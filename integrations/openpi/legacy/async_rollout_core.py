from __future__ import annotations

import dataclasses
import json
import math
import pathlib
import threading
import time
from typing import Any

import numpy as np


@dataclasses.dataclass(frozen=True)
class MergeStats:
    request_step: int
    current_step: int
    action_start: int
    action_end: int
    latency_steps: int
    inserted: int
    blended: int
    skipped_expired: int
    events: tuple[dict[str, Any], ...] = ()


@dataclasses.dataclass(frozen=True)
class BufferRead:
    action: np.ndarray | None
    held: bool
    missing: bool
    metadata: dict[str, Any] | None = None


@dataclasses.dataclass(frozen=True)
class TimedObservation:
    obs_id: int
    request_step: int
    capture_time: float
    send_time: float
    buffer_size: int
    robot_state: Any
    image_metadata: dict[str, Any] | None = None


@dataclasses.dataclass(frozen=True)
class TimedAction:
    chunk_id: int
    action_index: int
    target_step: int | None
    action: Any
    merge_type: str
    blend_weight: float | None
    source_obs_id: int
    latency_s: float
    delay_steps: int


@dataclasses.dataclass(frozen=True)
class ExecutedAction:
    control_step: int
    execute_time: float
    action: Any
    held: bool
    missing: bool
    buffer_size: int
    robot_pose_before: Any = None
    robot_pose_after: Any = None
    command_delta: Any = None
    tracking_error: Any = None
    raw_action: Any = None
    limited_action: Any = None
    limit_applied: bool = False
    position_delta_m: float | None = None
    rotation_delta_deg: float | None = None


def to_jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return to_jsonable(dataclasses.asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


class AsyncDebugWriter:
    def __init__(self, debug_dir: str | pathlib.Path | None, *, flush_interval: int = 1) -> None:
        self.enabled = debug_dir is not None
        self._flush_interval = max(int(flush_interval), 1)
        self._lock = threading.Lock()
        self._counts: dict[str, int] = {}
        self._files: dict[str, Any] = {}
        self._start_time = time.time()
        self.debug_dir: pathlib.Path | None = pathlib.Path(debug_dir) if debug_dir else None
        if self.debug_dir is not None:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            for name in ("observations", "actions", "executions", "chunks"):
                self._files[name] = (self.debug_dir / f"{name}.jsonl").open("a", encoding="utf-8")

    def write(self, stream: str, record: Any) -> None:
        if not self.enabled:
            return
        with self._lock:
            file = self._files[stream]
            file.write(json.dumps(to_jsonable(record), ensure_ascii=False, sort_keys=True) + "\n")
            count = self._counts.get(stream, 0) + 1
            self._counts[stream] = count
            if count % self._flush_interval == 0:
                file.flush()

    def write_summary(self, extra: dict[str, Any] | None = None) -> None:
        if not self.enabled or self.debug_dir is None:
            return
        summary = {
            "start_time": self._start_time,
            "end_time": time.time(),
            "counts": dict(self._counts),
        }
        if extra:
            summary.update(to_jsonable(extra))
        with (self.debug_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, sort_keys=True)

    def close(self, extra_summary: dict[str, Any] | None = None) -> None:
        if not self.enabled:
            return
        with self._lock:
            self.write_summary(extra_summary)
            for file in self._files.values():
                file.flush()
                file.close()


class LatencyEstimator:
    """Converts inference latency into action steps.

    Modes:
    - fixed: always returns fixed_steps.
    - instant: converts the current latency directly.
    - ema: converts an EMA-smoothed latency.
    """

    def __init__(self, *, mode: str, fixed_steps: int, control_hz: float, ema_alpha: float) -> None:
        if mode not in ("fixed", "instant", "ema"):
            raise ValueError(f"Unsupported latency estimator mode: {mode}")
        self._mode = mode
        self._fixed_steps = int(fixed_steps)
        self._control_hz = float(control_hz)
        self._ema_alpha = float(np.clip(ema_alpha, 0.0, 1.0))
        self._ema_latency_s: float | None = None

    @property
    def ema_latency_s(self) -> float | None:
        return self._ema_latency_s

    def observe(self, latency_s: float) -> int:
        latency_s = max(float(latency_s), 0.0)
        if self._mode == "fixed":
            return max(self._fixed_steps, 0)
        if self._mode == "instant":
            return max(int(math.ceil(latency_s * self._control_hz)), 0)
        if self._ema_latency_s is None:
            self._ema_latency_s = latency_s
        else:
            alpha = self._ema_alpha
            self._ema_latency_s = (1.0 - alpha) * self._ema_latency_s + alpha * latency_s
        return max(int(math.ceil(self._ema_latency_s * self._control_hz)), 0)


class ActionBuffer:
    """Thread-safe absolute-step action buffer with RTC-style overlap blending."""

    def __init__(
        self,
        *,
        min_buffer_steps: int,
        blend_horizon_steps: int,
        blend_schedule: str,
        empty_action_policy: str,
        action_smoothing: str,
        action_ema_alpha: float,
        cyclic_indices: tuple[int, ...] = (),
        cyclic_period: float = 360.0,
    ) -> None:
        if blend_schedule not in ("exp", "linear", "none"):
            raise ValueError(f"Unsupported blend schedule: {blend_schedule}")
        if empty_action_policy not in ("hold", "none"):
            raise ValueError(f"Unsupported empty action policy: {empty_action_policy}")
        if action_smoothing not in ("off", "ema"):
            raise ValueError(f"Unsupported action smoothing: {action_smoothing}")
        self._min_buffer_steps = max(int(min_buffer_steps), 0)
        self._blend_horizon_steps = max(int(blend_horizon_steps), 1)
        self._blend_schedule = blend_schedule
        self._empty_action_policy = empty_action_policy
        self._action_smoothing = action_smoothing
        self._action_ema_alpha = float(np.clip(action_ema_alpha, 0.0, 1.0))
        self._cyclic_indices = tuple(int(index) for index in cyclic_indices)
        self._cyclic_period = float(cyclic_period)
        self._buffer: dict[int, np.ndarray] = {}
        self._metadata: dict[int, dict[str, Any]] = {}
        self._last_action: np.ndarray | None = None
        self._last_smoothed_action: np.ndarray | None = None
        self._lock = threading.Lock()

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()
            self._metadata.clear()
            self._last_action = None
            self._last_smoothed_action = None

    def size(self) -> int:
        with self._lock:
            return len(self._buffer)

    def has_last_action(self) -> bool:
        with self._lock:
            return self._last_action is not None

    def pending_count_from(self, step: int) -> int:
        with self._lock:
            return sum(1 for target_step in self._buffer if target_step >= step)

    def merge_chunk(
        self,
        actions: np.ndarray,
        *,
        request_step: int,
        current_step: int,
        action_start: int,
        action_end: int,
        latency_steps: int,
        chunk_id: int | None = None,
        source_obs_id: int | None = None,
        latency_s: float | None = None,
    ) -> MergeStats:
        actions = np.asarray(actions, dtype=np.float64)
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)
        if actions.ndim != 2:
            raise ValueError(f"Expected actions to have shape (T, D), got {actions.shape}")

        request_step = int(request_step)
        current_step = int(current_step)
        action_start = max(int(action_start), 0)
        action_end = min(int(action_end), len(actions) - 1)
        latency_steps = max(int(latency_steps), 0)
        effective_start = action_start + latency_steps
        if action_end < effective_start or not len(actions):
            events = tuple(
                {
                    "chunk_id": chunk_id,
                    "source_obs_id": source_obs_id,
                    "action_index": action_index,
                    "target_step": None,
                    "merge_type": "skipped",
                    "blend_weight": None,
                    "latency_s": latency_s,
                    "delay_steps": latency_steps,
                    "action": actions[action_index].copy() if 0 <= action_index < len(actions) else None,
                }
                for action_index in range(action_start, action_end + 1)
            )
            return MergeStats(
                request_step=request_step,
                current_step=current_step,
                action_start=action_start,
                action_end=action_end,
                latency_steps=latency_steps,
                inserted=0,
                blended=0,
                skipped_expired=max(action_end - action_start + 1, 0),
                events=events,
            )

        inserted = 0
        blended = 0
        skipped_expired = max(effective_start - action_start, 0)
        events: list[dict[str, Any]] = []
        for action_index in range(action_start, effective_start):
            if action_index <= action_end:
                events.append(
                    {
                        "chunk_id": chunk_id,
                        "source_obs_id": source_obs_id,
                        "action_index": action_index,
                        "target_step": None,
                        "merge_type": "skipped",
                        "blend_weight": None,
                        "latency_s": latency_s,
                        "delay_steps": latency_steps,
                        "action": actions[action_index].copy(),
                    }
                )
        with self._lock:
            first_fill = self._last_action is None and not self._buffer
            frozen_until = current_step if first_fill else current_step + self._min_buffer_steps
            for action_index in range(effective_start, action_end + 1):
                target_step = request_step + action_index
                event_base = {
                    "chunk_id": chunk_id,
                    "source_obs_id": source_obs_id,
                    "action_index": action_index,
                    "target_step": target_step,
                    "latency_s": latency_s,
                    "delay_steps": latency_steps,
                    "action": actions[action_index].copy(),
                }
                if target_step < frozen_until:
                    skipped_expired += 1
                    events.append({**event_base, "merge_type": "skipped", "blend_weight": None})
                    continue
                new_action = actions[action_index].copy()
                old_action = self._buffer.get(target_step)
                if old_action is not None:
                    weight = self._new_action_weight(target_step=target_step, current_step=current_step)
                    self._buffer[target_step] = self._blend_actions(old_action, new_action, weight)
                    self._metadata[target_step] = {
                        **event_base,
                        "merge_type": "blended",
                        "blend_weight": weight,
                    }
                    events.append(self._metadata[target_step])
                    blended += 1
                else:
                    self._buffer[target_step] = new_action
                    self._metadata[target_step] = {
                        **event_base,
                        "merge_type": "inserted",
                        "blend_weight": None,
                    }
                    events.append(self._metadata[target_step])
                    inserted += 1
        return MergeStats(
            request_step=request_step,
            current_step=current_step,
            action_start=action_start,
            action_end=action_end,
            latency_steps=latency_steps,
            inserted=inserted,
            blended=blended,
            skipped_expired=skipped_expired,
            events=tuple(events),
        )

    def pop(self, step: int) -> BufferRead:
        step = int(step)
        with self._lock:
            action = self._buffer.pop(step, None)
            metadata = self._metadata.pop(step, None)
            stale_steps = [target_step for target_step in self._buffer if target_step < step]
            for stale_step in stale_steps:
                del self._buffer[stale_step]
                self._metadata.pop(stale_step, None)
            missing = action is None
            held = False
            if action is None and self._empty_action_policy == "hold" and self._last_action is not None:
                action = self._last_action.copy()
                metadata = {"merge_type": "held", "target_step": step}
                held = True
            if action is None:
                return BufferRead(action=None, held=False, missing=missing, metadata=metadata)

            action = np.asarray(action, dtype=np.float64).copy()
            self._last_action = action.copy()
            if self._action_smoothing == "ema" and self._last_smoothed_action is not None:
                alpha = self._action_ema_alpha
                action = self._blend_actions(self._last_smoothed_action, action, alpha)
            self._last_smoothed_action = action.copy()
            return BufferRead(action=action, held=held, missing=missing, metadata=metadata)

    def _new_action_weight(self, *, target_step: int, current_step: int) -> float:
        if self._blend_schedule == "none":
            return 1.0
        unfrozen_start = current_step + self._min_buffer_steps
        progress = (target_step - unfrozen_start) / max(float(self._blend_horizon_steps), 1.0)
        progress = float(np.clip(progress, 0.0, 1.0))
        if progress >= 1.0:
            return 1.0
        if self._blend_schedule == "linear":
            return progress
        return float(1.0 - math.exp(-4.0 * progress))

    def _blend_actions(self, old_action: np.ndarray, new_action: np.ndarray, weight: float) -> np.ndarray:
        weight = float(np.clip(weight, 0.0, 1.0))
        blended = (1.0 - weight) * old_action + weight * new_action
        if not self._cyclic_indices:
            return blended
        for index in self._cyclic_indices:
            if index < 0 or index >= len(blended):
                continue
            delta = self._wrap_cyclic_delta(new_action[index] - old_action[index])
            blended[index] = self._normalize_cyclic(old_action[index] + weight * delta)
        return blended

    def _wrap_cyclic_delta(self, delta: float) -> float:
        period = self._cyclic_period
        return ((float(delta) + period / 2.0) % period) - period / 2.0

    def _normalize_cyclic(self, value: float) -> float:
        period = self._cyclic_period
        return ((float(value) + period / 2.0) % period) - period / 2.0


def limit_action_step(
    action: np.ndarray,
    previous_action: np.ndarray | None,
    *,
    max_position_step_m: float,
    max_rotation_step_deg: float,
    max_gripper_step: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    action = np.asarray(action, dtype=np.float64).copy()
    if previous_action is None:
        return action, {
            "limit_applied": False,
            "position_delta_m": 0.0,
            "rotation_delta_deg": 0.0,
            "gripper_delta": 0.0,
        }
    previous = np.asarray(previous_action, dtype=np.float64)
    limited = action.copy()
    position_delta_max = 0.0
    rotation_delta_max = 0.0
    gripper_delta_max = 0.0
    limit_applied = False
    arm_offsets = (0, 7) if len(action) >= 14 else (0,)
    for offset in arm_offsets:
        if offset + 6 >= len(action):
            continue
        pos_delta = limited[offset : offset + 3] - previous[offset : offset + 3]
        pos_norm = float(np.linalg.norm(pos_delta))
        position_delta_max = max(position_delta_max, pos_norm)
        if max_position_step_m > 0 and pos_norm > max_position_step_m:
            limited[offset : offset + 3] = previous[offset : offset + 3] + pos_delta * (max_position_step_m / pos_norm)
            limit_applied = True

        rpy_delta = _wrap_degrees(limited[offset + 3 : offset + 6] - previous[offset + 3 : offset + 6])
        rpy_norm = float(np.linalg.norm(rpy_delta))
        rotation_delta_max = max(rotation_delta_max, rpy_norm)
        if max_rotation_step_deg > 0 and rpy_norm > max_rotation_step_deg:
            rpy_delta = rpy_delta * (max_rotation_step_deg / rpy_norm)
            limited[offset + 3 : offset + 6] = _normalize_degrees(previous[offset + 3 : offset + 6] + rpy_delta)
            limit_applied = True

        gripper_index = offset + 6
        gripper_delta = float(limited[gripper_index] - previous[gripper_index])
        gripper_delta_max = max(gripper_delta_max, abs(gripper_delta))
        if max_gripper_step > 0 and abs(gripper_delta) > max_gripper_step:
            limited[gripper_index] = previous[gripper_index] + math.copysign(max_gripper_step, gripper_delta)
            limit_applied = True

    return limited, {
        "limit_applied": limit_applied,
        "position_delta_m": position_delta_max,
        "rotation_delta_deg": rotation_delta_max,
        "gripper_delta": gripper_delta_max,
    }


def action_tracking_error(command_action: np.ndarray | None, robot_pose: np.ndarray | None) -> dict[str, float] | None:
    if command_action is None or robot_pose is None:
        return None
    command = np.asarray(command_action, dtype=np.float64)
    pose = np.asarray(robot_pose, dtype=np.float64)
    if len(command) < 6 or len(pose) < 6:
        return None
    return {
        "position_error_m": float(np.linalg.norm(command[:3] - pose[:3])),
        "rotation_error_deg": float(np.linalg.norm(_wrap_degrees(command[3:6] - pose[3:6]))),
    }


def action_command_delta(action: np.ndarray, previous_action: np.ndarray | None) -> np.ndarray | None:
    if previous_action is None:
        return None
    delta = np.asarray(action, dtype=np.float64) - np.asarray(previous_action, dtype=np.float64)
    arm_offsets = (0, 7) if len(delta) >= 14 else (0,)
    for offset in arm_offsets:
        if offset + 6 <= len(delta):
            delta[offset + 3 : offset + 6] = _wrap_degrees(delta[offset + 3 : offset + 6])
    return delta


def _wrap_degrees(delta: np.ndarray) -> np.ndarray:
    return ((np.asarray(delta, dtype=np.float64) + 180.0) % 360.0) - 180.0


def _normalize_degrees(value: np.ndarray) -> np.ndarray:
    return ((np.asarray(value, dtype=np.float64) + 180.0) % 360.0) - 180.0
