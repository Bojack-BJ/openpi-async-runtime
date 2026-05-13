from __future__ import annotations

import dataclasses
import math
import threading

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


@dataclasses.dataclass(frozen=True)
class BufferRead:
    action: np.ndarray | None
    held: bool
    missing: bool


class LatencyEstimator:
    """Converts inference latency into action steps.

    If fixed_steps >= 0, dynamic latency measurement is disabled.
    """

    def __init__(self, *, fixed_steps: int, control_hz: float, ema_alpha: float) -> None:
        self._fixed_steps = int(fixed_steps)
        self._control_hz = float(control_hz)
        self._ema_alpha = float(np.clip(ema_alpha, 0.0, 1.0))
        self._ema_latency_s: float | None = None

    @property
    def ema_latency_s(self) -> float | None:
        return self._ema_latency_s

    def observe(self, latency_s: float) -> int:
        if self._fixed_steps >= 0:
            return self._fixed_steps
        latency_s = max(float(latency_s), 0.0)
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
        self._buffer: dict[int, np.ndarray] = {}
        self._last_action: np.ndarray | None = None
        self._last_smoothed_action: np.ndarray | None = None
        self._lock = threading.Lock()

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()
            self._last_action = None
            self._last_smoothed_action = None

    def size(self) -> int:
        with self._lock:
            return len(self._buffer)

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
            return MergeStats(
                request_step=request_step,
                current_step=current_step,
                action_start=action_start,
                action_end=action_end,
                latency_steps=latency_steps,
                inserted=0,
                blended=0,
                skipped_expired=max(action_end - action_start + 1, 0),
            )

        inserted = 0
        blended = 0
        skipped_expired = max(effective_start - action_start, 0)
        frozen_until = current_step + self._min_buffer_steps
        with self._lock:
            for action_index in range(effective_start, action_end + 1):
                target_step = request_step + action_index
                if target_step < frozen_until:
                    skipped_expired += 1
                    continue
                new_action = actions[action_index].copy()
                old_action = self._buffer.get(target_step)
                if old_action is not None:
                    weight = self._new_action_weight(target_step=target_step, current_step=current_step)
                    self._buffer[target_step] = (1.0 - weight) * old_action + weight * new_action
                    blended += 1
                else:
                    self._buffer[target_step] = new_action
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
        )

    def pop(self, step: int) -> BufferRead:
        step = int(step)
        with self._lock:
            action = self._buffer.pop(step, None)
            stale_steps = [target_step for target_step in self._buffer if target_step < step]
            for stale_step in stale_steps:
                del self._buffer[stale_step]
            missing = action is None
            held = False
            if action is None and self._empty_action_policy == "hold" and self._last_action is not None:
                action = self._last_action.copy()
                held = True
            if action is None:
                return BufferRead(action=None, held=False, missing=missing)

            action = np.asarray(action, dtype=np.float64).copy()
            self._last_action = action.copy()
            if self._action_smoothing == "ema" and self._last_smoothed_action is not None:
                alpha = self._action_ema_alpha
                action = (1.0 - alpha) * self._last_smoothed_action + alpha * action
            self._last_smoothed_action = action.copy()
            return BufferRead(action=action, held=held, missing=missing)

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
