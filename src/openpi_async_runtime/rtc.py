"""Real-time chunk conditioning on client and policy-server boundaries."""

from __future__ import annotations

import dataclasses
import threading
from typing import Any

import numpy as np


RTC_ROLLOUT_KEY = "__rtc_rollout"


@dataclasses.dataclass(frozen=True)
class RTCConditionResult:
    condition: np.ndarray
    weight: np.ndarray
    applied: bool
    metadata: dict[str, Any]


@dataclasses.dataclass(frozen=True)
class RTCChunkState:
    session_id: str
    generation: int
    base_step: int
    actions: np.ndarray


def build_rtc_action_condition(
    *,
    previous_actions: np.ndarray | None,
    previous_base_step: int | None,
    request_step: int,
    delay_steps: int,
    soft_horizon_steps: int,
    free_tail_steps: int,
    action_horizon: int,
    action_dim: int,
) -> RTCConditionResult:
    """Build frozen, soft-guided, and free RTC regions in normalized action space."""
    delay_steps = max(int(delay_steps), 0)
    action_horizon = max(int(action_horizon), 0)
    action_dim = max(int(action_dim), 0)
    free_tail_steps = max(int(free_tail_steps), 0)
    soft_horizon_steps = max(int(soft_horizon_steps), 1)
    condition = np.zeros((action_horizon, action_dim), dtype=np.float32)
    weight = np.zeros((action_horizon,), dtype=np.float32)
    metadata: dict[str, Any] = {
        "request_step": int(request_step),
        "delay_steps": delay_steps,
        "soft_horizon_steps": soft_horizon_steps,
        "free_tail_steps": free_tail_steps,
        "action_horizon": action_horizon,
        "condition_steps": 0,
        "frozen_steps": 0,
        "soft_steps": 0,
        "free_steps": min(free_tail_steps, action_horizon),
        "skip_reason": None,
    }
    if delay_steps >= action_horizon:
        metadata["skip_reason"] = "delay_ge_horizon"
        return RTCConditionResult(condition, weight, False, metadata)
    if previous_actions is None or previous_base_step is None:
        metadata["skip_reason"] = "no_previous_chunk"
        return RTCConditionResult(condition, weight, False, metadata)

    previous_actions = np.asarray(previous_actions, dtype=np.float32)
    soft_end = max(action_horizon - free_tail_steps, delay_steps)
    copy_dim = min(action_dim, previous_actions.shape[-1])
    for action_index in range(action_horizon):
        if action_index >= soft_end:
            continue
        previous_index = int(request_step) + action_index - int(previous_base_step)
        if previous_index < 0 or previous_index >= len(previous_actions):
            continue
        condition[action_index, :copy_dim] = previous_actions[previous_index, :copy_dim]
        if action_index < delay_steps:
            weight[action_index] = 1.0
            metadata["frozen_steps"] += 1
        else:
            weight[action_index] = float(np.exp(-max(action_index - delay_steps, 0) / soft_horizon_steps))
            metadata["soft_steps"] += 1

    metadata["condition_steps"] = int(np.count_nonzero(weight > 0))
    metadata["previous_base_step"] = int(previous_base_step)
    metadata["request_delta_steps"] = int(request_step) - int(previous_base_step)
    applied = bool(metadata["condition_steps"])
    if not applied:
        metadata["skip_reason"] = "no_overlapping_previous_actions"
    return RTCConditionResult(condition, weight, applied, metadata)


class RTCSessionConditioner:
    """Thread-safe server-side RTC state independent of policy frameworks."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        fixed_delay_steps: int = -1,
        soft_horizon_steps: int = 5,
        free_tail_steps: int = 5,
    ) -> None:
        self.enabled = bool(enabled)
        self.fixed_delay_steps = int(fixed_delay_steps)
        self.soft_horizon_steps = max(int(soft_horizon_steps), 1)
        self.free_tail_steps = max(int(free_tail_steps), 0)
        self._state: RTCChunkState | None = None
        self._lock = threading.Lock()

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "delay_steps": self.fixed_delay_steps,
            "soft_horizon_steps": self.soft_horizon_steps,
            "free_tail_steps": self.free_tail_steps,
        }

    def prepare(self, request: dict[str, Any] | None, *, action_horizon: int, action_dim: int):
        if not self.enabled or not isinstance(request, dict) or not request.get("enabled", True):
            return None, None
        request_step = int(request.get("request_step", 0))
        session_id = str(request.get("session_id") or "default")
        generation = int(request.get("generation", 0))
        delay_steps = int(request.get("delay_steps", self.fixed_delay_steps))
        if self.fixed_delay_steps >= 0:
            delay_steps = self.fixed_delay_steps
        soft_horizon_steps = int(request.get("soft_horizon_steps", self.soft_horizon_steps))
        free_tail_steps = int(request.get("free_tail_steps", self.free_tail_steps))
        with self._lock:
            previous = self._state
            if previous is not None and (previous.session_id != session_id or previous.generation != generation):
                previous = None
                self._state = None
        if previous is not None:
            action_horizon = int(action_horizon) or int(previous.actions.shape[0])
            action_dim = int(action_dim) or int(previous.actions.shape[-1])
        result = build_rtc_action_condition(
            previous_actions=None if previous is None else previous.actions,
            previous_base_step=None if previous is None else previous.base_step,
            request_step=request_step,
            delay_steps=delay_steps,
            soft_horizon_steps=soft_horizon_steps,
            free_tail_steps=free_tail_steps,
            action_horizon=action_horizon,
            action_dim=action_dim,
        )
        response = {
            **result.metadata,
            "enabled": True,
            "requested": True,
            "applied": result.applied,
            "session_id": session_id,
            "generation": generation,
            "_state_update": {
                "session_id": session_id,
                "generation": generation,
                "base_step": request_step,
                "delay_steps": max(delay_steps, 0),
            },
        }
        return result, response

    def finalize(self, raw_actions: np.ndarray, response: dict[str, Any] | None):
        if response is None:
            return raw_actions, None
        update = response.pop("_state_update")
        full_actions = np.asarray(raw_actions, dtype=np.float32).copy()
        with self._lock:
            self._state = RTCChunkState(
                session_id=str(update["session_id"]),
                generation=int(update["generation"]),
                base_step=int(update["base_step"]),
                actions=full_actions,
            )
        if response.get("applied", False):
            delay_steps = int(update["delay_steps"])
            if delay_steps >= len(raw_actions):
                response["applied"] = False
                response["skip_reason"] = "delay_ge_returned_horizon"
                return raw_actions[:0], response
            response["action_base_step"] = int(update["base_step"]) + delay_steps
            return raw_actions[delay_steps:], response
        response["action_base_step"] = int(update["base_step"])
        return raw_actions, response


class RTCClientConditioner:
    """Client-side RTC metadata and response timeline adapter."""

    def __init__(self, session_id: str, *, soft_horizon_steps: int = 5, free_tail_steps: int = 5) -> None:
        self.session_id = session_id
        self.soft_horizon_steps = max(int(soft_horizon_steps), 1)
        self.free_tail_steps = max(int(free_tail_steps), 0)

    def request(self, *, generation: int, request_step: int, delay_steps: int) -> dict[str, Any]:
        return {
            "enabled": True,
            "session_id": self.session_id,
            "generation": int(generation),
            "request_step": int(request_step),
            "delay_steps": max(int(delay_steps), 0),
            "soft_horizon_steps": self.soft_horizon_steps,
            "free_tail_steps": self.free_tail_steps,
        }

    @staticmethod
    def response_timeline(response: dict[str, Any], *, request_step: int, latency_steps: int, action_start: int):
        payload = response.get("rtc", {})
        applied = bool(payload.get("applied", False))
        return {
            "rtc": payload,
            "applied": applied,
            "base_step": int(response.get("action_base_step", request_step)) if applied else int(request_step),
            "latency_steps": 0 if applied else int(latency_steps),
            "action_start": 0 if applied else int(action_start),
        }
