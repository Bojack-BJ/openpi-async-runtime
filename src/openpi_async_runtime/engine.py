"""Generic asynchronous observation/inference/control orchestration."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import dataclasses
import threading
import time
from typing import Any

import numpy as np

from openpi_async_runtime.core import ActionBuffer
from openpi_async_runtime.core import LatencyEstimator
from openpi_async_runtime.core import limit_action_step
from openpi_async_runtime.core import should_advance_control_step
from openpi_async_runtime.protocol import EventSink
from openpi_async_runtime.protocol import PolicyTransport
from openpi_async_runtime.protocol import RobotBackend
from openpi_async_runtime.protocol import TrajectoryInstaller
from openpi_async_runtime.rtc import RTCClientConditioner
from openpi_async_runtime.rtc import RTC_ROLLOUT_KEY


@dataclasses.dataclass(frozen=True)
class AsyncEngineConfig:
    control_hz: float = 20.0
    inference_interval_steps: int = 10
    action_start: int = 0
    action_end: int | None = None
    max_position_step_m: float = 0.0
    max_rotation_step_deg: float = 0.0
    max_gripper_step: float = 0.0
    max_inference_delay_steps: int = -1


class AsyncRolloutEngine:
    """Robot- and transport-neutral asynchronous action-chunk engine.

    Deterministic ``run_inference_once`` and ``run_control_once`` methods make
    integrations testable without threads. ``start`` simply schedules those
    operations on independent fixed-rate loops.
    """

    def __init__(
        self,
        *,
        transport: PolicyTransport,
        robot: RobotBackend,
        action_buffer: ActionBuffer,
        latency_estimator: LatencyEstimator,
        config: AsyncEngineConfig = AsyncEngineConfig(),
        rtc: RTCClientConditioner | None = None,
        trajectory_installer: TrajectoryInstaller | None = None,
        event_sink: EventSink | None = None,
        response_actions: Callable[[Mapping[str, Any]], np.ndarray] | None = None,
    ) -> None:
        if config.control_hz <= 0:
            raise ValueError("control_hz must be positive")
        self.transport = transport
        self.robot = robot
        self.action_buffer = action_buffer
        self.latency_estimator = latency_estimator
        self.config = config
        self.rtc = rtc
        self.trajectory_installer = trajectory_installer
        self.event_sink = event_sink
        self.response_actions = response_actions or self._default_response_actions
        self._step = 0
        self._generation = 0
        self._last_action: np.ndarray | None = None
        self._last_delay_steps = 0
        self._state_lock = threading.Lock()
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._threads: list[threading.Thread] = []

    @property
    def step(self) -> int:
        with self._state_lock:
            return self._step

    @property
    def generation(self) -> int:
        with self._state_lock:
            return self._generation

    def reset(self) -> None:
        self.robot.reset()
        self.action_buffer.clear()
        with self._state_lock:
            self._step = 0
            self._generation += 1
            self._last_action = None
        self._emit("reset", {"generation": self.generation})

    def run_inference_once(self) -> Mapping[str, Any]:
        request_step = self.step
        generation = self.generation
        observation = dict(self.robot.observe())
        observation["__async_rollout"] = {
            "request_step": request_step,
            "control_hz": self.config.control_hz,
            "generation": generation,
            "buffer_size": self.action_buffer.pending_count_from(request_step),
        }
        if self.rtc is not None:
            observation[RTC_ROLLOUT_KEY] = self.rtc.request(
                generation=generation,
                request_step=request_step,
                delay_steps=self._last_delay_steps,
            )
        started = time.perf_counter()
        response = dict(self.transport.infer(observation))
        latency_s = time.perf_counter() - started
        if generation != self.generation:
            return {"discarded": True, "reason": "generation_changed"}
        latency_steps = self.latency_estimator.observe(latency_s)
        if self.config.max_inference_delay_steps >= 0:
            latency_steps = min(latency_steps, self.config.max_inference_delay_steps)
        self._last_delay_steps = latency_steps
        timeline = (
            self.rtc.response_timeline(
                response,
                request_step=request_step,
                latency_steps=latency_steps,
                action_start=self.config.action_start,
            )
            if self.rtc is not None
            else {
                "rtc": {},
                "applied": False,
                "base_step": request_step,
                "latency_steps": latency_steps,
                "action_start": self.config.action_start,
            }
        )
        actions = self.response_actions(response)
        current_step = self.step
        stats = self.action_buffer.merge_chunk(
            actions,
            request_step=int(timeline["base_step"]),
            current_step=current_step,
            action_start=int(timeline["action_start"]),
            action_end=len(actions) - 1 if self.config.action_end is None else self.config.action_end,
            latency_steps=int(timeline["latency_steps"]),
            latency_s=latency_s,
        )
        planner = None
        if self.trajectory_installer is not None:
            planner = self.trajectory_installer.install(
                actions,
                base_step=int(timeline["base_step"]),
                current_step=current_step,
            )
        event = {
            "request_step": request_step,
            "current_step": current_step,
            "latency_s": latency_s,
            "latency_steps": latency_steps,
            "inserted": stats.inserted,
            "blended": stats.blended,
            "skipped": stats.skipped_expired,
            "rtc": timeline["rtc"],
            "planner": planner,
        }
        self._emit("chunk", event)
        return event

    def run_control_once(self) -> Mapping[str, Any]:
        step = self.step
        read = self.action_buffer.pop(step)
        has_future = read.missing and self.action_buffer.next_pending_step_after(step) is not None
        if read.action is None:
            if should_advance_control_step(read, has_future_action=has_future):
                self._advance_step()
            event = {"step": step, "executed": False, "held": read.held, "future_gap": has_future}
            self._emit("control", event)
            return event
        action, limits = limit_action_step(
            np.asarray(read.action, dtype=np.float64),
            self._last_action,
            max_position_step_m=self.config.max_position_step_m,
            max_rotation_step_deg=self.config.max_rotation_step_deg,
            max_gripper_step=self.config.max_gripper_step,
        )
        self.robot.execute(action, step=step, metadata=read.metadata)
        with self._state_lock:
            self._last_action = action.copy()
        self._advance_step()
        event = {"step": step, "executed": True, "action": action, "limits": limits}
        self._emit("control", event)
        return event

    def start(self) -> None:
        if any(thread.is_alive() for thread in self._threads):
            return
        self._stop.clear()
        self._threads = [
            threading.Thread(target=self._inference_loop, name="async-policy", daemon=True),
            threading.Thread(target=self._control_loop, name="async-control", daemon=True),
        ]
        for thread in self._threads:
            thread.start()

    def pause(self) -> None:
        self._paused.set()

    def resume(self) -> None:
        self._paused.clear()

    def stop(self, *, timeout: float = 2.0) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=timeout)
        self.robot.close()

    def _inference_loop(self) -> None:
        next_step = 0
        while not self._stop.is_set():
            if self._paused.is_set() or self.step < next_step:
                time.sleep(0.005)
                continue
            self.run_inference_once()
            next_step = self.step + max(int(self.config.inference_interval_steps), 1)

    def _control_loop(self) -> None:
        period = 1.0 / self.config.control_hz
        next_tick = time.perf_counter()
        while not self._stop.is_set():
            if not self._paused.is_set():
                self.run_control_once()
            next_tick += period
            time.sleep(max(next_tick - time.perf_counter(), 0.0))

    def _advance_step(self) -> None:
        with self._state_lock:
            self._step += 1

    def _emit(self, event: str, payload: Mapping[str, Any]) -> None:
        if self.event_sink is not None:
            self.event_sink.emit(event, payload)

    @staticmethod
    def _default_response_actions(response: Mapping[str, Any]) -> np.ndarray:
        value = response["actions"] if "actions" in response else response["action"]
        actions = np.asarray(value, dtype=np.float64)
        return actions.reshape(1, -1) if actions.ndim == 1 else actions
