from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from openpi_async_runtime.conditioning import project_action_condition
from openpi_async_runtime.core import ActionBuffer
from openpi_async_runtime.core import LatencyEstimator
from openpi_async_runtime.engine import AsyncEngineConfig
from openpi_async_runtime.engine import AsyncRolloutEngine
from openpi_async_runtime.rtc import build_rtc_action_condition
from openpi_async_runtime.rtc import RTCClientConditioner
from openpi_async_runtime.rtc import RTCSessionConditioner
from openpi_async_runtime.rtc import RTC_ROLLOUT_KEY


def _buffer() -> ActionBuffer:
    return ActionBuffer(
        min_buffer_steps=0,
        blend_horizon_steps=2,
        blend_schedule="none",
        empty_action_policy="none",
        action_smoothing="off",
        action_ema_alpha=0.5,
    )


def test_rtc_condition_aligns_previous_chunk_on_absolute_timeline() -> None:
    previous = np.arange(12, dtype=np.float32).reshape(6, 2)

    result = build_rtc_action_condition(
        previous_actions=previous,
        previous_base_step=10,
        request_step=12,
        delay_steps=2,
        soft_horizon_steps=2,
        free_tail_steps=1,
        action_horizon=5,
        action_dim=2,
    )

    assert result.applied
    np.testing.assert_allclose(result.condition[:4], previous[2:6])
    np.testing.assert_allclose(result.weight, [1.0, 1.0, 1.0, np.exp(-0.5), 0.0])
    assert result.metadata["frozen_steps"] == 2
    assert result.metadata["soft_steps"] == 2


def test_rtc_session_uses_full_previous_chunk_and_returns_only_live_suffix() -> None:
    conditioner = RTCSessionConditioner(enabled=True, free_tail_steps=0)
    first_request = {
        "enabled": True,
        "session_id": "session-a",
        "generation": 0,
        "request_step": 0,
        "delay_steps": 2,
    }
    first_condition, first_response = conditioner.prepare(first_request, action_horizon=5, action_dim=1)
    assert first_condition is not None and not first_condition.applied
    first_actions, first_response = conditioner.finalize(np.arange(5, dtype=np.float32)[:, None], first_response)
    np.testing.assert_allclose(first_actions[:, 0], np.arange(5))
    assert first_response is not None and first_response["action_base_step"] == 0

    second_request = {**first_request, "request_step": 2}
    second_condition, second_response = conditioner.prepare(second_request, action_horizon=5, action_dim=1)
    assert second_condition is not None and second_condition.applied
    np.testing.assert_allclose(second_condition.condition[:3, 0], [2.0, 3.0, 4.0])

    returned, second_response = conditioner.finalize(np.arange(10, 15, dtype=np.float32)[:, None], second_response)
    np.testing.assert_allclose(returned[:, 0], [12.0, 13.0, 14.0])
    assert second_response is not None and second_response["action_base_step"] == 4


def test_condition_projection_is_numpy_backend_neutral() -> None:
    x_t = np.zeros((1, 3, 1), dtype=np.float32)
    noise = np.full_like(x_t, 2.0)
    condition = np.full_like(x_t, 6.0)
    weight = np.asarray([[1.0, 0.5, 0.0]], dtype=np.float32)

    projected = project_action_condition(
        x_t,
        time=np.asarray(0.25, dtype=np.float32),
        noise=noise,
        action_condition=condition,
        action_condition_weight=weight,
    )

    np.testing.assert_allclose(projected[:, :, 0], [[5.0, 2.5, 0.0]])


class _FakeTransport:
    def __init__(self) -> None:
        self.observations: list[dict[str, Any]] = []

    def infer(self, observation: dict[str, Any]) -> Mapping[str, Any]:
        self.observations.append(observation)
        return {"actions": np.asarray([[1.0, 2.0], [3.0, 4.0]])}


class _FakeRobot:
    def __init__(self) -> None:
        self.executed: list[tuple[int, np.ndarray]] = []
        self.reset_count = 0
        self.closed = False

    def observe(self) -> dict[str, Any]:
        return {"state": np.asarray([0.0])}

    def execute(self, action: np.ndarray, *, step: int, metadata: Mapping[str, Any] | None = None) -> None:
        del metadata
        self.executed.append((step, action.copy()))

    def reset(self) -> None:
        self.reset_count += 1

    def close(self) -> None:
        self.closed = True


def test_engine_runs_with_transport_and_robot_adapters() -> None:
    transport = _FakeTransport()
    robot = _FakeRobot()
    engine = AsyncRolloutEngine(
        transport=transport,
        robot=robot,
        action_buffer=_buffer(),
        latency_estimator=LatencyEstimator(mode="fixed", fixed_steps=0, control_hz=20.0, ema_alpha=0.5),
        config=AsyncEngineConfig(control_hz=20.0),
        rtc=RTCClientConditioner("session-a"),
    )

    chunk = engine.run_inference_once()
    first = engine.run_control_once()
    second = engine.run_control_once()

    assert chunk["inserted"] == 2
    assert RTC_ROLLOUT_KEY in transport.observations[0]
    assert first["executed"] and second["executed"]
    assert [step for step, _ in robot.executed] == [0, 1]
    np.testing.assert_allclose(robot.executed[0][1], [1.0, 2.0])
    np.testing.assert_allclose(robot.executed[1][1], [3.0, 4.0])
