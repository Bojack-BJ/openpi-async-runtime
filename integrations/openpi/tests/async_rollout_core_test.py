from __future__ import annotations

import math

import numpy as np

from async_rollout_core import ActionBuffer
from async_rollout_core import LatencyEstimator


def _buffer(**kwargs) -> ActionBuffer:
    defaults = {
        "min_buffer_steps": 2,
        "blend_horizon_steps": 4,
        "blend_schedule": "none",
        "empty_action_policy": "hold",
        "action_smoothing": "off",
        "action_ema_alpha": 0.5,
    }
    defaults.update(kwargs)
    return ActionBuffer(**defaults)


def test_fixed_delay_skips_expired_actions() -> None:
    buffer = _buffer(min_buffer_steps=0)
    actions = np.arange(5, dtype=np.float64).reshape(5, 1)

    stats = buffer.merge_chunk(
        actions,
        request_step=10,
        current_step=10,
        action_start=0,
        action_end=4,
        latency_steps=2,
    )

    assert stats.inserted == 3
    assert buffer.pop(10).action is None
    assert buffer.pop(11).action is None
    np.testing.assert_allclose(buffer.pop(12).action, [2.0])
    np.testing.assert_allclose(buffer.pop(13).action, [3.0])
    np.testing.assert_allclose(buffer.pop(14).action, [4.0])


def test_dynamic_latency_ema_steps() -> None:
    estimator = LatencyEstimator(fixed_steps=-1, control_hz=20.0, ema_alpha=0.5)

    assert estimator.observe(0.11) == math.ceil(0.11 * 20.0)
    assert estimator.observe(0.21) == math.ceil((0.5 * 0.11 + 0.5 * 0.21) * 20.0)


def test_linear_overlap_blending_trusts_old_near_current() -> None:
    buffer = _buffer(min_buffer_steps=2, blend_horizon_steps=4, blend_schedule="linear")
    old_actions = np.ones((8, 1), dtype=np.float64)
    new_actions = np.full((8, 1), 10.0, dtype=np.float64)

    buffer.merge_chunk(old_actions, request_step=0, current_step=0, action_start=0, action_end=7, latency_steps=0)
    buffer.merge_chunk(new_actions, request_step=0, current_step=0, action_start=0, action_end=7, latency_steps=0)

    assert buffer.pop(0).action is None
    assert buffer.pop(1).action is None
    np.testing.assert_allclose(buffer.pop(2).action, [1.0])
    np.testing.assert_allclose(buffer.pop(4).action, [5.5])
    np.testing.assert_allclose(buffer.pop(6).action, [10.0])


def test_exp_overlap_blending_replaces_far_future() -> None:
    buffer = _buffer(min_buffer_steps=1, blend_horizon_steps=2, blend_schedule="exp")
    old_actions = np.ones((5, 1), dtype=np.float64)
    new_actions = np.full((5, 1), 10.0, dtype=np.float64)

    buffer.merge_chunk(old_actions, request_step=0, current_step=0, action_start=0, action_end=4, latency_steps=0)
    buffer.merge_chunk(new_actions, request_step=0, current_step=0, action_start=0, action_end=4, latency_steps=0)

    assert buffer.pop(0).action is None
    np.testing.assert_allclose(buffer.pop(1).action, [1.0])
    near = buffer.pop(2).action[0]
    assert 1.0 < near < 10.0
    np.testing.assert_allclose(buffer.pop(3).action, [10.0])


def test_empty_buffer_holds_last_action() -> None:
    buffer = _buffer(empty_action_policy="hold")
    actions = np.asarray([[3.0]], dtype=np.float64)
    buffer.merge_chunk(actions, request_step=0, current_step=0, action_start=0, action_end=0, latency_steps=0)

    np.testing.assert_allclose(buffer.pop(0).action, [3.0])
    held = buffer.pop(1)
    assert held.held
    assert held.missing
    np.testing.assert_allclose(held.action, [3.0])


def test_min_buffer_steps_prevents_near_term_overwrite() -> None:
    buffer = _buffer(min_buffer_steps=3, blend_schedule="none")
    actions = np.arange(5, dtype=np.float64).reshape(5, 1)

    stats = buffer.merge_chunk(actions, request_step=0, current_step=0, action_start=0, action_end=4, latency_steps=0)

    assert stats.skipped_expired == 3
    assert buffer.pop(0).action is None
    assert buffer.pop(1).action is None
    assert buffer.pop(2).action is None
    np.testing.assert_allclose(buffer.pop(3).action, [3.0])


def test_action_ema_smoothing() -> None:
    buffer = _buffer(action_smoothing="ema", action_ema_alpha=0.25, min_buffer_steps=0)
    actions = np.asarray([[0.0], [4.0]], dtype=np.float64)
    buffer.merge_chunk(actions, request_step=0, current_step=0, action_start=0, action_end=1, latency_steps=0)

    np.testing.assert_allclose(buffer.pop(0).action, [0.0])
    np.testing.assert_allclose(buffer.pop(1).action, [1.0])
