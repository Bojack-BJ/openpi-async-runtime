from __future__ import annotations

import math

import numpy as np

from scripts.rollout.async_rollout_core import ActionBuffer
from scripts.rollout.async_rollout_core import LatencyEstimator
from scripts.rollout.async_rollout_core import TimedAction
from scripts.rollout.async_rollout_core import action_command_delta
from scripts.rollout.async_rollout_core import limit_action_step
from scripts.rollout.async_rollout_core import plan_joint_cubic_trajectory
from scripts.rollout.async_rollout_core import should_advance_control_step
from scripts.rollout.async_rollout_core import to_jsonable


def _buffer(**kwargs) -> ActionBuffer:
    defaults = {
        "min_buffer_steps": 2,
        "blend_horizon_steps": 4,
        "blend_schedule": "none",
        "empty_action_policy": "hold",
        "action_smoothing": "off",
        "action_ema_alpha": 0.5,
        "cyclic_indices": (),
        "cyclic_period": 360.0,
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
    assert [event["merge_type"] for event in stats.events] == ["skipped", "skipped", "inserted", "inserted", "inserted"]
    assert buffer.pop(10).action is None
    assert buffer.pop(11).action is None
    np.testing.assert_allclose(buffer.pop(12).action, [2.0])
    np.testing.assert_allclose(buffer.pop(13).action, [3.0])
    np.testing.assert_allclose(buffer.pop(14).action, [4.0])


def test_dynamic_latency_ema_steps() -> None:
    estimator = LatencyEstimator(mode="ema", fixed_steps=0, control_hz=20.0, ema_alpha=0.5)

    assert estimator.observe(0.11) == math.ceil(0.11 * 20.0)
    assert estimator.observe(0.21) == math.ceil((0.5 * 0.11 + 0.5 * 0.21) * 20.0)


def test_instant_latency_uses_current_sample_only() -> None:
    estimator = LatencyEstimator(mode="instant", fixed_steps=0, control_hz=20.0, ema_alpha=0.5)

    assert estimator.observe(2.0) == 40
    assert estimator.observe(0.11) == math.ceil(0.11 * 20.0)


def test_fixed_latency_ignores_observed_latency() -> None:
    estimator = LatencyEstimator(mode="fixed", fixed_steps=3, control_hz=20.0, ema_alpha=0.5)

    assert estimator.observe(0.0) == 3
    assert estimator.observe(2.0) == 3


def test_linear_overlap_blending_trusts_old_near_current() -> None:
    buffer = _buffer(min_buffer_steps=2, blend_horizon_steps=4, blend_schedule="linear")
    old_actions = np.ones((8, 1), dtype=np.float64)
    new_actions = np.full((8, 1), 10.0, dtype=np.float64)

    buffer.merge_chunk(old_actions, request_step=0, current_step=0, action_start=0, action_end=7, latency_steps=0)
    buffer.merge_chunk(new_actions, request_step=0, current_step=0, action_start=0, action_end=7, latency_steps=0)

    np.testing.assert_allclose(buffer.pop(0).action, [1.0])
    np.testing.assert_allclose(buffer.pop(1).action, [1.0])
    np.testing.assert_allclose(buffer.pop(2).action, [1.0])
    np.testing.assert_allclose(buffer.pop(4).action, [5.5])
    np.testing.assert_allclose(buffer.pop(6).action, [10.0])


def test_exp_overlap_blending_replaces_far_future() -> None:
    buffer = _buffer(min_buffer_steps=1, blend_horizon_steps=2, blend_schedule="exp")
    old_actions = np.ones((5, 1), dtype=np.float64)
    new_actions = np.full((5, 1), 10.0, dtype=np.float64)

    buffer.merge_chunk(old_actions, request_step=0, current_step=0, action_start=0, action_end=4, latency_steps=0)
    buffer.merge_chunk(new_actions, request_step=0, current_step=0, action_start=0, action_end=4, latency_steps=0)

    np.testing.assert_allclose(buffer.pop(0).action, [1.0])
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


def test_control_step_only_advances_for_buffered_action() -> None:
    buffer = _buffer(empty_action_policy="hold")
    buffer.merge_chunk(np.asarray([[3.0]]), request_step=0, current_step=0, action_start=0, action_end=0, latency_steps=0)

    assert should_advance_control_step(buffer.pop(0))
    assert not should_advance_control_step(buffer.pop(1))
    assert not should_advance_control_step(_buffer(empty_action_policy="none").pop(0))


def test_contiguous_actions_from_skips_gap_and_caps_window() -> None:
    buffer = _buffer(min_buffer_steps=0)
    buffer.merge_chunk(
        np.asarray([[3.0], [4.0], [5.0]], dtype=np.float64),
        request_step=3,
        current_step=0,
        action_start=0,
        action_end=2,
        latency_steps=0,
    )

    start_step, actions = buffer.contiguous_actions_from(0, max_steps=2)

    assert start_step == 3
    np.testing.assert_allclose(actions, [[3.0], [4.0]])


def test_min_buffer_steps_prevents_near_term_overwrite() -> None:
    buffer = _buffer(min_buffer_steps=3, blend_schedule="none")
    old_actions = np.arange(5, dtype=np.float64).reshape(5, 1)
    new_actions = np.full((5, 1), 10.0, dtype=np.float64)

    first_stats = buffer.merge_chunk(old_actions, request_step=0, current_step=0, action_start=0, action_end=4, latency_steps=0)
    stats = buffer.merge_chunk(new_actions, request_step=0, current_step=0, action_start=0, action_end=4, latency_steps=0)

    assert first_stats.skipped_expired == 0
    assert stats.skipped_expired == 3
    assert len(stats.events) == 5
    np.testing.assert_allclose(buffer.pop(0).action, [0.0])
    np.testing.assert_allclose(buffer.pop(1).action, [1.0])
    np.testing.assert_allclose(buffer.pop(2).action, [2.0])
    np.testing.assert_allclose(buffer.pop(3).action, [10.0])


def test_action_ema_smoothing() -> None:
    buffer = _buffer(action_smoothing="ema", action_ema_alpha=0.25, min_buffer_steps=0)
    actions = np.asarray([[0.0], [4.0]], dtype=np.float64)
    buffer.merge_chunk(actions, request_step=0, current_step=0, action_start=0, action_end=1, latency_steps=0)

    np.testing.assert_allclose(buffer.pop(0).action, [0.0])
    np.testing.assert_allclose(buffer.pop(1).action, [1.0])


def test_cyclic_overlap_blending_uses_shortest_angle_path() -> None:
    buffer = _buffer(min_buffer_steps=0, blend_schedule="linear", blend_horizon_steps=2, cyclic_indices=(0,))
    old_actions = np.asarray([[179.0], [179.0], [179.0]], dtype=np.float64)
    new_actions = np.asarray([[-179.0], [-179.0], [-179.0]], dtype=np.float64)

    buffer.merge_chunk(old_actions, request_step=0, current_step=0, action_start=0, action_end=2, latency_steps=0)
    buffer.merge_chunk(new_actions, request_step=0, current_step=0, action_start=0, action_end=2, latency_steps=0)

    np.testing.assert_allclose(buffer.pop(0).action, [179.0])
    np.testing.assert_allclose(buffer.pop(1).action, [-180.0])
    np.testing.assert_allclose(buffer.pop(2).action, [-179.0])


def test_cyclic_ema_smoothing_uses_shortest_angle_path() -> None:
    buffer = _buffer(
        min_buffer_steps=0,
        action_smoothing="ema",
        action_ema_alpha=0.5,
        cyclic_indices=(0,),
    )
    actions = np.asarray([[179.0], [-179.0]], dtype=np.float64)
    buffer.merge_chunk(actions, request_step=0, current_step=0, action_start=0, action_end=1, latency_steps=0)

    np.testing.assert_allclose(buffer.pop(0).action, [179.0])
    np.testing.assert_allclose(buffer.pop(1).action, [-180.0])


def test_timed_action_json_serialization() -> None:
    record = TimedAction(
        chunk_id=1,
        action_index=2,
        target_step=12,
        action=np.asarray([1.0, 2.0]),
        merge_type="inserted",
        blend_weight=None,
        source_obs_id=3,
        latency_s=0.1,
        delay_steps=2,
    )

    encoded = to_jsonable(record)

    assert encoded["action"] == [1.0, 2.0]
    assert encoded["chunk_id"] == 1


def test_limit_action_position_rotation_and_gripper() -> None:
    previous = np.asarray([0.0, 0.0, 0.0, 179.0, 0.0, 0.0, 0.0], dtype=np.float64)
    action = np.asarray([0.03, 0.04, 0.0, -179.0, 0.0, 0.0, 1.0], dtype=np.float64)

    limited, info = limit_action_step(
        action,
        previous,
        max_position_step_m=0.01,
        max_rotation_step_deg=1.0,
        max_gripper_step=0.2,
    )

    assert info["limit_applied"]
    np.testing.assert_allclose(np.linalg.norm(limited[:3] - previous[:3]), 0.01)
    np.testing.assert_allclose(limited[3], -180.0)
    np.testing.assert_allclose(limited[6], 0.2)


def test_limit_action_no_limit_is_identity() -> None:
    previous = np.asarray([0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0], dtype=np.float64)
    action = np.asarray([1.0, 2.0, 3.0, 20.0, 30.0, 40.0, 1.0], dtype=np.float64)

    limited, info = limit_action_step(
        action,
        previous,
        max_position_step_m=0.0,
        max_rotation_step_deg=0.0,
        max_gripper_step=0.0,
    )

    assert not info["limit_applied"]
    np.testing.assert_allclose(limited, action)


def test_action_command_delta_wraps_rpy() -> None:
    previous = np.asarray([0.0, 0.0, 0.0, 179.0, 0.0, 0.0, 0.0], dtype=np.float64)
    action = np.asarray([0.1, 0.0, 0.0, -179.0, 0.0, 0.0, 0.2], dtype=np.float64)

    delta = action_command_delta(action, previous)

    np.testing.assert_allclose(delta[:3], [0.1, 0.0, 0.0])
    np.testing.assert_allclose(delta[3:6], [2.0, 0.0, 0.0])
    np.testing.assert_allclose(delta[6], 0.2)


def test_joint_cubic_trajectory_preserves_start_velocity_and_waypoint_timing() -> None:
    trajectory = plan_joint_cubic_trajectory(
        np.asarray([0.0, 0.0]),
        np.asarray([0.5, -0.25]),
        np.asarray([[1.0, 0.0], [2.0, 1.0]]),
        waypoint_dt_s=0.5,
        sample_hz=20.0,
    )

    np.testing.assert_allclose(trajectory.positions[0], [0.0, 0.0])
    np.testing.assert_allclose(trajectory.velocities[0], [0.5, -0.25])
    waypoint_index = int(np.where(np.isclose(trajectory.times_s, 0.5))[0][0])
    np.testing.assert_allclose(trajectory.positions[waypoint_index], [1.0, 0.0])
    np.testing.assert_allclose(trajectory.positions[-1], [2.0, 1.0])
    np.testing.assert_allclose(trajectory.velocities[-1], [0.0, 0.0])


def test_joint_cubic_trajectory_holds_position_until_start_delay() -> None:
    trajectory = plan_joint_cubic_trajectory(
        np.asarray([0.2]),
        np.asarray([0.3]),
        np.asarray([[0.8]]),
        waypoint_dt_s=0.5,
        sample_hz=20.0,
        start_delay_s=0.2,
    )

    np.testing.assert_allclose(trajectory.duration_s, 0.7)
    before_start_q, before_start_dq = trajectory.sample(0.19)
    np.testing.assert_allclose(before_start_q, [0.2])
    np.testing.assert_allclose(before_start_dq, [0.0])
    after_start_q, _after_start_dq = trajectory.sample(0.3)
    assert after_start_q[0] > 0.2


def test_joint_cubic_trajectory_rejects_velocity_limit_violation() -> None:
    with np.testing.assert_raises_regex(ValueError, "velocity"):
        plan_joint_cubic_trajectory(
            np.asarray([0.0]),
            np.asarray([0.0]),
            np.asarray([[1.0]]),
            waypoint_dt_s=0.1,
            sample_hz=100.0,
            max_velocity_rad_s=1.0,
        )
