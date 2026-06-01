from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import torch

from openpi.models import pi0
from openpi.models_pytorch import pi0_pytorch
from openpi.policies import policy as _policy


def test_rtc_condition_absolute_step_alignment_and_masks() -> None:
    previous = np.arange(12, dtype=np.float32).reshape(6, 2)

    result = _policy.build_rtc_action_condition(
        previous_actions=previous,
        previous_base_step=10,
        request_step=12,
        delay_steps=2,
        soft_horizon_steps=2,
        free_tail_steps=1,
        action_horizon=6,
        action_dim=2,
    )

    assert result.applied
    np.testing.assert_allclose(result.condition[0], previous[2])
    np.testing.assert_allclose(result.condition[1], previous[3])
    np.testing.assert_allclose(result.condition[2], previous[4])
    np.testing.assert_allclose(result.weight[:2], [1.0, 1.0])
    assert 0.0 < result.weight[2] <= 1.0
    assert result.weight[-1] == 0.0
    assert result.metadata["frozen_steps"] == 2
    assert result.metadata["condition_steps"] == 4


def test_rtc_condition_missing_previous_steps_are_unguided() -> None:
    previous = np.arange(6, dtype=np.float32).reshape(3, 2)

    result = _policy.build_rtc_action_condition(
        previous_actions=previous,
        previous_base_step=0,
        request_step=10,
        delay_steps=1,
        soft_horizon_steps=2,
        free_tail_steps=1,
        action_horizon=4,
        action_dim=2,
    )

    assert not result.applied
    assert result.metadata["skip_reason"] == "no_overlapping_previous_actions"
    np.testing.assert_allclose(result.weight, np.zeros(4))


def test_jax_action_condition_projection() -> None:
    x_t = jnp.ones((1, 2, 1), dtype=jnp.float32)
    noise = jnp.zeros_like(x_t)
    condition = jnp.asarray([[[5.0], [7.0]]], dtype=jnp.float32)
    weight = jnp.asarray([[1.0, 0.0]], dtype=jnp.float32)

    projected = pi0.apply_action_condition(
        x_t,
        time=jnp.asarray(0.0, dtype=jnp.float32),
        noise=noise,
        action_condition=condition,
        action_condition_weight=weight,
    )

    np.testing.assert_allclose(np.asarray(projected[0, 0]), [5.0])
    np.testing.assert_allclose(np.asarray(projected[0, 1]), [1.0])


def test_pytorch_action_condition_projection() -> None:
    x_t = torch.ones((1, 2, 1), dtype=torch.float32)
    noise = torch.zeros_like(x_t)
    condition = torch.tensor([[[5.0], [7.0]]], dtype=torch.float32)
    weight = torch.tensor([[1.0, 0.0]], dtype=torch.float32)

    projected = pi0_pytorch.apply_action_condition(
        x_t,
        time=torch.tensor(0.0, dtype=torch.float32),
        noise=noise,
        action_condition=condition,
        action_condition_weight=weight,
    )

    np.testing.assert_allclose(projected[0, 0].numpy(), [5.0])
    np.testing.assert_allclose(projected[0, 1].numpy(), [1.0])


class _FakeTorchPolicyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = type("Config", (), {"action_horizon": 6, "action_dim": 2})()
        self.conditions: list[torch.Tensor | None] = []
        self.weights: list[torch.Tensor | None] = []

    def sample_actions(self, device, observation, noise=None, num_steps=10, action_condition=None, action_condition_weight=None):
        del observation, noise, num_steps
        self.conditions.append(action_condition.detach().cpu() if action_condition is not None else None)
        self.weights.append(action_condition_weight.detach().cpu() if action_condition_weight is not None else None)
        return torch.arange(12, dtype=torch.float32, device=device).reshape(1, 6, 2)


def _fake_obs() -> dict:
    return {
        "state": np.zeros((2,), dtype=np.float32),
        "image": {"front": np.zeros((4, 4, 3), dtype=np.uint8)},
        "image_mask": {"front": np.asarray(True)},
    }


def test_policy_rtc_trims_returned_actions_and_sets_action_base_step() -> None:
    model = _FakeTorchPolicyModel()
    policy = _policy.Policy(
        model,
        is_pytorch=True,
        rtc_chunk_conditioning=True,
        rtc_soft_horizon_steps=2,
        rtc_free_tail_steps=1,
    )

    first = policy.infer(
        {
            **_fake_obs(),
            _policy.RTC_ROLLOUT_KEY: {"session_id": "s", "generation": 0, "request_step": 0, "delay_steps": 0},
        }
    )
    assert not first["rtc"]["applied"]
    assert first["actions"].shape == (6, 2)

    second = policy.infer(
        {
            **_fake_obs(),
            _policy.RTC_ROLLOUT_KEY: {"session_id": "s", "generation": 0, "request_step": 2, "delay_steps": 2},
        }
    )

    assert second["rtc"]["applied"]
    assert second["action_base_step"] == 4
    assert second["actions"].shape == (4, 2)
    assert model.conditions[-1] is not None
    assert model.weights[-1] is not None
    np.testing.assert_allclose(model.weights[-1].numpy()[0, :2], [1.0, 1.0])
