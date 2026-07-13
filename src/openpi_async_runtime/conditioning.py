"""Backend-neutral action-conditioning math."""

from __future__ import annotations

from typing import Any


def project_action_condition(
    x_t: Any,
    *,
    time: Any,
    noise: Any,
    action_condition: Any | None,
    action_condition_weight: Any | None,
) -> Any:
    """Project selected action positions toward a flow-matching bridge.

    The implementation intentionally relies only on array operators shared by
    NumPy, JAX, and PyTorch, keeping the runtime free of framework dependencies.
    """
    if action_condition is None or action_condition_weight is None:
        return x_t
    time_expanded = time.reshape((1, 1, 1)) if getattr(time, "ndim", 0) == 0 else time[:, None, None]
    weight = action_condition_weight
    if weight.ndim == 2:
        weight = weight[..., None]
    conditioned = time_expanded * noise + (1.0 - time_expanded) * action_condition
    return x_t * (1.0 - weight) + conditioned * weight
