from collections.abc import Sequence
import dataclasses
import logging
import pathlib
import threading
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
from PIL import Image
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy
_BILINEAR_RESAMPLING = getattr(Image, "Resampling", Image).BILINEAR
RTC_ROLLOUT_KEY = "__rtc_rollout"


@dataclasses.dataclass(frozen=True)
class RTCConditionResult:
    condition: np.ndarray
    weight: np.ndarray
    applied: bool
    metadata: dict[str, Any]


@dataclasses.dataclass
class _RTCChunkState:
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
    """Build the frozen/soft/free RTC inpainting mask in normalized action space."""
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
        return RTCConditionResult(condition=condition, weight=weight, applied=False, metadata=metadata)
    if previous_actions is None or previous_base_step is None:
        metadata["skip_reason"] = "no_previous_chunk"
        return RTCConditionResult(condition=condition, weight=weight, applied=False, metadata=metadata)

    previous_actions = np.asarray(previous_actions, dtype=np.float32)
    soft_end = max(action_horizon - free_tail_steps, delay_steps)
    previous_len = previous_actions.shape[0]
    previous_dim = previous_actions.shape[-1]
    copy_dim = min(action_dim, previous_dim)
    for action_index in range(action_horizon):
        if action_index >= soft_end:
            continue
        previous_index = int(request_step) + action_index - int(previous_base_step)
        if previous_index < 0 or previous_index >= previous_len:
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
    return RTCConditionResult(condition=condition, weight=weight, applied=applied, metadata=metadata)


class VisualIntermediateRecorder:
    """Saves visual encoder feature maps during inference."""

    def __init__(self, output_dir: str | pathlib.Path, *, feature_name: str = "pre_logits_2d"):
        self._output_dir = pathlib.Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._feature_name = feature_name
        self._step = 0

    def record(
        self,
        model: _model.BaseModel | torch.nn.Module,
        observation: _model.Observation,
        *,
        is_pytorch: bool,
    ) -> None:
        step_dir = self._output_dir / f"step{self._step}"
        step_dir.mkdir(parents=True, exist_ok=True)

        for image_name, image_value in observation.images.items():
            feature_map = self._extract_feature_map(model, image_value, image_name, is_pytorch=is_pytorch)
            image = self._to_uint8_image(self._to_numpy(image_value[0]))
            heatmap = self._feature_map_to_heatmap(feature_map)
            overlay = self._overlay_heatmap(image, heatmap)

            np.save(step_dir / f"{image_name}_{self._feature_name}.npy", feature_map.astype(np.float32))
            Image.fromarray(image).save(step_dir / f"{image_name}_input.png")
            Image.fromarray(heatmap).save(step_dir / f"{image_name}_{self._feature_name}.png")
            Image.fromarray(overlay).save(step_dir / f"{image_name}_{self._feature_name}_overlay.png")

        self._step += 1

    def _extract_feature_map(
        self,
        model: _model.BaseModel | torch.nn.Module,
        image_value: jax.Array | torch.Tensor,
        image_name: str,
        *,
        is_pytorch: bool,
    ) -> np.ndarray:
        if is_pytorch:
            feature = model.paligemma_with_expert.embed_image(image_value)
            return self._to_spatial_map(self._to_numpy(feature[0]))

        _, intermediates = model.PaliGemma.img(image_value, train=False)
        feature = intermediates.get(self._feature_name)
        if feature is None:
            raise ValueError(
                f"Feature '{self._feature_name}' is not available for image '{image_name}'. "
                f"Available keys: {sorted(intermediates)}"
            )
        return self._to_spatial_map(self._to_numpy(feature[0]))

    def _to_spatial_map(self, feature: np.ndarray) -> np.ndarray:
        if feature.ndim == 3:
            return feature
        if feature.ndim == 2:
            num_tokens, channels = feature.shape
            side = int(round(np.sqrt(num_tokens)))
            if side * side != num_tokens:
                raise ValueError(f"Expected square number of visual tokens, got {num_tokens}.")
            return feature.reshape(side, side, channels)
        raise ValueError(f"Unsupported feature shape: {feature.shape}")

    def _feature_map_to_heatmap(self, feature_map: np.ndarray) -> np.ndarray:
        intensity = np.linalg.norm(feature_map, axis=-1)
        intensity -= intensity.min()
        max_value = intensity.max()
        if max_value > 0:
            intensity /= max_value
        intensity = np.asarray(intensity * 255.0, dtype=np.uint8)

        heat_r = intensity
        heat_g = np.zeros_like(intensity)
        heat_b = 255 - intensity
        heatmap = np.stack([heat_r, heat_g, heat_b], axis=-1)
        return np.asarray(Image.fromarray(heatmap).resize(_model.IMAGE_RESOLUTION[::-1], _BILINEAR_RESAMPLING))

    def _overlay_heatmap(self, image: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
        overlay = 0.6 * image.astype(np.float32) + 0.4 * heatmap.astype(np.float32)
        return np.asarray(np.clip(overlay, 0, 255), dtype=np.uint8)

    def _to_uint8_image(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 3 and image.shape[0] in (1, 3):
            image = np.transpose(image, (1, 2, 0))
        if image.dtype == np.uint8:
            return image
        image = np.asarray((image + 1.0) * 127.5, dtype=np.float32)
        return np.asarray(np.clip(image, 0, 255), dtype=np.uint8)

    def _to_numpy(self, value: jax.Array | torch.Tensor | np.ndarray) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().float().numpy()
        return np.asarray(value)


def _to_pytorch_input_leaf(x, device: str):
    array = np.asarray(x)
    if array.dtype.kind in {"U", "S", "O"}:
        return array[None, ...]
    return torch.from_numpy(array).to(device)[None, ...]


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
        intermediate_recorder: VisualIntermediateRecorder | None = None,
        rtc_chunk_conditioning: bool = False,
        rtc_delay_steps: int = -1,
        rtc_soft_horizon_steps: int = 5,
        rtc_free_tail_steps: int = 5,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device
        self._intermediate_recorder = intermediate_recorder
        self._rtc_chunk_conditioning = bool(rtc_chunk_conditioning)
        self._rtc_delay_steps = int(rtc_delay_steps)
        self._rtc_soft_horizon_steps = max(int(rtc_soft_horizon_steps), 1)
        self._rtc_free_tail_steps = max(int(rtc_free_tail_steps), 0)
        self._rtc_state: _RTCChunkState | None = None
        self._rtc_lock = threading.Lock()
        self._metadata = {
            **self._metadata,
            "rtc": {
                "enabled": self._rtc_chunk_conditioning,
                "delay_steps": self._rtc_delay_steps,
                "soft_horizon_steps": self._rtc_soft_horizon_steps,
                "free_tail_steps": self._rtc_free_tail_steps,
            },
        }

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    def _to_backend_sample_array(self, value: np.ndarray, *, add_batch_dim: bool = True):
        value = np.asarray(value, dtype=np.float32)
        if add_batch_dim:
            value = value[None, ...]
        if self._is_pytorch_model:
            return torch.from_numpy(value).to(self._pytorch_device)
        return jnp.asarray(value)

    def _prepare_rtc_condition(self, rtc_meta: dict[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any] | None]:
        if not self._rtc_chunk_conditioning or not isinstance(rtc_meta, dict) or not rtc_meta.get("enabled", True):
            return {}, None

        request_step = int(rtc_meta.get("request_step", 0))
        session_id = str(rtc_meta.get("session_id") or "default")
        generation = int(rtc_meta.get("generation", 0))
        delay_steps = int(rtc_meta.get("delay_steps", self._rtc_delay_steps))
        if self._rtc_delay_steps >= 0:
            delay_steps = self._rtc_delay_steps
        soft_horizon_steps = int(rtc_meta.get("soft_horizon_steps", self._rtc_soft_horizon_steps))
        free_tail_steps = int(rtc_meta.get("free_tail_steps", self._rtc_free_tail_steps))

        with self._rtc_lock:
            previous = self._rtc_state
            if previous is not None and (previous.session_id != session_id or previous.generation != generation):
                previous = None
                self._rtc_state = None
            previous_actions = None if previous is None else previous.actions
            previous_base_step = None if previous is None else previous.base_step

        model_config = getattr(self._model, "config", None)
        action_horizon = int(
            getattr(self._model, "action_horizon", 0)
            or getattr(model_config, "action_horizon", 0)
            or (previous_actions.shape[0] if previous_actions is not None else 0)
        )
        action_dim = int(
            getattr(self._model, "action_dim", 0)
            or getattr(model_config, "action_dim", 0)
            or (previous_actions.shape[-1] if previous_actions is not None else 0)
        )
        result = build_rtc_action_condition(
            previous_actions=previous_actions,
            previous_base_step=previous_base_step,
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
        }
        sample_kwargs = {}
        if result.applied:
            sample_kwargs["action_condition"] = self._to_backend_sample_array(result.condition)
            sample_kwargs["action_condition_weight"] = self._to_backend_sample_array(result.weight)
        response["_state_update"] = {
            "session_id": session_id,
            "generation": generation,
            "base_step": request_step,
            "delay_steps": max(delay_steps, 0),
            "applied": result.applied,
        }
        return sample_kwargs, response

    def _finalize_rtc_outputs(self, raw_actions: np.ndarray, rtc_response: dict[str, Any] | None) -> tuple[np.ndarray, dict[str, Any] | None]:
        if rtc_response is None:
            return raw_actions, None
        state_update = rtc_response.pop("_state_update")
        full_actions = np.asarray(raw_actions, dtype=np.float32).copy()
        with self._rtc_lock:
            self._rtc_state = _RTCChunkState(
                session_id=str(state_update["session_id"]),
                generation=int(state_update["generation"]),
                base_step=int(state_update["base_step"]),
                actions=full_actions,
            )
        if rtc_response.get("applied", False):
            delay_steps = int(state_update["delay_steps"])
            if delay_steps >= len(raw_actions):
                rtc_response["applied"] = False
                rtc_response["skip_reason"] = "delay_ge_returned_horizon"
                return raw_actions[:0], rtc_response
            rtc_response["action_base_step"] = int(state_update["base_step"]) + delay_steps
            return raw_actions[delay_steps:], rtc_response
        rtc_response["action_base_step"] = int(state_update["base_step"])
        return raw_actions, rtc_response

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        rtc_meta = inputs.pop(RTC_ROLLOUT_KEY, None) if isinstance(inputs, dict) else None
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            if isinstance(inputs, dict) and "raw_prompt" in inputs:
                inputs = {k: v for k, v in inputs.items() if k != "raw_prompt"}
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: _to_pytorch_input_leaf(x, self._pytorch_device), inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        rtc_sample_kwargs, rtc_response = self._prepare_rtc_condition(rtc_meta)
        sample_kwargs.update(rtc_sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        if self._intermediate_recorder is not None:
            self._intermediate_recorder.record(
                self._model,
                observation,
                is_pytorch=self._is_pytorch_model,
            )
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        outputs["actions"], rtc_response = self._finalize_rtc_outputs(outputs["actions"], rtc_response)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        if rtc_response is not None:
            outputs["rtc"] = rtc_response
            outputs["action_base_step"] = rtc_response.get("action_base_step")
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
