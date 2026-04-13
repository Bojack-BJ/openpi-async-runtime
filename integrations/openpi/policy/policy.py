from collections.abc import Sequence
import logging
import pathlib
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

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
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

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
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
