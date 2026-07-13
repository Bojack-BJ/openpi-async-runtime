"""Stable interfaces between the async engine, policies, and robots."""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
from typing import Any, Protocol, runtime_checkable

import numpy as np


@dataclasses.dataclass(frozen=True)
class ActionChunk:
    """A policy response placed on an absolute control-step timeline."""

    actions: np.ndarray
    request_step: int
    base_step: int | None = None
    latency_s: float = 0.0
    metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)


@runtime_checkable
class PolicyTransport(Protocol):
    """Transport capable of returning an action chunk for an observation."""

    def infer(self, observation: dict[str, Any]) -> Mapping[str, Any]: ...


@runtime_checkable
class RobotBackend(Protocol):
    """Minimal robot adapter used by the generic control engine."""

    def observe(self) -> dict[str, Any]: ...

    def execute(self, action: np.ndarray, *, step: int, metadata: Mapping[str, Any] | None = None) -> None: ...

    def reset(self) -> None: ...

    def close(self) -> None: ...


class TrajectoryInstaller(Protocol):
    """Optional backend that atomically replaces the executable future suffix."""

    def install(self, actions: np.ndarray, *, base_step: int, current_step: int) -> Mapping[str, Any]: ...


class EventSink(Protocol):
    """Receives structured runtime events without constraining storage."""

    def emit(self, event: str, payload: Mapping[str, Any]) -> None: ...
