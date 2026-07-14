# OpenPI Async Runtime

A model-, transport-, and robot-agnostic runtime for asynchronous action-chunk inference. It owns the reusable timing and state machinery: latency compensation, absolute-step buffering, overlap blending, RTC chunk conditioning, safety limits, and independent inference/control loops.

The package has one runtime dependency (`numpy`) and does not import OpenPI, JAX, PyTorch, FastTouch, xArm, or a network client.

Install it from a checkout with `pip install -e .`, or use `uv sync --dev` to include the test tools.

Documentation:

- [Runtime API guide](docs/runtime_api.md) describes the reusable package boundary.
- [OpenPI async rollout guide](docs/async_rollout.md) preserves the complete xArm/FastTouch startup, parameter, timing, servo, and debugging notes.
- [OpenPI examples](examples/openpi/README.md) are hardware-integration snapshots and are not installed with the package.

## Integration boundary

Implement two small adapters:

```python
class Transport:
    def infer(self, observation):
        return websocket_policy.infer(observation)


class Robot:
    def observe(self):
        return {"state": read_state(), "image": read_images()}

    def execute(self, action, *, step, metadata=None):
        send_action(action)

    def reset(self): ...
    def close(self): ...
```

Then assemble the engine with the buffering and latency policies appropriate for the robot:

```python
from openpi_async_runtime import ActionBuffer, AsyncRolloutEngine, LatencyEstimator

engine = AsyncRolloutEngine(
    transport=Transport(),
    robot=Robot(),
    action_buffer=ActionBuffer(
        min_buffer_steps=2,
        blend_horizon_steps=5,
        blend_schedule="exp",
        empty_action_policy="hold",
        action_smoothing="off",
        action_ema_alpha=0.5,
    ),
    latency_estimator=LatencyEstimator(
        mode="ema",
        fixed_steps=0,
        control_hz=20.0,
        ema_alpha=0.5,
    ),
)
engine.start()
```

`run_inference_once()` and `run_control_once()` expose the same engine without threads, which makes hardware adapters deterministic to test. An optional `TrajectoryInstaller` can atomically install joint-space trajectories, while an `EventSink` can route structured runtime events to JSONL, metrics, or a UI.

OpenPI-specific model conditioning remains integration code in the development repository. Reference snapshots of the hardware clients are kept under `examples/openpi`; the runtime package itself remains independently installable and does not import them.

## Repository history

This repository was extracted from the OpenPI development repository with path-filtered Git history. The reusable buffer began in the original async rollout scripts. See [the OpenPI integration notes](integrations/openpi/README.md) for the adapter boundary and the `examples/openpi` snapshots for concrete hardware wiring.
