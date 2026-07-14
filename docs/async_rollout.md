# Asynchronous Rollout Runtime

`openpi-async-runtime` separates policy inference from fixed-rate robot control. The package owns reusable timing and state machinery; camera capture, policy transport, robot SDK calls, and OpenPI model code stay in adapters.

## Runtime boundary

```text
observation source -> PolicyTransport -> action chunk
                                      -> ActionBuffer
control clock -> RobotBackend <- action for absolute step
```

The reusable package provides:

- absolute-step action buffering;
- fixed, instantaneous, or EMA latency compensation;
- overlap blending and optional EMA output smoothing;
- independent inference and control loops;
- RTC chunk conditioning;
- safety limits and structured runtime events;
- deterministic single-step APIs for tests.

It deliberately does not import OpenPI, robot SDKs, JAX, PyTorch, or a websocket client.

## Minimal integration

Implement a transport and robot adapter:

```python
class Transport:
    def infer(self, observation):
        return policy_client.infer(observation)


class Robot:
    def observe(self):
        return {"state": read_state(), "images": read_images()}

    def execute(self, action, *, step, metadata=None):
        send_action(action)

    def reset(self): ...
    def close(self): ...
```

Then assemble the engine:

```python
from openpi_async_runtime import ActionBuffer, AsyncRolloutEngine, LatencyEstimator

engine = AsyncRolloutEngine(
    transport=Transport(),
    robot=Robot(),
    action_buffer=ActionBuffer(
        min_buffer_steps=2,
        blend_horizon_steps=8,
        blend_schedule="exp",
        empty_action_policy="hold",
        action_smoothing="off",
        action_ema_alpha=0.5,
    ),
    latency_estimator=LatencyEstimator(
        mode="fixed",
        fixed_steps=0,
        control_hz=20.0,
        ema_alpha=0.5,
    ),
)
engine.start()
```

Use `run_inference_once()` and `run_control_once()` for deterministic adapter tests without background threads.

## Time alignment

Inference records the control step and monotonic timestamp at observation capture. When the action chunk returns, the runtime maps every action to an absolute control step.

```text
target_step = request_step + action_index
```

Latency compensation may skip actions that correspond to already elapsed control steps:

| Mode | Delay estimate | Recommended use |
| --- | --- | --- |
| `fixed` | configured step count | hardware bring-up and reproducible comparisons |
| `instant` | current request latency | variable remote inference |
| `ema` | smoothed request latency | stable production latency |

Start hardware tuning with `fixed_steps=0`. Increase it only after logs show a consistent phase lag. Cap dynamic compensation so a latency spike cannot discard most of a chunk.

## Buffer and blending

`min_buffer_steps` freezes actions close to execution so a newly returned chunk cannot overwrite the control thread's immediate future. When chunks overlap, `blend_horizon_steps` blends old and new actions with the selected schedule.

Recommended bring-up order:

1. Position control, fixed delay zero, smoothing off.
2. Confirm the buffer rarely reports missing actions.
3. Enable overlap blending.
4. Switch to servo or joint-waypoint execution.
5. Add latency compensation, then output smoothing if needed.

If the buffer is empty, use `hold` during initial hardware tests. More aggressive fallback behavior belongs in the robot adapter because safe behavior is hardware-specific.

## RTC conditioning

RTC conditioning projects a raw model chunk against already committed actions before installation. The package exposes framework-neutral conditioning and RTC session/client adapters; model-specific forward passes remain outside the runtime.

Keep these concerns separate:

- model conditioning decides what chunk should be proposed;
- latency estimation decides which prefix is already stale;
- the buffer decides which absolute steps may still be replaced;
- the robot adapter decides how actions become SDK commands.

## OpenPI integration

OpenPI's xArm and FastTouch clients remain in `scripts/rollout` of the development repository. They provide cameras, websocket transport, robot SDK execution, plan-servo behavior, and hardware debug output while importing the reusable runtime from this package.

Initialize the submodule from OpenPI with:

```bash
git submodule update --init --recursive
```

## Validation

Run the standalone tests with:

```bash
uv run --python 3.11 pytest -q tests/test_runtime.py
```

Hardware adapters should additionally test observation shape, action units, empty-buffer behavior, reset behavior, and emergency-stop handling on the target robot.
