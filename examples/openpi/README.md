# OpenPI Hardware Examples

These files are reference snapshots of the OpenPI async rollout clients at OpenPI commit `26f8857`. They show complete xArm, FastTouch, and depth-camera integrations while keeping vendor and OpenPI dependencies out of the installable `openpi_async_runtime` package.

They are examples, not hardware-agnostic demos. Running them can move a real robot.

## Layout

```text
fasttouch/
  pi0_rollout_client_fasttouch_rpy.py
  pi0_rollout_client_fasttouch_rpy_async.py
  tcp_compensation.py
xarm/
  pi0_rollout_client_xarm_rpy.py
  pi0_rollout_client_xarm_rpy_async.py
  pinocchio_urdf_ik.py
  assets/xarm6_kinematics.urdf
depth_xarm/
  rollout_multi.py
tools/
  plot_async_rollout_debug.py
```

The async FastTouch and xArm clients import their colocated synchronous client as `base`; both files are therefore required. The standalone runtime itself is imported from `openpi_async_runtime`.

## Shared requirements

- an installed `openpi-async-runtime` checkout;
- `openpi_client` from OpenPI;
- NumPy, SciPy, and OpenCV;
- a compatible OpenPI policy server;
- Linux camera and terminal interfaces used by the scripts.

The OpenPI policy server is intentionally not copied here because model loading and policy configuration remain owned by OpenPI.

## FastTouch

FastTouch additionally requires the vendor `startouchclass.SingleArm` module and the correct CAN interfaces. Put the vendor interface directory on `PYTHONPATH`, then run:

```bash
python examples/openpi/fasttouch/pi0_rollout_client_fasttouch_rpy_async.py \
  --description "your task" \
  --left_can can0 \
  --right_can can1
```

Review camera device IDs, server address, port, TCP compensation, arm selection, and action units before enabling motion.

## xArm

xArm requires the deployed BestMan/xArm SDK environment. The synchronous base snapshot contains a site-specific BestMan import path that must be changed for the target machine.

```bash
python examples/openpi/xarm/pi0_rollout_client_xarm_rpy_async.py \
  --description "your task" \
  --xarm_control_mode position
```

Begin in position mode with conservative rate and buffer settings. Switch to servo or plan-servo only after verifying state/action conventions and emergency-stop behavior. The full parameter sequence is documented in [`docs/async_rollout.md`](../../docs/async_rollout.md).

## Depth xArm

`depth_xarm/rollout_multi.py` additionally requires the xArm Python SDK and a local Depth Anything V2 checkout/checkpoint. Its observation schema must match the policy checkpoint.

## Debug plots

```bash
python examples/openpi/tools/plot_async_rollout_debug.py \
  --debug-dir /tmp/openpi_async_debug/xarm_run01
```

## Synchronization policy

For now, OpenPI's `scripts/rollout` remains the deployment source and these files are a reviewable example snapshot. When changing the deployed client, update the corresponding snapshot in the same change and record the source OpenPI commit here. A future refactor can replace both copies with thin robot adapters around `AsyncRolloutEngine`.
