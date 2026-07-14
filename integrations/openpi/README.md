# OpenPI integration

The standalone runtime deliberately does not import OpenPI. The development repository connects the two projects at three narrow boundaries:

1. OpenPI's policy client implements `PolicyTransport.infer`.
2. Each robot SDK implements `RobotBackend.observe`, `execute`, `reset`, and `close`; an optional planner implements `TrajectoryInstaller.install`.
3. OpenPI's policy server uses `RTCSessionConditioner`, while the Pi0 JAX and PyTorch samplers call `project_action_condition`.

Reference snapshots of the FastTouch/xArm clients, URDF helper, depth client, and debugging tool live under [`examples/openpi`](../../examples/openpi/README.md). They require OpenPI's policy server/client and vendor SDKs, and are deliberately excluded from the installable runtime package. Production adapters should live with their owning model or robot repository and depend on `openpi-async-runtime` as a normal package.
