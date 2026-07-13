# OpenPI integration

The standalone runtime deliberately does not import OpenPI. The development repository connects the two projects at three narrow boundaries:

1. OpenPI's policy client implements `PolicyTransport.infer`.
2. Each robot SDK implements `RobotBackend.observe`, `execute`, `reset`, and `close`; an optional planner implements `TrajectoryInstaller.install`.
3. OpenPI's policy server uses `RTCSessionConditioner`, while the Pi0 JAX and PyTorch samplers call `project_action_condition`.

The historical FastTouch/xArm clients, policy wiring, server flags, URDF helper, and debugging tool are retained in this repository's earlier commits. They are not kept on `main` because they require the full OpenPI tree and vendor SDKs. Production adapters should live with their owning model or robot repository and depend on `openpi-async-runtime` as a normal package.
