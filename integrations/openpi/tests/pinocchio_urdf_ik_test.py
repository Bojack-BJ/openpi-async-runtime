from __future__ import annotations

import pathlib
import sys
import types
import xml.etree.ElementTree as ET

import numpy as np

from scripts.rollout.pinocchio_urdf_ik import normalize_xarm_tcp_offset
from scripts.rollout.pinocchio_urdf_ik import parse_tcp_offset_mm_rpy_deg
from scripts.rollout.pinocchio_urdf_ik import PinocchioUrdfIK
from scripts.rollout.pinocchio_urdf_ik import rpy_deg_to_rotation_matrix
from scripts.rollout.pinocchio_urdf_ik import tcp_pose_to_flange_pose_matrix


URDF_PATH = pathlib.Path(__file__).parents[1] / "rollout" / "assets" / "xarm6_kinematics.urdf"


def test_parse_explicit_tcp_offset_converts_mm_to_meters() -> None:
    offset = parse_tcp_offset_mm_rpy_deg("100,-20,30,1,2,3")

    np.testing.assert_allclose(offset, [0.1, -0.02, 0.03, 1.0, 2.0, 3.0])


def test_normalize_xarm_tcp_offset_converts_radians_to_degrees() -> None:
    offset = normalize_xarm_tcp_offset([100.0, 0.0, 0.0, np.pi / 2.0, 0.0, 0.0], angles_are_radian=True)

    np.testing.assert_allclose(offset, [0.1, 0.0, 0.0, 90.0, 0.0, 0.0])


def test_tcp_target_is_converted_to_flange_target() -> None:
    flange_target = tcp_pose_to_flange_pose_matrix(
        [0.3, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
    )

    np.testing.assert_allclose(flange_target[:3, 3], [0.2, 0.0, 0.0])


def test_rpy_rotation_uses_xyz_convention() -> None:
    rotation = rpy_deg_to_rotation_matrix([0.0, 0.0, 90.0])

    np.testing.assert_allclose(rotation @ np.asarray([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-8)


def test_vendored_xarm6_urdf_has_six_revolute_joints() -> None:
    root = ET.parse(URDF_PATH).getroot()
    revolute_joints = [joint.attrib["name"] for joint in root.findall("joint") if joint.attrib["type"] == "revolute"]

    assert revolute_joints == ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]


def test_pinocchio_solver_moves_seed_toward_target() -> None:
    class FakeSE3:
        def __init__(self, rotation, translation):
            self.rotation = np.asarray(rotation, dtype=np.float64)
            self.translation = np.asarray(translation, dtype=np.float64)

        def inverse(self):
            rotation = self.rotation.T
            return FakeSE3(rotation, -(rotation @ self.translation))

        def __mul__(self, other):
            return FakeSE3(self.rotation @ other.rotation, self.translation + self.rotation @ other.translation)

        def actInv(self, other):
            return self.inverse() * other

    class FakeModel:
        nq = 6
        nv = 6
        frames = [object()]
        lowerPositionLimit = np.full(6, -2.0)
        upperPositionLimit = np.full(6, 2.0)

        def getFrameId(self, tip_link):
            assert tip_link == "link6"
            return 0

        def createData(self):
            return types.SimpleNamespace(oMf=[FakeSE3(np.eye(3), np.zeros(3))])

    def forward_kinematics(_model, data, q):
        data.oMf[0] = FakeSE3(np.eye(3), np.asarray([q[0], q[1], q[2]]))

    fake_pin = types.SimpleNamespace(
        SE3=FakeSE3,
        ReferenceFrame=types.SimpleNamespace(LOCAL=object()),
        buildModelFromUrdf=lambda _path: FakeModel(),
        forwardKinematics=forward_kinematics,
        updateFramePlacements=lambda _model, _data: None,
        log6=lambda transform: types.SimpleNamespace(vector=np.concatenate([transform.translation, np.zeros(3)])),
        computeFrameJacobian=lambda _model, _data, _q, _frame_id, _reference: np.eye(6),
        integrate=lambda _model, q, velocity: q + velocity,
    )
    previous_pin = sys.modules.get("pinocchio")
    sys.modules["pinocchio"] = fake_pin
    try:
        solver = PinocchioUrdfIK(
            URDF_PATH,
            tip_link="link6",
            max_iterations=10,
            tolerance=1e-5,
            damping=1e-6,
            step_size=1.0,
        )
        solution = solver.solve([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], q_seed_rad=np.zeros(6))
    finally:
        if previous_pin is None:
            del sys.modules["pinocchio"]
        else:
            sys.modules["pinocchio"] = previous_pin

    np.testing.assert_allclose(solution[:3], [1.0, 0.0, 0.0], atol=1e-5)
