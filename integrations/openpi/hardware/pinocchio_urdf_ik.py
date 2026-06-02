from __future__ import annotations

import pathlib

import numpy as np


def rpy_deg_to_rotation_matrix(rpy_deg) -> np.ndarray:
    roll, pitch, yaw = np.deg2rad(np.asarray(rpy_deg, dtype=np.float64))
    sr, cr = np.sin(roll), np.cos(roll)
    sp, cp = np.sin(pitch), np.cos(pitch)
    sy, cy = np.sin(yaw), np.cos(yaw)
    return np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def parse_tcp_offset_mm_rpy_deg(text: str) -> np.ndarray:
    values = np.asarray([float(part.strip()) for part in str(text).split(",")], dtype=np.float64)
    if values.shape != (6,):
        raise ValueError(f"Expected TCP offset x_mm,y_mm,z_mm,roll_deg,pitch_deg,yaw_deg, got: {text!r}")
    values[:3] /= 1000.0
    return values


def normalize_xarm_tcp_offset(raw_offset, *, angles_are_radian: bool) -> np.ndarray:
    values = np.asarray(raw_offset, dtype=np.float64).copy()
    if values.shape != (6,):
        raise ValueError(f"Expected xArm tcp_offset shape (6,), got {values.shape}")
    values[:3] /= 1000.0
    if angles_are_radian:
        values[3:] = np.rad2deg(values[3:])
    return values


def pose_xyz_m_rpy_deg_to_matrix(pose) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (6,):
        raise ValueError(f"Expected pose shape (6,), got {pose.shape}")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rpy_deg_to_rotation_matrix(pose[3:])
    transform[:3, 3] = pose[:3]
    return transform


def tcp_pose_to_flange_pose_matrix(tcp_pose_xyz_m_rpy_deg, tcp_offset_xyz_m_rpy_deg) -> np.ndarray:
    return pose_xyz_m_rpy_deg_to_matrix(tcp_pose_xyz_m_rpy_deg) @ np.linalg.inv(
        pose_xyz_m_rpy_deg_to_matrix(tcp_offset_xyz_m_rpy_deg)
    )


class PinocchioUrdfIK:
    """Seeded local-frame damped least-squares IK loaded from a URDF chain."""

    def __init__(
        self,
        urdf_path: str | pathlib.Path,
        *,
        tip_link: str,
        max_iterations: int,
        tolerance: float,
        damping: float,
        step_size: float,
        tcp_offset_xyz_m_rpy_deg: np.ndarray | None = None,
    ) -> None:
        try:
            import pinocchio as pin
        except ImportError as exc:
            raise RuntimeError(
                "Pinocchio IK backend requires the optional `pin` package. "
                "Install it on the rollout host with `python -m pip install pin`."
            ) from exc

        urdf_path = pathlib.Path(urdf_path).expanduser().resolve()
        if not urdf_path.is_file():
            raise FileNotFoundError(f"Pinocchio URDF does not exist: {urdf_path}")
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        if tolerance <= 0.0:
            raise ValueError("tolerance must be > 0")
        if damping <= 0.0:
            raise ValueError("damping must be > 0")
        if step_size <= 0.0:
            raise ValueError("step_size must be > 0")

        model = pin.buildModelFromUrdf(str(urdf_path))
        frame_id = int(model.getFrameId(tip_link))
        if frame_id >= len(model.frames):
            raise ValueError(f"URDF has no tip link frame {tip_link!r}: {urdf_path}")
        if model.nq != 6 or model.nv != 6:
            raise ValueError(f"Expected a fixed-base 6-DoF URDF, got nq={model.nq}, nv={model.nv}: {urdf_path}")

        self._pin = pin
        self._model = model
        self._data = model.createData()
        self._frame_id = frame_id
        self._max_iterations = int(max_iterations)
        self._tolerance = float(tolerance)
        self._damping = float(damping)
        self._step_size = float(step_size)
        tcp_offset = (
            np.zeros(6, dtype=np.float64)
            if tcp_offset_xyz_m_rpy_deg is None
            else np.asarray(tcp_offset_xyz_m_rpy_deg, dtype=np.float64)
        )
        if tcp_offset.shape != (6,):
            raise ValueError(f"Expected TCP offset shape (6,), got {tcp_offset.shape}")
        self.urdf_path = urdf_path
        self.tip_link = str(tip_link)
        self.tcp_offset_xyz_m_rpy_deg = tcp_offset.copy()

    def solve(self, pose_xyz_m_rpy_deg, *, q_seed_rad: np.ndarray) -> np.ndarray:
        pose = np.asarray(pose_xyz_m_rpy_deg, dtype=np.float64)
        q = np.asarray(q_seed_rad, dtype=np.float64).copy()
        if pose.shape != (6,):
            raise ValueError(f"Expected pose shape (6,), got {pose.shape}")
        if q.shape != (self._model.nq,):
            raise ValueError(f"Expected seed shape ({self._model.nq},), got {q.shape}")

        q = np.clip(q, self._model.lowerPositionLimit, self._model.upperPositionLimit)
        target_matrix = tcp_pose_to_flange_pose_matrix(pose, self.tcp_offset_xyz_m_rpy_deg)
        target = self._pin.SE3(target_matrix[:3, :3], target_matrix[:3, 3])
        residual = np.inf
        for _ in range(self._max_iterations):
            self._pin.forwardKinematics(self._model, self._data, q)
            self._pin.updateFramePlacements(self._model, self._data)
            frame_error = self._data.oMf[self._frame_id].actInv(target)
            error = self._pin.log6(frame_error).vector
            residual = float(np.linalg.norm(error))
            if residual < self._tolerance:
                return q
            jacobian = self._pin.computeFrameJacobian(
                self._model,
                self._data,
                q,
                self._frame_id,
                self._pin.ReferenceFrame.LOCAL,
            )
            velocity = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + self._damping * np.eye(6),
                error,
            )
            q = self._pin.integrate(self._model, q, velocity * self._step_size)
            q = np.clip(q, self._model.lowerPositionLimit, self._model.upperPositionLimit)
        raise RuntimeError(
            f"Pinocchio IK did not converge after {self._max_iterations} iterations: residual={residual:.6g}"
        )
