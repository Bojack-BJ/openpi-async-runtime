#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Async RTC-style rollout client for xArm.

This script leaves the legacy synchronous xArm rollout untouched and replaces
the rollout loop with an inference thread, fixed-rate control thread, and
RTC-style action buffer.
"""

from __future__ import annotations

import argparse
import contextlib
import pathlib
import sys
import termios
import threading
import time
import tty
import uuid

import numpy as np

from async_rollout_core import ActionBuffer
from async_rollout_core import AsyncDebugWriter
from async_rollout_core import ExecutedAction
from async_rollout_core import LatencyEstimator
from async_rollout_core import TimedAction
from async_rollout_core import TimedObservation
from async_rollout_core import active_joint_vector
from async_rollout_core import action_command_delta
from async_rollout_core import action_tracking_error
from async_rollout_core import align_joint_waypoints_to_install_step
from async_rollout_core import call_with_supported_optional_kwargs
from async_rollout_core import command_stream_handoff_state
from async_rollout_core import limit_action_step
from async_rollout_core import max_joint_waypoint_delta
from async_rollout_core import plan_joint_cubic_trajectory
from async_rollout_core import prepare_live_handoff_actions
from async_rollout_core import should_advance_control_step
from pinocchio_urdf_ik import PinocchioUrdfIK
from pinocchio_urdf_ik import normalize_xarm_tcp_offset
from pinocchio_urdf_ik import parse_tcp_offset_mm_rpy_deg
import pi0_rollout_client_xarm_rpy as base


GRIPPER_UPDATE_INTERVAL_S = 0.1
GRIPPER_UPDATE_THRESHOLD = 0.02
XARM_SINGLE_RPY_ACTION_INDICES = (3, 4, 5)
XARM_DUAL_RPY_ACTION_INDICES = (3, 4, 5, 10, 11, 12)
_unsupported_xarm_ik_kwargs: set[str] = set()
_warned_xarm_ik_dropped_kwargs: set[str] = set()
DEFAULT_XARM6_URDF = pathlib.Path(__file__).with_name("assets") / "xarm6_kinematics.urdf"


def _add_async_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--control_hz", type=float, default=20.0, help="Fixed robot command frequency in Hz")
    parser.add_argument("--inference_interval_steps", type=int, default=16, help="Try one policy inference every N control steps")
    parser.add_argument("--min_buffer_steps", type=int, default=4, help="Do not overwrite actions this close to execution")
    parser.add_argument("--empty_action_policy", choices=("hold", "none"), default="hold", help="Behavior when no action exists for current step")
    parser.add_argument("--inference_delay_mode", choices=("fixed", "instant", "ema"), default="instant", help="How to convert policy latency to skipped action steps")
    parser.add_argument("--inference_delay_steps", type=int, default=0, help="Fixed delay steps used when --inference_delay_mode fixed")
    parser.add_argument("--max_inference_delay_steps", type=int, default=4, help="Upper bound for dynamic delay compensation; <0 disables the cap")
    parser.add_argument("--reset_delay_on_empty_buffer", action=argparse.BooleanOptionalAction, default=True, help="Use zero delay after startup/buffer underrun to avoid jumping into future actions")
    parser.add_argument("--latency_ema_alpha", type=float, default=0.2, help="EMA alpha for dynamic inference latency")
    parser.add_argument("--chunk_blend_horizon_steps", type=int, default=10, help="Overlap horizon for RTC-style chunk blending")
    parser.add_argument("--chunk_blend_schedule", choices=("exp", "linear", "none"), default="exp", help="How new chunks blend into existing buffered actions")
    parser.add_argument("--action_smoothing", choices=("off", "ema"), default="off", help="Optional output action smoothing")
    parser.add_argument("--action_ema_alpha", type=float, default=0.35, help="EMA alpha used when --action_smoothing ema")
    parser.add_argument("--async_log_interval_s", type=float, default=1.0, help="Async rollout status log interval")
    parser.add_argument("--async_debug_dir", default=None, help="Write async rollout debug JSONL files under this directory")
    parser.add_argument("--async_debug_readback_every_n_steps", type=int, default=0, help="Read robot pose every N control steps for tracking debug; 0 disables")
    parser.add_argument("--async_debug_flush_interval", type=int, default=1, help="Flush debug JSONL files every N records")
    parser.add_argument("--async_debug_include_images", action="store_true", help="Record image shapes/dtypes only; image bytes are not written")
    parser.add_argument("--max_position_step_m", type=float, default=0.0, help="Per-tick L2 position limit in meters; 0 disables")
    parser.add_argument("--max_rotation_step_deg", type=float, default=0.0, help="Per-tick RPY L2 rotation limit in degrees; 0 disables")
    parser.add_argument("--max_gripper_step", type=float, default=0.0, help="Per-tick gripper limit; 0 disables")
    parser.add_argument("--rtc_chunk_conditioning", action="store_true", help="Enable server-side RTC FM inpainting from previous chunks")
    parser.add_argument("--rtc_delay_steps", type=int, default=-1, help="RTC delay override; <0 uses the last measured delay estimate")
    parser.add_argument("--rtc_soft_horizon_steps", type=int, default=5, help="RTC soft guidance exponential decay horizon")
    parser.add_argument("--rtc_free_tail_steps", type=int, default=5, help="RTC free tail length with zero guidance")


def _set_xarm_control_mode(bestman: base.Bestman_Real_Xarm6, mode: int) -> None:
    robot = bestman.robot
    if hasattr(robot, "motion_enable"):
        try:
            robot.motion_enable(enable=True)
        except TypeError:
            robot.motion_enable(True)
    robot.set_mode(mode)
    robot.set_state(0)
    time.sleep(0.05)


def _set_xarm_servo_pose(bestman: base.Bestman_Real_Xarm6, pos_m, rpy_deg) -> None:
    x_m, y_m, z_m = [float(v) for v in pos_m]
    roll, pitch, yaw = [float(v) for v in rpy_deg]
    pose = [x_m * 1000.0, y_m * 1000.0, z_m * 1000.0, roll, pitch, yaw]
    try:
        bestman.robot.set_servo_cartesian(pose, is_radian=False)
    except TypeError:
        bestman.robot.set_servo_cartesian(pose)


def _set_xarm_joint_servo(bestman: base.Bestman_Real_Xarm6, q_rad: np.ndarray) -> None:
    q_rad = active_joint_vector(q_rad, axis=bestman.robot.axis, name="xArm joint servo target")
    code = bestman.robot.set_servo_angle_j(q_rad.tolist(), is_radian=True)
    if code != 0:
        raise RuntimeError(f"xArm set_servo_angle_j failed: code={code}")


def _read_xarm_joint_state(bestman: base.Bestman_Real_Xarm6) -> tuple[np.ndarray, np.ndarray]:
    code, state = bestman.robot.get_joint_states(is_radian=True)
    if code != 0 or len(state) < 2:
        raise RuntimeError(f"xArm get_joint_states failed: code={code}")
    q_rad = active_joint_vector(state[0], axis=bestman.robot.axis, name="xArm q")
    dq_rad_s = active_joint_vector(state[1], axis=bestman.robot.axis, name="xArm dq")
    return q_rad, dq_rad_s


def _solve_xarm_joint_waypoints(
    bestman: base.Bestman_Real_Xarm6,
    actions: np.ndarray,
    *,
    action_offset: int,
    q_seed_rad: np.ndarray,
    max_joint_step_rad: float,
    robot_lock: threading.Lock | None = None,
) -> np.ndarray:
    q_seed_rad = np.asarray(q_seed_rad, dtype=np.float64)
    waypoints = []
    for waypoint_index, action in enumerate(np.asarray(actions, dtype=np.float64)):
        x_m, y_m, z_m, roll, pitch, yaw = action[action_offset : action_offset + 6]
        pose = [x_m * 1000.0, y_m * 1000.0, z_m * 1000.0, roll, pitch, yaw]
        optional_ik_kwargs = {
            keyword: value
            for keyword, value in (("limited", True), ("ref_angles", q_seed_rad.tolist()))
            if keyword not in _unsupported_xarm_ik_kwargs
        }
        with robot_lock if robot_lock is not None else contextlib.nullcontext():
            (code, q_rad), dropped_kwargs = call_with_supported_optional_kwargs(
                bestman.robot.get_inverse_kinematics,
                pose,
                input_is_radian=False,
                return_is_radian=True,
                optional_kwargs=tuple(optional_ik_kwargs),
                **optional_ik_kwargs,
            )
        for keyword in dropped_kwargs:
            _unsupported_xarm_ik_kwargs.add(keyword)
            if keyword not in _warned_xarm_ik_dropped_kwargs:
                print(
                    "[ASYNC][plan-servo][WARN] "
                    f"xArm IK SDK does not support optional kwarg {keyword!r}; retrying without it"
                )
                _warned_xarm_ik_dropped_kwargs.add(keyword)
        if code != 0:
            raise RuntimeError(f"xArm IK failed at waypoint {waypoint_index}: code={code}, pose={pose}")
        q_rad = active_joint_vector(q_rad, axis=len(q_seed_rad), name="xArm IK solution")
        joint_step_rad = max_joint_waypoint_delta(q_seed_rad, q_rad[None, :])
        if max_joint_step_rad > 0.0 and joint_step_rad > max_joint_step_rad:
            raise RuntimeError(
                "xArm IK branch jump at waypoint "
                f"{waypoint_index}: joint_step={joint_step_rad:.4f}rad exceeds "
                f"--plan_servo_max_ik_joint_step_rad={max_joint_step_rad:.4f}"
            )
        q_seed_rad = q_rad
        waypoints.append(q_seed_rad.copy())
    return np.stack(waypoints)


def _solve_local_joint_waypoints(
    solver: PinocchioUrdfIK,
    actions: np.ndarray,
    *,
    action_offset: int,
    q_seed_rad: np.ndarray,
    max_joint_step_rad: float,
) -> np.ndarray:
    q_seed_rad = np.asarray(q_seed_rad, dtype=np.float64)
    waypoints = []
    for waypoint_index, action in enumerate(np.asarray(actions, dtype=np.float64)):
        pose = action[action_offset : action_offset + 6]
        q_rad = solver.solve(pose, q_seed_rad=q_seed_rad)
        joint_step_rad = max_joint_waypoint_delta(q_seed_rad, q_rad[None, :])
        if max_joint_step_rad > 0.0 and joint_step_rad > max_joint_step_rad:
            raise RuntimeError(
                "Pinocchio IK branch jump at waypoint "
                f"{waypoint_index}: joint_step={joint_step_rad:.4f}rad exceeds "
                f"--plan_servo_max_ik_joint_step_rad={max_joint_step_rad:.4f}"
            )
        q_seed_rad = q_rad
        waypoints.append(q_seed_rad.copy())
    return np.stack(waypoints)


def _read_xarm_tcp_configuration(robot, *, override: str) -> tuple[np.ndarray, object]:
    if override != "auto":
        return parse_tcp_offset_mm_rpy_deg(override), getattr(robot, "tcp_load", None)
    try:
        raw_offset = getattr(robot, "tcp_offset")
        angles_are_radian = bool(robot.default_is_radian)
    except Exception as exc:
        raise RuntimeError(
            "Unable to read xArm tcp_offset automatically. Pass "
            '--plan_servo_tcp_offset "x_mm,y_mm,z_mm,roll_deg,pitch_deg,yaw_deg".'
        ) from exc
    return normalize_xarm_tcp_offset(raw_offset, angles_are_radian=angles_are_radian), getattr(
        robot,
        "tcp_load",
        None,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--description", required=True, help="任务自然语言指令")
    parser.add_argument("--arm_mode", choices=("dual", "single"), default="dual", help="dual=双臂 14D；single=单臂 7D")
    parser.add_argument("--single_arm", choices=("robot_0", "robot_1"), default=None, help="single 模式必须显式指定使用哪只 xArm")
    parser.add_argument("--single_image_key", default="front", help="single 模式发给 policy 的 image key，FastUMI 单臂默认 front")
    parser.add_argument("--robot_ip", default="192.168.1.240", help="robot_0 对应 xArm 控制盒 IP")
    parser.add_argument("--robot_ip_2", default="192.168.1.217", help="robot_1 对应 xArm 控制盒 IP")
    parser.add_argument("--server_ip", default=base.SERVER_IP, help="policy server host")
    parser.add_argument("--port", type=int, default=base.PORT, help="policy server port")
    parser.add_argument("--camera_dev0", type=int, default=base.DEV, help="robot_0 图像对应的视频设备编号")
    parser.add_argument("--camera_dev1", type=int, default=base.DEV + 2, help="robot_1 图像对应的视频设备编号")
    parser.add_argument("--init_pose_left", default="0.250,0.200,0.145,180,-90,0.0", help="robot_0 初始位姿: x,y,z,roll,pitch,yaw（米、度）")
    parser.add_argument("--init_pose_right", default="0.4,0.0,0.146,180,-90,0.0", help="robot_1 初始位姿: x,y,z,roll,pitch,yaw（米、度）")
    parser.add_argument("--skip_init_move", action="store_true", help="启动时不自动移动到 init_pose；s 复位仍使用 init_pose")
    parser.add_argument("--action_start", type=int, default=0, help="本次推理返回的 action 序列起始下标（含）")
    parser.add_argument("--action_end", type=int, default=50, help="本次推理返回的 action 序列结束下标（含）")
    parser.add_argument(
        "--xarm_control_mode",
        choices=("position", "servo", "online_cartesian", "plan-servo"),
        default="position",
        help="position=mode 0；servo=mode 1 笛卡尔伺服；online_cartesian=mode 7；plan-servo=SDK IK + 外部 joint planner + mode 1 joint servo",
    )
    parser.add_argument("--plan_servo_hz", type=float, default=100.0, help="plan-servo 下发 joint sample 的频率，默认 100Hz")
    parser.add_argument("--plan_servo_model_action_dt", type=float, default=-1.0, help="每个模型 action waypoint 的时间间隔；默认 -1 自动使用 1 / control_hz")
    parser.add_argument("--plan_servo_chunk_max_waypoints", type=int, default=50, help="plan-servo 每次重规划最多使用的连续未来模型 action 数")
    parser.add_argument("--plan_servo_ik_backend", choices=("sdk", "pinocchio"), default="sdk", help="plan-servo IK backend；pinocchio 使用本地 URDF seeded IK")
    parser.add_argument("--plan_servo_urdf", default=str(DEFAULT_XARM6_URDF), help="pinocchio IK 使用的 URDF 路径")
    parser.add_argument("--plan_servo_urdf_tip_link", default="link6", help="pinocchio IK 目标法兰 link；默认 link6")
    parser.add_argument("--plan_servo_tcp_offset", default="auto", help="pinocchio IK 法兰到 TCP offset；auto 从 xArm 读取，否则传 x_mm,y_mm,z_mm,roll_deg,pitch_deg,yaw_deg")
    parser.add_argument("--plan_servo_pinocchio_max_iterations", type=int, default=200, help="pinocchio IK 每个 waypoint 最大迭代数")
    parser.add_argument("--plan_servo_pinocchio_tolerance", type=float, default=9e-3, help="pinocchio IK SE(3) residual 收敛阈值")
    parser.add_argument("--plan_servo_pinocchio_damping", type=float, default=1e-6, help="pinocchio IK damped least-squares 阻尼")
    parser.add_argument("--plan_servo_pinocchio_step_size", type=float, default=0.1, help="pinocchio IK 每次迭代步长")
    parser.add_argument("--plan_servo_max_ik_joint_step_rad", type=float, default=0.5, help="相邻 IK waypoint 最大 joint 跳变；默认 0.5rad，0 表示关闭")
    parser.add_argument("--plan_servo_max_joint_velocity_rad_s", type=float, default=1.0, help="计划 joint 最大速度校验阈值；默认 1.0rad/s，0 表示关闭校验")
    parser.add_argument("--plan_servo_max_joint_acceleration_rad_s2", type=float, default=40.0, help="计划 joint 最大加速度校验阈值；默认 40rad/s^2，0 表示关闭校验")
    parser.add_argument("--plan_servo_max_tracking_error_rad", type=float, default=0.15, help="上一条 command 与真实 joint 的最大允许误差；默认 0.15rad，0 表示关闭校验")
    parser.add_argument("--plan_servo_debug_readback_every_n_samples", type=int, default=0, help="debug 模式每 N 个 joint-servo sample 读取一次真实 joint；默认 0 关闭")
    parser.add_argument("--plan_servo_pinocchio_step_size", type=float, default=0.2, help="pinocchio IK 每次迭代步长")
    parser.add_argument("--plan_servo_max_ik_joint_step_rad", type=float, default=0.4, help="相邻 IK waypoint 最大 joint 跳变；默认 0.5rad，0 表示关闭")
    parser.add_argument("--plan_servo_max_joint_velocity_rad_s", type=float, default=5.0, help="计划 joint 最大速度校验阈值；默认 1.0rad/s，0 表示关闭校验")
    parser.add_argument("--plan_servo_max_joint_acceleration_rad_s2", type=float, default=80.0, help="计划 joint 最大加速度校验阈值；默认 40rad/s^2，0 表示关闭校验")
    parser.add_argument("--plan_servo_stale_timeout_s", type=float, default=0.5, help="计划结束后最多保持末点的时长；超时后停止发送陈旧 joint target")
    parser.add_argument("--mask_overlay", action="store_true", help="请求 policy server 使用 SAM3 mask overlay 后再推理")
    parser.add_argument("--mask_view", default=None, help="需要做 SAM3 overlay 的 image key；single 默认 front，dual 下建议显式指定 robot_0/robot_1")
    parser.add_argument("--mask_prompt_point", default=None, help="初始正点 prompt，格式 x,y；不填则弹窗点击")
    parser.add_argument("--mask_prompt_text", default=None, help="SAM3 text prompt，例如 sponge；text_select_video 模式必填")
    parser.add_argument("--mask_alpha", type=float, default=0.35, help="SAM3 overlay alpha")
    parser.add_argument("--mask_debug_dir", default=None, help="保存 SAM3 preview 输入/输出调试图")
    parser.add_argument("--mask_track_between_actions", action="store_true", help="async 模式不在控制线程内做 tracking-only，保留参数兼容旧命令")
    parser.add_argument("--mask_track_every_n_actions", type=int, default=1, help="旧脚本 tracking-only 参数，async 模式不使用")
    _add_async_args(parser)
    args = parser.parse_args()

    is_dual = args.arm_mode == "dual"
    if not is_dual and args.single_arm is None:
        parser.error("--arm_mode single 需要显式指定 --single_arm robot_0 或 robot_1")
    if args.mask_overlay and is_dual and args.mask_view is None:
        parser.error("--arm_mode dual 开启 --mask_overlay 时需要显式指定 --mask_view robot_0 或 robot_1")
    if args.mask_track_between_actions:
        print("[WARN] async rollout 不在 control thread 内执行 mask tracking-only；会继续在 policy inference 时请求 overlay")
    if args.control_hz <= 0:
        parser.error("--control_hz must be > 0")
    if args.inference_interval_steps < 1:
        parser.error("--inference_interval_steps must be >= 1")
    if args.inference_delay_mode != "fixed" and args.inference_delay_steps != 0:
        print(
            "[WARN] --inference_delay_steps is ignored unless "
            "--inference_delay_mode fixed; use --max_inference_delay_steps to cap instant/ema modes"
        )
    if (
        args.inference_delay_mode == "fixed"
        and args.max_inference_delay_steps >= 0
        and args.inference_delay_steps > args.max_inference_delay_steps
    ):
        print(
            f"[WARN] fixed delay {args.inference_delay_steps} will be capped to "
            f"--max_inference_delay_steps {args.max_inference_delay_steps}"
        )
    if args.rtc_soft_horizon_steps < 1:
        parser.error("--rtc_soft_horizon_steps must be >= 1")
    if args.rtc_free_tail_steps < 0:
        parser.error("--rtc_free_tail_steps must be >= 0")
    if args.plan_servo_hz <= 0.0:
        parser.error("--plan_servo_hz must be > 0")
    if args.plan_servo_model_action_dt == 0.0:
        parser.error("--plan_servo_model_action_dt must be > 0, or < 0 to use 1 / control_hz")
    if args.plan_servo_chunk_max_waypoints < 1:
        parser.error("--plan_servo_chunk_max_waypoints must be >= 1")
    if args.plan_servo_pinocchio_max_iterations < 1:
        parser.error("--plan_servo_pinocchio_max_iterations must be >= 1")
    if args.plan_servo_pinocchio_tolerance <= 0.0:
        parser.error("--plan_servo_pinocchio_tolerance must be > 0")
    if args.plan_servo_pinocchio_damping <= 0.0:
        parser.error("--plan_servo_pinocchio_damping must be > 0")
    if args.plan_servo_pinocchio_step_size <= 0.0:
        parser.error("--plan_servo_pinocchio_step_size must be > 0")
    if args.plan_servo_max_ik_joint_step_rad < 0.0:
        parser.error("--plan_servo_max_ik_joint_step_rad must be >= 0")
    if args.plan_servo_max_joint_velocity_rad_s < 0.0:
        parser.error("--plan_servo_max_joint_velocity_rad_s must be >= 0")
    if args.plan_servo_max_joint_acceleration_rad_s2 < 0.0:
        parser.error("--plan_servo_max_joint_acceleration_rad_s2 must be >= 0")
    if args.plan_servo_max_tracking_error_rad < 0.0:
        parser.error("--plan_servo_max_tracking_error_rad must be >= 0")
    if args.plan_servo_debug_readback_every_n_samples < 0:
        parser.error("--plan_servo_debug_readback_every_n_samples must be >= 0")
    if args.plan_servo_stale_timeout_s < 0.0:
        parser.error("--plan_servo_stale_timeout_s must be >= 0")
    if args.plan_servo_model_action_dt > 0.0 and not np.isclose(
        args.plan_servo_model_action_dt,
        1.0 / args.control_hz,
        rtol=1e-4,
        atol=1e-6,
    ):
        print(
            "[WARN] --plan_servo_model_action_dt differs from 1 / control_hz. "
            "Current plan-servo keeps ActionBuffer step timing and gripper updates on control_hz, "
            "so a different dt only changes the joint planner's internal waypoint spacing."
        )

    single_arm = args.single_arm
    single_arm_index = base._arm_index(single_arm) if single_arm is not None else 0
    effective_mask_view = args.mask_view or args.single_image_key
    init_pos0, init_rpy0 = base.parse_pose_xyz_rpy_deg(args.init_pose_left)
    init_pos1, init_rpy1 = base.parse_pose_xyz_rpy_deg(args.init_pose_right)
    init_by_arm = {"robot_0": (init_pos0, init_rpy0), "robot_1": (init_pos1, init_rpy1)}

    arms: dict[str, base.Bestman_Real_Xarm6] = {}
    if is_dual or single_arm == "robot_0":
        arms["robot_0"] = base.Bestman_Real_Xarm6(args.robot_ip, None, None)
    if is_dual or single_arm == "robot_1":
        arms["robot_1"] = base.Bestman_Real_Xarm6(args.robot_ip_2, None, None)
    if args.xarm_control_mode == "plan-servo":
        for arm_name, arm in arms.items():
            required_methods = ["get_joint_states", "set_servo_angle_j"]
            if args.plan_servo_ik_backend == "sdk":
                required_methods.append("get_inverse_kinematics")
            missing = [
                name
                for name in required_methods
                if not hasattr(arm.robot, name)
            ]
            if missing:
                raise RuntimeError(f"{arm_name} xArm SDK 缺少 {missing}，不能使用 --xarm_control_mode plan-servo")
    local_ik_solvers = {}
    local_ik_tcp_offsets = {}
    if args.xarm_control_mode == "plan-servo" and args.plan_servo_ik_backend == "pinocchio":
        for arm_name, arm in arms.items():
            tcp_offset, tcp_load = _read_xarm_tcp_configuration(arm.robot, override=args.plan_servo_tcp_offset)
            print(
                f"[ASYNC][plan-servo] {arm_name} pinocchio TCP offset "
                f"[m,deg]={tcp_offset.tolist()} payload={tcp_load}"
            )
            local_ik_tcp_offsets[arm_name] = tcp_offset.tolist()
            local_ik_solvers[arm_name] = PinocchioUrdfIK(
                args.plan_servo_urdf,
                tip_link=args.plan_servo_urdf_tip_link,
                max_iterations=args.plan_servo_pinocchio_max_iterations,
                tolerance=args.plan_servo_pinocchio_tolerance,
                damping=args.plan_servo_pinocchio_damping,
                step_size=args.plan_servo_pinocchio_step_size,
                tcp_offset_xyz_m_rpy_deg=tcp_offset,
            )

    robot_lock = threading.Lock()
    control_step = {"value": 0}
    generation = {"value": 0}
    step_lock = threading.Lock()
    stop_event = threading.Event()
    paused_event = threading.Event()
    runtime_control_mode_active = {"value": False}
    plan_servo_lock = threading.Lock()
    plan_servo_state: dict[str, object] = {
        "installed_at": None,
        "trajectories": {},
        "last_commands": {},
        "version": 0,
    }
    debug_writer = AsyncDebugWriter(args.async_debug_dir, flush_interval=args.async_debug_flush_interval)
    debug_counts = {"obs_id": 0, "chunk_id": 0}
    last_executed_action = {"value": None}
    last_completed_chunk_id = {"value": None}
    last_delay_steps = {"value": None}
    last_planner_delay_steps = {"value": 0}
    rtc_session_id = uuid.uuid4().hex
    gripper_state: dict[str, dict[str, float | None]] = {
        "robot_0": {"open": None, "time": 0.0},
        "robot_1": {"open": None, "time": 0.0},
    }

    def get_step() -> int:
        with step_lock:
            return int(control_step["value"])

    def set_step(step: int) -> None:
        with step_lock:
            control_step["value"] = int(step)

    def get_generation() -> int:
        with step_lock:
            return int(generation["value"])

    def bump_generation() -> None:
        with step_lock:
            generation["value"] += 1

    def advance_step() -> None:
        with step_lock:
            control_step["value"] += 1

    def next_debug_id(name: str) -> int:
        with step_lock:
            value = int(debug_counts[name])
            debug_counts[name] = value + 1
            return value

    def requested_runtime_xarm_mode() -> int | None:
        return {"servo": 1, "online_cartesian": 7, "plan-servo": 1}.get(args.xarm_control_mode)

    def clear_plan_servo_trajectory() -> None:
        with plan_servo_lock:
            plan_servo_state["installed_at"] = None
            plan_servo_state["trajectories"] = {}
            plan_servo_state["last_commands"] = {}
            plan_servo_state["version"] = int(plan_servo_state["version"]) + 1

    def enter_runtime_control_mode() -> None:
        mode = requested_runtime_xarm_mode()
        if mode is None or runtime_control_mode_active["value"]:
            return
        with robot_lock:
            for arm in arms.values():
                _set_xarm_control_mode(arm, mode)
        runtime_control_mode_active["value"] = True
        print(f"[ASYNC][xArm] {args.xarm_control_mode} mode enabled: mode={mode}")

    def exit_runtime_control_mode() -> None:
        if requested_runtime_xarm_mode() is None or not runtime_control_mode_active["value"]:
            return
        try:
            with robot_lock:
                for arm in arms.values():
                    _set_xarm_control_mode(arm, 0)
            print("[ASYNC][xArm] restored position mode")
        except Exception as exc:
            print(f"[ASYNC][xArm][WARN] failed to restore position mode: {exc}")
        finally:
            runtime_control_mode_active["value"] = False

    def reset_active_arms() -> None:
        clear_plan_servo_trajectory()
        exit_runtime_control_mode()
        with robot_lock:
            if is_dual:
                base.reset_arms_to_init(arms["robot_0"], arms["robot_1"], init_pos0, init_rpy0, init_pos1, init_rpy1)
            else:
                assert single_arm is not None
                init_pos, init_rpy = init_by_arm[single_arm]
                base.reset_arm_to_init(arms[single_arm], init_pos, init_rpy, arm_index=single_arm_index)

    if not args.skip_init_move:
        print(
            f"[INFO] 移动到初始位姿: "
            f"{'robot_0=' + args.init_pose_left + ' robot_1=' + args.init_pose_right if is_dual else single_arm + '=' + base._select_by_arm(single_arm, args.init_pose_left, args.init_pose_right)}"
        )
        reset_active_arms()
        print("[INFO] 已到达初始位姿")

    policy_client = base.websocket_client_policy.WebsocketClientPolicy(host=args.server_ip, port=args.port)
    print(f"[INFO] 已连接策略服务器：ws://{args.server_ip}:{args.port}")
    metadata = policy_client.get_server_metadata() if (args.mask_overlay or args.rtc_chunk_conditioning) else {}
    if args.mask_overlay:
        if not metadata.get("mask_overlay", {}).get("enabled", False):
            raise RuntimeError("client 开启了 --mask_overlay，但 server 未启用 --mask-overlay")
        if metadata.get("mask_overlay", {}).get("tracking_mode") == "text_select_video" and not args.mask_prompt_text:
            raise RuntimeError("server 使用 text_select_video，client 需要提供 --mask_prompt_text")
    if args.rtc_chunk_conditioning and not metadata.get("rtc", {}).get("enabled", False):
        raise RuntimeError("client 开启了 --rtc_chunk_conditioning，但 server 未启用 --rtc-chunk-conditioning")

    camera_devs = {"robot_0": args.camera_dev0, "robot_1": args.camera_dev1}
    caps = {}
    if is_dual:
        caps["robot_0"] = base.init_yu12_camera(camera_devs["robot_0"])
        caps["robot_1"] = base.init_yu12_camera(camera_devs["robot_1"])
    else:
        assert single_arm is not None
        caps[single_arm] = base.init_yu12_camera(camera_devs[single_arm])
    for _ in range(50):
        for cap in caps.values():
            _ = cap.read()
    print("[INFO] 摄像头预热完成")

    input("按 Enter 开始")
    mask_debug_dir = pathlib.Path(args.mask_debug_dir) if args.mask_debug_dir else None
    if args.mask_overlay:
        if is_dual:
            preview_images = {
                "robot_0": base._capture_resized(caps["robot_0"]),
                "robot_1": base._capture_resized(caps["robot_1"]),
            }
        else:
            assert single_arm is not None
            preview_images = {args.single_image_key: base._capture_resized(caps[single_arm])}
        base._preview_mask_overlay(
            policy_client,
            image_rgb=preview_images[effective_mask_view],
            view=effective_mask_view,
            point_xy=base._parse_xy(args.mask_prompt_point),
            text_prompt=args.mask_prompt_text,
            alpha=args.mask_alpha,
            debug_dir=mask_debug_dir,
        )
        print("[INFO] SAM3 mask overlay 已确认")

    action_buffer = ActionBuffer(
        min_buffer_steps=args.min_buffer_steps,
        blend_horizon_steps=args.chunk_blend_horizon_steps,
        blend_schedule=args.chunk_blend_schedule,
        empty_action_policy=args.empty_action_policy,
        action_smoothing=args.action_smoothing,
        action_ema_alpha=args.action_ema_alpha,
        cyclic_indices=XARM_DUAL_RPY_ACTION_INDICES if is_dual else XARM_SINGLE_RPY_ACTION_INDICES,
        cyclic_period=360.0,
    )
    latency_estimator = LatencyEstimator(
        mode=args.inference_delay_mode,
        fixed_steps=args.inference_delay_steps,
        control_hz=args.control_hz,
        ema_alpha=args.latency_ema_alpha,
    )

    def _image_metadata(image_obs: dict) -> dict:
        if not args.async_debug_include_images:
            return {
                key: {"shape": list(np.asarray(value).shape), "dtype": str(np.asarray(value).dtype)}
                for key, value in image_obs.items()
            }
        return {
            key: {
                "shape": list(np.asarray(value).shape),
                "dtype": str(np.asarray(value).dtype),
                "min": int(np.min(value)),
                "max": int(np.max(value)),
            }
            for key, value in image_obs.items()
        }

    def read_debug_robot_pose() -> np.ndarray | None:
        if args.async_debug_readback_every_n_steps <= 0:
            return None
        try:
            if is_dual:
                with robot_lock:
                    pos0, rpy0, _quat0 = base._read_xarm_pose(arms["robot_0"])
                    pos1, rpy1, _quat1 = base._read_xarm_pose(arms["robot_1"])
                    g0 = base._read_xarm_gripper_open(arms["robot_0"])
                    g1 = base._read_xarm_gripper_open(arms["robot_1"])
                return np.asarray([*pos0, *rpy0, g0, *pos1, *rpy1, g1], dtype=np.float64)
            assert single_arm is not None
            with robot_lock:
                pos, rpy, _quat = base._read_xarm_pose(arms[single_arm])
                gripper = base._read_xarm_gripper_open(arms[single_arm])
            return np.asarray([*pos, *rpy, gripper], dtype=np.float64)
        except Exception as exc:
            print(f"[ASYNC][debug][WARN] failed to read xArm pose: {exc}")
            return None

    def read_observation():
        if is_dual:
            with robot_lock:
                pos0, rpy0, quat0 = base._read_xarm_pose(arms["robot_0"])
                pos1, rpy1, quat1 = base._read_xarm_pose(arms["robot_1"])
                gripper0 = base._read_xarm_gripper_open(arms["robot_0"])
                gripper1 = base._read_xarm_gripper_open(arms["robot_1"])
            state_vec = np.array([*pos0, *quat0, gripper0, *pos1, *quat1, gripper1], dtype=np.float32)
            image_obs = {
                "robot_0": base.image_tools.convert_to_uint8(base._capture_resized(caps["robot_0"])),
                "robot_1": base.image_tools.convert_to_uint8(base._capture_resized(caps["robot_1"])),
            }
        else:
            assert single_arm is not None
            with robot_lock:
                pos, _rpy, quat = base._read_xarm_pose(arms[single_arm])
                gripper = base._read_xarm_gripper_open(arms[single_arm])
            state_vec = np.array([*pos, *quat, gripper], dtype=np.float32)
            image_obs = {
                args.single_image_key: base.image_tools.convert_to_uint8(base._capture_resized(caps[single_arm])),
            }
        obs = {"state": state_vec, "image": image_obs, "prompt": args.description}
        if args.mask_overlay:
            obs[base.MASK_OVERLAY_KEY] = {"enabled": True, "view": effective_mask_view, "alpha": args.mask_alpha}
            if args.mask_prompt_text:
                obs[base.MASK_OVERLAY_KEY]["text"] = args.mask_prompt_text
            if mask_debug_dir is not None:
                obs[base.MASK_OVERLAY_KEY]["return_image"] = True
                obs[base.MASK_OVERLAY_KEY]["return_overlay"] = True
        return obs, image_obs

    def rtc_request_delay_steps() -> int:
        if args.rtc_delay_steps >= 0:
            delay_steps = int(args.rtc_delay_steps)
        elif last_delay_steps["value"] is not None:
            delay_steps = int(last_delay_steps["value"])
        elif args.inference_delay_mode == "fixed":
            delay_steps = int(args.inference_delay_steps)
        else:
            delay_steps = 0
        if args.max_inference_delay_steps >= 0:
            delay_steps = min(delay_steps, args.max_inference_delay_steps)
        return max(delay_steps, 0)

    def update_total_delay_steps(policy_delay_steps: int, planner_delay_steps: int = 0) -> int:
        total_delay_steps = max(int(policy_delay_steps), 0) + max(int(planner_delay_steps), 0)
        if args.max_inference_delay_steps >= 0:
            total_delay_steps = min(total_delay_steps, args.max_inference_delay_steps)
        last_planner_delay_steps["value"] = max(int(planner_delay_steps), 0)
        last_delay_steps["value"] = total_delay_steps
        return total_delay_steps

    def maybe_set_gripper(arm_name: str, g_open: float) -> None:
        state = gripper_state[arm_name]
        now = time.monotonic()
        last_open = state["open"]
        last_time = float(state["time"] or 0.0)
        if (
            last_open is None
            or abs(float(g_open) - float(last_open)) >= GRIPPER_UPDATE_THRESHOLD
            or now - last_time >= GRIPPER_UPDATE_INTERVAL_S
        ):
            base._set_xarm_gripper(arms[arm_name], float(g_open), arm_index=base._arm_index(arm_name), wait=False)
            state["open"] = float(g_open)
            state["time"] = now

    def set_pose(arm_name: str, pos_m, rpy_deg) -> None:
        if args.xarm_control_mode == "servo":
            _set_xarm_servo_pose(arms[arm_name], pos_m, rpy_deg)
        else:
            base._set_xarm_pose(arms[arm_name], pos_m, rpy_deg, wait=False)

    def execute_action(action: np.ndarray) -> None:
        if args.xarm_control_mode == "plan-servo":
            with robot_lock:
                if is_dual:
                    maybe_set_gripper("robot_0", float(np.clip(action[6], 0.0, 1.0)))
                    maybe_set_gripper("robot_1", float(np.clip(action[13], 0.0, 1.0)))
                else:
                    assert single_arm is not None
                    maybe_set_gripper(single_arm, float(np.clip(action[6], 0.0, 1.0)))
            return
        if is_dual:
            x0, y0, z0, r0, p0, yy0, g0, x1, y1, z1, r1, p1, yy1, g1 = action[:14]
            with robot_lock:
                set_pose("robot_0", [x0, y0, z0], [r0, p0, yy0])
                set_pose("robot_1", [x1, y1, z1], [r1, p1, yy1])
                maybe_set_gripper("robot_0", float(np.clip(g0, 0.0, 1.0)))
                maybe_set_gripper("robot_1", float(np.clip(g1, 0.0, 1.0)))
        else:
            assert single_arm is not None
            x, y, z, roll, pitch, yaw, g_open = action[:7]
            with robot_lock:
                set_pose(single_arm, [x, y, z], [roll, pitch, yaw])
                maybe_set_gripper(single_arm, float(np.clip(g_open, 0.0, 1.0)))

    def plan_servo_model_action_dt() -> float:
        if args.plan_servo_model_action_dt > 0.0:
            return float(args.plan_servo_model_action_dt)
        return 1.0 / args.control_hz

    def update_plan_servo_trajectory(current_step: int, *, expected_generation: int) -> dict[str, object]:
        """Replace the future joint-servo trajectory from the latest merged Cartesian suffix."""
        planning_started_at = time.monotonic()
        planning_start_step = get_step()
        start_step, actions = action_buffer.contiguous_actions_from(
            current_step,
            max_steps=args.plan_servo_chunk_max_waypoints,
        )
        if start_step is None or len(actions) == 0:
            return {"updated": False, "reason": "no_contiguous_actions"}
        handoff_anchor_step = start_step
        actions, start_step, live_handoff_input_skipped = prepare_live_handoff_actions(
            actions,
            start_step=start_step,
            planning_start_step=planning_start_step,
        )
        if len(actions) == 0:
            return {"updated": False, "reason": "no_future_actions_after_live_handoff"}
        with robot_lock:
            initial_states = {arm_name: _read_xarm_joint_state(arm) for arm_name, arm in arms.items()}
        joint_waypoints = {}
        for arm_name, arm in arms.items():
            action_offset = 0 if arm_name == "robot_0" or not is_dual else 7
            if args.plan_servo_ik_backend == "pinocchio":
                joint_waypoints[arm_name] = _solve_local_joint_waypoints(
                    local_ik_solvers[arm_name],
                    actions,
                    action_offset=action_offset,
                    q_seed_rad=initial_states[arm_name][0],
                    max_joint_step_rad=args.plan_servo_max_ik_joint_step_rad,
                )
            else:
                joint_waypoints[arm_name] = _solve_xarm_joint_waypoints(
                    arm,
                    actions,
                    action_offset=action_offset,
                    q_seed_rad=initial_states[arm_name][0],
                    max_joint_step_rad=args.plan_servo_max_ik_joint_step_rad,
                    robot_lock=robot_lock,
                )
        if expected_generation != get_generation():
            return {"updated": False, "reason": "stale_generation_after_ik"}
        install_step = get_step()
        aligned = {
            arm_name: align_joint_waypoints_to_install_step(
                waypoints,
                first_target_step=start_step,
                install_step=install_step,
                control_hz=args.control_hz,
            )
            for arm_name, waypoints in joint_waypoints.items()
        }
        joint_waypoints = {arm_name: result[0] for arm_name, result in aligned.items()}
        start_step = next(iter(aligned.values()))[1]
        expired_waypoints = next(iter(aligned.values()))[2]
        start_delay_s = next(iter(aligned.values()))[3]
        if any(len(waypoints) == 0 for waypoints in joint_waypoints.values()):
            planner_latency_s = time.monotonic() - planning_started_at
            return {
                "updated": False,
                "reason": "all_waypoints_expired_during_ik",
                "planner_latency_s": planner_latency_s,
                "planner_latency_steps": max(int(np.ceil(planner_latency_s * args.control_hz)), 0),
            }
        ik_waypoints = len(next(iter(joint_waypoints.values())))
        waypoint_dt_s = plan_servo_model_action_dt()
        if start_delay_s > 0.0:
            print(
                "[ASYNC][plan-servo][WARN] future-gap fallback active: "
                f"install_step={install_step} start_step={start_step} hold={start_delay_s:.3f}s"
            )
        with plan_servo_lock:
            # Freeze sender progression while reading the physical state and
            # atomically replacing the command stream.
            with robot_lock:
                start_states = {arm_name: _read_xarm_joint_state(arm) for arm_name, arm in arms.items()}
            last_commands = dict(plan_servo_state["last_commands"])
            try:
                handoff_states = {}
                for arm_name in arms:
                    q_command, dq_command = last_commands.get(arm_name, (None, None))
                    handoff_states[arm_name] = command_stream_handoff_state(
                        start_states[arm_name][0],
                        start_states[arm_name][1],
                        q_command,
                        dq_command,
                        max_tracking_error_rad=args.plan_servo_max_tracking_error_rad,
                    )
            except ValueError:
                # Do not keep advancing an old command stream after the physical
                # arm has fallen behind it.
                plan_servo_state["installed_at"] = None
                plan_servo_state["trajectories"] = {}
                plan_servo_state["last_commands"] = {}
                raise
            planned_waypoints = len(next(iter(joint_waypoints.values())))
            trajectories = {
                arm_name: plan_joint_cubic_trajectory(
                    handoff_states[arm_name][0],
                    handoff_states[arm_name][1],
                    joint_waypoints[arm_name],
                    waypoint_dt_s=waypoint_dt_s,
                    sample_hz=args.plan_servo_hz,
                    start_delay_s=start_delay_s,
                    max_velocity_rad_s=args.plan_servo_max_joint_velocity_rad_s,
                    max_acceleration_rad_s2=args.plan_servo_max_joint_acceleration_rad_s2,
                )
                for arm_name in arms
            }
            installed_at = time.monotonic()
            plan_servo_state["installed_at"] = installed_at
            plan_servo_state["trajectories"] = trajectories
            plan_servo_state["version"] = int(plan_servo_state["version"]) + 1
            version = int(plan_servo_state["version"])
        planner_latency_s = installed_at - planning_started_at
        planner_latency_steps = max(int(np.ceil(planner_latency_s * args.control_hz)), 0)
        return {
            "updated": True,
            "version": version,
            "start_step": start_step,
            "handoff_anchor_step": handoff_anchor_step,
            "waypoints": planned_waypoints,
            "ik_waypoints": ik_waypoints,
            "live_handoff_anchor": start_delay_s <= 0.0,
            "live_handoff_input_skipped": live_handoff_input_skipped,
            "handoff_sources": {arm_name: state[3] for arm_name, state in handoff_states.items()},
            "max_tracking_error_rad": max(state[2] for state in handoff_states.values()),
            "expired_waypoints_during_ik": expired_waypoints,
            "planning_start_step": planning_start_step,
            "install_step": install_step,
            "planner_latency_s": planner_latency_s,
            "planner_latency_steps": planner_latency_steps,
            "model_action_dt_s": waypoint_dt_s,
            "start_delay_s": start_delay_s,
            "sample_hz": args.plan_servo_hz,
            "duration_s": max(trajectory.duration_s for trajectory in trajectories.values()),
            "max_velocity_rad_s": max(trajectory.max_velocity_rad_s for trajectory in trajectories.values()),
            "max_acceleration_rad_s2": max(
                trajectory.max_acceleration_rad_s2 for trajectory in trajectories.values()
            ),
        }

    def plan_servo_sender_loop() -> None:
        period_s = 1.0 / args.plan_servo_hz
        next_tick = time.perf_counter()
        last_warning_time = 0.0
        sample_index = 0
        while not stop_event.is_set():
            if paused_event.is_set():
                next_tick = time.perf_counter() + period_s
                time.sleep(min(period_s, 0.02))
                continue
            try:
                with plan_servo_lock:
                    installed_at = plan_servo_state["installed_at"]
                    trajectories = dict(plan_servo_state["trajectories"])
                    if installed_at is not None and trajectories:
                        elapsed_s = time.monotonic() - float(installed_at)
                        duration_s = max(trajectory.duration_s for trajectory in trajectories.values())
                        if elapsed_s > duration_s + args.plan_servo_stale_timeout_s:
                            plan_servo_state["installed_at"] = None
                            plan_servo_state["trajectories"] = {}
                            print(
                                "[ASYNC][plan-servo][WARN] stale trajectory expired: "
                                f"elapsed={elapsed_s:.3f}s duration={duration_s:.3f}s"
                            )
                        else:
                            samples = {
                                arm_name: trajectory.sample(elapsed_s) for arm_name, trajectory in trajectories.items()
                            }
                            readback_enabled = (
                                debug_writer.enabled
                                and args.plan_servo_debug_readback_every_n_samples > 0
                                and sample_index % args.plan_servo_debug_readback_every_n_samples == 0
                            )
                            with robot_lock:
                                for arm_name, (q_rad, _dq_rad_s) in samples.items():
                                    _set_xarm_joint_servo(arms[arm_name], q_rad)
                                actual_states = (
                                    {
                                        arm_name: _read_xarm_joint_state(arm)
                                        for arm_name, arm in arms.items()
                                    }
                                    if readback_enabled
                                    else {}
                                )
                            plan_servo_state["last_commands"] = {
                                arm_name: (q_rad.copy(), dq_rad_s.copy())
                                for arm_name, (q_rad, dq_rad_s) in samples.items()
                            }
                            version = int(plan_servo_state["version"])
                            command_time = time.monotonic()
                            for arm_name, (q_rad, dq_rad_s) in samples.items():
                                q_actual, dq_actual = actual_states.get(arm_name, (None, None))
                                debug_writer.write(
                                    "joint_servo",
                                    {
                                        "sample_index": sample_index,
                                        "command_time": command_time,
                                        "arm_name": arm_name,
                                        "trajectory_version": version,
                                        "trajectory_elapsed_s": elapsed_s,
                                        "q_command_rad": q_rad,
                                        "dq_command_rad_s": dq_rad_s,
                                        "q_actual_rad": q_actual,
                                        "dq_actual_rad_s": dq_actual,
                                    },
                                )
                            sample_index += 1
            except Exception as exc:
                now = time.monotonic()
                if now - last_warning_time >= 1.0:
                    print(f"[ASYNC][plan-servo][WARN] {exc}")
                    last_warning_time = now
            next_tick += period_s
            sleep_s = next_tick - time.perf_counter()
            if sleep_s > 0.0:
                time.sleep(sleep_s)
            else:
                next_tick = time.perf_counter()

    def inference_loop() -> None:
        infer_index = 0
        next_request_step = 0
        while not stop_event.is_set():
            if paused_event.is_set():
                time.sleep(0.02)
                next_request_step = get_step()
                continue
            current_step = get_step()
            if current_step < next_request_step and action_buffer.pending_count_from(current_step) > args.min_buffer_steps:
                time.sleep(0.005)
                continue
            request_step = current_step
            request_generation = get_generation()
            next_request_step = request_step + args.inference_interval_steps
            try:
                obs_id = next_debug_id("obs_id")
                capture_time = time.perf_counter()
                obs, image_obs = read_observation()
                timed_obs = TimedObservation(
                    obs_id=obs_id,
                    request_step=request_step,
                    capture_time=capture_time,
                    send_time=time.perf_counter(),
                    buffer_size=action_buffer.pending_count_from(request_step),
                    robot_state=obs.get("state"),
                    image_metadata=_image_metadata(image_obs),
                )
                obs["__async_rollout"] = {
                    "obs_id": obs_id,
                    "request_step": request_step,
                    "control_hz": args.control_hz,
                    "prev_chunk_id": last_completed_chunk_id["value"],
                    "prev_leftover_steps": action_buffer.pending_count_from(request_step),
                    "delay_mode": args.inference_delay_mode,
                    "delay_steps": last_delay_steps["value"],
                    "planner_delay_steps": last_planner_delay_steps["value"],
                }
                if args.rtc_chunk_conditioning:
                    obs["__rtc_rollout"] = {
                        "enabled": True,
                        "session_id": rtc_session_id,
                        "generation": request_generation,
                        "request_step": request_step,
                        "delay_steps": rtc_request_delay_steps(),
                        "soft_horizon_steps": args.rtc_soft_horizon_steps,
                        "free_tail_steps": args.rtc_free_tail_steps,
                    }
                debug_writer.write("observations", timed_obs)
                request_time = time.perf_counter()
                resp = policy_client.infer(obs)
                latency_s = time.perf_counter() - request_time
                if request_generation != get_generation():
                    print(f"[ASYNC][infer] discard stale response idx={infer_index}")
                    infer_index += 1
                    continue
                latency_steps = latency_estimator.observe(latency_s)
                current_merge_step = get_step()
                buffer_empty = action_buffer.pending_count_from(current_merge_step) == 0
                if args.reset_delay_on_empty_buffer and (buffer_empty or not action_buffer.has_last_action()):
                    latency_steps = 0
                if args.max_inference_delay_steps >= 0:
                    latency_steps = min(latency_steps, args.max_inference_delay_steps)
                update_total_delay_steps(latency_steps)
                mask_payload = resp.get(base.MASK_OVERLAY_KEY, {}) if args.mask_overlay else {}
                if args.mask_overlay and mask_debug_dir is not None:
                    base._dump_mask_rollout_debug(
                        debug_dir=mask_debug_dir,
                        view=effective_mask_view,
                        infer_index=infer_index,
                        payload=mask_payload,
                    )
                if args.mask_overlay and mask_payload.get("needs_reprompt"):
                    print(f"[MASK] tracking 丢失，重新打点: {mask_payload.get('reprompt_reason')}")
                    base._preview_mask_overlay(
                        policy_client,
                        image_rgb=np.asarray(image_obs[effective_mask_view]),
                        view=effective_mask_view,
                        point_xy=None,
                        text_prompt=args.mask_prompt_text,
                        alpha=args.mask_alpha,
                        debug_dir=mask_debug_dir,
                    )
                    infer_index += 1
                    continue
                actions_all = resp["actions"] if "actions" in resp else resp["action"]
                actions_all = np.array(actions_all, dtype=np.float64, copy=True)
                if actions_all.ndim == 1:
                    actions_all = actions_all.reshape(1, -1)
                actions_all = base._adjust_gripper_actions(actions_all, arm_mode=args.arm_mode, single_arm_index=single_arm_index)
                chunk_id = next_debug_id("chunk_id")
                resp_server_timing = resp.get("server_timing", {})
                rtc_payload = resp.get("rtc", {})
                rtc_applied = bool(rtc_payload.get("applied", False))
                merge_request_step = int(resp.get("action_base_step", request_step)) if rtc_applied else request_step
                merge_latency_steps = 0 if rtc_applied else latency_steps
                merge_action_start = 0 if rtc_applied else args.action_start
                merge_action_end = args.action_end
                stats = action_buffer.merge_chunk(
                    actions_all,
                    request_step=merge_request_step,
                    current_step=current_merge_step,
                    action_start=merge_action_start,
                    action_end=merge_action_end,
                    latency_steps=merge_latency_steps,
                    chunk_id=chunk_id,
                    source_obs_id=obs_id,
                    latency_s=latency_s,
                )
                planner_update = None
                if args.xarm_control_mode == "plan-servo":
                    planner_started_at = time.monotonic()
                    try:
                        planner_update = update_plan_servo_trajectory(
                            current_merge_step,
                            expected_generation=request_generation,
                        )
                    except Exception as exc:
                        planner_latency_s = time.monotonic() - planner_started_at
                        planner_update = {
                            "updated": False,
                            "reason": str(exc),
                            "planner_latency_s": planner_latency_s,
                            "planner_latency_steps": max(int(np.ceil(planner_latency_s * args.control_hz)), 0),
                        }
                        print(f"[ASYNC][plan-servo][WARN] keeping previous trajectory: {exc}")
                    planner_latency_s = float(
                        planner_update.setdefault("planner_latency_s", time.monotonic() - planner_started_at)
                    )
                    planner_update.setdefault(
                        "planner_latency_steps",
                        max(int(np.ceil(planner_latency_s * args.control_hz)), 0),
                    )
                    planner_update["total_delay_steps"] = update_total_delay_steps(
                        latency_steps,
                        int(planner_update["planner_latency_steps"]),
                    )
                debug_writer.write(
                    "chunks",
                    {
                        "chunk_id": chunk_id,
                        "obs_id": obs_id,
                        "request_step": request_step,
                        "action_base_step": merge_request_step,
                        "current_merge_step": current_merge_step,
                        "latency_s": latency_s,
                        "delay_steps": latency_steps,
                        "merge_delay_steps": merge_latency_steps,
                        "server_timing": resp_server_timing,
                        "async_rollout_echo": resp.get("async_rollout_echo"),
                        "rtc": rtc_payload,
                        "planner_update": planner_update,
                        "inserted": stats.inserted,
                        "blended": stats.blended,
                        "skipped": stats.skipped_expired,
                        "buffer": action_buffer.pending_count_from(get_step()),
                    },
                )
                for event in stats.events:
                    debug_writer.write(
                        "actions",
                        TimedAction(
                            chunk_id=chunk_id,
                            action_index=int(event["action_index"]),
                            target_step=event["target_step"],
                            action=event["action"],
                            merge_type=str(event["merge_type"]),
                            blend_weight=event["blend_weight"],
                            source_obs_id=obs_id,
                            latency_s=latency_s,
                            delay_steps=merge_latency_steps,
                        ),
                    )
                last_completed_chunk_id["value"] = chunk_id
                print(
                    "[ASYNC][infer] "
                    f"idx={infer_index} request_step={request_step} latency={latency_s:.3f}s "
                    f"delay_steps={latency_steps} merge_delay={merge_latency_steps} rtc={rtc_applied} "
                    f"inserted={stats.inserted} blended={stats.blended} "
                    f"skipped={stats.skipped_expired} buffer={action_buffer.pending_count_from(get_step())}"
                )
                if planner_update is not None:
                    print(f"[ASYNC][plan-servo] {planner_update}")
                infer_index += 1
            except Exception as exc:
                print(f"[ASYNC][infer][WARN] {exc}")
                time.sleep(0.1)

    def control_loop() -> None:
        period_s = 1.0 / args.control_hz
        next_tick = time.perf_counter()
        last_log = 0.0
        future_gap_active = False
        while not stop_event.is_set():
            if paused_event.is_set():
                next_tick = time.perf_counter() + period_s
                time.sleep(0.02)
                continue
            step = get_step()
            read = action_buffer.pop(step)
            next_pending_step = action_buffer.next_pending_step_after(step) if read.missing else None
            if next_pending_step is not None and not future_gap_active:
                print(
                    "[ASYNC][control][WARN] future action gap fallback: "
                    f"holding step={step} until buffered step={next_pending_step}"
                )
            future_gap_active = next_pending_step is not None
            if read.action is None:
                if should_advance_control_step(read, has_future_action=future_gap_active):
                    advance_step()
                now = time.perf_counter()
                if args.async_log_interval_s <= 0.0 or now - last_log >= args.async_log_interval_s:
                    last_log = now
                    print(f"[ASYNC][control] step={step} missing_action=True buffer={action_buffer.pending_count_from(step)}")
                next_tick += period_s
                sleep_s = next_tick - time.perf_counter()
                if sleep_s > 0:
                    time.sleep(sleep_s)
                else:
                    next_tick = time.perf_counter()
                continue
            raw_action = np.asarray(read.action, dtype=np.float64).copy()
            limited_action, limit_info = limit_action_step(
                raw_action,
                last_executed_action["value"],
                max_position_step_m=args.max_position_step_m,
                max_rotation_step_deg=args.max_rotation_step_deg,
                max_gripper_step=args.max_gripper_step,
            )
            readback_enabled = (
                args.async_debug_readback_every_n_steps > 0
                and step % args.async_debug_readback_every_n_steps == 0
            )
            robot_pose_before = read_debug_robot_pose() if readback_enabled else None
            execute_time = time.perf_counter()
            try:
                execute_action(limited_action)
            except Exception as exc:
                print(f"[ASYNC][control][WARN] step={step} {exc}")
            robot_pose_after = read_debug_robot_pose() if readback_enabled else None
            tracking_error = action_tracking_error(limited_action, robot_pose_after)
            command_delta = action_command_delta(limited_action, last_executed_action["value"])
            last_executed_action["value"] = limited_action.copy()
            debug_writer.write(
                "executions",
                ExecutedAction(
                    control_step=step,
                    execute_time=execute_time,
                    action=limited_action,
                    held=read.held,
                    missing=read.missing,
                    buffer_size=action_buffer.pending_count_from(step),
                    robot_pose_before=robot_pose_before,
                    robot_pose_after=robot_pose_after,
                    command_delta=command_delta,
                    tracking_error=tracking_error,
                    raw_action=raw_action,
                    limited_action=limited_action,
                    limit_applied=bool(limit_info["limit_applied"]),
                    position_delta_m=limit_info["position_delta_m"],
                    rotation_delta_deg=limit_info["rotation_delta_deg"],
                ),
            )
            if should_advance_control_step(read, has_future_action=future_gap_active):
                advance_step()
            now = time.perf_counter()
            if args.async_log_interval_s <= 0.0 or now - last_log >= args.async_log_interval_s:
                last_log = now
                print(f"[ASYNC][control] step={step} held={read.held} missing={read.missing} buffer={action_buffer.pending_count_from(step)}")
            next_tick += period_s
            sleep_s = next_tick - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_tick = time.perf_counter()

    old_term = None
    stdin_fd = sys.stdin.fileno()
    inference_thread = threading.Thread(target=inference_loop, name="async-rollout-inference", daemon=True)
    control_thread = threading.Thread(target=control_loop, name="async-rollout-control", daemon=True)
    plan_servo_thread = threading.Thread(target=plan_servo_sender_loop, name="xarm-plan-servo", daemon=True)
    try:
        if sys.stdin.isatty():
            old_term = termios.tcgetattr(stdin_fd)
            tty.setcbreak(stdin_fd)
        enter_runtime_control_mode()
        print("[INFO] async rollout started. 按 s：复位并暂停；按 c：继续；按 q：退出")
        inference_thread.start()
        control_thread.start()
        if args.xarm_control_mode == "plan-servo":
            plan_servo_thread.start()
        while not stop_event.is_set():
            ch = base._stdin_read_char_nonblocking()
            if ch in ("q", "Q"):
                stop_event.set()
                break
            if ch in ("s", "S"):
                paused_event.set()
                action_buffer.clear()
                bump_generation()
                set_step(0)
                reset_active_arms()
                print("[INFO] 已复位到初始位姿，推理已暂停；按 c 继续")
            elif ch in ("c", "C"):
                action_buffer.clear()
                clear_plan_servo_trajectory()
                bump_generation()
                set_step(0)
                enter_runtime_control_mode()
                paused_event.clear()
                print("[INFO] 继续 async rollout")
            time.sleep(0.02)
    finally:
        stop_event.set()
        inference_thread.join(timeout=2.0)
        control_thread.join(timeout=2.0)
        if args.xarm_control_mode == "plan-servo":
            plan_servo_thread.join(timeout=2.0)
        clear_plan_servo_trajectory()
        exit_runtime_control_mode()
        if old_term is not None:
            termios.tcsetattr(stdin_fd, termios.TCSADRAIN, old_term)
        for cap in caps.values():
            cap.release()
        debug_writer.close(
            {
                "control_hz": args.control_hz,
                "inference_interval_steps": args.inference_interval_steps,
                "xarm_control_mode": args.xarm_control_mode,
                "arm_mode": args.arm_mode,
                "plan_servo_hz": args.plan_servo_hz,
                "plan_servo_model_action_dt": plan_servo_model_action_dt(),
                "plan_servo_chunk_max_waypoints": args.plan_servo_chunk_max_waypoints,
                "plan_servo_ik_backend": args.plan_servo_ik_backend,
                "plan_servo_urdf": args.plan_servo_urdf,
                "plan_servo_urdf_tip_link": args.plan_servo_urdf_tip_link,
                "plan_servo_tcp_offset": args.plan_servo_tcp_offset,
                "plan_servo_tcp_offsets_m_deg": local_ik_tcp_offsets,
                "plan_servo_max_ik_joint_step_rad": args.plan_servo_max_ik_joint_step_rad,
                "plan_servo_max_joint_velocity_rad_s": args.plan_servo_max_joint_velocity_rad_s,
                "plan_servo_max_joint_acceleration_rad_s2": args.plan_servo_max_joint_acceleration_rad_s2,
                "plan_servo_max_tracking_error_rad": args.plan_servo_max_tracking_error_rad,
                "plan_servo_debug_readback_every_n_samples": args.plan_servo_debug_readback_every_n_samples,
                "plan_servo_stale_timeout_s": args.plan_servo_stale_timeout_s,
            }
        )
        print("[INFO] 结束，摄像头已释放。")


if __name__ == "__main__":
    main()
