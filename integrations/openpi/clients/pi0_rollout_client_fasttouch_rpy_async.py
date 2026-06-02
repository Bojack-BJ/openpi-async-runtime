#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Async RTC-style rollout client for FastTouch.

This script intentionally does not modify the legacy synchronous rollout client.
It reuses its robot/camera/mask helpers and replaces only the main control flow
with an inference thread, a fixed-rate control thread, and an RTC-style action buffer.
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
from async_rollout_core import action_command_delta
from async_rollout_core import action_tracking_error
from async_rollout_core import limit_action_step
from async_rollout_core import plan_joint_cubic_trajectory
from async_rollout_core import should_advance_control_step
import pi0_rollout_client_fasttouch_rpy as base


FASTTOUCH_SINGLE_RPY_ACTION_INDICES = (3, 4, 5)
FASTTOUCH_DUAL_RPY_ACTION_INDICES = (3, 4, 5, 10, 11, 12)
FASTTOUCH_GRIPPER_UPDATE_INTERVAL_S = 0.1
FASTTOUCH_GRIPPER_UPDATE_THRESHOLD = 0.02


def _set_fasttouch_joint_servo(arm, q_rad: np.ndarray, dq_rad_s: np.ndarray) -> None:
    result = arm.set_joint_raw(
        np.asarray(q_rad, dtype=np.float64).tolist(),
        np.asarray(dq_rad_s, dtype=np.float64).tolist(),
    )
    if isinstance(result, bool) and not result:
        raise RuntimeError("FastTouch set_joint_raw returned False")
    if isinstance(result, int) and result != 0:
        raise RuntimeError(f"FastTouch set_joint_raw failed: code={result}")


def _read_fasttouch_joint_state(
    arm,
    *,
    arm_name: str,
    robot_lock: threading.Lock | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    with robot_lock if robot_lock is not None else contextlib.nullcontext():
        q_rad = np.asarray(arm.get_joint_positions(), dtype=np.float64)
        dq_rad_s = np.asarray(arm.get_joint_velocities(), dtype=np.float64)
    if q_rad.ndim != 1:
        raise RuntimeError(f"{arm_name} FastTouch joint positions have invalid shape: {q_rad.shape}")
    if dq_rad_s.shape != q_rad.shape:
        raise RuntimeError(f"{arm_name} FastTouch joint velocity has invalid shape: {dq_rad_s.shape}")
    return q_rad, dq_rad_s


def _parse_init_pose(value: str) -> tuple[list[float], list[float]]:
    parts = [float(v) for v in value.split(",")]
    if len(parts) != 6:
        raise ValueError(f"init_pose needs 6 values x,y,z,roll,pitch,yaw, got: {value}")
    x, y, z, roll, pitch, yaw = parts
    return [x, y, z], [roll, pitch, yaw]


def _add_async_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--control_hz", type=float, default=20.0, help="Fixed robot command frequency in Hz")
    parser.add_argument("--inference_interval_steps", type=int, default=8, help="Try one policy inference every N control steps")
    parser.add_argument("--min_buffer_steps", type=int, default=2, help="Do not overwrite actions this close to execution")
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
    parser.add_argument(
        "--execution_backend",
        choices=("cartesian_raw", "joint_waypoint_chunk", "plan_servo"),
        default="cartesian_raw",
        help="cartesian_raw=逐 tick 笛卡尔下发；joint_waypoint_chunk=SDK planner；plan_servo=SDK IK + 外部 joint planner + raw joint servo",
    )
    # Model actions are low-rate waypoints. The SDK planner interpolates them into
    # its own high-rate execution trajectory and keeps a short old-trajectory prefix
    # when a newer policy chunk replaces the future suffix.
    parser.add_argument(
        "--model_infer_action_dt",
        "--trajectory_dt",
        dest="model_infer_action_dt",
        type=float,
        default=0.05,
        help="每个模型 action waypoint 的时间间隔（秒）；通常等于 1 / control_hz。--trajectory_dt 是兼容旧参数名",
    )
    parser.add_argument("--joint_waypoint_speed_percent", type=float, default=-1.0, help=">0 时让 SDK 按速度比例规划；默认 -1 使用 model_infer_action_dt * waypoint_count 作为轨迹时长")
    parser.add_argument(
        "--chunk_switch_delay_sec",
        "--joint_chunk_switch_delay_sec",
        dest="chunk_switch_delay_sec",
        type=float,
        default=0.05,
        help="SDK 安装新 chunk 时保留旧活动轨迹前缀的时长（秒）；旧前缀不可被新规划覆盖",
    )
    parser.add_argument(
        "--planner_chunk_max_waypoints",
        "--joint_chunk_max_waypoints",
        dest="planner_chunk_max_waypoints",
        type=int,
        default=20,
        help="SDK planner 每次从 ActionBuffer 读取的最大连续未来模型 action 数",
    )
    parser.add_argument("--ik_retries", type=int, default=5, help="每个 waypoint 的 IK 额外重试次数")
    parser.add_argument("--ik_retry_sleep_s", type=float, default=0.0, help="IK 失败重试间隔秒数")
    parser.add_argument("--plan_servo_hz", type=float, default=100.0, help="plan_servo 下发 joint sample 的频率，默认 100Hz")
    parser.add_argument("--plan_servo_max_joint_velocity_rad_s", type=float, default=0.0, help="外部 planner joint 最大速度校验阈值；0 表示关闭")
    parser.add_argument("--plan_servo_max_joint_acceleration_rad_s2", type=float, default=0.0, help="外部 planner joint 最大加速度校验阈值；0 表示关闭")
    parser.add_argument("--plan_servo_stale_timeout_s", type=float, default=0.5, help="计划结束后最多保持末点的时长；超时后停止发送陈旧 target")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--description", required=True, help="任务自然语言指令")
    parser.add_argument("--arm_mode", choices=("dual", "single"), default="dual", help="dual=双臂 14D；single=单臂 7D")
    parser.add_argument("--single_arm", choices=("robot_0", "robot_1"), default=None, help="single 模式必须显式指定使用哪只 arm")
    parser.add_argument("--single_image_key", default="front", help="single 模式发给 policy 的 image key，FastUMI 单臂默认 front")
    parser.add_argument("--left_can", default="can0", help="左臂（robot_0）CAN 接口")
    parser.add_argument("--right_can", default="can1", help="右臂（robot_1）CAN 接口")
    parser.add_argument("--server_ip", default=base.SERVER_IP, help="policy server host")
    parser.add_argument("--port", type=int, default=base.PORT, help="policy server port")
    parser.add_argument("--camera_dev0", type=int, default=base.DEV, help="robot_0 图像对应的视频设备编号")
    parser.add_argument("--camera_dev1", type=int, default=base.DEV + 2, help="robot_1 图像对应的视频设备编号")
    parser.add_argument("--init_pose_left", type=str, default="0.3,0.0,0.16,0.0,0.0,0.0", help="robot_0 初始位姿: x,y,z,roll,pitch,yaw（弧度）")
    parser.add_argument("--init_pose_right", type=str, default="0.3,0.0,0.16,0.0,0.0,0.0", help="robot_1 初始位姿: x,y,z,roll,pitch,yaw（弧度）")
    parser.add_argument(
        "--tcp_offset",
        type=str,
        default="0.0,0.0,0.0",
        help="法兰系下「法兰原点→夹爪尖 TCP」的平移 x,y,z（米）。",
    )
    parser.add_argument("--tcp_debug", action="store_true", help="打印法兰/TCP 互转调试信息")
    parser.add_argument("--action_start", type=int, default=0, help="本次推理返回的 action 序列起始下标（含）")
    parser.add_argument("--action_end", type=int, default=20, help="本次推理返回的 action 序列结束下标（含）")
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
    if args.model_infer_action_dt <= 0.0:
        parser.error("--model_infer_action_dt must be > 0")
    if args.joint_waypoint_speed_percent > 1.0:
        parser.error("--joint_waypoint_speed_percent must be <= 1.0")
    if args.chunk_switch_delay_sec < 0.0:
        parser.error("--chunk_switch_delay_sec must be >= 0")
    if args.planner_chunk_max_waypoints < 1:
        parser.error("--planner_chunk_max_waypoints must be >= 1")
    if args.plan_servo_hz <= 0.0:
        parser.error("--plan_servo_hz must be > 0")
    if args.plan_servo_stale_timeout_s < 0.0:
        parser.error("--plan_servo_stale_timeout_s must be >= 0")

    single_arm = args.single_arm
    single_arm_index = base._arm_index(single_arm) if single_arm is not None else 0
    effective_mask_view = args.mask_view or args.single_image_key
    tcp_off = base.parse_tool_offset_xyz(args.tcp_offset)
    init_pos0, init_euler0 = _parse_init_pose(args.init_pose_left)
    init_pos1, init_euler1 = _parse_init_pose(args.init_pose_right)
    init_by_arm = {
        "robot_0": (init_pos0, init_euler0),
        "robot_1": (init_pos1, init_euler1),
    }

    arms: dict[str, base.SingleArm] = {}
    if is_dual or single_arm == "robot_0":
        arms["robot_0"] = base.SingleArm(can_interface_=args.left_can)
    if is_dual or single_arm == "robot_1":
        arms["robot_1"] = base.SingleArm(can_interface_=args.right_can)
    time.sleep(2.0)
    if args.execution_backend in ("joint_waypoint_chunk", "plan_servo"):
        for arm_name, arm in arms.items():
            required = ["solve_ik", "get_joint_positions"]
            if args.execution_backend == "joint_waypoint_chunk":
                required.append("update_joint_waypoint_chunk_with_gripper")
            else:
                required.extend(["get_joint_velocities", "set_joint_raw"])
            missing = [name for name in required if not hasattr(arm, name)]
            if missing:
                raise RuntimeError(
                    f"{arm_name} 当前 Startouch SDK 缺少 {missing}，不能使用 --execution_backend {args.execution_backend}。"
                )

    robot_lock = threading.Lock()
    control_step = {"value": 0}
    generation = {"value": 0}
    step_lock = threading.Lock()
    stop_event = threading.Event()
    paused_event = threading.Event()
    plan_servo_lock = threading.Lock()
    plan_servo_state: dict[str, object] = {"installed_at": None, "trajectories": {}, "version": 0}
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

    def clear_plan_servo_trajectory() -> None:
        with plan_servo_lock:
            plan_servo_state["installed_at"] = None
            plan_servo_state["trajectories"] = {}
            plan_servo_state["version"] = int(plan_servo_state["version"]) + 1

    def reset_active_arms() -> None:
        clear_plan_servo_trajectory()
        with robot_lock:
            if is_dual:
                base.reset_arms_to_init(arms["robot_0"], arms["robot_1"], init_pos0, init_euler0, init_pos1, init_euler1)
            else:
                assert single_arm is not None
                init_pos, init_euler = init_by_arm[single_arm]
                base.reset_arm_to_init(arms[single_arm], init_pos, init_euler)
        for state in gripper_state.values():
            state["open"] = None
            state["time"] = 0.0

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
        cyclic_indices=FASTTOUCH_DUAL_RPY_ACTION_INDICES if is_dual else FASTTOUCH_SINGLE_RPY_ACTION_INDICES,
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
                    pose_euler0 = arms["robot_0"].get_ee_pose_euler()
                    pose_euler1 = arms["robot_1"].get_ee_pose_euler()
                    g0 = float(arms["robot_0"].get_gripper_position())
                    g1 = float(arms["robot_1"].get_gripper_position())
                pos0 = np.asarray(pose_euler0[0], dtype=np.float64)
                rpy0 = np.rad2deg(np.asarray(pose_euler0[1], dtype=np.float64))
                pos1 = np.asarray(pose_euler1[0], dtype=np.float64)
                rpy1 = np.rad2deg(np.asarray(pose_euler1[1], dtype=np.float64))
                return np.asarray([*pos0, *rpy0, g0, *pos1, *rpy1, g1], dtype=np.float64)
            assert single_arm is not None
            with robot_lock:
                pose_euler = arms[single_arm].get_ee_pose_euler()
                gripper = float(arms[single_arm].get_gripper_position())
            pos = np.asarray(pose_euler[0], dtype=np.float64)
            rpy = np.rad2deg(np.asarray(pose_euler[1], dtype=np.float64))
            return np.asarray([*pos, *rpy, gripper], dtype=np.float64)
        except Exception as exc:
            print(f"[ASYNC][debug][WARN] failed to read FastTouch pose: {exc}")
            return None

    def read_observation():
        if is_dual:
            with robot_lock:
                pos0, quat_wxyz0 = arms["robot_0"].get_ee_pose_quat()
                qw0, qx0, qy0, qz0 = quat_wxyz0
                quat0 = np.array([qx0, qy0, qz0, qw0])
                p_tcp0 = base.flange_position_to_tcp(pos0, quat_wxyz0, tcp_off)
                pos1, quat_wxyz1 = arms["robot_1"].get_ee_pose_quat()
                qw1, qx1, qy1, qz1 = quat_wxyz1
                quat1 = np.array([qx1, qy1, qz1, qw1])
                p_tcp1 = base.flange_position_to_tcp(pos1, quat_wxyz1, tcp_off)
                gripper_open0 = float(arms["robot_0"].get_gripper_position())
                gripper_open1 = float(arms["robot_1"].get_gripper_position())
            gripper_open0 = float(np.clip(np.where(gripper_open0 < 0.3, gripper_open0 + 0.2, gripper_open0), 0.0, 1.0))
            gripper_open1 = float(np.clip(np.where(gripper_open1 < 0.3, gripper_open1 + 0.2, gripper_open1), 0.0, 1.0))
            state_vec = np.array([*p_tcp0, *quat0, gripper_open0, *p_tcp1, *quat1, gripper_open1], dtype=np.float32)
            image_obs = {
                "robot_0": base.image_tools.convert_to_uint8(base._capture_resized(caps["robot_0"])),
                "robot_1": base.image_tools.convert_to_uint8(base._capture_resized(caps["robot_1"])),
            }
        else:
            assert single_arm is not None
            with robot_lock:
                pos, quat_wxyz = arms[single_arm].get_ee_pose_quat()
                qw, qx, qy, qz = quat_wxyz
                quat = np.array([qx, qy, qz, qw])
                p_tcp = base.flange_position_to_tcp(pos, quat_wxyz, tcp_off)
                gripper_open = float(arms[single_arm].get_gripper_position())
            gripper_open = float(np.clip(np.where(gripper_open < 0.3, gripper_open + 0.2, gripper_open), 0.0, 1.0))
            state_vec = np.array([*p_tcp, *quat, gripper_open], dtype=np.float32)
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
            or abs(float(g_open) - float(last_open)) >= FASTTOUCH_GRIPPER_UPDATE_THRESHOLD
            or now - last_time >= FASTTOUCH_GRIPPER_UPDATE_INTERVAL_S
        ):
            arms[arm_name].setGripperPosition(float(g_open))
            state["open"] = float(g_open)
            state["time"] = now

    def execute_gripper_action(action: np.ndarray) -> None:
        with robot_lock:
            if is_dual:
                maybe_set_gripper("robot_0", float(np.clip(action[6], 0.0, 1.0)))
                maybe_set_gripper("robot_1", float(np.clip(action[13], 0.0, 1.0)))
                return
            assert single_arm is not None
            maybe_set_gripper(single_arm, float(np.clip(action[6], 0.0, 1.0)))

    def execute_action(action: np.ndarray) -> None:
        if is_dual:
            x0, y0, z0, r0, p0, yy0, g0, x1, y1, z1, r1, p1, yy1, g1 = action[:14]
            euler0 = np.deg2rad([r0, p0, yy0]).tolist()
            euler1 = np.deg2rad([r1, p1, yy1]).tolist()
            pos0 = base.tcp_position_to_flange([x0, y0, z0], euler0, tcp_off).tolist()
            pos1 = base.tcp_position_to_flange([x1, y1, z1], euler1, tcp_off).tolist()
            with robot_lock:
                arms["robot_0"].set_end_effector_pose_euler_raw(pos=pos0, euler=euler0)
                arms["robot_1"].set_end_effector_pose_euler_raw(pos=pos1, euler=euler1)
                arms["robot_0"].setGripperPosition(float(np.clip(g0, 0.0, 1.0)))
                arms["robot_1"].setGripperPosition(float(np.clip(g1, 0.0, 1.0)))
        else:
            assert single_arm is not None
            x, y, z, roll, pitch, yaw, g_open = action[:7]
            euler = np.deg2rad([roll, pitch, yaw]).tolist()
            pos = base.tcp_position_to_flange([x, y, z], euler, tcp_off).tolist()
            with robot_lock:
                arms[single_arm].set_end_effector_pose_euler_raw(pos=pos, euler=euler)
                arms[single_arm].setGripperPosition(float(np.clip(g_open, 0.0, 1.0)))

    def update_joint_waypoint_chunk_planner(current_step: int) -> dict[str, object]:
        """Install a replannable SDK trajectory from the current ActionBuffer suffix."""
        start_step, actions = action_buffer.contiguous_actions_from(
            current_step,
            max_steps=args.planner_chunk_max_waypoints,
        )
        if start_step is None or len(actions) == 0:
            return {"updated": False, "reason": "no_contiguous_actions"}
        trajectory_time_sec = args.model_infer_action_dt * len(actions)
        motion_kwargs = {
            **base._joint_waypoints_motion_kwargs(trajectory_time_sec, args.joint_waypoint_speed_percent),
            "switch_delay_sec": args.chunk_switch_delay_sec,
        }
        with robot_lock:
            if is_dual:
                left_poses, right_poses, left_grippers, right_grippers = base._build_dual_pose_trajectories(actions, tcp_off)
                left_joints = base._solve_joint_waypoints_from_poses(
                    arms["robot_0"],
                    left_poses,
                    "left",
                    ik_retries=args.ik_retries,
                    ik_retry_sleep_s=args.ik_retry_sleep_s,
                )
                right_joints = base._solve_joint_waypoints_from_poses(
                    arms["robot_1"],
                    right_poses,
                    "right",
                    ik_retries=args.ik_retries,
                    ik_retry_sleep_s=args.ik_retry_sleep_s,
                )
                durations = base._run_blocking_calls_concurrently(
                    [
                        (
                            "left update_joint_waypoint_chunk_with_gripper",
                            arms["robot_0"].update_joint_waypoint_chunk_with_gripper,
                            (left_joints.tolist(), left_grippers.tolist()),
                            motion_kwargs,
                        ),
                        (
                            "right update_joint_waypoint_chunk_with_gripper",
                            arms["robot_1"].update_joint_waypoint_chunk_with_gripper,
                            (right_joints.tolist(), right_grippers.tolist()),
                            motion_kwargs,
                        ),
                    ]
                )
            else:
                assert single_arm is not None
                poses, grippers = base._build_single_pose_trajectory(actions, tcp_off)
                joints = base._solve_joint_waypoints_from_poses(
                    arms[single_arm],
                    poses,
                    single_arm,
                    ik_retries=args.ik_retries,
                    ik_retry_sleep_s=args.ik_retry_sleep_s,
                )
                durations = {
                    single_arm: arms[single_arm].update_joint_waypoint_chunk_with_gripper(
                        joints.tolist(),
                        grippers.tolist(),
                        **motion_kwargs,
                    )
                }
        return {
            "updated": True,
            "start_step": start_step,
            "waypoints": len(actions),
            "model_infer_action_dt": args.model_infer_action_dt,
            "trajectory_time_sec": trajectory_time_sec,
            "chunk_switch_delay_sec": args.chunk_switch_delay_sec,
            "durations": durations,
        }

    def update_plan_servo_trajectory(current_step: int, *, expected_generation: int) -> dict[str, object]:
        """Replace the external joint-servo trajectory from the latest buffered suffix."""
        planning_started_at = time.monotonic()
        planning_start_step = get_step()
        start_step, actions = action_buffer.contiguous_actions_from(
            current_step,
            max_steps=args.planner_chunk_max_waypoints,
        )
        if start_step is None or len(actions) == 0:
            return {"updated": False, "reason": "no_contiguous_actions"}
        if is_dual:
            left_poses, right_poses, _left_grippers, _right_grippers = base._build_dual_pose_trajectories(
                actions,
                tcp_off,
            )
            poses_by_arm = {"robot_0": left_poses, "robot_1": right_poses}
        else:
            assert single_arm is not None
            poses, _grippers = base._build_single_pose_trajectory(actions, tcp_off)
            poses_by_arm = {single_arm: poses}
        joint_waypoints = {
            arm_name: base._solve_joint_waypoints_from_poses(
                arm,
                poses_by_arm[arm_name],
                arm_name,
                ik_retries=args.ik_retries,
                ik_retry_sleep_s=args.ik_retry_sleep_s,
                require_move_joint_waypoints=False,
                robot_lock=robot_lock,
            )
            for arm_name, arm in arms.items()
        }
        with robot_lock:
            start_states = {
                arm_name: _read_fasttouch_joint_state(arm, arm_name=arm_name)
                for arm_name, arm in arms.items()
            }
        if expected_generation != get_generation():
            return {"updated": False, "reason": "stale_generation_after_ik"}
        install_step = get_step()
        expired_waypoints = max(install_step - start_step, 0)
        if expired_waypoints:
            joint_waypoints = {
                arm_name: waypoints[expired_waypoints:] for arm_name, waypoints in joint_waypoints.items()
            }
            start_step += expired_waypoints
        if any(len(waypoints) == 0 for waypoints in joint_waypoints.values()):
            planner_latency_s = time.monotonic() - planning_started_at
            return {
                "updated": False,
                "reason": "all_waypoints_expired_during_ik",
                "planner_latency_s": planner_latency_s,
                "planner_latency_steps": max(int(np.ceil(planner_latency_s * args.control_hz)), 0),
            }
        start_delay_s = max(start_step - install_step, 0) / args.control_hz
        if start_delay_s > 0.0:
            print(
                "[ASYNC][plan-servo][WARN] future-gap fallback active: "
                f"install_step={install_step} start_step={start_step} hold={start_delay_s:.3f}s"
            )
        trajectories = {
            arm_name: plan_joint_cubic_trajectory(
                start_states[arm_name][0],
                start_states[arm_name][1],
                joint_waypoints[arm_name],
                waypoint_dt_s=args.model_infer_action_dt,
                sample_hz=args.plan_servo_hz,
                start_delay_s=start_delay_s,
                max_velocity_rad_s=args.plan_servo_max_joint_velocity_rad_s,
                max_acceleration_rad_s2=args.plan_servo_max_joint_acceleration_rad_s2,
            )
            for arm_name in arms
        }
        installed_at = time.monotonic()
        planner_latency_s = installed_at - planning_started_at
        planner_latency_steps = max(int(np.ceil(planner_latency_s * args.control_hz)), 0)
        with plan_servo_lock:
            plan_servo_state["installed_at"] = installed_at
            plan_servo_state["trajectories"] = trajectories
            plan_servo_state["version"] = int(plan_servo_state["version"]) + 1
            version = int(plan_servo_state["version"])
        return {
            "updated": True,
            "version": version,
            "planning_start_step": planning_start_step,
            "install_step": install_step,
            "start_step": start_step,
            "waypoints": len(next(iter(joint_waypoints.values()))),
            "expired_waypoints_during_ik": expired_waypoints,
            "planner_latency_s": planner_latency_s,
            "planner_latency_steps": planner_latency_steps,
            "model_infer_action_dt": args.model_infer_action_dt,
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
        while not stop_event.is_set():
            if paused_event.is_set():
                next_tick = time.perf_counter() + period_s
                time.sleep(min(period_s, 0.02))
                continue
            with plan_servo_lock:
                installed_at = plan_servo_state["installed_at"]
                trajectories = dict(plan_servo_state["trajectories"])
            if installed_at is not None and trajectories:
                elapsed_s = time.monotonic() - float(installed_at)
                duration_s = max(trajectory.duration_s for trajectory in trajectories.values())
                if elapsed_s > duration_s + args.plan_servo_stale_timeout_s:
                    with plan_servo_lock:
                        if plan_servo_state["installed_at"] == installed_at:
                            plan_servo_state["installed_at"] = None
                            plan_servo_state["trajectories"] = {}
                    print(
                        "[ASYNC][plan-servo][WARN] stale trajectory expired: "
                        f"elapsed={elapsed_s:.3f}s duration={duration_s:.3f}s"
                    )
                    continue
                try:
                    samples = {
                        arm_name: trajectory.sample(elapsed_s) for arm_name, trajectory in trajectories.items()
                    }
                    with robot_lock:
                        for arm_name, (q_rad, dq_rad_s) in samples.items():
                            _set_fasttouch_joint_servo(arms[arm_name], q_rad, dq_rad_s)
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
                if args.execution_backend in ("joint_waypoint_chunk", "plan_servo"):
                    planner_started_at = time.monotonic()
                    try:
                        if args.execution_backend == "joint_waypoint_chunk":
                            planner_update = update_joint_waypoint_chunk_planner(current_merge_step)
                        else:
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
                        print(f"[ASYNC][planner][WARN] {exc}")
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
                        "server_timing": resp.get("server_timing", {}),
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
                    print(f"[ASYNC][planner] {planner_update}")
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
                if args.execution_backend == "cartesian_raw":
                    execute_action(limited_action)
                elif args.execution_backend == "plan_servo":
                    execute_gripper_action(limited_action)
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
    plan_servo_thread = threading.Thread(target=plan_servo_sender_loop, name="fasttouch-plan-servo", daemon=True)
    try:
        if sys.stdin.isatty():
            old_term = termios.tcgetattr(stdin_fd)
            tty.setcbreak(stdin_fd)
        print("[INFO] async rollout started. 按 s：复位并暂停；按 c：继续；按 q：退出")
        inference_thread.start()
        control_thread.start()
        if args.execution_backend == "plan_servo":
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
                paused_event.clear()
                print("[INFO] 继续 async rollout")
            time.sleep(0.02)
    finally:
        stop_event.set()
        inference_thread.join(timeout=2.0)
        control_thread.join(timeout=2.0)
        if args.execution_backend == "plan_servo":
            plan_servo_thread.join(timeout=2.0)
        clear_plan_servo_trajectory()
        if old_term is not None:
            termios.tcsetattr(stdin_fd, termios.TCSADRAIN, old_term)
        for cap in caps.values():
            cap.release()
        for arm in arms.values():
            arm.cleanup()
        debug_writer.close(
            {
                "control_hz": args.control_hz,
                "inference_interval_steps": args.inference_interval_steps,
                "arm_mode": args.arm_mode,
                "execution_backend": args.execution_backend,
                "model_infer_action_dt": args.model_infer_action_dt,
                "chunk_switch_delay_sec": args.chunk_switch_delay_sec,
                "planner_chunk_max_waypoints": args.planner_chunk_max_waypoints,
                "plan_servo_hz": args.plan_servo_hz,
                "plan_servo_stale_timeout_s": args.plan_servo_stale_timeout_s,
            }
        )
        print("[INFO] 结束，摄像头与机械臂已释放。")


if __name__ == "__main__":
    main()
