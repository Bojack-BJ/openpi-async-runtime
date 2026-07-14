#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xArm 在线 rollout client，支持双臂和单臂。

这个脚本保持老 xArm 版本的机械臂控制调用：
    - Bestman_Real_Xarm6(...)
    - robot.get_position()
    - get_gripper_position_robotiq()
    - robot.set_position(...)
    - gripper_goto_robotiq(...)

上层功能与 fasttouch rollout 对齐：
    - action_start/action_end 选择执行 chunk 子区间
    - s/c 非阻塞暂停、复位、继续
    - SAM3 mask overlay 初始点击确认与持续请求
    - action 序列 reshape/copy 与夹爪维修正
    - 相机最新帧抓取、224x224 输入
    - dual: image['robot_0']/image['robot_1']，16 维 state，14 维 action
    - single: image['front']，8 维 state，7 维 action
"""

import argparse
import pathlib
import select
import sys
import termios
import time
import tty

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from openpi_client import image_tools, websocket_client_policy

# ====== 根据你的实际路径修改 ======
sys.path.append("/home/lumos/lxt/BestMan_Xarm/RoboticsToolBox/")
from Bestman_real_xarm6 import Bestman_Real_Xarm6  # noqa: E402


# ====== policy server 参数 ======
SERVER_IP = "180.184.74.93"
PORT = 8005

# ====== 摄像头（YU12 / I420）参数 ======
DEV = 0
W, H, FPS = 1280, 1280, 100

MASK_OVERLAY_KEY = "__mask_overlay"


def init_yu12_camera(dev: int):
    """按 YU12(I420) 模式初始化摄像头。"""
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开 /dev/video{dev}")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YU12"))  # YU12 == I420
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    fcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    fcc = "".join(chr((fcc_int >> (8 * i)) & 0xFF) for i in range(4))
    print(f"[INFO] /dev/video{dev} FOURCC from driver: {fcc}")
    return cap


def grab_rgb(cap):
    """从 YU12(I420) 摄像头抓一帧，返回 RGB 图像 (H, W, 3, uint8)。"""
    ok, raw = cap.read()
    if not ok:
        raise RuntimeError("摄像头读取失败")

    yuv = np.ascontiguousarray(raw).reshape(H * 3 // 2, W)
    bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def grab_rgb_latest(cap, flush_n=4):
    """丢弃旧帧后抓取最新一帧，并输出 RGB 图像。"""
    for _ in range(flush_n):
        cap.grab()
    ok, raw = cap.read()
    if not ok:
        raise RuntimeError("摄像头读取失败")
    yuv = np.ascontiguousarray(raw).reshape(H * 3 // 2, W)
    bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _stdin_read_char_nonblocking():
    """终端模式下非阻塞读一个字符；非 tty 或无可读数据时返回 None。"""
    if not sys.stdin.isatty():
        return None
    readable, _, _ = select.select([sys.stdin], [], [], 0)
    if not readable:
        return None
    try:
        return sys.stdin.read(1)
    except Exception:
        return None


def _stdin_wants_reset_pause() -> bool:
    ch = _stdin_read_char_nonblocking()
    return ch in ("s", "S")


def _parse_xy(value: str | None) -> tuple[int, int] | None:
    if value is None:
        return None
    x_str, y_str = value.split(",", maxsplit=1)
    return int(float(x_str)), int(float(y_str))


def _select_point(image_rgb: np.ndarray, *, title: str) -> tuple[int, int]:
    clicked: list[tuple[int, int]] = []
    display = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked[:] = [(int(x), int(y))]

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(title, on_mouse)
    while not clicked:
        frame = display.copy()
        cv2.putText(
            frame,
            "Click target point, q/esc to cancel",
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        cv2.imshow(title, frame)
        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord("q")):
            cv2.destroyWindow(title)
            raise RuntimeError("Mask prompt selection cancelled.")
    cv2.destroyWindow(title)
    return clicked[0]


def _array_stats(name: str, value) -> str:
    arr = np.asarray(value)
    if arr.size == 0:
        return f"{name}: shape={arr.shape} dtype={arr.dtype} empty"
    return (
        f"{name}: shape={arr.shape} dtype={arr.dtype} "
        f"min={arr.min()} max={arr.max()} mean={float(arr.mean()):.2f} nonzero={int(np.count_nonzero(arr))}"
    )


def _as_uint8_image(image) -> np.ndarray:
    image_np = np.asarray(image)
    if image_np.ndim == 3 and image_np.shape[0] in (3, 4) and image_np.shape[-1] not in (3, 4):
        image_np = np.moveaxis(image_np, 0, -1)
    if image_np.dtype != np.uint8:
        if np.issubdtype(image_np.dtype, np.floating):
            if image_np.size and image_np.min() >= -1.0 and image_np.max() <= 1.0:
                image_np = (image_np + 1.0) * 127.5
            elif image_np.size and image_np.min() >= 0.0 and image_np.max() <= 1.0:
                image_np = image_np * 255.0
        image_np = np.clip(image_np, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image_np)


def _rgb_to_bgr_for_cv2(image) -> np.ndarray:
    image_np = _as_uint8_image(image)
    if image_np.ndim == 2:
        return cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
    if image_np.shape[-1] == 4:
        image_np = image_np[..., :3]
    return cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)


def _save_debug_image(path: pathlib.Path, image_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image_np = _as_uint8_image(image_rgb)
    if image_np.ndim == 2:
        cv2.imwrite(str(path), image_np)
    else:
        cv2.imwrite(str(path), cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))


def _show_overlay_preview(overlay, *, window: str) -> str:
    display = _rgb_to_bgr_for_cv2(overlay)
    cv2.putText(
        display,
        "y/Enter: accept   r: retry   q/Esc: quit",
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
    )
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    while True:
        cv2.imshow(window, display)
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("y"), 13, 10):
            cv2.destroyWindow(window)
            return "y"
        if key == ord("r"):
            cv2.destroyWindow(window)
            return "r"
        if key in (ord("q"), 27):
            cv2.destroyWindow(window)
            return "q"


def _dump_mask_preview_debug(
    *,
    debug_dir: pathlib.Path,
    view: str,
    point_xy: tuple[int, int],
    client_image: np.ndarray,
    payload: dict,
) -> None:
    overlay = np.asarray(payload["overlay"])
    mask = np.asarray(payload.get("mask", np.zeros(client_image.shape[:2], dtype=np.uint8)))
    server_image = payload.get("image")
    print(
        "[MASK DEBUG]",
        f"view={view}",
        f"point={point_xy}",
        f"initialized={payload.get('initialized')}",
        f"score={payload.get('score')}",
        f"bbox={payload.get('bbox_xyxy')}",
    )
    print("[MASK DEBUG]", _array_stats("client_image", client_image))
    if server_image is not None:
        print("[MASK DEBUG]", _array_stats("server_image", server_image))
    print("[MASK DEBUG]", _array_stats("overlay", overlay))
    print("[MASK DEBUG]", _array_stats("mask", mask))

    prefix = debug_dir / f"{view}_x{point_xy[0]}_y{point_xy[1]}"
    _save_debug_image(prefix.with_name(prefix.name + "_client.png"), client_image)
    if server_image is not None:
        _save_debug_image(prefix.with_name(prefix.name + "_server.png"), np.asarray(server_image))
    _save_debug_image(prefix.with_name(prefix.name + "_overlay.png"), overlay)
    cv2.imwrite(str(prefix.with_name(prefix.name + "_mask.png")), mask)
    print(f"[MASK DEBUG] saved preview images under {debug_dir}")


def _dump_mask_rollout_debug(
    *,
    debug_dir: pathlib.Path,
    view: str,
    infer_index: int,
    payload: dict,
) -> None:
    overlay = payload.get("overlay")
    if overlay is None:
        print(f"[MASK DEBUG] infer={infer_index:06d} has no overlay payload; keys={sorted(payload)}")
        return

    overlay = np.asarray(overlay)
    mask = np.asarray(payload.get("mask", np.zeros(overlay.shape[:2], dtype=np.uint8)))
    server_image = payload.get("image")
    print(
        "[MASK DEBUG]",
        f"infer={infer_index:06d}",
        f"view={view}",
        f"initialized={payload.get('initialized')}",
        f"score={payload.get('score')}",
        f"bbox={payload.get('bbox_xyxy')}",
        _array_stats("overlay", overlay),
        _array_stats("mask", mask),
    )

    prefix = debug_dir / "rollout" / f"{infer_index:06d}_{view}"
    if server_image is not None:
        _save_debug_image(prefix.with_name(prefix.name + "_server.png"), np.asarray(server_image))
    _save_debug_image(prefix.with_name(prefix.name + "_overlay.png"), overlay)
    cv2.imwrite(str(prefix.with_name(prefix.name + "_mask.png")), mask)


def _preview_mask_overlay(
    policy_client,
    *,
    image_rgb: np.ndarray,
    view: str,
    point_xy: tuple[int, int] | None,
    text_prompt: str | None,
    alpha: float,
    debug_dir: pathlib.Path | None,
) -> None:
    while True:
        if point_xy is None:
            point_xy = _select_point(image_rgb, title=f"Select SAM3 point: {view}")
        mask_request = {
            "enabled": True,
            "view": view,
            "reset": True,
            "points": np.asarray([point_xy], dtype=np.float32),
            "point_labels": np.asarray([1], dtype=np.int32),
            "preview_only": True,
            "alpha": float(alpha),
        }
        if text_prompt:
            mask_request["text"] = text_prompt
        if debug_dir is not None:
            mask_request["return_image"] = True

        obs = {
            "image": {view: image_tools.convert_to_uint8(image_rgb)},
            MASK_OVERLAY_KEY: mask_request,
        }
        resp = policy_client.infer(obs)
        payload = resp.get(MASK_OVERLAY_KEY, {})
        overlay = payload.get("overlay")
        if overlay is None:
            raise RuntimeError(f"Policy server did not return mask overlay preview: keys={sorted(resp)}")
        if payload.get("needs_reprompt"):
            print(f"[MASK] server 请求重新打点: {payload.get('reprompt_reason')}")
            answer = input("SAM3 未得到有效 mask。[r=重新点击 / q=退出]: ").strip().lower()
            if answer in ("q", "quit"):
                raise RuntimeError("Mask overlay preview rejected.")
            point_xy = None
            continue
        if debug_dir is not None:
            _dump_mask_preview_debug(
                debug_dir=debug_dir,
                view=view,
                point_xy=point_xy,
                client_image=image_rgb,
                payload=payload,
            )

        answer = _show_overlay_preview(overlay, window=f"SAM3 overlay preview: {view}")
        if answer == "y":
            return
        if answer == "q":
            raise RuntimeError("Mask overlay preview rejected.")
        point_xy = None


def _capture_mask_view_image(caps, *, view: str, is_dual: bool, single_arm: str | None) -> np.ndarray:
    if is_dual:
        if view not in caps:
            raise KeyError(f"mask view {view!r} 没有对应摄像头；available={sorted(caps)}")
        return image_tools.convert_to_uint8(_capture_resized(caps[view]))
    if single_arm is None:
        raise ValueError("single-arm mask tracking requires single_arm")
    return image_tools.convert_to_uint8(_capture_resized(caps[single_arm]))


def _send_mask_track_only(
    policy_client,
    *,
    image_rgb: np.ndarray,
    view: str,
    text_prompt: str | None,
    alpha: float,
    debug_dir: pathlib.Path | None,
    track_index: int,
) -> dict:
    mask_request = {
        "enabled": True,
        "view": view,
        "track_only": True,
        "alpha": float(alpha),
    }
    if text_prompt:
        mask_request["text"] = text_prompt
    if debug_dir is not None:
        mask_request["return_image"] = True
        mask_request["return_overlay"] = True

    resp = policy_client.infer(
        {
            "image": {view: image_tools.convert_to_uint8(image_rgb)},
            MASK_OVERLAY_KEY: mask_request,
        }
    )
    payload = resp.get(MASK_OVERLAY_KEY, {})
    if debug_dir is not None:
        _dump_mask_rollout_debug(
            debug_dir=debug_dir,
            view=view,
            infer_index=track_index,
            payload=payload,
        )
    return payload


def parse_pose_xyz_rpy_deg(value: str) -> tuple[list[float], list[float]]:
    parts = [float(v) for v in value.split(",")]
    if len(parts) != 6:
        raise ValueError(f"pose 需要 6 个值 x,y,z,roll,pitch,yaw，实际得到: {value}")
    x, y, z, roll, pitch, yaw = parts
    return [x, y, z], [roll, pitch, yaw]


def _arm_index(arm_name: str) -> int:
    if arm_name == "robot_0":
        return 0
    if arm_name == "robot_1":
        return 1
    raise ValueError(f"Unsupported arm name: {arm_name}")


def _select_by_arm(arm_name: str, robot_0_value, robot_1_value):
    return robot_0_value if _arm_index(arm_name) == 0 else robot_1_value


def _read_xarm_pose(bestman: Bestman_Real_Xarm6) -> tuple[list[float], list[float], np.ndarray]:
    """读取 xArm 位姿；位置返回米，RPY 返回度，quat 返回 xyzw。"""
    bestman.robot.set_state(0)
    code, qpos_raw = bestman.robot.get_position()
    if code != 0:
        print(f"[WARN] xArm get_position return code={code}")
    x_mm, y_mm, z_mm, roll, pitch, yaw = qpos_raw
    pos_m = (np.asarray([x_mm, y_mm, z_mm], dtype=np.float64) / 1000.0).tolist()
    rpy_deg = [float(roll), float(pitch), float(yaw)]
    quat_xyzw = R.from_euler("xyz", rpy_deg, degrees=True).as_quat()
    return pos_m, rpy_deg, quat_xyzw


def _read_xarm_gripper_open(bestman: Bestman_Real_Xarm6) -> float:
    """读取 Robotiq 夹爪，转换为策略使用的 open=1 / close=0。"""
    width_raw = bestman.get_gripper_position_robotiq()  # 255~0
    return float(np.clip(1.0 - width_raw / 255.0, 0.0, 1.0))


def _gripper_open_to_robotiq_cmd(g_open: float, *, arm_index: int) -> int:
    """保持老 xArm 版本的 Robotiq open->cmd 标定。"""
    cmd = int((1.0 - float(g_open)) * 255)
    if arm_index == 0 and cmd > 180:
        cmd += 70
    if arm_index == 1 and cmd > 170:
        cmd += 30
    return int(np.clip(cmd, 0, 255))


def _set_xarm_pose(
    bestman: Bestman_Real_Xarm6,
    pos_m,
    rpy_deg,
    *,
    speed: float = 1333,
    mvacc: float = 66666,
    wait: bool = False,
) -> None:
    """保持老 xArm 版本的 set_position 调用；位置米转毫米，RPY 按度下发。"""
    x_m, y_m, z_m = [float(v) for v in pos_m]
    roll, pitch, yaw = [float(v) for v in rpy_deg]
    bestman.robot.set_state(0)
    bestman.robot.set_position(
        x_m * 1000.0,
        y_m * 1000.0,
        z_m * 1000.0,
        roll,
        pitch,
        yaw,
        speed=speed,
        mvacc=mvacc,
        wait=wait,
    )


def _set_xarm_gripper(
    bestman: Bestman_Real_Xarm6,
    g_open: float,
    *,
    arm_index: int,
    wait: bool = False,
) -> None:
    cmd = _gripper_open_to_robotiq_cmd(g_open, arm_index=arm_index)
    bestman.gripper_goto_robotiq(cmd, wait=wait, wait_motion=False)


def reset_arms_to_init(
    bestman0: Bestman_Real_Xarm6,
    bestman1: Bestman_Real_Xarm6,
    init_pos0,
    init_rpy_deg0,
    init_pos1,
    init_rpy_deg1,
) -> None:
    _set_xarm_pose(bestman0, init_pos0, init_rpy_deg0, wait=False)
    _set_xarm_pose(bestman1, init_pos1, init_rpy_deg1, wait=False)
    time.sleep(2.0)
    _set_xarm_gripper(bestman0, 1.0, arm_index=0, wait=False)
    _set_xarm_gripper(bestman1, 1.0, arm_index=1, wait=False)


def reset_arm_to_init(
    bestman: Bestman_Real_Xarm6,
    init_pos,
    init_rpy_deg,
    *,
    arm_index: int,
) -> None:
    _set_xarm_pose(bestman, init_pos, init_rpy_deg, wait=False)
    time.sleep(2.0)
    _set_xarm_gripper(bestman, 1.0, arm_index=arm_index, wait=False)


def interp_and_move_one(
    bestman: Bestman_Real_Xarm6,
    start_pos,
    start_rpy_deg,
    target_pos,
    target_rpy_deg,
    g_open: float,
    *,
    arm_index: int,
    step_size: float,
    dt: float,
) -> None:
    """线性位置插值 + 线性 RPY 插值，下发单臂（RPY 单位为度）。"""
    start_pos_arr = np.asarray(start_pos, dtype=float)
    target_pos_arr = np.asarray(target_pos, dtype=float)
    start_rpy_arr = np.asarray(start_rpy_deg, dtype=float)
    target_rpy_arr = np.asarray(target_rpy_deg, dtype=float)

    if step_size <= 0:
        steps = 1
    else:
        distance = np.linalg.norm(target_pos_arr - start_pos_arr)
        steps = max(int(np.ceil(distance / max(step_size, 1e-6))), 1)

    interp_ts = np.linspace(0.0, 1.0, steps + 1)
    pos_arr = np.linspace(start_pos_arr, target_pos_arr, steps + 1)
    rpy_arr = np.outer(1.0 - interp_ts, start_rpy_arr) + np.outer(interp_ts, target_rpy_arr)

    for idx in range(steps + 1):
        _set_xarm_pose(bestman, pos_arr[idx], rpy_arr[idx], wait=False)
        time.sleep(dt)

    _set_xarm_gripper(bestman, g_open, arm_index=arm_index, wait=False)


def interp_and_move_both(
    bestman0: Bestman_Real_Xarm6,
    bestman1: Bestman_Real_Xarm6,
    start_pos0,
    start_rpy_deg0,
    target_pos0,
    target_rpy_deg0,
    start_pos1,
    start_rpy_deg1,
    target_pos1,
    target_rpy_deg1,
    g0: float,
    g1: float,
    step_size: float,
    dt: float,
) -> None:
    """线性位置插值 + 线性 RPY 插值，同步下发双臂（RPY 单位为度）。"""
    p0s = np.asarray(start_pos0, dtype=float)
    p0e = np.asarray(target_pos0, dtype=float)
    r0s = np.asarray(start_rpy_deg0, dtype=float)
    r0e = np.asarray(target_rpy_deg0, dtype=float)
    p1s = np.asarray(start_pos1, dtype=float)
    p1e = np.asarray(target_pos1, dtype=float)
    r1s = np.asarray(start_rpy_deg1, dtype=float)
    r1e = np.asarray(target_rpy_deg1, dtype=float)

    if step_size <= 0:
        steps = 1
    else:
        dist0 = np.linalg.norm(p0e - p0s)
        dist1 = np.linalg.norm(p1e - p1s)
        steps = max(int(np.ceil(max(dist0, dist1) / max(step_size, 1e-6))), 1)

    ts = np.linspace(0.0, 1.0, steps + 1)
    pos0_arr = np.linspace(p0s, p0e, steps + 1)
    pos1_arr = np.linspace(p1s, p1e, steps + 1)
    rpy0_arr = np.outer(1.0 - ts, r0s) + np.outer(ts, r0e)
    rpy1_arr = np.outer(1.0 - ts, r1s) + np.outer(ts, r1e)

    for idx in range(steps + 1):
        _set_xarm_pose(bestman0, pos0_arr[idx], rpy0_arr[idx], wait=False)
        _set_xarm_pose(bestman1, pos1_arr[idx], rpy1_arr[idx], wait=False)
        time.sleep(dt)

    _set_xarm_gripper(bestman0, g0, arm_index=0, wait=False)
    _set_xarm_gripper(bestman1, g1, arm_index=1, wait=False)


def _slice_actions(actions_all: np.ndarray, start: int, end: int) -> tuple[int, np.ndarray] | None:
    n_act = len(actions_all)
    if start < 0 or end < 0:
        print(f"[WARN] 忽略非法 action 区间: action_start={start} action_end={end}")
        return None
    if start > end:
        print(f"[WARN] action_start 不能大于 action_end: {start} > {end}")
        return None
    if n_act == 0:
        print("[WARN] 策略返回的 action 序列为空")
        return None
    lo = start
    hi = min(end, n_act - 1)
    if lo >= n_act:
        print(f"[WARN] action_start 越界: start={start} 序列长度={n_act}")
        return None
    if lo > hi:
        print(f"[WARN] 无 action 可执行: 区间 [{start},{end}] 与长度 {n_act} 无交集")
        return None
    return lo, actions_all[lo : hi + 1]


def _capture_resized(cap) -> np.ndarray:
    return cv2.resize(grab_rgb_latest(cap), (224, 224), interpolation=cv2.INTER_AREA)


def _adjust_gripper_actions(actions_all: np.ndarray, *, arm_mode: str, single_arm_index: int) -> np.ndarray:
    if arm_mode == "single":
        if actions_all.shape[-1] < 7:
            raise ValueError(f"single mode expects action dim >= 7, got shape={actions_all.shape}")
        threshold = 0.8 if single_arm_index == 0 else 0.95
        actions_all[..., 6] = np.where(actions_all[..., 6] < threshold, actions_all[..., 6] - 0.2, actions_all[..., 6])
        actions_all[..., 6] = np.clip(actions_all[..., 6], 0.0, 1.0)
        return actions_all

    if actions_all.shape[-1] < 14:
        raise ValueError(f"dual mode expects action dim >= 14, got shape={actions_all.shape}")
    actions_all[..., 6] = np.where(actions_all[..., 6] < 0.8, actions_all[..., 6] - 0.2, actions_all[..., 6])
    actions_all[..., 6] = np.clip(actions_all[..., 6], 0.0, 1.0)
    actions_all[..., 13] = np.where(actions_all[..., 13] < 0.95, actions_all[..., 13] - 0.2, actions_all[..., 13])
    actions_all[..., 13] = np.clip(actions_all[..., 13], 0.0, 1.0)
    return actions_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--description", required=True, help="任务自然语言指令")
    parser.add_argument("--arm_mode", choices=("dual", "single"), default="dual", help="dual=双臂 14D；single=单臂 7D")
    parser.add_argument("--single_arm", choices=("robot_0", "robot_1"), default=None, help="single 模式必须显式指定使用哪只 xArm")
    parser.add_argument("--single_image_key", default="front", help="single 模式发给 policy 的 image key，FastUMI 单臂默认 front")
    parser.add_argument("--robot_ip", default="192.168.1.240", help="robot_0 对应 xArm 控制盒 IP")
    parser.add_argument("--robot_ip_2", default="192.168.1.240", help="robot_1 对应 xArm 控制盒 IP")
    parser.add_argument("--server_ip", default=SERVER_IP, help="policy server host")
    parser.add_argument("--port", type=int, default=PORT, help="policy server port")
    parser.add_argument("--camera_dev0", type=int, default=DEV, help="robot_0 图像对应的视频设备编号")
    parser.add_argument("--camera_dev1", type=int, default=DEV + 2, help="robot_1 图像对应的视频设备编号")
    parser.add_argument("--dt", type=float, default=0.025, help="启用 --enable_interp 时，插值每步的睡眠时间（秒）")
    parser.add_argument("--enable_interp", action="store_true", help="启用客户端线性插值；默认关闭，交给 xArm SDK 自身规划")
    parser.add_argument("--interp_step_size", type=float, default=0.0025, help="启用 --enable_interp 时，插值最大平移步长（米）")
    parser.add_argument("--init_pose_left", default="0.25,0.2,0.145,180,-90,0.0", help="robot_0 初始位姿: x,y,z,roll,pitch,yaw（米、度）")
    parser.add_argument("--init_pose_right", default="0.4,0.0,0.146,180,-90,0.0", help="robot_1 初始位姿: x,y,z,roll,pitch,yaw（米、度）")
    parser.add_argument("--skip_init_move", action="store_true", help="启动时不自动移动到 init_pose；s 复位仍使用 init_pose")
    parser.add_argument("--action_start", type=int, default=10, help="本次推理返回的 action 序列起始下标（含）")
    parser.add_argument("--action_end", type=int, default=30, help="本次推理返回的 action 序列结束下标（含）")
    parser.add_argument("--mask_overlay", action="store_true", help="请求 policy server 使用 SAM3 mask overlay 后再推理")
    parser.add_argument("--mask_view", default=None, help="需要做 SAM3 overlay 的 image key；single 默认 front，dual 下建议显式指定 robot_0/robot_1")
    parser.add_argument("--mask_prompt_point", default=None, help="初始正点 prompt，格式 x,y；不填则弹窗点击")
    parser.add_argument("--mask_prompt_text", default=None, help="SAM3 text prompt，例如 sponge；text_select_video 模式必填")
    parser.add_argument("--mask_alpha", type=float, default=0.35, help="SAM3 overlay alpha")
    parser.add_argument("--mask_debug_dir", default=None, help="显式提供目录时，保存 SAM3 preview 输入/输出调试图并请求 server 回传原图")
    parser.add_argument("--mask_track_between_actions", action="store_true", help="每执行若干 action 后额外发送一帧给 SAM3 追踪，只更新 mask 不跑 policy")
    parser.add_argument("--mask_track_every_n_actions", type=int, default=1, help="开启 --mask_track_between_actions 后，每 N 个 action 发送一次 tracking-only 图像")
    args = parser.parse_args()

    is_dual = args.arm_mode == "dual"
    if not is_dual and args.single_arm is None:
        parser.error("--arm_mode single 需要显式指定 --single_arm robot_0 或 robot_1，避免误用默认 robot_0")
    if args.mask_overlay and is_dual and args.mask_view is None:
        parser.error("--arm_mode dual 开启 --mask_overlay 时需要显式指定 --mask_view robot_0 或 robot_1")
    if args.mask_track_between_actions and not args.mask_overlay:
        parser.error("--mask_track_between_actions 需要同时开启 --mask_overlay")
    if args.mask_track_every_n_actions < 1:
        parser.error("--mask_track_every_n_actions 必须 >= 1")

    single_arm = args.single_arm
    single_arm_index = _arm_index(single_arm) if single_arm is not None else 0
    effective_mask_view = args.mask_view or args.single_image_key

    init_pos0, init_rpy0 = parse_pose_xyz_rpy_deg(args.init_pose_left)
    init_pos1, init_rpy1 = parse_pose_xyz_rpy_deg(args.init_pose_right)
    init_by_arm = {
        "robot_0": (init_pos0, init_rpy0),
        "robot_1": (init_pos1, init_rpy1),
    }

    arms: dict[str, Bestman_Real_Xarm6] = {}
    if is_dual or single_arm == "robot_0":
        arms["robot_0"] = Bestman_Real_Xarm6(args.robot_ip, None, None)
    if is_dual or single_arm == "robot_1":
        arms["robot_1"] = Bestman_Real_Xarm6(args.robot_ip_2, None, None)

    def reset_active_arms() -> None:
        if is_dual:
            reset_arms_to_init(arms["robot_0"], arms["robot_1"], init_pos0, init_rpy0, init_pos1, init_rpy1)
            return
        assert single_arm is not None
        init_pos, init_rpy = init_by_arm[single_arm]
        reset_arm_to_init(arms[single_arm], init_pos, init_rpy, arm_index=single_arm_index)

    if not args.skip_init_move:
        if is_dual:
            print(f"[INFO] 移动到初始位姿: robot_0={args.init_pose_left} robot_1={args.init_pose_right}")
        else:
            print(f"[INFO] 移动到初始位姿: {single_arm}={_select_by_arm(single_arm, args.init_pose_left, args.init_pose_right)}")
        reset_active_arms()
        print("[INFO] 已到达初始位姿")

    policy_client = websocket_client_policy.WebsocketClientPolicy(
        host=args.server_ip,
        port=args.port,
    )
    print(f"[INFO] 已连接策略服务器：ws://{args.server_ip}:{args.port}")
    if args.mask_overlay:
        metadata = policy_client.get_server_metadata()
        if not metadata.get("mask_overlay", {}).get("enabled", False):
            raise RuntimeError("client 开启了 --mask_overlay，但 server 未启用 --mask-overlay")
        if metadata.get("mask_overlay", {}).get("tracking_mode") == "text_select_video" and not args.mask_prompt_text:
            raise RuntimeError("server 使用 text_select_video，client 需要提供 --mask_prompt_text")

    camera_devs = {
        "robot_0": args.camera_dev0,
        "robot_1": args.camera_dev1,
    }
    caps = {}
    if is_dual:
        caps["robot_0"] = init_yu12_camera(camera_devs["robot_0"])
        caps["robot_1"] = init_yu12_camera(camera_devs["robot_1"])
    else:
        assert single_arm is not None
        caps[single_arm] = init_yu12_camera(camera_devs[single_arm])

    for _ in range(50):
        for cap in caps.values():
            _ = cap.read()
    print("[INFO] 摄像头预热完成，开始循环")

    input("按 Enter 开始")
    mask_debug_dir = pathlib.Path(args.mask_debug_dir) if args.mask_debug_dir else None
    if args.mask_overlay:
        if is_dual:
            preview_images = {
                "robot_0": _capture_resized(caps["robot_0"]),
                "robot_1": _capture_resized(caps["robot_1"]),
            }
        else:
            assert single_arm is not None
            preview_images = {args.single_image_key: _capture_resized(caps[single_arm])}
        if effective_mask_view not in preview_images:
            raise ValueError(f"--mask_view must be one of {sorted(preview_images)}, got {effective_mask_view}")
        _preview_mask_overlay(
            policy_client,
            image_rgb=preview_images[effective_mask_view],
            view=effective_mask_view,
            point_xy=_parse_xy(args.mask_prompt_point),
            text_prompt=args.mask_prompt_text,
            alpha=args.mask_alpha,
            debug_dir=mask_debug_dir,
        )
        print("[INFO] SAM3 mask overlay 已确认，后续推理会持续请求 overlay")
    mask_track_index = 0

    stdin_fd = sys.stdin.fileno()
    old_term = None
    try:
        if sys.stdin.isatty():
            old_term = termios.tcgetattr(stdin_fd)
            tty.setcbreak(stdin_fd)
        print("[INFO] 按 s：复位到初始位姿并暂停推理；按 c：继续推理")

        paused = False
        infer_index = 0
        while True:
            ch = _stdin_read_char_nonblocking()
            if ch in ("s", "S"):
                reset_active_arms()
                paused = True
                print("[INFO] 已复位到初始位姿，推理已暂停；按 c 继续")
                continue
            if ch in ("c", "C"):
                paused = False
                print("[INFO] 继续推理")

            if paused:
                time.sleep(0.02)
                continue

            if is_dual:
                pos0, rpy_deg0, quat0 = _read_xarm_pose(arms["robot_0"])
                pos1, rpy_deg1, quat1 = _read_xarm_pose(arms["robot_1"])
                gripper_open0 = _read_xarm_gripper_open(arms["robot_0"])
                gripper_open1 = _read_xarm_gripper_open(arms["robot_1"])

                state_vec = np.array(
                    [
                        *pos0,
                        *quat0,
                        gripper_open0,
                        *pos1,
                        *quat1,
                        gripper_open1,
                    ],
                    dtype=np.float32,
                )
                cur_pos0, cur_rpy0 = list(pos0), list(rpy_deg0)
                cur_pos1, cur_rpy1 = list(pos1), list(rpy_deg1)
                image_obs = {
                    "robot_0": image_tools.convert_to_uint8(_capture_resized(caps["robot_0"])),
                    "robot_1": image_tools.convert_to_uint8(_capture_resized(caps["robot_1"])),
                }
            else:
                assert single_arm is not None
                pos_single, rpy_single, quat_single = _read_xarm_pose(arms[single_arm])
                gripper_open_single = _read_xarm_gripper_open(arms[single_arm])
                state_vec = np.array([*pos_single, *quat_single, gripper_open_single], dtype=np.float32)
                cur_pos_single, cur_rpy_single = list(pos_single), list(rpy_single)
                image_obs = {
                    args.single_image_key: image_tools.convert_to_uint8(_capture_resized(caps[single_arm])),
                }

            print("[INFO] 机器人状态：", state_vec)
            obs = {
                "state": state_vec,
                "image": image_obs,
                "prompt": args.description,
            }
            if args.mask_overlay:
                obs[MASK_OVERLAY_KEY] = {
                    "enabled": True,
                    "view": effective_mask_view,
                    "alpha": args.mask_alpha,
                }
                if args.mask_prompt_text:
                    obs[MASK_OVERLAY_KEY]["text"] = args.mask_prompt_text
                if mask_debug_dir is not None:
                    obs[MASK_OVERLAY_KEY]["return_image"] = True
                    obs[MASK_OVERLAY_KEY]["return_overlay"] = True

            resp = policy_client.infer(obs)
            mask_payload = resp.get(MASK_OVERLAY_KEY, {}) if args.mask_overlay else {}
            if args.mask_overlay and mask_debug_dir is not None:
                _dump_mask_rollout_debug(
                    debug_dir=mask_debug_dir,
                    view=effective_mask_view,
                    infer_index=infer_index,
                    payload=mask_payload,
                )
            if args.mask_overlay and mask_payload.get("needs_reprompt"):
                print(f"[MASK] tracking 丢失，重新打点: {mask_payload.get('reprompt_reason')}")
                _preview_mask_overlay(
                    policy_client,
                    image_rgb=np.asarray(image_obs[effective_mask_view]),
                    view=effective_mask_view,
                    point_xy=None,
                    text_prompt=args.mask_prompt_text,
                    alpha=args.mask_alpha,
                    debug_dir=mask_debug_dir,
                )
                print("[INFO] SAM3 mask overlay 已重新初始化，继续推理")
                infer_index += 1
                continue
            infer_index += 1
            actions_all = resp["actions"] if "actions" in resp else resp["action"]
            actions_all = np.array(actions_all, dtype=np.float64, copy=True)
            if actions_all.ndim == 1:
                actions_all = actions_all.reshape(1, -1)

            actions_all = _adjust_gripper_actions(actions_all, arm_mode=args.arm_mode, single_arm_index=single_arm_index)

            action_slice_result = _slice_actions(actions_all, args.action_start, args.action_end)
            if action_slice_result is None:
                continue
            lo, action_slice = action_slice_result

            for step_idx, action in enumerate(action_slice, start=lo):
                if _stdin_wants_reset_pause():
                    reset_active_arms()
                    paused = True
                    print("[INFO] 已复位到初始位姿，推理已暂停；按 c 继续")
                    break

                print(f"第 {step_idx} 步, action:", action)
                if is_dual:
                    x_a0, y_a0, z_a0, roll0, pitch0, yaw0, g_open0, x_a1, y_a1, z_a1, roll1, pitch1, yaw1, g_open1 = action

                    target_pos0 = [x_a0, y_a0, z_a0]
                    target_pos1 = [x_a1, y_a1, z_a1]
                    target_rpy0 = [roll0, pitch0, yaw0]
                    target_rpy1 = [roll1, pitch1, yaw1]
                    g0 = float(np.clip(g_open0, 0.0, 1.0))
                    g1 = float(np.clip(g_open1, 0.0, 1.0))

                    if args.enable_interp:
                        interp_and_move_both(
                            arms["robot_0"],
                            arms["robot_1"],
                            cur_pos0,
                            cur_rpy0,
                            target_pos0,
                            target_rpy0,
                            cur_pos1,
                            cur_rpy1,
                            target_pos1,
                            target_rpy1,
                            g0,
                            g1,
                            step_size=args.interp_step_size,
                            dt=args.dt,
                        )
                    else:
                        _set_xarm_pose(arms["robot_0"], target_pos0, target_rpy0, wait=False)
                        _set_xarm_pose(arms["robot_1"], target_pos1, target_rpy1, wait=False)
                        _set_xarm_gripper(arms["robot_0"], g0, arm_index=0, wait=False)
                        _set_xarm_gripper(arms["robot_1"], g1, arm_index=1, wait=False)
                    cur_pos0, cur_rpy0 = target_pos0, target_rpy0
                    cur_pos1, cur_rpy1 = target_pos1, target_rpy1
                else:
                    assert single_arm is not None
                    x_a, y_a, z_a, roll, pitch, yaw, g_open = action[:7]
                    target_pos = [x_a, y_a, z_a]
                    target_rpy = [roll, pitch, yaw]
                    gripper_open = float(np.clip(g_open, 0.0, 1.0))
                    if args.enable_interp:
                        interp_and_move_one(
                            arms[single_arm],
                            cur_pos_single,
                            cur_rpy_single,
                            target_pos,
                            target_rpy,
                            gripper_open,
                            arm_index=single_arm_index,
                            step_size=args.interp_step_size,
                            dt=args.dt,
                        )
                    else:
                        _set_xarm_pose(arms[single_arm], target_pos, target_rpy, wait=False)
                        _set_xarm_gripper(arms[single_arm], gripper_open, arm_index=single_arm_index, wait=False)
                    cur_pos_single, cur_rpy_single = target_pos, target_rpy

                if (
                    args.mask_overlay
                    and args.mask_track_between_actions
                    and ((step_idx - lo + 1) % args.mask_track_every_n_actions == 0)
                ):
                    track_image = _capture_mask_view_image(
                        caps,
                        view=effective_mask_view,
                        is_dual=is_dual,
                        single_arm=single_arm,
                    )
                    track_payload = _send_mask_track_only(
                        policy_client,
                        image_rgb=track_image,
                        view=effective_mask_view,
                        text_prompt=args.mask_prompt_text,
                        alpha=args.mask_alpha,
                        debug_dir=mask_debug_dir,
                        track_index=1_000_000 + mask_track_index,
                    )
                    mask_track_index += 1
                    if track_payload.get("needs_reprompt"):
                        print(f"[MASK] dense tracking 丢失，重新打点: {track_payload.get('reprompt_reason')}")
                        _preview_mask_overlay(
                            policy_client,
                            image_rgb=track_image,
                            view=effective_mask_view,
                            point_xy=None,
                            text_prompt=args.mask_prompt_text,
                            alpha=args.mask_alpha,
                            debug_dir=mask_debug_dir,
                        )
                        print("[INFO] SAM3 mask overlay 已重新初始化，继续执行当前 rollout")

    finally:
        if old_term is not None:
            termios.tcsetattr(stdin_fd, termios.TCSADRAIN, old_term)
        for cap in caps.values():
            cap.release()
        cv2.destroyAllWindows()
        print("[INFO] 结束，摄像头已释放。")


if __name__ == "__main__":
    main()
