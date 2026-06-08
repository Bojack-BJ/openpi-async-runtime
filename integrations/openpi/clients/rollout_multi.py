#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Async RTC-style xArm rollout with front RGB, depth, and bottom-center crops.

This script fuses the single-arm depth/crop rollout with the async rollout
action queue.  It keeps the policy observation schema used by the depth-crop
checkpoint:

    obs["image"]["front"]              HxWx3 uint8 RGB
    obs["image"]["front_crop"]         HxWx3 uint8 RGB
    obs["image"]["front_depth"]        HxWx1 uint8 depth
    obs["image"]["front_depth_crop"]   HxWx1 uint8 depth crop
    obs["state"]                       [x,y,z,qx,qy,qz,qw,gripper_open]

The control side is single-arm xArm Cartesian servo.  Policy inference runs in
its own thread and merges returned action chunks into a fixed-rate ActionBuffer.
RTC server-side chunk conditioning can be enabled with --rtc_chunk_conditioning
when serving with scripts/serve_policy_multi.py --rtc-chunk-conditioning.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from queue import Empty
from queue import Queue
import select
import sys
import termios
import threading
import time
import traceback
import tty
import uuid

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from xarm.wrapper import XArmAPI

from async_rollout_core import ActionBuffer
from async_rollout_core import AsyncDebugWriter
from async_rollout_core import ExecutedAction
from async_rollout_core import LatencyEstimator
from async_rollout_core import TimedAction
from async_rollout_core import TimedObservation
from async_rollout_core import action_command_delta
from async_rollout_core import action_tracking_error
from async_rollout_core import limit_action_step
from async_rollout_core import should_advance_control_step
from openpi_client import image_tools
from openpi_client import websocket_client_policy


SERVER_IP = "180.184.74.93"
PORT = 8002
DEV = 0
W, H, FPS = 1280, 1280, 100
IMAGE_OUTPUT_SIZE = 224
BOTTOM_CENTER_CROP_RATIO = 0.50
PREVIEW_WINDOW_NAME = "rollout_multi: front | crop | depth"
CONTROL_KEY_QUEUE: Queue[str] = Queue()
PREVIEW_KEY_PUMP_ACTIVE = threading.Event()


def _resolve_default_da2_repo() -> Path:
    env_repo = os.environ.get("DEPTH_ANYTHING_V2_ROOT", "").strip()
    if env_repo:
        return Path(env_repo)
    return Path(__file__).resolve().parent / "Depth-Anything-V2"


def _resolve_da2_device(device_arg: str):
    import torch

    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--da2-device=cuda specified but CUDA is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_depth_anything2_model(da2_repo: Path, encoder: str, device):
    da2_repo = Path(da2_repo).resolve()
    if da2_repo.exists() and str(da2_repo) not in sys.path:
        sys.path.insert(0, str(da2_repo))
    try:
        from depth_anything_v2.dpt import DepthAnythingV2
    except Exception as exc:
        raise RuntimeError(f"Failed to import depth_anything_v2 from --da2-repo={da2_repo}") from exc

    model_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
    }
    if encoder not in model_configs:
        raise ValueError(f"unknown --da2-encoder={encoder}")
    ckpt = da2_repo / "checkpoints" / f"depth_anything_v2_{encoder}.pth"
    if not ckpt.is_file():
        raise FileNotFoundError(f"DA2 checkpoint not found: {ckpt}")

    import torch

    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch.load(str(ckpt), map_location="cpu"))
    return model.to(device).eval()


def normalize_depth_to_uint8(depth, *, invert: bool = False, d_min=None, d_max=None) -> np.ndarray:
    if d_min is None:
        d_min = float(np.min(depth))
    if d_max is None:
        d_max = float(np.max(depth))
    if d_max <= d_min + 1e-8:
        out = np.zeros_like(depth, dtype=np.uint8)
    else:
        out = ((depth - d_min) / (d_max - d_min) * 255.0).clip(0, 255).astype(np.uint8)
    if invert:
        out = 255 - out
    return out


class DepthAnything2Runtime:
    def __init__(
        self,
        *,
        da2_repo: Path,
        encoder: str,
        device_arg: str,
        input_size: int,
        fp16: bool,
        norm_mode: str,
        fixed_min: float | None,
        fixed_max: float | None,
        percentile_min: float,
        percentile_max: float,
        invert: bool,
        depth_scale: float,
        depth_shift: float,
    ) -> None:
        import torch
        from torchvision.transforms import Compose

        self.torch = torch
        self.device = _resolve_da2_device(device_arg)
        self.model = build_depth_anything2_model(da2_repo, encoder, self.device)
        from depth_anything_v2.util.transform import NormalizeImage
        from depth_anything_v2.util.transform import PrepareForNet
        from depth_anything_v2.util.transform import Resize

        self.fp16 = bool(fp16)
        self.norm_mode = norm_mode
        self.fixed_min = fixed_min
        self.fixed_max = fixed_max
        self.percentile_min = percentile_min
        self.percentile_max = percentile_max
        self.invert = bool(invert)
        self.depth_scale = float(depth_scale)
        self.depth_shift = float(depth_shift)
        self.transform = Compose(
            [
                Resize(
                    width=input_size,
                    height=input_size,
                    resize_target=False,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=14,
                    resize_method="lower_bound",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ]
        )

    def infer_depth_1c(self, frame_rgb: np.ndarray) -> np.ndarray:
        import torch.nn.functional as F

        h, w = frame_rgb.shape[:2]
        image = frame_rgb.astype(np.float32) / 255.0
        x_np = self.transform({"image": image})["image"]
        x = self.torch.from_numpy(x_np[None]).float().to(self.device, non_blocking=True)
        with self.torch.inference_mode():
            if self.fp16 and self.device.type == "cuda":
                with self.torch.autocast(device_type="cuda", dtype=self.torch.float16):
                    depth = self.model(x)
            else:
                depth = self.model(x)
        depth = F.interpolate(depth[:, None], size=(h, w), mode="bilinear", align_corners=True)[0, 0]
        depth = depth.float().cpu().numpy() * self.depth_scale + self.depth_shift
        if self.norm_mode == "frame":
            depth_u8 = normalize_depth_to_uint8(depth, invert=self.invert)
        elif self.norm_mode == "fixed":
            if self.fixed_min is None or self.fixed_max is None:
                raise ValueError("--da2-norm-mode=fixed requires --da2-fixed-min and --da2-fixed-max")
            depth_u8 = normalize_depth_to_uint8(depth, invert=self.invert, d_min=self.fixed_min, d_max=self.fixed_max)
        elif self.norm_mode == "percentile":
            p_min, p_max = np.percentile(depth, [self.percentile_min, self.percentile_max])
            depth_u8 = normalize_depth_to_uint8(depth, invert=self.invert, d_min=float(p_min), d_max=float(p_max))
        else:
            raise ValueError(f"Unsupported --da2-norm-mode={self.norm_mode}")
        return depth_u8[..., None]


def init_yu12_camera(dev: int):
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开 /dev/video{dev}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YU12"))
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


def read_yu12_rgb_frame(cap) -> np.ndarray:
    ok, raw = cap.read()
    if not ok:
        raise RuntimeError("摄像头读取失败")
    yuv = np.ascontiguousarray(raw).reshape(H * 3 // 2, W)
    bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def bottom_center_crop_resize_rgb(
    frame_rgb: np.ndarray,
    *,
    out_hw: int = IMAGE_OUTPUT_SIZE,
    crop_ratio: float = BOTTOM_CENTER_CROP_RATIO,
) -> np.ndarray:
    h, w = frame_rgb.shape[:2]
    crop_h = max(1, int(round(h * crop_ratio)))
    crop_w = max(1, int(round(w * crop_ratio)))
    y0 = h - crop_h
    x0 = (w - crop_w) // 2
    crop = frame_rgb[y0:h, x0 : x0 + crop_w]
    return cv2.resize(crop, (out_hw, out_hw), interpolation=cv2.INTER_AREA).astype(np.uint8)


def bottom_center_crop_resize_depth(
    depth_1c: np.ndarray,
    *,
    out_hw: int = IMAGE_OUTPUT_SIZE,
    crop_ratio: float = BOTTOM_CENTER_CROP_RATIO,
) -> np.ndarray:
    depth = np.asarray(depth_1c)
    if depth.ndim == 3:
        depth = depth[..., 0]
    h, w = depth.shape[:2]
    crop_h = max(1, int(round(h * crop_ratio)))
    crop_w = max(1, int(round(w * crop_ratio)))
    y0 = h - crop_h
    x0 = (w - crop_w) // 2
    crop = depth[y0:h, x0 : x0 + crop_w]
    return cv2.resize(crop, (out_hw, out_hw), interpolation=cv2.INTER_AREA)[..., None].astype(np.uint8)


def _control_key_from_code(key_code: int) -> str | None:
    key = int(key_code) & 0xFF
    if key in (ord("s"), ord("S")):
        return "s"
    if key in (ord("c"), ord("C")):
        return "c"
    if key in (ord("q"), ord("Q"), 27):
        return "q"
    return None


def _queue_control_key_from_code(key_code: int) -> None:
    key = _control_key_from_code(key_code)
    if key is not None:
        CONTROL_KEY_QUEUE.put(key)


def _stdin_read_char_nonblocking() -> str | None:
    if not sys.stdin.isatty():
        return None
    readable, _, _ = select.select([sys.stdin], [], [], 0)
    if not readable:
        return None
    try:
        return sys.stdin.read(1)
    except Exception:
        return None


def read_control_key(wait_ms: int = 1) -> str | None:
    try:
        return CONTROL_KEY_QUEUE.get_nowait()
    except Empty:
        pass
    if PREVIEW_KEY_PUMP_ACTIVE.is_set():
        if wait_ms > 0:
            time.sleep(wait_ms / 1000.0)
    else:
        key = _control_key_from_code(cv2.waitKey(wait_ms) & 0xFF)
        if key is not None:
            return key
    ch = _stdin_read_char_nonblocking()
    if ch in ("s", "S"):
        return "s"
    if ch in ("c", "C"):
        return "c"
    if ch in ("q", "Q", "\x1b"):
        return "q"
    return None


def _rgb_to_bgr_preview(image_rgb):
    image = image_tools.convert_to_uint8(np.asarray(image_rgb))
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[-1] == 1:
        return cv2.cvtColor(image[..., 0], cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def _put_preview_label(image_bgr, label: str):
    out = image_bgr.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 24), (0, 0, 0), thickness=-1)
    cv2.putText(out, label, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def show_rollout_preview(front_rgb, front_crop_rgb, front_depth_1c, *, enabled: bool = True) -> None:
    if not enabled:
        return
    front = _rgb_to_bgr_preview(front_rgb)
    crop = _rgb_to_bgr_preview(front_crop_rgb)
    depth = _rgb_to_bgr_preview(front_depth_1c)
    h, w = front.shape[:2]
    crop_h = max(1, int(round(h * BOTTOM_CENTER_CROP_RATIO)))
    crop_w = max(1, int(round(w * BOTTOM_CENTER_CROP_RATIO)))
    y0 = h - crop_h
    x0 = (w - crop_w) // 2
    front_box = front.copy()
    cv2.rectangle(front_box, (x0, y0), (x0 + crop_w - 1, h - 1), (0, 255, 0), 2)
    preview = np.hstack(
        [
            _put_preview_label(front_box, "front"),
            _put_preview_label(crop, "front_crop"),
            _put_preview_label(depth, "front_depth"),
        ]
    )
    cv2.imshow(PREVIEW_WINDOW_NAME, preview)


class RealtimeCameraPreview:
    def __init__(self, cam, *, visual_enabled: bool = True) -> None:
        self.cam = cam
        self.visual_enabled = visual_enabled
        self._lock = threading.Condition()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="camera-preview", daemon=True)
        self._latest_rgb_raw = None
        self._latest_front = None
        self._latest_crop = None
        self._latest_depth = np.zeros((IMAGE_OUTPUT_SIZE, IMAGE_OUTPUT_SIZE, 1), dtype=np.uint8)
        self._seq = 0
        self._timestamp = 0.0
        self._error = None

    def start(self) -> None:
        if self.visual_enabled:
            PREVIEW_KEY_PUMP_ACTIVE.set()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        if self.visual_enabled:
            PREVIEW_KEY_PUMP_ACTIVE.clear()

    def update_depth(self, depth_1c) -> None:
        depth = image_tools.convert_to_uint8(np.asarray(depth_1c))
        if depth.ndim == 2:
            depth = depth[..., None]
        with self._lock:
            self._latest_depth = depth.copy()

    def frame_seq(self) -> int:
        with self._lock:
            return self._seq

    def wait_for_new_frame(self, *, after_seq: int | None = None, timeout: float = 1.0):
        deadline = time.monotonic() + float(timeout)
        with self._lock:
            if after_seq is None:
                after_seq = self._seq
            while self._seq <= after_seq and self._error is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._lock.wait(timeout=remaining)
            if self._error is not None:
                raise RuntimeError(f"摄像头线程失败: {self._error}") from self._error
            if self._latest_rgb_raw is None:
                raise RuntimeError("摄像头线程尚未取得图像")
            if self._seq <= after_seq:
                raise TimeoutError("等待最新摄像头帧超时")
            return self._latest_rgb_raw.copy(), self._seq, time.monotonic() - self._timestamp

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                frame_rgb_raw = read_yu12_rgb_frame(self.cam)
                front = cv2.resize(frame_rgb_raw, (IMAGE_OUTPUT_SIZE, IMAGE_OUTPUT_SIZE), interpolation=cv2.INTER_AREA)
                crop = bottom_center_crop_resize_rgb(frame_rgb_raw)
                with self._lock:
                    self._latest_rgb_raw = frame_rgb_raw
                    self._latest_front = front
                    self._latest_crop = crop
                    self._seq += 1
                    self._timestamp = time.monotonic()
                    depth = self._latest_depth.copy()
                    self._lock.notify_all()
                if self.visual_enabled:
                    show_rollout_preview(front, crop, depth, enabled=True)
                    _queue_control_key_from_code(cv2.waitKey(1))
                else:
                    time.sleep(0.001)
        except Exception as exc:
            with self._lock:
                self._error = exc
                self._lock.notify_all()


def parse_pose_deg(value: str) -> list[float]:
    parts = [float(v) for v in value.split(",")]
    if len(parts) != 6:
        raise ValueError(f"pose 需要 6 个值 x,y,z,r,p,y，实际得到: {value}")
    x, y, z, roll, pitch, yaw = parts
    return [x * 1000.0, y * 1000.0, z * 1000.0, roll, pitch, yaw]


def check_xarm_code(arm, code, label: str, *, raise_on_error: bool = False):
    if isinstance(code, (tuple, list)) and len(code) >= 1:
        code = code[0]
    if code != 0:
        msg = f"[WARN] {label} code={code}, state={arm.state}, error={arm.error_code}, warn={arm.warn_code}"
        print(msg)
        if raise_on_error:
            raise RuntimeError(msg)
    return code


def set_robotiq_position(arm, pos: int, label: str, *, wait: bool = False, raise_on_error: bool = False):
    ret = arm.robotiq_set_position(pos, wait=wait, auto_enable=True)
    code = ret[0] if isinstance(ret, (tuple, list)) and len(ret) >= 1 else ret
    status = getattr(arm, "robotiq_status", {}) or {}
    if code != 0 and (code == 102 or status.get("gFLT") == 5):
        print(f"[INFO] {label} 检测到 Robotiq 未就绪，重新激活后重试")
        if hasattr(arm, "robotiq_reset"):
            check_xarm_code(arm, arm.robotiq_reset(), f"{label} robotiq_reset")
            time.sleep(0.1)
        if hasattr(arm, "robotiq_set_activate"):
            check_xarm_code(arm, arm.robotiq_set_activate(wait=True), f"{label} robotiq_set_activate")
            time.sleep(0.1)
        ret = arm.robotiq_set_position(pos, wait=wait, auto_enable=True)
    return check_xarm_code(arm, ret, label, raise_on_error=raise_on_error)


def init_xarm(robot_ip: str, init_pose: list[float], *, gripper_open: bool = True):
    arm = XArmAPI(robot_ip)
    time.sleep(0.5)
    arm.clean_warn()
    arm.clean_error()
    arm.motion_enable(enable=True)
    arm.set_mode(0)
    arm.set_state(0)
    time.sleep(0.2)
    print(f"[INFO] 移动到初始位姿: {init_pose}")
    check_xarm_code(arm, arm.set_position(*init_pose, wait=True), "set_position(init)", raise_on_error=True)
    if gripper_open:
        set_robotiq_position(arm, 0, "robotiq open", wait=True)
    time.sleep(0.5)
    return arm


def reset_arm_to_init(arm, init_pose: list[float]) -> None:
    arm.set_mode(0)
    arm.set_state(0)
    time.sleep(0.1)
    check_xarm_code(arm, arm.set_position(*init_pose, wait=True), "reset set_position", raise_on_error=True)
    set_robotiq_position(arm, 0, "reset robotiq open", wait=True)


def get_xarm_pose_deg(arm) -> np.ndarray:
    code, pose = arm.get_position(is_radian=False)
    check_xarm_code(arm, code, "get_position", raise_on_error=True)
    if pose is None or len(pose) != 6:
        raise RuntimeError(f"get_position 返回非法 pose: {pose}")
    return np.asarray(pose, dtype=np.float64)


def get_xarm_gripper_position(arm) -> float | None:
    if hasattr(arm, "robotiq_get_status"):
        try:
            ret = arm.robotiq_get_status(number_of_registers=3)
            if isinstance(ret, (tuple, list)) and len(ret) >= 1:
                check_xarm_code(arm, ret[0], "robotiq_get_status")
            pos = (getattr(arm, "robotiq_status", {}) or {}).get("gPO")
        except Exception as exc:
            print(f"[WARN] 读取 Robotiq 状态失败: {exc}")
            return None
    elif hasattr(arm, "get_gripper_position"):
        try:
            ret = arm.get_gripper_position()
            pos = ret[1] if isinstance(ret, (tuple, list)) and len(ret) >= 2 else ret
        except Exception as exc:
            print(f"[WARN] 读取夹爪失败: {exc}")
            return None
    else:
        return None
    try:
        return float(pos)
    except (TypeError, ValueError):
        return None


def get_xarm_gripper_open(arm, *, default: float = 1.0) -> float:
    pos = get_xarm_gripper_position(arm)
    if pos is None:
        return float(default)
    return float(np.clip(1.0 - pos / 255.0, 0.0, 1.0))


def xarm_pose_to_state_vec(pose_deg: np.ndarray, gripper_open: float) -> np.ndarray:
    x_mm, y_mm, z_mm, roll, pitch, yaw = pose_deg
    quat = Rotation.from_euler("xyz", [roll, pitch, yaw], degrees=True).as_quat()
    return np.array([x_mm / 1000.0, y_mm / 1000.0, z_mm / 1000.0, *quat, gripper_open], dtype=np.float32)


def gripper_open_to_robotiq(g_open: float) -> int:
    return int(round((1.0 - float(np.clip(g_open, 0.0, 1.0))) * 255.0))


def set_xarm_servo_pose(arm, action: np.ndarray) -> None:
    x_m, y_m, z_m, roll, pitch, yaw = [float(v) for v in action[:6]]
    pose = [x_m * 1000.0, y_m * 1000.0, z_m * 1000.0, roll, pitch, yaw]
    try:
        code = arm.set_servo_cartesian(pose, is_radian=False)
    except TypeError:
        code = arm.set_servo_cartesian(pose)
    check_xarm_code(arm, code, "set_servo_cartesian", raise_on_error=True)


def maybe_set_gripper(arm, g_open: float, state: dict[str, float | None], *, interval_s: float, threshold: float) -> None:
    now = time.monotonic()
    last_open = state.get("open")
    last_time = float(state.get("time") or 0.0)
    if last_open is None or abs(float(g_open) - float(last_open)) >= threshold or now - last_time >= interval_s:
        set_robotiq_position(arm, gripper_open_to_robotiq(g_open), "robotiq_set_position", wait=False)
        state["open"] = float(g_open)
        state["time"] = now


def add_async_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--control_hz", type=float, default=20.0)
    parser.add_argument("--inference_interval_steps", type=int, default=16)
    parser.add_argument("--min_buffer_steps", type=int, default=4)
    parser.add_argument("--empty_action_policy", choices=("hold", "none"), default="hold")
    parser.add_argument("--inference_delay_mode", choices=("fixed", "instant", "ema"), default="instant")
    parser.add_argument("--inference_delay_steps", type=int, default=0)
    parser.add_argument("--max_inference_delay_steps", type=int, default=4)
    parser.add_argument("--reset_delay_on_empty_buffer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--latency_ema_alpha", type=float, default=0.2)
    parser.add_argument("--chunk_blend_horizon_steps", type=int, default=10)
    parser.add_argument("--chunk_blend_schedule", choices=("exp", "linear", "none"), default="exp")
    parser.add_argument("--action_smoothing", choices=("off", "ema"), default="off")
    parser.add_argument("--action_ema_alpha", type=float, default=0.35)
    parser.add_argument("--async_log_interval_s", type=float, default=1.0)
    parser.add_argument("--async_debug_dir", default=None)
    parser.add_argument("--async_debug_flush_interval", type=int, default=1)
    parser.add_argument("--async_debug_include_images", action="store_true")
    parser.add_argument("--max_position_step_m", type=float, default=0.0)
    parser.add_argument("--max_rotation_step_deg", type=float, default=0.0)
    parser.add_argument("--max_gripper_step", type=float, default=0.0)
    parser.add_argument("--rtc_chunk_conditioning", action="store_true")
    parser.add_argument("--rtc_delay_steps", type=int, default=-1)
    parser.add_argument("--rtc_soft_horizon_steps", type=int, default=5)
    parser.add_argument("--rtc_free_tail_steps", type=int, default=5)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--description", required=True)
    parser.add_argument("--robot_ip", default="192.168.1.240")
    parser.add_argument("--camera_dev", type=int, default=DEV)
    parser.add_argument("--init_pose", default="0.25,0.2,0.145,180.0,-90.0,0.0")
    parser.add_argument("--server_ip", default=SERVER_IP)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--action_start", type=int, default=0)
    parser.add_argument("--action_end", type=int, default=50)
    parser.add_argument("--camera-frame-timeout", type=float, default=1.0)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--no_gripper", action="store_true")
    parser.add_argument("--no_gripper_value", type=float, default=1.0)
    parser.add_argument("--gripper_update_interval_s", type=float, default=0.1)
    parser.add_argument("--gripper_update_threshold", type=float, default=0.02)
    parser.add_argument("--gripper_subtract_below", type=float, default=0.8)
    parser.add_argument("--gripper_subtract_amount", type=float, default=0.05)
    parser.add_argument("--depth-image-key", default="front_depth")
    parser.add_argument("--visualize-preview", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--da2-repo", type=Path, default=None)
    parser.add_argument("--da2-encoder", choices=("vits", "vitb", "vitl", "vitg"), default="vitl")
    parser.add_argument("--da2-input-size", type=int, default=672)
    parser.add_argument("--da2-device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--da2-fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--da2-norm-mode", choices=("frame", "fixed", "percentile"), default="percentile")
    parser.add_argument("--da2-fixed-min", type=float, default=None)
    parser.add_argument("--da2-fixed-max", type=float, default=None)
    parser.add_argument("--da2-percentile-min", type=float, default=1.0)
    parser.add_argument("--da2-percentile-max", type=float, default=99.0)
    parser.add_argument("--da2-invert", action="store_true")
    parser.add_argument("--da2-depth-scale", type=float, default=1.0)
    parser.add_argument("--da2-depth-shift", type=float, default=0.0)
    add_async_args(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.control_hz <= 0:
        raise ValueError("--control_hz must be > 0")
    if args.inference_interval_steps < 1:
        raise ValueError("--inference_interval_steps must be >= 1")
    if args.depth_image_key != "front_depth":
        raise ValueError("depth/crop policies currently expect --depth-image-key=front_depth")

    init_pose = parse_pose_deg(args.init_pose)
    args.da2_repo = args.da2_repo or _resolve_default_da2_repo()
    if not args.da2_repo.is_dir():
        raise FileNotFoundError(f"--da2-repo is not a directory: {args.da2_repo}")

    arm = None
    cam = None
    preview = None
    old_term = None
    stdin_fd = None
    stop_event = threading.Event()
    paused_event = threading.Event()
    robot_lock = threading.Lock()
    step_lock = threading.Lock()
    control_step = {"value": 0}
    generation = {"value": 0}
    debug_counts = {"obs_id": 0, "chunk_id": 0}
    last_completed_chunk_id = {"value": None}
    last_delay_steps = {"value": None}
    last_executed_action = {"value": None}
    gripper_state = {"open": None, "time": 0.0}
    rtc_session_id = uuid.uuid4().hex

    action_buffer = ActionBuffer(
        min_buffer_steps=args.min_buffer_steps,
        blend_horizon_steps=args.chunk_blend_horizon_steps,
        blend_schedule=args.chunk_blend_schedule,
        empty_action_policy=args.empty_action_policy,
        action_smoothing=args.action_smoothing,
        action_ema_alpha=args.action_ema_alpha,
        cyclic_indices=(3, 4, 5),
        cyclic_period=360.0,
    )
    latency_estimator = LatencyEstimator(
        mode=args.inference_delay_mode,
        fixed_steps=args.inference_delay_steps,
        control_hz=args.control_hz,
        ema_alpha=args.latency_ema_alpha,
    )
    debug_writer = AsyncDebugWriter(args.async_debug_dir, flush_interval=args.async_debug_flush_interval)

    def get_step() -> int:
        with step_lock:
            return int(control_step["value"])

    def advance_step() -> None:
        with step_lock:
            control_step["value"] += 1

    def get_generation() -> int:
        with step_lock:
            return int(generation["value"])

    def bump_generation() -> None:
        with step_lock:
            generation["value"] += 1

    def next_debug_id(name: str) -> int:
        with step_lock:
            value = int(debug_counts[name])
            debug_counts[name] = value + 1
            return value

    def image_metadata(image_obs: dict[str, np.ndarray]) -> dict:
        return {
            key: {"shape": list(np.asarray(value).shape), "dtype": str(np.asarray(value).dtype)}
            for key, value in image_obs.items()
        }

    def reset_arm() -> None:
        action_buffer.clear()
        if arm is not None:
            with robot_lock:
                arm.set_mode(0)
                arm.set_state(0)
                reset_arm_to_init(arm, init_pose)
        bump_generation()
        with step_lock:
            control_step["value"] = 0
        last_executed_action["value"] = None

    def read_observation():
        if arm is not None:
            with robot_lock:
                pose = get_xarm_pose_deg(arm)
                gripper_open = float(args.no_gripper_value) if args.no_gripper else get_xarm_gripper_open(arm)
        else:
            pose = np.asarray(init_pose, dtype=np.float64)
            gripper_open = float(args.no_gripper_value)
        state_vec = xarm_pose_to_state_vec(pose, gripper_open)

        request_seq = preview.frame_seq()
        rgb_raw, frame_seq, frame_age = preview.wait_for_new_frame(
            after_seq=request_seq,
            timeout=args.camera_frame_timeout,
        )
        depth_raw = depth_runtime.infer_depth_1c(rgb_raw)
        front = cv2.resize(rgb_raw, (IMAGE_OUTPUT_SIZE, IMAGE_OUTPUT_SIZE), interpolation=cv2.INTER_AREA).astype(np.uint8)
        front_crop = bottom_center_crop_resize_rgb(rgb_raw)
        front_depth = cv2.resize(depth_raw, (IMAGE_OUTPUT_SIZE, IMAGE_OUTPUT_SIZE), interpolation=cv2.INTER_AREA)
        if front_depth.ndim == 2:
            front_depth = front_depth[..., None]
        front_depth_crop = bottom_center_crop_resize_depth(depth_raw)
        preview.update_depth(front_depth)
        image_obs = {
            "front": image_tools.convert_to_uint8(front),
            "front_crop": image_tools.convert_to_uint8(front_crop),
            "front_depth": image_tools.convert_to_uint8(front_depth),
            "front_depth_crop": image_tools.convert_to_uint8(front_depth_crop),
        }
        obs = {"state": state_vec, "image": image_obs, "prompt": args.description}
        return obs, image_obs, {"frame_seq": frame_seq, "frame_age_s": frame_age}

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
                obs, image_obs, camera_meta = read_observation()
                obs["__async_rollout"] = {
                    "obs_id": obs_id,
                    "request_step": request_step,
                    "control_hz": args.control_hz,
                    "prev_chunk_id": last_completed_chunk_id["value"],
                    "prev_leftover_steps": action_buffer.pending_count_from(request_step),
                    "delay_mode": args.inference_delay_mode,
                    "delay_steps": last_delay_steps["value"],
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
                debug_writer.write(
                    "observations",
                    TimedObservation(
                        obs_id=obs_id,
                        request_step=request_step,
                        capture_time=capture_time,
                        send_time=time.perf_counter(),
                        buffer_size=action_buffer.pending_count_from(request_step),
                        robot_state=obs.get("state"),
                        image_metadata={**image_metadata(image_obs), "camera": camera_meta},
                    ),
                )
                request_time = time.perf_counter()
                resp = policy_client.infer(obs)
                latency_s = time.perf_counter() - request_time
                if request_generation != get_generation():
                    print(f"[MULTI][infer] discard stale response idx={infer_index}")
                    infer_index += 1
                    continue
                latency_steps = latency_estimator.observe(latency_s)
                current_merge_step = get_step()
                buffer_empty = action_buffer.pending_count_from(current_merge_step) == 0
                if args.reset_delay_on_empty_buffer and (buffer_empty or not action_buffer.has_last_action()):
                    latency_steps = 0
                if args.max_inference_delay_steps >= 0:
                    latency_steps = min(latency_steps, args.max_inference_delay_steps)
                last_delay_steps["value"] = latency_steps
                actions_all = resp["actions"] if "actions" in resp else resp["action"]
                actions_all = np.asarray(actions_all, dtype=np.float64).copy()
                if actions_all.ndim == 1:
                    actions_all = actions_all.reshape(1, -1)
                if actions_all.shape[-1] < 7:
                    raise ValueError(f"Expected action dim >= 7, got {actions_all.shape}")
                if not args.no_gripper and args.gripper_subtract_below is not None:
                    gripper_actions = actions_all[..., 6]
                    mask = gripper_actions < args.gripper_subtract_below
                    gripper_actions[mask] = gripper_actions[mask] - args.gripper_subtract_amount
                    actions_all[..., 6] = np.clip(actions_all[..., 6], 0.0, 1.0)
                chunk_id = next_debug_id("chunk_id")
                rtc_payload = resp.get("rtc", {})
                rtc_applied = bool(rtc_payload.get("applied", False))
                merge_request_step = int(resp.get("action_base_step", request_step)) if rtc_applied else request_step
                merge_latency_steps = 0 if rtc_applied else latency_steps
                merge_action_start = 0 if rtc_applied else args.action_start
                stats = action_buffer.merge_chunk(
                    actions_all,
                    request_step=merge_request_step,
                    current_step=current_merge_step,
                    action_start=merge_action_start,
                    action_end=args.action_end,
                    latency_steps=merge_latency_steps,
                    chunk_id=chunk_id,
                    source_obs_id=obs_id,
                    latency_s=latency_s,
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
                    "[MULTI][infer] "
                    f"idx={infer_index} request_step={request_step} latency={latency_s:.3f}s "
                    f"delay_steps={latency_steps} merge_delay={merge_latency_steps} rtc={rtc_applied} "
                    f"inserted={stats.inserted} blended={stats.blended} skipped={stats.skipped_expired} "
                    f"buffer={action_buffer.pending_count_from(get_step())}"
                )
                infer_index += 1
            except Exception as exc:
                print(f"[MULTI][infer][WARN] {type(exc).__name__}: {exc}")
                print(traceback.format_exc(limit=8).rstrip())
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
            future_gap_active = next_pending_step is not None
            if read.action is None:
                if should_advance_control_step(read, has_future_action=future_gap_active):
                    advance_step()
                now = time.perf_counter()
                if args.async_log_interval_s <= 0.0 or now - last_log >= args.async_log_interval_s:
                    last_log = now
                    print(f"[MULTI][control] step={step} missing_action=True buffer={action_buffer.pending_count_from(step)}")
                next_tick += period_s
                sleep_s = next_tick - time.perf_counter()
                time.sleep(max(sleep_s, 0.0))
                if sleep_s <= 0.0:
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
            pose_before = None
            pose_after = None
            try:
                if arm is not None:
                    with robot_lock:
                        pose_before = get_xarm_pose_deg(arm)
                        if not args.dry_run:
                            set_xarm_servo_pose(arm, limited_action)
                            if not args.no_gripper:
                                maybe_set_gripper(
                                    arm,
                                    float(limited_action[6]),
                                    gripper_state,
                                    interval_s=args.gripper_update_interval_s,
                                    threshold=args.gripper_update_threshold,
                                )
                        pose_after = get_xarm_pose_deg(arm)
                else:
                    print(f"[DRY] step={step} action={limited_action[:7].tolist()}")
                debug_writer.write(
                    "executions",
                    ExecutedAction(
                        control_step=step,
                        execute_time=time.perf_counter(),
                        action=limited_action,
                        held=read.held,
                        missing=read.missing,
                        buffer_size=action_buffer.pending_count_from(step),
                        robot_pose_before=pose_before,
                        robot_pose_after=pose_after,
                        command_delta=action_command_delta(limited_action, last_executed_action["value"]),
                        tracking_error=action_tracking_error(limited_action, pose_before),
                        raw_action=raw_action,
                        limited_action=limited_action,
                        limit_applied=bool(limit_info.get("limit_applied", False)),
                        position_delta_m=limit_info.get("position_delta_m"),
                        rotation_delta_deg=limit_info.get("rotation_delta_deg"),
                    ),
                )
                last_executed_action["value"] = limited_action.copy()
                if should_advance_control_step(read, has_future_action=future_gap_active):
                    advance_step()
            except Exception as exc:
                print(f"[MULTI][control][WARN] {type(exc).__name__}: {exc}")
            now = time.perf_counter()
            if args.async_log_interval_s <= 0.0 or now - last_log >= args.async_log_interval_s:
                last_log = now
                print(
                    f"[MULTI][control] step={step} held={read.held} missing={read.missing} "
                    f"buffer={action_buffer.pending_count_from(step)}"
                )
            next_tick += period_s
            sleep_s = next_tick - time.perf_counter()
            time.sleep(max(sleep_s, 0.0))
            if sleep_s <= 0.0:
                next_tick = time.perf_counter()

    try:
        if not args.dry_run:
            print(f"[INFO] 连接 xArm: {args.robot_ip}")
            arm = init_xarm(args.robot_ip, init_pose)
            arm.set_mode(1)
            arm.set_state(0)
            time.sleep(0.05)
            print("[INFO] xArm servo Cartesian mode enabled")

        policy_client = websocket_client_policy.WebsocketClientPolicy(host=args.server_ip, port=args.port)
        print(f"[INFO] 已连接策略服务器：ws://{args.server_ip}:{args.port}")
        metadata = policy_client.get_server_metadata() if args.rtc_chunk_conditioning else {}
        if args.rtc_chunk_conditioning and not metadata.get("rtc", {}).get("enabled", False):
            raise RuntimeError("client 开启 --rtc_chunk_conditioning，但 server 未启用 --rtc-chunk-conditioning")

        print(f"[INFO] 加载 Depth Anything V2: repo={args.da2_repo}, encoder={args.da2_encoder}")
        depth_runtime = DepthAnything2Runtime(
            da2_repo=args.da2_repo,
            encoder=args.da2_encoder,
            device_arg=args.da2_device,
            input_size=args.da2_input_size,
            fp16=args.da2_fp16,
            norm_mode=args.da2_norm_mode,
            fixed_min=args.da2_fixed_min,
            fixed_max=args.da2_fixed_max,
            percentile_min=args.da2_percentile_min,
            percentile_max=args.da2_percentile_max,
            invert=args.da2_invert,
            depth_scale=args.da2_depth_scale,
            depth_shift=args.da2_depth_shift,
        )
        print(f"[INFO] Depth Anything V2 loaded on {depth_runtime.device}")

        cam = init_yu12_camera(args.camera_dev)
        for _ in range(60):
            _ = cam.read()
        input("按 Enter 开始")
        preview = RealtimeCameraPreview(cam, visual_enabled=args.visualize_preview)
        preview.start()
        _ = preview.wait_for_new_frame(after_seq=0, timeout=args.camera_frame_timeout)
        print("[INFO] 预览已启动。按 s：复位并暂停；按 c：继续；按 q：退出")

        if sys.stdin.isatty():
            stdin_fd = sys.stdin.fileno()
            old_term = termios.tcgetattr(stdin_fd)
            tty.setcbreak(stdin_fd)

        infer_thread = threading.Thread(target=inference_loop, name="multi-inference", daemon=True)
        control_thread = threading.Thread(target=control_loop, name="multi-control", daemon=True)
        infer_thread.start()
        control_thread.start()

        while not stop_event.is_set():
            key = read_control_key(wait_ms=20)
            if key == "q":
                stop_event.set()
                break
            if key == "s":
                paused_event.set()
                print("[INFO] reset + pause")
                reset_arm()
                continue
            if key == "c":
                paused_event.clear()
                print("[INFO] continue")
        infer_thread.join(timeout=2.0)
        control_thread.join(timeout=2.0)
    finally:
        stop_event.set()
        debug_writer.close(
            {
                "control_hz": args.control_hz,
                "inference_interval_steps": args.inference_interval_steps,
                "rtc_chunk_conditioning": args.rtc_chunk_conditioning,
            }
        )
        if old_term is not None and stdin_fd is not None:
            termios.tcsetattr(stdin_fd, termios.TCSADRAIN, old_term)
        if preview is not None:
            preview.stop()
        if cam is not None:
            cam.release()
        cv2.destroyAllWindows()
        if arm is not None:
            try:
                arm.set_mode(0)
                arm.set_state(0)
            finally:
                arm.disconnect()
        print("[INFO] rollout_multi exited")


if __name__ == "__main__":
    main()
