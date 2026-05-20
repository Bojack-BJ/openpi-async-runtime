#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

import numpy as np


def _read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64)
    if arr.size == 0:
        return None
    return arr


def _wrap_degrees(delta: np.ndarray) -> np.ndarray:
    return ((np.asarray(delta, dtype=np.float64) + 180.0) % 360.0) - 180.0


def _save_timeline(chunks: list[dict], executions: list[dict], output_dir: pathlib.Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=False)
    if chunks:
        chunk_steps = [row.get("request_step", i) for i, row in enumerate(chunks)]
        axes[0].plot(chunk_steps, [row.get("latency_s", 0.0) for row in chunks], marker="o", label="latency_s")
        axes[0].set_ylabel("latency_s")
        axes[1].plot(chunk_steps, [row.get("delay_steps", 0) for row in chunks], marker="o", label="delay_steps")
        axes[1].set_ylabel("delay_steps")
    if executions:
        steps = [row.get("control_step", i) for i, row in enumerate(executions)]
        axes[2].plot(steps, [row.get("buffer_size", 0) for row in executions], label="buffer")
        axes[2].set_ylabel("buffer")
        axes[3].plot(steps, [1 if row.get("missing") else 0 for row in executions], label="missing")
        axes[3].plot(steps, [1 if row.get("held") else 0 for row in executions], label="held")
        axes[3].set_ylabel("flags")
        axes[3].legend()
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "timeline.png", dpi=160)
    plt.close(fig)


def _save_action_delta(executions: list[dict], output_dir: pathlib.Path) -> dict[str, float]:
    import matplotlib.pyplot as plt

    steps = []
    pos_delta = []
    rot_delta = []
    grip_delta = []
    for row in executions:
        steps.append(row.get("control_step", len(steps)))
        pos_delta.append(float(row.get("position_delta_m") or 0.0))
        rot_delta.append(float(row.get("rotation_delta_deg") or 0.0))
        delta = _array(row.get("command_delta"))
        grip_delta.append(float(abs(delta[6])) if delta is not None and len(delta) > 6 else 0.0)

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    axes[0].plot(steps, pos_delta, label="position_delta_m")
    axes[1].plot(steps, rot_delta, label="rotation_delta_deg", color="tab:orange")
    axes[2].plot(steps, grip_delta, label="gripper_delta", color="tab:green")
    for ax in axes:
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "action_delta.png", dpi=160)
    plt.close(fig)
    return {
        "max_position_delta_m": max(pos_delta, default=0.0),
        "max_rotation_delta_deg": max(rot_delta, default=0.0),
        "max_gripper_delta": max(grip_delta, default=0.0),
    }


def _save_command_vs_actual(executions: list[dict], output_dir: pathlib.Path) -> dict[str, float]:
    import matplotlib.pyplot as plt

    rows = [row for row in executions if row.get("robot_pose_after") is not None]
    if not rows:
        return {"max_position_error_m": 0.0, "max_rotation_error_deg": 0.0}

    steps = [row.get("control_step", i) for i, row in enumerate(rows)]
    command = np.asarray([row.get("limited_action") or row.get("action") for row in rows], dtype=np.float64)
    actual = np.asarray([row["robot_pose_after"] for row in rows], dtype=np.float64)
    dims = min(command.shape[1], actual.shape[1], 6)

    fig, axes = plt.subplots(dims, 1, figsize=(14, 2.3 * dims), sharex=True)
    if dims == 1:
        axes = [axes]
    labels = ["x", "y", "z", "roll", "pitch", "yaw"]
    for i in range(dims):
        axes[i].plot(steps, command[:, i], label=f"cmd_{labels[i]}")
        axes[i].plot(steps, actual[:, i], label=f"actual_{labels[i]}", alpha=0.7)
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "command_vs_actual.png", dpi=160)
    plt.close(fig)

    pos_err = np.linalg.norm(command[:, :3] - actual[:, :3], axis=1) if dims >= 3 else np.zeros(len(rows))
    rot_err = np.linalg.norm(_wrap_degrees(command[:, 3:6] - actual[:, 3:6]), axis=1) if dims >= 6 else np.zeros(len(rows))
    return {
        "max_position_error_m": float(np.max(pos_err)) if len(pos_err) else 0.0,
        "max_rotation_error_deg": float(np.max(rot_err)) if len(rot_err) else 0.0,
    }


def _save_chunk_merge(actions: list[dict], output_dir: pathlib.Path) -> None:
    import matplotlib.pyplot as plt

    if not actions:
        return
    colors = {"inserted": "tab:blue", "blended": "tab:orange", "skipped": "tab:red", "held": "tab:green"}
    fig, ax = plt.subplots(figsize=(14, 7))
    for merge_type, color in colors.items():
        rows = [row for row in actions if row.get("merge_type") == merge_type and row.get("target_step") is not None]
        if not rows:
            continue
        ax.scatter(
            [row["target_step"] for row in rows],
            [row.get("chunk_id", 0) for row in rows],
            s=10,
            color=color,
            label=merge_type,
            alpha=0.75,
        )
    ax.set_xlabel("target_step")
    ax.set_ylabel("chunk_id")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "chunk_merge.png", dpi=160)
    plt.close(fig)


def _write_summary(debug_dir: pathlib.Path, output_dir: pathlib.Path, chunks: list[dict], executions: list[dict], metrics: dict) -> None:
    latencies = [float(row.get("latency_s", 0.0)) for row in chunks]
    missing_count = sum(1 for row in executions if row.get("missing"))
    held_count = sum(1 for row in executions if row.get("held"))
    total_exec = max(len(executions), 1)
    top_spikes = sorted(
        executions,
        key=lambda row: float(row.get("position_delta_m") or 0.0) + 0.01 * float(row.get("rotation_delta_deg") or 0.0),
        reverse=True,
    )[:10]
    lines = [
        "# Async Rollout Debug Summary",
        "",
        f"- debug_dir: `{debug_dir}`",
        f"- chunks: {len(chunks)}",
        f"- executions: {len(executions)}",
        f"- latency_mean_s: {float(np.mean(latencies)) if latencies else 0.0:.4f}",
        f"- latency_max_s: {max(latencies, default=0.0):.4f}",
        f"- missing_ratio: {missing_count / total_exec:.4f}",
        f"- held_ratio: {held_count / total_exec:.4f}",
    ]
    for key, value in metrics.items():
        lines.append(f"- {key}: {value:.6f}")
    lines.extend(["", "## Top Delta Events", ""])
    for row in top_spikes:
        lines.append(
            f"- step={row.get('control_step')} pos_delta={row.get('position_delta_m')} "
            f"rot_delta={row.get('rotation_delta_deg')} held={row.get('held')} missing={row.get('missing')}"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    debug_dir = pathlib.Path(args.debug_dir)
    output_dir = pathlib.Path(args.output_dir) if args.output_dir else debug_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    chunks = _read_jsonl(debug_dir / "chunks.jsonl")
    actions = _read_jsonl(debug_dir / "actions.jsonl")
    executions = _read_jsonl(debug_dir / "executions.jsonl")

    _save_timeline(chunks, executions, output_dir)
    delta_metrics = _save_action_delta(executions, output_dir)
    tracking_metrics = _save_command_vs_actual(executions, output_dir)
    _save_chunk_merge(actions, output_dir)
    _write_summary(debug_dir, output_dir, chunks, executions, {**delta_metrics, **tracking_metrics})
    print(f"Wrote async rollout plots to {output_dir}")


if __name__ == "__main__":
    main()
