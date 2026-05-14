# Async RTC-Style Rollout

这套脚本把 policy 推理和机器人控制拆成两个线程：

- `inference thread`：采集 observation，调用 policy server，得到 action chunk。
- `control thread`：按固定频率从 action buffer 取 action，下发机器人；不等待推理。

入口脚本：

```bash
python scripts/pi0_rollout_client_fasttouch_rpy_async.py ...
python scripts/pi0_rollout_client_xarm_rpy_async.py ...
```

原始同步脚本不受影响。

## 推荐起步参数

先用低频、无动态延迟、无 smoothing 验证控制链路：

```bash
--control_hz 10 \
--inference_interval_steps 6 \
--inference_delay_steps 0 \
--action_start 0 \
--action_end 20 \
--chunk_blend_horizon_steps 6 \
--chunk_blend_schedule exp \
--action_smoothing off \
--async_log_interval_s 0.5
```

确认能连续运动后，再逐步改：

```bash
--control_hz 20 \
--inference_delay_steps -1
```

如果 `inserted=0` 或机器人不动，通常是动态延迟估计超过 action chunk 可用范围。先固定 `--inference_delay_steps 0/2/4` 排查。

## 核心参数

- `--control_hz`：机器人控制线程频率，单位 Hz。`20` 表示每 `0.05s` 执行一个 action step。这个值应接近训练数据/模型 action 的时间频率；太高会更容易把 inference latency 换算成很多 delay steps。
- `--inference_interval_steps`：每隔多少个 control steps 启动一次新推理。例：`control_hz=20` 且 `inference_interval_steps=8`，约每 `0.4s` 请求一个新 chunk。
- `--action_start` / `--action_end`：从 policy 返回的 action chunk 中使用哪个闭区间。注意动态延迟会在 `action_start` 基础上继续跳过若干步。
- `--inference_delay_steps`：延迟补偿步数。`0` 表示不跳过 action；`>=0` 使用固定值；`-1` 使用动态 EMA：`ceil(EMA(inference_latency_s) * control_hz)`。
- `--latency_ema_alpha`：动态延迟 EMA 系数，默认 `0.2`。越小越稳定，越大越跟随当前推理耗时。
- `--min_buffer_steps`：保护最近 N 个将要执行的 steps，不让新 chunk 覆盖，避免 race。默认 `2`。
- `--empty_action_policy`：buffer 当前 step 没 action 时的策略。默认 `hold`，复用上一条 action；如果还没有上一条 action，就跳过该 tick。

## Chunk Blending

新 chunk 到达时会按绝对 control step merge 到 action buffer：

```text
insert_step = request_step + action_index
effective_start = action_start + latency_steps
```

重叠区域通过 soft blending 融合旧 action 和新 action：

- `--chunk_blend_schedule none`：新 chunk 直接覆盖旧 chunk。
- `--chunk_blend_schedule linear`：从旧 action 线性过渡到新 action。
- `--chunk_blend_schedule exp`：指数过渡，默认值；更接近 RTC soft mask 的直觉。
- `--chunk_blend_horizon_steps`：过渡长度。越大越平滑，但越滞后；越小越跟随新预测，但 chunk 边界更容易跳。

调参建议：

- chunk 边界跳：增大 `--chunk_blend_horizon_steps`，例如 `6 -> 10 -> 14`。
- 整体滞后：减小 `--chunk_blend_horizon_steps`，或用 `linear/none`。

## Output Smoothing

这是 buffer merge 后的最后一道低通滤波，默认关闭：

```bash
--action_smoothing ema \
--action_ema_alpha 0.35
```

- `--action_smoothing off`：默认，不做额外滤波。
- `--action_smoothing ema`：对最终下发 action 做 EMA。
- `--action_ema_alpha`：越小越稳但越滞后；`1.0` 等于几乎不平滑。

建议先调 `chunk_blend_*`，再开 EMA。EMA 会掩盖 buffer/latency 问题，不建议一开始就开。

## xArm 专用参数

`scripts/pi0_rollout_client_xarm_rpy_async.py` 额外支持：

```bash
--xarm_control_mode position
--xarm_control_mode servo
```

- `position`：默认，使用 `set_position(..., wait=False)`，兼容性最高。
- `servo`：使用 `set_servo_cartesian`，进入脚本后切 `mode=1`，退出/reset 时恢复 `mode=0`。延迟更低，但需要 SDK/固件支持，且更依赖稳定的 action stream。

如果 `servo` 抖，先确认 async buffer 能稳定供 action，再调：

```bash
--control_hz 20 \
--inference_delay_steps 0 \
--chunk_blend_horizon_steps 10 \
--action_smoothing ema \
--action_ema_alpha 0.25
```

## 日志怎么看

推理日志：

```text
[ASYNC][infer] idx=0 request_step=0 latency=1.230s delay_steps=13 inserted=8 blended=0 skipped=13 buffer=8
```

- `latency`：本次 policy server 耗时。
- `delay_steps`：延迟补偿跳过了多少 action steps。
- `inserted`：新插入 buffer 的 action 数。
- `blended`：和旧 buffer 重叠融合的 action 数。
- `skipped`：因为延迟或 min-buffer 被丢掉的 action 数。
- `buffer`：从当前 step 往后的待执行 action 数。

控制日志：

```text
[ASYNC][control] step=23 held=False missing=False buffer=7
```

- `missing=True`：当前 step 没有 action。
- `held=True`：当前 step 没 action，但复用了上一条 action。
- `buffer=0`：推理供不上，控制线程快要断粮。

常见问题：

- 只打印一行 infer 后不动：看 `inserted` 是否为 `0`。若为 `0`，降低 `--control_hz`、固定较小 `--inference_delay_steps`，或增大 `--action_end`。
- 一顿一顿：看 `missing=True` 是否频繁出现。若频繁出现，降低 `--control_hz` 或减小 `--inference_interval_steps`。
- 边界跳：增大 `--chunk_blend_horizon_steps`。
