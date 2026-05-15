# Async RTC-Style Rollout

本文档覆盖两类 rollout 策略：

- 同步脚本的 `dt` 限速 workaround：用于临时缓解 `set_position(wait=False)` 指令堆积。
- 异步 RTC-style rollout：长期方案，把 policy 推理和机器人控制解耦。

## 同步脚本 Workaround

`scripts/pi0_rollout_client_xarm_rpy.py` 的 normal 版默认用 `set_position(..., wait=False)`。`wait=False` 只是不阻塞 Python，并不保证 xArm 控制器立刻丢弃旧目标；如果 for-loop 很快塞入一整个 action chunk，机械臂可能还在执行旧 chunk，Python 已经开始下一轮推理，于是 chunk 切换时会看起来“回退”。

临时缓解方式是启用客户端插值和 `dt` 限速，让每个目标按固定节奏下发：

```bash
python scripts/pi0_rollout_client_xarm_rpy.py \
  --description "Put the target object into the target slot" \
  --arm_mode single \
  --single_arm robot_1 \
  --single_image_key front \
  --robot_ip_2 192.168.1.240 \
  --server_ip 180.184.74.93 \
  --port 8005 \
  --action_start 10 \
  --action_end 30 \
  --enable_interp \
  --dt 0.05 \
  --interp_step_size 0.0025
```

插值规则：

- `interp_step_size` 控制相邻目标之间最大平移步长，单位米。默认 `0.0025` 表示超过 2.5mm 就细分。
- `dt` 是每个插值小步之间的 sleep 秒数。`dt=0.05` 约等于 20Hz 下发。
- 当前实现对每个 action 从上一目标线性插值到当前目标，RPY 也是线性插值；它不是完整 RTC，也不会解决推理阻塞。

如果只想限速而不想额外细分，可以设：

```bash
--enable_interp \
--dt 0.05 \
--interp_step_size 0
```

但当前同步实现里 `step_size<=0` 仍会下发 start 和 target 两个点，所以严格频率会低于 `1/dt`。更稳定的长期方案是使用 async rollout，尤其是 xArm 的 `servo` 模式。

## Async 系统框架

异步脚本不改原始同步脚本，新增：

```bash
python scripts/pi0_rollout_client_fasttouch_rpy_async.py ...
python scripts/pi0_rollout_client_xarm_rpy_async.py ...
```

运行时内部有两个线程和一个共享 buffer：

```text
camera/robot state
       |
       v
inference thread -- policy server --> action chunk
       |                               |
       | request_step, latency         v
       +------------------------> ActionBuffer(step -> action)
                                        |
                                        v
control thread -- fixed control_hz --> robot SDK
```

- `inference thread`：在某个 control step 采集 observation，调用 policy server，得到 action chunk。
- `ActionBuffer`：用绝对 control step 存储 action，并对新旧 chunk 的重叠区做 blend。
- `control thread`：按 `--control_hz` 固定频率取当前 step 的 action 并下发机器人；不等待 policy inference。

## 推荐运行命令

xArm 低风险起步，先用 `position` 模式验证 buffer 连续性：

```bash
python scripts/pi0_rollout_client_xarm_rpy_async.py \
  --description "Put the target object into the target slot" \
  --arm_mode single \
  --single_arm robot_1 \
  --single_image_key front \
  --robot_ip_2 192.168.1.240 \
  --server_ip 180.184.74.93 \
  --port 8005 \
  --action_start 0 \
  --action_end 49 \
  --control_hz 10 \
  --inference_interval_steps 8 \
  --inference_delay_mode fixed \
  --inference_delay_steps 0 \
  --max_inference_delay_steps 4 \
  --reset_delay_on_empty_buffer \
  --min_buffer_steps 2 \
  --chunk_blend_horizon_steps 8 \
  --chunk_blend_schedule exp \
  --action_smoothing off \
  --xarm_control_mode position \
  --async_log_interval_s 0.5
```

确认 `missing=True` 很少出现后，再切到 servo：

```bash
python scripts/pi0_rollout_client_xarm_rpy_async.py \
  --description "Put the target object into the target slot" \
  --arm_mode single \
  --single_arm robot_1 \
  --single_image_key front \
  --robot_ip_2 192.168.1.240 \
  --server_ip 180.184.74.93 \
  --port 8005 \
  --action_start 0 \
  --action_end 49 \
  --control_hz 20 \
  --inference_interval_steps 8 \
  --inference_delay_mode instant \
  --inference_delay_steps 0 \
  --max_inference_delay_steps 4 \
  --reset_delay_on_empty_buffer \
  --min_buffer_steps 2 \
  --chunk_blend_horizon_steps 10 \
  --chunk_blend_schedule exp \
  --action_smoothing ema \
  --action_ema_alpha 0.25 \
  --xarm_control_mode servo \
  --async_log_interval_s 0.5
```

FastTouch 用同一套时间对齐参数：

```bash
python scripts/pi0_rollout_client_fasttouch_rpy_async.py \
  --description "Put the target object into the target slot" \
  --arm_mode single \
  --single_arm robot_1 \
  --single_image_key front \
  --right_can can1 \
  --server_ip 180.184.74.93 \
  --port 8005 \
  --action_start 0 \
  --action_end 49 \
  --control_hz 20 \
  --inference_interval_steps 8 \
  --inference_delay_mode fixed \
  --inference_delay_steps 0 \
  --max_inference_delay_steps 4 \
  --reset_delay_on_empty_buffer \
  --chunk_blend_horizon_steps 10 \
  --chunk_blend_schedule exp \
  --action_smoothing off
```

## 时间对齐策略

### Observation 时间

inference thread 在发起请求时记录：

```text
request_step = current_control_step
request_time = monotonic_time
```

observation 中的 `state` 和 `image` 对应 `request_step` 附近的真实机器人状态。policy 返回的是从该 observation 出发预测的未来 action chunk：

```text
actions[0], actions[1], ..., actions[T-1]
```

### Inference 延迟补偿

policy 返回时计算本次推理耗时：

```text
latency_s = now - request_time
```

延迟步数有三种模式：

- `--inference_delay_mode fixed`：直接使用 `--inference_delay_steps`。实机调试最稳，常用 `0/2/4`。
- `--inference_delay_mode instant`：使用本次 latency：`ceil(latency_s * control_hz)`。默认模式，不受偶发长尾推理的历史影响。
- `--inference_delay_mode ema`：使用 `ceil(EMA(latency_s) * control_hz)`。只适合 latency 很稳定的环境；如果偶发一次 2-3s 长尾，后面几次可能继续过度跳过。
- 上限：`--max_inference_delay_steps` 会限制补偿最多跳过多少步；默认 `4`，避免 20Hz 下 0.5s 推理直接跳过 10 个动作。
- 空 buffer 重置：默认开启 `--reset_delay_on_empty_buffer`。启动或断流后，下一次 chunk 不再按历史 latency 跳到很远的未来动作，而是先恢复连续控制。

新 chunk 的可用起点是：

```text
effective_start = action_start + latency_steps
```

也就是说，已经因为推理延迟错过的 action 会被跳过，不再塞进 buffer。

### Action 到绝对 Step 的映射

每个 action 按绝对 control step 写入：

```text
target_step = request_step + action_index
```

例如 `request_step=100`，`action_start=0`，`latency_steps=5`，则 `actions[0:5]` 被丢弃，`actions[5]` 写到 `step=105`。

`--min_buffer_steps` 会保护马上要执行的 step：

```text
frozen_until = current_step + min_buffer_steps
```

`target_step < frozen_until` 的 action 不会覆盖旧 buffer，避免 control thread 正在读取附近 action 时被新 chunk 改写。

## 核心参数

- `--control_hz`：机器人控制线程频率，单位 Hz。`20` 表示每 `0.05s` 执行一个 action step。这个值应接近训练数据/模型 action 的时间频率；太高会更容易把 inference latency 换算成很多 delay steps。
- `--inference_interval_steps`：每隔多少个 control steps 启动一次新推理。例：`control_hz=20` 且 `inference_interval_steps=8`，约每 `0.4s` 请求一个新 chunk。
- `--action_start` / `--action_end`：从 policy 返回的 action chunk 中使用哪个闭区间。注意动态延迟会在 `action_start` 基础上继续跳过若干步。
- `--inference_delay_mode`：`fixed/instant/ema`。默认 `instant`。
- `--inference_delay_steps`：`fixed` 模式下使用的固定延迟补偿步数。`0` 表示不跳过 action。
- `--max_inference_delay_steps`：延迟补偿上限。默认 `4`；`<0` 表示不限制。实机上不建议无限动态跳步，否则 policy latency 抖动会变成 action index 抖动。
- `--reset_delay_on_empty_buffer` / `--no-reset_delay_on_empty_buffer`：启动或 buffer 断流后是否把 delay 临时置 0。默认开启，防止恢复时直接跳进未来动作。
- `--latency_ema_alpha`：动态延迟 EMA 系数，默认 `0.2`。越小越稳定，越大越跟随当前推理耗时。
- `--min_buffer_steps`：保护最近 N 个将要执行的 steps，不让新 chunk 覆盖，避免 race。默认 `2`。
- `--empty_action_policy`：buffer 当前 step 没 action 时的策略。默认 `hold`，复用上一条 action；如果还没有上一条 action，就跳过该 tick。

## 控制端策略

control thread 只做固定频率控制，不做推理：

```text
period_s = 1 / control_hz
for each tick:
    action = buffer.pop(current_step)
    if missing:
        hold last action or wait at current action step
    optional output smoothing
    send robot command
    current_step += 1
```

xArm 有两种下发模式：

- `--xarm_control_mode position`：使用 `set_position(..., wait=False)`。兼容性最好，但普通运动规划可能仍有控制器端队列/滞后。
- `--xarm_control_mode servo`：使用 `set_servo_cartesian`，脚本进入时切 `set_mode(1)`，退出/reset 时恢复 `set_mode(0)`。SDK 语义是 servo mode 下只追最新指令，更适合固定频率控制。

夹爪命令会节流，不会每个 control tick 都发，避免 Robotiq 命令拖慢主控制循环。

注意：`control_hz` 必须和模型 action 的训练频率匹配。若模型 action 是 10Hz 语义，用 `control_hz=20` 会把一个 chunk 以两倍速度执行，表现为“动得特别快”。若 `10Hz` 又抖，优先开 `servo`、增大 `chunk_blend_horizon_steps` 或开启轻量 EMA，而不是直接把 `control_hz` 提到 20。

上一次修复后的 step 语义：

- 没有任何 action 可执行时，control thread 只等待，不推进 `control_step`。否则启动阶段 policy 还没返回，`control_step` 会先跑到 10/20，首个 chunk 回来就被当成过期动作丢掉。
- `held=True missing=True` 时，只重复上一条命令维持机器人，不推进 `control_step`。否则一次 buffer 断流会让 action 时间轴继续向前跑，新 chunk 回来又会被大量跳过。
- 首次填充 buffer 时不应用 `min_buffer_steps` 冻结区。否则第一条可用 action 会从 `step=2` 或更后开始，而 control thread 卡在 `step=0`。
- 启动或 buffer 断流后默认临时把 delay 置 0，先恢复连续控制，再让后续 chunk 做正常延迟补偿。

## Chunk 对齐和 Blend

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

xArm/FastTouch 的 RPY action 会按角度周期处理。也就是说 `179 -> -179` 会沿最短路径融合，不会线性穿过 `0` 度；否则在欧拉角接近 `±180` 或 pitch 接近 `±90` 时，chunk overlap 很容易制造一次假的大幅 yaw/roll 旋转。

chunk merge 的原则：

- 越靠近当前执行 step，越信任旧 buffer，避免临近动作突然被改。
- 越远离当前执行 step，越信任新 chunk，让策略能及时更新未来动作。
- `none` 适合排查问题；`linear` 更可解释；`exp` 是默认，更接近 RTC soft mask 的直觉。

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
--inference_delay_mode fixed \
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
- `buffer=0`：推理供不上，控制线程快要断粮。当前实现不会在没有可用 action 时消耗新的 action step，避免首个/恢复 chunk 被整段判定为过期。

常见问题：

- 只打印一行 infer 后不动：看 `inserted` 是否为 `0`。若为 `0`，降低 `--control_hz`、使用 `--inference_delay_mode fixed --inference_delay_steps 0`，或增大 `--action_end`。
- 一顿一顿：看 `missing=True` 是否频繁出现。若频繁出现，降低 `--control_hz` 或减小 `--inference_interval_steps`。
- `skipped` 很大并且随后 `buffer=0`：推理耗时已经吃掉了 chunk 后半段。例如 `control_hz=20` 且一次推理耗时 `2.5s`，控制线程已经前进约 `50` 步；如果 `action_end` 只有 `49`，这个 chunk 基本没有未来 action 可用。先用 `--control_hz 10` 或增大 `--action_end` 验证。
- 边界跳：增大 `--chunk_blend_horizon_steps`。
- normal 同步版 chunk 切换回退：先用 `--enable_interp --dt 0.05` 限速；如果仍有旧 chunk 滞后，切 async + `--xarm_control_mode servo`。
- `control_hz=20` 明显太快：说明 action 时间语义更接近 10Hz。先用 `--control_hz 10 --xarm_control_mode servo --chunk_blend_horizon_steps 10 --action_smoothing ema --action_ema_alpha 0.25`，再按抖动情况微调。
