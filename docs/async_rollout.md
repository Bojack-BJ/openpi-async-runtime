# Async RTC-Style Rollout

本文档覆盖两类 rollout 策略：

- 同步脚本的 `dt` 限速 workaround：用于临时缓解 `set_position(wait=False)` 指令堆积。
- 异步 RTC-style rollout：长期方案，把 policy 推理和机器人控制解耦。

## 同步脚本 Workaround

`scripts/rollout/pi0_rollout_client_xarm_rpy.py` 的 normal 版默认用 `set_position(..., wait=False)`。`wait=False` 只是不阻塞 Python，并不保证 xArm 控制器立刻丢弃旧目标；如果 for-loop 很快塞入一整个 action chunk，机械臂可能还在执行旧 chunk，Python 已经开始下一轮推理，于是 chunk 切换时会看起来“回退”。

临时缓解方式是启用客户端插值和 `dt` 限速，让每个目标按固定节奏下发：

```bash
python scripts/rollout/pi0_rollout_client_xarm_rpy.py \
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
python scripts/rollout/pi0_rollout_client_fasttouch_rpy_async.py ...
python scripts/rollout/pi0_rollout_client_xarm_rpy_async.py ...
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
python scripts/rollout/pi0_rollout_client_xarm_rpy_async.py \
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
  --min_buffer_steps 2 \
  --chunk_blend_horizon_steps 8 \
  --chunk_blend_schedule exp \
  --action_smoothing off \
  --xarm_control_mode position \
  --async_log_interval_s 0.5
```

确认 `missing=True` 很少出现后，再切到 servo。先固定不跳过 action，排除 latency 补偿造成的跳变：

```bash
python scripts/rollout/pi0_rollout_client_xarm_rpy_async.py \
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
  --min_buffer_steps 2 \
  --chunk_blend_horizon_steps 10 \
  --chunk_blend_schedule exp \
  --action_smoothing ema \
  --action_ema_alpha 0.25 \
  --async_debug_dir /tmp/openpi_async_debug/xarm_run01 \
  --async_debug_readback_every_n_steps 2 \
  --xarm_control_mode servo \
  --async_log_interval_s 0.5
```

FastTouch 用同一套时间对齐参数：

```bash
python scripts/rollout/pi0_rollout_client_fasttouch_rpy_async.py \
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
  --chunk_blend_horizon_steps 10 \
  --chunk_blend_schedule exp \
  --action_smoothing off
```

如果 fixed 0 明显滞后，再试当前 latency 补偿。注意这里不要再传 `--inference_delay_steps`，它只在 `fixed` 模式生效：

```bash
--inference_delay_mode instant \
--max_inference_delay_steps 2 \
--reset_delay_on_empty_buffer
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

| mode | delay 来源 | `--inference_delay_steps` | 适用场景 |
| --- | --- | --- | --- |
| `fixed` | 固定步数 | 生效 | 实机调试最稳，常用 `0/2/4` |
| `instant` | `ceil(latency_s * control_hz)` | 不生效 | 默认模式，只看本次 latency |
| `ema` | `ceil(EMA(latency_s) * control_hz)` | 不生效 | latency 很稳定时才用 |

`--max_inference_delay_steps` 会限制补偿最多跳过多少步；默认 `4`，避免 20Hz 下 0.5s 推理直接跳过 10 个动作。`--reset_delay_on_empty_buffer` 默认开启，启动或断流后下一次 chunk 不再按历史 latency 跳到很远的未来动作，而是先恢复连续控制。

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
- `--inference_delay_steps`：仅在 `--inference_delay_mode fixed` 时生效。`0` 表示不跳过 action；`instant/ema` 下会被忽略。
- `--max_inference_delay_steps`：延迟补偿上限。默认 `4`；`<0` 表示不限制。对 `instant/ema` 最有用，也会 cap 过大的 fixed delay。
- `--reset_delay_on_empty_buffer` / `--no-reset_delay_on_empty_buffer`：启动或 buffer 断流后是否把 delay 临时置 0。默认开启，防止恢复时直接跳进未来动作。
- `--latency_ema_alpha`：动态延迟 EMA 系数，默认 `0.2`。越小越稳定，越大越跟随当前推理耗时。
- `--min_buffer_steps`：保护最近 N 个将要执行的 steps，不让新 chunk 覆盖，避免 race。默认 `2`。
- `--empty_action_policy`：buffer 当前 step 没 action 时的策略。默认 `hold`，复用上一条 action；如果还没有上一条 action，就跳过该 tick。
- `--async_debug_dir`：启用 JSONL debug 输出目录；不传则完全关闭。
- `--async_debug_readback_every_n_steps`：每 N 个 control steps 读取一次真实机器人 pose；`0` 关闭 readback，避免额外阻塞。
- `--async_debug_flush_interval`：JSONL flush 间隔，默认每条 flush，最利于崩溃后保留日志。
- `--async_debug_include_images`：只记录图像 shape/dtype/min/max，不保存图像字节；默认关闭。
- `--max_position_step_m` / `--max_rotation_step_deg` / `--max_gripper_step`：控制端每 tick 限幅，默认 `0` 表示关闭。

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

xArm 有四种下发模式：

- `--xarm_control_mode position`：使用 `set_position(..., wait=False)`。兼容性最好，但普通运动规划可能仍有控制器端队列/滞后。
- `--xarm_control_mode servo`：使用 `set_servo_cartesian`，脚本进入时切 `set_mode(1)`，退出/reset 时恢复 `set_mode(0)`。SDK 语义是 servo mode 下只追最新指令，更适合固定频率控制。
- `--xarm_control_mode online_cartesian`：使用 `set_position(..., wait=False)`，脚本进入时切 `set_mode(7)`，退出/reset 时恢复 `set_mode(0)`。新目标会中断旧目标并由 xArm 控制器在线重规划，通常比 `servo` 更平滑。
- `--xarm_control_mode plan-servo`：使用 xArm SDK IK、外部 joint planner 和 `set_servo_angle_j()`。新 chunk 到达后从实时 `q/dq` 重规划，按显式模型 action dt 生成高频 joint samples，适合精细操作。

夹爪命令会节流，不会每个 control tick 都发，避免 Robotiq 命令拖慢主控制循环。

注意：`control_hz` 必须和模型 action 的训练频率匹配。若模型 action 是 10Hz 语义，用 `control_hz=20` 会把一个 chunk 以两倍速度执行，表现为“动得特别快”。若 `10Hz` 又抖，优先试 `online_cartesian`、增大 `chunk_blend_horizon_steps` 或开启轻量 EMA，而不是直接把 `control_hz` 提到 20。

如果 `latency` 已经稳定、`delay_steps` 也很小但仍然抽搐，原因通常不是 latency，而是控制端在追一个跳变的绝对 pose：

- policy 相邻 action 本身不平滑，servo 会快速追最新目标。
- 新旧 chunk 的同一未来 step 差异较大，blend horizon 太短会在边界跳。
- RPY 在 pitch 接近 `±90°` 时有欧拉角奇异性，即使做了逐轴 wrap，roll/yaw 仍可能出现等价表示切换。
- `control_hz` 高于模型 action 频率时，轨迹被压缩执行，看起来会更猛。

排查顺序：先用 `--inference_delay_mode fixed --inference_delay_steps 0` 排除 delay；再用 `--chunk_blend_horizon_steps 14` 和 `--action_smoothing ema --action_ema_alpha 0.15` 降低 chunk/action 跳变；如果仍抽搐，需要在控制端加每 tick 最大位移/最大旋转限幅，或改成 quaternion/Slerp 处理姿态。

控制端限幅默认关闭。若要先压住明显 spike，可从保守值开始：

```bash
--max_position_step_m 0.005 \
--max_rotation_step_deg 5 \
--max_gripper_step 0.05
```

上一次修复后的 step 语义：

- 没有任何 action 可执行时，control thread 只等待，不推进 `control_step`。否则启动阶段 policy 还没返回，`control_step` 会先跑到 10/20，首个 chunk 回来就被当成过期动作丢掉。
- `held=True missing=True` 且 buffer 没有 future action 时，只重复上一条命令维持机器人，不推进 `control_step`。否则一次 buffer 断流会让 action 时间轴继续向前跑，新 chunk 回来又会被大量跳过。
- 空 buffer 恢复填充时不应用 `min_buffer_steps` 冻结区。否则恢复后的第一条可用 action 会从未来 step 开始，而 control thread 卡在当前 step。
- 如果 buffer 中确实只有 future action，control thread 会打印 `future action gap fallback` WARN，并在 hold 当前命令的同时推进到第一个可用 step。这是异常兜底，不是正常 chunk 衔接路径。
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

## RTC FM Inpainting

现有 `ActionBuffer` 的 delay/merge/blend 是 rollout 侧补偿；`RTC FM inpainting` 是 policy server 侧补偿。开启后，server 会在 flow-matching 采样时使用上一段 raw normalized action chunk 作为条件，生成下一段 chunk 时显式分三段：

- `frozen d`：由推理延迟导致必然会继续执行的旧 action，权重 `1`，server 不返回给 client。
- `soft region`：上一段 chunk 仍覆盖的未来 action，按指数权重引导，但允许模型更新。
- `free tail s`：上一段 chunk 覆盖不到的尾部 action，权重 `0`，完全重新生成。

server 启动：

```bash
python scripts/serve_policy.py \
  --rtc-chunk-conditioning \
  --rtc-soft-horizon-steps 5 \
  --rtc-free-tail-steps 5 \
  policy:checkpoint \
  --policy.config <config> \
  --policy.dir <checkpoint>
```

async client 开启：

```bash
--rtc_chunk_conditioning \
--rtc_soft_horizon_steps 5 \
--rtc_free_tail_steps 5
```

FastTouch async 和 xArm async 都支持这组 RTC client 参数。RTC 在 policy server 的 flow-matching 采样层工作，与机器人执行 backend 独立：xArm 可以将 RTC 与 `position`、`servo`、`online_cartesian` 或 `plan-servo` 任一模式组合使用。

默认 `--rtc_delay_steps -1` 表示 client 使用上一轮实测 latency steps 作为本轮 RTC 的 `d`。如果要固定调试，可显式传：

```bash
--rtc_delay_steps 2
```

当 server 返回 `rtc.applied=true` 时，返回的 `actions[0]` 已经是 `request_step + d` 的第一个可用 action，client 会用 `action_base_step` 写入 buffer，并把 merge latency 置 `0`。当 `rtc.applied=false`（首个 chunk、reset、无 overlap、`d >= H` 等）时，client 退回原有 `inference_delay_mode` 跳过过期 action。

## FastTouch 新版 SDK Joint Waypoints Mode

同步 FastTouch rollout 支持可选的新版 SDK 轨迹执行后端。默认仍使用原来的逐 action 笛卡尔下发：

```bash
python scripts/rollout/pi0_rollout_client_fasttouch_rpy.py \
  --description "<task>" \
  --execution_backend cartesian_raw
```

如果 Startouch SDK 提供 `solve_ik()`、`move_joint_waypoints()` 和 `get_joint_positions()`，可以启用整段 IK + 关节路点规划：

```bash
python scripts/rollout/pi0_rollout_client_fasttouch_rpy.py \
  --description "<task>" \
  --execution_backend joint_waypoints \
  --model_infer_action_dt 0.05 \
  --ik_retries 5 \
  --ik_retry_sleep_s 0
```

`joint_waypoints` mode 会将本次 action slice 的 TCP/RPY 目标转换为法兰 pose，逐点调用 `solve_ik()`，再将整段关节轨迹交给 `move_joint_waypoints()` 执行。双臂会并发执行，夹爪按轨迹时长同步下发。

- `--model_infer_action_dt`：每个模型 action waypoint 对应的时间间隔，默认 `0.05s`。旧参数名 `--trajectory_dt` 仍可用。
- `--joint_waypoint_speed_percent`：`>0` 时传给 SDK 的 `speed_percent`；默认 `-1`，使用 `model_infer_action_dt * waypoint_count` 计算 `time_sec`。
- `--ik_retries`：每个 waypoint 首次失败后的额外 IK 重试次数，默认 `5`。
- `--ik_retry_sleep_s`：IK 重试间隔秒数，默认 `0`。

同步 `joint_waypoints` mode 仍然适合整段阻塞执行，不会在轨迹执行中间做 mask tracking-only。对于 async rollout，不要直接使用这个阻塞接口；新版 Startouch SDK `0.1.6` 提供了专门的非阻塞滚动 chunk planner。

### Async FastTouch Joint Waypoint Chunk Planner

FastTouch async rollout 默认仍使用逐 tick 笛卡尔下发：

```bash
python scripts/rollout/pi0_rollout_client_fasttouch_rpy_async.py \
  --description "<task>" \
  --execution_backend cartesian_raw
```

如果 Startouch SDK 提供 `solve_ik()`、`get_joint_positions()` 和 `update_joint_waypoint_chunk_with_gripper()`，可以开启 SDK 原生滚动规划：

```bash
python scripts/rollout/pi0_rollout_client_fasttouch_rpy_async.py \
  --description "<task>" \
  --execution_backend joint_waypoint_chunk \
  --model_infer_action_dt 0.05 \
  --chunk_switch_delay_sec 0.05 \
  --planner_chunk_max_waypoints 20 \
  --ik_retries 5
```

每次 policy response 合并进 `ActionBuffer` 后，client 会读取连续的未来 action 窗口，将 TCP/RPY 转成法兰 pose，逐 waypoint 以当前关节和上一 IK 解为 seed 求解关节目标，再调用 SDK 的 `update_joint_waypoint_chunk_with_gripper()`。该调用非阻塞：SDK 保留活动轨迹的一段短前缀，然后重新规划并替换未来 suffix。控制线程仍按 `control_hz` 推进绝对 step 和记录日志，但不会重复发送笛卡尔 raw command。

- `--control_hz`：`ActionBuffer` 消费模型 action 和推进 rollout 绝对 step 的频率，默认 `20Hz`。
- `--inference_interval_steps`：每隔多少个 rollout step 尝试请求一次新 policy chunk，默认 `8`；`20Hz` 下约为每 `0.4s` 一次。
- `--model_infer_action_dt`：每个 VLA action waypoint 在 SDK planner 时间轴上的间隔，默认 `0.05s`。通常应设为 `1 / control_hz`；旧参数名 `--trajectory_dt` 仍可用。
- `--joint_waypoint_speed_percent`：`>0` 时改用 SDK 速度比例；默认 `-1`，使用 `model_infer_action_dt * waypoint_count`。
- `--chunk_switch_delay_sec`：安装新 chunk 时保留旧活动轨迹前缀的时间，默认 `0.05s`。该前缀不可被新规划覆盖；旧参数名 `--joint_chunk_switch_delay_sec` 仍可用。
- `--planner_chunk_max_waypoints`：每次滚动规划从 `ActionBuffer` 读取的最大连续未来 action 数，默认 `20`。结合默认 dt，对应最长 `1.0s` 规划时域；旧参数名 `--joint_chunk_max_waypoints` 仍可用。
- `--ik_retries`、`--ik_retry_sleep_s`：单个 waypoint IK 失败后的重试策略。

该 backend 和 server-side RTC FM inpainting 可以同时开启。RTC 在模型采样层处理 frozen latency prefix、soft-guided overlap 和 free tail；SDK planner 在执行层保留不可抢占前缀并平滑替换后续轨迹。两层 latency 参数需要根据真机日志联合调节。

默认参数下，三个时间轴如下：

```text
control_hz=20
  -> 每 0.05s 消费一个模型 action
inference_interval_steps=8
  -> 理想情况下每 0.4s 请求一次新 policy chunk
planner_chunk_max_waypoints=20, model_infer_action_dt=0.05
  -> 每次最多安装 1.0s 的未来轨迹，SDK 内部再高频插值执行
```

### Async FastTouch External Plan Servo

Startouch SDK `0.1.6` 还提供 `get_joint_velocities()` 和 `set_joint_raw(q_target, dq_target)`。如果需要显式使用当前速度接续新 chunk，可以绕过 SDK 内置的零边界速度 planner，改用外部 planner：

```bash
python scripts/rollout/pi0_rollout_client_fasttouch_rpy_async.py \
  --description "<task>" \
  --execution_backend plan_servo \
  --control_hz 20 \
  --model_infer_action_dt 0.05 \
  --planner_chunk_max_waypoints 20 \
  --plan_servo_hz 100 \
  --rtc_chunk_conditioning
```

`plan_servo` 逐 waypoint 调用 SDK `solve_ik()`，从安装时刻实时读取 `q/dq`，构造 cubic Hermite joint 轨迹，再由独立线程调用 `set_joint_raw(q, dq)` 高频透传。IK 和 planner 计算期间旧轨迹继续发送；新轨迹 ready 后会按等待期间已经推进的 control step 丢掉过期 waypoint，再从实时 `q/dq` 接管。

- `--plan_servo_hz`：raw joint target 下发频率，默认 `100Hz`。
- `--plan_servo_max_joint_velocity_rad_s`：可选 joint 最大速度校验阈值；默认 `0` 表示关闭。
- `--plan_servo_max_joint_acceleration_rad_s2`：可选 joint 最大加速度校验阈值；默认 `0` 表示关闭。
- `--plan_servo_stale_timeout_s`：轨迹结束后允许保持末点的时长，默认 `0.5s`。

如果 `ActionBuffer` 意外出现 future gap，新轨迹会短暂 hold 实时起点等待对应 step，并输出 `future-gap fallback active` warning。正常 RTC 连续 chunk 不应触发该 fallback。每轮 planner 实际耗时会折算成 `planner_latency_steps`，与 policy latency 相加后写回下一轮 RTC delay。

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

`scripts/rollout/pi0_rollout_client_xarm_rpy_async.py` 额外支持：

```bash
--xarm_control_mode position
--xarm_control_mode servo
--xarm_control_mode online_cartesian
--xarm_control_mode plan-servo
```

- `position`：默认，使用 `set_position(..., wait=False)`，兼容性最高。
- `servo`：使用 `set_servo_cartesian`，进入脚本后切 `mode=1`，退出/reset 时恢复 `mode=0`。延迟更低，但需要 SDK/固件支持，且更依赖稳定的 action stream。
- `online_cartesian`：使用 `set_position(..., wait=False)`，进入脚本后切 `mode=7`，退出/reset 时恢复 `mode=0`。每次新目标会中断当前目标并由控制器在线规划，适合希望降低 chunk 切换顿挫但不想直接使用 servo 的场景。
- `plan-servo`：使用 `get_inverse_kinematics(ref_angles=上一点 IK 解)` 逐 waypoint 求解，再从实时 `q/dq` 构造 cubic Hermite joint 轨迹，最后通过独立高频线程调用 `set_servo_angle_j()`。它显式使用模型 action 时间尺度，比 `online_cartesian` 更适合精细操作。

推荐先测试 xArm 原生在线规划：

```bash
python scripts/rollout/pi0_rollout_client_xarm_rpy_async.py \
  --description "<task>" \
  --xarm_control_mode online_cartesian \
  --control_hz 10 \
  --rtc_chunk_conditioning \
  --rtc_soft_horizon_steps 5 \
  --rtc_free_tail_steps 5
```

需要显式时间规划时，使用 `plan-servo`：

```bash
python scripts/rollout/pi0_rollout_client_xarm_rpy_async.py \
  --description "<task>" \
  --xarm_control_mode plan-servo \
  --control_hz 20 \
  --plan_servo_hz 100 \
  --plan_servo_model_action_dt -1 \
  --plan_servo_chunk_max_waypoints 20 \
  --plan_servo_stale_timeout_s 0.5 \
  --rtc_chunk_conditioning \
  --rtc_soft_horizon_steps 5 \
  --rtc_free_tail_steps 5
```

- `--plan_servo_hz`：joint sample 下发频率，默认 `100Hz`。
- `--plan_servo_model_action_dt`：模型相邻 action waypoint 的时间间隔。默认 `-1`，自动使用 `1 / control_hz`；例如 `20Hz -> 0.05s`。
- `--plan_servo_chunk_max_waypoints`：每次重规划读取的最大连续未来 action 数，默认 `20`。
- `--plan_servo_max_joint_velocity_rad_s`：可选 joint 最大速度校验阈值；默认 `0` 表示关闭。
- `--plan_servo_max_joint_acceleration_rad_s2`：可选 joint 最大加速度校验阈值；默认 `0` 表示关闭。
- `--plan_servo_stale_timeout_s`：轨迹结束后允许保持末点的时长，默认 `0.5s`；超时后 sender 停止发送陈旧 joint target。

`plan-servo` 的首段速度来自实时 `get_joint_states()` 返回的 `dq`，而不是强制设为零。中间 waypoint 速度由相邻 IK 点估计，最后一个 waypoint 速度收敛到零。IK 和 planner 计算期间旧轨迹继续发送；新轨迹 ready 后会丢掉等待期间已经过期的 waypoint，再从实时 `q/dq` 接管。新计划 IK 失败或超过显式速度/加速度阈值时，client 会保留上一条有效轨迹。

部分 xArm SDK 版本的 `get_inverse_kinematics()` 不接受 `limited` 或 `ref_angles`。client 会在 SDK 明确报告 optional keyword 不支持时打印 WARN 并降级重试；支持这些参数的版本仍优先使用完整 IK 调用。

如果 `ActionBuffer` 意外出现 future gap，新轨迹会短暂 hold 实时起点等待对应 step，并输出 `future-gap fallback active` warning。正常 RTC 连续 chunk 不应触发该 fallback。每轮 planner 实际耗时会折算成 `planner_latency_steps`，与 policy latency 相加后写回下一轮 RTC delay。

当前 planner 是轻量 cubic Hermite 实现，不是完整 jerk-limited OTG。真机验证稳定后，如果还需要更严格的 jerk 限制，可以把 planner 替换为 Ruckig；RTC、IK 和 sender 线程无需重写。

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

## Debug 可视化

启用 debug 后会写出：

```text
observations.jsonl
actions.jsonl
executions.jsonl
chunks.jsonl
summary.json
```

每条 observation/action/execution 都带 monotonic timestamp、control step、chunk id、merge type、command action 和可选真实 pose readback。服务端会把 client 发来的 `__async_rollout` metadata 原样 echo 到 `async_rollout_echo`，用于对齐 request/response。

生成图：

```bash
python scripts/rollout/plot_async_rollout_debug.py \
  --debug-dir /tmp/openpi_async_debug/xarm_run01
```

输出到 `plots/`：

- `timeline.png`：latency、delay、buffer、missing/held。
- `action_delta.png`：每 tick position/rpy/gripper delta。
- `command_vs_actual.png`：command pose 和 readback pose 对比。
- `chunk_merge.png`：chunk/action/target_step 的 inserted/blended/skipped 分布。
- `summary.md`：关键统计和 top spike events。

常见问题：

- 只打印一行 infer 后不动：看 `inserted` 是否为 `0`。若为 `0`，降低 `--control_hz`、使用 `--inference_delay_mode fixed --inference_delay_steps 0`，或增大 `--action_end`。
- 一顿一顿：看 `missing=True` 是否频繁出现。若频繁出现，降低 `--control_hz` 或减小 `--inference_interval_steps`。
- `skipped` 很大并且随后 `buffer=0`：推理耗时已经吃掉了 chunk 后半段。例如 `control_hz=20` 且一次推理耗时 `2.5s`，控制线程已经前进约 `50` 步；如果 `action_end` 只有 `49`，这个 chunk 基本没有未来 action 可用。先用 `--control_hz 10` 或增大 `--action_end` 验证。
- 边界跳：增大 `--chunk_blend_horizon_steps`。
- normal 同步版 chunk 切换回退：先用 `--enable_interp --dt 0.05` 限速；如果仍有旧 chunk 滞后，切 async + `--xarm_control_mode online_cartesian`。
- `control_hz=20` 明显太快：说明 action 时间语义更接近 10Hz。先用 `--control_hz 10 --xarm_control_mode online_cartesian --chunk_blend_horizon_steps 10 --action_smoothing ema --action_ema_alpha 0.25`，再按抖动情况微调。
