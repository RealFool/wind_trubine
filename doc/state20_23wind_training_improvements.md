# state20 相对旧 23 风况训练的改进说明

更新时间：2026-09-22

## 1. 对比对象

本文比较两个相同规模的 23 风况、4 轮、600 s 训练：

| 项目 | 旧实验 | state20 实验 |
|---|---|---|
| 旧路径 | `RL/experiments/b4_residual_td3_bc_23wind_4round_600s_20260908_230201` | `RL/experiments/b4_residual_td3_bc_state20_23wind_4round_600s_diagnostic_20260919_121426` |
| 数据合同 | `b4_centered_residual_bc_v1` | `b4_centered_residual_bc_state20_v2` |
| observation | 历史 obs12 | 20 维 state-aware observation |
| torque request high | 21,000,000 Nm | 21,586,451.33303 Nm |
| Actor Q-health target-Q threshold | 10.0 | 20.0 |
| Actor optimizer health gate | 未配置 | 已配置，固定 BC batch 健康检查 |
| execution replay contract | 未配置 | 已配置，按执行模式区分 replay source |
| logging | compact/有限摘要 | `legacy_long_form` 完整长表 |

这次改进解决的是 Actor 更新被大面积阻断、合法 baseline 被误记为 fallback、replay 数据断流等训练稳定性问题。它不等于最终功率-载荷策略验收已经通过。

## 2. 旧实验的问题链

旧实验的 Critic 本身没有发生数值健康失败：Critic Q-health degraded 为 0，Critic guard rate 为 0。但 Actor gate 使用 `max_target_q_abs_mean=10.0`，而 target-Q 后期超过了该阈值。

旧实验的结果：

```text
总控制步数                         2,759,930
总 update events                  68,998
总 fallback                       711,751 (25.79%)
Q-health blocked fallback          689,936
Actor 实际更新                     11,399
Actor gate open events             46,000 / 68,998
最终 Actor gate                    closed
存入 replay 的 transition          2,048,109
拒绝 transition                    711,821
```

旧代码路径形成了以下级联：

```text
target-Q 超过 Actor gate 阈值
  -> Actor Q-health gate 关闭
  -> Actor update 被阻断
  -> 合法 B4 baseline 被记为 fallback
  -> skip_safety_fallback 拒绝 replay
  -> 后期有效在线数据减少
  -> 第 4 轮 Actor 执行和训练被遮挡
```

这不是普通的物理安全事故，而是训练健康 gate、控制执行和 replay filter 之间的耦合问题。

## 3. state20 的关键改进

### 3.1 使用一致的 state20 合同

state20 不再拼接旧 obs12 checkpoint、旧 normalization 和新配置，而是整套绑定：

```text
RL/data/state_action_envelope_3_to_25_v1.json
RL/data/state_aware_control_observation_schema_v1.json
RL/data/resolved_control_contract_state20_600s_v1.json
RL/data/region_aware_joint_baseline_state20_3_to_25_v3.json
RL/data/b4_centered_residual_bc_state20_v2/manifest.json
当时使用的 state20 v2 checkpoint（当前已退役清理；本报告记录仅用于历史对照）
```

resolved contract fingerprint：

```text
9b1618eedc9e8c128e7220e56930350c4991a6db4ecad3922014d3ee5da5d142
```

checkpoint、B4 baseline、observation width、previous-action normalization 和 predictor interface 必须匹配同一个 fingerprint。

### 3.2 修复动作范围和动作变化范围

旧合同：

```text
action_high       = [21,000,000 Nm, 0.43 rad]
action_delta_high = [15,000 Nm, 0.0003 rad]
```

state20 合同：

```text
action_high       = [21,586,451.33303 Nm, 0.43 rad]
action_delta_high = [90,000 Nm, 0.000698 rad]
```

21 MNm 截断会损失 PID demand label 的有效范围。state20 使用已审计的 engineering request high，并继续经过 safety projection、rate limit 和 float64 wire serialization。

### 3.3 新增 Actor optimizer health gate

state20 增加：

```json
{
  "schema_version": "actor_optimizer_health_gate_v1",
  "enabled": true,
  "recovery_consecutive_healthy_critic_updates": 200,
  "require_fixed_offline_batch_health_check": true,
  "fixed_offline_batch_size": 256
}
```

它的职责是只冻结 Actor optimizer update：

- Critic 数值检查或固定 BC batch 检查不健康时，暂时不更新 Actor。
- 连续 200 个健康 Critic update 后恢复 Actor update。
- 不把“暂时不更新 Actor”自动变成“禁止执行有效 Actor policy”。
- `actor_execution_allowed` 单独控制 policy 是否进入 safety controller。

关键代码位置：[train_rl.py](../RL/train_rl.py#L631-L640)、[train_rl.py](../RL/train_rl.py#L1565-L1578)、[critic_q_health.py](../RL/utils/critic_q_health.py#L365-L505)。

### 3.4 Actor Q-health threshold 从 10 提高到 20

state20 使用：

```text
train.actor_q_health_gate.max_target_q_abs_mean = 20.0
agent.max_target_q_abs_mean = 20.0
```

两者职责不同：

- `train.actor_q_health_gate.max_target_q_abs_mean` 是 Actor 趋势 gate 条件。
- `agent.max_target_q_abs_mean` 是 Critic update 的硬数值 guard。

因此不是取消 Critic 保护，而是避免 target-Q 尚未达到 Critic 硬 guard 时过早关闭 Actor gate。

### 3.5 execution-mode replay contract

对应配置键为 `train.execution_replay_contract`，schema version 为
`execution_mode_replay_contract_v1`。

先把 replay 想成“训练记忆”。每个 replay transition 大致记录：

```text
当时看到的状态 obs
当时实际执行的动作 action
得到的 reward
执行后看到的状态 next_obs
以及这一步动作是由谁/什么安全模式产生的 source tag
```

旧实验的问题是：只要 `control_fallback_active=1`，旧的
`skip_safety_fallback=true` 就会直接丢弃这条 transition。但 fallback 不一定代表
Actor 出错，例如转速过低时，控制器主动进入 recovery hold 是一种有意的保护动作。
旧逻辑把“有意的低速保护”和“Actor/Q-health 真出问题”混成了同一种失败。

state20 明确区分以下执行模式：

```text
actor
explicit_baseline_stage
actor_execution_rollback_baseline
low_speed_recovery_hold
```

它们分别可以这样理解：

| execution mode | 通俗含义 | 例子 |
|---|---|---|
| `actor` | Actor 的动作正常被发送 | Actor 给出 torque/pitch residual，经过 projection 后执行 |
| `explicit_baseline_stage` | 系统有意让 baseline 接管 | 启动、停止或明确要求 baseline 的阶段 |
| `actor_execution_rollback_baseline` | Actor 结果不适合直接执行，退回安全 baseline | Actor 输出非法、不可执行或需要 rollback，但环境这一步仍有效 |
| `low_speed_recovery_hold` | 转速过低时暂时保持/卸载，帮助风机恢复 | 风况 3、4 的低速保护与 recovery hold |

因此，state20 的 replay 流程是：

```text
1. 先决定这一步实际使用哪一种 execution mode。
2. 把实际发送的 action 和对应 source tag 一起写入 transition。
3. 如果发生 post-step timeout、invalid、unsafe stop 或 condition boundary，拒绝保存。
4. 如果只是合法的 recovery hold，且 post-step 有效，则允许按对应 source 保存。
5. 采样器可以保留并返回 source tag，让程序知道这条数据来自 Actor、baseline、rollback 还是 recovery。

这里要特别澄清：**当前 tag 不是 Actor/Critic 的输入特征**。它不会自动让模型“理解这
是 recovery 数据”，也不会在 `TD3Agent.update()` 里切换成另一套 loss。当前 state20
代码对 tag 的使用分成三层：

```text
执行时：决定这条 transition 是否有资格进入 replay
采样时：保留 source tag；如果配置 source_max_fraction，可限制某类 source 在 batch 中的比例
更新时：TD3 当前只使用 obs/action/reward/next_obs/done，source tag 不直接进入网络或 loss
```

当前 state20 配置没有设置 `condition_replay.source_max_fraction`，所以 source tag
主要用于：

- 在写入前区分哪些 execution mode 可以作为 Critic replay；
- 统计每个 condition 的 Actor、baseline、rollback、recovery 数量；
- 为后续 source 配额、分层采样或单独评估提供依据；
- 审计“这次训练到底学到了哪些实际执行数据”。

以 state20 round4 checkpoint 的 replay snapshot 为例，920,000 条存量 transition 的
实际构成为：

| source tag | 存量条数 | 存量占比 |
|---|---:|---:|
| `actor` | 910,827 | 99.0029% |
| `low_speed_recovery_hold` | 9,173 | 0.9971% |
| `explicit_baseline` | 0 | 0% |
| `actor_execution_rollback_baseline` | 0 | 0% |
| `online_bladed` | 0 | 0% |

| source tag | 存量条数 | 存量占比 |
|---|---:|---:|
| `actor` | 910,827 | 99.0029% |
| `recovery` | 9,173 | 0.9971% |
| `baseline` | 0 | 0% |
| `rollback` | 0 | 0% |

这些是 replay **存量占比**，不是一个 batch 的实际抽样占比。当前 sampler 的行为是：

- 先保证 23 个 condition 都有机会进入 batch；
- 每个 condition 至少抽取 5 条，batch size 为 128；
- 对某个 condition，如果同时存在 `actor` 和 `low_speed_recovery_hold`，就在这两类中近似均匀抽取；
- 如果该 condition 只有 `actor`，就只能抽 `actor`；
- 当前没有 `source_max_fraction`，所以没有显式的 25%/50% source 配额。

本次 round4 中 23 个 condition 里有 16 个同时包含 Actor 和 recovery，7 个只有 Actor。
因此按当前 sampler 粗略估计，一个 batch 的 source 比例约为：

```text
actor                       约 65.2%
low_speed_recovery_hold     约 34.8%
explicit_baseline           0%
actor_execution_rollback    0%
```

这只是当前采样算法和当前存量结构共同产生的近似值，不是正式设计目标；而且每个 batch
会有随机波动。最重要的是：即使 `low_speed_recovery_hold` 在存量中只有约 1%，它在
包含该 source 的 condition 内仍可能获得接近一半的抽样机会。这正是 source tag 会影响
训练数据分布的地方，但 TD3 的网络和 loss 仍然不直接读取 tag。

因此，tag 的直接价值不是让模型立刻学会四种模式，而是防止不同性质的数据在进入训练前
被混为一类或被全部误删。若希望模型真正区别这些模式，还需要额外设计，例如把 execution
mode 编码加入 observation、配置 source-aware sampling，或对不同 source 使用明确的
loss/权重；当前实现尚未做这些扩展。

这就是为什么 state20 的 `queue_condition_rejected_fallback_count` 不能简单等同于全部
fallback：一部分 low-speed recovery transition 是有效的 replay 数据。反过来，旧实验
把大量 Q-health blocked baseline 统一判为 fallback，又用 fallback filter 全部拒绝，形成
了“Actor 不更新 -> 数据更少 -> 后期更难恢复”的断流。

关键代码在 `_should_store_transition()`：启用 execution replay contract 后，代码先检查
post-step outcome 和 execution mode，而不是先用一个笼统的 `skip_safety_fallback` 丢弃
所有 fallback。这样解决的是“数据分类错误和 replay 断流”，不是取消安全保护。

### 3.6 state20 paired PID reference

先把 PID reference 想成“这一次 RL 测试对应的标准答案”。所谓 paired，不是只保存一个
按风速查表的数字，而是把同一个 condition 的功率、三片叶根 My、原始 CSV 和时间窗绑在
同一条 reference entry 中。

旧实验也有 PID reference，但主要是旧的、分开的 legacy reference：功率 reference 一套、
叶根 peak reference 另一套。这样容易出现“功率拿了一套 reference、载荷又拿另一套
reference，时间窗或来源不完全一致”的问题。它不一定每一步都错，但审计时很难证明
RL 和 PID 是严格一一对应的。

state20 使用：

```text
RL/data/paired_pid_power_load_reference_3_to_25_600s_v2.json
condition_id + condition_instance + exact 600 s window
```

可以把一次 condition 的比较理解为：

```text
RL condition_id=8, sim_time=100.00 s
  对比
PID condition_id=8, 同一 reference entry, sim_time=100.00 s
```

而不是：

```text
RL 当前风速大约 8 m/s
  对比
某个按风速插值得到的 PID 平均值
```

训练时主要使用三类信息：

1. `condition_id`：这是哪个固定 Bladed case/风况。
2. `sim_time`：当前仿真进行到了这个 case 的哪个时间点，用来查对应 PID power。
3. 同一 entry 的三片 Blade root My peak reference：用来计算每片叶根当前峰值是否超过 PID 对照。

`condition_instance_id` 用来区分同一个 condition 在不同 round 或不同 queue slot 中的
一次具体出现；它不是把 wind8 误认为 wind16，而是给重复执行的 wind8 加上本次运行身份。

这样做的好处是：

- RL reward 的 power 对照和 load 对照来自同一个配对来源；
- 600 s 训练不会误用另一时间窗的 180 s peak；
- 每个 condition 的 energy、三片 My peak 和 reference hash 可以一起审计；
- 如果 condition 或时间戳错了，应该停止该 condition，而不是继续写 replay。

需要区分两种 identity 强度：

- 本次 state20 训练是 diagnostic prefix identity，只验证前缀身份，适合开发诊断；
- 正式 release 需要 strict full-trace identity，验证完整 Time + I_AverageWindSpeed trace。

所以 paired PID reference 解决的是“RL 与 PID 是否公平、可追溯地比较”；它本身不等于
已经完成正式 release，也不保证 Actor 一定满足 My 硬约束。

### 3.7 完整 long-form telemetry

state20 配置使用 `legacy_long_form`，保存：

```text
metrics.csv
queue_train_step_metrics.csv
queue_train_update_metrics.csv
queue_condition_metrics.csv
queue_condition_checkpoint_metrics.csv
queue_round_*_summary.json
queue_training_summary.json
```

因此 Actor gate、fallback、每轮 update、condition energy/My 和 replay rejection 都可以事后核对。

## 4. 结果对比

| 指标 | 旧实验 | state20 实验 | 变化 |
|---|---:|---:|---:|
| 完成 rounds / cases | 4 / 92 | 4 / 92 | 保持 |
| Critic degraded | 0 | 0 | 保持 |
| 总控制步数 | 2,759,930 | 2,759,788 | 同规模 |
| update events | 68,998 | 68,994 | 同规模 |
| Actor 实际更新 | 11,399 | 16,603 | +45.65% |
| Actor gate open events | 46,000 | 68,794 | +49.55% |
| Q-health blocked update events | 22,998 | 200 | -99.13% |
| 总 fallback | 711,751 (25.79%) | 93,269 (3.38%) | -86.90% |
| replay rejected fallback | 711,729 | 70,626 | -90.08% |
| stored transitions | 2,048,109 | 2,689,070 | +31.30% |
| skipped / retried conditions | 0 / 0 | 0 / 0 | 保持 |

state20 关键证据：

```text
Actor optimizer gate open        68,795 / 68,994 = 99.71%
Actor Q-health gate open         68,794 / 68,994 = 99.71%
Critic degraded                 0
Target-Q max                   19.0346 < 20
第4轮 Actor updates             5,560
```

旧实验第 4 轮不是完全没有 Actor update，而是 gate 关闭后大量合法 Actor/B4 execution 被标记为 fallback。state20 第 4 轮仍保持 Actor update 和 Actor execution。

## 5. 已解决与未解决

### 已解决或显著缓解

1. **Actor training interruption**：Q-health blocked update 从 22,998 降至 200。
2. **Gate-blocked fallback cascade**：旧实验的 689,936 个 Q-health blocked fallback step 不再出现；state20 的 policy-blocked Q-health fallback 为 0。
3. **Replay data starvation**：stored transition 从 2.05M 增加到 2.69M。
4. **Contract mismatch**：state20 fingerprint、manifest 和 checkpoint metadata 阻断 obs12/v1 与 state20/v2 混用。
5. **可审计性**：从 compact aggregate 改为完整 long-form metrics。

### 尚未解决

1. **策略性能验收仍未通过**：平均能量约比 PID 低 0.712%；三片 Blade root My 平均相对 PID 高约 4.07% / 4.78% / 5.02%。
2. **低风速 fallback 仍存在**：state20 实际 fallback 93,269 步，主要是 startup/stopped baseline 和 low-speed recovery hold，集中在 wind3/wind4。
3. **Actor observation 仍不直接包含三片 My 峰值和 headroom**，因此尚不能宣称硬载荷约束。
4. **仍是 diagnostic**：production predictor、load guard calibration、strict full-trace identity 和 release manifest 尚未完成。

## 6. 结论

旧实验的核心问题是训练健康 gate、控制执行和 replay filter 耦合，导致 target-Q 增长后 Actor gate 关闭，继而形成假性 fallback 和 replay 断流。state20 的修复是一个闭环：

```text
state20 contract/fingerprint
  -> 一致的 action / observation / paired reference
  -> 独立 Actor optimizer health gate
  -> Actor target-Q threshold 与 Critic hard guard 分层
  -> execution-mode replay source 分离
  -> long-form telemetry
  -> Actor update 与 policy execution 不再被同一故障级联切断
```

可复核证据：

- [旧实验分析](../RL/experiments/b4_residual_td3_bc_23wind_4round_600s_20260908_230201/b4_residual_smoke_analysis.json)
- [旧实验配置](../RL/experiments/b4_residual_td3_bc_23wind_4round_600s_20260908_230201/config.json)
- [state20 实验配置](../RL/experiments/b4_residual_td3_bc_state20_23wind_4round_600s_diagnostic_20260919_121426/config.json)
- [state20 训练汇总](../RL/experiments/b4_residual_td3_bc_state20_23wind_4round_600s_diagnostic_20260919_121426/queue_training_summary.json)
- [state20 update 长表](../RL/experiments/b4_residual_td3_bc_state20_23wind_4round_600s_diagnostic_20260919_121426/queue_train_update_metrics.csv)
- [state20 step 长表](../RL/experiments/b4_residual_td3_bc_state20_23wind_4round_600s_diagnostic_20260919_121426/queue_train_step_metrics.csv)
- [控制策略验收口径](控制策略指标_更新版.md)
