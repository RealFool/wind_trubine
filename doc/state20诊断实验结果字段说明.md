# state20 23 风况 × 4 轮 × 600 秒诊断实验：结果文件与字段说明

## 先看这三点

1. **这是 2026-09-19 的历史诊断实验，不是正式验收。** 对应目录为 `RL/experiments/b4_residual_td3_bc_state20_23wind_4round_600s_diagnostic_20260919_121426/`；以该目录的 `config.json`、实际 CSV 表头及小表字段为准。后续源码可能新增字段、规则和严格身份校验，不能倒填到这次实验。
2. **训练 CSV 是“长表”。** 每行只记一个数值，四列恒为 `stage,step,name,value`。不能把 `name` 当作文件表头，也不能把每一行当作独立的仿真步。`metrics.csv` 是各阶段的合集，按阶段命名的 CSV 是对应阶段的副本；不能把两者相加。
3. **不要在编辑器中打开或整表加载大日志。** `metrics.csv` 约 40.8 GB，`queue_train_step_metrics.csv` 约 40.6 GB，`queue_train_update_metrics.csv` 约 181 MB；查看字段可从本说明、小型 JSON、`queue_condition_metrics.csv`（约 252 KB）及表头入手。本文对大表仅依据表头、有界抽样和产生日志的代码解释字段族；**抽样没见到某字段，不等于整个大表不存在它**。本说明没有运行训练、推理或 Bladed，也没有加载 `.pt` 权重。

常用单位：电功率 W（$10^6\,\mathrm W=1\,\mathrm{MW}$），能量 J（$3.6\times10^9\,\mathrm J=1\,\mathrm{MWh}$），扭矩/叶根 My Nm（$10^6\,\mathrm{Nm}=1\,\mathrm{MNm}$），桨距 rad，时间 s 或 ms。数值型开关常用 `1.0` 表示是、`0.0` 表示否；比例 `0.05` 是 5%，只有以 `_percent` 命名的字段通常直接按“百分数点数值”阅读。

## 1. 文件去哪儿找

| 目录中的文件/位置 | 内容；优先怎样看 |
| --- | --- |
| `config.json` | 本次运行冻结的配置：风况、轮数、控制边界、训练调度、奖励、日志形式和诊断身份口径；先读它。 |
| `queue_training_summary.json` | 全程累计的步数、回放数、更新数、跳过/重试、过程有效性；最方便的总览。 |
| `queue_round_1_summary.json` ～ `queue_round_4_summary.json` | 各轮结束时的**累计值**、该轮对偶约束更新及各风况回放存量；轮 4 数值不能再与轮 1 相加。 |
| `bc_prediction_preview_summary.json` | BC 预测预览样本数和归一化残差误差；不是闭环电量或载荷误差。 |
| `queue_condition_metrics.csv` | 每个已完成工况的汇总，实际有 42 种 `name`，各 92 条（23 风况 × 4 轮）；优先读它。 |
| `queue_condition_checkpoint_metrics.csv` | 每个工况进度检查点，实际 7 种 `name`，各 92 条。 |
| `queue_train_step_metrics.csv` | 逐控制步的环境/动作/奖励/UDP 长表，体积巨大。 |
| `queue_train_update_metrics.csv` | Critic/Actor 优化及健康门控的逐更新长表；不是逐仿真步表。 |
| `metrics.csv` | 上述阶段日志的总汇；非新增的一份独立实验。 |
| `checkpoints/` | 92 个工况进度检查点 + 4 个整轮模型检查点，共 96 个 `.pt`。名字标记轮次、风况、实例和检查点序号；这里不读取权重。 |
| `figures/` | 该目录未见训练图；缺图不代表没有训练日志。 |
| `inference/` | 风 8、16、24 各 4 轮的独立 `wind<风况>_round<轮次>_retry/` 推理文件夹，以及各风况一个 `wind<风况>_round_progress_retry/` 比较文件夹；它们**不是**另外 12 个训练工况。 |

### 长表的阅读方法

四列长表通用字段 `stage`、`step`、`name`、`value` 出现在 `metrics.csv` 以及 `queue_train_step_metrics.csv`、`queue_train_update_metrics.csv`、`queue_condition_metrics.csv`、`queue_condition_checkpoint_metrics.csv` 中。具体指标按 `stage` 区分；合并总表与分阶段副本不能重复累计。

| 列 | 意思 |
| --- | --- |
| `stage` | 记录来源，例如 `queue_train_step`（一步）、`queue_train_update`（一次优化）、`queue_condition`（完成一个工况）、`queue_condition_checkpoint`（保存检查点）；并非“训练轮次”。 |
| `step` | 从本次训练开始累计的**全局控制步编号**。更新事件和工况结束事件可能共用同一个 `step`；不是仿真时间或风况编号。 |
| `name` | 一种指标的实际名字，例如 `condition_rl_energy_j`；它并非 CSV 独立列。 |
| `value` | 该指标的数值。记录器跳过非有限值，因此缺行不应简单按零填补。布尔标志也写成浮点数。 |

同一工况的 `queue_condition` 行可先按 `stage+step` 组合起来，再看 `condition_id`、`condition_instance_id`：`condition_id` 是物理风况 ID，`condition_instance_id` 是这 92 次加载的实例 ID，不是轮次。日志中第一个完成的实例为 `condition_id=12`、`condition_instance_id=1`、`step=30001`；**不能假设执行顺序从风 3 开始**。跨表匹配应带上阶段、全局步和实例，避免把同名指标误并。

## 2. 本次配置如何解读

**字段来源：** 配置键（如 `algorithm`、`state20_queue_contract`、`env.*`、`agent.*`）来自 `config.json`；BC 预览数值来自 `bc_prediction_preview_summary.json`；实际 `queue_update_count` 来自 `queue_round_N_summary.json` 和 `queue_training_summary.json`。

- `algorithm=td3`、`training_mode=bc_warmstart_rl`；`bc.resume_checkpoint` 指向当时的 **state20 residual v2** BC 权重。`bc.steps=0` 指本次运行不再新增离线 BC 预训练步数，不等于没有加载 BC 权重；预览有 512 条样本，`torque_residual_mae≈0.05327`、`pitch_residual_mae≈0.03709`，均是**归一化残差**误差。
- `state20_queue_contract` 为 `state20_queue_training_contract_v1`；`observation_width=20`，`condition_ids=3..25` 共 23 种、`rounds=4`、每次目标 `train_sim_s=600`。`queue.execution_mode=independent_case_reload`，实际实例运行顺序以记录为准。`env.bladed_step_duration_ms=20` 对应目标 50 个**仿真步/仿真秒**；不能据此推断计算机实际每秒能输出 50 条命令。`env.episode_length=512` 是常规单段长度配置，不是该 queue 的 600 秒上限。
- `queue.critic_start_after_rounds=1`、`actor_start_after_rounds=1`：首轮采集；从下一轮进入可更新阶段。`train.update_every_steps=30`、`updates_per_step=1`、`agent.policy_delay=4`：满足条件时每 30 个全局步调度一次 Critic，Actor 只在相应的延迟更新槽且门控允许时更新。`queue_update_count=68994` 是**更新事件/Critic 调度**，不是 Actor 优化次数；首轮为 0。名义计划更新数 `expected_scheduled_updates=69000`，不要拿计划数替代实际数。
- `env.action_high=[21586451.33303,0.43]` 分别是请求扭矩 Nm 和桨距 rad 的边界；`agent.actor_residual_target_scale=0.25`；`control.ood_threshold=6.0`。超过 OOD 阈值会影响 Actor 执行，但 Actor 参数更新门控、OOD 回退、低速保护是不同概念。
- `env.reward.power_reward_mode=relative_pid_energy`、`blade_root_load_mode=my_peak_increment`，还配有逐叶根载荷和电量的对偶约束，阈值 `epsilon_load_per_blade=[0,0,0]`、`epsilon_energy_percent=0`；详情见下文公式。`load_predictor.mode=shadow`、接口 v2，`predictor_artifact_sha256=null`、`hard_candidate_enabled=false`、`realized_plan_online_authorized=false`：不能把这些诊断性配置说成预测器硬约束已上线。
- `logging.mode=legacy_long_form`；**不是**后续的 `compact_streaming` 格式。`state20_queue_contract.identity_validation_level=diagnostic`、`policy_acceptance=false`：这里的风速前缀参考匹配并非严格完整时间-风速轨迹、工程身份和同窗 PID 验收。

## 3. 逐工况汇总：42 个实际字段

**字段所在文件：** `queue_condition_metrics.csv`。每种指标有 92 次；下表的 `{i}` 取 1、2、3，即三片叶片。以下分组**完整覆盖这张小表的 42 个实际 `name`**。

### 3.1 工况身份和进度（6 个）

| `name` | 解释 |
| --- | --- |
| `condition_id` | 风况 ID（3～25）；不是该风况实际测得的瞬时风速。 |
| `condition_instance_id` | 排队/重新加载实例的编号；结合风况和轮次识别一次运行。 |
| `queue_completed_round` | **当前这一条**完成后是否刚好结束整个轮次（0/1）。绝大多数工况为 0。 |
| `queue_completed_rounds` | 截至当前已完整结束的轮数；在一轮中途仍保留之前完成轮数。 |
| `queue_conditions_completed_in_round` | 当前轮已完成几个风况，最后一个到 23。 |
| `queue_train_elapsed_s` | 本次风况实际 TRAIN 阶段经过的仿真秒；目标 600 s，允许配置中的小偏差，例如末条约 599.94 s。 |

### 3.2 功率积分、电量差与违约（5 个）

| `name` | 解释 |
| --- | --- |
| `condition_pid_energy_j` | 配对 PID 参考功率逐步乘当前有效时间间隔后求和，单位 J。 |
| `condition_rl_energy_j` | RL 已实现电功率按相同时间间隔逐步求和，单位 J。 |
| `condition_delta_energy_j` | $E_{\mathrm{RL}}-E_{\mathrm{PID}}$，J；**负数**为发电量少，正数为多。 |
| `condition_energy_deficit_percent` | $\max(-\Delta E/\max(\lvert E_{\mathrm{PID}}\rvert,1)\times100,0)$；只反映不足的百分数，例如 `0.7` 是亏 0.7%，盈余时为 0。 |
| `condition_g_energy` | $\max(\text{deficit\_percent}-\epsilon_E,0)$；扣掉允许容差的非负约束违约。本次 $\epsilon_E=0$，数值以百分数点计，不是 0～1 的比例。 |

**口径提醒：**此处是逐步 $\sum P\Delta t$ 的训练约束累计，采样值/有效步与后续“独立推理 + 同窗 PID”采用的原始时刻梯形积分并不必然完全相同。不要直接把这五项当成正式验收表。

### 3.3 三片叶根 My 峰值、相对变化和违约（12 个）

| `name`（每片叶片 `{i}`） | 解释 |
| --- | --- |
| `condition_pid_peak_blade_{i}_nm` | 配对 PID 的该片叶根 $\lvert M_y\rvert$ 峰值参考，Nm。 |
| `condition_rl_peak_blade_{i}_nm` | 本次工况记录的 RL 叶根 $\lvert M_y\rvert$ 峰值，Nm。 |
| `condition_delta_load_blade_{i}` | $L_{\mathrm{RL},i}/L_{\mathrm{PID},i}-1$，**比例**；`0.04` 表示峰值增加约 4%（变差），`-0.04` 表示降低约 4%（改善）。 |
| `condition_g_load_blade_{i}` | $\max(\text{delta\_load}_i-\epsilon_i,0)$；本次容差 $\epsilon_i=0$，所以只保留超出 PID 峰值的部分。为零**不说明电量合格**。 |

特别注意：独立推理汇总里的 `load_reduction_percent` 是 $(1-L_{\mathrm{RL}}/L_{\mathrm{PID}})\times100$，**与 `condition_delta_load_blade_{i}` 符号相反**，且一边是百分数、一边是比例。每风况相对比例先平均、先汇总峰值再计算比例，结果也不会严格相同。

### 3.4 对偶乘子与完成状态（5 个）

| `name` | 解释 |
| --- | --- |
| `dual_lambda_load_blade_1`、`dual_lambda_load_blade_2`、`dual_lambda_load_blade_3` | 记录**完成当前工况时**载荷约束的非负乘子；它们是训练软惩罚权重，不是 Nm，也不是三片叶的实际载荷。 |
| `dual_lambda_energy` | 同理，电量缺口对应的非负惩罚乘子。 |
| `dual_constraint_condition_complete` | 已把本工况完整结果计入约束统计时为 1。 |

乘子在**整轮边界**才根据该轮完整工况的平均违约更新并截到配置上限；因此工况表尾行的 `dual_lambda_*` 与同轮 `queue_round_N_summary.json` 中**更新后**的值可能不同。逐步的 `dual_step_penalty` 是从训练 reward 扣除的软惩罚，不能保证实际发送前动作满足最终电量/极限载荷的硬约束。

### 3.5 逐步统计与回放（14 个）

| `name` | 解释 |
| --- | --- |
| `queue_condition_raw_frame_count`、`queue_condition_train_frame_count` | 本实例逐步处理的帧/步骤数与其中 TRAIN 阶段的数目；可能含边界步骤，不要机械用 600/0.02 来强制相等。 |
| `queue_condition_eligible_transition_count` | 同一 TRAIN 工况内、未跨边界，可形成训练转移的步数；“合格采集机会”还不是最终入库。 |
| `queue_condition_stored_transition_count`、`queue_condition_rejected_transition_count` | 本工况实际存入 replay 与未存入的转移数。**历史累计采集数**，不同于下面的当前 replay 存量。 |
| `queue_condition_rejected_boundary_count`、`queue_condition_rejected_non_train_count` | 因跨工况/阶段边界，或非 TRAIN 不可入库。 |
| `queue_condition_rejected_fallback_count`、`queue_condition_rejected_unsafe_count`、`queue_condition_rejected_negative_power_count` | 因控制回退、危险停机或负功率而被过滤；归因是按代码中的优先级匹配，不能重复累计同一步的多个原因。 |
| `queue_condition_rejected_invalid_count`、`queue_condition_rejected_other_count` | 非有限转移、无效环境数值或其余原因导致的未存储。 |
| `queue_condition_replay_count` | 此刻该风况回放缓冲区里**仍保留**的样本数，满额上限为 40000，旧样本可被后续样本覆盖；不能把四轮末的存量相加当总采样量。 |
| `queue_condition_replay_online_bladed_count` | 当前缓冲区里来源标记恰为 `online_bladed` 的数。该实验按**执行模式**赋 replay 来源标签，`0` **不等于没有从 Bladed 采到训练数据**，更不能无视 `queue_condition_replay_count`。 |

一个实数例子：首个完成工况风 12 的 `condition_pid_energy_j≈8.8395×10^9`、`condition_rl_energy_j≈8.6376×10^9`，故 `condition_delta_energy_j≈-2.019×10^8` J（电量较 PID 少）；其叶片 1 `condition_delta_load_blade_1≈-0.03189`，即峰值约下降 3.19%，`condition_g_load_blade_1=0`。**电量与叶片的方向可以不同**，不能只看一列评判策略。

## 4. 工况检查点：7 个实际字段

**字段所在文件：** `queue_condition_checkpoint_metrics.csv`。每个工况 1 个检查点（配置 `condition_checkpoint_count=1`），字段如下：

| `name` | 解释 |
| --- | --- |
| `queue_checkpoint_round_index` | 本检查点属于第几轮（从 1 起）。 |
| `queue_checkpoint_condition_id`、`queue_checkpoint_condition_instance_id` | 对应风况 ID 和重新加载实例 ID。 |
| `queue_checkpoint_index`、`queue_checkpoint_count` | 本工况第几个进度检查点、预期共几个；本次通常 1/1。 |
| `queue_checkpoint_target_train_elapsed_s` | 计划触发该检查点的 TRAIN 仿真时间，单位 s。 |
| `queue_checkpoint_actual_train_elapsed_s` | 实际保存时已运行的 TRAIN 仿真时间，单位 s；边界处可能略异于目标。 |

检查点 JSON/CSV 字段**不是模型参数明细**；对应文件在 `checkpoints/`，整轮文件如 `queue_round_4.pt`。这张表 92×7 行只说明保存记录，不代表 `.pt` 文件已通过独立推理验收。

## 5. 每轮和全程 JSON 字段

**字段来源：** `queue_round_N_summary.json`（轮次累计、约束、健康及回放字段）、`queue_training_summary.json`（全程累计字段）、`bc_prediction_preview_summary.json`（BC 预览字段）。

`queue_round_N_summary.json` 每轮都包含同一批基础字段（本历史文件每份 75 个键：29 个基础/约束/健康键 + 23 风况 × 2 个回放键）；**轮次 JSON 中没出现新版批次来源采样统计列**。具体读法：

| 键/字段族 | 解释 |
| --- | --- |
| `queue_completed_rounds`、`queue_total_rounds` | 已完成轮数 / 计划轮数；分别为 1～4 / 4。 |
| `queue_total_steps` | 从实验开始累计的控制步数。 |
| `queue_total_stored_transitions`、`queue_total_rejected_transitions` | 迄今累计存入/拒绝的转移数，不是当前 replay 存量。 |
| `queue_skipped_condition_count`、`queue_retried_condition_attempt_count` | 不完整风况跳过次数 / 尝试重试次数；本次都是 0。 |
| `queue_require_zero_skipped_conditions_for_updates`、`queue_require_zero_retried_condition_attempts_for_updates` | 本轮更新前要求“跳过/重试数必须为零”的配置标志，**不是观测到的计数**。 |
| `queue_updates_quarantined_after_skip`、`queue_updates_quarantined_after_retry`、`queue_updates_quarantined_after_failure` | 遇到异常后隔离更新的标志/状态；本次是 0。 |
| `queue_training_valid` | 训练流程按自身安全/完整性门槛被标记有效；不等于策略性能达标。 |
| `queue_all_conditions_covered` | 各风况回放是否达到配置最低可用量，而不仅是“见过风况 ID”。 |
| `queue_update_count` | 累计的 update/Critic 事件数；轮 1～4 分别是 0、22998、45996、68994；**不是每轮独立数，更不是 Actor 次数**。 |
| `dual_constraint_updated`、`dual_constraint_complete_conditions` | 这轮是否执行了乘子更新 / 纳入平均的完整工况数（23）。 |
| `dual_mean_g_load_blade_{1,2,3}`、`dual_mean_g_energy` | 该轮完整工况非负违约量的算术均值；电量一项单位为百分数点，其余为比例。 |
| `dual_lambda_load_blade_{1,2,3}`、`dual_lambda_energy` | 本轮结束、按平均违约更新后的四个乘子值。 |
| `queue_critic_q_health_degraded`、`queue_critic_q_health_failure_triggered`、`queue_critic_q_health_passed`、`queue_critic_q_health_continue_freeze_actor` | Critic 更新健康降级/触发失败、是否通过、是否进入“继续但冻结 Actor”模式；这次未出现 Critic 健康降级。 |
| `queue_condition_<id>_replay_count`、`queue_condition_<id>_online_bladed_replay_count` | **当前轮边界**每个风况的 replay 存量及 `online_bladed` 来源标签数量；`id` 为 3～25。本次轮 4 各风况存量 40000、`online_bladed` 为 0，是标签口径差异。 |

`queue_training_summary.json` 是最终累计简表：复用上表的 `queue_completed_rounds`、`queue_total_rounds`、`queue_total_steps`、`queue_total_stored_transitions`、`queue_total_rejected_transitions`、`queue_update_count`、跳过/重试与隔离/健康字段，另外含 `queue_abort_on_retried_condition_attempt`（配置的“重试则中止”标志）及 `queue_model_candidate_valid`（内部候选的**过程**有效标志，不是策略发布许可）。本次最终是 **4/4 轮、2,759,788 步、2,689,070 条累计入库、70,718 条累计拒绝、68,994 次 update**，无跳过或重试；性能指标必须另看推理和同窗 PID 对比。

`bc_prediction_preview_summary.json` 只有 `preview_samples`、`torque_residual_mae`、`pitch_residual_mae`：是训练前/加载后的 BC 残差预览质量指标，不是 Nm、rad 的物理动作 MAE，也不是本实验 Actor 四轮的进步曲线。

## 6. 逐控制步日志：怎样从很多 `name` 找到想看的值

**字段所在文件：** `queue_train_step_metrics.csv`。它以全局 `step` 记一次动作执行后的观测、控制链、安全结果和奖励。大表的不同位置有界抽样见到约 244 种 `name`，下表按含义归纳实际常见字段和**同名字段族**；并非声称对 40 GB 全表做过全集统计。

### 6.1 哪个风况、什么时间、采集是否有效

**字段所在文件：** `queue_train_step_metrics.csv`。

| 字段/字段族 | 怎么读 |
| --- | --- |
| `queue_condition_round_index`、`queue_condition_id`、`queue_condition_instance_id` | 当前轮次、风况 ID、实例 ID；与全局 `step` 一起定位现场。 |
| `queue_stage_code` | 优化调度阶段：`1=collect` 仅采集；`2=critic` 可优化 Critic；`3=actor` 可按延迟和门控优化 Actor。**不等于**这一步的动作由谁实际执行；“critic 阶段”仍可能发送 Actor 动作。 |
| `udp_queue_epoch`、`udp_condition_id`、`udp_condition_instance_id`、`udp_phase_code`、`udp_phase_elapsed_s` | **当前输入帧**的 queue 轮次/风况/实例/阶段/阶段内仿真时间。`udp_next_*` 是执行该动作后**下一个帧**的对应元数据。`phase_code: 0=无、1=SETTLE、2=TRAIN、3=END`。`udp_queue_protocol_v2` 及 `udp_next_queue_protocol_v2` 是协议版本标志。 |
| `udp_sim_time`、`udp_next_sim_time`、`udp_frame_time` | 当前帧时间、下一帧时间、下一帧的原始 `Time`；单位 s。后两项通常对应动作后的环境，和当前帧不可随意混用。 |
| `condition_train_active`、`condition_boundary`、`condition_start`、`condition_end` | 下一帧是否处于 TRAIN、是否跨阶段/工况边界、是否开始/结束工况（0/1）；边界步骤不可当普通有效转移。 |
| `transition_store_eligible`、`transition_store_reason_code` | 当前和下一帧都在同一 TRAIN 且不跨边界才 eligible；理由码 `1=eligible、2=边界、3=非 TRAIN`。**eligible 不保证最终入 replay**：还要通过数值、安全、回退等过滤。 |
| `wind_t`、`wind_next` | 此实现从**动作后的下一帧**取的风速，两者可相同，单位 m/s；不能仅凭字段名就假设是动作发送前的风速。 |
| `power_t_metric_raw`、`power_metric_raw` | 动作前输入帧、动作后奖励帧的 `Electrical power`，W。`power`、`power_reward_metric`、`power_state_real` 分别是奖励采用的动作后功率、动作后的奖励功率通道和动作后 `I_RealPower` 状态，W；这些名称相近但不要无条件互换。 |
| `queue_all_conditions_covered`、`queue_training_valid`、`queue_updates_quarantined_after_skip/retry/failure` | 当前回放覆盖、训练过程标记和异常后优化隔离状态；**不是**实时性能/发布许可。`queue_train_timeouts_since_last_log` 是距上次打印窗口的超时数，不是全程总数。 |
| `queue_case_identity_validation_active`、`queue_case_identity_validated`、`queue_case_identity_expected_wind`、`queue_case_identity_observed_wind`、`queue_case_identity_expected_max_abs_error`、`queue_case_identity_closest_*`、`queue_case_identity_reference_offset`、`queue_case_identity_wind_sample_index`、`queue_case_identity_condition_id/instance_id` | 基于少量风速样本的参考匹配诊断：预期/实测风速与最大绝对误差单位 m/s，offset/index 是样本序号，closest 为最接近的候选。`validated=1` 只说明**本配置的诊断性前缀校验**完成，不是全 600 s 时间-风速、Bladed 工程和配对 PID 同一物理实例均已严格证明。 |

### 6.2 从 Actor 提议到真正发送：最重要的动作链

**字段所在文件：** `queue_train_step_metrics.csv`。下面的 `{x}` 同时适用于 `torque`、`pitch`；**扭矩物理值是 Nm，桨距物理值是 rad**。不要把策略网络的归一化数值直接乘以电功率或当作已发送命令。

| 字段族（`control_` 前缀，另有 `action_*`） | 意义与单位 |
| --- | --- |
| `control_baseline_action_{x}` | B4/当前保底控制建议的物理动作，Nm/rad。`control_bc_baseline_action_{x}`、`control_bc_baseline_residual_{x}` 是可选 BC 基线/其归一化残差诊断，值为 0 时不一定表示真正启用了零动作。`control_pid_history_action_{x}` 及 `control_pid_history_index/action_index/action_offset_steps/wind/wind_error/observation_available` 属于 PID 历史回放诊断，不能直接当在线 Actor 输出。 |
| `control_policy_action_{x}` | Actor 提议的**归一化残差**；不是物理扭矩或桨距。`control_state_residual_limit_{x}` 是该状态允许的物理残差幅度（Nm/rad）。 |
| `control_policy_delta_{x}` | Actor 提议残差换算后的物理增量，约为 `policy_action × state_residual_limit`，Nm/rad；**即使随后回退，这个诊断提议仍可能非零**。`control_pre_safety_action_{x}` 是 B4 基线加此原始提议，尚未进行安全处理。 |
| `control_residual_{x}` | 安全仲裁后**准入**的归一化残差；回退时可置零。`control_residual_admitted` 是是否准入的 0/1 标志；不要与 `control_policy_action_{x}` 混淆。 |
| `control_pitch_protection_floor`、`control_pitch_protection_state`、`control_pitch_protection_reason`、`control_pitch_protection_lift`、`control_pitch_protection_applied`、`control_post_pitch_protection_action_{x}` | 桨距下限（rad）、保护状态/原因码、加抬量（rad）、是否实际施加以及施加后的物理动作。状态码通常 `0=无下限、1=下限存在但未抬高、2=下限生效`；原因码是控制器内部规则，不能把它当成状态码。 |
| `control_low_speed_stage`、`control_low_speed_scale`、`control_low_speed_reason`、`control_low_rotor_speed_fallback_active`、`control_low_rotor_speed_torque_unload_active`、`control_post_safety_action_{x}` | 低转速的阶段、缩放比例、原因码、回退/降扭标志，以及完成桨距和低速保护后的物理动作。阶段 `0=常态、1=预警、2=保护、3=紧急、4=恢复保持`；阶段 4 并非 Actor 已恢复执行。 |
| `control_projected_action_{x}`、`control_rate_limited_action_{x}`、`control_action_projection_applied`、`control_action_rate_limited` | 将保护后动作投影到安全物理边界、再处理动作变化速率后的数值和发生标志；速率限制可能还由环境二次处理。 |
| `control_action_{x}`、`control_sent_action_{x}`、`action_{x}`、`action_{x}_delta` | 分别为控制器准备输出的物理动作、整个控制链记录的实际发送动作、环境记录的实际发送动作、相对上一次发送的差量。分析执行效果以**实际发送**为准，而不是仅看 Actor 建议。 |
| `control_executed_action_{x}` | 将**实际执行的物理动作相对基线**反算得到的归一化残差，用于 replay；它也不是 Nm/rad，更不是归一化后的整机命令。 |

`control_above_rated_initial_pitch_floor_active/applied` 表示额定以上初始桨距下限是否存在/生效；`control_fallback_generating_torque_floor_applied` 表示回退路径生成扭矩下限曾介入。`control_residual_scale_diagnostic`、`control_residual_scale` 是诊断缩放开关和倍数；`control_state_observation_enabled` 表示带工况状态的观测已开启。不同阶段同名“动作”有顺序，**不是多份独立发送记录**。

### 6.3 为什么回退，以及几个不能混用的门控

**字段所在文件：** `queue_train_step_metrics.csv`。

| 字段族 | 意义 |
| --- | --- |
| `control_pre_send_execution_mode` | **这次发送前真正选中的执行模式编码**，见第 7 节；判断 Actor 是否真正执行优先看它和 `control_residual_admitted`。 |
| `control_fallback_active`、`control_fallback_bc_baseline`、`control_fallback_pid_history` | 是否实际回退，以及回退动作是否取自指定的 BC 或 PID 历史；这几项不是“Actor 参数更新门关闭”的同义词。 |
| `control_force_baseline/force_bc_baseline/force_pid_history`、`control_baseline_hold_initial` | 强制某种基线或保留最初基线的配置/当前选择；原因要与 `control_pre_send_execution_mode` 交叉看。 |
| `control_actor_policy_allowed`、`control_actor_execution_allowed`、`control_actor_execution_valid` | 当前工况允许使用 Actor、提议是否合法；允许不保证最终被执行（可能 OOD/低速/异常挡掉）。 |
| `control_actor_update_gate_open`、`control_actor_update_blocked_q_health` | **优化器**层面是否允许 Actor 参数更新及本步因健康问题阻断的标志；并不等于这一刻是否向 Bladed 发送了 Actor 动作。历史 `control_actor_policy_blocked_q_health` 为兼容性字段，不能仅凭其为零认定完全没有任何门控。 |
| `control_ood_detected`、`control_ood_score`、`control_ood_feature_index/value/median/iqr` | 离观测训练分布的稳健偏离：得分超过冻结配置的阈值 6 时回退；feature 的索引、当前值、中位数、四分位距帮助定位异常特征。得分本身无物理单位，也不是风速。 |
| `control_filtered_wind_speed_mps`、`control_operating_state_changed`、`control_region_changed`、`control_optimization_state_code`、`control_region_code` | 滤波风速 m/s、工况/分区切换标志及分类编码，见第 7 节。 |
| `control_negative_power_steps/patience`、`control_low_power_steps/patience`、`control_unsafe_stop` | 功率异常/低功率的连续步数与停止耐心阈值、安全停机结果；不等同于功率调节的百分比。 |
| `control_post_step_timeout/invalid/unsafe_stop/condition_boundary` | **发送并观察下一帧后**才知道的超时、无效、危险停止、工况边界（0/1）；不能把它们当发送前已知的硬约束。 |

### 6.4 奖励项：原始物理量和训练目标要分开

**字段所在文件：** `queue_train_step_metrics.csv`。这一组的 `reward_*_raw` **不总是同一单位**；`*_term` 往往经过比例化/裁剪，具体还会再乘配置权重。历史奖励模式曾含转速与独立负功率项；当前活动 reward 以相对 PID 电量、三叶峰值增量、持续载荷/功率缺口和动作连续性为主：

| 字段族 | 通俗解释 |
| --- | --- |
| `reward_power_metric_raw`、`reward_power_state_raw`、`reward_power_raw`、`reward_power_t_raw`、`reward_pid_power_reference_raw` | 分别为下一帧奖励电功率、下一帧实功率状态、奖励选中的电功率、上一帧电功率、该时刻配对 PID 功率，单位 W。当前配置奖励选用 `Electrical power`；功率字段不等于能量。 |
| `reward_relative_power_raw`、`reward_energy_delta_t_s`、`reward_relative_energy_increment_j`、`reward_energy_increment_j`、`reward_energy_increment_valid`、`reward_power_term` | 相对 PID 功率差 $P_{\mathrm{RL}}-P_{\mathrm{PID}}$（W）、有效步长（s）、$(P_{\mathrm{RL}}-P_{\mathrm{PID}})\Delta t$（J）、本奖励模式下**同一个相对增量**（J）、时间步有效标志，以及裁剪归一化后的奖赏项。`reward_energy_increment_j` **不是**绝对 RL 发电量！ |
| `reward_blade_root_{1,2,3}_raw`、`reward_blade_root_raw`、`reward_blade_root_ratio` | 三片叶的瞬时 My 绝对值 Nm；聚合的瞬时叶根载荷 Nm；运行峰值/参考峰值的无量纲比例。`blade_root_metrics_available/nonzero/abs` 是传感器通道与该聚合值的简单检查。 |
| `reward_blade_root_{i}_peak_nm`、`reward_blade_root_{i}_peak_reference_nm`、`reward_blade_root_{i}_peak_excess`、`reward_blade_root_{i}_peak_excess_increment` | 第 `i` 片运行至当前的 $\lvert M_y\rvert$ 最大值 Nm、配对 PID 峰值参考 Nm、超参考比例及本步**新增加的**超参考比例。仅一小步没新增峰值，不能推出整段安全。 |
| `reward_blade_root_peak_raw`、`reward_blade_root_peak_excess`、`reward_blade_root_peak_excess_increment`、`reward_blade_root_term`、`reward_blade_root_continuous_term` | 三叶运行峰值/超额/新增超额的聚合及用于奖励的无量纲惩罚；本配置峰值聚合取三片的最大，连续项按相对 PID 峰值超额计算。它们不是正式极限载荷验收结果。 |
| `reward_acc_raw/term`、`reward_shaft_raw/term`、`reward_pitch_rate_raw/term`、`reward_torque_rate_raw/term` | 机舱加速度（原始单位 m/s²）、轴系代理量（按配置算法，并非测量到的叶根 My）、相邻发送桨距变化量 rad、扭矩变化量 Nm，以及对应的归一化项；这里 `rate_raw` 是**每步差量**，不是先除以秒的导数。当前 R2 acceleration/shaft 基础 reward 权重为零，相关项即便记录也不影响 reward。 |
| `reward_power_available_raw/fraction`、`reward_command_power_raw`、`reward_command_excess_raw/term`、`reward_power_tracking_raw/term` | 估计可用功率 W、实现功率占其比例、扭矩×转速估算命令功率 W、超可用功率 W、实际与估算可用功率之差 W及归一化项；**估计可用功率不是独立实测发电量**。 |
| `reward_negative_power_raw/term` | 仅旧日志字段：负发电功率幅值（W）及旧版独立惩罚项。当前 reward 不输出这些字段，也不对负功率增加独立 reward 扣分；主相对 PID 功率项和持续功率缺口项仍按定义作用。 |
| `reward_persistent_power_deficit_raw/ewma/term/clipped` | 扣除死区后相对配对 PID 的持续功率不足（原始/平滑均 W）、归一化惩罚、是否截断标志；不等于全工况电量违约百分数。 |
| `reward_rotor_speed_raw/reference/deadband`、`reward_region_rotor_speed_term`、`reward_rotor_speed_variation_term`、`reward_sustained_low_rotor_speed_term` | 旧日志中的转速诊断/惩罚字段；当前 reward 不再生成这些字段，也不计算转速惩罚。当前状态分类与安全低速保护仍使用转速。 |
| `reward_regional_operating_state_code`、`reward_regional_control_region_code` | 旧 reward breakdown 中的工况/控制分区编码；当前 reward 不生成这些字段。当前控制状态仍由 `control_optimization_state_code`、`control_region_code` 表示。 |
| `reward_state_power_multiplier`、`reward_state_blade_root_multiplier`、`reward_state_continuous_load_multiplier`、`reward_state_persistent_power_deficit_multiplier`、`reward_state_action_continuity_multiplier` | 当前状态对各奖励/惩罚项的无量纲权重倍率；它们不是新的实测功率或载荷。 |
| `reward_total` | 训练步的合计奖赏；在此训练日志中可能**已经减去**额外 `dual_step_penalty`。数值提高并不自动证明发电量增益、My 峰值通过或 50 Hz 达标。 |

`dual_step_load_increment` 是三叶新超载增量的最大值（无量纲），`dual_step_power_deficit_fraction` 是当前 $\max(P_{\mathrm{PID}}-P_{\mathrm{RL}},0)/P_{\mathrm{PID}}$（PID 功率为正时，无量纲），`dual_step_penalty` 是基于乘子扣掉的训练软惩罚。三个字段都不能直接当作“实际发送前已被阻止的动作数”。

### 6.5 UDP 链路和环境结果

**字段所在文件：** `queue_train_step_metrics.csv`。

| 字段族 | 解释及单位 |
| --- | --- |
| `udp_connected`、`udp_handshake_complete`、`udp_frame_id`、`udp_next_frame_id`、`udp_frame_lag`、`udp_dropped_stale_frames` | 连接/握手状态，当前/下一帧号、序号间隙、丢弃过时帧数；帧号缺失可用 `-1` 代表未提供。 |
| `udp_receive_monotonic_s`、`udp_action_sent_monotonic_s`、`udp_action_ready_lag_ms`、`udp_action_apply_lag_ms` | 本步输入帧**被接收**的单调时钟 s、动作**已发送**时的单调时钟 s、动作准备好的接收后延迟 ms、两时间戳之差 ms。字段名中的 `apply_lag` 在此实现是**接收至发送延迟**，并非 Bladed 叶片执行器真正响应完成的时间。 |
| `udp_receive_wait_ms`、`udp_wait_limit_s`、`udp_inter_frame_timeout`、`udp_condition_load_active/timeout`、`udp_pause_active`、`udp_pause_wait_ms` | 动作发送**之后等下一帧**的耗时、允许等待上限及是否超时；换工况等待/暂停的状态与超阈等待。`udp_receive_wait_ms` 不是边缘发送前时延，不能与上一行相加推断正式 T1。 |
| `udp_action_next_torque/pitch`、`udp_action_next_torque_delta/pitch_delta` | 实际发送物理命令 Nm/rad 及相对上次变化量；可用于检查环境的二次投影。 |
| `env_invalid_prediction`、`env_invalid_reward`、`env_input_clip_fraction`、`env_output_clip_fraction`、`env_model_clip_event/count` | 环境数值有效性/裁剪遥测；当前 Bladed UDP 实现的部分 `clip` 字段固定写零，**不能把零当作设备层面从未限幅的物理证明**。超时分支可另外给 `env_timeout`。 |

## 7. 状态码速查：不要只看一个“模式”字段

**字段所在文件：** 未加 `info_` 前缀的训练字段来自 `queue_train_step_metrics.csv`；独立推理中对应的状态码（若被记录）位于 `bladed_inference_steps.csv` 的 `info_*` 列或 `step_info_json`。

| 字段 | 编码及含义 |
| --- | --- |
| `control_optimization_state_code` | `-1=不可用`，`0=startup_stopped`、`1=below_rated`、`2=transition`、`3=above_rated`、`4=negative_power_abnormal`。 |
| `control_region_code` | `-1=不可用`，`0=below_rated`、`1=transition`、`2=above_rated`。 |
| `reward_regional_operating_state_code` | `-1=无`，`0=startup_cut_in`、`1=normal_generation`、`2=transition`、`3=above_rated_regulation`、`4=shutdown_stopped`、`5=negative_power_abnormal`；是奖励内部重新分类，**不按** `control_optimization_state_code` 的顺序。 |
| `reward_regional_control_region_code` | `-1=无`，`0=below_rated`、`1=transition`、`2=above_rated`。 |
| `control_pre_send_execution_mode` | `1=actor`（残差准入）；`2=explicit_baseline_stage`（显式基线）；`3=actor_execution_rollback_baseline`（Actor 执行异常回滚）；`4=ood_baseline`；`5=low_speed_warning_hold`；`6=low_speed_protection_hold`；`7=low_speed_emergency_hold`；`8=low_speed_recovery_hold`；`9=startup_stopped_baseline`；`10=negative_power_abnormal_baseline`。只有代码 1 代表这一步准入了 Actor 残差。 |
| `control_low_speed_stage` | `0=正常`、`1=预警`、`2=保护`、`3=紧急`、`4=恢复保持`；与执行模式 5～8 对应，但可能被更高优先级的启停/功率异常状态覆盖。 |
| `udp_phase_code`/`udp_next_phase_code` | `0=没有阶段信息`、`1=SETTLE`、`2=TRAIN`、`3=END`。与训练调度 `queue_stage_code` 是**完全不同的两套码**。 |

## 8. 逐优化事件日志：Critic、Actor、BC 和健康门控

**字段所在文件：** `queue_train_update_metrics.csv`（`stage=queue_train_update`）。一次更新可生成很多行，且同一 `step` 并不意味着一次 Actor 更新。大表有界抽样见到约 48 种键；产生日志的代码可能输出其他少见健康/跳过字段。历史实际配置：`train.batch_size=128`、`agent.policy_delay=4`、`agent.max_target_q_abs_mean=20`，Critic 健康检查窗口 200 次，Actor 优化器健康门控要求连续 200 次健康 Critic 更新且固定离线 batch 健康。

| 字段族 | 意义；怎样避免误读 |
| --- | --- |
| `queue_stage_code` | 与逐步表相同，表示发生更新时的训练阶段；`2` 不会更新 Actor，`3` 也只是**允许**尝试 Actor。 |
| `critic1_loss`、`critic2_loss` | 两个 Critic 拟合目标回报的损失；数值不是发电量误差。`critic1_td_error_abs_mean`、`critic2_td_error_abs_mean` 是 TD 误差绝对值的 batch 均值（Q/回报尺度）。 |
| `q1_mean`、`q2_mean`、`critic_q_abs_gap_mean` | 当前 batch 上两个 Q 估计的均值及二者绝对差均值；Q 是折扣回报的模型值，**单位不是 W/J/Nm**。 |
| `target_q_mean`、`target_q_abs_mean` | 前者是 Critic 训练目标（奖励 + 折扣的目标 Q）的 batch 均值；后者是目标双 Critic 中较小者的**绝对值均值**，两者公式不同。硬阈值 20 用在后者；超限事件可记 `update_skipped_q_guard=1`，不应只看 `target_q_mean` 的正负。 |
| `critic_obs_scaled_abs_mean/max` | Critic 归一化观测值的绝对均值/最大值，供输入分布检查；不是 OOD 控制分数。 |
| `actor_update_slot_due`、`actor_update_attempted`、`actor_update_deferred` | 延迟更新是否轮到 Actor、是否进入 Actor 尝试、是否因阶段/门控而推迟。部分字段只在对应分支出现；**三者都不能替代 `actor_loss` 来统计成功更新次数**。 |
| `actor_q_mean`、`actor_q_abs_mean`、`actor_rl_loss` | Actor 当前提议在 Critic 上的 Q 均值、绝对均值及未乘 BC 平衡系数前的 $-\overline Q$；不是实际发送命令或实测电功率。 |
| `actor_bc_loss`、`actor_bc_lambda`、`actor_bc_weight`、`actor_bc_effective_weight` | BC 目标偏离损失、按 Q 尺度自适应的系数、调度的原始 BC 权重、经过缩放/目标比例/封顶后的有效 BC 权重。 |
| `actor_rl_term`、`actor_bc_term`、`actor_bc_to_rl_ratio`、`actor_loss` | TD3+BC 的 RL 项与 BC 项（已加权）、二者绝对大小之比、最终优化损失；成功时 $\text{actor\_loss}=\text{actor\_rl\_term}+\text{actor\_bc\_term}$。**只有成功执行 Actor 优化的事件才含 `actor_loss`**；`actor_rl_loss` 存在不能单独证明已更新。 |
| `actor_update_gate_open_before_critic/after_critic`、`actor_update_blocked_q_health`、`actor_update_blocked_control_health`、`actor_update_blocked_execution_mode`、`actor_update_requested_after_critic` | Critic 本次更新前后门控、Actor 受 Q/数值健康或本步执行模式阻断、真正请求更新的状态；请同时检查“轮到更新槽、是否成功、具体执行模式”。 |
| `actor_optimizer_health_gate_enabled/open`、`actor_optimizer_health_gate_closed_reason_code`、`actor_optimizer_health_consecutive_healthy_critic_updates`、`actor_optimizer_health_required_consecutive_healthy_critic_updates`、`actor_optimizer_health_fixed_offline_batch_healthy`、`actor_optimizer_health_critic_update_numerically_healthy`、`actor_optimizer_health_recovery_count` | 本次**实际起作用**的 Actor 优化器健康门：是否启用/打开、闭门原因、健康连续次数/恢复门槛、固定离线 batch 检查、Critic 数值检查、恢复次数。闭门码 `0=已打开、1=初始恢复期、2=Critic 不健康、3=离线 batch 不健康`。 |
| `actor_q_health_gate_open_before_update` | **历史兼容命名**，记录的是上面有效门控的更新前快照，**不是**另一套趋势型 `ActorQHealthGate` 已独立放行的证据。该实验的配置虽含趋势型 gate，但有 optimizer gate 时未另外实例化它。 |
| `critic_q_health_window_scheduled/successful/guarded`、`critic_q_health_window_success_rate/guard_rate`、`critic_q_health_window_ready/evaluated/violated`、`critic_q_health_consecutive_violations`、`critic_q_health_failure_triggered/degraded`、`critic_q_health_continue_freeze_actor` | 最近窗口内调度/成功/受保护跳过次数、成功率/保护率、窗口就绪/违约、连续违约与降级处理。配置窗口长 200，guard-rate 上限 0.05，success-rate 下限 0.90；**这些比例不是电量提升比例**。 |

理解顺序：先看 `queue_stage_code` 和更新槽，再看 `target_q_abs_mean`/健康状态、门控前后，再看是否有 `critic1_loss` 与 `actor_loss`。本次四轮最终 `queue_update_count=68994`，不是 68994 次 Actor 更新；首批受恢复门控拦下的 Actor 槽不能解释为“整轮 Actor 没学”。训练输出的过程健康、物理性能和部署实时性是三个不同判据。

## 9. 独立推理文件：与训练长表完全不同的格式

`inference/wind8_round1_retry/`、风 16/24 的对应目录均是从相应轮的检查点发起的**单风况**运行；目录名末尾 `_retry` 是产物命名，**不等于训练的 `queue_retried_condition_attempt_count` 曾非零**。各目录常见文件：

| 文件 | 用途 |
| --- | --- |
| `bladed_inference_steps.csv` | **宽表**：一行包括一个时刻的 19 个 Bladed 原始物理量、控制动作/时延和 `info_*`。抽查三个目录均有 216 列；不要与四列训练长表混读，也不要在编辑器整表打开。 |
| `bladed_raw_frames.csv` | 同一仿真记录的 19 列 Bladed 原始帧；不含 Actor 提议和时延。 |
| `episode_summary.csv` | 每次推理的一行汇总（历史例子 26 列），可快速看 30000 控制步/30001 帧、回退/超时/终止。 |
| `run_summary.json` | `status`、`error`、起止 UTC/实际耗时、请求/完成 episode、总步/总记录数、输入输出文件路径、`inference_mode`/`baseline_only`、内嵌 episode 和 edge 摘要。`status=completed` 只是程序跑完。 |
| `checkpoint_metadata.json` | 检查点来源路径、载入 UTC、checkpoint 内是否带配置、训练时的 `total_steps`、schema/控制指纹、推理配置来源、运行设备及模式；用于追溯所用权重，不是性能评分。 |
| `effective_config.json` | **这次独立推理**的实际运行配置，不应与根目录训练 `config.json` 等同。例如已抽查目录为单风况 `env.queue_lifecycle_enabled=false`、`load_predictor=null`；若 `info_load_predictor_*` 有列但为空，不应说预测器当时参与控制。 |
| `edge_performance_summary.json` | 接收到输入帧至发出动作的时延分布、输出频率、T1 各子项与总判定；见第 10 节。 |

### 9.1 推理宽表开头 19 个物理通道

**字段所在文件：** 这 19 个 Bladed 原始物理通道同时记录在 `bladed_inference_steps.csv` 和 `bladed_raw_frames.csv`；前者还包含控制与时延字段。

| 列名 | 意义与单位 |
| --- | --- |
| `Time` | Bladed **仿真**时间，s；不是 Linux/Windows 墙上时钟。 |
| `I_RotorAzimuth`、`I_RotorSpeed`、`I_YawError` | 转子方位角 rad、转子转速 rad/s、偏航误差 rad。 |
| `I_ElectricalTorque`、`I_GeneratorSpeed` | 实测电磁扭矩 Nm 和发电机转速 rad/s；实测扭矩不等于扭矩请求值。 |
| `O_PcsTorqueDemand`、`O_PitchPositionDemand1` | 帧里的扭矩/桨距需求通道，Nm/rad；是**原始帧字段**，不应仅凭 `O_` 名字就断言它等于本行本次发送的策略输出，后者看 `sent_action_*`。 |
| `I_PitchPosition1`、`I_PitchPosition2`、`I_PitchPosition3` | 三片叶的实测桨距，rad。 |
| `I_NacelleForeAftAcceleration`、`I_NacelleSideSideAcceleration` | 机舱前后、侧向加速度，m/s²。 |
| `I_AverageWindSpeed` | 平均风速，m/s；不等于文件夹名字的固定常数。 |
| `Electrical power`、`I_RealPower` | 用于奖励/指标的电功率与实功率状态，W；通常接近，仍应按指标明确选用哪列。 |
| `Blade root 1 My (Nm)`、`Blade root 2 My (Nm)`、`Blade root 3 My (Nm)` | 三叶片叶根 $M_y$ 的带符号瞬时值，Nm；极限载荷比较使用各片在同一窗口内的 $\max\lvert M_y\rvert$，不是三列的平均瞬时值。 |

### 9.2 其他宽表列：一行究竟是“发动作”还是“末端观测”

**字段所在文件：** 本节宽表字段来自 `bladed_inference_steps.csv`；`info_*` 与 `step_info_json` 也在该文件中。

| 字段族 | 解释 |
| --- | --- |
| `record_type`、`source_frame_role` | `control_step / control_input` 是**当前输入帧**据以发出动作的一行；`terminal_observation / post_action_terminal` 是最后一次动作后的补充观测，通常无新推理动作/时延。计算 30000 控制步、电量或 T1 时必须按各脚本规则过滤，不能简单把 30001 行都算动作。 |
| `inference_episode`、`inference_step`、`inference_rollout_segment`、`inference_global_control_step`、`inference_record_index` | episode 序号、该 episode 中的控制步、按配置拆分的片段、运行内累计控制步、运行内记录行号。episode/segment/行号均不是风况 ID；单风况示例片段长度配置为 512 步。 |
| `wall_clock_utc`、`wall_clock_unix_s` | 帧写入记录时的实际 UTC 字符串/Unix 秒。不能直接与 `Time`（仿真秒）或 `t_rx_monotonic_s`（单调秒）相减。 |
| `policy_inference_ms`、`safety_projection_ms` | Actor 前向推理和安全控制器的本地处理耗时，ms；`policy_inference_ms` **不是**接收至发送全链路时延。 |
| `udp_step_ms`、`control_step_ms` | `env.step()` 的发送及等待下一帧、整步控制循环的本地耗时，ms；包含等仿真/网络后续帧的部分，不等于发送动作的边缘时延。 |
| `t_rx_monotonic_s`、`t_tx_monotonic_s`、`t_e2e_ms` | 接收当前输入、真正发送动作的单调时钟，以及 $(t_{\rm tx}-t_{\rm rx})\times1000$；这是该项目 T1 的 P99 时延输入。与 `info_udp_receive_wait_ms`（发出动作**以后**的等帧时间）不同。 |
| `policy_residual_{x}_norm`、`safe_residual_{x}_norm`、`executed_action_{x}_norm` | 分别为 Actor 原始提议、安全准入残差、相对 B4 基线反算的实际残差，均是**归一化残差**，不是 Nm/rad 的完整控制命令。 |
| `baseline_action_{x}`、`policy_delta_{x}`、`pre_safety_action_{x}`、`post_safety_action_{x}`、`post_pitch_protection_action_{x}`、`projected_action_{x}`、`rate_limited_action_{x}`、`sent_action_{x}` | 类似第 6 节的“基线→原始增量→保护→投影→限速→发送”动作链，都是 Nm/rad；历史推理表头未单列 `post_pitch_protection_action_{x}`，可在下述 `info_` 或 `step_info_json` 看**确实存在的**记录。 |
| `requested_action_{x}`、`sent_action_{x}_delta` | 前者是传给环境的控制器物理请求，后者是本步实际发送相对上次发送的物理变化；二次投影/限速时 `requested` 不应替代 `sent`。 |
| `pitch_protection_floor/state/applied`、`low_speed_stage/scale/reason` | 安全保护诊断，与训练步同名 `control_` 字段对应；含状态码的列不能当作桨距角。 |
| `step_reward`、`step_done`、`step_invalid`、`terminal_reason` | 一步环境奖励、是否结束、预测/奖励数值是否无效及结束原因（如 `max_steps_reached`/`udp_timeout`/`safety_stop`）；不是正式电量或载荷合格标志。 |
| `info_<name>`、`step_info_json` | `info_` 前缀复制选定的环境/安全控制 `name`（该示例 146 列），末尾 JSON 字符串保存本步可用的完整有限数值 info；列不存在或单元格空白**不能按 0 补**。例如本次独立推理的 `info_control_ood_score`、`info_control_fallback_active` 可查，`control_pre_send_execution_mode` 未单列为 `info_` 时可看 `step_info_json` 中是否记录。 |

具体 `info_*` 子族（均来自 `bladed_inference_steps.csv`，沿用第 6 节的单位和前后时序）包括 `info_udp_*`（本帧/下一帧时间、队列身份与接收/发送时延）、`info_control_*`（B4 基线与实际回退/安全动作；该历史推理表头实际有 30 列）、`info_reward_*`（奖励分解，该表头实际有 50 列）、`info_action_*`、`info_wind_*`、`info_power_*`、`info_blade_root_*`、`info_condition_*`、`info_transition_*`、`info_env_*`，以及 `info_load_predictor_shadow_*`（列存在但这次检查的单风况推理未配置 predictor，具体值可为空）。`info_condition_train_active=1` 在非 queue 单例推理中不能据此说“又进行了一轮训练”。此历史推理表头未提供完整严格 identity 的专用列，也不能由 `info_queue_case_identity_*` 在后来源码里出现推断这里已做过 full-trace 验证。

**字段所在文件：** `episode_summary.csv`。其 26 列可按四组理解：① `episode`、`start_wall_clock_utc`、`end_wall_clock_utc`、`control_steps`、`record_count`（步数与记录行数的区别）；② `episode_reward`、`reward_per_step`、`episode_duration_ms`、`control_step_ms_mean/max`、`policy_inference_ms_mean`、`udp_step_ms_mean`、`t_e2e_ms_mean/max`（奖励与耗时的各自统计）；③ `fallback_step_count/fraction`、`unsafe_stop`、`timeout_step_count`、`invalid_step_count`、`terminal_reason`（控制结果）；④ `queue_mode`、`queue_condition_id/instance_id`、`queue_epoch`、`queue_phase_code`、`rollout_segments`（运行形态/队列元数据）。`queue_mode=0` 的单工况推理不应强行解释不存在的 queue ID。示例有 `control_steps=30000`、`record_count=30001`：后一行是末端观测。

## 10. 边缘时延判定及轮次进展文件

### 10.1 `edge_performance_summary.json` 的各字段

**字段所在文件：** `edge_performance_summary.json`。

| 字段 | 如何解释 |
| --- | --- |
| `mode`、`records_csv` | 边缘评估协议版本及输入推理宽表路径。 |
| `control_step_count`、`valid_sample_count`、`invalid_timing_count` | 只选 `record_type=control_step` 的步数、有效接收/发送时戳数、缺失/非有限/倒序的无效时戳数；末端观测不计。 |
| `latency_ms.min/p50/p95/p99/max` | 由 $(t_{\rm tx}-t_{\rm rx})\times1000$ 得到的时延分布，ms；`p99` 不是单步最大值。 |
| `output_frequency_hz` | 从有效动作发送时刻算 $(N-1)/(t_{\rm tx,last}-t_{\rm tx,first})$，**实际墙上时间**频率 Hz；不等于 20 ms 仿真时间分辨率。 |
| `valid`、`reason` | 时戳/顺序等指标本身是否有效及无效说明；`valid=true` 不代表 T1 成功。 |
| `criteria.minimum_sample_count`、`criteria.p99_e2e_ms_max`、`criteria.output_frequency_hz_min`、`criteria.T1_rule` | 判据来源：至少 5000 条有效控制步、P99 接收→发送不超过 20 ms、实际输出频率至少 50 Hz，且时序有效。 |
| `sample_count_pass`、`p99_latency_pass`、`output_frequency_pass`、`T1_pass` | 三个子项及**同时满足**时的总判断。 |

例如风 8 第 1 轮独立推理：30000 条有效控制步，`valid=true`、P99 ≈ 8.08 ms、`p99_latency_pass=true`，但 `output_frequency_hz≈27.97`、`output_frequency_pass=false`，因此 **`T1_pass=false`**。这不与“平均边缘发送前时延只有数毫秒”矛盾，也不应因为仿真 20 ms 一步就认为达到 50 Hz。该例最大单步时延约 163.89 ms，也不能用最大值替代 P99。

### 10.2 `wind<风况>_round_progress_retry/` 的轮次对照

**字段所在文件：** `round_progress_summary.csv`（表格列）和 `round_progress_summary.json`（JSON 键）。每个目录还含三张图 `round_progress_power_load.png`、`round_progress_relative_to_pid.png`、`round_progress_blade_root_peaks.png`。CSV **一行对应一轮独立推理**，不是训练 step 表。JSON 有 `mode`、`pid_csv`（对比 PID 路径）、`pid_start_row_zero_based`（PID 对齐窗口**从第几行起，0 起**）、`rounds`（每轮下述数据对象），以及图和结果文件路径 `absolute_plot/relative_plot/blade_plot/summary_csv/summary_json`。

| CSV 列族（JSON `rounds` 中同义字段） | 算法/单位 |
| --- | --- |
| `round`、`inference_dir`、`matched_steps`、`duration_s` | 检查点轮次、单次推理目录、与 PID 所取窗口的控制步数、推理持续仿真秒。要检查风况和 PID 路径/窗口是否真正可比。 |
| `rl_mean_power_mw`、`pid_mean_power_mw`、`power_delta_percent` | 两条轨迹平均功率 MW，变化百分数 $(\overline P_{\mathrm{RL}}/\overline P_{\mathrm{PID}}-1)\times100$，正数是 RL 功率高。 |
| `rl_energy_mwh`、`pid_energy_mwh`、`energy_delta_percent` | 按各自严格递增的原始仿真 `Time` 对功率用**梯形积分**算得 MWh；变化百分数 $(E_{\mathrm{RL}}/E_{\mathrm{PID}}-1)\times100$，正数是增发。不能与第 3 节逐步累积 J 值不加说明地混合。 |
| `rl_blade_{1,2,3}_peak_mnm`、`pid_blade_{1,2,3}_peak_mnm` | 同窗每片 $\max\lvert M_y\rvert$，MNm；JSON 里是 `rl_blade_peak_mnm`、`pid_blade_peak_mnm` 三元素数组，顺序为叶片 1、2、3。 |
| `rl_mean_blade_peak_mnm`、`pid_mean_blade_peak_mnm`、`load_reduction_percent` | 三片各自峰值的算术平均 MNm，再计算 $(1-\overline L_{\mathrm{RL}}/\overline L_{\mathrm{PID}})\times100$；**正数是载荷降低，负数是变差**。 |
| `blade_{1,2,3}_load_reduction_percent` | 每片 $(1-L_{\mathrm{RL},i}/L_{\mathrm{PID},i})\times100$；JSON 为 `blade_load_reduction_percent` 三元素数组。 |
| `fallback_fraction`、`ood_mean` | 控制步回退比例（0～1）及 OOD 分数均值；回退率不等于 load-reduction。 |

例如风 8 四轮该汇总的 `energy_delta_percent` 均为负（第 1 轮约 -0.474%，第 4 轮约 -0.584%），`load_reduction_percent` 也均为负；这只是**该风况的独立推理诊断性趋势**。同一 PID 窗口、相同风况、完整身份/轨迹和固定评价标准未逐一确认时，不可以将这些图直接当成 23 风况正式验收报告。

## 11. 一页式阅读顺序和结论边界

1. 从根目录 `config.json` 确认该次确实是 state20 / 23×4×600、历史 BC v2、诊断身份口径；从 `queue_training_summary.json` 看 4/4 轮和训练过程指标。**不要**拿后来新候选 predictor、compact 日志或新版推理程序字段来填补旧结果。
2. 看小表 `queue_condition_metrics.csv`：先定位 `condition_id + condition_instance_id`，读 `condition_delta_energy_j`（负为电量少）、三片 `condition_delta_load_blade_{i}`（**正为载荷更差**），再看约束违约 `condition_g_*`、是否回退/过滤、当前 replay 存量。按轮平均、按风况平均、先总量后算比例是不同算法；报数字必须写明口径。
3. 想诊断原因，再有选择地取**某一全局步的** `queue_train_step_metrics.csv` 和 `queue_train_update_metrics.csv`，看实际执行模式、已发送物理动作、UDP 时序、Critic/Actor 更新门控；不要通过 40 GB 文件的某一条采样“推断”全部步数。逐步奖励是代理优化信号，不是最终电量/My/实时性验收证明。
4. 看 `inference/` 的单风况 `episode_summary.csv`、`run_summary.json`、`edge_performance_summary.json` 和 `round_progress_summary.csv/json`；看宽表时过滤 `record_type=control_step`，区分仿真时间和真实输出频率。`queue_model_candidate_valid=1`、程序 `status=completed`、Critic 门控健康、毫秒 P99 合格都**不能单独说明策略已发布或 T1 已通过**。
5. 最终工程结论必须另依同窗 PID、三叶根 $|M_y|$ 峰值、真实控制输出和严格的实例/全轨迹身份、无失效/回退门槛以及目标设备 50 Hz 实测来判断。这次 `identity_validation_level=diagnostic`、`policy_acceptance=false`，不提供完整 strict trace 身份或正式验收结果。更具体的指标方法参见[推理指标计算方式](推理指标计算方式.md)，问题与门槛参见[问题总结](问题总结.md)、[控制策略指标（更新版）](控制策略指标_更新版.md)。