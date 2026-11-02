# RL 控制与训练

## 当前统一发行版：WT15_STATE20_RL_20261023_R2

当前唯一项目发行 ID 是 **WT15_STATE20_RL_20261023_R2**。版本规则及工件 SHA 清单见[发行说明](configs/WT15_STATE20_RL_20261023_R2/README.md)和[release manifest](configs/WT15_STATE20_RL_20261023_R2/release_manifest.json)。

- 当前主配置为 state20 + 状态/区域条件 B4 + 冻结 BC v3 warm-start + TD3+BC + 安全控制器；[23 风况 × 600 s rounds-configurable profile](configs/WT15_STATE20_RL_20261023_R2/state20_physical_23wind_round600_training.json) 默认 4 轮、支持 1–8 轮，其 queue/bundle/report 同属上述发行版。
- `physical_fir_v2` 是并行的只读载荷影子，不参与动作、奖励或回放；soft/hard、独立审计、counterfactual 和策略验收均未授权。
- 架构、模块开关、实际运行进度和遗留版本边界：[模型架构与物理影子说明](../doc/RL模型架构与物理影子模型说明.md)
- 操作步骤及当前发行版边界：[训练操作笔记](../doc/训练操作笔记.md)

该发行版目前是**离线对齐、未启动的开发候选**，其物理启动保持 blocked。`WT15_STATE20_RL_20261022_R1` 是保留原 SHA 的六轮历史归档，`physical_23wind6_600s_20261016T155444Z_939992` 是绑定旧候选身份的批次，二者都不能作为 R2 的续跑入口；此前可核验的历史摘要为 4/6 轮，不计入 R2 运行进度。R2 尚未准备或启动任何 RUN_ID。

## 当前数据与模型工件

- state20 BC 数据合同：[BC 数据 manifest](data/b4_centered_residual_bc_state20_v2/manifest.json)，20 维因果观测，训练集来自 `raw`，独立 holdout 来自 `raw_infer`。
- 状态/区域统计基线：[state20 B4 baseline](data/region_aware_joint_baseline_state20_3_to_25_v3.json)
- 冻结行为克隆初始策略：[BC v3 checkpoint](BC_checkpoint/bladed_pid_b4_state20_residual_v3.pt)
- 控制状态/动作包络：[state-action artifact](data/state_action_envelope_3_to_25_v1.json)
- 当前影子后端：[physical_fir_v2 bundle](configs/WT15_STATE20_RL_20261023_R2/physical_load_shadow_bundle.json)

## 入口

- 在线训练：`RL.train_rl`，由配置选择 queue-v2、TD3+BC、replay、奖励和可选只读影子。
- Bladed 推理：`RL.infer.run_bladed_inference`，载入冻结 checkpoint，通过同一安全控制器发送动作。
- BC holdout 评估：`RL.evaluate_b4_residual_bc`；不进行在线训练。
- B4/BC 数据构建、影子评价和作图脚本均为单独的离线工具，不会自动加入在线控制链。

## 历史版本说明

`WT15_STATE20_RL_20261023_R2` 是当前唯一项目发行号；`WT15_STATE20_RL_20261022_R1` 是六轮历史归档。`state20` 是 20 维观测合同；BC `v3`、B4 `v3`、数据 `v2`、预测器接口 `v3`、D0-R/FIR `v2`、UDP `queue-v2` 等是组件自身的兼容或协议标识，保留原名以维持 SHA、checkpoint 和 schema 校验，不代表并行项目发行版。`candidate_v11`、`candidate_v12` 是历史 profile 迭代目录；当前 23wind rounds-configurable route 已绑定统一发行目录。

旧 obs12、state20 v2 配置与 checkpoint 已从活动源码/配置目录清理；历史训练输出仍留在 RL/experiments 中供结果审计，但不再承诺可从已清理的旧模型文件复跑。不要将旧 21 MNm 动作边界、obs12 normalization 或旧 fingerprint 与 state20/v3 工件混用。

当前发行版未配置 NN 载荷 checkpoint。共享的 predictor interface/dataset 合同与通用 loader 代码仍服务 state20 合同/离线工具；其他历史实验和 UDP 兼容分支仍按各自审计边界保留。
