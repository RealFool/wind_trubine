# WT15_STATE20_RL_20261022_R1

> **历史归档（immutable）：**本目录保留 2026-10-22 的六轮 profile、release manifest、源码 SHA 和授权状态，供审计使用。它不再是活动 route；不要刷新或复用 R1 receipt。当前唯一活动 profile 是 [WT15_STATE20_RL_20261023_R2](../WT15_STATE20_RL_20261023_R2/README.md)，同一 rounds-configurable runtime 默认 4 轮并可按配置选择其他受支持轮数。

这是原 WT15_STATE20_RL_20261022_R1 发行包，用来归档当时的 15 MW state20 控制链、23 风况队列、BC warm-start 和物理载荷影子身份。其组件 schema/模型编号保持原样，详见 [release_manifest.json](release_manifest.json)。

## R1 历史主配置（只读）

- 23 风况 × 6 轮 × 600 s 的开发配置：[state20_physical_23wind_6round_600s_training.json](state20_physical_23wind_6round_600s_training.json)
- Queue manifest：[state20_physical_23wind_6round_600s_training_manifest.json](state20_physical_23wind_6round_600s_training_manifest.json)
- 物理 shadow bundle：[physical_load_shadow_bundle.json](physical_load_shadow_bundle.json)
- Profile SHA/源码身份报告：[profile_build_report.json](profile_build_report.json)

原主配置沿用当时的 state20 BC v3 checkpoint、B4 和 queue-v2 合同；`physical_fir_v2` 仅为只读 shadow。R1 中 `physical_shadow_launch.authorized=false`、`policy_acceptance=false`、`independent_audit=false`、`soft/hard=false`。历史 SHA 和授权字段不得为适配 R2 而修改。

## 版本规则

- `WT15_STATE20_RL_20261022_R1` 是当时的冻结发行 ID；当前活动发行 ID 已切换为 `WT15_STATE20_RL_20261023_R2`。
- `state20` 表示 20 维观测合同；BC `v3`、数据 `v2`、B4 `v3`、接口 `v3`、物理 FIR/D0-R `v2`、UDP queue-v2 均是各自组件的 schema/工件兼容标识，不是并列的项目发行号。
- `physical_shadow_state20_23wind6_candidate_v11` 与 `..._v12` 是保留的候选迭代目录，不再是当前 route。旧 `physical_23wind6_600s_20261016T155444Z_939992` 准备包仍保留审计，但绑定旧候选身份，不能拿来继续当前发行版。
- 四风况 frozen profiles 保留为独立参考路线，不与 23 风况主配置混为同一 queue。

R1 不作为新的实际运行入口。若需后续实际运行，使用 R2 的统一 rounds 参数化 profile 和新 RUN_ID；不要对 R1 manifest/profile report/receipt 做就地刷新，也不要跨 R1/R2 复用 receipt。
