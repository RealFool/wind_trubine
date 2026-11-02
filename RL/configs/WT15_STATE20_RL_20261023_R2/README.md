# WT15_STATE20_RL_20261023_R2

这是当前唯一活动的 23 风况物理 shadow training profile。训练链为 state20 + 状态/区域条件 B4 residual + frozen BC warm-start + TD3+BC + SafeResidualController，并使用不影响动作、奖励和 replay 的 read-only `physical_fir_v2` shadow。

当前 reward 以配对 PID 相对功率、叶根 My 峰值、持续功率缺口和动作连续性为目标，不含独立转速惩罚或独立负功率扣分。转速状态输入、低速保护、`negative_power_abnormal` 分类/回退仍保留；源码与 profile/runtime SHA 已按此配置重建。reward 改动尚未获得新的 Bladed 闭环性能证据。

## 单一 profile 与 rounds 参数

- 唯一 profile：`23wind_round600_training`。
- 默认 `rounds=4`、92 slots；同一 builder/validator/operator receipt/DLL `full` schedule 支持 1–8 轮。9 轮及以上超出固定 18 GiB worst-case physical-shadow record budget，会被拒绝。
- 默认配置：[state20_physical_23wind_round600_training.json](state20_physical_23wind_round600_training.json)
- 默认 manifest：[state20_physical_23wind_round600_training_manifest.json](state20_physical_23wind_round600_training_manifest.json)
- 源码/config/manifest/DLL SHA 闭合报告：[profile_build_report.json](profile_build_report.json)
- 只读 physical-FIR bundle：[physical_load_shadow_bundle.json](physical_load_shadow_bundle.json)
- 当前活动 release identity：[release_manifest.json](release_manifest.json)

每个 RUN_ID 的 manifest 记录该批实际 rounds、展开后的逐 slot 顺序和 slot 数；runtime identity、operator confirmation、receipt 及 Windows persisted queue state 绑定同一个 schedule hash。改变 rounds 或源码必须创建新 RUN_ID/package 并重新确认；禁止原地改写配置、复用 receipt 或续用 DLL 队列 state。

## 历史 R1

[WT15_STATE20_RL_20261022_R1](../WT15_STATE20_RL_20261022_R1/README.md) 是 23×6 的 immutable historical archive。其 config/manifest/profile report/release SHA 原样保留用于审计，不再是并行活动代码路线。R2 receipt 不适用于 R1，R1 receipt 也不能用于 R2。

## 授权边界

`release_ready=false`、`physical_shadow_launch.authorized=false`、`policy_acceptance=false`、`independent_audit=false`、`soft_authorized=false`、`hard_authorized=false`。profile build 和 RUN_ID preparation 均不会启动 Bladed。实际在线运行前须完成人工工程/DLL/seed/case identity 确认及适用的独立运行许可；operator receipt 只是开发运行确认，不构成六门禁通过、策略验收、独立审计或物理安全授权。
