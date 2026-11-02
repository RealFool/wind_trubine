# D0-R：物理载荷裕量动态改进验证 v2

**范围：仅开发验证，不是新的独立 audit，不授权生产接线。**

- 动态候选：fir_lag40_ridge0.0001；有限窗线性 FIR，无神经网络、无递归模态求解。
- 数据：train [3, 4, 6, 7, 9, 10, 12, 14, 16, 18, 20, 22, 24]；validation [5, 13, 21]；calibration [8, 15, 23]。
- 旧 audit [11, 17, 19, 25] **未打开/未哈希/未参与评价**。
- validation 用于选择历史长度/岭参数，随后指标和覆盖只能称“复用验证集开发结果”。
- train 拟合物理曲面、每个未来帧的有符号载荷增量残差、条件误差尺度；calibration 只冻结余量。
- 输入仍为现有白名单；使用历史差分/相位/加速度等内部派生量，不增加 UDP/Actor/观测变量。
- 原始 demand 来源及工程身份仍待核定；只验证已有 PID 轨迹，不验证未执行 RL 候选因果效果。
- 点预测决策：**PROMISING_ON_REUSED_VALIDATION_ONLY**。
- 联合余量决策：**PROMISING_REUSED_VALIDATION_MARGIN_ONLY**。
- 接线决策：**BLOCKED_NO_FRESH_INDEPENDENT_AUDIT**。

## 1. 因果模型与选择规则

- 保持 v1 实测 My 锚定物理 rollout，回归未来 20 ms 各帧的有符号残差增量，最后取 max_abs。
- 历史只读到 origin；未来 My 仅作为监督目标，未来 wind/actual/action 不进入特征。
- 所有候选在同一支持域 origins 上比较，避免通过排除难样本制造改善。
- 选择分数为逐 case/短窗等权的 NMAE + 0.25×归一化低估 p95；只从六个冻结候选中选。
- 错误尺度是 train 残差的线性 log-scale 拟合，校准样本再决定分位数；不是新实测变量或硬界。

## 2. 复用验证集的点预测

| H | v1 NMAE | 动态 FIR NMAE | 改善 | 动态 MAE (MNm) | 动态低估 p95 (MNm，三片平均) |
|---|---:|---:|---:|---:|---:|
| 0.10s | 0.011503 | 0.006320 | 45.06% | 0.2991 | 1.1088 |
| 0.20s | 0.019608 | 0.013953 | 28.84% | 0.6565 | 2.5301 |
| 0.50s | 0.037181 | 0.028521 | 23.29% | 1.3215 | 4.6486 |

v1 是本轮相同 train/validation、相同 origin 上重算的基线；不是与旧 audit 数字交叉比较。0.50 s 仅诊断。

| case | H | v1 NMAE | 动态 NMAE |
|---|---|---:|---:|
| wind5 | 0.10s | 0.006840 | 0.004018 |
| wind5 | 0.20s | 0.012754 | 0.009379 |
| wind13 | 0.10s | 0.009354 | 0.005017 |
| wind13 | 0.20s | 0.015661 | 0.010923 |
| wind21 | 0.10s | 0.018316 | 0.009925 |
| wind21 | 0.20s | 0.030408 | 0.021555 |

## 3. 三片×双窗联合余量

在相同可用误差尺度/模型域内，比较逐片 95%、全六维固定联合余量、条件化全六维联合余量。数值是在复用 validation 上观察的经验覆盖，不具有独立置信保证。

| 模型 | 余量方法 | 区域 | origin | 六维联合覆盖 | 平均余量 (MNm) |
|---|---|---|---:|---:|---:|
| v1_greybox | legacy_per_blade95_nm | below_rated | 6759 | 90.52% | 1.4071 |
| v1_greybox | legacy_per_blade95_nm | transition | 3451 | 92.67% | 3.1903 |
| v1_greybox | legacy_per_blade95_nm | above_rated | 7757 | 81.78% | 3.5021 |
| v1_greybox | joint_constant95_nm | below_rated | 6759 | 96.98% | 2.4857 |
| v1_greybox | joint_constant95_nm | transition | 3451 | 99.30% | 5.3641 |
| v1_greybox | joint_constant95_nm | above_rated | 7757 | 95.19% | 5.9680 |
| v1_greybox | joint_conditional95 | below_rated | 6759 | 98.03% | 3.9115 |
| v1_greybox | joint_conditional95 | transition | 3451 | 97.42% | 5.5350 |
| v1_greybox | joint_conditional95 | above_rated | 7757 | 95.17% | 7.3669 |
| dynamic_fir | legacy_per_blade95_nm | below_rated | 6759 | 89.21% | 0.9732 |
| dynamic_fir | legacy_per_blade95_nm | transition | 3451 | 91.42% | 2.1472 |
| dynamic_fir | legacy_per_blade95_nm | above_rated | 7757 | 80.25% | 2.4859 |
| dynamic_fir | joint_constant95_nm | below_rated | 6759 | 96.86% | 1.8933 |
| dynamic_fir | joint_constant95_nm | transition | 3451 | 99.01% | 4.2307 |
| dynamic_fir | joint_constant95_nm | above_rated | 7757 | 95.29% | 4.6626 |
| dynamic_fir | joint_conditional95 | below_rated | 6759 | 98.71% | 2.7550 |
| dynamic_fir | joint_conditional95 | transition | 3451 | 97.83% | 3.8263 |
| dynamic_fir | joint_conditional95 | above_rated | 7757 | 95.44% | 4.6583 |

未知/排除比例和逐窗逐片覆盖详见机器报告；没有覆盖组不能用零余量填补。加大余量不等于计算精度提高，需同时检查误报及平均宽度。

## 4. 近峰与开发门禁

| H | 0.95 近峰事件 | 按时 | 晚报 | 无模型机会 | 按时 recall |
|---|---:|---:|---:|---:|---:|
| 0.10s | 40 | 40 | 0 | 0 | 100.00% |
| 0.20s | 40 | 40 | 0 | 0 | 100.00% |

这是未限制未来计划匹配的 recorded-PID 近峰 proxy 和理想离线提前量；跨无效段机会不计，起点删失事件单列并排除。不是严格超 PID recall；严格正例为 0，recall 不可估计。

| 检查 | 范围 | 实测 | 门槛 | 状态 |
|---|---|---:|---:|---|
| peak_MAE_reduction_vs_v1 | 0.10 | 0.454568 | 0.1 | PASS |
| underestimate_p95_reduction_vs_v1 | 0.10 | 0.420606 | 0.1 | PASS |
| known_origin_fraction | 0.10 | 0.998333 | 0.9 | PASS |
| case_MAE_ratio_to_v1 | wind5/0.10 | 0.587461 | 1.05 | PASS |
| case_MAE_ratio_to_v1 | wind13/0.10 | 0.536371 | 1.05 | PASS |
| case_MAE_ratio_to_v1 | wind21/0.10 | 0.541800 | 1.05 | PASS |
| peak_MAE_reduction_vs_v1 | 0.20 | 0.291510 | 0.1 | PASS |
| underestimate_p95_reduction_vs_v1 | 0.20 | 0.241334 | 0.1 | PASS |
| known_origin_fraction | 0.20 | 0.998333 | 0.9 | PASS |
| case_MAE_ratio_to_v1 | wind5/0.20 | 0.735520 | 1.05 | PASS |
| case_MAE_ratio_to_v1 | wind13/0.20 | 0.697428 | 1.05 | PASS |
| case_MAE_ratio_to_v1 | wind21/0.20 | 0.708878 | 1.05 | PASS |
| 3_blades_2_windows_empirical_coverage | below_rated | 0.987128 | 0.95 | PASS |
| 3_blades_2_windows_empirical_coverage | transition | 0.978267 | 0.95 | PASS |
| 3_blades_2_windows_empirical_coverage | above_rated | 0.954364 | 0.95 | PASS |
| margin_known_fraction | all | 1.000000 | 0.9 | PASS |
| near_peak_upper_false_positive_fraction | all | 0.035586 | 0.1 | PASS |

## 5. 身份与下一步边界

- 冻结开发协议 SHA-256：`0cbd7ccea351940bea5d13c4fb7b16c1225e3472c4bf1365112d8d07e2c2956a`。
- 实际读取 19 个 case / 570019 行，清单 SHA-256：`bd1a4537d02987dd675dd9e0acab64987602bb4042644f56294a0f5743b27bab`。
- 动态模型 SHA-256：`c8286005ce23afc13490906461d80b87245d6bca72a10d2f7f71b53d41baad36`；各文件哈希、读取阶段记录在报告。
- 运行 21.87 s，Linux 峰值 RSS 270.02 MiB；不是 Windows T1。
- 旧 v1 协议/工件/报告不覆盖；本轮无 Bladed/生产 UDP/Actor/BC/TCN 更新。
- 没有新独立数据证据，不能把“validation 选出来的最好模型”当作已达可用精度。
- 保留全部候选分数/失败门禁；若动态点预测改善但覆盖失败，继续检查条件尺度/校准域，不调到旧 audit 通过。
- 即使开发点预测和余量均达标，仍需动作/工程来源核定、未执行动作局部验证及获准后的独立真实 evidence，soft/hard 仍关闭。
