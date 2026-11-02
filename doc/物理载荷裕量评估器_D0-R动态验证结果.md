# 物理载荷裕量评估器：D0-R 动态验证结果

日期：2026-10-10。结论：**物理锚定＋因果低阶线性动态增量明显改善短窗精度，联合余量达到本轮开发门槛；但仍只是复用 validation 的发展证据，不能授权生产预警或未执行候选动作的安全判定。**

实施计划：[doc/载荷峰值预测预警影子模型实施计划.md](doc/载荷峰值预测预警影子模型实施计划.md)。本轮结果：[RL/diagnostics/physical_load_margin_dynamic_v2/README.md](RL/diagnostics/physical_load_margin_dynamic_v2/README.md)；完整机器报告：[RL/diagnostics/physical_load_margin_dynamic_v2/report.json](RL/diagnostics/physical_load_margin_dynamic_v2/report.json)；原有摘要图：[RL/diagnostics/physical_load_margin_dynamic_v2/diagnostic.png](RL/diagnostics/physical_load_margin_dynamic_v2/diagnostic.png)。

冻结 validation 细化图见 [RL/diagnostics/physical_load_margin_dynamic_v2_visualizations/README.md](RL/diagnostics/physical_load_margin_dynamic_v2_visualizations/README.md)；该次只读 SHA 已冻结 CSV/报告，不重拟合、不重校准、不触碰旧 audit/raw：

绘图实现：[RL/plot_physical_load_margin_dynamic.py](RL/plot_physical_load_margin_dynamic.py)。它只读取冻结 D0-R `report.json` 和 `validation_samples.csv`；复现须给绘图器一个新输出目录，已有结果不会覆盖。

- [点预测精度与低估尾部](RL/diagnostics/physical_load_margin_dynamic_v2_visualizations/prediction_metrics.png)
- [三片×双短窗经验联合覆盖与余量宽度](RL/diagnostics/physical_load_margin_dynamic_v2_visualizations/joint_margin_validation.png)
- [逐叶片 origin 的预测—实际未来峰值散点](RL/diagnostics/physical_load_margin_dynamic_v2_visualizations/prediction_vs_actual_scatter.png)
- **[PID全窗极限载荷预测锚点放大图](RL/diagnostics/physical_load_margin_dynamic_v2_visualizations/pid_peak_prediction_anchors.png)**
- [18个峰值锚点逐点明细](RL/diagnostics/physical_load_margin_dynamic_v2_visualizations/pid_peak_anchor_examples.csv)；[输入及代码身份/范围清单](RL/diagnostics/physical_load_margin_dynamic_v2_visualizations/plot_manifest.json)

## 1. 本轮实际验证了什么

1. 保持 v1 动压/相位灰箱、实际执行器与转子闭式响应、当前实测 My 锚定不变，增加**直接多步线性 FIR 残差增量**。
2. 利用当前/历史三片 My、静态残差、因果载荷差分/变化率、风速差分、已有机舱加速度、当前桨距/转速/相位等内部派生量。**没有新增外部状态、扰动、奖励、传感器或 UDP/Actor 输入。**
3. train 拟合条件误差尺度，calibration 冻结三片×0.10/0.20 s 六维联合经验余量；比较逐片 95%、固定联合 95%、条件化联合 95%。
4. 六个候选均使用相同 origins 比较；两点预测模型及三种余量的比较使用共同模型/尺度支持域，不通过删掉难样本制造改善。

入口：[RL/demo_physical_load_margin_dynamic.py](RL/demo_physical_load_margin_dynamic.py)。计算模块：[RL/utils/physical_load_margin_dynamic.py](RL/utils/physical_load_margin_dynamic.py)。冻结协议：[RL/configs/physical_load_margin_dynamic_demo_v2.json](RL/configs/physical_load_margin_dynamic_demo_v2.json)。

### 数据角色与证据级别

| 角色 | wind ID | 用途 |
|---|---|---|
| train | 3、4、6、7、9、10、12、14、16、18、20、22、24 | 拟合物理曲面、FIR、误差尺度 |
| validation | 5、13、21 | 选择冻结候选；随后评价仅作为复用验证集开发诊断 |
| calibration | 8、15、23 | 只冻结经验余量，不调点模型 |
| 旧 audit，禁止读取 | 11、17、19、25 | 本轮不打开、不哈希、不评价 |

实际访问 19 个 CSV、570,019 行，均准确 [0,600] s、dt=0.02 s，白名单非有限输入=0、缺帧=0。只解析原 17 个字段，忽略 `I_PitchPosition2/3`。旧 v1 audit 已公开，**不能在改版后仍当作新的独立 holdout**；本轮也不将 calibration 偷换成独立 audit。

### 模型与选择

对每个严格未来帧 $k=1,\ldots,25$，训练目标为未来有符号 My 减去当前实测 My 锚定的物理 rollout；未来 My 只作为 train 监督目标，不进入特征。预测再对各窗取 max_abs。

$$
\widehat M^{v2}_{j,t+k}=\widehat M^{phys}_{j,t+k}(u_{\mathrm{PID,ZOH}})
 +10^7\mathbf x_{j,t}^{\mathsf T}\mathbf b_{j,s,k}.
$$

这是有限窗直接回归，不是递归 ARX、真实模态估计或完整气弹性模型；没有数值递归发散并不等于系统稳定保证。**v2 没有替代动作预测 API**，只验证录制 PID origin demand 的近似 ZOH 情形。

选择规则：逐 case/0.10、0.20 s 等权的 NMAE＋0.25×归一化正部低估 p95，严格从以下六个预先冻结候选中选；0.50 s 不参与选择。

| 历史帧 lags | ridge | 开发选择分数 |
|---|---:|---:|
| 1、2、5、10 | 0.0001 | 0.019190 |
| 1、2、5、10 | 0.001 | 0.019491 |
| 1、2、5、10 | 0.01 | 0.020375 |
| 1、2、5、10、20、40 | 0.0001 | **0.018922，选中** |
| 1、2、5、10、20、40 | 0.001 | 0.019310 |
| 1、2、5、10、20、40 | 0.01 | 0.020196 |

选中 `fir_lag40_ridge0.0001`，最长历史 0.80 s、52 个内部线性基函数。最小共同评分域比例约 99.83%。完整选择记录：[RL/diagnostics/physical_load_margin_dynamic_v2/frozen_selection.json](RL/diagnostics/physical_load_margin_dynamic_v2/frozen_selection.json)。

## 2. 点预测和低估尾部：短窗改善是真实计算结果

极值放大图将 wind5/13/21、三片叶片的 PID 全窗最大载荷用作**离线目标锚点**，并显示目标峰前 origin 的0.10/0.20 s预测、实际未来窗标签、origin时已观测累计峰值和 PID 条件经验余量示意。锚点选取为有效、模型支持、近似 ZOH 的最晚采样时刻 $t$，要求 $t<t_{PID,max}\le t+H$ 且该 future-window 的实际峰值等于同叶片 PID 全窗峰值。共18个锚点；距离真实离线 PID 峰时刻0.02–0.10 s，中位0.06 s；点预测为该叶片 PID 最大值的98.39%–100.46%。示意的点预测＋条件经验余量为103.96%–115.39%，**不是 RL 覆盖保证或安全边界**。

这些锚点是看完整条 PID 轨迹后事后筛选的 validation origins，峰时刻与窗口 My 是离线标签；它们不是在线预警事件召回率，不证明该提前量足以克服通信、执行器和结构响应滞后，也不等于 RL 候选动作因果验证。重复/重叠 origin 不独立；更多限定见上述可视化 README。

下表全部为本轮同一 train 重新拟合的 v1 基线与 v2 在**相同 validation origins**上的比较，不与旧 audit 数字交叉比较。MAE/p95/p99 单位 MNm；p95/p99 为三片各自分位数的平均，并非独立置信界。

| H | origin 数 | v1 MAE | v2 MAE | MAE 降幅 | v1 低估 p95 | v2 低估 p95 | p95 降幅 | v2 低估 p99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.10 s | 17,970 | 0.5483 | **0.2991** | **45.46%** | 1.9137 | **1.1088** | **42.06%** | 1.9898 |
| 0.20 s | 17,967 | 0.9266 | **0.6565** | **29.15%** | 3.3349 | **2.5301** | **24.13%** | 4.3085 |
| 0.50 s，仅诊断 | 17,955 | 1.7386 | 1.3215 | 23.99% | 6.1072 | 4.6486 | 23.88% | 7.5875 |

0.10/0.20 s NMAE 为 0.006320/0.013953，即约 0.632%/1.395% 的同 case PID 极值。Nm-MAE 与 NMAE 的降幅略有差异，因为各 case 的参考峰值不同，不能混用口径。wind5/13/21 两个短窗均改善，无 case 超过预注册最大 1.05 倍退化门槛。

### 更紧的 PID 计划匹配诊断

在未来 demand 与 origin demand 偏差≤20 kNm/0.00035 rad 的评价子集上，短窗改善仍存在；这些未来动作只用于筛评价标签，不进入模型支持或特征。

| H | 更紧子集 origin 数 | v1 MAE | v2 MAE | MAE 降幅 |
|---|---:|---:|---:|---:|
| 0.10 s | 9,114 | 0.4400 | 0.2427 | 44.84% |
| 0.20 s | 4,622 | 0.6308 | 0.4501 | 28.64% |

仍有必须公开的限制：

- 普通近峰（≥0.95 PID）低估 p95：0.10 s 从 1.4515 降到 1.0068 MNm；0.20 s 从 3.0541 降到 2.5885 MNm。
- **更紧子集的 0.20 s 近峰低估 p95 仅从 1.8272 降到 1.7743 MNm，约改善 2.90%。** 平均精度进步不代表所有关键尾部同幅改善。
- **0.50 s 普通近峰低估 p95 从 6.0183 增到 6.6659 MNm，反而恶化约 10.76%。** 不扩展到长窗预警，不用整体 MAE 改善掩盖该退化。
- 百万 Nm 级低估尾部仍远大于此前小候选动作的 kNm 级数值敏感度。不能据此声称已具备细小候选动作的因果排序精度，也不能从误差量级直接断言增量误差必然同样大。

## 3. 联合校准有效，但“条件化”不一定更紧

train 用固定岭回归拟合 log(max(abs peak error,100 kNm)) 的因果条件尺度 $s_{j,H}(x_t)$；calibration 对下式取经验 95% 分位数 $q_s$：

$$
z_t=\max_{j\in\{1,2,3\},H\in\{0.10,0.20\}}
 \frac{(P^{true}_{j,t,H}-\widehat P_{j,t,H})_+}{s_{j,H}(x_t)},
\qquad e_{j,t,H}=q_s s_{j,H}(x_t).
$$

这只校准每个 eligible origin 的六个值同时覆盖，不校准整轨迹最大值、全部时刻不超限或独立 seed 置信度。overlapping origins 不独立。

| 区域 | 评价 origins | 逐片 95% 的六维覆盖 | 固定联合覆盖 | 条件联合覆盖 | 固定平均余量 MNm | 条件平均余量 MNm |
|---|---:|---:|---:|---:|---:|---:|
| below_rated | 6,759 | 89.21% | 96.86% | **98.71%** | 1.8933 | 2.7550 |
| transition | 3,451 | 91.42% | 99.01% | **97.83%** | 4.2307 | 3.8263 |
| above_rated | 7,757 | 80.25% | 95.29% | **95.44%** | 4.6626 | 4.6583 |

- **固定联合余量已在三类区域达到开发覆盖门槛。** 不能把“逐片余量改成联合余量”的收益全部归功于条件化模型。
- 条件化在 below_rated 更宽（约＋45.5%），换来更高经验覆盖；transition 更窄（约－9.6%）；above_rated 宽度近乎相同。未证明条件化在所有区域都更经济。
- 全区域平均：固定联合覆盖 96.60%、平均余量 3.5379 MNm、近峰负样本警报比例 1.42%；条件联合覆盖 97.13%、平均余量 3.7825 MNm、负样本警报比例 **3.56%**。需权衡宽度/覆盖/误报，不能只看 coverage。
- 同方法对比 v1，动态模型的条件余量平均由 5.7151 降到 3.7825 MNm，改善约 33.8%；不是硬安全界。
- 已知预测约 99.83%；**余量可用率 100% 的分母只是 17,967 个共同点预测/近似计划合格 origins**，不是整个原始数据或真实运行的 100% 可靠率。
- cal 计数：below 为 wind8 的 5,943＋wind15 的 61；transition 仅 wind15 的 889；above 为 wind15 的 5,036＋wind23 的 5,986。仍没有多 seed 独立保证。

冻结余量：[RL/diagnostics/physical_load_margin_dynamic_v2/frozen_joint_calibration.json](RL/diagnostics/physical_load_margin_dynamic_v2/frozen_joint_calibration.json)。逐行点预测/余量/未知域：[RL/diagnostics/physical_load_margin_dynamic_v2/validation_samples.csv](RL/diagnostics/physical_load_margin_dynamic_v2/validation_samples.csv)。

## 4. 开发门禁、事件和资源

预注册的两短窗 MAE/p95 各至少改善 10%、每 case 不退化超过 5%、各区域六维经验覆盖≥95%、余量可用≥90%、负样本警报≤10%均通过。结论分别为：

- 点预测：`PROMISING_ON_REUSED_VALIDATION_ONLY`。
- 联合余量：`PROMISING_REUSED_VALIDATION_MARGIN_ONLY`。
- 接线：**`BLOCKED_NO_FRESH_INDEPENDENT_AUDIT`，始终不改变。**

0.95 近峰 proxy 共 40 个逐叶片事件，两个短窗均 40/40 按时、晚报=0；理想仿真提前量中位数 0.06/0.16 s。事件统计不限未来计划匹配，但按有效连续段计数；跨 invalid 段机会不计，起点删失单列。本轮无删失。严格超自身完整 PID 窗极值正例=0、recall 不可估计，不能把近峰 100% 当作严格超限检出率或在线保护效果。

验证：[RL/tests/test_physical_load_margin_dynamic.py](RL/tests/test_physical_load_margin_dynamic.py)新增 20 项 fixture 均通过；与原 v1 合计 39 项通过。RL 全量回归 **306 项、1 项跳过、其余通过**（16.07 s，BLAS 单线程）；包含本机 Fake UDP/临时数据，不是物理 Bladed。

真实离线实验 21.87 s，Linux 峰值 RSS 270.02 MiB，输出约 40.39 MiB。该指标不等于逐步推理延迟、目标 Windows T1 或可用预警提前量。

冻结身份均已复核：

- 开发协议规范化 SHA-256：`0cbd7ccea351940bea5d13c4fb7b16c1225e3472c4bf1365112d8d07e2c2956a`。
- 实际 19-case 输入清单 SHA-256：`bd1a4537d02987dd675dd9e0acab64987602bb4042644f56294a0f5743b27bab`。
- 动态模型 SHA-256：`c8286005ce23afc13490906461d80b87245d6bca72a10d2f7f71b53d41baad36`。
- 输出工件、实际物理模型、尺度/校准绑定、7 个相关源码文件及 19 个允许 raw 的哈希全部匹配报告；当时控制链权重保持未变。本结果是历史 D0-R 验证记录，不构成当前 NN 模型身份或动作保护授权。

## 5. 下一步与当前停止点

本轮回答了“已有历史是否能显著提高载荷短窗计算精度”和“正确的六维联合余量是否能改善经验覆盖”：**可以，且已实跑；但不能据此直接宣布已达到候选动作保护的可用精度。**

下一步按顺序：

1. 核定历史记录器 `O_*` 的 demand/actual 来源、测量→发送→setter 作用区间及真实工程身份。示例工程数值/SHA 核验不证明全部历史 case 一致。
2. 固定本轮候选网格和模型，不再根据已看过的 validation/calibration 扩网格试到通过。没有新的独立数据时只保留发展结论。
3. 在同白名单、同真实工程/准确 PID 窗口下获得新独立轨迹，再检验短窗点误差、条件/固定联合余量、近峰尾部、误报及 unknown；0.50 s 暂不扩围。
4. 未执行动作局部因果证据和来源/时序未闭合前，v2 不移用到 RL 候选动作。生产合同/旁路/Windows/资源门禁仍按原计划逐项验收。

**未修改生产控制、Actor/BC、reward、UDP 或状态/扰动清单；未启动 Bladed；未授权 soft/hard。开发通过不是生产通过，统计余量不是确定保护。**