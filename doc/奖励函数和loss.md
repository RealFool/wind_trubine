# 当前奖励函数和 Loss

本文以当前活动 profile [WT15_STATE20_RL_20261023_R2](../RL/configs/WT15_STATE20_RL_20261023_R2/state20_physical_23wind_round600_training.json) 中声明的 state20/BC v3/TD3+BC 配置为准。schedule 默认 23 风况×4 轮×600 s，rounds 可配置；轮数变化不改变此处的 reward/Loss 数学定义。profile 仍是离线开发候选，`authorized=false`，不能写作训练完成或策略验收。R1 六轮身份和 SHA 保留在[历史归档](../RL/configs/WT15_STATE20_RL_20261022_R1/release_manifest.json)。实现入口：[RewardFunction](../RL/utils/reward.py)、[TD3Agent](../RL/agents/td3.py)、[训练队列](../RL/train_rl.py)。

## 1. 动作与观测

Actor 输入是 20 维 state-aware observation：10 个基础物理量、2 个上一步实际发送动作特征、5 个 optimization-state one-hot 和 3 个 control-region one-hot。B4 按状态/区域预测基线动作，Actor 输出二维有界 residual。保护、动作边界和速率限制后才得到实际发送动作。

设 B4 基线动作为 $b_t$，状态包络为 $L(s_t)$，Actor normalized residual 为 $r_t$：

$$
a_t^{desired}=b_t+r_t\odot L(s_t),\qquad r_t=0.25\tanh(f_\theta(o_t))\in[-0.25,0.25]^2.
$$

Replay 记录的是实际发送动作相对于同帧 B4 基线反算的 residual，而不是未经保护的 Actor 原始输出。当前工程 request torque high 为 $21{,}586{,}451.33303\ \mathrm{Nm}$，pitch high 为 $0.43\ \mathrm{rad}$；它们与未来 My 安全界不是一回事。

## 2. 总奖励与当前权重

总体形式是相对 PID 功率收益减去载荷、持续功率缺口和动作连续性代价：

$$
R_t=w_P C_P-w_{My}C_{My}-w_{My,cont}C_{My,cont}-w_{deficit}C_{deficit}-w_{pitch}C_{pitch}-w_{torque}C_{torque}-w_{dual}C_{dual}.
$$

当前配置中主要基础权重为：相对 PID 电量 1.0、叶根峰值增量 1.0、持续载荷超额 0.05、持续功率缺口 0.1、桨距/扭矩变化各 0.0005。转速误差、转速变化、持续低速及独立负功率惩罚已从 reward 中移除；负功率仍可使主相对 PID 功率项或持续缺口项变差。加速度、轴系、指令超可用功率、功率跟踪和 power-floor 权重为 0。状态条件化 reward 对活动功率、载荷、持续缺口及动作变化分量再乘状态权重。

Dual controller 在完整 condition/round 边界按全窗能量和三片峰值调整惩罚权重。它是自适应软惩罚，不能强制满足载荷上限，不能作为安全证明。

## 3. 相对 PID 电量奖励

当前模式为 `relative_pid_energy`，PID reference 来自同工况、同仿真时间戳的 paired reference，严格查找，不插值。设当前 transition 的实测功率为 $P_{t+1}$，PID 参考功率为 $P^{PID}_{t+1}$：

$$
\Delta E_t^{relative}=(P_{t+1}-P^{PID}_{t+1})\Delta t,
\qquad
C_P=\operatorname{clip}\left(\frac{\Delta E_t^{relative}}{300000\ \mathrm{J}},-2,2\right),
\quad\Delta t=0.02\ \mathrm{s}.
$$

$C_P>0$ 表示该时刻 RL 功率高于 PID，$C_P<0$ 表示较低。PID reference 缺失、condition 不匹配或时间无效时应 fail-closed，不伪造参考值。15 MW 是额定功率归一化参考；当前主功率项不是简单的 $P/15\ \mathrm{MW}$。

实现仍支持 `energy_increment`、`available_fraction` 等其他模式，但当前物理开发配置不选用它们。

## 4. 叶根 My 载荷奖励

使用三片叶根 My 的绝对值，并按 condition 逐叶片维护当前运行峰值：

$$
M_{j,t}=\max_{\tau\le t}|My_{j,\tau}|,\qquad
 e_{j,t}=\max\left(\frac{M_{j,t}}{M^{PID}_j}-1,0\right).
$$

当前 `my_peak_increment` 对本步新增加的正超额量惩罚：

$$
C_{My}=\max_j\max(e_{j,t}-e_{j,t-1},0).
$$

另有较轻的持续超额项 $C_{My,cont}=\max_j e_{j,t}$。前者惩罚新增峰值风险，后者持续反映累计峰值超 PID 的程度。两者都是 reward 信号，未把未来预测值用作阻断条件。

## 5. 其他代价项

### 5.1 持续功率缺口

deadband 为 $0.01\times15\ \mathrm{MW}=150\ \mathrm{kW}$。扣除 deadband 后对功率亏损做时间常数为 2 s 的 EWMA，再除以额定功率并裁剪到 $[0,1]$。condition reset 时 EWMA 清零。

### 5.2 动作连续性与独立安全逻辑

桨距/扭矩变化代价分别按 $0.0003\ \mathrm{rad}$ 和 $15\ \mathrm{kNm}$ 归一。转速参考偏差、相邻帧转速变化、持续低速 EWMA 和独立负功率项均不再构成 reward 分量。

本项删除只涉及学习奖励：低速保护、负功率异常分类、Actor residual 回退与安全停止仍由运行状态机和 `SafeResidualController` 独立负责，不因 reward 改动而移除或放宽。

## 6. BC v3 预训练

离线 BC 使用 B4-centered residual 标签，先将 PID demand 按工程边界处理，再除以状态相关 residual envelope 并裁剪。当前冻结 checkpoint 为 [state20 BC v3](../RL/BC_checkpoint/bladed_pid_b4_state20_residual_v3.pt)。其独立 holdout 为 [90003 个样本的评估](../RL/data/b4_centered_residual_bc_state20_v2/val_bc_evaluation.json)：normalized torque/pitch residual MAE 约为 0.05568 / 0.03585。此结果是 teacher-forced 离线拟合误差，不是闭环性能。

v3 预训练配置为 5000 个 BC 更新、batch 256、SmoothL1、冻结 Critic、Actor gradient clip 10。当前在线 queue 配置 `bc.steps=0`、加载 v3 checkpoint 且不导出覆盖文件，所以每次 RL 训练不重跑预训练，但 TD3 Actor loss 仍含 BC 正则项。

$$
L_{BC}=\operatorname{mean}_{i,j}\operatorname{SmoothL1}(\hat r_{i,j}-r^{target}_{i,j}).
$$

## 7. TD3+BC 在线损失

双 Critic 的 target action 加 clipped Gaussian policy-smoothing noise，取两个 target Q 的较小值：

$$
\epsilon\sim\operatorname{clip}(\mathcal{N}(0,0.005^2),-0.01,0.01),\qquad
 a'=\operatorname{clip}(\pi_{\theta'}(o')+\epsilon,-0.25,0.25),
$$

$$
y=R+0.999(1-d)\min(Q'_1(o',a'),Q'_2(o',a')).
$$

两个 Critic 使用 SmoothL1（delta 1.0），学习率 $5\times10^{-5}$，梯度范数裁剪 5。Actor 学习率 $3\times10^{-5}$，每 4 个 Critic update 槽才尝试一次。Actor loss：

$$
L_{RL}=-\lambda\,\mathbb{E}[Q_1(o,\pi(o))],\qquad
\lambda=\frac{1}{\mathbb{E}|Q_1(o,\pi(o))|+10^{-6}},
$$

$$
L_{actor}=L_{RL}+w_{BC}L_{BC}.
$$

`td3_bc_target_ratio=0.5` 是损失项的动态比例下限，不表示最终动作有 50% 来自 PID，也不保证闭环性能；权重上限为 100,000。当前现场探索噪声为 0；`policy_noise=0.005`、`noise_clip=0.01` 仅用于 TD3 target-action smoothing。

## 8. 健康门与配置边界

当前在线配置使用 batch 128、每 30 个环境步最多安排 1 个 Critic update，启用 Critic Q-health 与 Actor optimizer health gate。配置还保留 `actor_q_health_gate` 字段，但在 Actor optimizer gate 已创建时，训练入口不会再创建第二个 Actor-Q gate。`max_target_q_abs_mean=20` 与 `max_actor_q_abs_mean=120` 是数值护栏，不是策略验收指标。

历史 obs12 smoke 的 action normalization、reward reference、数据 manifest、BC checkpoint 和 smoke schedule 均已退出当前路线，不再作为“当前配置”描述。`RL/analyze_b4_residual_smoke.py` 仍可用于读取旧实验输出；它是历史离线分析工具，不是当前 state20 evaluator。
