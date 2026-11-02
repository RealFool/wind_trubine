# 从原始 PID 数据重建 state20 训练链（WT15）

**文档日期：2026-10-23**  
**适用对象：**当前 15 MW Bladed 工程、state20 观测合同、B4 残差 BC 与 TD3+BC 训练路线。  
**目的：**说明从已经采集的原始 PID CSV 逐步重建状态/动作标定、PID 参照、B4、BC 数据、BC Actor 和后续 RL 训练所需的工件。

> 本文从 PID 原始 CSV 已经存在开始；不包含启动 Bladed 采集 PID 数据的步骤。前处理使用固定 canonical 路径并按本流程顺序重建、覆盖 baseline/BC 数据/config/checkpoint。活动训练统一为一个 23 风况 × 600 秒 queue-v2 + TD3+BC + read-only `physical_fir_v2` shadow route；`rounds` 是该 route 的配置参数，默认 4，范围 1–8，并由同一个 profile builder、validator、RUN_ID package 和 Windows `full` schedule 处理。R1 六轮 release 文件保留为历史归档，不再是另一条活动代码路线。

> **场景（23×4）：**本文对应 23 风况 × 600 s、`full`、92 slot（默认 4 轮）；4 风况 × 4 轮（16 slot、`smoke_4x4`）的操作见[训练操作笔记](训练操作笔记.md)。两套并列，不互相替代。

## 1. 先读清楚当前状态

当前活动 profile 位于 [WT15_STATE20_RL_20261023_R2](../RL/configs/WT15_STATE20_RL_20261023_R2/README.md)，默认目标为 23 风况 × 4 轮 × 600 秒（92 slots）；也可对同一 profile 配置其他受支持轮数。**raw PID 前处理、B4 residual 数据、resolved control contract、BC config 和 BC checkpoint 按固定 canonical 路径覆盖**，不再要求另建前处理版本目录。R1 6-round profile 和 receipt 已归档，保留旧 SHA，不会被新 route 改写或复用。

覆盖会改变 canonical 文件 SHA 和 resolved control fingerprint，因此历史 R1 manifest/profile report 与新 canonical 工件不再哈希闭合。不要拿旧 R1 receipt 替新配置背书；要记录新的 RUN_ID、实际 rounds、queue manifest 和 artifact hashes。[R1 归档发行身份](../RL/configs/WT15_STATE20_RL_20261022_R1/release_manifest.json#L1-L72) [当前 R2 活动身份](../RL/configs/WT15_STATE20_RL_20261023_R2/release_manifest.json)

当前 R2 是**离线开发候选，不是已授权的 Bladed 训练批次**：`release_ready=false`，真实 Bladed 尚未为该发行版启动，策略验收、独立审计、soft/hard 和 RL margin coverage 均未授权。[R2 release manifest](../RL/configs/WT15_STATE20_RL_20261023_R2/release_manifest.json)

**区域分类仍使用当前 state20 contract 的滤波风速与滞回阈值；没有因 R2 rounds 参数化而改变。** 对 raw 全量 23 个 condition 做的 rotor-only 可分性检查显示，转速单变量不能稳定分开 transition 与 above-rated；超速 trip 倍率也不等同于三个控制分区。详细样本统计、condition-level 留一验证及如需更换时的依赖重建范围见[基于转速的运行分区可行性分析](基于转速的运行分区可行性分析.md)。该分析未修改 classifier 或模型工件。

流程分三段：

- **第 2–7 节：**从 raw/raw_infer 重建 provenance、state/action、paired PID、S2/U1、转速参考、B4 和 BC residual 数据，写回固定 canonical 路径。
- **第 8–9 节：**覆盖 resolved control contract 和 BC config，强制重新做 BC-only 训练并覆盖 canonical BC checkpoint；BC 日志自动落到带 timestamp 的实验目录。
- **第 10 节：**用同一个 rounds 参数化 physical-FIR profile 生成 blocked `RUN_ID` package；operator receipt 逐次绑定实际 rounds、slots、manifest、源码和 DLL identity。此 preparation/finalize 不会自动启动训练或 Bladed。

这里的“重建”限定于当前 WT15 工程。动作上限、桨距范围、功率/风速尺度、状态区阈值、B4 统计参数和 PID 参考均与当前机组及数据绑定；它不是跨功率机组通用的配方。另请先核对 `gear_ratio=97` 与 Bladed 工程字段的语义差异，不要未经确认就把它当作一致的物理齿轮比。[15MW 绑定点说明](RL模型架构与物理影子模型说明.md#L333-L347)

## 2. 准备环境和固定 canonical 路径

所有命令逐条执行；任一步失败即停止。输入为仓库中的 `bladed-data-new/raw` 与独立 holdout `bladed-data-new/raw_infer`，不能把 holdout 混入 raw 训练源。生成的前处理工件按下面的固定路径写回 `RL/data`，不创建额外数据目录。**本流程直接覆盖 canonical 文件**；每次从 provenance 起按顺序重建整个依赖链，不要只更新上游文件后保留旧的 B4/BC targets/config/checkpoint。

1. 进入仓库并激活环境：

   ```bash
   cd "$HOME/桌面/lyq/Wind_trubine/code/wind_trubine"
   ```

   ```bash
   conda activate wind
   ```

   ```bash
   unset RL_EXPERIMENT_SUFFIX RL_DATA_ROOT RL_DYNAMICS_CHECKPOINT
   ```

   ```bash
   export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
   ```

2. 设置输入与 canonical 输出路径。下列输出文件由后续步骤覆盖：

   ```bash
   export RAW_ROOT="$PWD/bladed-data-new/raw"
   ```

   ```bash
   export HOLDOUT_ROOT="$PWD/bladed-data-new/raw_infer"
   ```

   ```bash
   export DATA_DIR="$PWD/RL/data"
   ```

   ```bash
   export PROVENANCE_PATH="$DATA_DIR/pid_reference_provenance_3_to_25.json"
   ```

   ```bash
   export STATE_ACTION_PATH="$DATA_DIR/state_action_envelope_3_to_25_v1.json"
   ```

   ```bash
   export PAIRED_REFERENCE_PATH="$DATA_DIR/paired_pid_power_load_reference_3_to_25_600s_v2.json"
   ```

   ```bash
   export S2U1_PATH="$DATA_DIR/s2_u1_pid_envelope_3_to_25.json"
   ```

   ```bash
   export S2U1_REPORT_PATH="$DATA_DIR/s2_u1_pid_replay_3_to_25.json"
   ```

   ```bash
   export ROTOR_REFERENCE_PATH="$DATA_DIR/pid_rotor_speed_reference_3_to_25.json"
   ```

   ```bash
   export ROTOR_HOLDOUT_PATH="$DATA_DIR/pid_rotor_speed_reference_raw_infer_evaluation.json"
   ```

   ```bash
   export B4_PATH="$DATA_DIR/region_aware_joint_baseline_state20_3_to_25_v3.json"
   ```

   ```bash
   export BC_DATA_ROOT="$DATA_DIR/b4_centered_residual_bc_state20_v2"
   ```

   ```bash
   export BC_CONFIG_PATH="$PWD/RL/configs/b4_residual_smoke/bc_pretrain_b4_state20_residual_600s_v3.json"
   ```

   ```bash
   export BC_CHECKPOINT_PATH="$PWD/RL/BC_checkpoint/bladed_pid_b4_state20_residual_v3.pt"
   ```

3. 检查源目录名称、raw 23 个风况文件及独立 holdout 目录：

   ```bash
   test "$(basename "$RAW_ROOT")" = raw
   ```

   ```bash
   test -d "$RAW_ROOT"
   ```

   ```bash
   test "$(basename "$HOLDOUT_ROOT")" = raw_infer
   ```

   ```bash
   test -d "$HOLDOUT_ROOT"
   ```

   ```bash
   for condition_id in $(seq 3 25); do test -s "$RAW_ROOT/data_windspeed${condition_id}.csv" || { printf 'missing/empty: %s\n' "$RAW_ROOT/data_windspeed${condition_id}.csv" >&2; false; }; done
   ```

   raw 合同要求恰好有 `data_windspeed3.csv` 至 `data_windspeed25.csv`，不允许缺失或多出 `data_windspeed*.csv`；构建器还会检查时间戳、字段、有限值及 0–600 秒窗口。[raw 数据合同](../RL/utils/raw_dataset_contract.py#L8-L46)

4. 确认 canonical 目标属于本流程。**这些前处理文件会按顺序覆盖同名文件**；从 provenance 开始一路重建到 BC 数据和 config 后，再运行 BC 训练覆盖 checkpoint。不要在这条依赖链跑到一半时启动 RL，也不要把旧 R1 profile/receipt 当作新文件的身份凭证。

   R1 release manifest、profile build report 和物理启动 receipt 保持历史 SHA/fingerprint；前处理文件被覆盖不等于旧 release 自动更新或重新授权。R2 preparation 会绑定新 canonical 工件和当前源码 SHA。

## 3. 审计 raw 中的 PID 动作来源

先确认 CSV 的 `O_PcsTorqueDemand`、`O_PitchPositionDemand1` 到底是 recorded PID demand，还是当前控制器上报的 measurement alias。B4 baseline builder 会校验这份 provenance，因此不能跳过或手工改写结果。

1. 生成审计报告：

   ```bash
   python -m RL.audit_pid_reference_provenance --data-root "$RAW_ROOT" --controller-source bladed_env/udptest/udptest/controller.cpp --api-header bladed_env/BladedAPI/4.8/ExternalControllerApi.h --output "$PROVENANCE_PATH"
   ```

2. 查看机器判定：

   ```bash
   python -c 'import json, os; from pathlib import Path; p=Path(os.environ["PROVENANCE_PATH"]); d=json.loads(p.read_text()); print(json.dumps(d["decision"], ensure_ascii=False, indent=2))'
   ```

当前 B4 构建要求 provenance 的 `decision.udp_action_contract` 是 `recorded_pid_demand`，并要求 `baseline_reference_path=region_aware_surrogate_required`。若结果是 `unresolved`、`native_pid_demand` 或 manifest/工程来源不匹配，**停止**，先核对采集来源与合同；不要为了继续流程手改 JSON。[审计判定逻辑](../RL/audit_pid_reference_provenance.py#L200-L267) [B4 provenance 校验](../RL/build_region_aware_baseline.py#L66-L82)

## 4. 重建 state/action 工件和 paired PID 参照

这两类工件必须从同一份 23 风况 raw 生成，保证其 source manifest 一致。paired reference 包含逐时间戳功率、三片叶根 My 和风迹线签名，600 秒 JSON 可能较大；不要把它当普通小配置反复打开或复制。

1. 生成 state/action envelope。`raw_infer` 只用于独立校验/记录，不参与 raw PID 的训练来源：

   ```bash
   python -m RL.build_state_action_artifact --data-root "$RAW_ROOT" --raw-infer-root "$HOLDOUT_ROOT" --output "$STATE_ACTION_PATH"
   ```

2. 单独生成 0–600 秒 paired PID power/load reference：

   ```bash
   python -m RL.build_paired_pid_reference --data-root "$RAW_ROOT" --output "$PAIRED_REFERENCE_PATH" --time-window-start-s 0 --time-window-end-s 600
   ```

   若后续 profile 使用其他窗口，应为该窗口单独生成参照；不可把 600 秒 reference 当作 180 秒 reference。[paired PID CLI 与窗口参数](../RL/build_paired_pid_reference.py#L17-L31)

## 5. 重建 PID 辅助标定

当前 WT15 控制配置还引用 S2/U1 保护包络和 PID 转速参考。二者分别由 raw PID 数据生成；转速参考生成器会使用 `raw_infer` 检查 holdout，不会因此成为正式闭环验收。

1. 构建 S2/U1 calibration 和 replay report：

   ```bash
   python -m RL.build_s2_u1_calibration --data-root "$RAW_ROOT" --output "$S2U1_PATH" --report-output "$S2U1_REPORT_PATH"
   ```

2. 构建 PID rotor-speed schedule，并用独立 holdout 评估：

   ```bash
   python -m RL.build_pid_rotor_speed_reference --data-root "$RAW_ROOT" --output "$ROTOR_REFERENCE_PATH" --holdout-root "$HOLDOUT_ROOT" --holdout-output "$ROTOR_HOLDOUT_PATH"
   ```

   S2/U1 报告中的 `formal_closed_loop_acceptance` 仍是 false；观测 PID 回放不能估计改变 torque/pitch 后的反事实效果。[S2/U1 输出范围](../RL/build_s2_u1_calibration.py#L245-L284) [转速 schedule holdout 限定](../RL/build_pid_rotor_speed_reference.py#L190-L247)

## 6. 拟合 B4，并在独立 raw_infer 上做 teacher-forced 检查

1. 使用同一份 raw provenance 与 state/action 工件拟合新的区域条件联合 B4：

   ```bash
   python -m RL.build_region_aware_baseline --data-root "$RAW_ROOT" --provenance "$PROVENANCE_PATH" --state-action-artifact "$STATE_ACTION_PATH" --output "$B4_PATH" --ridge-alpha 1.0 --min-region-transitions 500 --torque-delta-limit 15000 --pitch-delta-limit 0.0003
   ```

   若任一控制区域不足最小 transition 数、raw manifest 和 state/action 不一致、或 provenance 不满足要求，构建会失败；不要降低门槛来掩盖数据覆盖问题。[B4 builder 参数](../RL/build_region_aware_baseline.py#L35-L52) [数据合同和拟合入口](../RL/build_region_aware_baseline.py#L253-L316)

2. 用 `raw_infer` 做独立的一步动作 holdout：

   ```bash
   python -m RL.evaluate_region_aware_baseline --baseline "$B4_PATH" --data-root "$HOLDOUT_ROOT" --output "$DATA_DIR/region_aware_joint_baseline_raw_infer_evaluation_v2.json" --step-duration-s 0.02
   ```

   这是使用 holdout 真实状态与上一步动作的 teacher-forced 诊断，不代表 B4 已经过闭环 Bladed 验证。[B4 holdout 评估范围](../RL/evaluate_region_aware_baseline.py#L184-L214)

## 7. 构建 B4 residual BC train/val 数据

BC target 是 recorded PID demand 相对 B4 的归一化 residual。raw 只做 train；raw_infer 只做 val。这里对 canonical BC residual 数据集显式传 `--overwrite`；builder 会覆盖 train/val CSV 与 manifest，并删除该数据目录下按旧输入生成的 `.rl_cache/offline_bc` 缓存，避免后续 loader 读到旧 targets/normalizer。

```bash
python -m RL.build_b4_residual_bc_data --train-root "$RAW_ROOT" --val-root "$HOLDOUT_ROOT" --baseline "$B4_PATH" --state-action-artifact "$STATE_ACTION_PATH" --output-root "$BC_DATA_ROOT" --torque-limit 15000 --pitch-limit 0.0003 --residual-target-scale 0.25 --overwrite
```

生成后检查 manifest 的 `schema_version`、`train_source_role=raw_pid_calibration`、`validation_source_role=raw_infer_holdout`、行数、逐文件 SHA、`baseline_sha256` 和 `state_action_artifact_sha256`。下方只读检查可以确认 600 秒 paired reference、state/action 与 B4 使用同一 raw manifest，并确认 BC 数据指向刚生成的 B4/state 工件：

```bash
python - <<'PY'
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ['DATA_DIR'])
state_path = Path(os.environ['STATE_ACTION_PATH'])
paired_path = Path(os.environ['PAIRED_REFERENCE_PATH'])
baseline_path = Path(os.environ['B4_PATH'])
bc_data_root = Path(os.environ['BC_DATA_ROOT'])
state = json.loads(state_path.read_text(encoding='utf-8'))
paired = json.loads(paired_path.read_text(encoding='utf-8'))
baseline = json.loads(baseline_path.read_text(encoding='utf-8'))
bc_data = json.loads((bc_data_root / 'manifest.json').read_text(encoding='utf-8'))
sha256 = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
assert state['metadata']['input_manifest_sha256'] == paired['metadata']['input_manifest_sha256']
assert state['metadata']['input_manifest_sha256'] == baseline['metadata']['input_manifest_sha256']
assert bc_data['baseline_sha256'] == sha256(baseline_path)
assert bc_data['state_action_artifact_sha256'] == sha256(state_path)
assert bc_data['train_source_role'] == 'raw_pid_calibration'
assert bc_data['validation_source_role'] == 'raw_infer_holdout'
print('raw/state/reference/B4/BC-data identity closure: PASS')
print('train_rows=', bc_data['train_rows'], 'val_rows=', bc_data['val_rows'])
PY
```

以上只证明文件和数据身份闭合，不证明策略安全、闭环性能或训练授权。[BC residual dataset builder](../RL/build_b4_residual_bc_data.py#L134-L200)

## 8. 覆盖并重新生成 resolved control contract 与 BC config

前处理全部输出到 state20 canonical 路径。按第 2–7 节顺序重新生成 state/action、paired PID、B4 和 BC residual 数据后，调用支持 `--overwrite` 的 builder 覆盖旧 resolved contract 与 BC config。这样下游 checkpoint 的 control fingerprint 与本次实际工件 SHA 一致；不要只覆盖 B4/data 后跳过本节。

1. 重算 control contract 和 BC-only 配置：

   ```bash
   python -m RL.build_state20_control_config --output "$BC_CONFIG_PATH" --overwrite --force-retrain
   ```

   builder 从固定 canonical 路径读取 state/action、paired reference、B4、BC data manifest、共享 observation schema 与 predictor interface，验证数据 manifest 的 B4/state SHA，再生成新的 resolved contract/fingerprint。配置沿用当前兼容 schema 标签，但 fingerprint 会因前置工件内容改变而更新。`--overwrite` 允许重写 canonical resolved contract/config；`--force-retrain` 将生成的 BC-only config 标记为无条件重新训练，以便下一节在旧 checkpoint 已存在时覆盖模型。两者用途不同；不传 `--force-retrain` 时，`train_rl` 会尝试加载兼容的 canonical checkpoint 并跳过 BC 训练。生成器的静态路径是有意设计的，因为本流程固定重建 canonical 工件，不提供任意多套工件目录切换。

2. 检查新合同与 config 已对应当前磁盘工件：

   ```bash
   python - <<'PY'
   import hashlib
   import json
   import os
   from pathlib import Path

   data = Path(os.environ['DATA_DIR'])
   config_path = Path(os.environ['BC_CONFIG_PATH'])
   contract = json.loads((data / 'resolved_control_contract_state20_600s_v3.json').read_text(encoding='utf-8'))
   config = json.loads(config_path.read_text(encoding='utf-8'))
   manifest = json.loads((Path(os.environ['BC_DATA_ROOT']) / 'manifest.json').read_text(encoding='utf-8'))
   sha256 = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
   assert contract['artifact_sha256']['state_artifact_sha256'] == sha256(Path(os.environ['STATE_ACTION_PATH']))
   assert contract['artifact_sha256']['paired_pid_reference_sha256'] == sha256(Path(os.environ['PAIRED_REFERENCE_PATH']))
   assert contract['artifact_sha256']['region_aware_baseline_sha256'] == sha256(Path(os.environ['B4_PATH']))
   assert manifest['baseline_sha256'] == sha256(Path(os.environ['B4_PATH']))
   assert manifest['state_action_artifact_sha256'] == sha256(Path(os.environ['STATE_ACTION_PATH']))
   assert config['control_contract_fingerprint'] == contract['control_contract_fingerprint']
   assert config['data']['root_dir'] == 'RL/data/b4_centered_residual_bc_state20_v2'
   assert config['training_mode'] == 'bc_pretrain_only'
   assert config['bc']['force_retrain'] is True
   print('canonical data -> resolved contract -> BC config: PASS')
   print('control_contract_fingerprint=', config['control_contract_fingerprint'])
   PY
   ```

R1 archive manifest/report/receipt 不会随前处理覆盖自动更新。上面的校验只确认配置和输入 SHA 闭合，不代表 BC 模型已训练或 Bladed 已授权。R2 使用新 `RUN_ID` 和 schedule identity；不能用 R1 receipt 验证 R2 配置。

## 9. BC 训练与 holdout

配置已由第 8 节按当前 canonical 文件重建。训练会读取固定 BC checkpoint 路径，并因 `bc.force_retrain=true` 跳过旧 checkpoint warm-start；训练完成后导出并覆盖新的 canonical BC checkpoint。BC logger 的实验目录使用自动 timestamp，避免与既有日志相撞。

1. 确认 canonical BC config/checkpoint 路径，并确保没有另一个训练进程同时读写该 checkpoint：

   ```bash
   printf 'BC_CONFIG=%s\nBC_CHECKPOINT=%s\n' "$BC_CONFIG_PATH" "$BC_CHECKPOINT_PATH"
   ```

   ```bash
   test -f "$BC_CONFIG_PATH" && test -d "$(dirname "$BC_CHECKPOINT_PATH")"
   ```

2. 使用 loopback 本地 UDP endpoint 做 BC-only 训练。此阶段不执行环境 step、不连接 Bladed，也不需要远端 UDP peer。端口 0 不会回写实际绑定端口；这里使用 BC config 中的 8888：

   ```bash
   python -u -m RL.train_rl --config "$BC_CONFIG_PATH" --udp-host 127.0.0.1 --udp-port 8888
   ```

   `--force-retrain` 仅将开关写入 `bc_pretrain_only` 配置；训练入口拒绝在在线 RL 模式使用它。训练日志保存在 `RL/experiments/bc_pretrain_b4_state20_residual_600s_v3_<timestamp>/`；BC 训练完成且进入导出阶段时，新 checkpoint 用原子替换写到 canonical 路径。若训练中断在导出之前，旧 checkpoint 文件保持原样；要进行 RL 前须检查本次 BC 日志、canonical checkpoint SHA 并完成 holdout 评估。[BC force-retrain 逻辑](../RL/train_rl.py#L4144-L4195) [原子 checkpoint 导出](../RL/train_rl.py#L3685-L3712) [实验 timestamp 命名](../RL/train_rl.py#L230-L244)

3. 对新 checkpoint 做独立 raw_infer BC 评估：

   ```bash
   python -m RL.evaluate_b4_residual_bc --config "$BC_CONFIG_PATH" --checkpoint "$BC_CHECKPOINT_PATH" --split val --batch-size 4096 --output "$DATA_DIR/bc_holdout_state20_current.json"
   ```

   保存模型 SHA、holdout 报告、配置和 data manifest SHA；holdout 指标是离线模型诊断，不是 Bladed 策略验收。[BC evaluator 参数](../RL/evaluate_b4_residual_bc.py#L84-L99)

## 10. 23 风况 × 4 轮 × 600 秒 TD3+BC 队列

本流程的 RL 队列目标固定为下表合同；“4 轮”是每个风况独立 reload 并训练 4 次，不是 4 个风况，也不改变 PID reference 的 600 秒时间窗。

| 字段 | 目标值 |
|---|---:|
| 风况 | condition 3–25，共 23 个 |
| 每个 slot 时长 | 600 秒 |
| rounds | 4 |
| slots | $23\times4=92$ |
| 轮次顺序 | base `[12, 11, 13, 10, 14, 9, 15, 8, 16, 7, 17, 6, 18, 5, 19, 4, 20, 3, 21, 22, 23, 24, 25]`，每轮旋转步长 7 |
| queue | `bladed_udp_19_si_my_queue_v2`，`independent_case_reload` |
| Windows manual batch | `manual_batch_profile=full`，`manual_batch_rounds=ROUNDS` |

默认值为 4 轮、92 slots、名义 69,000 次 Critic update。轮数可配置为 1–8；9 轮及以上被 18 GiB 固定 shadow storage budget 拒绝。只存在一个 profile 名 `23wind_round600_training`，DLL 一律使用 `manual_batch_profile=full`，rounds 另行传入。缺帧、skip/retry 或未完成 slot 不能按计划值记为完成。

### 10.1 生成唯一活动 profile 的 blocked RUN_ID package

第 2–9 节完成后，R2 profile builder 使用最新 canonical BC/B4/PID 工件。profile/build report 在 [R2 活动目录](../RL/configs/WT15_STATE20_RL_20261023_R2/README.md)固定；每次物理训练另用全新的 `RUN_ID`，把实际 rounds、manifest、来源文件和 runtime SHA 写进 package/receipt。

1. 选择轮数并生成新 RUN_ID：

   ```bash
   export ROUNDS=4
   ```

   ```bash
   export RUN_ID="state20_23w${ROUNDS}r600_$(date -u +%Y%m%dT%H%M%SZ)_${RANDOM}"
   ```

2. 确认 config 和 timestamped experiment 前缀都没有使用过：

   ```bash
   test ! -e "$PWD/RL/configs/$RUN_ID"
   ```

   ```bash
   test -z "$(find "$PWD/RL/experiments" -maxdepth 1 -mindepth 1 -type d -name "${RUN_ID}_*" -print -quit)"
   ```

3. 生成 blocked package；这一步不写 receipt、不启动 UDP/Bladed/训练：

   ```bash
   python -m RL.prepare_physical_operator_run --run-id "$RUN_ID" --rounds "$ROUNDS"
   ```

   package 会写到 `RL/configs/$RUN_ID/`，包含 `train.json`、round-specific `queue_manifest.json`、shadow bundle 副本和 preparation report。config 设 `output_dir=RL/experiments`、`experiment_name=RUN_ID`、`append_timestamp=true`，所以训练输出不会覆盖旧批次。

4. 离线核对实际 rounds、slots、update 数、schedule identity 和 blocked 状态：

   ```bash
   python - <<'PY'
   import json, os
   from pathlib import Path
   root = Path('RL/configs') / os.environ['RUN_ID']
   config = json.loads((root / 'train.json').read_text(encoding='utf-8'))
   report = json.loads((root / 'run_preparation_report.json').read_text(encoding='utf-8'))
   assert config['release_id'] == 'WT15_STATE20_RL_20261023_R2'
   assert config['queue']['total_condition_rounds'] == int(os.environ['ROUNDS'])
   assert report['rounds'] == int(os.environ['ROUNDS'])
   assert report['operator_route_contract']['slot_count'] == 23 * int(os.environ['ROUNDS'])
   assert config['append_timestamp'] is True
   assert config['physical_shadow_launch']['authorized'] is False
   assert report['operator_receipt_written'] is False
   assert report['physical_shadow_launch_authorized'] is False
   print('RUN_ID=', os.environ['RUN_ID'])
   print('rounds=', report['rounds'], 'slots=', report['operator_route_contract']['slot_count'])
   print('scheduled_critic_updates=', report['operator_route_contract']['expected_scheduled_updates'])
   print('schedule_sha256=', report['operator_route_contract']['schedule_sha256'])
   PY
   ```

5. 只有操作员再次确认当前 package 显示的 rounds/slots、工程、DLL、seed 和 case identity 后，才单独生成 operator receipt：

   ```bash
   python -m RL.finalize_physical_operator_run --run-id "$RUN_ID"
   ```

   Receipt 仅表示 operator-confirmed development route；不会自动授予独立验证、策略验收、soft/hard、counterfactual 或 RL margin coverage。任何 rounds/config/source 改动都必须新建 RUN_ID 并重新确认，不能原地刷新旧 receipt。

### 10.2 Windows 调度合同

Linux/Windows 共用一条 schedule：`manual_batch_profile=full`，轮数来自同一 `ROUNDS` 值。DLL 会把轮数和 `RUN_ID` 写入 persisted queue state identity；换轮数或重跑都必须用新 RUN_ID。以下 PowerShell 命令逐条执行；最后注销/重新登录 Windows，使 Bladed 从 Explorer 继承变量。若 machine scope 存在冲突值，先按项目配置处理，不用额外 command-line override 掩盖。

1. 设置 queue-v2 协议和统一 full profile：

   ```powershell
   [Environment]::SetEnvironmentVariable('BLADED_RL_UDP_PROTOCOL', 'bladed_udp_19_si_my_queue_v2', 'User')
   ```

   ```powershell
   [Environment]::SetEnvironmentVariable('BLADED_RL_EXPOSE_MX', '0', 'User')
   ```

   ```powershell
   [Environment]::SetEnvironmentVariable('BLADED_RL_POST_LOAD_DISCARD_SIM_S', '0', 'User')
   ```

   ```powershell
   [Environment]::SetEnvironmentVariable('BLADED_RL_MANUAL_BATCH_PROFILE', 'full', 'User')
   ```

   ```powershell
   [Environment]::SetEnvironmentVariable('BLADED_RL_TRAIN_SIM_S', '600', 'User')
   ```

2. 把 Linux 步骤 10.1 打印的同一个 RUN_ID 粘贴到 PowerShell：

   ```powershell
   $RunId = '粘贴 Linux 打印的 RUN_ID'
   ```

   ```powershell
   $Rounds = '粘贴 Linux 本次使用的 ROUNDS 数值'
   ```

   ```powershell
   if ($RunId -notmatch '^[A-Za-z0-9_-]{1,80}$') { throw 'RUN_ID 格式错误' }
   ```

   ```powershell
   if ($Rounds -notmatch '^[1-8]$') { throw 'ROUNDS 必须为 1 到 8' }
   ```

   ```powershell
   [Environment]::SetEnvironmentVariable('BLADED_RL_MANUAL_BATCH_RUN_ID', $RunId, 'User')
   ```

   ```powershell
   [Environment]::SetEnvironmentVariable('BLADED_RL_MANUAL_BATCH_ROUNDS', $Rounds, 'User')
   ```

3. 清除可能污染 queue identity 的 user-scope overrides：

   ```powershell
   [Environment]::SetEnvironmentVariable('BLADED_RL_QUEUE_EPOCH', $null, 'User')
   ```

   ```powershell
   [Environment]::SetEnvironmentVariable('BLADED_RL_CONDITION_INSTANCE_ID', $null, 'User')
   ```

   ```powershell
   [Environment]::SetEnvironmentVariable('BLADED_RL_CONDITION_ID', $null, 'User')
   ```

   ```powershell
   [Environment]::SetEnvironmentVariable('BLADED_RL_HOST', $null, 'User')
   ```

   ```powershell
   [Environment]::SetEnvironmentVariable('BLADED_RL_PORT', $null, 'User')
   ```

   ```powershell
   [Environment]::SetEnvironmentVariable('BLADED_RL_TIMEOUT_MS', $null, 'User')
   ```

同一 DLL `full` 档位支持动态 rounds，固定 23-condition base order 和 rotation stride=7；state file 按 RUN_ID/profile-rounds 隔离。每次新物理训练都生成新 RUN_ID，不续用耗尽状态。[DLL full/rounds 配置](../bladed_env/udptest/udptest/controller.cpp#L77-L96) [schedule 解析](../bladed_env/udptest/udptest/controller.cpp#L270-L320) [persisted state 与 slot rotation](../bladed_env/udptest/udptest/controller.cpp#L450-L535)

### 10.3 统一 route 与 R1 历史归档边界

本流程生成的 R2 profile 使用 `physical_fir_v2` read-only shadow；rounds 从配置和 manifest 一起读取，validator 同步计算 schedule hash、slot 数、critic updates、record 上限和日志预算。4 轮、6 轮和其他预算支持范围内的 rounds 是同一 route 的不同 schedule identity，不是多个 code/profile branch。R1 的 6-round manifest/profile report/release SHA 原样保留在 [R1 archive](../RL/configs/WT15_STATE20_RL_20261022_R1/release_manifest.json#L1-L72)，不得拿它的 receipt 背书 R2，也不能将 R2 receipt 改写到旧 RUN_ID。

历史 `state20_23wind_4round_600s_diagnostic` 仍只是历史结果，不作为新 R2 checkpoint/receipt/验收/续训起点。[历史四轮诊断说明](state20_23wind_training_improvements.md#L1-L11)

### 10.4 启动 Linux queue learner 与手动加载 Bladed cases

> **实际闭环提醒：**本节命令会启动带只读 physical-FIR shadow 的在线 TD3+BC 并等待 UDP/Bladed；这不是离线检查。只能使用完成配置校验、操作员确认并生成匹配 receipt 的新 R2 RUN_ID。操作员还须确认工程/DLL/seed/case identity、目标设备资源和在线运行许可；缺任一项则停在运行前，不启动 Bladed。Receipt 不等于六门禁验证或策略验收。

1. 在 Windows 完成 10.2 的用户变量设置、注销/重新登录，并确认 Bladed 使用的工程、DLL、case 风文件和 seed 与 strict PID reference 对应。
2. 只有本 RUN_ID 已存在且 SHA 与当前 config/schedule 完全匹配的 operator receipt，并完成人工工程/DLL/seed 确认后，Linux 才能启动 training listener：

```bash
python -u -m RL.train_rl --config "RL/configs/$RUN_ID/train.json"
```

   等待终端显示 config 已加载、UDP listener 正在等待/连接，再从 Windows Bladed 打开对应工程。严格按 `RL/configs/$RUN_ID/queue_manifest.json` 的 `round_id` 与 `condition_id` 顺序逐 case 加载；每个 slot 完成 600 秒 TRAIN 并发送 queue-v2 END 后，再加载 manifest 的下一个 slot，直到本次 ROUNDS 全部完成。

3. 任一 slot timeout、skip/retry、queue metadata/epoch 不匹配、时间窗缺帧或 strict identity 失败即停止，保留该实验目录并检查摘要；不要提高容差或继续把不完整批次当训练完成。
4. 完成后核对 `RL/experiments/${RUN_ID}_<timestamp>/queue_training_summary.json`、轮次 checkpoint、完成/skip/retry 数和 strict identity 结果。`23*ROUNDS` slot manifest 只是计划，不证明所有 slot 已完成。

本 route 属于 R2 的 physical-FIR development shadow 路线，但不是策略验收、独立审计或载荷安全授权；对应所有 authorization 保持 false。[R2 活动发行状态](../RL/configs/WT15_STATE20_RL_20261023_R2/release_manifest.json)

## 11. 训练后评价与归档

训练结束后仍要把不同证据分开记录：

1. **BC holdout：**`RL/data/bc_holdout_state20_current.json`，说明 BC 预测 raw_infer residual 的离线误差。
2. **B4 holdout：**`RL/data/region_aware_joint_baseline_raw_infer_evaluation_v2.json`，只说明 teacher-forced 下一步动作拟合。
3. **RL checkpoint 行为诊断：**使用 `RL.eval_offline_policy` 单独评价 checkpoint；只有训练确实生成 queue evaluation jobs 时，才使用 `--jobs-file`，不要假设 `offline_eval_jobs.jsonl` 一定存在。[offline evaluator 参数](../RL/eval_offline_policy.py#L197-L205)
4. **物理 shadow：**只有运行产生了有效 records、effective config 与 summary 后，才评价 shadow；输出仍是开发诊断，不会自动授予策略验收。[shadow evaluator](../RL/evaluate_physical_load_shadow.py#L505-L527)
5. **策略/发布验收：**需要符合独立场景身份和授权合同的 Bladed inference 证据；训练 summary、BC/B4 holdout 或 shadow 分数不能替代它。[多场景验收入口](../RL/evaluate_multi_scenario_acceptance.py#L225-L233)

每个训练 RUN_ID 的配置和 queue manifest 放在 `RL/configs/<RUN_ID>/`；`append_timestamp=true` 时实际训练日志和 RL checkpoint 位于 `RL/experiments/<RUN_ID>_<timestamp>/`。RUN_ID 同时绑定 Windows DLL persisted queue state 和 Linux config；每次 RL 实验必须使用新 RUN_ID。canonical 前处理/Baseline/BC 工件按本文明确步骤覆盖；RL 实验目录按 timestamp 隔离并保留旧实验，不新增 `runs/output` 层。[训练目录约定](训练操作笔记.md#L1-L25)

## 12. 结论：现在能从 raw 做到哪里

| 阶段 | 当前有可用独立 CLI | 固定 canonical 路径/覆盖策略 | 说明 |
|---|---:|---:|---|
| raw provenance | 是 | 固定文件覆盖 | 必须核验结果后才能拟合 B4 |
| state/action、paired PID 参照 | 是 | 固定文件覆盖 | 保持 raw 与 raw_infer 分离 |
| S2/U1、转速 schedule | 是 | 固定文件覆盖 | 离线 PID 标定，不是闭环验收 |
| B4 与 raw_infer 诊断 | 是 | 固定文件覆盖 | teacher-forced，只是离线诊断 |
| BC residual train/val 数据 | 是 | `--overwrite` | 覆盖 train/val、manifest 和旧 cache |
| resolved control contract / BC config | 是 | 是 | 使用 canonical 固定路径及 `--overwrite`，按 SHA 重算 fingerprint |
| BC checkpoint | 是 | 是 | BC-only `force_retrain=true`；完成后原子覆盖 canonical checkpoint，日志保留 timestamp |
| unified 23wind×rounds queue/profile | 是 | R2 profile 固定；每批新 RUN_ID | 一个 `23wind_round600_training` profile；`--rounds` 同步生成 queue manifest、TD3+BC update contract 和 shadow budget |
| physical-FIR TD3+BC Bladed development | 是 | 新 RUN_ID + timestamp output | 单一 queue-v2 / `physical_fir_v2` read-only shadow route；rounds 是 schedule 参数；仍不代表策略验收 |
| R1 `WT15_STATE20_RL_20261022_R1` | 只读归档 | 原 SHA/receipt 保留 | 历史 23wind×6round profile；不再是活动 code/config branch，不接受 R2 receipt |

因此，按本文固定 canonical 覆盖流程，已具备从 raw PID 重建 B4 residual BC 数据、重算 fingerprint/config、强制 BC-only 重训及 raw_infer holdout 的独立步骤；R2 统一物理 profile 默认生成 23wind×4round/92 slots，也可在同一个 profile 中配置其他受支持 rounds。每个运行包、schedule、operator receipt 都绑定实际 rounds 和源码 SHA。**每次前处理重建必须按依赖顺序更新，不能留下旧 BC targets/cache/checkpoint。**训练实验用新 RUN_ID 和 timestamp 隔离。此次只生成并验证离线 R2 profile，没有生成 operator receipt，没有启动训练或 Bladed。

`WT15_STATE20_RL_20261023_R2` 是当前唯一活动 profile，使用物理 FIR read-only shadow；`WT15_STATE20_RL_20261022_R1` 的六轮 profile/report/release SHA 保持为不可变历史归档。两者共享一套 rounds-configurable runtime，但 receipts/runtime identities 不互通。任何 schedule/source 修改都需新 RUN_ID 和新确认；authorization、策略验收、独立审计、soft/hard 和 RL margin coverage 仍为 false。
