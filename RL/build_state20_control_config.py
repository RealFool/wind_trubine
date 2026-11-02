from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from RL.utils.control_contract import (
    ENGINEERING_DEMAND_HIGH_NM,
    build_resolved_control_contract,
    load_static_control_contract,
    write_resolved_control_contract,
)
from RL.utils.load_predictor_contract import LoadPredictorInterface, PREDICTOR_INTERFACE_SCHEMA_VERSION
from RL.utils.state_action_artifact import state_aware_observation_schema


CONFIG_ROOT = PROJECT_ROOT / 'RL/configs/b4_residual_smoke'
DATA_ROOT = PROJECT_ROOT / 'RL/data'
STATIC_CONTRACT_PATH = CONFIG_ROOT / 'control_contract_static_v2.json'
STATE_ARTIFACT_PATH = DATA_ROOT / 'state_action_envelope_3_to_25_v1.json'
PAIRED_REFERENCE_PATH = DATA_ROOT / 'paired_pid_power_load_reference_3_to_25_600s_v2.json'
B4_BASELINE_PATH = DATA_ROOT / 'region_aware_joint_baseline_state20_3_to_25_v3.json'
OBSERVATION_SCHEMA_PATH = DATA_ROOT / 'state_aware_control_observation_schema_v1.json'
PREDICTOR_INTERFACE_PATH = DATA_ROOT / 'load_predictor_interface_v3.json'
PREDICTOR_DATASET_ROOT = DATA_ROOT / 'load_predictor_dataset_state20_stride10_v2'
PREDICTOR_DATASET_MANIFEST_PATH = PREDICTOR_DATASET_ROOT / 'manifest.json'
BC_DATA_ROOT = DATA_ROOT / 'b4_centered_residual_bc_state20_v2'
BC_DATA_MANIFEST_PATH = BC_DATA_ROOT / 'manifest.json'
RESOLVED_CONTRACT_PATH = DATA_ROOT / 'resolved_control_contract_state20_600s_v3.json'
BC_TEMPLATE_PATH = CONFIG_ROOT / 'bc_pretrain_b4_state20_template_v3.json'
BC_CONFIG_PATH = CONFIG_ROOT / 'bc_pretrain_b4_state20_residual_600s_v3.json'
BC_CHECKPOINT_PATH = PROJECT_ROOT / 'RL/BC_checkpoint/bladed_pid_b4_state20_residual_v3.pt'


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f'Required state20 artifact does not exist: {path}') from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f'Unable to read state20 artifact: {path}') from exc
    if not isinstance(value, dict):
        raise ValueError(f'State20 artifact must contain an object: {path}')
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def _resolved_contract(*, overwrite: bool = False) -> tuple[dict[str, Any], str]:
    static_contract, _ = load_static_control_contract(STATIC_CONTRACT_PATH)
    observation_schema = _load(OBSERVATION_SCHEMA_PATH)
    if observation_schema != state_aware_observation_schema():
        raise ValueError('State20 observation schema artifact does not match the code contract')
    predictor_interface = LoadPredictorInterface.from_path(PREDICTOR_INTERFACE_PATH)
    if predictor_interface.schema_version != PREDICTOR_INTERFACE_SCHEMA_VERSION:
        raise ValueError('State20 predictor interface schema is incompatible')
    resolved = build_resolved_control_contract(
        static_contract,
        artifact_paths={
            'state_artifact_sha256': STATE_ARTIFACT_PATH,
            'paired_pid_reference_sha256': PAIRED_REFERENCE_PATH,
            'region_aware_baseline_sha256': B4_BASELINE_PATH,
            'observation_schema_sha256': OBSERVATION_SCHEMA_PATH,
            'predictor_interface_schema_sha256': PREDICTOR_INTERFACE_PATH,
        },
        observation_schema=observation_schema,
        predictor_interface_schema_version=PREDICTOR_INTERFACE_SCHEMA_VERSION,
        project_root=PROJECT_ROOT,
    )
    if RESOLVED_CONTRACT_PATH.exists() and not overwrite:
        existing_resolved = _load(RESOLVED_CONTRACT_PATH)
        if existing_resolved != resolved:
            raise FileExistsError(
                f'Resolved control contract differs from the existing canonical file: {RESOLVED_CONTRACT_PATH}. '
                'Pass overwrite=True only when intentionally rebuilding all dependent canonical artifacts.'
            )
        fingerprint = str(existing_resolved['control_contract_fingerprint'])
    else:
        fingerprint = write_resolved_control_contract(RESOLVED_CONTRACT_PATH, resolved)
    return resolved, fingerprint


def _validate_b4_data(resolved: dict[str, Any]) -> dict[str, Any]:
    manifest = _load(BC_DATA_MANIFEST_PATH)
    if manifest.get('schema_version') != 'b4_centered_residual_bc_data_v2':
        raise ValueError('State20 BC data manifest schema is incompatible')
    if manifest.get('baseline_sha256') != resolved['artifact_sha256']['region_aware_baseline_sha256']:
        raise ValueError('State20 BC data manifest B4 hash is incompatible with the resolved contract')
    if manifest.get('state_action_artifact_sha256') != resolved['artifact_sha256']['state_artifact_sha256']:
        raise ValueError('State20 BC data manifest state artifact hash is incompatible with the resolved contract')
    if manifest.get('observation_schema') != resolved['observation_schema']:
        raise ValueError('State20 BC data manifest observation schema is incompatible with the resolved contract')
    if manifest.get('action_high') != [ENGINEERING_DEMAND_HIGH_NM, 0.43]:
        raise ValueError('State20 BC data manifest engineering action bound is incompatible')
    return manifest


def build_bc_config(
    *,
    output_path: Path = BC_CONFIG_PATH,
    overwrite: bool = False,
    force_retrain: bool = False,
) -> dict[str, Any]:
    resolved, fingerprint = _resolved_contract(overwrite=overwrite)
    bc_manifest = _validate_b4_data(resolved)
    config = copy.deepcopy(_load(BC_TEMPLATE_PATH))
    config.update({
        'experiment_name': 'bc_pretrain_b4_state20_residual_600s_v3',
        'training_mode': 'bc_pretrain_only',
        'device': 'cpu',
        'control_contract_static_path': _relative(STATIC_CONTRACT_PATH),
        'resolved_control_contract_path': _relative(RESOLVED_CONTRACT_PATH),
        'control_contract_fingerprint': fingerprint,
    })
    config['data'].update({
        'root_dir': _relative(BC_DATA_ROOT),
        'train_split': 'train',
        'eval_split': 'val',
    })
    action_low = [0.0, 0.0]
    action_high = [ENGINEERING_DEMAND_HIGH_NM, 0.43]
    config['env'].update({
        'queue_lifecycle_enabled': False,
        'require_condition_metadata': False,
        'udp_protocol_version': 'bladed_udp_19_si_my_v1',
        'include_previous_action_observation': True,
        'include_operating_state_observation': True,
        'state_action_artifact_path': _relative(STATE_ARTIFACT_PATH),
        'observation_action_low': action_low,
        'observation_action_high': action_high,
        'action_low': action_low,
        'action_high': action_high,
        'safe_action_low': action_low,
        'safe_action_high': action_high,
        'action_delta_low': [-90_000.0, -0.000698],
        'action_delta_high': [90_000.0, 0.000698],
    })
    reward = config['env']['reward']
    for key in (
        'pid_power_reference_manifest_path',
        'blade_root_peak_reference_path',
        'pid_blade_root_peak_reference_path',
    ):
        reward.pop(key, None)
    reward.update({
        'paired_pid_reference_path': _relative(PAIRED_REFERENCE_PATH),
        'blade_root_load_mode': 'my_peak_increment',
        'blade_root_continuous_mode': 'pid_excess',
        'state_conditioning': {
            'schema_version': 'state_conditioned_reward_v1',
            'weights': {
                'startup_stopped': {
                    'power': 0.0,
                    'blade_root': 0.0,
                    'continuous_load': 0.0,
                    'persistent_power_deficit': 0.0,
                    'action_continuity': 0.0,
                },
                'below_rated': {
                    'power': 1.0,
                    'blade_root': 0.5,
                    'continuous_load': 0.05,
                    'persistent_power_deficit': 0.1,
                    'action_continuity': 0.1,
                },
                'transition': {
                    'power': 0.75,
                    'blade_root': 0.75,
                    'continuous_load': 0.05,
                    'persistent_power_deficit': 0.1,
                    'action_continuity': 0.15,
                },
                'above_rated': {
                    'power': 0.5,
                    'blade_root': 1.0,
                    'continuous_load': 0.1,
                    'persistent_power_deficit': 0.25,
                    'action_continuity': 0.15,
                },
                'negative_power_abnormal': {
                    'power': 0.0,
                    'blade_root': 0.0,
                    'continuous_load': 0.0,
                    'persistent_power_deficit': 0.0,
                    'action_continuity': 0.0,
                },
            },
        },
        'dual_constraint': {
            'schema_version': 'dual_constraint_controller_v1',
            'eta': {
                'load_blade_1': 2.0,
                'load_blade_2': 2.0,
                'load_blade_3': 2.0,
                'energy': 0.5,
            },
            'lambda_max': {
                'load_blade_1': 10.0,
                'load_blade_2': 10.0,
                'load_blade_3': 10.0,
                'energy': 10.0,
            },
            'epsilon_load_per_blade': [0.0, 0.0, 0.0],
            'epsilon_energy_percent': 0.0,
            'initial_lambda': {
                'load_blade_1': 0.0,
                'load_blade_2': 0.0,
                'load_blade_3': 0.0,
                'energy': 0.0,
            },
            'aggregation': 'mean_complete_conditions_at_round_boundary',
        },
    })
    config['control'].update({
        'safe_action_low': action_low,
        'safe_action_high': action_high,
        'state_action_artifact_path': _relative(STATE_ARTIFACT_PATH),
        'region_aware_baseline_path': _relative(B4_BASELINE_PATH),
        'resolved_control_contract_fingerprint': fingerprint,
    })
    config['agent']['actor_residual_target_scale'] = float(bc_manifest['residual_target_scale'])
    config['control']['actor_residual_target_scale'] = float(bc_manifest['residual_target_scale'])
    config['control']['residual_limit'] = bc_manifest['residual_limit']
    config['train'] = {
        'total_episodes': 0,
        'online_rl_enabled': False,
        'replay_size': 1,
        'gpu_warmup_updates': 0,
        'execution_replay_contract': {
            'schema_version': 'execution_mode_replay_contract_v1',
            'use_execution_mode_replay': True,
            'td3_critic_replay_modes': [
                'actor',
                'explicit_baseline_stage',
                'actor_execution_rollback_baseline',
                'low_speed_recovery_hold',
            ],
            'reject_post_step_outcomes': True,
        },
        'actor_optimizer_health_gate': {
            'schema_version': 'actor_optimizer_health_gate_v1',
            'enabled': True,
            'recovery_consecutive_healthy_critic_updates': 200,
            'require_fixed_offline_batch_health_check': True,
            'fixed_offline_batch_size': 256,
        },
    }
    config['logging'] = {
        'mode': 'compact_streaming',
        'quantile_sample_capacity': 2048,
        'event_ring_capacity': 512,
        'episode_ring_capacity': 128,
        'debug_long_form_enabled': False,
        'debug_max_rows': 0,
        'debug_max_bytes': 0,
    }
    config['load_predictor'] = {
        'interface_path': _relative(PREDICTOR_INTERFACE_PATH),
        'interface_schema_version': PREDICTOR_INTERFACE_SCHEMA_VERSION,
        'dataset_root': _relative(PREDICTOR_DATASET_ROOT),
        'dataset_manifest_path': _relative(PREDICTOR_DATASET_MANIFEST_PATH),
        'dataset_manifest_sha256': _sha256(PREDICTOR_DATASET_MANIFEST_PATH),
        'dataset_stride': 10,
        'mode': 'shadow',
        'dataset_schema_version': 'load_predictor_dataset_v2',
        'history_frames': 50,
        'horizons_s': [0.1, 0.2, 0.5],
        'plan_input_semantics': 'candidate_plan+horizon_index+plan_kind_index',
        'candidate_plan_encoding': 'real_action_sequence_gru_normalized_by_engineering_bounds_v1',
        'candidate_plan_padding': 'zero_after_selected_horizon_v1',
        'horizon_conditioning': 'discrete_horizon_index_embedding_v1',
        'plan_kind_conditioning': 'realized_0_zoh_candidate_1_embedding_v1',
        'model_input_contract': (
            'candidate_plan_conditioned_tcn: '
            'history[50,31]+action_plan[max_horizon,2]+horizon_index+plan_kind_index'
        ),
        'response_label_semantics': 'measured_actual_torque_and_pitch_at_t_plus_horizon_v1',
        'peak_label_semantics': (
            'max_abs_blade_root_my_over_strict_future_frames_t_plus_dt_through_t_plus_horizon_v1'
        ),
        'soft_guard_horizons_s': [0.1, 0.2],
        'realized_plan_online_authorized': False,
        'online_candidate_plan_kind': 'zoh_candidate',
        'max_plan_mismatch': 0.05,
        'hard_candidate_enabled': False,
        'predictor_artifact_sha256': None,
        'predictor_artifact_path': None,
        'predictor_time_partitions_s': None,
        'shadow_audit_capacity_per_group': 35,
        'shadow_audit_max_records': 250_000,
        'shadow_audit_max_record_bytes': 64 * 1024,
        'shadow_audit_max_total_bytes': 256 * 1024 * 1024,
        'replay_capacity': 100_000,
        'replay_record_capacity': 512,
        'calibration_artifact_path': None,
        'calibration_artifact_sha256': None,
    }
    config['offline_eval'] = {'enabled': False, 'run_after_training': False}
    config['bc'].update({
        'enabled': True,
        'steps': 5_000,
        'checkpoint_steps': [2_000, 5_000],
        'batch_size': 256,
        'split': 'train',
        'cache_enabled': True,
        'resume_checkpoint': _relative(BC_CHECKPOINT_PATH),
        'export_checkpoint_path': _relative(BC_CHECKPOINT_PATH),
        'force_retrain': bool(force_retrain),
        'precomputed_action_target_columns': list(bc_manifest['target_columns']),
    })
    config['b4_residual_contract'] = {
        'schema_version': 'b4_centered_residual_control_v3',
        'formula': (
            'b4_action = clip(previous_sent_action + region_aware_b4_delta); '
            'desired_action = b4_action + actor_residual * state_conditioned_residual_limit'
        ),
        'replay_action': 'actual_sent_action_minus_same_step_b4_action_over_same_state_residual_limit',
        'data_manifest_path': _relative(BC_DATA_MANIFEST_PATH),
        'data_manifest_sha256': _sha256(BC_DATA_MANIFEST_PATH),
        'baseline_sha256': resolved['artifact_sha256']['region_aware_baseline_sha256'],
        'state_action_artifact_sha256': resolved['artifact_sha256']['state_artifact_sha256'],
        'paired_pid_reference_sha256': resolved['artifact_sha256']['paired_pid_reference_sha256'],
        'observation_schema_sha256': resolved['artifact_sha256']['observation_schema_sha256'],
        'control_contract_fingerprint': fingerprint,
        'actor_residual_target_scale': float(bc_manifest['residual_target_scale']),
        'previous_action_observation_retained': True,
        'state_observation_retained': True,
    }
    config.pop('queue', None)
    config.pop('b4_residual_smoke', None)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f'State20 BC config already exists: {output_path}. '
            'Pass overwrite=True only when intentionally rebuilding the canonical BC config.'
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return config


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Build the state20 BC warmstart config from the resolved control contract.',
    )
    parser.add_argument('--output', type=Path, default=BC_CONFIG_PATH)
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Refresh the canonical resolved contract and BC config from current canonical artifacts.',
    )
    parser.add_argument(
        '--force-retrain',
        action='store_true',
        help='Train a fresh BC actor even when the configured canonical checkpoint already exists.',
    )
    args = parser.parse_args()
    config = build_bc_config(
        output_path=args.output.expanduser().resolve(),
        overwrite=args.overwrite,
        force_retrain=args.force_retrain,
    )
    print(json.dumps({
        'output': str(args.output),
        'control_contract_fingerprint': config['control_contract_fingerprint'],
        'observation_width': 20,
        'bc_data_manifest': config['b4_residual_contract']['data_manifest_sha256'],
        'bc_checkpoint': config['bc']['export_checkpoint_path'],
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()