from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from RL.build_state20_control_config import BC_CHECKPOINT_PATH, BC_CONFIG_PATH
from RL.utils.load_predictor_contract import LoadPredictorInterface
from RL.utils.control_contract import resolve_configured_control_contract
from RL.utils.queue_case_identity import build_queue_case_identity_validator
from RL.utils.queue_manifest import expand_queue_manifest
from RL.utils.safe_residual_control import SafeResidualController


CONFIG_ROOT = PROJECT_ROOT / 'RL/configs/state20_queue_profiles_v3'
QUEUE_TEMPLATE_PATH = PROJECT_ROOT / (
    'RL/configs/physical_shadow_state20_v1_final_checked/'
    'state20_physical_4wind_4round_600s_training.json'
)
REFERENCE_ROOT = PROJECT_ROOT / 'RL/data'

PRESETS: dict[str, dict[str, Any]] = {
    '23wind_6round_600s': {
        'condition_ids': list(range(3, 26)),
        'base_condition_order': [
            12, 11, 13, 10, 14, 9, 15, 8, 16, 7, 17, 6,
            18, 5, 19, 4, 20, 3, 21, 22, 23, 24, 25,
        ],
        'rotation_stride': 7,
        'train_sim_s': 600.0,
        'paired_reference': 'RL/data/paired_pid_power_load_reference_3_to_25_600s_v2.json',
        'manual_batch_profile': 'full',
        'rounds': 6,
    },
    '23wind_4round_600s': {
        'condition_ids': list(range(3, 26)),
        'base_condition_order': [
            12, 11, 13, 10, 14, 9, 15, 8, 16, 7, 17, 6,
            18, 5, 19, 4, 20, 3, 21, 22, 23, 24, 25,
        ],
        'rotation_stride': 7,
        'train_sim_s': 600.0,
        'paired_reference': 'RL/data/paired_pid_power_load_reference_3_to_25_600s_v2.json',
        'manual_batch_profile': 'full_23x4',
        'rounds': 4,
    },
    '23wind_2round_600s': {
        'condition_ids': list(range(3, 26)),
        'base_condition_order': [
            12, 11, 13, 10, 14, 9, 15, 8, 16, 7, 17, 6,
            18, 5, 19, 4, 20, 3, 21, 22, 23, 24, 25,
        ],
        'rotation_stride': 7,
        'train_sim_s': 600.0,
        'paired_reference': 'RL/data/paired_pid_power_load_reference_3_to_25_600s_v2.json',
        'manual_batch_profile': 'full',
        'rounds': 2,
    },
    '23wind_1round_600s': {
        'condition_ids': list(range(3, 26)),
        'base_condition_order': [
            12, 11, 13, 10, 14, 9, 15, 8, 16, 7, 17, 6,
            18, 5, 19, 4, 20, 3, 21, 22, 23, 24, 25,
        ],
        'rotation_stride': 7,
        'train_sim_s': 600.0,
        'paired_reference': 'RL/data/paired_pid_power_load_reference_3_to_25_600s_v2.json',
        'manual_batch_profile': 'full',
        'rounds': 1,
    },
    '23wind_2round_600s_recovery_cap25': {
        'condition_ids': list(range(3, 26)),
        'base_condition_order': [
            12, 11, 13, 10, 14, 9, 15, 8, 16, 7, 17, 6,
            18, 5, 19, 4, 20, 3, 21, 22, 23, 24, 25,
        ],
        'rotation_stride': 7,
        'train_sim_s': 600.0,
        'paired_reference': 'RL/data/paired_pid_power_load_reference_3_to_25_600s_v2.json',
        'manual_batch_profile': 'full',
        'rounds': 2,
        'source_max_fraction': {'low_speed_recovery_hold': 0.25},
    },
    '4wind_4round_180s': {
        'condition_ids': [8, 11, 12, 16],
        'base_condition_order': [8, 11, 12, 16],
        'rotation_stride': 1,
        'train_sim_s': 180.0,
        'paired_reference': 'RL/data/paired_pid_power_load_reference_3_to_25_180s_v2.json',
        'manual_batch_profile': 'smoke_4x4',
        'rounds': 4,
    },
    '4wind_4round_600s': {
        'condition_ids': [8, 11, 12, 16],
        'base_condition_order': [8, 11, 12, 16],
        'rotation_stride': 1,
        'train_sim_s': 600.0,
        'paired_reference': 'RL/data/paired_pid_power_load_reference_3_to_25_600s_v2.json',
        'manual_batch_profile': 'smoke_4x4',
        'rounds': 4,
    },
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError(f'Expected JSON object: {path}')
    return value


def _validate_frozen_bc_checkpoint(checkpoint_path: Path, config: dict[str, Any]) -> str:
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f'Unable to load frozen state20 BC checkpoint: {checkpoint_path}') from exc
    if not isinstance(checkpoint, dict) or checkpoint.get('stage') != 'bc':
        raise ValueError('Frozen state20 BC checkpoint must be a BC-stage checkpoint mapping')
    if checkpoint.get('data_schema_version') != config.get('data_schema_version'):
        raise ValueError('Frozen state20 BC checkpoint data schema does not match the v3 candidate config')
    resolved_fingerprint = str(config.get('control_contract_fingerprint', ''))
    if (
        len(resolved_fingerprint) != 64
        or checkpoint.get('control_contract_fingerprint') != resolved_fingerprint
    ):
        raise ValueError('Frozen state20 BC checkpoint resolved-control fingerprint is incompatible')
    controller = SafeResidualController(config['env'], config.get('control', {}))
    if (
        checkpoint.get('actor_control_fingerprint') != controller.actor_control_fingerprint
        or checkpoint.get('control_fingerprint') != controller.control_fingerprint
    ):
        raise ValueError('Frozen state20 BC checkpoint Actor/control fingerprint is incompatible')
    agent_state = checkpoint.get('agent')
    actor_state = agent_state.get('actor') if isinstance(agent_state, dict) else None
    if not isinstance(actor_state, dict):
        raise ValueError('Frozen state20 BC checkpoint is missing the Actor state')
    obs_median = actor_state.get('obs_median')
    obs_iqr = actor_state.get('obs_iqr')
    if (
        not isinstance(obs_median, torch.Tensor)
        or not isinstance(obs_iqr, torch.Tensor)
        or obs_median.numel() != 20
        or obs_iqr.numel() != 20
    ):
        raise ValueError('Frozen state20 BC Actor observation-normalization dimension is incompatible')
    return _sha256(checkpoint_path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def _build_manifest(
    profile: str,
    settings: dict[str, Any],
    manifest_path: Path,
    identity_level: str,
) -> dict[str, Any]:
    condition_ids = [int(value) for value in settings['condition_ids']]
    rounds = int(settings.get('rounds', 4))
    return {
        'protocol_version': 'bladed_udp_19_si_my_queue_v2',
        'execution_mode': 'independent_case_reload',
        'total_condition_rounds': rounds,
        'base_condition_order': list(settings['base_condition_order']),
        'rotation_stride': int(settings['rotation_stride']),
        'queue_epoch_start': 1,
        'condition_instance_start': 1,
        'case_path_template': '',
        'bladed_command_template': '',
        'post_load_discard_sim_s': 0.0,
        'train_sim_s': float(settings['train_sim_s']),
        'expanded_slot_count': len(condition_ids) * rounds,
        'manual_batch_profile': settings['manual_batch_profile'],
        'manual_batch_rounds': rounds,
        'state20_profile': profile,
        'identity_validation_level': identity_level,
        'notes': (
            'State20 diagnostic queue training; prefix identity only; not release evidence.'
            if identity_level == 'diagnostic'
            else 'State20 B4-centered residual TD3+BC queue training; predictor remains shadow.'
        ),
    }


def _build_config(
    profile: str,
    settings: dict[str, Any],
    bc_config: dict[str, Any],
    template_config: dict[str, Any],
    manifest_path: Path,
    identity_level: str,
    bc_checkpoint_sha256: str,
    predictor_artifact_path: Path | None = None,
    calibration_artifact_path: Path | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    config = copy.deepcopy(bc_config)
    condition_ids = [int(value) for value in settings['condition_ids']]
    rounds = int(settings.get('rounds', 4))
    train_sim_s = float(settings['train_sim_s'])
    manifest_relative = _relative(manifest_path)
    reference_path = (PROJECT_ROOT / settings['paired_reference']).resolve()

    suffix = '_diagnostic' if identity_level == 'diagnostic' else ''
    config['experiment_name'] = f'b4_residual_td3_bc_state20_{profile}{suffix}'
    config['training_mode'] = 'bc_warmstart_rl'
    if run_id is not None:
        config.update(
            experiment_name=run_id,
            output_dir='RL/experiments',
            append_timestamp=True,
        )
    config['env'].update({
        'udp_protocol_version': 'bladed_udp_19_si_my_queue_v2',
        'queue_lifecycle_enabled': True,
        'require_condition_metadata': True,
        'udp_require_frame_id': True,
        'udp_strict_field_count': True,
    })
    config['env']['reward']['paired_pid_reference_path'] = _relative(reference_path)

    queue_train = copy.deepcopy(template_config['train'])
    queue_train.update({
        'online_rl_enabled': True,
        'online_eval_enabled': False,
        'eval_every': 0,
        'execution_replay_contract': copy.deepcopy(bc_config['train']['execution_replay_contract']),
        'actor_optimizer_health_gate': copy.deepcopy(bc_config['train']['actor_optimizer_health_gate']),
    })
    condition_replay = queue_train.get('condition_replay', {})
    if not isinstance(condition_replay, dict):
        raise ValueError('queue template train.condition_replay must be an object')
    source_max_fraction = settings.get('source_max_fraction')
    if source_max_fraction is not None:
        condition_replay = copy.deepcopy(condition_replay)
        condition_replay['source_max_fraction'] = dict(source_max_fraction)
        queue_train['condition_replay'] = condition_replay
    config['train'] = queue_train
    predictor_config = config.get('load_predictor', {})
    if not isinstance(predictor_config, dict):
        raise ValueError('Frozen BC config load_predictor must be an object')
    if predictor_artifact_path is not None:
        predictor_artifact_path = predictor_artifact_path.expanduser().resolve()
        if not predictor_artifact_path.is_file():
            raise FileNotFoundError(f'Load predictor artifact does not exist: {predictor_artifact_path}')
        predictor_config['predictor_artifact_path'] = _relative(predictor_artifact_path)
        predictor_config['predictor_artifact_sha256'] = _sha256(predictor_artifact_path)
        predictor_summary_path = predictor_artifact_path.with_name(
            f'{predictor_artifact_path.stem}.summary.json'
        )
        if not predictor_summary_path.is_file():
            raise FileNotFoundError(f'Frozen candidate predictor summary does not exist: {predictor_summary_path}')
        predictor_summary = _read_json(predictor_summary_path)
        configured_interface_path = Path(str(predictor_config['interface_path'])).expanduser()
        if not configured_interface_path.is_absolute():
            configured_interface_path = PROJECT_ROOT / configured_interface_path
        expected_interface_sha = _sha256(configured_interface_path)
        if (
            predictor_summary.get('checkpoint_sha256') != predictor_config['predictor_artifact_sha256']
            or
            predictor_summary.get('schema_version') != 'load_predictor_checkpoint_v4_candidate_conditioned'
            or predictor_summary.get('interface_schema_version') != predictor_config.get('interface_schema_version')
            or predictor_summary.get('interface_schema_version') != 'causal_load_predictor_interface_v3'
            or predictor_summary.get('interface_sha256') != expected_interface_sha
            or predictor_summary.get('dataset_schema_version') != 'load_predictor_dataset_v2'
        ):
            raise ValueError('Predictor artifact summary interface/dataset schema is incompatible with candidate profile')
        dataset_manifest_sha256 = predictor_summary.get('dataset_manifest_sha256')
        if not isinstance(dataset_manifest_sha256, str) or len(dataset_manifest_sha256) != 64:
            raise ValueError('Predictor summary is missing dataset_manifest_sha256 provenance')
        predictor_config['dataset_manifest_sha256'] = dataset_manifest_sha256
        predictor_dataset_root = predictor_summary.get('dataset_root')
        if not isinstance(predictor_dataset_root, str) or not predictor_dataset_root.strip():
            raise ValueError('Predictor summary is missing its training dataset root')
        predictor_config['dataset_root'] = _relative(Path(predictor_dataset_root))
        predictor_config['dataset_schema_version'] = predictor_summary['dataset_schema_version']
        frozen_dataset_manifest_path = Path(predictor_config['dataset_root']) / 'manifest.json'
        if not frozen_dataset_manifest_path.is_absolute():
            frozen_dataset_manifest_path = PROJECT_ROOT / frozen_dataset_manifest_path
        if (
            not frozen_dataset_manifest_path.is_file()
            or _sha256(frozen_dataset_manifest_path) != dataset_manifest_sha256
        ):
            raise ValueError('Predictor summary dataset root/manifest SHA do not identify the same frozen dataset')
        dataset_manifest = _read_json(frozen_dataset_manifest_path)
        if dataset_manifest.get('schema_version') != predictor_config['dataset_schema_version']:
            raise ValueError('Predictor dataset manifest schema differs from its summary')
        predictor_config['dataset_manifest_path'] = _relative(frozen_dataset_manifest_path)
        predictor_interface = LoadPredictorInterface.from_path(configured_interface_path)
        horizon_frames = predictor_interface.horizon_frames
        temporal_purge_s = (predictor_interface.history_frames - 1 + max(horizon_frames)) * predictor_interface.step_duration_s
        time_partitions_s = predictor_summary.get('time_partitions_s')
        if not isinstance(time_partitions_s, dict):
            raise ValueError('Predictor summary is missing frozen time_partitions_s')
        required_partition_names = ('train', 'validation', 'calibration', 'coverage_audit')
        if set(time_partitions_s) != set(required_partition_names):
            raise ValueError('Predictor summary time partitions are incomplete')
        partition_bounds = {
            name: (float(time_partitions_s[name][0]), float(time_partitions_s[name][1]))
            for name in required_partition_names
        }
        for previous, following in zip(required_partition_names, required_partition_names[1:]):
            if partition_bounds[following][0] - partition_bounds[previous][1] + 1.0e-12 < temporal_purge_s:
                raise ValueError('Predictor summary time partitions do not include the frozen causal purge')
        if not math.isclose(
            partition_bounds['train'][0],
            float(dataset_manifest['time_window_start_s']),
            rel_tol=0.0,
            abs_tol=1.0e-6,
        ) or not math.isclose(
            partition_bounds['coverage_audit'][1],
            float(dataset_manifest['time_window_end_s']) + predictor_interface.step_duration_s,
            rel_tol=0.0,
            abs_tol=1.0e-6,
        ):
            raise ValueError('Predictor summary partitions do not span the exact source PID window')
        expected_condition_ids = {int(value) for value in dataset_manifest['source_conditions']}
        condition_partitions = predictor_summary.get('condition_partitions')
        if not isinstance(condition_partitions, dict) or set(condition_partitions) != set(required_partition_names):
            raise ValueError('Predictor summary condition partitions are incomplete')
        flattened_conditions = [
            int(value) for name in required_partition_names for value in condition_partitions[name]
        ]
        if (
            len(flattened_conditions) != len(set(flattened_conditions))
            or set(flattened_conditions) != expected_condition_ids
            or set(predictor_summary.get('split_sample_counts', {})) != set(required_partition_names)
            or set(predictor_summary.get('plan_counts_by_split', {})) != set(required_partition_names)
        ):
            raise ValueError('Predictor summary condition/sample splits do not close against the source dataset')
        for name in required_partition_names:
            split_count = int(predictor_summary['split_sample_counts'][name])
            plan_counts = predictor_summary['plan_counts_by_split'][name]
            if (
                split_count <= 0
                or plan_counts.get('realized_plan') != 0
                or plan_counts.get('zoh_candidate') != split_count
            ):
                raise ValueError(f'Predictor summary split is empty or not ZOH-only: {name}')
        predictor_config['predictor_time_partitions_s'] = time_partitions_s
    else:
        predictor_config.pop('predictor_artifact_path', None)
        predictor_config['predictor_artifact_sha256'] = None
        predictor_config['predictor_time_partitions_s'] = None
    config['device'] = str(template_config.get('device', config.get('device', 'cpu')))
    if calibration_artifact_path is not None:
        calibration_artifact_path = calibration_artifact_path.expanduser().resolve()
        if not calibration_artifact_path.is_file():
            raise FileNotFoundError(f'Load guard calibration artifact does not exist: {calibration_artifact_path}')
        calibration_document = _read_json(calibration_artifact_path)
        if calibration_document.get('schema_version') != 'load_guard_calibration_v1':
            raise ValueError('Load guard calibration artifact schema is incompatible')
        predictor_config['calibration_artifact_path'] = _relative(calibration_artifact_path)
        predictor_config['calibration_artifact_sha256'] = _sha256(calibration_artifact_path)
    config['load_predictor'] = predictor_config
    config['logging'] = {
        'mode': 'compact_streaming',
        'quantile_sample_capacity': 2048,
        'event_ring_capacity': 512,
        'episode_ring_capacity': 128,
        'queue_condition_ring_capacity': 512,
        'queue_condition_max_record_bytes': 512 * 1024,
        'queue_condition_summaries_max_total_bytes': 256 * 1024 * 1024,
        'queue_event_pre_frames': 25,
        'queue_event_post_frames': 25,
        'queue_event_window_capacity': 16,
        'queue_event_window_max_record_bytes': 256 * 1024,
        'queue_event_windows_max_total_bytes': 512 * 1024 * 1024,
        'debug_long_form_enabled': False,
        'debug_max_rows': 0,
        'debug_max_bytes': 0,
    }

    config['bc'].update({
        'steps': 0,
        'checkpoint_steps': [],
        'resume_checkpoint': _relative(BC_CHECKPOINT_PATH),
        'export_checkpoint_path': '',
        'force_retrain': False,
    })

    if identity_level == 'strict':
        case_identity_validation = {
            'enabled': True,
            'validation_level': 'strict',
            'paired_pid_reference_path': _relative(reference_path),
            'fixed_simulation_time_offset_s': 0.0,
            'wind_absolute_tolerance': 0.01,
            'time_absolute_tolerance_s': 1.0e-6,
        }
    elif identity_level == 'diagnostic':
        case_identity_validation = {
            'enabled': True,
            'validation_level': 'prefix',
            'source_glob': 'bladed-data-new/raw/data_windspeed*.csv',
            'wind_sample_count': 3,
            'wind_absolute_tolerance': 0.01,
        }
    else:
        raise ValueError(f'Unknown identity validation level: {identity_level}')

    config['queue'] = {
        'manifest_path': manifest_relative,
        'execution_mode': 'independent_case_reload',
        'condition_ids': condition_ids,
        'total_condition_rounds': rounds,
        'train_sim_s': train_sim_s,
        'train_sim_tolerance_s': 0.1,
        'condition_checkpoint_count': 1,
        'skip_failed_condition_on_train_timeout': True,
        'require_zero_skipped_conditions_for_updates': True,
        'require_zero_retried_condition_attempts_for_updates': True,
        'abort_on_retried_condition_attempt': True,
        'reconcile_verified_physical_skip': False,
        'case_identity_validation': case_identity_validation,
        'critic_start_after_rounds': 1,
        'actor_start_after_rounds': 1,
        'collect_exploration_noise': 0.0,
        'critic_exploration_noise': 0.0,
        'actor_exploration_noise': 0.0,
    }
    steps_per_condition = int(round(train_sim_s / 0.02))
    update_every_steps = int(queue_train.get('update_every_steps', 30))
    config['state20_queue_contract'] = {
        'schema_version': 'state20_queue_training_contract_v1',
        'profile': profile,
        'condition_ids': condition_ids,
        'rounds': rounds,
        'train_sim_s': train_sim_s,
        'expected_scheduled_updates': (rounds - 1) * len(condition_ids) * steps_per_condition // update_every_steps,
        'paired_reference_path': _relative(reference_path),
        'paired_reference_sha256': _sha256(reference_path),
        'resolved_control_contract_fingerprint': config['control_contract_fingerprint'],
        'bc_warmstart_checkpoint_sha256': bc_checkpoint_sha256,
        'observation_width': 20,
        'predictor_mode': 'shadow',
        'identity_validation_level': identity_level,
        'policy_acceptance': False,
    }
    return config


def build_profile(
    profile: str,
    *,
    identity_level: str = 'strict',
    output_dir: Path | None = None,
    predictor_artifact_path: Path | None = None,
    calibration_artifact_path: Path | None = None,
    overwrite: bool = False,
    run_id: str | None = None,
) -> tuple[Path, Path]:
    if profile not in PRESETS:
        raise ValueError(f'Unknown state20 queue profile: {profile}')
    if run_id is not None:
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', run_id):
            raise ValueError('run_id must contain 1-80 ASCII letters, numbers, underscores, or hyphens')
        run_config_dir = (PROJECT_ROOT / 'RL/configs' / run_id).resolve()
        if output_dir is not None and output_dir.expanduser().resolve() != run_config_dir:
            raise ValueError('When run_id is provided, output_dir must be RL/configs/<RUN_ID>')
        experiment_root = PROJECT_ROOT / 'RL/experiments'
        if run_config_dir.exists() or (experiment_root / run_id).exists() or any(
            experiment_root.glob(f'{run_id}_*')
        ):
            raise FileExistsError(
                f'RUN_ID already has a config or timestamped experiment directory: {run_id}. '
                'Choose a fresh timestamped RUN_ID; do not resume or overwrite an earlier batch.'
            )
        output_dir = run_config_dir
    elif output_dir is None:
        output_dir = CONFIG_ROOT
    if not BC_CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f'State20 BC checkpoint does not exist: {BC_CHECKPOINT_PATH}. '
            'Run the state20 BC warmstart before building queue configs.'
        )
    # Read and validate the frozen BC config without invoking its builder: that
    # builder rewrites the canonical resolved contract and config in place.
    bc_config = _read_json(BC_CONFIG_PATH)
    resolved = resolve_configured_control_contract(bc_config, project_root=PROJECT_ROOT)
    if resolved is None or resolved[1] != bc_config.get('control_contract_fingerprint'):
        raise ValueError('Frozen state20 BC config/resolved control-contract fingerprint is invalid')
    if not BC_CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f'State20 BC checkpoint does not exist: {BC_CHECKPOINT_PATH}')
    bc_checkpoint_sha256 = _validate_frozen_bc_checkpoint(BC_CHECKPOINT_PATH, bc_config)
    template_config = _read_json(QUEUE_TEMPLATE_PATH)
    settings = PRESETS[profile]
    output_dir = output_dir.expanduser().resolve()
    suffix = '_diagnostic' if identity_level == 'diagnostic' else ''
    if run_id is None:
        manifest_path = output_dir / f'state20_{profile}{suffix}_manifest.json'
        output_path = output_dir / f'state20_{profile}{suffix}.json'
    else:
        manifest_path = output_dir / 'queue_manifest.json'
        output_path = output_dir / 'train.json'
    if (manifest_path.exists() or output_path.exists()) and not overwrite:
        raise FileExistsError(
            'Refusing to overwrite an existing queue config/manifest. '
            f'Choose a new output directory or explicitly enable overwrite: {output_dir}'
        )
    manifest = _build_manifest(profile, settings, manifest_path, identity_level)
    config = _build_config(
        profile,
        settings,
        bc_config,
        template_config,
        manifest_path,
        identity_level,
        bc_checkpoint_sha256,
        predictor_artifact_path,
        calibration_artifact_path,
        run_id,
    )
    slots = expand_queue_manifest(manifest)
    expected_slots = len(settings['condition_ids']) * int(settings.get('rounds', 4))
    if len(slots) != expected_slots:
        raise ValueError(f'Generated queue manifest has unexpected slot count: {manifest_path}')
    for round_index in range(1, int(settings.get('rounds', 4)) + 1):
        round_conditions = [slot.condition_id for slot in slots if slot.round_id == round_index]
        if len(round_conditions) != len(settings['condition_ids']) or set(round_conditions) != set(settings['condition_ids']):
            raise ValueError(f'Generated queue manifest round {round_index} has invalid conditions')
    build_queue_case_identity_validator(config['queue'], tuple(sorted(settings['condition_ids'])))
    if config['env']['udp_protocol_version'] != 'bladed_udp_19_si_my_queue_v2':
        raise ValueError('Generated state20 queue config did not enable queue protocol v2')
    if config['train']['online_rl_enabled'] is not True:
        raise ValueError('Generated state20 queue config did not enable online RL')

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_manifest_path = manifest_path.with_name(f'.{manifest_path.name}.{os.getpid()}.tmp')
    temporary_config_path = output_path.with_name(f'.{output_path.name}.{os.getpid()}.tmp')
    try:
        temporary_manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + '\n',
            encoding='utf-8',
        )
        temporary_config_path.write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + '\n',
            encoding='utf-8',
        )
        os.replace(temporary_manifest_path, manifest_path)
        os.replace(temporary_config_path, output_path)
    except Exception:
        temporary_manifest_path.unlink(missing_ok=True)
        temporary_config_path.unlink(missing_ok=True)
        raise

    return output_path, manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description='Build state20 queue TD3+BC configs and manifests.')
    parser.add_argument('--profile', choices=[*PRESETS, 'all'], default='all')
    parser.add_argument('--identity-level', choices=['strict', 'diagnostic', 'all'], default='all')
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--run-id', default='', help='Write config under RL/configs/<RUN_ID> and log to RL/experiments/<RUN_ID>.')
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite the selected queue profile config and manifest in a fixed preparation directory.',
    )
    parser.add_argument('--predictor-artifact', type=Path, default=None)
    parser.add_argument('--calibration-artifact', type=Path, default=None)
    args = parser.parse_args()
    profiles = list(PRESETS) if args.profile == 'all' else [args.profile]
    identity_levels = ['strict', 'diagnostic'] if args.identity_level == 'all' else [args.identity_level]
    outputs = [
        build_profile(
            profile,
            identity_level=identity_level,
            output_dir=args.output_dir,
            predictor_artifact_path=args.predictor_artifact,
            calibration_artifact_path=args.calibration_artifact,
            overwrite=args.overwrite,
            run_id=args.run_id.strip() or None,
        )
        for identity_level in identity_levels
        for profile in profiles
    ]
    for config_path, manifest_path in outputs:
        print(json.dumps({
            'config': _relative(config_path),
            'manifest': _relative(manifest_path),
            'profile': config_path.stem.removeprefix('state20_'),
        }, ensure_ascii=False))


if __name__ == '__main__':
    main()
