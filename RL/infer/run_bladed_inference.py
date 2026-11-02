from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from RL.agents import TD3Agent
from RL.envs.wind_turbine_env import BLADED_CSV_HEADER, WindTurbineEnv
from RL.infer.analyze_edge_performance import write_edge_performance_report
from RL.utils.config import load_config, normalize_user_path
from RL.utils.control_contract import resolve_configured_control_contract
from RL.utils.constants import DATA_SCHEMA_VERSION
from RL.utils.load_predictor_contract import LoadPredictorInterface
from RL.utils.load_predictor_shadow import LoadPredictorShadow
from RL.utils.load_predictor_artifact import load_candidate_conditioned_predictor
from RL.utils.operating_state import OPTIMIZATION_STATE_ENCODING_ORDER
from RL.utils.logger import ExperimentLogger
from RL.utils.physical_load_shadow import (
    build_physical_shadow, complete_physical_step, physical_backend_selected,
    physical_condition_metadata, preflight_physical_config, submit_physical_sent_frame,
)
from RL.utils.physical_load_shadow_launch import load_verified_physical_checkpoint, validate_physical_launch
from RL.utils.queue_case_identity import (
    QueueCaseIdentityMismatch,
    QueueStrictCaseIdentityValidator,
    build_queue_case_identity_validator,
)
from RL.utils.safe_residual_control import SafeResidualController, safe_action_metrics


STEP_INFO_FIELDS = [
    'reward_power_raw',
    'reward_power_metric_raw',
    'power_t_metric_raw',
    'power_metric_raw',
    'reward_power_state_raw',
    'reward_power_t_raw',
    'reward_power_term',
    'reward_energy_delta_t_s',
    'reward_energy_increment_j',
    'reward_energy_increment_valid',
    'reward_blade_root_raw',
    'reward_blade_root_ratio',
    'reward_blade_root_term',
    'reward_blade_root_peak_raw',
    'reward_blade_root_peak_excess',
    'reward_blade_root_peak_excess_increment',
    'reward_blade_root_1_raw',
    'reward_blade_root_1_peak_nm',
    'reward_blade_root_1_peak_reference_nm',
    'reward_blade_root_1_peak_excess',
    'reward_blade_root_1_peak_excess_increment',
    'reward_blade_root_2_raw',
    'reward_blade_root_2_peak_nm',
    'reward_blade_root_2_peak_reference_nm',
    'reward_blade_root_2_peak_excess',
    'reward_blade_root_2_peak_excess_increment',
    'reward_blade_root_3_raw',
    'reward_blade_root_3_peak_nm',
    'reward_blade_root_3_peak_reference_nm',
    'reward_blade_root_3_peak_excess',
    'reward_blade_root_3_peak_excess_increment',
    'reward_acc_raw',
    'reward_acc_term',
    'reward_shaft_raw',
    'reward_shaft_term',
    'reward_pitch_rate_raw',
    'reward_pitch_rate_term',
    'reward_torque_rate_raw',
    'reward_torque_rate_term',
    'reward_power_available_raw',
    'reward_power_available_fraction',
    'reward_command_power_raw',
    'reward_command_excess_raw',
    'reward_command_excess_term',
    'reward_power_tracking_raw',
    'reward_power_tracking_term',
    'reward_power_floor_raw',
    'reward_power_floor_term',
    'reward_blade_root_continuous_term',
    'reward_total',
    'wind_t',
    'wind_next',
    'action_torque',
    'action_pitch',
    'action_torque_delta',
    'action_pitch_delta',
    'power',
    'power_reward_metric',
    'power_state_real',
    'blade_root_metrics_available',
    'blade_root_metric_nonzero',
    'blade_root_metric_abs',
    'env_invalid_prediction',
    'env_invalid_reward',
    'env_timeout',
    'env_input_clip_fraction',
    'env_output_clip_fraction',
    'env_model_clip_event',
    'env_model_clip_count',
    'udp_connected',
    'udp_frame_time',
    'udp_frame_id',
    'udp_next_frame_id',
    'udp_sim_time',
    'udp_next_sim_time',
    'udp_receive_monotonic_s',
    'udp_action_sent_monotonic_s',
    'udp_frame_lag',
    'udp_dropped_stale_frames',
    'udp_action_apply_lag_ms',
    'udp_action_ready_lag_ms',
    'udp_receive_wait_ms',
    'udp_handshake_complete',
    'udp_inter_frame_timeout',
    'udp_condition_load_active',
    'udp_condition_load_timeout',
    'udp_wait_limit_s',
    'udp_pause_active',
    'udp_pause_wait_ms',
    'udp_action_next_torque',
    'udp_action_next_pitch',
    'udp_action_next_torque_delta',
    'udp_action_next_pitch_delta',
    'udp_queue_epoch',
    'udp_condition_instance_id',
    'udp_condition_id',
    'udp_phase_code',
    'udp_phase_elapsed_s',
    'udp_next_queue_epoch',
    'udp_next_condition_instance_id',
    'udp_next_condition_id',
    'udp_next_phase_code',
    'udp_next_phase_elapsed_s',
    'condition_boundary',
    'condition_start',
    'condition_end',
    'condition_train_active',
    'transition_store_eligible',
    'transition_store_reason_code',
    'control_ood_score',
    'control_fallback_active',
    'control_baseline_only',
    'control_baseline_action_torque',
    'control_baseline_action_pitch',
    'control_policy_delta_torque',
    'control_policy_delta_pitch',
    'control_pre_safety_action_torque',
    'control_pre_safety_action_pitch',
    'control_pitch_protection_floor',
    'control_pitch_protection_state',
    'control_pitch_protection_applied',
    'control_post_pitch_protection_action_torque',
    'control_post_pitch_protection_action_pitch',
    'control_low_speed_stage',
    'control_low_speed_scale',
    'control_low_speed_reason',
    'control_post_safety_action_torque',
    'control_post_safety_action_pitch',
    'control_projected_action_torque',
    'control_projected_action_pitch',
    'control_rate_limited_action_torque',
    'control_rate_limited_action_pitch',
    'control_sent_action_torque',
    'control_sent_action_pitch',
    'control_above_rated_initial_pitch_floor_active',
    'control_above_rated_initial_pitch_floor_applied',
    'control_unsafe_stop',
    'control_action_projection_applied',
    'control_action_rate_limited',
    'load_predictor_shadow_active',
    'load_predictor_shadow_history_count',
    'load_predictor_shadow_history_ready',
    'load_predictor_shadow_pending_count',
    'load_predictor_shadow_completed_record_count',
    'load_predictor_shadow_action_changed',
    'load_predictor_shadow_zoh_candidate_submitted',
    'load_predictor_shadow_prediction_available',
    'load_predictor_shadow_prediction_invalid',
    'load_predictor_shadow_predicted_future_torque_nm_h100',
    'load_predictor_shadow_predicted_future_pitch_rad_h100',
    'load_predictor_shadow_predicted_peak_ratio_blade_1_h100',
    'load_predictor_shadow_predicted_peak_ratio_blade_2_h100',
    'load_predictor_shadow_predicted_peak_ratio_blade_3_h100',
    'load_predictor_shadow_predicted_future_torque_nm_h200',
    'load_predictor_shadow_predicted_future_pitch_rad_h200',
    'load_predictor_shadow_predicted_peak_ratio_blade_1_h200',
    'load_predictor_shadow_predicted_peak_ratio_blade_2_h200',
    'load_predictor_shadow_predicted_peak_ratio_blade_3_h200',
    'queue_case_identity_validation_active',
    'queue_case_identity_validation_level_strict',
    'queue_case_identity_validated',
    'queue_case_identity_full_trace_complete',
    'queue_case_identity_trace_signature_match',
    'queue_case_identity_strict_case_matched',
    'queue_case_identity_reference_matched_only',
    'queue_case_identity_trace_sample_count',
    'queue_case_identity_expected_trace_sample_count',
    'queue_case_identity_mismatch',
]

INFERENCE_FIELDS = [
    'record_type',
    'source_frame_role',
    'inference_episode',
    'inference_step',
    'inference_rollout_segment',
    'inference_global_control_step',
    'inference_record_index',
    'wall_clock_utc',
    'wall_clock_unix_s',
    'policy_inference_ms',
    'safety_projection_ms',
    'udp_step_ms',
    'control_step_ms',
    't_rx_monotonic_s',
    't_tx_monotonic_s',
    't_e2e_ms',
    'policy_residual_torque_norm',
    'policy_residual_pitch_norm',
    'safe_residual_torque_norm',
    'safe_residual_pitch_norm',
    'executed_action_torque_norm',
    'executed_action_pitch_norm',
    'baseline_action_torque',
    'baseline_action_pitch',
    'policy_delta_torque',
    'policy_delta_pitch',
    'pre_safety_action_torque',
    'pre_safety_action_pitch',
    'pitch_protection_floor',
    'pitch_protection_state',
    'pitch_protection_applied',
    'low_speed_stage',
    'low_speed_scale',
    'low_speed_reason',
    'post_safety_action_torque',
    'post_safety_action_pitch',
    'projected_action_torque',
    'projected_action_pitch',
    'rate_limited_action_torque',
    'rate_limited_action_pitch',
    'requested_action_torque',
    'requested_action_pitch',
    'sent_action_torque',
    'sent_action_pitch',
    'sent_action_torque_delta',
    'sent_action_pitch_delta',
    'step_reward',
    'step_done',
    'step_invalid',
    'terminal_reason',
]

RECORD_FIELDS = [
    *BLADED_CSV_HEADER,
    *INFERENCE_FIELDS,
    *(f'info_{name}' for name in STEP_INFO_FIELDS),
    'step_info_json',
]

EPISODE_SUMMARY_FIELDS = [
    'episode',
    'start_wall_clock_utc',
    'end_wall_clock_utc',
    'control_steps',
    'record_count',
    'episode_reward',
    'reward_per_step',
    'episode_duration_ms',
    'control_step_ms_mean',
    'control_step_ms_max',
    'policy_inference_ms_mean',
    'udp_step_ms_mean',
    't_e2e_ms_mean',
    't_e2e_ms_max',
    'fallback_step_count',
    'fallback_fraction',
    'unsafe_stop',
    'timeout_step_count',
    'invalid_step_count',
    'terminal_reason',
    'queue_mode',
    'queue_condition_id',
    'queue_condition_instance_id',
    'queue_epoch',
    'queue_phase_code',
    'rollout_segments',
    'case_identity_status',
    'case_identity_full_trace_complete',
    'case_identity_strict_case_matched',
    'case_identity_pid_raw_sha256',
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')


def _finite_number(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _finite_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, value in metrics.items():
        numeric = _finite_number(value)
        if numeric is not None:
            result[key] = numeric
    return result


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open('w', encoding='utf-8') as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, allow_nan=False)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _build_agent(config: dict[str, Any], obs_dim: int, action_dim: int, device: torch.device) -> TD3Agent:
    algorithm = str(config.get('algorithm', '')).lower()
    if algorithm != 'td3':
        raise ValueError(f'Unsupported algorithm for live inference: {algorithm!r}. Expected td3.')
    return TD3Agent(obs_dim, action_dim, config['agent'], device)


def _build_load_predictor_shadow(
    config: dict[str, Any],
    env: WindTurbineEnv,
    device: torch.device,
) -> LoadPredictorShadow | None:
    predictor_config = config.get('load_predictor')
    if predictor_config is None:
        return None
    if physical_backend_selected(config):
        return None
    if not isinstance(predictor_config, dict):
        raise ValueError('load_predictor must be an object when configured')
    mode = str(predictor_config.get('mode', 'shadow')).strip().lower()
    if mode != 'shadow':
        raise ValueError(
            'Only load_predictor.mode=shadow is enabled for the first frozen inference candidate; '
            'soft/hard_candidate remain fail-closed until RL-distribution coverage audit and same-step identity pass'
        )
    if not env.queue_lifecycle_enabled:
        raise ValueError('load_predictor shadow requires queue lifecycle condition identity')
    interface_path = predictor_config.get('interface_path')
    if not isinstance(interface_path, str) or not interface_path.strip():
        raise ValueError('load_predictor.interface_path must be a non-empty path')
    path = Path(interface_path.strip()).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    interface = LoadPredictorInterface.from_path(path)
    interface_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if not env.include_operating_state_observation:
        raise ValueError('load_predictor shadow requires state-aware live observations')
    predictor_model: torch.nn.Module | None = None
    artifact_path_value = predictor_config.get('predictor_artifact_path')
    artifact_sha256_value = predictor_config.get('predictor_artifact_sha256')
    if artifact_path_value is not None or artifact_sha256_value is not None:
        if not isinstance(artifact_path_value, str) or not artifact_path_value.strip():
            raise ValueError('Configured load predictor artifact requires a non-empty predictor_artifact_path')
        if not isinstance(artifact_sha256_value, str) or len(artifact_sha256_value) != 64:
            raise ValueError('Configured load predictor artifact requires predictor_artifact_sha256')
        dataset_manifest_path_value = predictor_config.get('dataset_manifest_path')
        if not isinstance(dataset_manifest_path_value, str) or not dataset_manifest_path_value.strip():
            raise ValueError('Configured load predictor artifact requires its frozen dataset_manifest_path')
        dataset_manifest_path = Path(dataset_manifest_path_value.strip()).expanduser()
        if not dataset_manifest_path.is_absolute():
            dataset_manifest_path = PROJECT_ROOT / dataset_manifest_path
        if (
            not dataset_manifest_path.is_file()
            or _sha256_file(dataset_manifest_path) != predictor_config.get('dataset_manifest_sha256')
        ):
            raise ValueError('Configured predictor dataset manifest file/SHA does not match the frozen profile')
        artifact_path = Path(artifact_path_value.strip()).expanduser()
        if not artifact_path.is_absolute():
            artifact_path = PROJECT_ROOT / artifact_path
        loaded = load_candidate_conditioned_predictor(
            artifact_path,
            interface_path=path,
            expected_checkpoint_sha256=artifact_sha256_value,
            action_low=env.action_scaler.low,
            action_high=env.action_scaler.high,
            device=device,
            expected_dataset_manifest_sha256=predictor_config.get('dataset_manifest_sha256'),
            expected_time_partitions_s=predictor_config.get('predictor_time_partitions_s'),
        )
        interface = loaded.interface
        predictor_model = loaded.model
    return LoadPredictorShadow(
        interface,
        action_low=env.action_scaler.low,
        action_high=env.action_scaler.high,
        interface_sha256=interface_sha256,
        model=predictor_model,
        device=device,
        replay_capacity=int(predictor_config.get('replay_capacity', 100_000)),
        record_capacity=int(predictor_config.get('replay_record_capacity', 512)),
        audit_capacity_per_group=int(predictor_config.get('shadow_audit_capacity_per_group', 35)),
    )


def _strict_identity_sample(env: WindTurbineEnv) -> tuple[float, float, str | None]:
    payload = env.current_payload
    try:
        simulation_time_s = float(payload['Time'])
        wind_speed_mps = float(payload['I_AverageWindSpeed'])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError('Strict inference identity requires Time and I_AverageWindSpeed in the current payload') from exc
    if not math.isfinite(simulation_time_s) or not math.isfinite(wind_speed_mps):
        raise RuntimeError('Strict inference identity requires finite Time and I_AverageWindSpeed values')
    fingerprint_value = getattr(env, 'current_case_fingerprint', None)
    fingerprint = None if fingerprint_value is None else str(fingerprint_value).strip() or None
    return simulation_time_s, wind_speed_mps, fingerprint


def _agent_state_from_checkpoint(checkpoint: Any) -> dict[str, Any]:
    if not isinstance(checkpoint, dict):
        raise ValueError('Checkpoint must be a mapping containing the TD3 agent state.')
    agent_state = checkpoint.get('agent', checkpoint)
    if not isinstance(agent_state, dict) or 'actor' not in agent_state:
        raise ValueError('Checkpoint does not contain a TD3 agent state with an actor.')
    return agent_state


def _validate_data_schema(
    checkpoint: dict[str, Any],
    config: dict[str, Any],
    expected_control_fingerprint: str | None = None,
    expected_actor_control_fingerprint: str | None = None,
    expected_resolved_control_contract_fingerprint: str | None = None,
) -> None:
    config_schema_version = str(config.get('data_schema_version', '')).strip()
    if config_schema_version != DATA_SCHEMA_VERSION:
        raise ValueError(
            f'Inference config data_schema_version must be {DATA_SCHEMA_VERSION!r}; got {config_schema_version!r}.'
        )
    checkpoint_schema_version = str(checkpoint.get('data_schema_version', '')).strip()
    if checkpoint_schema_version != config_schema_version:
        raise ValueError(
            f'Checkpoint data schema is incompatible: {checkpoint_schema_version!r}; '
            f'expected {config_schema_version!r}. Use a checkpoint trained on the SI/My-only schema.'
        )
    if expected_resolved_control_contract_fingerprint is not None:
        checkpoint_resolved_fingerprint = str(
            checkpoint.get('control_contract_fingerprint', '')
        ).strip()
        if checkpoint_resolved_fingerprint != expected_resolved_control_contract_fingerprint:
            raise ValueError(
                'Checkpoint resolved control contract is incompatible with the configured '
                'state, paired-reference, B4, observation, and predictor interface artifacts.'
            )
    checkpoint_actor_control_fingerprint = str(checkpoint.get('actor_control_fingerprint', '')).strip()
    if expected_actor_control_fingerprint is not None and checkpoint_actor_control_fingerprint:
        if checkpoint_actor_control_fingerprint != expected_actor_control_fingerprint:
            raise ValueError(
                'Checkpoint actor control contract is incompatible with the configured wind-speed baseline '
                'and safety envelope.'
            )
    elif expected_control_fingerprint is not None:
        checkpoint_control_fingerprint = str(checkpoint.get('control_fingerprint', '')).strip()
        if checkpoint_control_fingerprint != expected_control_fingerprint:
            raise ValueError(
                'Checkpoint control contract is incompatible with the configured wind-speed baseline and safety envelope.'
            )


def _checkpoint_summary(checkpoint: Any, checkpoint_path: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {
        'checkpoint_path': str(checkpoint_path.resolve()),
        'checkpoint_loaded_at_utc': _utc_now(),
        'checkpoint_has_embedded_config': bool(isinstance(checkpoint, dict) and isinstance(checkpoint.get('config'), dict)),
    }
    if isinstance(checkpoint, dict):
        for key in (
            'stage',
            'episode',
            'total_steps',
            'bc_step',
            'bc_total_steps',
            'data_schema_version',
            'control_fingerprint',
            'actor_control_fingerprint',
            'online_training_fingerprint',
            'control_contract_fingerprint',
            'eval_checkpoint_metric',
            'eval_checkpoint_score',
        ):
            if key not in checkpoint:
                continue
            value = checkpoint[key]
            numeric = _finite_number(value)
            summary[key] = numeric if numeric is not None else str(value)
    return summary


def _resolve_device(config: dict[str, Any], requested_device: str) -> torch.device:
    device_name = requested_device.strip() or str(config.get('device', 'cpu'))
    if device_name.startswith('cuda') and not torch.cuda.is_available():
        device_name = 'cpu'
    return torch.device(device_name)


def _runtime_config(
    checkpoint: dict[str, Any],
    config_path: str,
    udp_host: str,
    udp_port: int,
) -> tuple[dict[str, Any], str]:
    if config_path.strip():
        config = load_config(config_path)
        config_source = str(normalize_user_path(config_path).resolve())
    else:
        embedded = checkpoint.get('config')
        if not isinstance(embedded, dict):
            raise ValueError('Provide --config because this checkpoint does not contain an embedded config.')
        config = copy.deepcopy(embedded)
        config_source = 'checkpoint:config'
    if not isinstance(config.get('env'), dict) or not isinstance(config.get('agent'), dict):
        raise ValueError('Inference config must contain env and agent objects.')
    if udp_host.strip():
        config['env']['udp_host'] = udp_host.strip()
    if udp_port > 0:
        config['env']['udp_port'] = int(udp_port)
    return config, config_source


def _apply_inference_torque_high(config: dict[str, Any], torque_high: float) -> dict[str, Any]:
    if not math.isfinite(torque_high) or torque_high <= 0.0:
        raise ValueError('--inference-torque-high must be a positive finite value in Nm.')
    if (
        config.get('control_contract_fingerprint') is not None
        or config.get('resolved_control_contract_path') is not None
        or config.get('control', {}).get('resolved_control_contract_fingerprint') is not None
    ):
        raise ValueError('--inference-torque-high cannot override a resolved control contract.')
    env_config = config.get('env')
    control_config = config.get('control')
    if not isinstance(env_config, dict) or not isinstance(control_config, dict):
        raise ValueError('Inference torque override requires env and control objects in the config.')
    action_low = list(env_config.get('action_low', []))
    action_high = list(env_config.get('action_high', []))
    if len(action_low) != 2 or len(action_high) != 2:
        raise ValueError('Inference torque override requires two-dimensional env action bounds.')
    current_high = float(action_high[0])
    if torque_high <= current_high:
        raise ValueError(
            f'--inference-torque-high must exceed the configured torque high {current_high:.0f} Nm.'
        )

    # Keep the actor's trained previous-action observation scale while widening only the sent-action envelope.
    env_config.setdefault('observation_action_low', copy.deepcopy(action_low))
    env_config.setdefault('observation_action_high', copy.deepcopy(action_high))
    action_high[0] = float(torque_high)
    env_config['action_high'] = action_high
    if isinstance(env_config.get('safe_action_high'), list):
        env_safe_high = list(env_config['safe_action_high'])
        env_safe_high[0] = float(torque_high)
        env_config['safe_action_high'] = env_safe_high

    control_safe_high = list(control_config.get('safe_action_high', action_high))
    if len(control_safe_high) != 2:
        raise ValueError('Inference torque override requires two-dimensional control safe bounds.')
    control_safe_high[0] = float(torque_high)
    control_config['safe_action_high'] = control_safe_high
    control_config['inference_torque_high'] = float(torque_high)
    return config


def _default_output_dir(checkpoint_path: Path, condition_id: int | None = None) -> Path:
    base_dir = checkpoint_path.parent.parent if checkpoint_path.parent.name == 'checkpoints' else checkpoint_path.parent
    checkpoint_suffix = checkpoint_path.stem.removeprefix('queue_')
    output_name = f'wind_{condition_id}_{checkpoint_suffix}' if condition_id is not None else checkpoint_suffix
    return base_dir / 'inference' / output_name


def _prepare_output_dir(output_dir: Path) -> Path:
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f'Inference output directory is not empty: {output_dir}')
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _configure_controller(controller: SafeResidualController, agent: TD3Agent) -> None:
    median = agent.actor.obs_median.detach().cpu().numpy().astype(np.float32, copy=True)
    iqr = agent.actor.obs_iqr.detach().cpu().numpy().astype(np.float32, copy=True)
    _configure_controller_observation_distribution(controller, median, iqr)


def _configure_controller_observation_distribution(
    controller: SafeResidualController,
    median: np.ndarray,
    iqr: np.ndarray,
) -> None:
    if median.shape != iqr.shape or not np.isfinite(median).all() or not np.isfinite(iqr).all():
        raise ValueError('Checkpoint actor observation normalizer is invalid.')
    controller.set_observation_distribution(median, iqr)


def _configure_controller_from_checkpoint(
    controller: SafeResidualController,
    checkpoint: dict[str, Any],
) -> None:
    actor_state = _agent_state_from_checkpoint(checkpoint).get('actor')
    if not isinstance(actor_state, dict):
        raise ValueError('Checkpoint actor state is missing or invalid.')
    median = actor_state.get('obs_median')
    iqr = actor_state.get('obs_iqr')
    if not isinstance(median, torch.Tensor) or not isinstance(iqr, torch.Tensor):
        raise ValueError('Checkpoint actor observation normalizer is missing.')
    _configure_controller_observation_distribution(
        controller,
        median.detach().cpu().numpy().astype(np.float32, copy=True),
        iqr.detach().cpu().numpy().astype(np.float32, copy=True),
    )


def _record_row(
    payload: dict[str, float],
    *,
    record_type: str,
    source_frame_role: str,
    episode: int,
    step: int,
    global_step: int,
    record_index: int,
    timing: dict[str, float] | None = None,
    policy_residual: np.ndarray | None = None,
    safe_residual: np.ndarray | None = None,
    executed_action: np.ndarray | None = None,
    requested_action: np.ndarray | None = None,
    info: dict[str, float] | None = None,
    reward: float | None = None,
    done: bool = False,
    terminal_reason: str = '',
    rollout_segment: int = 0,
) -> dict[str, Any]:
    info = _finite_metrics(info or {})
    row: dict[str, Any] = {name: payload.get(name, '') for name in BLADED_CSV_HEADER}
    row.update(
        {
            'record_type': record_type,
            'source_frame_role': source_frame_role,
            'inference_episode': int(episode),
            'inference_step': int(step),
            'inference_rollout_segment': int(rollout_segment),
            'inference_global_control_step': int(global_step),
            'inference_record_index': int(record_index),
            'wall_clock_utc': _utc_now(),
            'wall_clock_unix_s': time.time(),
            'policy_inference_ms': '' if timing is None else timing.get('policy_inference_ms', ''),
            'safety_projection_ms': '' if timing is None else timing.get('safety_projection_ms', ''),
            'udp_step_ms': '' if timing is None else timing.get('udp_step_ms', ''),
            'control_step_ms': '' if timing is None else timing.get('control_step_ms', ''),
            't_rx_monotonic_s': info.get('udp_receive_monotonic_s', ''),
            't_tx_monotonic_s': info.get('udp_action_sent_monotonic_s', ''),
            't_e2e_ms': info.get('udp_action_apply_lag_ms', ''),
            'policy_residual_torque_norm': '' if policy_residual is None else float(policy_residual[0]),
            'policy_residual_pitch_norm': '' if policy_residual is None else float(policy_residual[1]),
            'safe_residual_torque_norm': '' if safe_residual is None else float(safe_residual[0]),
            'safe_residual_pitch_norm': '' if safe_residual is None else float(safe_residual[1]),
            'executed_action_torque_norm': '' if executed_action is None else float(executed_action[0]),
            'executed_action_pitch_norm': '' if executed_action is None else float(executed_action[1]),
            'baseline_action_torque': info.get('control_baseline_action_torque', ''),
            'baseline_action_pitch': info.get('control_baseline_action_pitch', ''),
            'policy_delta_torque': info.get('control_policy_delta_torque', ''),
            'policy_delta_pitch': info.get('control_policy_delta_pitch', ''),
            'pre_safety_action_torque': info.get('control_pre_safety_action_torque', ''),
            'pre_safety_action_pitch': info.get('control_pre_safety_action_pitch', ''),
            'pitch_protection_floor': info.get('control_pitch_protection_floor', ''),
            'pitch_protection_state': info.get('control_pitch_protection_state', ''),
            'pitch_protection_applied': info.get('control_pitch_protection_applied', ''),
            'low_speed_stage': info.get('control_low_speed_stage', ''),
            'low_speed_scale': info.get('control_low_speed_scale', ''),
            'low_speed_reason': info.get('control_low_speed_reason', ''),
            'post_safety_action_torque': info.get('control_post_safety_action_torque', ''),
            'post_safety_action_pitch': info.get('control_post_safety_action_pitch', ''),
            'projected_action_torque': info.get('control_projected_action_torque', ''),
            'projected_action_pitch': info.get('control_projected_action_pitch', ''),
            'rate_limited_action_torque': info.get('control_rate_limited_action_torque', ''),
            'rate_limited_action_pitch': info.get('control_rate_limited_action_pitch', ''),
            'requested_action_torque': '' if requested_action is None else float(requested_action[0]),
            'requested_action_pitch': '' if requested_action is None else float(requested_action[1]),
            'sent_action_torque': info.get('action_torque', ''),
            'sent_action_pitch': info.get('action_pitch', ''),
            'sent_action_torque_delta': info.get('action_torque_delta', ''),
            'sent_action_pitch_delta': info.get('action_pitch_delta', ''),
            'step_reward': '' if reward is None else float(reward),
            'step_done': float(done),
            'step_invalid': float(
                info.get('env_invalid_prediction', 0.0) > 0.0 or info.get('env_invalid_reward', 0.0) > 0.0
            ),
            'terminal_reason': terminal_reason,
        }
    )
    for name in STEP_INFO_FIELDS:
        row[f'info_{name}'] = info.get(name, '')
    row['step_info_json'] = json.dumps(info, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return row


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def run_inference(
    config: dict[str, Any],
    checkpoint: dict[str, Any],
    checkpoint_path: Path,
    output_dir: Path,
    episodes: int,
    max_steps: int | None,
    device: torch.device,
    config_source: str,
    log_every_steps: int = 100,
    baseline_only: bool = False,
    checkpoint_control_fingerprint: str | None = None,
    checkpoint_actor_control_fingerprint: str | None = None,
) -> dict[str, Any]:
    if episodes <= 0:
        raise ValueError('episodes must be positive')
    if max_steps is not None and max_steps <= 0:
        raise ValueError('max_steps must be positive when provided')
    if log_every_steps < 0:
        raise ValueError('log_every_steps must be non-negative')
    if physical_backend_selected(config):
        config = copy.deepcopy(config)
        config['device'] = str(device)
        config['physical_shadow_execution_scope'] = {'entry': 'inference', 'episodes': episodes,
            'max_steps': max_steps, 'baseline_only': bool(baseline_only),
            'output_dir': str(output_dir.expanduser().resolve())}
        launch_identity = validate_physical_launch(config, project_root=PROJECT_ROOT,
            checkpoint_path=checkpoint_path, baseline_only=baseline_only)
        preflight_physical_config(config, project_root=PROJECT_ROOT)
        if not isinstance(launch_identity, dict):
            raise ValueError('Physical inference requires its validated execution identity')
        config['_physical_launch_operator_development_authorized'] = bool(
            launch_identity.get('operator_development_authorized', False)
        )
        checkpoint = load_verified_physical_checkpoint(checkpoint_path, launch_identity['actor_checkpoint_sha256'])
    # Production CLI inference always supplies a persisted frozen checkpoint.
    # Synthetic/in-memory callers may not; do not invent a file identity for
    # those runs, so their shadow rows remain ineligible for strict calibration.
    actor_policy_sha256 = _sha256_file(checkpoint_path) if checkpoint_path.is_file() else ''
    configured_control_contract = resolve_configured_control_contract(
        config,
        project_root=PROJECT_ROOT,
    )
    expected_resolved_control_contract_fingerprint = (
        None if configured_control_contract is None else configured_control_contract[1]
    )
    controller = SafeResidualController(config['env'], config.get('control', {}))
    checkpoint_contract_required = controller.checkpoint_contract_required(
        baseline_only=baseline_only,
    )
    expected_control_fingerprint = (
        checkpoint_control_fingerprint
        if checkpoint_control_fingerprint is not None
        else controller.control_fingerprint
        if checkpoint_contract_required
        else None
    )
    expected_actor_control_fingerprint = (
        checkpoint_actor_control_fingerprint
        if checkpoint_actor_control_fingerprint is not None
        else controller.actor_control_fingerprint
        if checkpoint_contract_required
        else None
    )
    _validate_data_schema(
        checkpoint,
        config,
        expected_control_fingerprint,
        expected_actor_control_fingerprint,
        expected_resolved_control_contract_fingerprint,
    )

    predictor_config = config.get('load_predictor', {})
    shadow_audit_max_records = int(predictor_config.get('shadow_audit_max_records', 250_000)
                                   if isinstance(predictor_config, dict) else 250_000)
    shadow_audit_max_record_bytes = int(predictor_config.get('shadow_audit_max_record_bytes', 64 * 1024)
                                        if isinstance(predictor_config, dict) else 64 * 1024)
    shadow_audit_max_total_bytes = int(predictor_config.get('shadow_audit_max_total_bytes', 256 * 1024 * 1024)
                                       if isinstance(predictor_config, dict) else 256 * 1024 * 1024)
    if any(value <= 0 for value in (shadow_audit_max_records, shadow_audit_max_record_bytes,
                                  shadow_audit_max_total_bytes)):
        raise ValueError('Load predictor shadow audit budgets must be positive')
    output_dir = _prepare_output_dir(output_dir)
    _write_json(output_dir / 'effective_config.json', config)
    checkpoint_meta = _checkpoint_summary(checkpoint, checkpoint_path)
    checkpoint_meta['config_source'] = config_source
    checkpoint_meta['device'] = str(device)
    checkpoint_meta['inference_mode'] = 'baseline_only' if baseline_only else 'policy'
    checkpoint_meta['inference_backend'] = 'pytorch'
    if 'inference_torque_high' in config.get('control', {}):
        checkpoint_meta['inference_torque_high'] = float(config['control']['inference_torque_high'])
    _write_json(output_dir / 'checkpoint_metadata.json', checkpoint_meta)

    env = WindTurbineEnv(config['env'], config.get('data', {}), device)
    queue_mode = bool(env.queue_lifecycle_enabled)
    physical_load_shadow = None
    queue_event_logger: ExperimentLogger | None = None
    try:
        load_predictor_shadow = _build_load_predictor_shadow(config, env, device)
        queue_config = config.get('queue', {})
        configured_condition_ids = tuple(sorted({int(value) for value in queue_config.get('condition_ids', ())}))
        identity_validator = (
            build_queue_case_identity_validator(
                queue_config,
                configured_condition_ids,
                operator_development_authorized=bool(
                    config.get('_physical_launch_operator_development_authorized', False)
                ),
            )
            if queue_mode and configured_condition_ids else None
        )
        strict_identity_enabled = isinstance(identity_validator, QueueStrictCaseIdentityValidator)
        agent = None
        if baseline_only:
            _configure_controller_from_checkpoint(controller, checkpoint)
        else:
            agent = _build_agent(config, env.obs_dim, env.action_dim, device)
            agent.load_state_dict(_agent_state_from_checkpoint(checkpoint))
            _configure_controller(controller, agent)
        logging_config = config.get('logging', {})
        if (
            physical_backend_selected(config)
            and isinstance(logging_config, dict)
            and logging_config.get('queue_event_full_state_enabled') is True
        ):
            queue_event_logger = ExperimentLogger(output_dir, logging_config)
    except Exception:
        env.close()
        raise
    print(
        f'[Infer] READY: checkpoint loaded, UDP listener bound at '
        f"{config['env'].get('udp_host', '0.0.0.0')}:{config['env'].get('udp_port', 8888)}, "
        f'device={device}, queue_mode={queue_mode}. Start Bladed now; waiting for the first frame.',
        flush=True,
    )

    records_path = output_dir / 'bladed_inference_steps.csv'
    raw_frames_path = output_dir / 'bladed_raw_frames.csv'
    episode_summary_path = output_dir / 'episode_summary.csv'
    run_started = time.perf_counter()
    episode_summaries: list[dict[str, Any]] = []
    identity_summaries: list[dict[str, Any]] = []
    predictor_shadow_finish_summaries: list[dict[str, Any]] = []
    shadow_audit_path = output_dir / 'load_predictor_shadow_audit.jsonl'
    shadow_audit_records_written = 0
    shadow_audit_bytes_written = 0

    def finish_shadow_slot(
        *,
        condition_id: int,
        condition_instance_id: int,
        identity_status: str,
        identity_error: str,
        identity_summary: dict[str, Any],
    ) -> dict[str, int]:
        nonlocal shadow_audit_records_written
        nonlocal shadow_audit_bytes_written
        if physical_load_shadow is not None:
            metadata = physical_condition_metadata(physical_load_shadow, condition_id=condition_id,
                condition_instance_id=condition_instance_id, identity_status=identity_status,
                identity_summary=identity_summary, queue_epoch=active_queue_epoch)
            metadata['identity_error'] = identity_error
            physical_load_shadow.finish_condition(metadata)
        if load_predictor_shadow is None:
            return {}
        finish_stats = load_predictor_shadow.finish_condition()
        audit_records = load_predictor_shadow.drain_audit_records()
        reference_identity: dict[str, Any] = {}
        if isinstance(identity_validator, QueueStrictCaseIdentityValidator):
            reference_entry = identity_validator.reference.entry(condition_id)
            reference_identity = {
                'scenario_id': reference_entry.scenario_id,
                'pid_raw_csv_path': str(reference_entry.pid_raw_csv_path),
                'time_window_start_s': float(reference_entry.time_window_start_s),
                'time_window_end_s': float(reference_entry.time_window_end_s),
                'expected_trace_signature': reference_entry.trace_signature,
                'expected_trace_sample_count': int(reference_entry.timestamps_s.size),
            }
        for raw_record in audit_records:
            record = {
                **raw_record,
                'identity_status': identity_status,
                'strict_case_matched': float(identity_status == 'strict_case_matched'),
                'identity_error': identity_error,
                'pid_raw_sha256': identity_summary.get('pid_raw_sha256', ''),
                **reference_identity,
                'predictor_interface_sha256': load_predictor_shadow.interface_sha256,
                'predictor_artifact_sha256': str(
                    predictor_config.get('predictor_artifact_sha256', '')
                    if isinstance(predictor_config, dict)
                    else ''
                ),
                'actor_policy_sha256': actor_policy_sha256,
                'actor_policy_version': 0,
            }
            serialized = json.dumps(
                record,
                ensure_ascii=False,
                allow_nan=False,
                separators=(',', ':'),
            ).encode('utf-8')
            record_bytes = len(serialized)
            if record_bytes > shadow_audit_max_record_bytes:
                raise ValueError(
                    'Load predictor shadow audit record exceeds configured byte budget: '
                    f'{record_bytes} > {shadow_audit_max_record_bytes}'
                )
            if shadow_audit_records_written + 1 > shadow_audit_max_records:
                raise ValueError('Load predictor shadow audit records exceed configured row budget')
            if shadow_audit_bytes_written + record_bytes + 1 > shadow_audit_max_total_bytes:
                raise ValueError('Load predictor shadow audit JSONL exceeds configured byte budget')
            with shadow_audit_path.open('ab') as handle:
                handle.write(serialized + b'\n')
            shadow_audit_records_written += 1
            shadow_audit_bytes_written += record_bytes + 1
        finish_summary = {
            **finish_stats,
            **load_predictor_shadow.audit_summary,
            'condition_id': int(condition_id),
            'condition_instance_id': int(condition_instance_id),
            'identity_status': identity_status,
            'audit_records_written_this_slot': len(audit_records),
        }
        predictor_shadow_finish_summaries.append(finish_summary)
        return finish_summary
    total_control_steps = 0
    total_records = 0
    status = 'completed'
    error_message = ''
    queue_started = False

    try:
        physical_load_shadow = build_physical_shadow(config, env,
            output_dir=output_dir, project_root=PROJECT_ROOT)
        with records_path.open('w', encoding='utf-8', newline='') as records_file, raw_frames_path.open(
            'w', encoding='utf-8', newline=''
        ) as raw_frames_file, episode_summary_path.open('w', encoding='utf-8', newline='') as episode_file:
            record_writer = csv.DictWriter(records_file, fieldnames=RECORD_FIELDS, extrasaction='ignore')
            raw_frame_writer = csv.DictWriter(raw_frames_file, fieldnames=BLADED_CSV_HEADER, extrasaction='ignore')
            episode_writer = csv.DictWriter(episode_file, fieldnames=EPISODE_SUMMARY_FIELDS, extrasaction='ignore')
            record_writer.writeheader()
            raw_frame_writer.writeheader()
            episode_writer.writeheader()

            for episode in range(1, episodes + 1):
                episode_started = time.perf_counter()
                episode_started_utc = _utc_now()
                if queue_mode:
                    obs, reset_info = env.start_pending_condition() if queue_started else env.reset()
                    queue_started = True
                    controller.reset_condition(obs, previous_action_real=env.last_sent_action)
                    active_train = bool(reset_info.get('condition_train_active', 0.0) > 0.0)
                    control_limit = max_steps
                else:
                    obs, reset_info = env.reset()
                    controller.reset_condition(obs, previous_action_real=env.last_sent_action)
                    active_train = True
                    control_limit = max_steps if max_steps is not None else env.episode_length
                active_condition_id = int(reset_info.get('udp_condition_id', -1.0))
                active_condition_instance_id = int(reset_info.get('udp_condition_instance_id', -1.0))
                active_queue_epoch = int(reset_info.get('udp_queue_epoch', -1.0))
                identity_status = 'not_configured'
                identity_error = ''
                identity_failed = False
                identity_summary_written = False
                shadow_slot_finished = False
                identity_final_metrics: dict[str, float] = {}
                identity_slot_summary: dict[str, Any] = {
                    'identity_status': identity_status,
                    'condition_id': active_condition_id,
                    'condition_instance_id': active_condition_instance_id,
                }
                if strict_identity_enabled and active_train:
                    assert isinstance(identity_validator, QueueStrictCaseIdentityValidator)
                    _, _, runtime_case_fingerprint = _strict_identity_sample(env)
                    identity_validator.reset_condition(
                        active_condition_id,
                        active_condition_instance_id,
                        runtime_case_fingerprint=runtime_case_fingerprint,
                    )
                    identity_status = 'strict_validation_in_progress'
                if load_predictor_shadow is not None:
                    load_predictor_shadow.reset_condition(
                        active_condition_id,
                        active_condition_instance_id,
                    )
                if physical_load_shadow is not None:
                    physical_load_shadow.reset_condition(active_condition_id, active_condition_instance_id)
                print(
                    f'[Infer] episode={episode} START condition={int(reset_info.get("udp_condition_id", -1))} '
                    f'epoch={int(reset_info.get("udp_queue_epoch", -1))} '
                    f'instance={int(reset_info.get("udp_condition_instance_id", -1))} '
                    f'wind={reset_info.get("initial_wind_speed", 0.0):.3f} '
                    f'phase_train={int(active_train)}',
                    flush=True,
                )
                done = False
                step = 0
                episode_reward = 0.0
                control_step_times: list[float] = []
                policy_times: list[float] = []
                udp_step_times: list[float] = []
                edge_e2e_times: list[float] = []
                fallback_steps = 0
                timeout_steps = 0
                invalid_steps = 0
                unsafe_stop = False
                terminal_reason = 'queue_step_limit_reached' if queue_mode else 'max_steps_reached'
                last_info: dict[str, float] = {}

                while not done and (control_limit is None or step < control_limit):
                    step += 1
                    total_control_steps += 1
                    cycle_started = time.perf_counter()
                    raw_payload = env.current_payload
                    origin_invalid_fields = tuple(env.current_payload_invalid_fields)
                    identity_step_metrics: dict[str, float] = {}
                    if strict_identity_enabled and active_train and not identity_failed:
                        assert isinstance(identity_validator, QueueStrictCaseIdentityValidator)
                        try:
                            sim_time_s, wind_speed_mps, runtime_case_fingerprint = _strict_identity_sample(env)
                            identity_step_metrics = identity_validator.observe(
                                active_condition_id,
                                active_condition_instance_id,
                                obs,
                                sim_time_s=sim_time_s,
                                wind_speed_mps=wind_speed_mps,
                                runtime_case_fingerprint=runtime_case_fingerprint,
                            )
                        except (QueueCaseIdentityMismatch, ValueError, RuntimeError) as exc:
                            identity_failed = True
                            identity_status = 'identity_mismatch'
                            identity_error = f'{type(exc).__name__}: {exc}'
                            identity_step_metrics = {
                                'queue_case_identity_validation_active': 1.0,
                                'queue_case_identity_validation_level_strict': 1.0,
                                'queue_case_identity_mismatch': 1.0,
                                **(exc.metrics if isinstance(exc, QueueCaseIdentityMismatch) else {}),
                            }

                    policy_started = time.perf_counter()
                    if baseline_only:
                        policy_residual = np.zeros(env.action_dim, dtype=np.float32)
                    else:
                        if agent is None:
                            raise RuntimeError('Policy inference agent is not initialized.')
                        obs_tensor = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
                        policy_residual = agent.select_action(obs_tensor, deterministic=True).squeeze(0).cpu().numpy()
                    policy_ended = time.perf_counter()

                    safety_started = time.perf_counter()
                    safe_action = controller.action_from_policy(
                        policy_residual,
                        obs,
                        allow_policy=False if baseline_only else (active_train if queue_mode else True),
                    )
                    baseline_action = safe_action.baseline_action_real
                    safety_ended = time.perf_counter()

                    udp_step_started = time.perf_counter()
                    action_sent_timestamps: list[float] = []

                    def sent_side_work(sent_monotonic_s: float) -> None:
                        action_sent_timestamps.append(sent_monotonic_s)
                        if physical_load_shadow is not None and active_train:
                            submit_physical_sent_frame(physical_load_shadow, env,
                                payload=raw_payload, condition_id=active_condition_id,
                                condition_instance_id=active_condition_instance_id,
                                sent_action_real=safe_action.action_real, actor_policy_version=0,
                                actor_policy_sha256=actor_policy_sha256, sent_monotonic_s=sent_monotonic_s)

                    step_result = env.step(
                        safe_action.action_norm,
                        on_action_sent=sent_side_work,
                        action_real=safe_action.action_real,
                    )
                    udp_step_ended = time.perf_counter()

                    info = dict(step_result.info)
                    info.update(identity_step_metrics)
                    if action_sent_timestamps:
                        info['udp_action_sent_monotonic_s'] = action_sent_timestamps[-1]
                    transition_store_eligible = bool(info.get('transition_store_eligible', 0.0) > 0.0)
                    unsafe_stop = controller.observe_step(float(info.get('power', 0.0))) if (
                        not queue_mode or (active_train and transition_store_eligible)
                    ) else False
                    if physical_load_shadow is not None and active_train:
                        complete_physical_step(physical_load_shadow, env, payload=raw_payload,
                                               info=info, unsafe_stop=unsafe_stop)
                        info.update(physical_load_shadow.diagnostics)
                    sent_action = np.asarray(
                        [
                            info.get('action_torque', safe_action.action_real[0]),
                            info.get('action_pitch', safe_action.action_real[1]),
                        ],
                        dtype=np.float64,
                    )
                    info.update(
                        {
                            'control_ood_score': float(safe_action.ood_score),
                            'actor_policy_version': 0.0,
                            'control_baseline_only': float(baseline_only),
                            **safe_action_metrics(safe_action, sent_action_real=sent_action),
                            'control_unsafe_stop': float(unsafe_stop),
                        }
                    )
                    if load_predictor_shadow is not None and active_train:
                        pid_peaks = env.reward_fn.pid_blade_root_peak_reference(active_condition_id)
                        state_encoding = np.zeros(8, dtype=np.float64)
                        if safe_action.optimization_state_code >= 0:
                            state_encoding[safe_action.optimization_state_code] = 1.0
                        if safe_action.control_region_code >= 0:
                            state_encoding[5 + safe_action.control_region_code] = 1.0
                        shadow_result = load_predictor_shadow.observe(
                            condition_id=active_condition_id,
                            condition_instance_id=active_condition_instance_id,
                            payload=raw_payload,
                            sent_action_real=sent_action,
                            projected_candidate_action_real=safe_action.action_real,
                            state_encoding=state_encoding,
                            pid_peak_abs_nm=np.asarray(pid_peaks, dtype=np.float64),
                            optimization_state=(
                                OPTIMIZATION_STATE_ENCODING_ORDER[int(safe_action.optimization_state_code)].value
                                if 0 <= int(safe_action.optimization_state_code) < len(
                                    OPTIMIZATION_STATE_ENCODING_ORDER
                                )
                                else 'unknown'
                            ),
                            execution_mode=safe_action.pre_send_execution_mode.value,
                            outcome_tag=(
                                'condition_boundary'
                                if info.get('condition_boundary', 0.0) > 0.0
                                else 'timeout'
                                if info.get('env_timeout', 0.0) > 0.0
                                or info.get('udp_inter_frame_timeout', 0.0) > 0.0
                                else 'invalid'
                                if info.get('env_invalid_prediction', 0.0) > 0.0
                                or info.get('env_invalid_reward', 0.0) > 0.0
                                else 'complete'
                            ),
                            actor_policy_version=0,
                            actor_policy_sha256=actor_policy_sha256,
                            predictor_artifact_sha256=str(
                                predictor_config.get('predictor_artifact_sha256', '')
                                if isinstance(predictor_config, dict)
                                else ''
                            ),
                        )
                        info.update(shadow_result.diagnostics)
                    last_info = info
                    condition_boundary = bool(queue_mode and info.get('condition_boundary', 0.0) > 0.0)
                    if queue_event_logger is not None:
                        observed_payload = None
                        observed_invalid_fields: tuple[str, ...] = ()
                        if info.get('udp_handshake_complete', 0.0) > 0.0:
                            observed_payload = env.current_payload
                            observed_invalid_fields = tuple(env.current_payload_invalid_fields)
                        wire_snapshot = env.last_successful_action_send
                        observed_sample_valid = bool(
                            observed_payload is not None
                            and not origin_invalid_fields
                            and not observed_invalid_fields
                            and info.get('transition_store_eligible', 0.0) > 0.0
                            and info.get('env_invalid_prediction', 0.0) <= 0.0
                            and info.get('env_invalid_reward', 0.0) <= 0.0
                            and info.get('udp_inter_frame_timeout', 0.0) <= 0.0
                        )
                        event_frame = {
                            'origin_payload': raw_payload,
                            'observed_payload': observed_payload,
                            'origin_invalid_fields': origin_invalid_fields,
                            'observed_invalid_fields': observed_invalid_fields,
                            'observed_sample_valid': observed_sample_valid,
                            'wire_action_real': wire_snapshot.get('action_real'),
                            'actor_policy_sha256': actor_policy_sha256,
                        }
                        event_metrics = {
                            'queue_condition_round_index': 1.0,
                            'queue_condition_id': float(active_condition_id),
                            'queue_condition_instance_id': float(active_condition_instance_id),
                            'queue_epoch': float(active_queue_epoch),
                            'actor_policy_version': 0.0,
                            'queue_event_sample_valid': float(observed_sample_valid),
                        }
                        event_metric_names = (
                            'udp_frame_id', 'udp_next_frame_id', 'udp_sim_time', 'udp_next_sim_time',
                            'udp_phase_elapsed_s', 'wind_t', 'wind_next', 'power', 'power_t_metric_raw',
                            'power_metric_raw', 'power_reward_metric', 'power_state_real', 'action_torque',
                            'action_pitch', 'action_torque_delta', 'action_pitch_delta',
                            'reward_acc_raw', 'reward_shaft_raw', 'reward_relative_power_raw',
                            'reward_relative_energy_increment_j', 'reward_persistent_power_deficit_ewma',
                            'reward_pid_power_reference_raw', 'control_pre_send_execution_mode',
                            'control_optimization_state_code', 'control_region_code', 'control_low_speed_stage',
                            'control_ood_detected', 'control_ood_score', 'control_baseline_action_torque',
                            'control_baseline_action_pitch', 'control_policy_delta_torque',
                            'control_policy_delta_pitch', 'control_residual_torque', 'control_residual_pitch',
                            'control_pre_safety_action_torque', 'control_pre_safety_action_pitch',
                            'control_projected_action_torque', 'control_projected_action_pitch',
                            'control_sent_action_torque', 'control_sent_action_pitch',
                            'control_action_projection_applied', 'control_action_rate_limited',
                            'control_post_step_timeout', 'control_post_step_invalid',
                            'control_post_step_unsafe_stop', 'control_post_step_condition_boundary',
                            'physical_load_shadow_result_available',
                            'physical_load_shadow_result_origin_time_s',
                            'physical_load_shadow_prediction_supported_h100',
                            'physical_load_shadow_prediction_supported_h200',
                            'physical_load_shadow_predicted_peak_nm_h100_blade_1',
                            'physical_load_shadow_predicted_peak_nm_h100_blade_2',
                            'physical_load_shadow_predicted_peak_nm_h100_blade_3',
                            'physical_load_shadow_predicted_peak_nm_h200_blade_1',
                            'physical_load_shadow_predicted_peak_nm_h200_blade_2',
                            'physical_load_shadow_predicted_peak_nm_h200_blade_3',
                        )
                        event_metrics.update({
                            name: info[name] for name in event_metric_names if name in info
                        })
                        for blade_index in range(1, 4):
                            for suffix in ('raw', 'peak_reference_nm'):
                                name = f'reward_blade_root_{blade_index}_{suffix}'
                                if name in info:
                                    event_metrics[name] = info[name]
                        queue_event_logger.log_metrics(
                            'queue_train_step', total_control_steps, event_metrics, event_frame=event_frame,
                        )
                        if (
                            condition_boundary
                            or unsafe_stop
                            or info.get('env_timeout', 0.0) > 0.0
                            or info.get('udp_inter_frame_timeout', 0.0) > 0.0
                        ):
                            queue_event_logger.finish_queue_event_slot(
                                termination_reason=(
                                    'condition_boundary' if condition_boundary else 'unsafe_stop' if unsafe_stop
                                    else 'timeout'
                                ),
                            )
                    if strict_identity_enabled and active_train and condition_boundary:
                        assert isinstance(identity_validator, QueueStrictCaseIdentityValidator)
                        next_condition_id = int(info.get('udp_next_condition_id', -1.0))
                        if not identity_failed:
                            try:
                                if next_condition_id == active_condition_id:
                                    sim_time_s, wind_speed_mps, runtime_case_fingerprint = _strict_identity_sample(env)
                                    identity_final_metrics.update(identity_validator.observe(
                                        active_condition_id,
                                        active_condition_instance_id,
                                        step_result.obs,
                                        sim_time_s=sim_time_s,
                                        wind_speed_mps=wind_speed_mps,
                                        runtime_case_fingerprint=runtime_case_fingerprint,
                                    ))
                                final_metrics, identity_slot_summary = identity_validator.finish_condition(
                                    active_condition_id,
                                    active_condition_instance_id,
                                )
                                identity_final_metrics.update(final_metrics)
                                identity_status = str(identity_slot_summary['identity_status'])
                                identity_slot_summary['identity_error'] = ''
                            except (QueueCaseIdentityMismatch, ValueError, RuntimeError) as exc:
                                identity_failed = True
                                identity_status = 'identity_mismatch'
                                identity_error = f'{type(exc).__name__}: {exc}'
                                identity_final_metrics.update({
                                    'queue_case_identity_validation_active': 1.0,
                                    'queue_case_identity_validation_level_strict': 1.0,
                                    'queue_case_identity_mismatch': 1.0,
                                    **(exc.metrics if isinstance(exc, QueueCaseIdentityMismatch) else {}),
                                })
                                identity_slot_summary = {
                                    'identity_status': identity_status,
                                    'condition_id': active_condition_id,
                                    'condition_instance_id': active_condition_instance_id,
                                    'identity_error': identity_error,
                                    **identity_final_metrics,
                                }
                        else:
                            identity_slot_summary = {
                                'identity_status': identity_status,
                                'condition_id': active_condition_id,
                                'condition_instance_id': active_condition_instance_id,
                                'identity_error': identity_error,
                                **identity_final_metrics,
                            }
                        info.update(identity_final_metrics)
                        info['queue_case_identity_strict_case_matched'] = float(
                            identity_status == 'strict_case_matched'
                        )
                        info['queue_case_identity_mismatch'] = float(identity_status == 'identity_mismatch')
                        identity_slot_summary.update({
                            'identity_status': identity_status,
                            'identity_error': identity_error,
                            'condition_id': active_condition_id,
                            'condition_instance_id': active_condition_instance_id,
                        })
                        identity_summary_path = output_dir / (
                            f'case_identity_condition_{active_condition_id:02d}_'
                            f'instance_{active_condition_instance_id:03d}.json'
                        )
                        _write_json(identity_summary_path, identity_slot_summary)
                        identity_slot_summary['summary_path'] = str(identity_summary_path)
                        identity_summaries.append(identity_slot_summary)
                        identity_summary_written = True
                    if unsafe_stop:
                        done = True
                        terminal_reason = 'safety_stop'
                    elif info.get('env_timeout', 0.0) > 0.0:
                        done = True
                        terminal_reason = 'udp_timeout'
                    elif queue_mode and condition_boundary:
                        done = active_train
                        if done:
                            terminal_reason = 'condition_complete'
                    else:
                        done = bool(step_result.done)
                    episode_reward += float(step_result.reward)

                    timing = {
                        'policy_inference_ms': (policy_ended - policy_started) * 1000.0,
                        'safety_projection_ms': (safety_ended - safety_started) * 1000.0,
                        'udp_step_ms': (udp_step_ended - udp_step_started) * 1000.0,
                        'control_step_ms': (udp_step_ended - cycle_started) * 1000.0,
                    }
                    control_step_times.append(timing['control_step_ms'])
                    policy_times.append(timing['policy_inference_ms'])
                    udp_step_times.append(timing['udp_step_ms'])
                    edge_e2e_ms = _finite_number(info.get('udp_action_apply_lag_ms'))
                    if edge_e2e_ms is not None and edge_e2e_ms >= 0.0:
                        edge_e2e_times.append(edge_e2e_ms)
                    fallback_steps += int(safe_action.fallback_active)
                    timeout_steps += int(info.get('env_timeout', 0.0) > 0.0)
                    invalid_steps += int(
                        info.get('env_invalid_prediction', 0.0) > 0.0 or info.get('env_invalid_reward', 0.0) > 0.0
                    )

                    if not queue_mode and step_result.done and not done:
                        terminal_reason = 'environment_done'

                    if (load_predictor_shadow is not None or physical_load_shadow is not None) and done:
                        finish_metrics = finish_shadow_slot(
                            condition_id=active_condition_id,
                            condition_instance_id=active_condition_instance_id,
                            identity_status=identity_status,
                            identity_error=identity_error,
                            identity_summary=identity_slot_summary,
                        )
                        info.update({
                            f'load_predictor_shadow_finish_{key}': float(value)
                            for key, value in finish_metrics.items()
                            if isinstance(value, (int, float))
                        })
                        shadow_slot_finished = True

                    total_records += 1
                    record_writer.writerow(
                        _record_row(
                            raw_payload,
                            record_type='control_step',
                            source_frame_role='control_input',
                            episode=episode,
                            step=step,
                            rollout_segment=(step - 1) // env.rollout_segment_steps + 1,
                            global_step=total_control_steps,
                            record_index=total_records,
                            timing=timing,
                            policy_residual=policy_residual,
                            safe_residual=safe_action.residual_norm,
                            executed_action=safe_action.executed_action_norm,
                            requested_action=safe_action.action_real,
                            info=info,
                            reward=step_result.reward,
                            done=done,
                            terminal_reason=terminal_reason if done else '',
                        )
                    )
                    raw_frame_writer.writerow({name: raw_payload.get(name, '') for name in BLADED_CSV_HEADER})
                    if log_every_steps > 0 and (step == 1 or step % log_every_steps == 0 or done):
                        print(
                            f'[Infer] live episode={episode} step={step} '
                            f'condition={int(info.get("udp_condition_id", -1))} '
                            f'wind={info.get("wind_next", info.get("wind_t", 0.0)):.3f} '
                            f'sim_time={info.get("udp_next_sim_time", info.get("udp_sim_time", 0.0)):.2f} '
                            f'ood={safe_action.ood_score:.3f} '
                            f'fallback={int(safe_action.fallback_active)} '
                            f'control_ms={timing["control_step_ms"]:.2f}',
                            flush=True,
                        )
                    if queue_mode and condition_boundary and not done:
                        obs, transition_info = env.start_pending_condition()
                        controller.reset_condition(obs, previous_action_real=env.last_sent_action)
                        active_train = bool(transition_info.get('condition_train_active', 0.0) > 0.0)
                        active_condition_id = int(transition_info.get('udp_condition_id', -1.0))
                        active_condition_instance_id = int(
                            transition_info.get('udp_condition_instance_id', -1.0)
                        )
                        active_queue_epoch = int(transition_info.get('udp_queue_epoch', -1.0))
                        if load_predictor_shadow is not None:
                            load_predictor_shadow.reset_condition(
                                active_condition_id,
                                active_condition_instance_id,
                            )
                        if physical_load_shadow is not None:
                            physical_load_shadow.reset_condition(active_condition_id, active_condition_instance_id)
                        if strict_identity_enabled and active_train:
                            assert isinstance(identity_validator, QueueStrictCaseIdentityValidator)
                            _, _, runtime_case_fingerprint = _strict_identity_sample(env)
                            identity_validator.reset_condition(
                                active_condition_id,
                                active_condition_instance_id,
                                runtime_case_fingerprint=runtime_case_fingerprint,
                            )
                            identity_status = 'strict_validation_in_progress'
                            identity_error = ''
                            identity_failed = False
                            identity_summary_written = False
                            identity_final_metrics = {}
                            identity_slot_summary = {
                                'identity_status': identity_status,
                                'condition_id': active_condition_id,
                                'condition_instance_id': active_condition_instance_id,
                            }
                        shadow_slot_finished = False
                    else:
                        obs = step_result.obs

                if strict_identity_enabled and active_train and not identity_summary_written:
                    if identity_status != 'identity_mismatch':
                        identity_status = 'identity_incomplete'
                        identity_error = (
                            'Inference condition ended before a complete strict identity boundary was observed'
                        )
                    identity_slot_summary.update({
                        'identity_status': identity_status,
                        'condition_id': active_condition_id,
                        'condition_instance_id': active_condition_instance_id,
                        'identity_error': identity_error,
                        'queue_case_identity_full_trace_complete': 0.0,
                        'queue_case_identity_strict_case_matched': 0.0,
                        **identity_final_metrics,
                    })
                    identity_summary_path = output_dir / (
                        f'case_identity_condition_{active_condition_id:02d}_'
                        f'instance_{active_condition_instance_id:03d}.json'
                    )
                    _write_json(identity_summary_path, identity_slot_summary)
                    identity_slot_summary['summary_path'] = str(identity_summary_path)
                    identity_summaries.append(identity_slot_summary)
                    identity_summary_written = True

                if (load_predictor_shadow is not None or physical_load_shadow is not None) and not shadow_slot_finished:
                    finish_shadow_slot(
                        condition_id=active_condition_id,
                        condition_instance_id=active_condition_instance_id,
                        identity_status=identity_status,
                        identity_error=identity_error,
                        identity_summary=identity_slot_summary,
                    )
                    shadow_slot_finished = True

                if last_info.get('env_timeout', 0.0) <= 0.0:
                    total_records += 1
                    record_writer.writerow(
                        _record_row(
                            env.current_payload,
                            record_type='terminal_observation',
                            source_frame_role='post_action_terminal',
                            episode=episode,
                            step=step,
                            rollout_segment=(step - 1) // env.rollout_segment_steps + 1 if step else 0,
                            global_step=total_control_steps,
                            record_index=total_records,
                            done=True,
                            terminal_reason=terminal_reason,
                        )
                    )
                    terminal_payload = env.current_payload
                    raw_frame_writer.writerow({name: terminal_payload.get(name, '') for name in BLADED_CSV_HEADER})

                episode_duration_ms = (time.perf_counter() - episode_started) * 1000.0
                episode_summary = {
                    'episode': episode,
                    'start_wall_clock_utc': episode_started_utc,
                    'end_wall_clock_utc': _utc_now(),
                    'control_steps': step,
                    'record_count': step + int(last_info.get('env_timeout', 0.0) <= 0.0),
                    'episode_reward': episode_reward,
                    'reward_per_step': episode_reward / step if step else 0.0,
                    'episode_duration_ms': episode_duration_ms,
                    'control_step_ms_mean': _mean(control_step_times),
                    'control_step_ms_max': max(control_step_times) if control_step_times else 0.0,
                    'policy_inference_ms_mean': _mean(policy_times),
                    'udp_step_ms_mean': _mean(udp_step_times),
                    't_e2e_ms_mean': _mean(edge_e2e_times),
                    't_e2e_ms_max': max(edge_e2e_times) if edge_e2e_times else 0.0,
                    'fallback_step_count': fallback_steps,
                    'fallback_fraction': fallback_steps / step if step else 0.0,
                    'unsafe_stop': float(unsafe_stop),
                    'timeout_step_count': timeout_steps,
                    'invalid_step_count': invalid_steps,
                    'terminal_reason': terminal_reason,
                    'queue_mode': float(queue_mode),
                    'queue_condition_id': last_info.get('udp_condition_id', ''),
                    'queue_condition_instance_id': last_info.get('udp_condition_instance_id', ''),
                    'queue_epoch': last_info.get('udp_queue_epoch', ''),
                    'queue_phase_code': last_info.get('udp_phase_code', ''),
                    'rollout_segments': (step - 1) // env.rollout_segment_steps + 1 if step else 0,
                    'case_identity_status': identity_status,
                    'case_identity_full_trace_complete': float(
                        identity_final_metrics.get('queue_case_identity_full_trace_complete', 0.0)
                    ),
                    'case_identity_strict_case_matched': float(
                        identity_final_metrics.get('queue_case_identity_strict_case_matched', 0.0)
                    ),
                    'case_identity_pid_raw_sha256': identity_slot_summary.get('pid_raw_sha256', ''),
                    'case_identity_error': identity_error,
                }
                episode_writer.writerow(episode_summary)
                records_file.flush()
                raw_frames_file.flush()
                episode_file.flush()
                episode_summaries.append(episode_summary)
                print(
                    f'[Infer] episode={episode} steps={step} reward={episode_reward:.4f} '
                    f'fallback={fallback_steps}/{step} reason={terminal_reason}',
                    flush=True,
                )
                if queue_mode and terminal_reason != 'condition_complete':
                    break
    except Exception as exc:
        status = 'failed'
        error_message = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        env.close()
        physical_shadow_summary = None
        if physical_load_shadow is not None:
            try:
                physical_shadow_summary = physical_load_shadow.close()
            except Exception as exc:
                physical_shadow_summary = {'backend': 'physical_fir_v2', 'finalized': False,
                    'faulted': True, 'summary_status': 'FINALIZATION_ERROR',
                    'last_error_type': type(exc).__name__}
        queue_event_capture_summary = None
        if queue_event_logger is not None:
            try:
                queue_event_logger.finish_queue_event_slot(termination_reason='run_finalized')
                queue_event_logger.flush()
                queue_event_capture_summary = {
                    'event_windows_seen': queue_event_logger._queue_event_windows_seen,
                    'event_windows_written': queue_event_logger._queue_event_windows_written,
                    'event_windows_dropped': queue_event_logger._queue_event_windows_dropped,
                    'event_bytes_written': queue_event_logger._queue_event_bytes_written,
                    'event_max_record_bytes_observed': queue_event_logger._queue_event_max_record_bytes_observed,
                    'event_window_max_record_bytes': queue_event_logger.logging_config[
                        'queue_event_window_max_record_bytes'
                    ],
                    'event_windows_max_total_bytes': queue_event_logger.logging_config[
                        'queue_event_windows_max_total_bytes'
                    ],
                    'event_windows_jsonl': str(output_dir / 'compact_queue_condition_event_windows.jsonl'),
                    'compact_summary_json': str(output_dir / 'compact_metrics_summary.json'),
                }
            except Exception as exc:
                queue_event_capture_summary = {
                    'finalized': False,
                    'error': f'{type(exc).__name__}: {exc}',
                }
        performance_summary_path = output_dir / 'edge_performance_summary.json'
        try:
            performance_summary = write_edge_performance_report(records_path, performance_summary_path)
        except (OSError, ValueError) as exc:
            performance_summary = {
                'mode': 'edge_performance_v1',
                'valid': False,
                'reason': f'{type(exc).__name__}: {exc}',
                'T1_pass': False,
            }
        run_summary = {
            'status': status,
            'error': error_message,
            'started_utc': checkpoint_meta['checkpoint_loaded_at_utc'],
            'finished_utc': _utc_now(),
            'elapsed_ms': (time.perf_counter() - run_started) * 1000.0,
            'episodes_requested': episodes,
            'episodes_completed': len(episode_summaries),
            'queue_mode': queue_mode,
            'total_control_steps': total_control_steps,
            'total_records': total_records,
            'records_csv': str(records_path),
            'raw_frames_csv': str(raw_frames_path),
            'episode_summary_csv': str(episode_summary_path),
            'checkpoint_metadata_json': str(output_dir / 'checkpoint_metadata.json'),
            'effective_config_json': str(output_dir / 'effective_config.json'),
            'edge_performance_summary_json': str(performance_summary_path),
            'edge_performance': performance_summary,
            'inference_mode': 'baseline_only' if baseline_only else 'policy',
            'inference_backend': 'pytorch',
            'baseline_only': baseline_only,
            'episode_summaries': episode_summaries,
            'case_identity_validation_level': 'strict' if strict_identity_enabled else 'not_strict',
            'case_identity_summaries': identity_summaries,
            'strict_case_matched_slot_count': sum(
                item.get('identity_status') == 'strict_case_matched'
                for item in identity_summaries
            ),
            'all_observed_train_slots_strict_case_matched': bool(identity_summaries)
            and all(item.get('identity_status') == 'strict_case_matched' for item in identity_summaries),
            'load_predictor_shadow_audit_jsonl': (
                str(shadow_audit_path) if load_predictor_shadow is not None else None
            ),
            'load_predictor_shadow_audit_records_written': shadow_audit_records_written,
            'load_predictor_shadow_audit_bytes_written': shadow_audit_bytes_written,
            'load_predictor_shadow_audit_max_records': shadow_audit_max_records,
            'load_predictor_shadow_audit_max_record_bytes': shadow_audit_max_record_bytes,
            'load_predictor_shadow_audit_max_total_bytes': shadow_audit_max_total_bytes,
            'load_predictor_shadow_finish_summaries': predictor_shadow_finish_summaries,
            'physical_load_shadow': physical_shadow_summary,
            'queue_event_capture': queue_event_capture_summary,
        }
        _write_json(output_dir / 'run_summary.json', run_summary)

    return run_summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run a saved TD3+BC checkpoint against the live Bladed UDP environment and save full frame records.',
    )
    parser.add_argument('--checkpoint', required=True, help='Saved TD3 checkpoint, such as checkpoints/best_eval.pt.')
    parser.add_argument('--config', default='', help='Optional runtime config. Defaults to config embedded in the checkpoint.')
    parser.add_argument(
        '--output-dir',
        default='',
        help='Empty directory for inference outputs. Defaults to the checkpoint experiment inference directory.',
    )
    parser.add_argument(
        '--condition-id',
        type=int,
        default=None,
        help='Condition ID used in the default output directory name; does not select or alter the Bladed case.',
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=1,
        help='Legacy episodes, or completed TRAIN conditions when using the v7 queue protocol.',
    )
    parser.add_argument(
        '--max-steps',
        type=int,
        default=0,
        help='Optional per-episode or per-condition diagnostic cap; 0 leaves queue conditions uncapped.',
    )
    parser.add_argument(
        '--log-every-steps',
        type=int,
        default=100,
        help='Print live control progress every N steps; 0 disables periodic logs (default: 100).',
    )
    parser.add_argument('--device', default='', help='Override device, for example cpu or cuda:0.')
    parser.add_argument('--udp-host', default='', help='Override env.udp_host for this run.')
    parser.add_argument('--udp-port', type=int, default=0, help='Override env.udp_port for this run.')
    parser.add_argument(
        '--inference-torque-high',
        type=float,
        default=0.0,
        help=(
            'Inference-only torque upper envelope in Nm. Keeps trained observation scaling and validates '
            'the checkpoint against the original actor contract; use only after checking the Bladed actuator limit.'
        ),
    )
    parser.add_argument(
        '--baseline-only',
        action='store_true',
        help='Disable actor inference and send the wind-speed baseline through the same safety layer.',
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    checkpoint_path = normalize_user_path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if not isinstance(checkpoint, dict):
        raise ValueError('Checkpoint must deserialize to a mapping.')
    config, config_source = _runtime_config(checkpoint, args.config, args.udp_host, args.udp_port)
    checkpoint_control_fingerprint = None
    checkpoint_actor_control_fingerprint = None
    if args.inference_torque_high > 0.0:
        contract_controller = SafeResidualController(config['env'], config.get('control', {}))
        checkpoint_control_fingerprint = contract_controller.control_fingerprint
        checkpoint_actor_control_fingerprint = contract_controller.actor_control_fingerprint
        config = _apply_inference_torque_high(config, args.inference_torque_high)
        config_source = f'{config_source};inference_torque_high={args.inference_torque_high:.0f}'
    device = _resolve_device(config, args.device)
    output_dir = (
        normalize_user_path(args.output_dir)
        if args.output_dir.strip()
        else _default_output_dir(checkpoint_path, args.condition_id)
    )
    summary = run_inference(
        config=config,
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        episodes=args.episodes,
        max_steps=args.max_steps or None,
        device=device,
        config_source=config_source,
        log_every_steps=args.log_every_steps,
        baseline_only=args.baseline_only,
        checkpoint_control_fingerprint=checkpoint_control_fingerprint,
        checkpoint_actor_control_fingerprint=checkpoint_actor_control_fingerprint,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()