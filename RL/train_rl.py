from __future__ import annotations

import argparse
import copy
from contextlib import ExitStack
import csv
from datetime import datetime
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from RL.agents import TD3Agent
from RL.data import OfflineBCDataset
from RL.envs import WindTurbineEnv
from RL.envs.wind_turbine_env import QUEUE_UDP_PROTOCOL_VERSION
from RL.replay import (
    ONLINE_BLADED_SOURCE,
    REPLAY_SOURCE_NAMES,
    ConditionStratifiedReplayBuffer,
    ReplayBuffer,
)
from RL.utils.condition_queue import ConditionQueueLifecycle
from RL.utils.config import load_config, normalize_user_path
from RL.utils.control_contract import resolve_configured_control_contract
from RL.utils.control_execution import (
    PreSendExecutionMode,
    execution_mode_replay_contract,
    post_step_outcome_metrics,
    pre_send_execution_mode_from_code,
    replay_source_for_execution_mode,
    td3_critic_replay_allowed,
)
from RL.utils.constants import DATA_SCHEMA_VERSION, OBSERVATION_VARIABLES
from RL.utils.dual_constraint import (
    ConditionConstraintAccumulator,
    DualConstraintConfig,
    DualConstraintController,
)
from RL.utils.critic_q_health import (
    ActorQHealthGate,
    ActorQHealthGateConfig,
    ActorOptimizerHealthGate,
    ActorOptimizerHealthGateConfig,
    CriticQHealthConfig,
    CriticQHealthMonitor,
    resolve_actor_gate_mode,
)
from RL.utils.evaluate import evaluate_policy
from RL.utils.logger import ExperimentLogger
from RL.utils.load_predictor_artifact import load_candidate_conditioned_predictor
from RL.utils.load_predictor_contract import LoadPredictorInterface
from RL.utils.load_predictor_shadow import LoadPredictorShadow
from RL.utils.metrics import summarize_metrics
from RL.utils.offline_policy_eval import evaluate_offline_policy, write_offline_policy_evaluation
from RL.utils.operating_state import OPTIMIZATION_STATE_ENCODING_ORDER
from RL.utils.physical_load_shadow import (
    PhysicalLoadShadow, build_physical_shadow, complete_physical_step, physical_backend_selected,
    physical_condition_metadata, preflight_physical_config, submit_physical_sent_frame,
)
from RL.utils.physical_load_shadow_launch import (
    OPERATOR_TRAINING_ROUTES,
    validate_physical_launch,
    validate_physical_training_binding,
)
from RL.utils.queue_case_identity import (
    QueueCaseIdentityMismatch,
    QueueStrictCaseIdentityValidator,
    build_queue_case_identity_validator,
)
from RL.utils.queue_manifest import expand_queue_manifest
from RL.utils.safe_residual_control import SafeResidualController, safe_action_metrics
from RL.utils.set_seed import set_seed


def build_agent(config: dict, obs_dim: int, action_dim: int, device: torch.device):
    algorithm = config['algorithm'].lower()
    if algorithm == 'td3':
        return TD3Agent(obs_dim, action_dim, config['agent'], device)
    raise ValueError(f'Unsupported algorithm: {config["algorithm"]}')


def _build_queue_load_predictor_shadow(
    config: dict,
    env: WindTurbineEnv,
    device: torch.device,
) -> tuple[LoadPredictorShadow | None, dict[str, object]]:
    predictor_config = config.get('load_predictor')
    if predictor_config is None:
        return None, {}
    if physical_backend_selected(config):
        return None, {'load_predictor_shadow_model_loaded': 0.0, 'load_predictor_shadow_nn_active': 0.0}
    if not isinstance(predictor_config, dict):
        raise ValueError('load_predictor must be an object')
    if str(predictor_config.get('mode', 'shadow')).strip().lower() != 'shadow':
        raise ValueError('Queue training permits predictor shadow only; soft/hard load guards are fail-closed')
    if not env.include_operating_state_observation:
        raise ValueError('Queue load predictor requires state-aware observations')
    interface_value = predictor_config.get('interface_path')
    if not isinstance(interface_value, str) or not interface_value.strip():
        raise ValueError('load_predictor.interface_path must be a non-empty path')
    interface_path = normalize_user_path(interface_value.strip())
    interface = LoadPredictorInterface.from_path(interface_path)
    interface_sha256 = hashlib.sha256(interface_path.read_bytes()).hexdigest()
    artifact_value = predictor_config.get('predictor_artifact_path')
    artifact_sha = predictor_config.get('predictor_artifact_sha256')
    predictor_model = None
    artifact_summary: dict[str, object] = {
        'load_predictor_shadow_model_loaded': 0.0,
        'load_predictor_shadow_model_sha256': '',
        'load_predictor_shadow_interface_sha256': '',
    }
    if artifact_value is not None or artifact_sha is not None:
        if not isinstance(artifact_value, str) or not artifact_value.strip():
            raise ValueError('Configured predictor artifact requires predictor_artifact_path')
        if not isinstance(artifact_sha, str) or not artifact_sha.strip():
            raise ValueError('Configured predictor artifact requires predictor_artifact_sha256')
        dataset_manifest_path_value = predictor_config.get('dataset_manifest_path')
        if not isinstance(dataset_manifest_path_value, str) or not dataset_manifest_path_value.strip():
            raise ValueError('Configured predictor artifact requires its frozen dataset_manifest_path')
        dataset_manifest_path = normalize_user_path(dataset_manifest_path_value.strip())
        if not dataset_manifest_path.is_file():
            raise FileNotFoundError(f'Configured predictor dataset manifest does not exist: {dataset_manifest_path}')
        if hashlib.sha256(dataset_manifest_path.read_bytes()).hexdigest() != predictor_config.get(
            'dataset_manifest_sha256'
        ):
            raise ValueError('Configured predictor dataset manifest SHA does not match the frozen queue config')
        loaded = load_candidate_conditioned_predictor(
            normalize_user_path(artifact_value.strip()),
            interface_path=interface_path,
            expected_checkpoint_sha256=artifact_sha.strip(),
            action_low=env.action_scaler.low,
            action_high=env.action_scaler.high,
            device=device,
            expected_dataset_manifest_sha256=predictor_config.get('dataset_manifest_sha256'),
            expected_time_partitions_s=predictor_config.get('predictor_time_partitions_s'),
        )
        interface = loaded.interface
        predictor_model = loaded.model
        artifact_summary.update({
            'load_predictor_shadow_model_loaded': 1.0,
            'load_predictor_shadow_model_sha256': loaded.checkpoint_sha256,
            'load_predictor_shadow_interface_sha256': hashlib.sha256(interface_path.read_bytes()).hexdigest(),
        })
    shadow = LoadPredictorShadow(
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
    return shadow, artifact_summary


def _load_pid_history_diagnostic(
    history_path: str | Path,
) -> tuple[Path, np.ndarray, np.ndarray, np.ndarray | None]:
    path = normalize_user_path(str(history_path))
    if not path.is_file():
        raise FileNotFoundError(f'PID history diagnostic file does not exist: {path}')
    required_columns = ('O_PcsTorqueDemand', 'O_PitchPositionDemand1', 'I_AverageWindSpeed')
    actions: list[tuple[float, float]] = []
    winds: list[float] = []
    observation_rows: list[np.ndarray] = []
    with path.open('r', encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing_columns = [column for column in required_columns if column not in fieldnames]
        if missing_columns:
            raise ValueError(
                f'PID history diagnostic file is missing required columns {missing_columns}: {path}'
            )
        observations_available = all(column in fieldnames for column in OBSERVATION_VARIABLES)
        for line_number, row in enumerate(reader, start=2):
            try:
                action = (float(row['O_PcsTorqueDemand']), float(row['O_PitchPositionDemand1']))
                wind = float(row['I_AverageWindSpeed'])
                reference_observation = (
                    np.asarray([float(row[column]) for column in OBSERVATION_VARIABLES], dtype=np.float32)
                    if observations_available
                    else None
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(f'PID history diagnostic has a non-numeric row at {path}:{line_number}') from exc
            if not math.isfinite(action[0]) or not math.isfinite(action[1]) or not math.isfinite(wind):
                raise ValueError(f'PID history diagnostic has a non-finite row at {path}:{line_number}')
            if reference_observation is not None:
                if not np.isfinite(reference_observation).all():
                    raise ValueError(f'PID history diagnostic has non-finite observations at {path}:{line_number}')
                azimuth_index = OBSERVATION_VARIABLES.index('I_RotorAzimuth')
                reference_observation[azimuth_index] = np.mod(
                    reference_observation[azimuth_index],
                    2.0 * math.pi,
                )
                observation_rows.append(reference_observation)
            actions.append(action)
            winds.append(wind)
    if not actions:
        raise ValueError(f'PID history diagnostic file has no data rows: {path}')
    reference_observations = (
        np.asarray(observation_rows, dtype=np.float32)
        if observations_available
        else None
    )
    return path, np.asarray(actions, dtype=np.float32), np.asarray(winds, dtype=np.float32), reference_observations


def build_experiment_name(config: dict) -> str:
    base_name = str(config['experiment_name'])
    suffix = os.environ.get('RL_EXPERIMENT_SUFFIX', '').strip()
    if suffix:
        base_name = f'{base_name}_{suffix}'
    if config.get('append_timestamp', True):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = f'{base_name}_{timestamp}'
    return base_name


def _require_data_schema_version(config: dict) -> str:
    configured_version = str(config.get('data_schema_version', '')).strip()
    if configured_version != DATA_SCHEMA_VERSION:
        raise ValueError(
            f'Config data_schema_version must be {DATA_SCHEMA_VERSION!r}; got {configured_version!r}. '
            'Regenerate the SI/My-only dataset and use a current configuration.'
        )
    return configured_version


def _format_metrics(metrics: dict[str, float], keys: list[str]) -> str:
    chunks: list[str] = []
    for key in keys:
        value = metrics.get(key)
        if value is None:
            continue
        value = float(value)
        if not math.isfinite(value):
            continue
        chunks.append(f'{key}={value:.4f}')
    return ' '.join(chunks)


def _is_finite_transition(obs: np.ndarray, action: np.ndarray, reward: float, next_obs: np.ndarray) -> bool:
    return (
        np.isfinite(obs).all()
        and np.isfinite(action).all()
        and math.isfinite(float(reward))
        and np.isfinite(next_obs).all()
    )


def _should_store_transition(step_info: dict[str, float], train_cfg: dict[str, object]) -> bool:
    if step_info.get('condition_boundary', 0.0) > 0.0:
        return False
    if 'transition_store_eligible' in step_info and step_info['transition_store_eligible'] <= 0.0:
        return False
    execution_replay = execution_mode_replay_contract(train_cfg)
    if execution_replay is not None:
        if any(
            step_info.get(key, 0.0) > 0.0
            for key in (
                'control_post_step_timeout',
                'control_post_step_invalid',
                'control_post_step_unsafe_stop',
                'control_post_step_condition_boundary',
            )
        ):
            return False
        try:
            execution_mode = pre_send_execution_mode_from_code(
                step_info['control_pre_send_execution_mode']
            )
        except (KeyError, TypeError, ValueError):
            return False
        return td3_critic_replay_allowed(execution_mode)
    replay_filter = train_cfg.get('replay_filter', {})
    if not isinstance(replay_filter, dict):
        return True
    if not bool(replay_filter.get('enabled', False)):
        return True
    if bool(replay_filter.get('skip_invalid_transition', True)):
        if step_info.get('env_invalid_prediction', 0.0) > 0.0 or step_info.get('env_invalid_reward', 0.0) > 0.0:
            return False
    if bool(replay_filter.get('skip_clip_event', True)) and step_info.get('env_model_clip_event', 0.0) > 0.0:
        return False
    if bool(replay_filter.get('skip_safety_fallback', True)) and step_info.get('control_fallback_active', 0.0) > 0.0:
        return False
    if bool(replay_filter.get('skip_unsafe_transition', True)) and step_info.get('control_unsafe_stop', 0.0) > 0.0:
        return False
    if (
        bool(replay_filter.get('skip_negative_power_transition', True))
        and step_info.get('control_negative_power_steps', 0.0) > 0.0
    ):
        return False
    max_input_clip_fraction = float(replay_filter.get('max_input_clip_fraction', 1.0))
    if step_info.get('env_input_clip_fraction', 0.0) > max_input_clip_fraction:
        return False
    max_output_clip_fraction = float(replay_filter.get('max_output_clip_fraction', 1.0))
    if step_info.get('env_output_clip_fraction', 0.0) > max_output_clip_fraction:
        return False
    return True


def _queue_condition_ids(config: dict) -> tuple[int, ...]:
    queue_cfg = config.get('queue', {})
    if not isinstance(queue_cfg, dict):
        raise ValueError('queue must be an object')
    configured_ids = queue_cfg.get('condition_ids', list(range(3, 26)))
    if not isinstance(configured_ids, (list, tuple)):
        raise ValueError('queue.condition_ids must be a list of integer condition IDs')
    condition_ids = tuple(sorted({int(condition_id) for condition_id in configured_ids}))
    if not condition_ids:
        raise ValueError('queue.condition_ids must contain at least one condition ID')
    return condition_ids


def _load_queue_manifest_slots(queue_cfg: dict[str, object]):
    manifest_value = str(queue_cfg.get('manifest_path', '')).strip()
    if not manifest_value:
        return None
    manifest_path = normalize_user_path(manifest_value)
    if not manifest_path.is_absolute():
        manifest_path = PROJECT_ROOT / manifest_path
    try:
        with manifest_path.open('r', encoding='utf-8') as file:
            manifest = json.load(file)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f'Queue manifest does not exist: {manifest_path}') from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f'Unable to read queue manifest {manifest_path}: {exc}') from exc
    if not isinstance(manifest, dict):
        raise ValueError('Queue manifest must contain an object')
    try:
        slots = expand_queue_manifest(manifest)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'Invalid queue manifest {manifest_path}: {exc}') from exc
    return manifest, slots


def _validate_queue_manifest_config(config: dict, queue_cfg: dict[str, object]):
    loaded_manifest = _load_queue_manifest_slots(queue_cfg)
    if loaded_manifest is None:
        return None
    manifest, slots = loaded_manifest

    expected_execution_mode = str(queue_cfg.get('execution_mode', manifest.get('execution_mode', '')))
    if manifest.get('execution_mode') != expected_execution_mode:
        raise ValueError('queue.execution_mode must match the queue manifest execution_mode')
    expected_rounds = int(queue_cfg.get('total_condition_rounds', 0))
    if expected_rounds <= 0:
        raise ValueError('queue.total_condition_rounds must be positive')
    expected_conditions = _queue_condition_ids(config)
    if len(slots) != expected_rounds * len(expected_conditions):
        raise ValueError(
            'Queue manifest slot count must equal queue.total_condition_rounds times '
            'the number of configured condition IDs'
        )
    for round_id in range(1, expected_rounds + 1):
        round_conditions = tuple(sorted(slot.condition_id for slot in slots if slot.round_id == round_id))
        if round_conditions != expected_conditions:
            raise ValueError(f'Queue manifest round {round_id} does not match queue.condition_ids')
    if 'train_sim_s' in queue_cfg:
        expected_train_sim_s = float(queue_cfg['train_sim_s'])
        if not math.isfinite(expected_train_sim_s) or expected_train_sim_s <= 0.0:
            raise ValueError('queue.train_sim_s must be positive and finite')
        if any(not math.isclose(slot.train_sim_s, expected_train_sim_s, abs_tol=1.0e-6) for slot in slots):
            raise ValueError('Queue manifest train_sim_s must match queue.train_sim_s')
    return slots


def _queue_phase_elapsed_s(info: dict[str, float]) -> float:
    value = float(info.get('udp_phase_elapsed_s', float('nan')))
    if not math.isfinite(value) or value < 0.0:
        raise RuntimeError('Queue TRAIN duration requires finite non-negative udp_phase_elapsed_s metadata')
    return value


def _strict_queue_identity_sample(env: WindTurbineEnv) -> tuple[float, float, str | None]:
    invalid_fields = getattr(env, 'current_payload_invalid_fields', ())
    if {'Time', 'I_AverageWindSpeed'}.intersection(invalid_fields):
        raise RuntimeError('Strict queue case identity rejects raw invalid Time or I_AverageWindSpeed')
    try:
        payload = env.current_payload
    except (AttributeError, RuntimeError) as exc:
        raise RuntimeError(
            'Strict queue case identity requires the environment current payload with Time and I_AverageWindSpeed'
        ) from exc
    try:
        sim_time_s = float(payload['Time'])
        wind_speed_mps = float(payload['I_AverageWindSpeed'])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            'Strict queue case identity requires finite Time and I_AverageWindSpeed payload fields'
        ) from exc
    if not math.isfinite(sim_time_s) or not math.isfinite(wind_speed_mps):
        raise RuntimeError(
            'Strict queue case identity requires finite Time and I_AverageWindSpeed payload fields'
        )
    raw_fingerprint = getattr(env, 'current_case_fingerprint', None)
    runtime_case_fingerprint = None if raw_fingerprint is None else str(raw_fingerprint).strip()
    return sim_time_s, wind_speed_mps, runtime_case_fingerprint or None


def _validate_strict_manifest_identity(slot, metadata: dict, *, context: str) -> None:
    expected = (slot.condition_id, slot.condition_instance_id, slot.queue_epoch)
    observed = tuple(metadata.get(key) for key in (
        'udp_condition_id', 'udp_condition_instance_id', 'udp_queue_epoch'))
    if observed != expected:
        raise RuntimeError(
            f'Queue manifest slot mismatch during {context}: expected={expected} observed={observed}'
        )


def _finish_physical_operator_identity(validator, env, slot, observation):
    # Use the pending packet's parse provenance, not condition_end (also true on reload).
    metadata = env.current_queue_metadata
    _validate_strict_manifest_identity(slot, metadata, context='physical END')
    if metadata.get('udp_phase_code') != 3.0 or metadata.get('udp_queue_protocol_v2') != 1.0:
        raise RuntimeError('Physical operator completion requires a genuine same-slot END packet')
    sim_time_s, wind_speed_mps, fingerprint = _strict_queue_identity_sample(env)
    entry = validator.reference.entry(slot.condition_id)
    endpoint = float(entry.timestamps_s[-1]) - validator.fixed_simulation_time_offset_s
    expected_count = len(entry.timestamps_s)
    time_tolerance = validator.time_absolute_tolerance_s
    end_offset = sim_time_s - endpoint
    if validator._sample_count < expected_count:
        # END may supply the final reference sample only when the packet itself is
        # exactly that endpoint. A later END marker is not a substitute for it.
        if not math.isclose(sim_time_s, endpoint, rel_tol=0.0, abs_tol=time_tolerance):
            raise RuntimeError(
                'Physical operator END cannot fill the missing trace endpoint: '
                f'end_time={sim_time_s:.6f} expected_endpoint={endpoint:.6f} '
                f'observed_trace_samples={validator._sample_count}/{expected_count}'
            )
        validator.observe(slot.condition_id, slot.condition_instance_id, observation,
                          sim_time_s=sim_time_s, wind_speed_mps=wind_speed_mps,
                          runtime_case_fingerprint=fingerprint)
        end_offset = 0.0
    elif validator._sample_count == expected_count:
        # Bladed may emit the queue END delimiter one output tick after the
        # requested interval. The final TRAIN trace sample must already be the
        # exact PID endpoint; accept only an END marker within one reference step
        # and do not add that marker to the PID trace/hash.
        last_trace_time = validator._last_simulation_time_s
        if last_trace_time is None or not math.isclose(
                last_trace_time + validator.fixed_simulation_time_offset_s,
                float(entry.timestamps_s[-1]), rel_tol=0.0, abs_tol=time_tolerance):
            raise RuntimeError(
                'Physical operator TRAIN trace did not reach its exact endpoint: '
                f'last_trace_time={last_trace_time!r} expected_endpoint={endpoint:.6f} '
                f'observed_trace_samples={validator._sample_count}/{expected_count}'
            )
        if expected_count < 2:
            raise RuntimeError('Physical operator PID reference needs at least two timestamps')
        trace_step = float(entry.timestamps_s[-1] - entry.timestamps_s[-2])
        if not math.isfinite(trace_step) or trace_step <= 0.0:
            raise RuntimeError('Physical operator PID reference has an invalid final frame interval')
        if end_offset < -time_tolerance or end_offset > trace_step + time_tolerance:
            raise RuntimeError(
                'Physical operator END marker is outside the endpoint-to-next-frame interval: '
                f'end_time={sim_time_s:.6f} expected_endpoint={endpoint:.6f} '
                f'end_offset={end_offset:.6f} max_offset={trace_step:.6f} '
                f'last_trace_time={last_trace_time:.6f}'
            )
        if abs(end_offset) <= time_tolerance:
            if wind_speed_mps != validator._last_observed_wind:
                raise RuntimeError('Physical operator END disagrees with the repeated trace endpoint')
            end_offset = 0.0
    else:
        raise RuntimeError(
            'Physical operator observed more TRAIN trace samples than the PID reference: '
            f'{validator._sample_count}>{expected_count}'
        )
    metrics, summary = validator.finish_condition(
        slot.condition_id, slot.condition_instance_id,
        runtime_case_fingerprint=fingerprint, end_received=True,
    )
    metrics.update({
        'queue_case_identity_end_marker_time_s': sim_time_s,
        'queue_case_identity_end_marker_offset_from_endpoint_s': end_offset,
    })
    summary.update({
        'end_marker_time_s': sim_time_s,
        'end_marker_offset_from_reference_endpoint_s': end_offset,
        'end_marker_is_reference_sample': float(abs(end_offset) <= time_tolerance),
    })
    return metrics, summary


def _step_environment(
    env,
    action_norm: np.ndarray,
    action_real: np.ndarray,
    on_action_sent=None,
):
    if bool(getattr(env, 'supports_physical_action', False)):
        return env.step(
            action_norm,
            on_action_sent=on_action_sent,
            action_real=action_real,
        )
    return env.step(action_norm, on_action_sent=on_action_sent)


def _validate_queue_train_duration(
    queue_cfg: dict[str, object],
    *,
    condition_id: int,
    condition_instance_id: int,
    phase_elapsed_start_s: float,
    phase_elapsed_end_s: float,
) -> float:
    elapsed_s = phase_elapsed_end_s - phase_elapsed_start_s
    if not math.isfinite(elapsed_s) or elapsed_s < 0.0:
        raise RuntimeError(
            f'Queue TRAIN phase elapsed time regressed for condition={condition_id} '
            f'instance={condition_instance_id}'
        )
    target_value = queue_cfg.get('train_sim_s')
    if target_value is None:
        return elapsed_s
    target_s = float(target_value)
    tolerance_s = float(queue_cfg.get('train_sim_tolerance_s', 0.1))
    if not math.isfinite(target_s) or target_s <= 0.0:
        raise ValueError('queue.train_sim_s must be positive and finite')
    if not math.isfinite(tolerance_s) or tolerance_s < 0.0:
        raise ValueError('queue.train_sim_tolerance_s must be finite and non-negative')
    if elapsed_s + tolerance_s < target_s:
        raise RuntimeError(
            f'Queue TRAIN duration is too short for condition={condition_id} '
            f'instance={condition_instance_id}: {elapsed_s:.6f}s < {target_s:.6f}s '
            f'(tolerance={tolerance_s:.6f}s)'
        )
    return elapsed_s


def _queue_condition_checkpoint_count(queue_cfg: dict[str, object]) -> int:
    raw_count = queue_cfg.get('condition_checkpoint_count', 0)
    if isinstance(raw_count, bool) or not isinstance(raw_count, int) or raw_count < 0:
        raise ValueError('queue.condition_checkpoint_count must be a non-negative integer')
    return raw_count


def _queue_condition_checkpoint_name(
    *,
    round_index: int,
    condition_id: int,
    condition_instance_id: int,
    checkpoint_index: int,
    checkpoint_count: int,
) -> str:
    return (
        f'queue_round_{round_index:02d}_condition_{condition_id:02d}_'
        f'instance_{condition_instance_id:03d}_checkpoint_{checkpoint_index:02d}_'
        f'of_{checkpoint_count:02d}.pt'
    )


def _build_online_replay_buffer(env: WindTurbineEnv, config: dict):
    train_cfg = config['train']
    if not env.queue_lifecycle_enabled:
        return ReplayBuffer(env.obs_dim, env.action_dim, int(train_cfg['replay_size']))

    condition_replay_cfg = train_cfg.get('condition_replay', {})
    if not isinstance(condition_replay_cfg, dict):
        raise ValueError('train.condition_replay must be an object for v7 queue training')
    sampling = str(condition_replay_cfg.get('sampling', 'condition_stratified')).lower()
    if sampling != 'condition_stratified':
        raise ValueError('v7 queue training requires train.condition_replay.sampling=condition_stratified')
    if 'capacity_per_condition' not in condition_replay_cfg:
        raise ValueError('v7 queue training requires train.condition_replay.capacity_per_condition')
    capacity_per_condition = int(condition_replay_cfg['capacity_per_condition'])
    min_valid_transitions = int(condition_replay_cfg.get('min_valid_transitions_per_condition', 20_000))
    if capacity_per_condition < min_valid_transitions:
        raise ValueError(
            'train.condition_replay.capacity_per_condition must be at least '
            'min_valid_transitions_per_condition'
        )
    return ConditionStratifiedReplayBuffer(
        env.obs_dim,
        env.action_dim,
        _queue_condition_ids(config),
        capacity_per_condition=capacity_per_condition,
        min_per_condition_in_batch=int(condition_replay_cfg.get('min_per_condition_in_batch', 5)),
        source_max_fraction=condition_replay_cfg.get('source_max_fraction'),
    )


def _queue_context_from_info(
    info: dict[str, float],
    *,
    expected_condition_ids: tuple[int, ...] | None = None,
) -> tuple[int, int, bool]:
    condition_id = int(info.get('udp_condition_id', -1.0))
    condition_instance_id = int(info.get('udp_condition_instance_id', -1.0))
    if condition_id < 0 or condition_instance_id < 0:
        raise RuntimeError('Queue training requires condition ID and condition instance metadata on every packet')
    if expected_condition_ids is not None and condition_id not in expected_condition_ids:
        raise RuntimeError(
            f'Queue condition mismatch: received condition={condition_id} '
            f'instance={condition_instance_id}, expected one of {list(expected_condition_ids)}. '
            'The Bladed DLL manual Batch Run profile or GUI case order does not match this training configuration.'
        )
    return condition_id, condition_instance_id, bool(info.get('condition_train_active', 0.0) > 0.0)


def _queue_stage_code(stage: str) -> float:
    return {'collect': 1.0, 'critic': 2.0, 'actor': 3.0}[stage]


def _queue_condition_stats() -> dict[str, float]:
    metrics = {
        'queue_condition_raw_frame_count': 0.0,
        'queue_condition_train_frame_count': 0.0,
        'queue_condition_eligible_transition_count': 0.0,
        'queue_condition_stored_transition_count': 0.0,
        'queue_condition_rejected_transition_count': 0.0,
        'queue_condition_rejected_boundary_count': 0.0,
        'queue_condition_rejected_non_train_count': 0.0,
        'queue_condition_rejected_fallback_count': 0.0,
        'queue_condition_rejected_unsafe_count': 0.0,
        'queue_condition_rejected_negative_power_count': 0.0,
        'queue_condition_rejected_invalid_count': 0.0,
        'queue_condition_rejected_other_count': 0.0,
        'queue_condition_batch_count': 0.0,
        'queue_condition_batch_sample_count': 0.0,
        'queue_condition_batch_quota_relaxed_count': 0.0,
        'queue_condition_execution_mode_transition_count': 0.0,
        'queue_condition_low_speed_stage_transition_count': 0.0,
        'queue_condition_actor_optimizer_gate_transition_count': 0.0,
        'queue_condition_last_execution_mode_code': -1.0,
        'queue_condition_last_low_speed_stage': -1.0,
        'queue_condition_last_actor_optimizer_gate_open': -1.0,
        'queue_condition_action_projection_count': 0.0,
        'queue_condition_action_rate_limited_count': 0.0,
        'queue_condition_ood_score_count': 0.0,
        'queue_condition_ood_score_sum': 0.0,
        'queue_condition_ood_score_min': 0.0,
        'queue_condition_ood_score_max': 0.0,
        'queue_condition_rl_pdt_diagnostic_energy_j': 0.0,
        'queue_condition_pid_pdt_diagnostic_energy_j': 0.0,
        'queue_condition_rl_trapezoid_energy_j': 0.0,
        'queue_condition_pid_trapezoid_energy_j': 0.0,
        'queue_condition_rl_trapezoid_valid_segment_count': 0.0,
        'queue_condition_pid_trapezoid_valid_segment_count': 0.0,
        'queue_condition_trapezoid_invalid_segment_count': 0.0,
        'queue_condition_trapezoid_missing_pid_reference_count': 0.0,
        'queue_condition_trapezoid_inputs_complete': 0.0,
        'queue_condition_trapezoid_energy_valid': 0.0,
        'queue_condition_rl_minus_pid_trapezoid_energy_j': 0.0,
        'queue_condition_trapezoid_delta_energy_percent': 0.0,
        'queue_condition_strict_per_blade_pass': 0.0,
        'queue_condition_load_predictor_observed_frame_count': 0.0,
        'queue_condition_load_predictor_pending_sample_count': 0.0,
        'queue_condition_load_predictor_completed_record_count': 0.0,
        'queue_condition_load_predictor_audit_record_count': 0.0,
        'queue_condition_load_predictor_audit_rows_written': 0.0,
        'queue_condition_load_predictor_calibration_eligible_record_count': 0.0,
        'queue_condition_load_predictor_dropped_pending_count': 0.0,
        'queue_condition_load_predictor_dropped_prediction_count': 0.0,
        'queue_condition_load_predictor_discontinuity_count': 0.0,
        'queue_condition_load_predictor_invalid_prediction_count': 0.0,
        'queue_condition_critic_update_attempt_count': 0.0,
        'queue_condition_critic_update_executed_count': 0.0,
        'queue_condition_critic_update_skipped_q_guard_count': 0.0,
        'queue_condition_critic_update_skipped_nonfinite_batch_count': 0.0,
        'queue_condition_critic_update_skipped_nonfinite_loss_count': 0.0,
        'queue_condition_critic_update_skipped_nonfinite_gradient_count': 0.0,
        'queue_condition_critic_update_skipped_nonfinite_parameter_count': 0.0,
        'queue_condition_critic_update_skipped_other_count': 0.0,
        'queue_condition_actor_update_requested_after_critic_count': 0.0,
        'queue_condition_actor_update_slot_due_count': 0.0,
        'queue_condition_actor_update_attempt_count': 0.0,
        'queue_condition_actor_update_blocked_critic_failure_count': 0.0,
        'queue_condition_actor_update_deferred_on_slot_count': 0.0,
        'queue_condition_actor_update_executed_count': 0.0,
        'queue_condition_actor_update_deferred_count': 0.0,
        'queue_condition_actor_update_blocked_q_health_count': 0.0,
        'queue_condition_actor_update_blocked_control_health_count': 0.0,
        'queue_condition_actor_update_skipped_q_guard_count': 0.0,
        'queue_condition_actor_update_skipped_nonfinite_count': 0.0,
        'queue_condition_actor_update_skipped_nonfinite_gradient_count': 0.0,
        'queue_condition_actor_update_skipped_nonfinite_parameter_count': 0.0,
        'queue_condition_actor_update_skipped_other_count': 0.0,
    }
    for source in REPLAY_SOURCE_NAMES:
        metrics[f'queue_condition_batch_source_{source}_sample_count'] = 0.0
        metrics[f'queue_condition_batch_source_{source}_quota_relaxed_count'] = 0.0
        metrics[f'queue_condition_stored_source_{source}_count'] = 0.0
        metrics[f'queue_condition_rejected_source_{source}_count'] = 0.0
    for mode in PreSendExecutionMode:
        metrics[f'queue_condition_execution_mode_{mode.value}_count'] = 0.0
        metrics[f'queue_condition_execution_mode_{mode.value}_duration_run_count'] = 0.0
        metrics[f'queue_condition_execution_mode_{mode.value}_duration_p50_s'] = 0.0
        metrics[f'queue_condition_execution_mode_{mode.value}_duration_p95_s'] = 0.0
        metrics[f'queue_condition_execution_mode_{mode.value}_duration_max_s'] = 0.0
        metrics[f'queue_condition_execution_mode_{mode.value}_duration_sample_count'] = 0.0
    for stage in range(5):
        metrics[f'queue_condition_low_speed_stage_{stage}_count'] = 0.0
    for blade_index in range(1, 4):
        metrics[f'queue_condition_blade_{blade_index}_peak_abs_nm'] = 0.0
        metrics[f'queue_condition_blade_{blade_index}_peak_time_s'] = -1.0
        metrics[f'queue_condition_blade_{blade_index}_pid_peak_reference_nm'] = 0.0
        metrics[f'queue_condition_blade_{blade_index}_peak_ratio_to_pid'] = 0.0
        metrics[f'queue_condition_blade_{blade_index}_delta_load_percent'] = 0.0
        metrics[f'queue_condition_blade_{blade_index}_first_pid_exceed_time_s'] = -1.0
        metrics[f'queue_condition_blade_{blade_index}_pid_exceed_seen'] = 0.0
    for feature_name in OBSERVATION_VARIABLES:
        metrics[f'queue_condition_ood_feature_{feature_name}_count'] = 0.0
        metrics[f'queue_condition_ood_feature_{feature_name}_max_score'] = 0.0
        metrics[f'queue_condition_ood_feature_{feature_name}_max_abs_z'] = 0.0
    metrics['queue_condition_execution_mode_current_run_duration_s'] = 0.0
    for mode_code, mode in enumerate(PreSendExecutionMode, start=1):
        metrics[f'_queue_condition_mode_{mode_code}_run_count'] = 0.0
        metrics[f'_queue_condition_mode_{mode_code}_current_run_duration_s'] = 0.0
        metrics[f'_queue_condition_mode_{mode_code}_max_run_duration_s'] = 0.0
        metrics[f'_queue_condition_mode_{mode_code}_duration_sample_count'] = 0.0
        for sample_index in range(128):
            metrics[f'_queue_condition_mode_{mode_code}_duration_sample_{sample_index:03d}'] = 0.0
    return metrics


def _close_queue_condition_mode_run(
    stats: dict[str, float],
    *,
    condition_id: int,
    mode_code: int,
) -> None:
    if mode_code < 1 or mode_code > len(PreSendExecutionMode):
        return
    duration = stats[f'_queue_condition_mode_{mode_code}_current_run_duration_s']
    if duration <= 0.0:
        return
    prefix = f'_queue_condition_mode_{mode_code}_'
    run_count = int(stats[prefix + 'run_count']) + 1
    stats[prefix + 'run_count'] = float(run_count)
    stats[prefix + 'max_run_duration_s'] = max(stats[prefix + 'max_run_duration_s'], duration)
    sample_count = int(stats[prefix + 'duration_sample_count'])
    if sample_count < 128:
        stats[f'{prefix}duration_sample_{sample_count:03d}'] = duration
        stats[prefix + 'duration_sample_count'] = float(sample_count + 1)
    else:
        seed = int(condition_id) * 1_000_003 + mode_code * 104_729 + run_count
        replacement_index = random.Random(seed).randrange(run_count)
        if replacement_index < 128:
            stats[f'{prefix}duration_sample_{replacement_index:03d}'] = duration
    stats[prefix + 'current_run_duration_s'] = 0.0


def _record_queue_condition_update_stats(
    stats: dict[str, float],
    update_metrics: dict[str, float],
) -> None:
    stats['queue_condition_critic_update_attempt_count'] += 1.0
    critic_executed = 'critic1_loss' in update_metrics and 'critic2_loss' in update_metrics
    stats['queue_condition_critic_update_executed_count'] += float(critic_executed)
    critic_skip_keys = (
        'update_skipped_q_guard',
        'update_skipped_nonfinite_batch',
        'update_skipped_nonfinite_loss',
        'update_skipped_nonfinite_gradient',
        'update_skipped_nonfinite_parameter',
    )
    if not critic_executed:
        skip_reason_count = 0
        for key in critic_skip_keys:
            if update_metrics.get(key, 0.0) > 0.0:
                stats[f'queue_condition_critic_{key}_count'] += 1.0
                skip_reason_count += 1
        if skip_reason_count == 0:
            stats['queue_condition_critic_update_skipped_other_count'] += 1.0
        elif skip_reason_count > 1:
            raise RuntimeError('A single Critic update was attributed to multiple skip reasons')
    critic_accounted = stats['queue_condition_critic_update_executed_count'] + sum(
        stats[f'queue_condition_critic_{key}_count']
        for key in critic_skip_keys
    ) + stats['queue_condition_critic_update_skipped_other_count']
    if critic_accounted != stats['queue_condition_critic_update_attempt_count']:
        raise RuntimeError('Critic update attempt/executed/skipped reason counts do not close')

    if critic_executed:
        stats['queue_condition_actor_update_requested_after_critic_count'] += float(
            update_metrics.get('actor_update_requested_after_critic', 0.0) > 0.0
        )
    slot_due = bool(update_metrics.get('actor_update_slot_due', 0.0) > 0.0)
    actor_requested = bool(update_metrics.get('actor_update_requested_after_critic', 0.0) > 0.0)
    stats['queue_condition_actor_update_slot_due_count'] += float(slot_due)
    actor_attempted = bool(update_metrics.get('actor_update_attempted', 0.0) > 0.0)
    stats['queue_condition_actor_update_attempt_count'] += float(actor_attempted)
    stats['queue_condition_actor_update_blocked_critic_failure_count'] += float(
        slot_due and not critic_executed
    )
    stats['queue_condition_actor_update_deferred_on_slot_count'] += float(
        slot_due and update_metrics.get('actor_update_deferred', 0.0) > 0.0
    )
    stats['queue_condition_actor_update_executed_count'] += float('actor_loss' in update_metrics)
    stats['queue_condition_actor_update_deferred_count'] += float(
        update_metrics.get('actor_update_deferred', 0.0) > 0.0
    )
    stats['queue_condition_actor_update_blocked_q_health_count'] += float(
        update_metrics.get('actor_update_blocked_q_health', 0.0) > 0.0
    )
    stats['queue_condition_actor_update_blocked_control_health_count'] += float(
        update_metrics.get('actor_update_blocked_control_health', 0.0) > 0.0
    )
    actor_skip_map = {
        'actor_update_skipped_q_guard': 'queue_condition_actor_update_skipped_q_guard_count',
        'actor_update_skipped_nonfinite': 'queue_condition_actor_update_skipped_nonfinite_count',
        'actor_update_skipped_nonfinite_gradient': 'queue_condition_actor_update_skipped_nonfinite_gradient_count',
        'actor_update_skipped_nonfinite_parameter': 'queue_condition_actor_update_skipped_nonfinite_parameter_count',
    }
    actor_skip_reason_count = 0
    for key, metric_name in actor_skip_map.items():
        if update_metrics.get(key, 0.0) > 0.0:
            stats[metric_name] += 1.0
            actor_skip_reason_count += 1
    if actor_skip_reason_count > 1:
        raise RuntimeError('A single Actor optimizer slot was attributed to multiple skip reasons')
    if (
        actor_attempted
        and 'actor_loss' not in update_metrics
        and update_metrics.get('actor_update_deferred', 0.0) <= 0.0
        and actor_skip_reason_count == 0
    ):
        stats['queue_condition_actor_update_skipped_other_count'] += 1.0
    actor_accounted = (
        stats['queue_condition_actor_update_executed_count']
        + stats['queue_condition_actor_update_skipped_q_guard_count']
        + stats['queue_condition_actor_update_skipped_nonfinite_count']
        + stats['queue_condition_actor_update_skipped_nonfinite_gradient_count']
        + stats['queue_condition_actor_update_skipped_nonfinite_parameter_count']
        + stats['queue_condition_actor_update_skipped_other_count']
    )
    if actor_accounted != stats['queue_condition_actor_update_attempt_count']:
        raise RuntimeError('Actor update attempt/executed/skipped reason counts do not close')
    actor_slot_accounted = (
        stats['queue_condition_actor_update_attempt_count']
        + stats['queue_condition_actor_update_blocked_critic_failure_count']
        + stats['queue_condition_actor_update_deferred_on_slot_count']
    )
    if actor_slot_accounted != stats['queue_condition_actor_update_slot_due_count']:
        raise RuntimeError('Actor scheduled-slot/attempt/critic-block/defer counts do not close')


def _pid_power_reference_at(env, condition_id: int, sim_time_s: float) -> float | None:
    reward_fn = getattr(env, 'reward_fn', None)
    reference_power = getattr(reward_fn, 'reference_power', None)
    if not callable(reference_power):
        return None
    try:
        value = reference_power(int(condition_id), float(sim_time_s))
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def _record_queue_condition_step_stats(
    stats: dict[str, float],
    step_info: dict[str, float],
    env,
    *,
    condition_id: int,
    transition_eligible: bool,
) -> None:
    try:
        sim_time_t = float(step_info.get('udp_sim_time', math.nan))
        sim_time_next_for_duration = float(step_info.get('udp_next_sim_time', math.nan))
        duration_delta_s = sim_time_next_for_duration - sim_time_t
    except (TypeError, ValueError):
        duration_delta_s = math.nan
    if not math.isfinite(duration_delta_s) or duration_delta_s <= 0.0:
        try:
            duration_delta_s = float(step_info.get('reward_energy_delta_t_s', 0.0))
        except (TypeError, ValueError):
            duration_delta_s = 0.0
    if not math.isfinite(duration_delta_s) or duration_delta_s <= 0.0:
        duration_delta_s = 0.02

    mode_code = float(step_info.get('control_pre_send_execution_mode', -1.0))
    if math.isfinite(mode_code) and mode_code >= 1.0:
        mode = pre_send_execution_mode_from_code(mode_code)
        stats[f'queue_condition_execution_mode_{mode.value}_count'] += 1.0
        previous_mode = stats['queue_condition_last_execution_mode_code']
        if previous_mode >= 0.0 and previous_mode != mode_code:
            stats['queue_condition_execution_mode_transition_count'] += 1.0
            _close_queue_condition_mode_run(
                stats,
                condition_id=condition_id,
                mode_code=int(previous_mode),
            )
        stats['queue_condition_last_execution_mode_code'] = mode_code
        stats['queue_condition_execution_mode_current_run_duration_s'] += duration_delta_s
        stats[f'_queue_condition_mode_{int(mode_code)}_current_run_duration_s'] += duration_delta_s

    low_speed_stage = float(step_info.get('control_low_speed_stage', -1.0))
    if math.isfinite(low_speed_stage) and low_speed_stage in range(5):
        stats[f'queue_condition_low_speed_stage_{int(low_speed_stage)}_count'] += 1.0
        previous_stage = stats['queue_condition_last_low_speed_stage']
        if previous_stage >= 0.0 and previous_stage != low_speed_stage:
            stats['queue_condition_low_speed_stage_transition_count'] += 1.0
        stats['queue_condition_last_low_speed_stage'] = low_speed_stage

    optimizer_gate_open = float(step_info.get('control_actor_update_gate_open', -1.0))
    if math.isfinite(optimizer_gate_open) and optimizer_gate_open in (0.0, 1.0):
        previous_gate = stats['queue_condition_last_actor_optimizer_gate_open']
        if previous_gate >= 0.0 and previous_gate != optimizer_gate_open:
            stats['queue_condition_actor_optimizer_gate_transition_count'] += 1.0
        stats['queue_condition_last_actor_optimizer_gate_open'] = optimizer_gate_open

    stats['queue_condition_action_projection_count'] += float(
        step_info.get('control_action_projection_applied', 0.0) > 0.0
    )
    stats['queue_condition_action_rate_limited_count'] += float(
        step_info.get('control_action_rate_limited', 0.0) > 0.0
    )
    ood_score = float(step_info.get('control_ood_score', math.nan))
    if math.isfinite(ood_score):
        count = stats['queue_condition_ood_score_count']
        stats['queue_condition_ood_score_count'] = count + 1.0
        stats['queue_condition_ood_score_sum'] += ood_score
        stats['queue_condition_ood_score_min'] = ood_score if count == 0.0 else min(
            stats['queue_condition_ood_score_min'], ood_score
        )
        stats['queue_condition_ood_score_max'] = max(stats['queue_condition_ood_score_max'], ood_score)
    try:
        feature_index = int(float(step_info.get('control_ood_feature_index', -1.0)))
        feature_value = float(step_info.get('control_ood_feature_value', math.nan))
        feature_median = float(step_info.get('control_ood_feature_median', math.nan))
        feature_iqr = float(step_info.get('control_ood_feature_iqr', math.nan))
    except (TypeError, ValueError, OverflowError):
        feature_index = -1
        feature_value = feature_median = feature_iqr = math.nan
    if (
        0 <= feature_index < len(OBSERVATION_VARIABLES)
        and all(math.isfinite(value) for value in (feature_value, feature_median, feature_iqr))
    ):
        feature_name = OBSERVATION_VARIABLES[feature_index]
        prefix = f'queue_condition_ood_feature_{feature_name}_'
        stats[prefix + 'count'] += 1.0
        stats[prefix + 'max_score'] = max(stats[prefix + 'max_score'], ood_score if math.isfinite(ood_score) else 0.0)
        stats[prefix + 'max_abs_z'] = max(
            stats[prefix + 'max_abs_z'],
            abs(feature_value - feature_median) / max(abs(feature_iqr), 1.0e-8),
        )

    try:
        sim_time_next = float(step_info.get('udp_next_sim_time', math.nan))
    except (TypeError, ValueError):
        sim_time_next = math.nan
    peak_timestamp = sim_time_next if math.isfinite(sim_time_next) else -1.0
    for blade_index in range(1, 4):
        value = step_info.get(f'reward_blade_root_{blade_index}_raw')
        try:
            blade_abs = abs(float(value))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(blade_abs):
            continue
        peak_key = f'queue_condition_blade_{blade_index}_peak_abs_nm'
        if blade_abs > stats[peak_key]:
            stats[peak_key] = blade_abs
            stats[f'queue_condition_blade_{blade_index}_peak_time_s'] = peak_timestamp
        reference_key = f'reward_blade_root_{blade_index}_peak_reference_nm'
        try:
            reference = float(step_info.get(reference_key, 0.0))
        except (TypeError, ValueError):
            reference = 0.0
        if math.isfinite(reference) and reference > 0.0:
            stats[f'queue_condition_blade_{blade_index}_pid_peak_reference_nm'] = reference
            exceed_key = f'queue_condition_blade_{blade_index}_pid_exceed_seen'
            if stats[exceed_key] == 0.0 and blade_abs > reference:
                stats[exceed_key] = 1.0
                stats[f'queue_condition_blade_{blade_index}_first_pid_exceed_time_s'] = peak_timestamp

    if not transition_eligible:
        return
    try:
        sim_time_t = float(step_info.get('udp_sim_time', math.nan))
        sim_time_next = float(step_info.get('udp_next_sim_time', math.nan))
        power_t = float(step_info.get('power_t_metric_raw', step_info.get('reward_power_t_raw', math.nan)))
        power_next = float(
            step_info.get(
                'power_metric_raw',
                step_info.get('reward_power_metric_raw', step_info.get('reward_power_raw', math.nan)),
            )
        )
    except (TypeError, ValueError):
        stats['queue_condition_trapezoid_invalid_segment_count'] += 1.0
        return
    delta_s = sim_time_next - sim_time_t
    if not all(math.isfinite(value) for value in (sim_time_t, sim_time_next, delta_s, power_t, power_next)) or delta_s <= 0.0:
        stats['queue_condition_trapezoid_invalid_segment_count'] += 1.0
        return

    stats['queue_condition_rl_trapezoid_energy_j'] += 0.5 * (power_t + power_next) * delta_s
    stats['queue_condition_rl_trapezoid_valid_segment_count'] += 1.0
    stats['queue_condition_rl_pdt_diagnostic_energy_j'] += power_next * delta_s
    try:
        pid_power_next = float(step_info.get('reward_pid_power_reference_raw', math.nan))
    except (TypeError, ValueError):
        pid_power_next = math.nan
    pid_power_t = _pid_power_reference_at(env, condition_id, sim_time_t)
    if pid_power_t is None or not math.isfinite(pid_power_next):
        stats['queue_condition_trapezoid_missing_pid_reference_count'] += 1.0
        return
    stats['queue_condition_pid_trapezoid_energy_j'] += 0.5 * (pid_power_t + pid_power_next) * delta_s
    stats['queue_condition_pid_trapezoid_valid_segment_count'] += 1.0
    stats['queue_condition_pid_pdt_diagnostic_energy_j'] += pid_power_next * delta_s


def _finalize_queue_condition_step_stats(stats: dict[str, float], *, condition_id: int) -> None:
    last_mode_code = int(stats['queue_condition_last_execution_mode_code'])
    if last_mode_code >= 1:
        _close_queue_condition_mode_run(
            stats,
            condition_id=int(condition_id),
            mode_code=last_mode_code,
        )
    for mode_code, mode in enumerate(PreSendExecutionMode, start=1):
        prefix = f'_queue_condition_mode_{mode_code}_'
        sample_count = int(stats[prefix + 'duration_sample_count'])
        samples = sorted(
            stats[f'{prefix}duration_sample_{sample_index:03d}']
            for sample_index in range(sample_count)
        )
        if samples:
            p50_index = min(len(samples) - 1, int(math.floor(0.50 * (len(samples) - 1))))
            p95_index = min(len(samples) - 1, int(math.ceil(0.95 * (len(samples) - 1))))
            stats[f'queue_condition_execution_mode_{mode.value}_duration_p50_s'] = samples[p50_index]
            stats[f'queue_condition_execution_mode_{mode.value}_duration_p95_s'] = samples[p95_index]
        stats[f'queue_condition_execution_mode_{mode.value}_duration_run_count'] = stats[prefix + 'run_count']
        stats[f'queue_condition_execution_mode_{mode.value}_duration_max_s'] = stats[prefix + 'max_run_duration_s']
        stats[f'queue_condition_execution_mode_{mode.value}_duration_sample_count'] = stats[prefix + 'duration_sample_count']
        for key in tuple(stats):
            if key.startswith(prefix):
                del stats[key]
    stats.pop('queue_condition_execution_mode_current_run_duration_s', None)
    expected_segments = stats['queue_condition_eligible_transition_count']
    stats['queue_condition_trapezoid_inputs_complete'] = float(
        expected_segments > 0.0
        and stats['queue_condition_rl_trapezoid_valid_segment_count'] == expected_segments
        and stats['queue_condition_pid_trapezoid_valid_segment_count'] == expected_segments
    )
    pid_energy_j = stats['queue_condition_pid_trapezoid_energy_j']
    if (
        stats['queue_condition_trapezoid_inputs_complete'] > 0.5
        and math.isfinite(pid_energy_j)
        and pid_energy_j > 0.0
    ):
        energy_delta_j = stats['queue_condition_rl_trapezoid_energy_j'] - pid_energy_j
        stats['queue_condition_rl_minus_pid_trapezoid_energy_j'] = energy_delta_j
        stats['queue_condition_trapezoid_delta_energy_percent'] = energy_delta_j / pid_energy_j * 100.0
        stats['queue_condition_trapezoid_energy_valid'] = 1.0
    strict_blade_pass = stats['queue_condition_trapezoid_inputs_complete'] > 0.5
    for blade_index in range(1, 4):
        peak = stats[f'queue_condition_blade_{blade_index}_peak_abs_nm']
        reference = stats[f'queue_condition_blade_{blade_index}_pid_peak_reference_nm']
        if peak > 0.0 and reference > 0.0:
            ratio = peak / reference
            stats[f'queue_condition_blade_{blade_index}_peak_ratio_to_pid'] = ratio
            stats[f'queue_condition_blade_{blade_index}_delta_load_percent'] = (ratio - 1.0) * 100.0
            strict_blade_pass = strict_blade_pass and ratio <= 1.0
        else:
            strict_blade_pass = False
    stats['queue_condition_strict_per_blade_pass'] = float(strict_blade_pass)


def _new_round_batch_source_stats(
    condition_ids: tuple[int, ...],
    source_names: tuple[str, ...],
) -> dict[str, object]:
    return {
        'batch_count': 0,
        'sample_count': 0,
        'source_counts': {source: 0 for source in source_names},
        'quota_relaxed_count': 0,
        'quota_relaxed_by_source': {source: 0 for source in source_names},
        'condition_quota_relaxed_count': {condition_id: 0 for condition_id in condition_ids},
        'condition_quota_relaxed_by_source': {
            condition_id: {source: 0 for source in source_names}
            for condition_id in condition_ids
        },
        'condition_sample_counts': {
            condition_id: {source: 0 for source in source_names}
            for condition_id in condition_ids
        },
        'condition_batch_counts': {condition_id: 0 for condition_id in condition_ids},
    }


def _round_batch_source_stats(
    stats_by_round: dict[int, dict[str, object]],
    round_index: int,
    condition_ids: tuple[int, ...],
    source_names: tuple[str, ...],
) -> dict[str, object]:
    stats = stats_by_round.get(int(round_index))
    if stats is None:
        stats = _new_round_batch_source_stats(condition_ids, source_names)
        stats_by_round[int(round_index)] = stats
    return stats


def _batch_source_metrics_for_condition(
    stats_by_round: dict[int, dict[str, object]],
    round_index: int,
    condition_id: int,
    source_names: tuple[str, ...],
) -> dict[str, float]:
    round_stats = stats_by_round.get(int(round_index))
    if round_stats is None:
        return {
            **{f'queue_condition_batch_source_{source}_sample_count': 0.0 for source in source_names},
            'queue_condition_batch_count': 0.0,
            'queue_condition_batch_sample_count': 0.0,
        }
    source_counts = round_stats['condition_sample_counts'][int(condition_id)]
    relaxed_by_source = round_stats['condition_quota_relaxed_by_source'][int(condition_id)]
    return {
        **{
            f'queue_condition_batch_source_{source}_sample_count': float(source_counts[source])
            for source in source_names
        },
        **{
            f'queue_condition_batch_source_{source}_quota_relaxed_count': float(relaxed_by_source[source])
            for source in source_names
        },
        'queue_condition_batch_count': float(round_stats['condition_batch_counts'][int(condition_id)]),
        'queue_condition_batch_sample_count': float(sum(source_counts.values())),
        'queue_condition_batch_quota_relaxed_count': float(
            round_stats['condition_quota_relaxed_count'][int(condition_id)]
        ),
    }


def _observe_dual_constraint_step(
    accumulator: ConditionConstraintAccumulator | None,
    step_info: dict[str, float],
) -> None:
    if accumulator is None:
        return
    pid_power = float(step_info.get('reward_pid_power_reference_raw', 0.0))
    rl_power = float(step_info.get('reward_power_metric_raw', step_info.get('power', 0.0)))
    delta_s = float(step_info.get('reward_energy_delta_t_s', 0.0))
    pid_peaks = np.asarray([
        step_info.get('reward_blade_root_1_peak_reference_nm', 0.0),
        step_info.get('reward_blade_root_2_peak_reference_nm', 0.0),
        step_info.get('reward_blade_root_3_peak_reference_nm', 0.0),
    ], dtype=np.float64)
    blade_root = np.asarray([
        step_info.get('reward_blade_root_1_raw', 0.0),
        step_info.get('reward_blade_root_2_raw', 0.0),
        step_info.get('reward_blade_root_3_raw', 0.0),
    ], dtype=np.float64)
    if delta_s <= 0.0 or np.any(pid_peaks <= 0.0) or not np.isfinite(pid_peaks).all():
        raise ValueError(
            'dual_constraint requires paired PID power, per-blade peak references, and positive transition duration'
        )
    accumulator.observe(
        pid_power_w=pid_power,
        rl_power_w=rl_power,
        blade_root_my_nm=blade_root,
        delta_s=delta_s,
        pid_peak_abs_nm=pid_peaks,
    )


def _finalize_queue_condition_constraints(
    accumulator: ConditionConstraintAccumulator | None,
    controller: DualConstraintController | None,
    config: DualConstraintConfig | None,
    *,
    complete: bool,
) -> tuple[ConditionConstraintAccumulator | None, dict[str, float]]:
    if controller is None:
        return accumulator, {}
    if not complete or accumulator is None:
        return ConditionConstraintAccumulator(), {'dual_constraint_condition_complete': 0.0}
    if config is None:
        raise ValueError('dual_constraint config is required when a condition is complete')
    metrics = accumulator.finalize(config)
    metrics.update(controller.complete_condition(metrics))
    metrics['dual_constraint_condition_complete'] = 1.0
    return ConditionConstraintAccumulator(), metrics


def _apply_dual_step_penalty(
    dual_controller: DualConstraintController | None,
    safe_action,
    step_result,
    step_info: dict[str, float],
) -> float:
    if dual_controller is None:
        return 0.0
    optimization_state_code = int(getattr(safe_action, 'optimization_state_code', -1))
    load_increment = max(
        float(step_info.get('reward_blade_root_1_peak_excess_increment', 0.0)),
        float(step_info.get('reward_blade_root_2_peak_excess_increment', 0.0)),
        float(step_info.get('reward_blade_root_3_peak_excess_increment', 0.0)),
    )
    pid_power = float(step_info.get('reward_pid_power_reference_raw', 0.0))
    rl_power = float(step_info.get('reward_power_raw', step_info.get('power', 0.0)))
    power_deficit_fraction = 0.0
    if math.isfinite(pid_power) and pid_power > 0.0 and math.isfinite(rl_power):
        power_deficit_fraction = max(pid_power - rl_power, 0.0) / pid_power
    penalty = dual_controller.step_penalty(
        optimization_state_code=optimization_state_code,
        g_load_increment=load_increment,
        power_deficit_fraction=power_deficit_fraction,
    )
    step_result.reward = float(step_result.reward - penalty)
    step_info['dual_step_penalty'] = float(penalty)
    step_info['dual_step_load_increment'] = float(load_increment)
    step_info['dual_step_power_deficit_fraction'] = float(power_deficit_fraction)
    return float(penalty)


def _queue_rejection_metric_key(step_info: dict[str, float], transition_finite: bool) -> str:
    if step_info.get('condition_boundary', 0.0) > 0.0:
        return 'queue_condition_rejected_boundary_count'
    if step_info.get('transition_store_eligible', 0.0) <= 0.0:
        return 'queue_condition_rejected_non_train_count'
    if step_info.get('control_fallback_active', 0.0) > 0.0:
        return 'queue_condition_rejected_fallback_count'
    if step_info.get('control_unsafe_stop', 0.0) > 0.0:
        return 'queue_condition_rejected_unsafe_count'
    if step_info.get('control_negative_power_steps', 0.0) > 0.0:
        return 'queue_condition_rejected_negative_power_count'
    if not transition_finite or step_info.get('env_invalid_prediction', 0.0) > 0.0 or step_info.get('env_invalid_reward', 0.0) > 0.0:
        return 'queue_condition_rejected_invalid_count'
    return 'queue_condition_rejected_other_count'


def _queue_exploration_noise(queue_cfg: dict[str, object], train_cfg: dict[str, object], stage: str) -> float:
    default_collect_noise = float(train_cfg.get('warmup_exploration_noise', 0.002))
    if stage == 'collect':
        return float(queue_cfg.get('collect_exploration_noise', default_collect_noise))
    if stage == 'critic':
        return float(queue_cfg.get('critic_exploration_noise', default_collect_noise))
    return float(queue_cfg.get('actor_exploration_noise', train_cfg.get('exploration_noise', 0.005)))


def _queue_checkpoint_state(
    agent,
    config: dict,
    controller: SafeResidualController,
    lifecycle: ConditionQueueLifecycle,
    replay_buffer: ConditionStratifiedReplayBuffer,
    total_steps: int,
    checkpoint_metadata: dict[str, object] | None = None,
    skipped_condition_count: int = 0,
    retried_condition_attempt_count: int = 0,
    training_valid: bool = True,
) -> dict:
    state = {
        'agent': agent.state_dict(),
        'config': config,
        'data_schema_version': config['data_schema_version'],
        'control_fingerprint': controller.control_fingerprint,
        'actor_control_fingerprint': controller.actor_control_fingerprint,
        'online_training_fingerprint': controller.online_training_fingerprint,
        'control_contract_fingerprint': controller.resolved_control_contract_fingerprint,
        'queue_state': {
            'completed_rounds': lifecycle.completed_rounds,
            'total_rounds': lifecycle.total_rounds,
            'condition_counts': replay_buffer.condition_counts(),
            'condition_source_counts': replay_buffer.condition_source_counts(),
            'skipped_condition_count': int(skipped_condition_count),
            'retried_condition_attempt_count': int(retried_condition_attempt_count),
            'training_valid': bool(training_valid),
        },
        'total_steps': total_steps,
    }
    if checkpoint_metadata is not None:
        state['queue_checkpoint'] = checkpoint_metadata
    return state


def _append_offline_eval_job(
    logger: ExperimentLogger,
    checkpoint_name: str,
    completed_rounds: int,
) -> None:
    job = {
        'checkpoint_path': str(logger.checkpoint_dir / checkpoint_name),
        'completed_rounds': completed_rounds,
        'requested_at_utc': datetime.utcnow().isoformat(timespec='seconds') + 'Z',
    }
    with (logger.output_dir / 'offline_eval_jobs.jsonl').open('a', encoding='utf-8') as file:
        file.write(json.dumps(job, ensure_ascii=False) + '\n')


def run_queue_training(
    agent,
    env: WindTurbineEnv,
    controller: SafeResidualController,
    replay_buffer: ConditionStratifiedReplayBuffer,
    config: dict,
    logger: ExperimentLogger,
    device: torch.device,
    bc_iterator=None,
    bc_dataset: OfflineBCDataset | None = None,
    offline_eval_dataset: OfflineBCDataset | None = None,
    physical_load_shadow: PhysicalLoadShadow | None = None,
) -> dict[str, float]:
    if not env.queue_lifecycle_enabled:
        raise ValueError('run_queue_training requires a queue-lifecycle environment')
    if not isinstance(replay_buffer, ConditionStratifiedReplayBuffer):
        raise ValueError('run_queue_training requires ConditionStratifiedReplayBuffer')
    if physical_load_shadow is not None and not physical_backend_selected(config):
        raise ValueError('Physical worker/config backend mismatch; dual NN/physical paths are forbidden')
    if physical_backend_selected(config):
        validate_physical_launch(config, project_root=PROJECT_ROOT)
        if physical_load_shadow is None:
            raise ValueError('Physical backend selected without its read-only worker; NN fallback is forbidden')
        validate_physical_training_binding(config, env, controller, agent, physical_load_shadow,
                                           device, project_root=PROJECT_ROOT)

    train_cfg = config['train']
    execution_replay = execution_mode_replay_contract(train_cfg)
    reward_config = config.get('env', {}).get('reward', {})
    if not isinstance(reward_config, dict):
        raise ValueError('env.reward must be an object')
    dual_constraint_config = DualConstraintConfig.from_reward_config(reward_config)
    dual_constraint_controller = (
        None
        if dual_constraint_config is None
        else DualConstraintController(dual_constraint_config)
    )
    critic_q_health_monitor = CriticQHealthMonitor(
        CriticQHealthConfig.from_train_config(train_cfg)
    )
    actor_gate_mode = resolve_actor_gate_mode(train_cfg)
    optimizer_health_config = ActorOptimizerHealthGateConfig.from_train_config(train_cfg)
    actor_optimizer_health_gate = (
        ActorOptimizerHealthGate(optimizer_health_config)
        if actor_gate_mode['mode'] == 'optimizer' and optimizer_health_config is not None
        else None
    )
    actor_q_health_gate = (
        ActorQHealthGate(ActorQHealthGateConfig.from_train_config(train_cfg))
        if actor_gate_mode['mode'] == 'trend'
        else None
    )
    print(
        f"[Queue] actor_gate={actor_gate_mode['mode']} "
        f"trend_configured={actor_gate_mode['trend_configured']} "
        f"optimizer_configured={actor_gate_mode['optimizer_configured']}"
        + (f" note={actor_gate_mode['note']}" if actor_gate_mode['note'] else ''),
        flush=True,
    )
    queue_cfg = config.get('queue', {})
    if not isinstance(queue_cfg, dict):
        raise ValueError('queue must be an object')
    load_predictor_shadow, load_predictor_artifact_summary = _build_queue_load_predictor_shadow(
        config,
        env,
        device,
    )
    load_predictor_config = config.get('load_predictor', {})
    if not isinstance(load_predictor_config, dict):
        load_predictor_config = {}
    load_predictor_audit_path = logger.output_dir / 'load_predictor_shadow_audit.jsonl'
    load_predictor_audit_max_records = int(load_predictor_config.get('shadow_audit_max_records', 250_000))
    load_predictor_audit_max_record_bytes = int(
        load_predictor_config.get('shadow_audit_max_record_bytes', 64 * 1024)
    )
    load_predictor_audit_max_total_bytes = int(
        load_predictor_config.get('shadow_audit_max_total_bytes', 256 * 1024 * 1024)
    )
    if load_predictor_shadow is not None and any(value <= 0 for value in (
        load_predictor_audit_max_records,
        load_predictor_audit_max_record_bytes,
        load_predictor_audit_max_total_bytes,
    )):
        raise ValueError('Load predictor shadow audit budgets must be positive')
    load_predictor_audit_records_written = 0
    load_predictor_audit_bytes_written = 0
    manifest_slots = _validate_queue_manifest_config(config, queue_cfg)
    condition_replay_cfg = train_cfg.get('condition_replay', {})
    if not isinstance(condition_replay_cfg, dict):
        raise ValueError('train.condition_replay must be an object')
    offline_eval_cfg = config.get('offline_eval', {})
    if not isinstance(offline_eval_cfg, dict):
        raise ValueError('offline_eval must be an object')
    if bool(train_cfg.get('online_eval_enabled', False)) or int(train_cfg.get('eval_every', 0)) > 0:
        raise ValueError('v7 queue training forbids shared-UDP online evaluation; set online_eval_enabled=false and eval_every=0')
    fixed_offline_health_obs: torch.Tensor | None = None
    fixed_offline_health_actions: torch.Tensor | None = None
    if actor_optimizer_health_gate is not None:
        if bc_dataset is None:
            raise ValueError('actor optimizer health gate requires the configured BC dataset')
        fixed_batch_size = actor_optimizer_health_gate.config.fixed_offline_batch_size
        if len(bc_dataset) < fixed_batch_size:
            raise ValueError(
                'actor optimizer health gate fixed offline batch is larger than the configured BC dataset'
            )
        fixed_indices = np.linspace(0, len(bc_dataset) - 1, num=fixed_batch_size, dtype=np.int64)
        fixed_offline_health_obs = torch.from_numpy(
            np.asarray(bc_dataset.obs[fixed_indices], dtype=np.float32)
        ).to(device)
        if bc_dataset.action_targets is None:
            fixed_actions_real = torch.from_numpy(
                np.asarray(bc_dataset.actions[fixed_indices], dtype=np.float32)
            ).to(device)
            fixed_offline_health_actions = controller.action_target_tensor(
                fixed_actions_real,
                obs=fixed_offline_health_obs,
            )
        else:
            fixed_offline_health_actions = torch.from_numpy(
                np.asarray(bc_dataset.action_targets[fixed_indices], dtype=np.float32)
            ).to(device)

    lifecycle = ConditionQueueLifecycle(
        _queue_condition_ids(config),
        total_rounds=int(queue_cfg.get('total_condition_rounds', 0)),
        critic_start_after_rounds=int(queue_cfg.get('critic_start_after_rounds', 1)),
        actor_start_after_rounds=int(queue_cfg.get('actor_start_after_rounds', 2)),
    )
    operator_development_route = config.get('physical_shadow_launch', {}).get('route')
    operator_development_authorized = operator_development_route in OPERATOR_TRAINING_ROUTES
    case_identity_validator = build_queue_case_identity_validator(
        queue_cfg, lifecycle.condition_ids,
        operator_development_authorized=(physical_backend_selected(config) and
            operator_development_authorized),
    )
    strict_case_identity = isinstance(case_identity_validator, QueueStrictCaseIdentityValidator)
    physical_operator_identity = strict_case_identity and physical_backend_selected(config) and (
        operator_development_authorized
    )
    full_manifest_identity = strict_case_identity and (
        physical_operator_identity or getattr(env, 'udp_protocol_version',
            config.get('env', {}).get('udp_protocol_version')) == QUEUE_UDP_PROTOCOL_VERSION
    )
    skip_failed_condition_on_train_timeout = queue_cfg.get('skip_failed_condition_on_train_timeout', False)
    if not isinstance(skip_failed_condition_on_train_timeout, bool):
        raise ValueError('queue.skip_failed_condition_on_train_timeout must be a boolean')
    require_zero_skipped_conditions_for_updates = queue_cfg.get(
        'require_zero_skipped_conditions_for_updates',
        False,
    )
    if not isinstance(require_zero_skipped_conditions_for_updates, bool):
        raise ValueError('queue.require_zero_skipped_conditions_for_updates must be a boolean')
    require_zero_retried_condition_attempts_for_updates = queue_cfg.get(
        'require_zero_retried_condition_attempts_for_updates',
        require_zero_skipped_conditions_for_updates,
    )
    if not isinstance(require_zero_retried_condition_attempts_for_updates, bool):
        raise ValueError('queue.require_zero_retried_condition_attempts_for_updates must be a boolean')
    abort_on_retried_condition_attempt = queue_cfg.get(
        'abort_on_retried_condition_attempt',
        require_zero_retried_condition_attempts_for_updates,
    )
    if not isinstance(abort_on_retried_condition_attempt, bool):
        raise ValueError('queue.abort_on_retried_condition_attempt must be a boolean')
    reconcile_verified_physical_skip = queue_cfg.get('reconcile_verified_physical_skip', False)
    if not isinstance(reconcile_verified_physical_skip, bool):
        raise ValueError('queue.reconcile_verified_physical_skip must be a boolean')
    if reconcile_verified_physical_skip:
        if manifest_slots is None:
            raise ValueError('queue.reconcile_verified_physical_skip requires queue.manifest_path')
        if case_identity_validator is None:
            raise ValueError(
                'queue.reconcile_verified_physical_skip requires queue.case_identity_validation.enabled=true'
            )
        if env.udp_protocol_version != QUEUE_UDP_PROTOCOL_VERSION:
            raise ValueError(
                'queue.reconcile_verified_physical_skip requires '
                f'env.udp_protocol_version={QUEUE_UDP_PROTOCOL_VERSION!r}'
            )
    if strict_case_identity:
        if manifest_slots is None:
            raise ValueError('strict queue case identity requires queue.manifest_path')
        if reconcile_verified_physical_skip:
            raise ValueError(
                'strict queue case identity cannot use prefix-based queue.reconcile_verified_physical_skip'
            )
        assert isinstance(case_identity_validator, QueueStrictCaseIdentityValidator)
        for slot in manifest_slots:
            entry = case_identity_validator.reference.entry(slot.condition_id)
            if physical_operator_identity:
                expected_times = np.arange(round(slot.train_sim_s / 0.02) + 1, dtype=np.float64) * 0.02
                if (len(entry.timestamps_s) != len(expected_times) or
                        not np.allclose(entry.timestamps_s, expected_times, rtol=0.0, atol=1.0e-6)):
                    raise ValueError('Physical operator reference requires every 0.02s frame including both endpoints')
            if (
                not math.isclose(entry.time_window_start_s, 0.0, rel_tol=0.0, abs_tol=1.0e-6)
                or not math.isclose(
                    entry.time_window_end_s - entry.time_window_start_s,
                    slot.train_sim_s,
                    rel_tol=0.0,
                    abs_tol=1.0e-6,
                )
            ):
                raise ValueError(
                    'strict queue case identity paired PID reference window must be [0, queue-slot train_sim_s] '
                    f'for condition={slot.condition_id}'
                )
    min_valid_transitions = int(condition_replay_cfg.get('min_valid_transitions_per_condition', 20_000))
    batch_size = int(train_cfg.get('batch_size', 128))
    if batch_size <= 0:
        raise ValueError('train.batch_size must be positive')
    online_rl_enabled = bool(train_cfg.get('online_rl_enabled', True))
    updates_per_step = max(1, int(train_cfg.get('updates_per_step', 1)))
    update_every_steps = max(1, int(train_cfg.get('update_every_steps', 30)))
    step_log_every = max(1, int(train_cfg.get('step_log_every', 20)))
    condition_checkpoint_count = _queue_condition_checkpoint_count(queue_cfg)
    condition_checkpoint_interval_s = None
    if condition_checkpoint_count > 0:
        raw_train_sim_s = queue_cfg.get('train_sim_s')
        if raw_train_sim_s is None:
            raise ValueError('queue.condition_checkpoint_count requires queue.train_sim_s')
        train_sim_s = float(raw_train_sim_s)
        if not math.isfinite(train_sim_s) or train_sim_s <= 0.0:
            raise ValueError('queue.train_sim_s must be positive and finite')
        condition_checkpoint_interval_s = train_sim_s / condition_checkpoint_count
    current_td3_bc_weight = float(config['agent'].get('td3_bc_weight', 1.0))
    force_baseline = bool(queue_cfg.get('force_baseline', False))
    force_bc_baseline = bool(queue_cfg.get('force_bc_baseline', False))
    pid_history_diagnostic_raw = queue_cfg.get('pid_history_diagnostic_path')
    force_pid_history = pid_history_diagnostic_raw is not None
    pid_history_path = None
    pid_history_actions = None
    pid_history_winds = None
    pid_history_observations = None
    pid_history_max_wind_error = None
    pid_history_action_offset_steps = 0
    if force_pid_history:
        if not isinstance(pid_history_diagnostic_raw, str) or not pid_history_diagnostic_raw.strip():
            raise ValueError('queue.pid_history_diagnostic_path must be a non-empty CSV path')
        (
            pid_history_path,
            pid_history_actions,
            pid_history_winds,
            pid_history_observations,
        ) = _load_pid_history_diagnostic(pid_history_diagnostic_raw)
        pid_history_max_wind_error = float(queue_cfg.get('pid_history_diagnostic_max_wind_error', 1.0))
        if not math.isfinite(pid_history_max_wind_error) or pid_history_max_wind_error < 0.0:
            raise ValueError('queue.pid_history_diagnostic_max_wind_error must be a finite non-negative number')
        raw_action_offset = queue_cfg.get('pid_history_diagnostic_action_offset_steps', 0)
        if isinstance(raw_action_offset, bool) or not isinstance(raw_action_offset, int) or raw_action_offset < 0:
            raise ValueError('queue.pid_history_diagnostic_action_offset_steps must be a non-negative integer')
        pid_history_action_offset_steps = raw_action_offset
    residual_scale_diagnostic_raw = queue_cfg.get('residual_scale_diagnostic')
    residual_scale_diagnostic = None
    if residual_scale_diagnostic_raw is not None:
        if isinstance(residual_scale_diagnostic_raw, bool):
            raise ValueError('queue.residual_scale_diagnostic must be a finite number between 0 and 1')
        residual_scale_diagnostic = float(residual_scale_diagnostic_raw)
        if not math.isfinite(residual_scale_diagnostic) or not 0.0 <= residual_scale_diagnostic <= 1.0:
            raise ValueError('queue.residual_scale_diagnostic must be a finite number between 0 and 1')
    force_diagnostic_count = sum(
        (
            force_baseline,
            force_bc_baseline,
            force_pid_history,
            residual_scale_diagnostic is not None,
        )
    )
    if force_diagnostic_count > 1:
        raise ValueError('Only one queue diagnostic action mode can be enabled at a time')
    hold_initial_baseline = bool(queue_cfg.get('force_baseline_hold_initial', False))
    if hold_initial_baseline and not force_baseline:
        raise ValueError('queue.force_baseline_hold_initial requires queue.force_baseline=true')
    frozen_bc_actor = None
    if force_bc_baseline:
        actor = getattr(agent, 'actor', None)
        if not isinstance(actor, torch.nn.Module):
            raise ValueError('queue.force_bc_baseline requires an agent with a torch actor module')
        frozen_bc_actor = copy.deepcopy(actor).to(device).eval()
        for parameter in frozen_bc_actor.parameters():
            parameter.requires_grad_(False)
    diagnostic_mode = force_baseline or force_bc_baseline or force_pid_history or residual_scale_diagnostic is not None

    total_steps = 0
    total_stored = 0
    total_rejected = 0
    update_count = 0
    actor_update_executed_count = 0
    total_skipped_conditions = 0
    total_retried_condition_attempts = 0
    train_timeout_count_since_log = 0
    obs, condition_info = env.reset()
    controller.reset_condition(obs, previous_action_real=env.last_sent_action)
    held_baseline_action = controller.baseline_action_for_obs(obs) if hold_initial_baseline else None
    pid_history_index = 0
    active_condition_id, active_condition_instance_id, active_train = _queue_context_from_info(
        condition_info,
        expected_condition_ids=lifecycle.condition_ids,
    )
    active_train_phase_elapsed_start_s = _queue_phase_elapsed_s(condition_info) if active_train else None
    active_train_phase_elapsed_latest_s = active_train_phase_elapsed_start_s
    active_condition_stats = _queue_condition_stats()
    condition_constraint_accumulator = (
        None if dual_constraint_controller is None else ConditionConstraintAccumulator()
    )
    batch_source_stats_by_round: dict[int, dict[str, object]] = {}
    next_condition_checkpoint_index = 1
    if case_identity_validator is not None:
        if strict_case_identity:
            _, _, runtime_case_fingerprint = _strict_queue_identity_sample(env)
            assert isinstance(case_identity_validator, QueueStrictCaseIdentityValidator)
            case_identity_validator.reset_condition(
                active_condition_id,
                active_condition_instance_id,
                runtime_case_fingerprint=runtime_case_fingerprint,
            )
        else:
            case_identity_validator.reset_condition(active_condition_id, active_condition_instance_id)
    if load_predictor_shadow is not None:
        load_predictor_shadow.reset_condition(active_condition_id, active_condition_instance_id)
    if physical_load_shadow is not None:
        physical_load_shadow.reset_condition(active_condition_id, active_condition_instance_id)
    manifest_slot_index = 0
    manifest_slot_validation_required = reconcile_verified_physical_skip or strict_case_identity

    def validate_active_manifest_slot(context: str) -> None:
        if not manifest_slot_validation_required:
            return
        assert manifest_slots is not None
        if manifest_slot_index >= len(manifest_slots):
            raise RuntimeError(f'Queue manifest has no active slot during {context}')
        expected_slot = manifest_slots[manifest_slot_index]
        if full_manifest_identity:
            metadata = getattr(env, 'current_queue_metadata', condition_info)
            _validate_strict_manifest_identity(expected_slot, metadata, context=context)
        if active_condition_id != expected_slot.condition_id:
            raise RuntimeError(
                f'Queue manifest slot mismatch during {context}: slot={manifest_slot_index + 1} '
                f'expected_condition={expected_slot.condition_id} observed_condition={active_condition_id}'
            )

    def advance_manifest_slot_after_active_condition() -> None:
        nonlocal manifest_slot_index
        if not manifest_slot_validation_required:
            return
        validate_active_manifest_slot('slot advance')
        manifest_slot_index += 1

    validate_active_manifest_slot('initial activation')

    def queue_condition_identity_metadata(
        identity_status: str,
        identity_summary: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            'identity_status': identity_status,
            'condition_id': int(active_condition_id),
            'condition_instance_id': int(active_condition_instance_id),
        }
        reward_fn = getattr(env, 'reward_fn', None)
        paired_reference = getattr(reward_fn, 'paired_pid_reference', None)
        if paired_reference is not None:
            entry = paired_reference.entry(active_condition_id)
            metadata.update({
                'scenario_id': entry.scenario_id,
                'pid_raw_csv_path': str(entry.pid_raw_csv_path),
                'pid_raw_sha256': entry.pid_raw_sha256,
                'time_window_start_s': float(entry.time_window_start_s),
                'time_window_end_s': float(entry.time_window_end_s),
                'expected_trace_signature': entry.trace_signature,
            })
        if identity_summary:
            for key in (
                'identity_status',
                'observed_trace_signature',
                'expected_trace_signature',
                'trace_signature_match',
                'expected_trace_sample_count',
                'observed_trace_sample_count',
                'fixed_simulation_time_offset_s',
                'pid_raw_sha256',
                'pid_raw_csv_path',
                'time_window_start_s',
                'time_window_end_s',
                'scenario_id',
            ):
                if key in identity_summary:
                    metadata[key] = identity_summary[key]
        return metadata

    def finish_queue_load_predictor_condition(
        identity_status: str,
        identity_summary: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        nonlocal load_predictor_audit_records_written
        nonlocal load_predictor_audit_bytes_written
        if physical_load_shadow is not None:
            physical_load_shadow.finish_condition(physical_condition_metadata(
                physical_load_shadow, condition_id=active_condition_id,
                condition_instance_id=active_condition_instance_id, identity_status=identity_status,
                identity_summary=identity_summary, queue_epoch=int(condition_info.get('udp_queue_epoch', -1))))
        if load_predictor_shadow is None:
            return {}
        finish_stats = load_predictor_shadow.finish_condition()
        audit_records = load_predictor_shadow.drain_audit_records()
        pid_raw_sha256 = ''
        pid_identity_metadata: dict[str, Any] = {}
        if isinstance(case_identity_validator, QueueStrictCaseIdentityValidator):
            reference_entry = case_identity_validator.reference.entry(active_condition_id)
            pid_raw_sha256 = reference_entry.pid_raw_sha256
            pid_identity_metadata = {
                'scenario_id': reference_entry.scenario_id,
                'pid_raw_csv_path': str(reference_entry.pid_raw_csv_path),
                'time_window_start_s': float(reference_entry.time_window_start_s),
                'time_window_end_s': float(reference_entry.time_window_end_s),
                'expected_trace_signature': reference_entry.trace_signature,
                'expected_trace_sample_count': int(reference_entry.timestamps_s.size),
            }
        serialized_rows: list[bytes] = []
        calibration_eligible_count = 0
        for raw_record in audit_records:
            record = {
                **raw_record,
                'identity_status': identity_status,
                'strict_case_matched': float(identity_status == 'strict_case_matched'),
                'pid_raw_sha256': pid_raw_sha256,
                **pid_identity_metadata,
                'predictor_interface_sha256': load_predictor_shadow.interface_sha256,
                'predictor_artifact_sha256': str(
                    load_predictor_artifact_summary.get('load_predictor_shadow_model_sha256', '')
                ),
            }
            serialized = json.dumps(
                record,
                ensure_ascii=False,
                allow_nan=False,
                separators=(',', ':'),
            ).encode('utf-8')
            if len(serialized) > load_predictor_audit_max_record_bytes:
                raise ValueError(
                    'Queue load predictor shadow audit record exceeds configured size budget: '
                    f'{len(serialized)} > {load_predictor_audit_max_record_bytes}'
                )
            serialized_rows.append(serialized + b'\n')
            calibration_eligible_count += int(bool(record.get('calibration_eligible', False)))
        append_bytes = sum(len(row) for row in serialized_rows)
        if load_predictor_audit_records_written + len(serialized_rows) > load_predictor_audit_max_records:
            raise ValueError('Queue load predictor shadow audit exceeds configured row budget')
        if load_predictor_audit_bytes_written + append_bytes > load_predictor_audit_max_total_bytes:
            raise ValueError('Queue load predictor shadow audit exceeds configured byte budget')
        if serialized_rows:
            with load_predictor_audit_path.open('ab') as handle:
                handle.writelines(serialized_rows)
        load_predictor_audit_records_written += len(serialized_rows)
        load_predictor_audit_bytes_written += append_bytes
        metrics = {
            'queue_condition_load_predictor_dropped_pending_count': float(
                finish_stats.get('dropped_pending_count', 0)
            ),
            'queue_condition_load_predictor_dropped_prediction_count': float(
                finish_stats.get('pending_prediction_dropped_count', 0)
            ),
            'queue_condition_load_predictor_resolved_sample_count': float(
                finish_stats.get('resolved_record_count', 0)
            ),
            'queue_condition_load_predictor_discontinuity_count': float(
                finish_stats.get('discontinuity_count', 0)
            ),
            'queue_condition_load_predictor_audit_record_count': float(
                active_condition_stats['queue_condition_load_predictor_audit_record_count']
            ),
            'queue_condition_load_predictor_audit_rows_written': float(len(serialized_rows)),
            'queue_condition_load_predictor_calibration_eligible_record_count': float(
                calibration_eligible_count
            ),
        }
        active_condition_stats.update(metrics)
        return metrics

    def updates_quarantined_after_skip() -> bool:
        return require_zero_skipped_conditions_for_updates and total_skipped_conditions > 0

    def updates_quarantined_after_retry() -> bool:
        return (
            require_zero_retried_condition_attempts_for_updates
            and total_retried_condition_attempts > 0
        )

    def updates_quarantined_after_failure() -> bool:
        return updates_quarantined_after_skip() or updates_quarantined_after_retry()

    def queue_training_valid() -> bool:
        return not updates_quarantined_after_failure()

    def critic_q_health_summary() -> dict[str, float]:
        degraded = critic_q_health_monitor.degraded
        return {
            'queue_critic_q_health_degraded': float(degraded),
            'queue_critic_q_health_failure_triggered': float(
                critic_q_health_monitor.failure_triggered
            ),
            'queue_critic_q_health_passed': float(not degraded),
            'queue_critic_q_health_continue_freeze_actor': float(
                critic_q_health_monitor.config.failure_mode == 'continue_freeze_actor'
            ),
        }

    def save_abort_diagnostics(reason_code: float, step_info: dict[str, float]) -> None:
        logger.save_summary(
            'queue_abort_diagnostics.json',
            {
                'queue_abort_reason_code': reason_code,
                'queue_abort_step': float(total_steps),
                'queue_abort_condition_id': float(active_condition_id),
                'queue_abort_condition_instance_id': float(active_condition_instance_id),
                'queue_abort_power': float(step_info.get('power', 0.0)),
                'queue_abort_power_reward_metric': float(step_info.get('power_reward_metric', 0.0)),
                'queue_abort_power_state_real': float(step_info.get('power_state_real', 0.0)),
                'queue_abort_ood_score': float(step_info.get('control_ood_score', 0.0)),
                'queue_abort_ood_feature_index': float(step_info.get('control_ood_feature_index', -1.0)),
                'queue_abort_ood_feature_value': float(step_info.get('control_ood_feature_value', 0.0)),
                'queue_abort_ood_feature_median': float(step_info.get('control_ood_feature_median', 0.0)),
                'queue_abort_ood_feature_iqr': float(step_info.get('control_ood_feature_iqr', 0.0)),
                'queue_abort_fallback_active': float(step_info.get('control_fallback_active', 0.0)),
                'queue_abort_negative_power_steps': float(controller.negative_power_steps),
                'queue_abort_negative_power_patience': float(controller.negative_power_patience),
                'queue_abort_low_power_steps': float(controller.low_power_steps),
                'queue_abort_low_power_patience': float(controller.low_power_patience),
                **active_condition_stats,
            },
        )
        logger.flush()

    def reconcile_retried_condition_to_verified_physical_skip(
        recovered_obs: np.ndarray,
        recovery_info: dict[str, float],
        *,
        trigger_code: float,
    ) -> tuple[np.ndarray, dict[str, float], list[np.ndarray] | None]:
        if not reconcile_verified_physical_skip:
            return recovered_obs, recovery_info, None
        assert case_identity_validator is not None
        assert manifest_slots is not None
        validate_active_manifest_slot('physical skip probe')
        if manifest_slot_index + 1 >= len(manifest_slots):
            raise RuntimeError('Queue physical skip probe cannot advance beyond the final manifest slot')

        recovered_condition_id, recovered_condition_instance_id, _recovered_train = _queue_context_from_info(
            recovery_info,
            expected_condition_ids=lifecycle.condition_ids,
        )
        if recovered_condition_id != active_condition_id:
            raise RuntimeError('Queue physical skip probe requires a same-condition retry metadata frame')
        if recovered_condition_instance_id == active_condition_instance_id:
            raise RuntimeError('Queue physical skip probe requires a new retry condition instance')

        probe_observations = [recovered_obs]
        latest_obs = recovered_obs
        latest_info = recovery_info
        for _ in range(1, case_identity_validator.wind_sample_count):
            latest_obs, latest_info = env.probe_pending_condition()
            probe_condition_id, probe_condition_instance_id, _probe_train = _queue_context_from_info(
                latest_info,
                expected_condition_ids=lifecycle.condition_ids,
            )
            if (
                probe_condition_id != recovered_condition_id
                or probe_condition_instance_id != recovered_condition_instance_id
            ):
                raise RuntimeError(
                    'Queue physical skip probe changed queue metadata before physical identity was verified'
                )
            probe_observations.append(latest_obs)

        matched_condition_id, probe_metrics = case_identity_validator.classify_observation_prefix(
            probe_observations
        )
        expected_next_slot = manifest_slots[manifest_slot_index + 1]
        diagnostics = {
            'queue_physical_skip_probe_trigger_code': trigger_code,
            'queue_physical_skip_probe_active_condition_id': float(active_condition_id),
            'queue_physical_skip_probe_retry_instance_id': float(recovered_condition_instance_id),
            'queue_physical_skip_probe_expected_next_slot_index': float(manifest_slot_index + 2),
            'queue_physical_skip_probe_expected_next_condition_id': float(expected_next_slot.condition_id),
            'queue_physical_skip_probe_approved': 0.0,
            **probe_metrics,
        }
        artifact_name = (
            f'queue_physical_skip_probe_condition_{active_condition_id:02d}_'
            f'instance_{active_condition_instance_id:03d}.json'
        )
        if matched_condition_id != expected_next_slot.condition_id:
            logger.save_summary(artifact_name, diagnostics)
            logger.log_metrics('queue_physical_skip_probe', total_steps, diagnostics)
            logger.flush()
            return latest_obs, latest_info, None

        remapped_obs, remapped_info = env.advance_pending_incomplete_slot()
        remapped_condition_id, remapped_condition_instance_id, _remapped_train = _queue_context_from_info(
            remapped_info,
            expected_condition_ids=lifecycle.condition_ids,
        )
        if remapped_condition_id != expected_next_slot.condition_id:
            raise RuntimeError(
                'Queue verified physical skip remapped to an unexpected metadata condition: '
                f'expected={expected_next_slot.condition_id} got={remapped_condition_id}'
            )
        if remapped_condition_instance_id == recovered_condition_instance_id:
            raise RuntimeError(
                'Queue verified physical skip remap did not advance to a new condition instance'
            )
        diagnostics.update(
            {
                'queue_physical_skip_probe_approved': 1.0,
                'queue_physical_skip_remapped_condition_id': float(remapped_condition_id),
                'queue_physical_skip_remapped_condition_instance_id': float(remapped_condition_instance_id),
            }
        )
        logger.save_summary(artifact_name, diagnostics)
        logger.log_metrics('queue_physical_skip_probe', total_steps, diagnostics)
        logger.flush()
        print(
            f'[Queue] verified physical skip condition={active_condition_id} '
            f'instance={active_condition_instance_id} '
            f'physical_next_condition={matched_condition_id} '
            f'remapped_instance={remapped_condition_instance_id}',
            flush=True,
        )
        return remapped_obs, remapped_info, probe_observations

    def save_condition_checkpoint(
        *,
        checkpoint_index: int,
        target_elapsed_s: float,
        actual_elapsed_s: float,
    ) -> None:
        round_index = lifecycle.completed_rounds + 1
        checkpoint_name = _queue_condition_checkpoint_name(
            round_index=round_index,
            condition_id=active_condition_id,
            condition_instance_id=active_condition_instance_id,
            checkpoint_index=checkpoint_index,
            checkpoint_count=condition_checkpoint_count,
        )
        checkpoint_metadata = {
            'checkpoint_kind': 'condition_progress',
            'round_index': round_index,
            'condition_id': active_condition_id,
            'condition_instance_id': active_condition_instance_id,
            'checkpoint_index': checkpoint_index,
            'checkpoint_count': condition_checkpoint_count,
            'target_train_elapsed_s': target_elapsed_s,
            'actual_train_elapsed_s': actual_elapsed_s,
            'total_steps': total_steps,
        }
        logger.save_checkpoint(
            checkpoint_name,
            _queue_checkpoint_state(
                agent,
                config,
                controller,
                lifecycle,
                replay_buffer,
                total_steps,
                checkpoint_metadata=checkpoint_metadata,
                skipped_condition_count=total_skipped_conditions,
                retried_condition_attempt_count=total_retried_condition_attempts,
                training_valid=queue_training_valid(),
            ),
        )
        logger.log_metrics(
            'queue_condition_checkpoint',
            total_steps,
            {
                'queue_checkpoint_round_index': float(round_index),
                'queue_checkpoint_condition_id': float(active_condition_id),
                'queue_checkpoint_condition_instance_id': float(active_condition_instance_id),
                'queue_checkpoint_index': float(checkpoint_index),
                'queue_checkpoint_count': float(condition_checkpoint_count),
                'queue_checkpoint_target_train_elapsed_s': target_elapsed_s,
                'queue_checkpoint_actual_train_elapsed_s': actual_elapsed_s,
            },
        )
        logger.flush()
        print(
            f'[Queue-Checkpoint] saved={checkpoint_name} round={round_index} '
            f'condition={active_condition_id} instance={active_condition_instance_id} '
            f'part={checkpoint_index}/{condition_checkpoint_count} '
            f'train_elapsed_s={actual_elapsed_s:.3f}',
            flush=True,
        )

    def finalize_completed_round(completion) -> None:
        if not completion.completed_round or diagnostic_mode:
            return
        checkpoint_name = f'queue_round_{completion.completed_rounds}.pt'
        logger.save_checkpoint(
            checkpoint_name,
            _queue_checkpoint_state(
                agent,
                config,
                controller,
                lifecycle,
                replay_buffer,
                total_steps,
                skipped_condition_count=total_skipped_conditions,
                retried_condition_attempt_count=total_retried_condition_attempts,
                training_valid=queue_training_valid(),
            ),
        )
        round_summary = {
            'queue_completed_rounds': float(completion.completed_rounds),
            'queue_total_rounds': float(lifecycle.total_rounds),
            'queue_total_steps': float(total_steps),
            'queue_total_stored_transitions': float(total_stored),
            'queue_total_rejected_transitions': float(total_rejected),
            'queue_skipped_condition_count': float(total_skipped_conditions),
            'queue_retried_condition_attempt_count': float(total_retried_condition_attempts),
            'queue_require_zero_skipped_conditions_for_updates': float(
                require_zero_skipped_conditions_for_updates
            ),
            'queue_require_zero_retried_condition_attempts_for_updates': float(
                require_zero_retried_condition_attempts_for_updates
            ),
            'queue_updates_quarantined_after_skip': float(updates_quarantined_after_skip()),
            'queue_updates_quarantined_after_retry': float(updates_quarantined_after_retry()),
            'queue_updates_quarantined_after_failure': float(updates_quarantined_after_failure()),
            'queue_training_valid': float(queue_training_valid()),
            'queue_all_conditions_covered': float(
                replay_buffer.all_conditions_covered(min_valid_transitions)
            ),
            'queue_update_count': float(update_count),
        }
        sampling_stats = batch_source_stats_by_round.pop(completion.completed_rounds, None)
        if sampling_stats is not None:
            round_sample_total = int(sampling_stats['sample_count'])
            round_summary['queue_batch_count'] = float(sampling_stats['batch_count'])
            round_summary['queue_batch_sample_count'] = float(round_sample_total)
            round_summary['queue_batch_source_quota_relaxed_count'] = float(
                sampling_stats['quota_relaxed_count']
            )
            for source in REPLAY_SOURCE_NAMES:
                sampled = int(sampling_stats['source_counts'][source])
                relaxed = int(sampling_stats['quota_relaxed_by_source'][source])
                round_summary[f'queue_batch_source_{source}_count'] = float(sampled)
                round_summary[f'queue_batch_source_{source}_fraction'] = (
                    0.0 if round_sample_total <= 0 else sampled / round_sample_total
                )
                round_summary[f'queue_batch_source_quota_relaxed_{source}_count'] = float(relaxed)
            for condition_id, source_counts in sampling_stats['condition_sample_counts'].items():
                round_summary[
                    f'queue_condition_{condition_id}_batch_count'
                ] = float(sampling_stats['condition_batch_counts'][condition_id])
                round_summary[
                    f'queue_condition_{condition_id}_batch_quota_relaxed_count'
                ] = float(sampling_stats['condition_quota_relaxed_count'][condition_id])
                for source in REPLAY_SOURCE_NAMES:
                    round_summary[
                        f'queue_condition_{condition_id}_batch_source_{source}_count'
                    ] = float(source_counts[source])
                    round_summary[
                        f'queue_condition_{condition_id}_batch_source_{source}_quota_relaxed_count'
                    ] = float(
                        sampling_stats['condition_quota_relaxed_by_source'][condition_id][source]
                    )
        if dual_constraint_controller is not None:
            round_summary.update(dual_constraint_controller.update_at_round_boundary())
        round_summary.update(critic_q_health_summary())
        for condition_id, count in replay_buffer.condition_counts().items():
            round_summary[f'queue_condition_{condition_id}_replay_count'] = float(count)
            source_counts = replay_buffer.condition_source_counts()[condition_id]
            round_summary[f'queue_condition_{condition_id}_online_bladed_replay_count'] = float(
                source_counts[ONLINE_BLADED_SOURCE]
            )
        logger.save_summary(f'queue_round_{completion.completed_rounds}_summary.json', round_summary)
        if bool(offline_eval_cfg.get('enabled', False)) and queue_training_valid():
            _append_offline_eval_job(logger, checkpoint_name, completion.completed_rounds)
        logger.flush()

    def finalize_condition_constraints(*, complete: bool) -> dict[str, float]:
        nonlocal condition_constraint_accumulator
        condition_constraint_accumulator, metrics = _finalize_queue_condition_constraints(
            condition_constraint_accumulator,
            dual_constraint_controller,
            dual_constraint_config,
            complete=complete,
        )
        return metrics

    def save_skipped_condition_diagnostics(
        *,
        recovered_condition_id: int,
        recovered_condition_instance_id: int,
        train_elapsed_s: float,
        recovery_wait_ms: float,
        trigger_code: float,
    ) -> None:
        _finalize_queue_condition_step_stats(
            active_condition_stats,
            condition_id=active_condition_id,
        )
        source_counts = replay_buffer.condition_source_counts()[active_condition_id]
        round_index = (
            int(manifest_slots[manifest_slot_index].round_id)
            if manifest_slots is not None and manifest_slot_index < len(manifest_slots)
            else lifecycle.completed_rounds + 1
        )
        diagnostics = {
            'queue_skip_reason_code': 1.0,
            'queue_skip_trigger_code': trigger_code,
            'queue_skip_step': float(total_steps),
            'queue_condition_round_index': float(round_index),
            'queue_condition_id': float(active_condition_id),
            'queue_condition_instance_id': float(active_condition_instance_id),
            'queue_skip_condition_id': float(active_condition_id),
            'queue_skip_condition_instance_id': float(active_condition_instance_id),
            'queue_skip_recovered_condition_id': float(recovered_condition_id),
            'queue_skip_recovered_condition_instance_id': float(recovered_condition_instance_id),
            'queue_skip_last_train_elapsed_s': train_elapsed_s,
            'queue_skip_recovery_wait_ms': recovery_wait_ms,
            'queue_skip_power': float(step_info.get('power', 0.0)),
            'queue_skip_ood_score': float(step_info.get('control_ood_score', 0.0)),
            'queue_skip_fallback_active': float(step_info.get('control_fallback_active', 0.0)),
            'queue_skip_replay_count': float(replay_buffer.condition_counts()[active_condition_id]),
            'queue_skip_replay_online_bladed_count': float(source_counts[ONLINE_BLADED_SOURCE]),
            **active_condition_stats,
            **_batch_source_metrics_for_condition(
                batch_source_stats_by_round,
                round_index,
                active_condition_id,
                replay_buffer.source_names,
            ),
        }
        logger.save_summary(
            f'queue_skipped_condition_{active_condition_id:02d}_instance_{active_condition_instance_id:03d}.json',
            diagnostics,
        )
        logger.log_metrics(
            'queue_condition_skip',
            total_steps,
            diagnostics,
            record_metadata=queue_condition_identity_metadata('incomplete_condition'),
        )

    def skip_failed_condition_and_activate(
        recovered_obs: np.ndarray,
        recovery_info: dict[str, float],
        *,
        train_elapsed_s: float,
        recovery_wait_ms: float,
        trigger_code: float,
        identity_probe_observations: list[np.ndarray] | None = None,
    ) -> bool:
        nonlocal total_skipped_conditions
        nonlocal obs
        nonlocal held_baseline_action
        nonlocal pid_history_index
        nonlocal active_condition_id
        nonlocal active_condition_instance_id
        nonlocal active_train
        nonlocal active_train_phase_elapsed_start_s
        nonlocal active_train_phase_elapsed_latest_s
        nonlocal active_condition_stats
        nonlocal next_condition_checkpoint_index

        recovered_condition_id, recovered_condition_instance_id, recovered_train = _queue_context_from_info(
            recovery_info,
            expected_condition_ids=lifecycle.condition_ids,
        )
        if (
            recovered_condition_id == active_condition_id
            and recovered_condition_instance_id == active_condition_instance_id
        ):
            raise RuntimeError('Queue failed-condition skip did not advance to a new condition instance')

        advance_manifest_slot_after_active_condition()
        total_skipped_conditions += 1
        dual_constraint_skip_metrics = finalize_condition_constraints(complete=False)
        finish_queue_load_predictor_condition('incomplete_condition')
        save_skipped_condition_diagnostics(
            recovered_condition_id=recovered_condition_id,
            recovered_condition_instance_id=recovered_condition_instance_id,
            train_elapsed_s=train_elapsed_s,
            recovery_wait_ms=recovery_wait_ms,
            trigger_code=trigger_code,
        )
        completion = lifecycle.complete_condition(active_condition_id, active_condition_instance_id)
        skip_round_metrics = dict(dual_constraint_skip_metrics)
        skip_round_metrics.update(_batch_source_metrics_for_condition(
            batch_source_stats_by_round,
            lifecycle.completed_rounds + 1,
            active_condition_id,
            replay_buffer.source_names,
        ))
        logger.log_metrics(
            'queue_condition_skip',
            total_steps,
            {
                'queue_skip_completed_round': float(completion.completed_round),
                'queue_skip_completed_rounds': float(completion.completed_rounds),
                'queue_skip_conditions_completed_in_round': float(
                    completion.conditions_completed_in_round
                ),
                'queue_skipped_condition_count': float(total_skipped_conditions),
                **skip_round_metrics,
            },
            record_metadata=queue_condition_identity_metadata('incomplete_condition'),
        )
        finalize_completed_round(completion)
        print(
            f'[Queue] skipped failed condition={active_condition_id} '
            f'instance={active_condition_instance_id} '
            f'train_elapsed_s={train_elapsed_s:.3f} '
            f'next_condition={recovered_condition_id} '
            f'next_instance={recovered_condition_instance_id} '
            f'wait_ms={recovery_wait_ms:.1f}',
            flush=True,
        )
        logger.flush()
        if lifecycle.finished:
            return True

        obs = recovered_obs
        controller.reset_condition(obs, previous_action_real=env.last_sent_action)
        held_baseline_action = controller.baseline_action_for_obs(obs) if hold_initial_baseline else None
        pid_history_index = 0
        active_condition_id = recovered_condition_id
        active_condition_instance_id = recovered_condition_instance_id
        active_train = recovered_train
        active_train_phase_elapsed_start_s = _queue_phase_elapsed_s(recovery_info) if active_train else None
        active_train_phase_elapsed_latest_s = active_train_phase_elapsed_start_s
        active_condition_stats = _queue_condition_stats()
        if load_predictor_shadow is not None:
            load_predictor_shadow.reset_condition(active_condition_id, active_condition_instance_id)
        if physical_load_shadow is not None:
            physical_load_shadow.reset_condition(active_condition_id, active_condition_instance_id)
        next_condition_checkpoint_index = 1
        if case_identity_validator is not None:
            if identity_probe_observations is None:
                case_identity_validator.reset_condition(active_condition_id, active_condition_instance_id)
            else:
                case_identity_validator.prime_condition(
                    active_condition_id,
                    active_condition_instance_id,
                    identity_probe_observations,
                )
        validate_active_manifest_slot('skip recovery')
        return False

    def retry_failed_condition_and_activate(
        recovered_obs: np.ndarray,
        recovery_info: dict[str, float],
        *,
        train_elapsed_s: float,
        recovery_wait_ms: float,
        trigger_code: float,
        abort_after_retry: bool,
    ) -> None:
        nonlocal total_retried_condition_attempts
        nonlocal obs
        nonlocal held_baseline_action
        nonlocal pid_history_index
        nonlocal active_condition_id
        nonlocal active_condition_instance_id
        nonlocal active_train
        nonlocal active_train_phase_elapsed_start_s
        nonlocal active_train_phase_elapsed_latest_s
        nonlocal active_condition_stats
        nonlocal next_condition_checkpoint_index

        recovered_condition_id, recovered_condition_instance_id, recovered_train = _queue_context_from_info(
            recovery_info,
            expected_condition_ids=lifecycle.condition_ids,
        )
        if recovered_condition_id != active_condition_id:
            raise RuntimeError('Queue retry must retain the failed condition ID')
        if recovered_condition_instance_id == active_condition_instance_id:
            raise RuntimeError('Queue retry did not advance to a new condition instance')

        total_retried_condition_attempts += 1
        source_counts = replay_buffer.condition_source_counts()[active_condition_id]
        retry_round_index = (
            int(manifest_slots[manifest_slot_index].round_id)
            if manifest_slots is not None and manifest_slot_index < len(manifest_slots)
            else lifecycle.completed_rounds + 1
        )
        dual_constraint_retry_metrics = finalize_condition_constraints(complete=False)
        finish_queue_load_predictor_condition('incomplete_condition')
        _finalize_queue_condition_step_stats(
            active_condition_stats,
            condition_id=active_condition_id,
        )
        diagnostics = {
            'queue_retry_reason_code': 1.0,
            'queue_retry_trigger_code': trigger_code,
            'queue_retry_step': float(total_steps),
            'queue_retry_round_index': float(retry_round_index),
            'queue_condition_round_index': float(retry_round_index),
            'queue_condition_id': float(active_condition_id),
            'queue_condition_instance_id': float(active_condition_instance_id),
            'queue_retry_condition_id': float(active_condition_id),
            'queue_retry_failed_condition_instance_id': float(active_condition_instance_id),
            'queue_retry_recovered_condition_instance_id': float(recovered_condition_instance_id),
            'queue_retry_last_train_elapsed_s': train_elapsed_s,
            'queue_retry_recovery_wait_ms': recovery_wait_ms,
            'queue_retry_power': float(step_info.get('power', 0.0)),
            'queue_retry_ood_score': float(step_info.get('control_ood_score', 0.0)),
            'queue_retry_fallback_active': float(step_info.get('control_fallback_active', 0.0)),
            'queue_retry_replay_count': float(replay_buffer.condition_counts()[active_condition_id]),
            'queue_retry_replay_online_bladed_count': float(source_counts[ONLINE_BLADED_SOURCE]),
            'queue_retried_condition_attempt_count': float(total_retried_condition_attempts),
            **dual_constraint_retry_metrics,
            **active_condition_stats,
            **_batch_source_metrics_for_condition(
                batch_source_stats_by_round,
                retry_round_index,
                active_condition_id,
                replay_buffer.source_names,
            ),
        }
        logger.save_summary(
            f'queue_retried_condition_{active_condition_id:02d}_'
            f'instance_{active_condition_instance_id:03d}_to_{recovered_condition_instance_id:03d}.json',
            diagnostics,
        )
        logger.log_metrics(
            'queue_condition_retry',
            total_steps,
            diagnostics,
            record_metadata=queue_condition_identity_metadata('incomplete_condition'),
        )
        print(
            f'[Queue] retrying failed condition={active_condition_id} '
            f'instance={active_condition_instance_id} '
            f'next_instance={recovered_condition_instance_id} '
            f'train_elapsed_s={train_elapsed_s:.3f} '
            f'wait_ms={recovery_wait_ms:.1f}',
            flush=True,
        )
        logger.flush()

        if abort_after_retry:
            save_abort_diagnostics(4.0, step_info)
            logger.log_metrics(
                'queue_retry_abort',
                total_steps,
                {
                    'queue_retry_abort_condition_id': float(active_condition_id),
                    'queue_retry_abort_failed_condition_instance_id': float(active_condition_instance_id),
                    'queue_retry_abort_recovered_condition_instance_id': float(
                        recovered_condition_instance_id
                    ),
                    'queue_retry_abort_trigger_code': trigger_code,
                    'queue_retry_abort_train_elapsed_s': train_elapsed_s,
                    'queue_retry_abort_recovery_wait_ms': recovery_wait_ms,
                },
            )
            logger.flush()
            raise RuntimeError(
                f'Queue training aborted after retried condition={active_condition_id} '
                f'instance={active_condition_instance_id} '
                f'next_instance={recovered_condition_instance_id}; '
                'strict retry safety is enabled, so no action will be sent to the retry instance'
            )

        obs = recovered_obs
        controller.reset_condition(obs, previous_action_real=env.last_sent_action)
        held_baseline_action = controller.baseline_action_for_obs(obs) if hold_initial_baseline else None
        pid_history_index = 0
        active_condition_id = recovered_condition_id
        active_condition_instance_id = recovered_condition_instance_id
        active_train = recovered_train
        active_train_phase_elapsed_start_s = _queue_phase_elapsed_s(recovery_info) if active_train else None
        active_train_phase_elapsed_latest_s = active_train_phase_elapsed_start_s
        active_condition_stats = _queue_condition_stats()
        if load_predictor_shadow is not None:
            load_predictor_shadow.reset_condition(active_condition_id, active_condition_instance_id)
        if physical_load_shadow is not None:
            physical_load_shadow.reset_condition(active_condition_id, active_condition_instance_id)
        next_condition_checkpoint_index = 1
        if case_identity_validator is not None:
            case_identity_validator.reset_condition(active_condition_id, active_condition_instance_id)
        validate_active_manifest_slot('retry recovery')

    print(
        f'[Queue] start rounds={lifecycle.total_rounds} conditions={list(lifecycle.condition_ids)} '
        f'min_valid_per_condition={min_valid_transitions} batch_size={batch_size} '
        f'force_baseline={int(force_baseline)} force_bc_baseline={int(force_bc_baseline)} '
        f'force_pid_history={int(force_pid_history)} '
        f'hold_initial_baseline={int(hold_initial_baseline)} '
        f'residual_scale_diagnostic={residual_scale_diagnostic} '
        f'pid_history_path={pid_history_path} '
        f'pid_history_action_offset_steps={pid_history_action_offset_steps} '
        f'case_identity_validation={int(case_identity_validator is not None)} '
        f'reconcile_verified_physical_skip={int(reconcile_verified_physical_skip)} '
        f'skip_failed_condition_on_train_timeout={int(skip_failed_condition_on_train_timeout)} '
        f'require_zero_skipped_conditions_for_updates={int(require_zero_skipped_conditions_for_updates)} '
        f'require_zero_retried_condition_attempts_for_updates='
        f'{int(require_zero_retried_condition_attempts_for_updates)} '
        f'abort_on_retried_condition_attempt={int(abort_on_retried_condition_attempt)} '
        f'condition_checkpoint_count={condition_checkpoint_count}',
        flush=True,
    )
    while not lifecycle.finished:
        validate_active_manifest_slot('before control/update')
        actor_policy_version_for_step = actor_update_executed_count
        load_predictor_payload = (
            env.current_payload
            if (load_predictor_shadow is not None or physical_load_shadow is not None) and active_train
            else None
        )
        case_identity_metrics: dict[str, float] = {
            'queue_case_identity_validation_active': float(case_identity_validator is not None),
            'queue_case_identity_validated': 0.0,
        }
        if case_identity_validator is not None and active_train:
            try:
                if strict_case_identity:
                    sim_time_s, wind_speed_mps, runtime_case_fingerprint = _strict_queue_identity_sample(env)
                    assert isinstance(case_identity_validator, QueueStrictCaseIdentityValidator)
                    case_identity_metrics = case_identity_validator.observe(
                        active_condition_id,
                        active_condition_instance_id,
                        obs,
                        sim_time_s=sim_time_s,
                        wind_speed_mps=wind_speed_mps,
                        runtime_case_fingerprint=runtime_case_fingerprint,
                    )
                else:
                    case_identity_metrics = case_identity_validator.observe(
                        active_condition_id,
                        active_condition_instance_id,
                        obs,
                    )
            except QueueCaseIdentityMismatch as exc:
                mismatch_metrics = {
                    **exc.metrics,
                    'queue_case_identity_mismatch': 1.0,
                    'queue_case_identity_total_steps': float(total_steps),
                    'queue_case_identity_skipped_condition_count': float(total_skipped_conditions),
                    'queue_case_identity_retried_condition_attempt_count': float(total_retried_condition_attempts),
                }
                logger.save_summary(
                    f'queue_case_identity_mismatch_condition_{active_condition_id:02d}_'
                    f'instance_{active_condition_instance_id:03d}.json',
                    mismatch_metrics,
                )
                logger.log_metrics('queue_case_identity_mismatch', total_steps, mismatch_metrics)
                logger.flush()
                raise RuntimeError(str(exc)) from exc
        all_conditions_covered = replay_buffer.all_conditions_covered(min_valid_transitions)
        updates_quarantined = updates_quarantined_after_failure()
        stage = lifecycle.training_stage(all_conditions_covered and not updates_quarantined)
        noise_std = 0.0 if diagnostic_mode or not active_train else _queue_exploration_noise(queue_cfg, train_cfg, stage)
        obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device)
        bc_baseline_action = None
        bc_baseline_residual = None
        pid_history_action = None
        pid_history_action_index = None
        pid_history_wind = None
        pid_history_wind_error = None
        pid_history_observation_metrics: dict[str, float] = {
            'control_pid_history_observation_available': 0.0,
        }
        if force_pid_history:
            if pid_history_actions is None or pid_history_winds is None or pid_history_max_wind_error is None:
                raise RuntimeError('PID history diagnostic is not initialized')
            if pid_history_index >= len(pid_history_actions):
                raise RuntimeError(
                    f'PID history diagnostic exhausted at frame={pid_history_index}; '
                    f'available_frames={len(pid_history_actions)}'
                )
            pid_history_action_index = pid_history_index + pid_history_action_offset_steps
            if pid_history_action_index >= len(pid_history_actions):
                raise RuntimeError(
                    'PID history diagnostic action history exhausted at '
                    f'observation_frame={pid_history_index}; action_frame={pid_history_action_index}; '
                    f'available_frames={len(pid_history_actions)}'
                )
            wind_index = OBSERVATION_VARIABLES.index('I_AverageWindSpeed')
            if obs.shape[0] <= wind_index:
                raise ValueError('PID history diagnostic requires the configured observation schema')
            pid_history_action = pid_history_actions[pid_history_action_index]
            pid_history_wind = float(pid_history_winds[pid_history_index])
            pid_history_wind_error = abs(float(obs[wind_index]) - pid_history_wind)
            if pid_history_wind_error > pid_history_max_wind_error:
                raise RuntimeError(
                    f'PID history diagnostic wind mismatch at frame={pid_history_index}: '
                    f'online={float(obs[wind_index]):.6f} reference={pid_history_wind:.6f} '
                    f'error={pid_history_wind_error:.6f} limit={pid_history_max_wind_error:.6f}'
                )
            if pid_history_observations is not None:
                reference_observation = pid_history_observations[pid_history_index]
                if obs.shape != reference_observation.shape:
                    raise ValueError('PID history diagnostic observation shape is incompatible with the active environment')
                observation_error = obs.astype(np.float32, copy=False) - reference_observation
                azimuth_index = OBSERVATION_VARIABLES.index('I_RotorAzimuth')
                observation_error[azimuth_index] = np.arctan2(
                    np.sin(observation_error[azimuth_index]),
                    np.cos(observation_error[azimuth_index]),
                )
                max_error_index = int(np.argmax(np.abs(observation_error)))
                pid_history_observation_metrics = {
                    'control_pid_history_observation_available': 1.0,
                    'control_pid_history_obs_max_abs_error': float(np.abs(observation_error[max_error_index])),
                    'control_pid_history_obs_max_error_index': float(max_error_index),
                    'control_pid_history_obs_max_error_value': float(observation_error[max_error_index]),
                    'control_pid_history_obs_max_reference_value': float(reference_observation[max_error_index]),
                }
                for feature_name, reference_value, error_value in zip(
                    OBSERVATION_VARIABLES,
                    reference_observation,
                    observation_error,
                ):
                    pid_history_observation_metrics[
                        f'control_pid_history_obs_reference_{feature_name}'
                    ] = float(reference_value)
                    pid_history_observation_metrics[
                        f'control_pid_history_obs_error_{feature_name}'
                    ] = float(error_value)
        if frozen_bc_actor is not None:
            with torch.inference_mode():
                bc_residual_action = torch.nan_to_num(
                    frozen_bc_actor(obs_tensor),
                    nan=0.0,
                    posinf=1.0,
                    neginf=-1.0,
                ).clamp(-1.0, 1.0)
                bc_baseline_residual = bc_residual_action.squeeze(0).cpu().numpy()
                bc_baseline_action = controller.action_real_tensor(
                    bc_residual_action,
                    obs_tensor,
                ).squeeze(0).cpu().numpy()
        if force_bc_baseline or force_pid_history:
            policy_action = np.zeros(env.action_dim, dtype=np.float32)
            actor_execution_valid = True
        else:
            action_norm = agent.select_action(
                obs_tensor,
                noise_std=noise_std,
                deterministic=noise_std <= 0.0,
            )
            raw_policy_action = action_norm.squeeze(0).detach().cpu().numpy()
            actor_execution_valid = bool(
                raw_policy_action.shape == (env.action_dim,)
                and np.isfinite(raw_policy_action).all()
            )
            policy_action = (
                raw_policy_action
                if actor_execution_valid
                else np.zeros(env.action_dim, dtype=np.float32)
            )
        if residual_scale_diagnostic is not None:
            policy_action = policy_action * residual_scale_diagnostic
        baseline_action = held_baseline_action
        actor_execution_allowed = (
            active_train
            and not force_baseline
            and not force_bc_baseline
            and not force_pid_history
        )
        actor_update_gate_open = (
            actor_optimizer_health_gate.allows_actor_update
            if actor_optimizer_health_gate is not None
            else (
                not critic_q_health_monitor.degraded
                and actor_q_health_gate is not None
                and actor_q_health_gate.allows_actor_update
            )
        )
        safe_action = controller.action_from_policy(
            policy_action,
            obs,
            allow_policy=actor_execution_allowed,
            baseline_action_real=baseline_action,
            fallback_action_real=(
                pid_history_action
                if force_pid_history
                else bc_baseline_action if force_bc_baseline else None
            ),
            baseline_execution_mode=PreSendExecutionMode.EXPLICIT_BASELINE_STAGE,
            actor_execution_valid=actor_execution_valid,
        )
        baseline_action = safe_action.baseline_action_real
        next_total_steps = total_steps + 1
        update_scheduled = (
            online_rl_enabled
            and not diagnostic_mode
            and active_train
            and stage in {'critic', 'actor'}
            and not updates_quarantined
            and len(replay_buffer) >= batch_size
            and next_total_steps % update_every_steps == 0
        )
        update_metrics_records: list[dict[str, float]] = []

        def update_during_udp_wait(_action_sent_monotonic_s: float) -> None:
            nonlocal update_count
            nonlocal actor_update_executed_count
            if not update_scheduled:
                return
            for _ in range(updates_per_step):
                batch = replay_buffer.sample(batch_size)
                training_round_index = lifecycle.completed_rounds + 1
                round_sampling_stats = _round_batch_source_stats(
                    batch_source_stats_by_round,
                    training_round_index,
                    replay_buffer.condition_ids,
                    replay_buffer.source_names,
                )
                source_sample_counts = batch['source_sample_counts'].tolist()
                source_quota_relaxed_by_source = batch['source_quota_relaxed_by_source'].tolist()
                condition_source_sample_counts = batch['condition_source_sample_counts'].numpy()
                condition_quota_relaxed_counts = batch['condition_quota_relaxed_count'].numpy()
                condition_source_quota_relaxed_counts = batch[
                    'condition_source_quota_relaxed_by_source'
                ].numpy()
                round_sampling_stats['batch_count'] += 1
                round_sampling_stats['sample_count'] += batch_size
                round_sampling_stats['quota_relaxed_count'] += int(
                    batch['source_quota_relaxed_count'].item()
                )
                for source_index, source_name in enumerate(replay_buffer.source_names):
                    round_sampling_stats['source_counts'][source_name] += int(
                        source_sample_counts[source_index]
                    )
                    round_sampling_stats['quota_relaxed_by_source'][source_name] += int(
                        source_quota_relaxed_by_source[source_index]
                    )
                for condition_index, condition_id in enumerate(replay_buffer.condition_ids):
                    condition_source_counts = round_sampling_stats['condition_sample_counts'][condition_id]
                    round_sampling_stats['condition_quota_relaxed_count'][condition_id] += int(
                        condition_quota_relaxed_counts[condition_index]
                    )
                    if int(condition_source_sample_counts[condition_index].sum()) > 0:
                        round_sampling_stats['condition_batch_counts'][condition_id] += 1
                    for source_index, source_name in enumerate(replay_buffer.source_names):
                        condition_source_counts[source_name] += int(
                            condition_source_sample_counts[condition_index, source_index]
                        )
                        round_sampling_stats['condition_quota_relaxed_by_source'][condition_id][source_name] += int(
                            condition_source_quota_relaxed_counts[condition_index, source_index]
                        )

                actor_update_blocked_execution_mode = (
                    safe_action.pre_send_execution_mode != PreSendExecutionMode.ACTOR
                )
                actor_update_execution_allowed = (
                    stage == 'actor' and not actor_update_blocked_execution_mode
                )

                def actor_update_preflight(metrics: dict[str, float]) -> bool:
                    gate_open_before_critic = (
                        actor_optimizer_health_gate.allows_actor_update
                        if actor_optimizer_health_gate is not None
                        else (
                            not critic_q_health_monitor.degraded
                            and actor_q_health_gate is not None
                            and actor_q_health_gate.allows_actor_update
                        )
                    )
                    critic_q_health_metrics = critic_q_health_monitor.observe(metrics)
                    metrics.update(critic_q_health_metrics)
                    if metrics.get('critic_q_health_failure_triggered', 0.0) > 0.0:
                        logger.save_summary(
                            'queue_critic_q_health_degraded.json',
                            {
                                'queue_critic_q_health_failure_step': float(total_steps),
                                'queue_critic_q_health_failure_condition_id': float(active_condition_id),
                                'queue_critic_q_health_failure_condition_instance_id': float(
                                    active_condition_instance_id
                                ),
                                **critic_q_health_metrics,
                            },
                        )
                        print(
                            '[Queue-QHealth] persistent critic guard detected; '
                            'continuing queue with actor frozen and guarded critic updates skipped',
                            flush=True,
                        )
                    if actor_optimizer_health_gate is not None:
                        assert fixed_offline_health_obs is not None
                        assert fixed_offline_health_actions is not None
                        fixed_offline_batch_healthy = agent.fixed_offline_batch_is_healthy(
                            fixed_offline_health_obs,
                            fixed_offline_health_actions,
                        )
                        gate_metrics = actor_optimizer_health_gate.observe(
                            metrics,
                            fixed_offline_batch_healthy=fixed_offline_batch_healthy,
                        )
                        metrics.update(gate_metrics)
                        gate_open_after_critic = actor_optimizer_health_gate.allows_actor_update
                    else:
                        assert actor_q_health_gate is not None
                        gate_metrics = actor_q_health_gate.observe(metrics)
                        metrics.update(gate_metrics)
                        gate_open_after_critic = (
                            not critic_q_health_monitor.degraded
                            and actor_q_health_gate.allows_actor_update
                        )
                    metrics['actor_update_gate_open_before_critic'] = float(
                        gate_open_before_critic
                    )
                    metrics['actor_update_gate_open_after_critic'] = float(
                        gate_open_after_critic
                    )
                    # Preserve the legacy field for downstream reports; it is an
                    # update-time snapshot, not evidence of a separate gate.
                    metrics['actor_q_health_gate_open_before_update'] = float(
                        gate_open_before_critic
                    )
                    metrics['actor_update_blocked_q_health'] = float(
                        stage == 'actor' and not gate_open_after_critic
                    )
                    metrics['actor_update_blocked_control_health'] = float(
                        stage == 'actor' and actor_update_blocked_execution_mode
                    )
                    metrics['actor_update_blocked_execution_mode'] = float(
                        stage == 'actor' and actor_update_blocked_execution_mode
                    )
                    metrics['actor_update_requested_after_critic'] = float(
                        actor_update_execution_allowed and gate_open_after_critic
                    )
                    for source_index, source_name in enumerate(replay_buffer.source_names):
                        sample_count = int(source_sample_counts[source_index])
                        metrics[f'queue_batch_source_{source_name}_count'] = float(sample_count)
                        metrics[f'queue_batch_source_{source_name}_fraction'] = float(
                            sample_count / batch_size
                        )
                        relaxed_count = int(source_quota_relaxed_by_source[source_index])
                        metrics[f'queue_batch_source_quota_relaxed_{source_name}_count'] = float(
                            relaxed_count
                        )
                    quota_relaxed_count = int(batch['source_quota_relaxed_count'].item())
                    metrics['queue_batch_source_quota_relaxed_count'] = float(quota_relaxed_count)
                    if int(condition_quota_relaxed_counts.sum()) != quota_relaxed_count:
                        raise RuntimeError('Replay quota-relaxed counts do not close across conditions')
                    if int(condition_source_quota_relaxed_counts.sum()) != quota_relaxed_count:
                        raise RuntimeError('Replay quota-relaxed source counts do not close across conditions')
                    for condition_index, condition_id in enumerate(replay_buffer.condition_ids):
                        if int(condition_source_sample_counts[condition_index].sum()) > 0:
                            metrics[f'queue_batch_condition_{condition_id}_sample_count'] = float(
                                condition_source_sample_counts[condition_index].sum()
                            )
                        metrics[
                            f'queue_batch_condition_{condition_id}_quota_relaxed_count'
                        ] = float(condition_quota_relaxed_counts[condition_index])
                    metrics['queue_stage_code'] = _queue_stage_code(stage)
                    return gate_open_after_critic

                if config['algorithm'].lower() == 'td3' and getattr(agent, 'td3_bc_enabled', False) and bc_iterator is not None:
                    bc_obs, _bc_actions_real, bc_actions_target = _next_bc_batch(bc_iterator, controller, device)
                    metrics = agent.update(
                        SimpleNamespace(**batch),
                        bc_obs=bc_obs,
                        bc_actions=bc_actions_target,
                        bc_weight_override=current_td3_bc_weight,
                        update_actor=actor_update_execution_allowed,
                        actor_update_preflight=actor_update_preflight,
                    )
                else:
                    metrics = agent.update(
                        SimpleNamespace(**batch),
                        update_actor=actor_update_execution_allowed,
                        actor_update_preflight=actor_update_preflight,
                    )
                _record_queue_condition_update_stats(active_condition_stats, metrics)
                update_metrics_records.append(metrics)
                update_count += 1
                actor_update_executed_count += int('actor_loss' in metrics)

        def sent_side_work(sent_monotonic_s: float) -> None:
            # Enqueue immediately after a successful send, before optimizer
            # work or waiting for the next physical frame. Never change action.
            if physical_load_shadow is not None and load_predictor_payload is not None:
                submit_physical_sent_frame(physical_load_shadow, env,
                    payload=load_predictor_payload, condition_id=active_condition_id,
                    condition_instance_id=active_condition_instance_id,
                    sent_action_real=safe_action.action_real,
                    actor_policy_version=actor_policy_version_for_step,
                    actor_policy_sha256='', sent_monotonic_s=sent_monotonic_s)
            if update_scheduled:
                update_during_udp_wait(sent_monotonic_s)

        step_result = _step_environment(
            env,
            safe_action.action_norm,
            safe_action.action_real,
            on_action_sent=sent_side_work if update_scheduled or physical_load_shadow is not None else None,
        )
        step_info = step_result.info
        train_timeout_detected = step_info.get('udp_inter_frame_timeout', 0.0) > 0.0
        train_timeout_count_since_log += int(train_timeout_detected)
        transition_store_eligible = step_info.get('transition_store_eligible', 0.0) > 0.0
        if transition_store_eligible:
            _apply_dual_step_penalty(
                dual_constraint_controller,
                safe_action,
                step_result,
                step_info,
            )
        if dual_constraint_controller is not None and transition_store_eligible:
            _observe_dual_constraint_step(condition_constraint_accumulator, step_info)
        active_condition_stats['queue_condition_raw_frame_count'] += 1.0
        active_condition_stats['queue_condition_train_frame_count'] += float(active_train)
        active_condition_stats['queue_condition_eligible_transition_count'] += float(transition_store_eligible)
        step_info.update(
            {
                'udp_sim_time': float(step_info.get('udp_sim_time', math.nan)),
                'udp_next_sim_time': float(step_info.get('udp_next_sim_time', math.nan)),
            }
        )
        safety_stop = controller.observe_step(float(step_info.get('power', 0.0))) if transition_store_eligible else False
        if physical_load_shadow is not None and load_predictor_payload is not None:
            complete_physical_step(physical_load_shadow, env, payload=load_predictor_payload,
                                  info=step_info, unsafe_stop=safety_stop)
            step_info.update(physical_load_shadow.diagnostics)
        active_condition_round_index = (
            int(manifest_slots[manifest_slot_index].round_id)
            if manifest_slots is not None and manifest_slot_index < len(manifest_slots)
            else lifecycle.completed_rounds + 1
        )
        step_info.update(
            {
                'queue_condition_round_index': float(active_condition_round_index),
                'queue_condition_id': float(active_condition_id),
                'queue_condition_instance_id': float(active_condition_instance_id),
                'control_ood_score': safe_action.ood_score,
                'control_ood_feature_index': float(safe_action.ood_feature_index),
                'control_ood_feature_value': safe_action.ood_feature_value,
                'control_ood_feature_median': safe_action.ood_feature_median,
                'control_ood_feature_iqr': safe_action.ood_feature_iqr,
                'control_force_baseline': float(force_baseline),
                'control_force_bc_baseline': float(force_bc_baseline),
                'control_force_pid_history': float(force_pid_history),
                'control_baseline_hold_initial': float(hold_initial_baseline),
                'control_actor_policy_allowed': float(actor_execution_allowed),
                'control_actor_execution_allowed': float(actor_execution_allowed),
                'control_actor_execution_valid': float(actor_execution_valid),
                'control_actor_update_gate_open': float(actor_update_gate_open),
                'control_actor_policy_blocked_q_health': 0.0,
                'control_actor_update_blocked_q_health': float(
                    stage == 'actor' and not actor_update_gate_open
                ),
                'control_residual_scale_diagnostic': float(residual_scale_diagnostic is not None),
                'control_residual_scale': 1.0 if residual_scale_diagnostic is None else residual_scale_diagnostic,
                'control_baseline_action_torque': float(baseline_action[0]),
                'control_baseline_action_pitch': float(baseline_action[1]),
                'control_bc_baseline_action_torque': (
                    0.0 if bc_baseline_action is None else float(bc_baseline_action[0])
                ),
                'control_bc_baseline_action_pitch': (
                    0.0 if bc_baseline_action is None else float(bc_baseline_action[1])
                ),
                'control_bc_baseline_residual_torque': (
                    0.0 if bc_baseline_residual is None else float(bc_baseline_residual[0])
                ),
                'control_bc_baseline_residual_pitch': (
                    0.0 if bc_baseline_residual is None else float(bc_baseline_residual[1])
                ),
                'control_pid_history_index': -1.0 if pid_history_action is None else float(pid_history_index),
                'control_pid_history_action_index': (
                    -1.0 if pid_history_action_index is None else float(pid_history_action_index)
                ),
                'control_pid_history_action_offset_steps': float(pid_history_action_offset_steps),
                'control_pid_history_action_torque': (
                    0.0 if pid_history_action is None else float(pid_history_action[0])
                ),
                'control_pid_history_action_pitch': (
                    0.0 if pid_history_action is None else float(pid_history_action[1])
                ),
                'control_pid_history_wind': 0.0 if pid_history_wind is None else pid_history_wind,
                'control_pid_history_wind_error': (
                    0.0 if pid_history_wind_error is None else pid_history_wind_error
                ),
                **pid_history_observation_metrics,
                **case_identity_metrics,
                'control_fallback_bc_baseline': float(
                    force_bc_baseline and bc_baseline_action is not None and safe_action.fallback_active
                ),
                'control_fallback_pid_history': float(
                    pid_history_action is not None and safe_action.fallback_active
                ),
                'control_fallback_active': float(safe_action.fallback_active),
                'control_low_rotor_speed_fallback_active': float(
                    safe_action.low_rotor_speed_fallback_active
                ),
                'control_low_rotor_speed_torque_unload_active': float(
                    safe_action.low_rotor_speed_torque_unload_active
                ),
                'control_above_rated_initial_pitch_floor_active': float(
                    safe_action.above_rated_initial_pitch_floor_active
                ),
                'control_above_rated_initial_pitch_floor_applied': float(
                    safe_action.above_rated_initial_pitch_floor_applied
                ),
                'control_negative_power_steps': float(controller.negative_power_steps),
                'control_negative_power_patience': float(controller.negative_power_patience),
                'control_low_power_steps': float(controller.low_power_steps),
                'control_low_power_patience': float(controller.low_power_patience),
                'control_action_torque': float(safe_action.action_real[0]),
                'control_action_pitch': float(safe_action.action_real[1]),
                'control_residual_torque': float(safe_action.residual_norm[0]),
                'control_residual_pitch': float(safe_action.residual_norm[1]),
                'control_policy_action_torque': float(safe_action.policy_action_norm[0]),
                'control_policy_action_pitch': float(safe_action.policy_action_norm[1]),
                'control_executed_action_torque': float(safe_action.executed_action_norm[0]),
                'control_executed_action_pitch': float(safe_action.executed_action_norm[1]),
                'control_action_projection_applied': float(safe_action.action_projection_applied),
                'control_action_rate_limited': float(safe_action.action_rate_limited),
                **safe_action_metrics(
                    safe_action,
                    sent_action_real=np.asarray(
                        [
                            step_info.get('action_torque', safe_action.action_real[0]),
                            step_info.get('action_pitch', safe_action.action_real[1]),
                        ],
                        dtype=np.float64,
                    ),
                ),
                'control_unsafe_stop': float(safety_stop),
                **post_step_outcome_metrics(step_info, unsafe_stop=safety_stop),
                'queue_stage_code': _queue_stage_code(stage),
                'queue_all_conditions_covered': float(all_conditions_covered),
                'queue_updates_quarantined_after_skip': float(updates_quarantined_after_skip()),
                'queue_updates_quarantined_after_retry': float(updates_quarantined_after_retry()),
                'queue_updates_quarantined_after_failure': float(updates_quarantined),
                'queue_training_valid': float(queue_training_valid()),
                'queue_train_timeouts_since_last_log': float(train_timeout_count_since_log),
            }
        )
        transition_finite = _is_finite_transition(
            obs,
            safe_action.executed_action_norm,
            step_result.reward,
            step_result.obs,
        )
        if (
            load_predictor_shadow is not None
            and load_predictor_payload is not None
            and transition_store_eligible
            and transition_finite
            and not train_timeout_detected
            and step_info.get('env_invalid_prediction', 0.0) <= 0.0
            and step_info.get('env_invalid_reward', 0.0) <= 0.0
        ):
            pid_peaks = env.reward_fn.pid_blade_root_peak_reference(active_condition_id)
            state_encoding = np.zeros(8, dtype=np.float64)
            optimization_state_code = int(safe_action.optimization_state_code)
            if 0 <= optimization_state_code < len(OPTIMIZATION_STATE_ENCODING_ORDER):
                state_encoding[optimization_state_code] = 1.0
            control_region_code = int(safe_action.control_region_code)
            if 0 <= control_region_code < 3:
                state_encoding[5 + control_region_code] = 1.0
            shadow_result = load_predictor_shadow.observe(
                condition_id=active_condition_id,
                condition_instance_id=active_condition_instance_id,
                payload=load_predictor_payload,
                sent_action_real=np.asarray([
                    step_info.get('action_torque', safe_action.action_real[0]),
                    step_info.get('action_pitch', safe_action.action_real[1]),
                ], dtype=np.float64),
                projected_candidate_action_real=safe_action.projected_action_real,
                state_encoding=state_encoding,
                pid_peak_abs_nm=np.asarray(pid_peaks, dtype=np.float64),
                optimization_state=(
                    OPTIMIZATION_STATE_ENCODING_ORDER[optimization_state_code].value
                    if 0 <= optimization_state_code < len(OPTIMIZATION_STATE_ENCODING_ORDER)
                    else 'unknown'
                ),
                execution_mode=safe_action.pre_send_execution_mode.value,
                outcome_tag='unsafe_stop' if safety_stop else 'complete',
                actor_policy_version=actor_policy_version_for_step,
                actor_policy_sha256=str(load_predictor_config.get('actor_policy_checkpoint_sha256', '')),
                predictor_artifact_sha256=str(
                    load_predictor_artifact_summary.get('load_predictor_shadow_model_sha256', '')
                ),
            )
            step_info.update(shadow_result.diagnostics)
            active_condition_stats['queue_condition_load_predictor_observed_frame_count'] += 1.0
            active_condition_stats['queue_condition_load_predictor_pending_sample_count'] += float(
                shadow_result.pending_sample_id is not None
            )
            active_condition_stats['queue_condition_load_predictor_completed_record_count'] += float(
                len(shadow_result.completed_records)
            )
            active_condition_stats['queue_condition_load_predictor_audit_record_count'] += float(
                shadow_result.completed_audit_record_count
            )
            active_condition_stats['queue_condition_load_predictor_calibration_eligible_record_count'] += float(
                shadow_result.completed_audit_calibration_eligible_count
            )
            active_condition_stats['queue_condition_load_predictor_invalid_prediction_count'] += float(
                shadow_result.diagnostics.get('load_predictor_shadow_prediction_invalid', 0.0)
            )
        _record_queue_condition_step_stats(
            active_condition_stats,
            step_info,
            env,
            condition_id=active_condition_id,
            transition_eligible=transition_store_eligible,
        )
        if active_train and not train_timeout_detected:
            active_train_phase_elapsed_latest_s = _queue_phase_elapsed_s(step_info)
        should_store = (
            transition_store_eligible
            and transition_finite
            and not diagnostic_mode
            and _should_store_transition(step_info, train_cfg)
        )
        replay_source = (
            ONLINE_BLADED_SOURCE
            if execution_replay is None
            else replay_source_for_execution_mode(safe_action.pre_send_execution_mode)
        )
        should_store = should_store and replay_source is not None
        if should_store:
            replay_buffer.add(
                obs,
                safe_action.executed_action_norm,
                step_result.reward,
                step_result.obs,
                False,
                condition_id=active_condition_id,
                condition_instance_id=active_condition_instance_id,
                phase_elapsed_s=float(step_info.get('udp_phase_elapsed_s', np.nan)),
                source=replay_source,
            )
            total_stored += 1
            active_condition_stats['queue_condition_stored_transition_count'] += 1.0
            active_condition_stats[f'queue_condition_stored_source_{replay_source}_count'] += 1.0
        else:
            total_rejected += 1
            active_condition_stats['queue_condition_rejected_transition_count'] += 1.0
            rejection_key = _queue_rejection_metric_key(step_info, transition_finite)
            active_condition_stats[rejection_key] += 1.0
            if replay_source is not None:
                active_condition_stats[f'queue_condition_rejected_source_{replay_source}_count'] += 1.0

        total_steps = next_total_steps
        logger.log_metrics('queue_train_step', total_steps, step_info | {'reward_total': step_result.reward})
        for update_metrics in update_metrics_records:
            logger.log_metrics('queue_train_update', total_steps, update_metrics)
        if (
            not diagnostic_mode
            and active_train
            and condition_checkpoint_interval_s is not None
            and active_train_phase_elapsed_start_s is not None
            and not train_timeout_detected
            and step_info.get('condition_boundary', 0.0) <= 0.0
        ):
            if active_train_phase_elapsed_latest_s is None:
                raise RuntimeError('Queue TRAIN condition is missing its latest phase elapsed time')
            condition_elapsed_s = active_train_phase_elapsed_latest_s - active_train_phase_elapsed_start_s
            while next_condition_checkpoint_index < condition_checkpoint_count:
                target_elapsed_s = condition_checkpoint_interval_s * next_condition_checkpoint_index
                if condition_elapsed_s + 1.0e-9 < target_elapsed_s:
                    break
                save_condition_checkpoint(
                    checkpoint_index=next_condition_checkpoint_index,
                    target_elapsed_s=target_elapsed_s,
                    actual_elapsed_s=condition_elapsed_s,
                )
                next_condition_checkpoint_index += 1
        if force_pid_history:
            pid_history_index += 1
        if (
            total_steps == 1
            or total_steps % step_log_every == 0
            or step_info.get('condition_boundary', 0.0) > 0.0
            or train_timeout_detected
        ):
            print(
                f'[Queue] step={total_steps} stage={stage} condition={active_condition_id} '
                f'stored={total_stored} rejected={total_rejected} reward={step_result.reward:.4f} '
                f'power={step_info.get("power", 0.0):.2f} '
                f'ood={step_info.get("control_ood_score", 0.0):.3f} '
                f'ood_feature={safe_action.ood_feature_name} '
                f'force_baseline={int(force_baseline)} '
                f'force_bc_baseline={int(force_bc_baseline)} '
                f'force_pid_history={int(force_pid_history)} '
                f'fallback={step_info.get("control_fallback_active", 0.0):.0f} '
                f'negative_power={controller.negative_power_steps}/{controller.negative_power_patience} '
                f'train_timeouts_window={train_timeout_count_since_log}',
                flush=True,
            )
            train_timeout_count_since_log = 0

        if safety_stop:
            save_abort_diagnostics(1.0, step_info)
            raise RuntimeError(
                f'Queue training stopped by safety controller in condition={active_condition_id} '
                f'instance={active_condition_instance_id}: power={step_info.get("power", 0.0):.6f} '
                f'negative_power_steps={controller.negative_power_steps}/{controller.negative_power_patience} '
                f'low_power_steps={controller.low_power_steps}/{controller.low_power_patience} '
                f'ood={safe_action.ood_score:.6f} fallback={int(safe_action.fallback_active)}'
            )
        if train_timeout_detected:
            if skip_failed_condition_on_train_timeout and active_train:
                logger.flush()
                recovery_method = getattr(env, 'recover_after_train_timeout', None)
                if not callable(recovery_method):
                    raise RuntimeError(
                        'queue.skip_failed_condition_on_train_timeout requires '
                        'WindTurbineEnv.recover_after_train_timeout()'
                    )
                try:
                    recovered_obs, recovery_info = recovery_method()
                    recovered_condition_id, recovered_condition_instance_id, recovered_train = _queue_context_from_info(
                        recovery_info,
                        expected_condition_ids=lifecycle.condition_ids,
                    )
                except (RuntimeError, TimeoutError) as exc:
                    save_abort_diagnostics(3.0, step_info)
                    raise TimeoutError(
                        f'Queue training could not recover after timeout in condition={active_condition_id} '
                        f'instance={active_condition_instance_id}'
                    ) from exc

                condition_changed = recovery_info.get('udp_timeout_recovery_condition_changed', 0.0) > 0.0
                recovery_wait_ms = float(recovery_info.get('udp_timeout_recovery_wait_ms', 0.0))
                logger.log_metrics(
                    'queue_timeout_recovery',
                    total_steps,
                    {
                        'queue_timeout_recovery_condition_changed': float(condition_changed),
                        'queue_timeout_recovery_wait_ms': recovery_wait_ms,
                        'queue_timeout_recovery_condition_id': float(recovered_condition_id),
                        'queue_timeout_recovery_condition_instance_id': float(recovered_condition_instance_id),
                    },
                )

                if not condition_changed:
                    if (
                        recovered_condition_id != active_condition_id
                        or recovered_condition_instance_id != active_condition_instance_id
                        or not recovered_train
                    ):
                        save_abort_diagnostics(3.0, step_info)
                        raise RuntimeError(
                            'Queue timeout recovery reported a resumed condition with incompatible identity or phase'
                        )
                    active_train_phase_elapsed_latest_s = _queue_phase_elapsed_s(recovery_info)
                    obs = recovered_obs
                    print(
                        f'[Queue] timeout recovered same instance condition={active_condition_id} '
                        f'instance={active_condition_instance_id} '
                        f'wait_ms={recovery_wait_ms:.1f}',
                        flush=True,
                    )
                    continue

                if active_train_phase_elapsed_start_s is None or active_train_phase_elapsed_latest_s is None:
                    raise RuntimeError('Queue TRAIN timeout skip is missing phase elapsed metadata from the last valid frame')
                skipped_train_elapsed_s = max(
                    0.0,
                    active_train_phase_elapsed_latest_s - active_train_phase_elapsed_start_s,
                )
                if recovered_condition_id == active_condition_id:
                    recovered_obs, recovery_info, identity_probe_observations = (
                        reconcile_retried_condition_to_verified_physical_skip(
                            recovered_obs,
                            recovery_info,
                            trigger_code=1.0,
                        )
                    )
                    if identity_probe_observations is not None:
                        if skip_failed_condition_and_activate(
                            recovered_obs,
                            recovery_info,
                            train_elapsed_s=skipped_train_elapsed_s,
                            recovery_wait_ms=recovery_wait_ms,
                            trigger_code=3.0,
                            identity_probe_observations=identity_probe_observations,
                        ):
                            break
                        continue
                    retry_failed_condition_and_activate(
                        recovered_obs,
                        recovery_info,
                        train_elapsed_s=skipped_train_elapsed_s,
                        recovery_wait_ms=recovery_wait_ms,
                        trigger_code=1.0,
                        abort_after_retry=abort_on_retried_condition_attempt,
                    )
                    continue
                if skip_failed_condition_and_activate(
                    recovered_obs,
                    recovery_info,
                    train_elapsed_s=skipped_train_elapsed_s,
                    recovery_wait_ms=recovery_wait_ms,
                    trigger_code=1.0,
                ):
                    break
                continue
            save_abort_diagnostics(2.0, step_info)
            raise TimeoutError(
                f'Queue training timed out while waiting for condition={active_condition_id} '
                f'instance={active_condition_instance_id}'
            )

        obs = step_result.obs
        if step_info.get('condition_boundary', 0.0) <= 0.0:
            continue

        case_identity_final_metrics: dict[str, float] = {}
        case_identity_summary: dict[str, Any] = {}
        if strict_case_identity and active_train:
            assert isinstance(case_identity_validator, QueueStrictCaseIdentityValidator)
            next_condition_id = int(step_info.get('udp_next_condition_id', -1.0))
            try:
                if physical_operator_identity:
                    case_identity_final_metrics, case_identity_summary = _finish_physical_operator_identity(
                        case_identity_validator, env, manifest_slots[manifest_slot_index], step_result.obs,
                    )
                elif next_condition_id == active_condition_id:
                    if full_manifest_identity:
                        _validate_strict_manifest_identity(manifest_slots[manifest_slot_index],
                            env.current_queue_metadata, context='strict endpoint')
                    sim_time_s, wind_speed_mps, runtime_case_fingerprint = _strict_queue_identity_sample(env)
                    case_identity_validator.observe(
                        active_condition_id,
                        active_condition_instance_id,
                        step_result.obs,
                        sim_time_s=sim_time_s,
                        wind_speed_mps=wind_speed_mps,
                        runtime_case_fingerprint=runtime_case_fingerprint,
                    )
                if not physical_operator_identity:
                    case_identity_final_metrics, case_identity_summary = case_identity_validator.finish_condition(
                        active_condition_id,
                        active_condition_instance_id,
                    )
            except QueueCaseIdentityMismatch as exc:
                mismatch_metrics = {
                    **exc.metrics,
                    'queue_case_identity_mismatch': 1.0,
                    'queue_case_identity_total_steps': float(total_steps),
                }
                logger.save_summary(
                    f'queue_case_identity_mismatch_condition_{active_condition_id:02d}_'
                    f'instance_{active_condition_instance_id:03d}.json',
                    mismatch_metrics,
                )
                logger.log_metrics('queue_case_identity_mismatch', total_steps, mismatch_metrics)
                logger.flush()
                raise RuntimeError(str(exc)) from exc
            logger.save_summary(
                f'queue_case_identity_condition_{active_condition_id:02d}_'
                f'instance_{active_condition_instance_id:03d}.json',
                case_identity_summary,
            )
            logger.log_metrics('queue_case_identity_final', total_steps, case_identity_final_metrics)

        if active_train:
            if active_train_phase_elapsed_start_s is None:
                raise RuntimeError('Queue TRAIN condition is missing its phase elapsed start time')
            phase_elapsed_end_s = _queue_phase_elapsed_s(step_info)
            try:
                train_elapsed_s = _validate_queue_train_duration(
                    queue_cfg,
                    condition_id=active_condition_id,
                    condition_instance_id=active_condition_instance_id,
                    phase_elapsed_start_s=active_train_phase_elapsed_start_s,
                    phase_elapsed_end_s=phase_elapsed_end_s,
                )
            except RuntimeError as exc:
                next_condition_instance_id = int(step_info.get('udp_next_condition_instance_id', -1.0))
                next_instance_arrived = (
                    next_condition_instance_id >= 0
                    and next_condition_instance_id != active_condition_instance_id
                )
                if (
                    not skip_failed_condition_on_train_timeout
                    or not next_instance_arrived
                    or 'Queue TRAIN duration is too short' not in str(exc)
                ):
                    raise
                try:
                    recovered_obs, recovery_info = env.start_pending_condition()
                except RuntimeError as recovery_exc:
                    save_abort_diagnostics(3.0, step_info)
                    raise RuntimeError(
                        f'Queue training could not activate the next instance after a short TRAIN condition='
                        f'{active_condition_id} instance={active_condition_instance_id}'
                    ) from recovery_exc
                skipped_train_elapsed_s = max(0.0, phase_elapsed_end_s - active_train_phase_elapsed_start_s)
                recovered_condition_id, _recovered_condition_instance_id, _recovered_train = _queue_context_from_info(
                    recovery_info,
                    expected_condition_ids=lifecycle.condition_ids,
                )
                if recovered_condition_id == active_condition_id:
                    recovered_obs, recovery_info, identity_probe_observations = (
                        reconcile_retried_condition_to_verified_physical_skip(
                            recovered_obs,
                            recovery_info,
                            trigger_code=2.0,
                        )
                    )
                    if identity_probe_observations is not None:
                        if skip_failed_condition_and_activate(
                            recovered_obs,
                            recovery_info,
                            train_elapsed_s=skipped_train_elapsed_s,
                            recovery_wait_ms=0.0,
                            trigger_code=3.0,
                            identity_probe_observations=identity_probe_observations,
                        ):
                            break
                        continue
                    retry_failed_condition_and_activate(
                        recovered_obs,
                        recovery_info,
                        train_elapsed_s=skipped_train_elapsed_s,
                        recovery_wait_ms=0.0,
                        trigger_code=2.0,
                        abort_after_retry=abort_on_retried_condition_attempt,
                    )
                    continue
                if skip_failed_condition_and_activate(
                    recovered_obs,
                    recovery_info,
                    train_elapsed_s=skipped_train_elapsed_s,
                    recovery_wait_ms=0.0,
                    trigger_code=2.0,
                ):
                    break
                continue
            if not diagnostic_mode and condition_checkpoint_interval_s is not None:
                while next_condition_checkpoint_index <= condition_checkpoint_count:
                    save_condition_checkpoint(
                        checkpoint_index=next_condition_checkpoint_index,
                        target_elapsed_s=condition_checkpoint_interval_s * next_condition_checkpoint_index,
                        actual_elapsed_s=train_elapsed_s,
                    )
                    next_condition_checkpoint_index += 1
            completed_condition_round_index = (
                int(manifest_slots[manifest_slot_index].round_id)
                if manifest_slots is not None and manifest_slot_index < len(manifest_slots)
                else lifecycle.completed_rounds + 1
            )
            condition_identity_status = (
                'strict_case_matched'
                if case_identity_final_metrics.get('queue_case_identity_strict_case_matched', 0.0) > 0.5
                else 'reference_matched_only'
                if case_identity_validator is not None
                else 'not_configured'
            )
            if case_identity_summary.get('identity_status') == 'development_unverified':
                condition_identity_status = 'development_unverified'
            predictor_identity_status = condition_identity_status
            if load_predictor_shadow is not None and (
                not strict_case_identity or condition_identity_status != 'strict_case_matched'
            ):
                predictor_identity_status = 'reference_matched_only'
            finish_queue_load_predictor_condition(predictor_identity_status, case_identity_summary)
            _finalize_queue_condition_step_stats(
                active_condition_stats,
                condition_id=active_condition_id,
            )
            advance_manifest_slot_after_active_condition()
            dual_constraint_condition_metrics = finalize_condition_constraints(complete=True)
            completion = lifecycle.complete_condition(active_condition_id, active_condition_instance_id)
            condition_source_counts = replay_buffer.condition_source_counts()[completion.condition_id]
            logger.log_metrics(
                'queue_condition',
                total_steps,
                {
                    'condition_id': float(completion.condition_id),
                    'condition_instance_id': float(completion.condition_instance_id),
                    'queue_condition_round_index': float(completed_condition_round_index),
                    'queue_completed_round': float(completion.completed_round),
                    'queue_completed_rounds': float(completion.completed_rounds),
                    'queue_conditions_completed_in_round': float(completion.conditions_completed_in_round),
                    'queue_condition_replay_count': float(replay_buffer.condition_counts()[completion.condition_id]),
                    'queue_condition_replay_online_bladed_count': float(
                        condition_source_counts[ONLINE_BLADED_SOURCE]
                    ),
                    **active_condition_stats,
                    **_batch_source_metrics_for_condition(
                        batch_source_stats_by_round,
                        completed_condition_round_index,
                        completion.condition_id,
                        replay_buffer.source_names,
                    ),
                    'queue_train_elapsed_s': train_elapsed_s,
                    **case_identity_final_metrics,
                    **dual_constraint_condition_metrics,
                },
                record_metadata=queue_condition_identity_metadata(
                    condition_identity_status,
                    case_identity_summary,
                ),
            )
            if completion.completed_round:
                finalize_completed_round(completion)
                if lifecycle.finished:
                    break

        obs, condition_info = env.start_pending_condition()
        controller.reset_condition(obs, previous_action_real=env.last_sent_action)
        held_baseline_action = controller.baseline_action_for_obs(obs) if hold_initial_baseline else None
        pid_history_index = 0
        active_condition_id, active_condition_instance_id, active_train = _queue_context_from_info(
            condition_info,
            expected_condition_ids=lifecycle.condition_ids,
        )
        active_train_phase_elapsed_start_s = _queue_phase_elapsed_s(condition_info) if active_train else None
        active_train_phase_elapsed_latest_s = active_train_phase_elapsed_start_s
        active_condition_stats = _queue_condition_stats()
        condition_constraint_accumulator = (
            None if dual_constraint_controller is None else ConditionConstraintAccumulator()
        )
        if load_predictor_shadow is not None:
            load_predictor_shadow.reset_condition(active_condition_id, active_condition_instance_id)
        if physical_load_shadow is not None:
            physical_load_shadow.reset_condition(active_condition_id, active_condition_instance_id)
        next_condition_checkpoint_index = 1
        if case_identity_validator is not None:
            case_identity_validator.reset_condition(active_condition_id, active_condition_instance_id)
        validate_active_manifest_slot('normal condition activation')

    if (
        not diagnostic_mode
        and bool(offline_eval_cfg.get('enabled', False))
        and bool(offline_eval_cfg.get('run_after_training', True))
        and queue_training_valid()
    ):
        if offline_eval_dataset is None:
            raise ValueError('offline_eval requires a configured validation dataset')
        evaluation = evaluate_offline_policy(
            agent,
            offline_eval_dataset,
            controller,
            device,
            batch_size=int(offline_eval_cfg.get('batch_size', 1024)),
        )
        write_offline_policy_evaluation(logger.output_dir / 'offline_eval_final', evaluation)
        logger.save_summary('offline_eval_final_summary.json', evaluation.summary)
        logger.log_metrics('offline_eval_final', total_steps, evaluation.summary)

    final_source_counts = replay_buffer.condition_source_counts()
    final_summary = {
        'queue_completed_rounds': float(lifecycle.completed_rounds),
        'queue_total_rounds': float(lifecycle.total_rounds),
        'queue_total_steps': float(total_steps),
        'queue_total_stored_transitions': float(total_stored),
        'queue_total_rejected_transitions': float(total_rejected),
        'queue_update_count': float(update_count),
        'queue_skipped_condition_count': float(total_skipped_conditions),
        'queue_retried_condition_attempt_count': float(total_retried_condition_attempts),
        'queue_require_zero_skipped_conditions_for_updates': float(
            require_zero_skipped_conditions_for_updates
        ),
        'queue_require_zero_retried_condition_attempts_for_updates': float(
            require_zero_retried_condition_attempts_for_updates
        ),
        'queue_abort_on_retried_condition_attempt': float(abort_on_retried_condition_attempt),
        'queue_updates_quarantined_after_skip': float(updates_quarantined_after_skip()),
        'queue_updates_quarantined_after_retry': float(updates_quarantined_after_retry()),
        'queue_updates_quarantined_after_failure': float(updates_quarantined_after_failure()),
        'queue_training_valid': float(queue_training_valid()),
        'queue_all_conditions_covered': float(replay_buffer.all_conditions_covered(min_valid_transitions)),
        'queue_model_candidate_valid': float(
            queue_training_valid() and not critic_q_health_monitor.degraded
        ),
        'queue_load_predictor_shadow_enabled': float(load_predictor_shadow is not None),
        'queue_load_predictor_model_loaded': float(
            load_predictor_artifact_summary.get('load_predictor_shadow_model_loaded', 0.0)
        ),
        'queue_load_predictor_shadow_audit_records_written': float(load_predictor_audit_records_written),
        'queue_load_predictor_shadow_audit_bytes_written': float(load_predictor_audit_bytes_written),
        'queue_load_predictor_shadow_audit_max_records': float(load_predictor_audit_max_records),
        'queue_load_predictor_shadow_audit_max_record_bytes': float(load_predictor_audit_max_record_bytes),
        'queue_load_predictor_shadow_audit_max_total_bytes': float(load_predictor_audit_max_total_bytes),
        'queue_load_predictor_shadow_audit_file_present': float(load_predictor_audit_path.is_file()),
        'queue_physical_load_shadow_enabled': float(physical_load_shadow is not None),
        'queue_physical_load_shadow_action_changed': 0.0,
    }
    final_summary.update(critic_q_health_summary())
    for condition_id, source_counts in final_source_counts.items():
        final_summary[f'queue_condition_{condition_id}_online_bladed_replay_count'] = float(
            source_counts[ONLINE_BLADED_SOURCE]
        )
    return final_summary


def _get_eval_checkpoint_score(
    eval_summary: dict[str, float],
    selection_metric: str,
    expected_valid_episodes: int,
) -> float | None:
    try:
        score = float(eval_summary[selection_metric])
        invalid_fraction = float(eval_summary.get('episode_invalid_mean', 0.0))
        valid_episodes = float(eval_summary.get('reward_valid_episode_count', 0.0))
    except (KeyError, TypeError, ValueError):
        return None
    if (
        not math.isfinite(score)
        or not math.isfinite(invalid_fraction)
        or invalid_fraction > 0.0
        or valid_episodes < float(expected_valid_episodes)
    ):
        return None
    return score


def _build_bc_dataloader(config: dict) -> tuple[OfflineBCDataset, DataLoader]:
    bc_config = config['bc']
    env_config = config['env']
    dataset = OfflineBCDataset(
        root_dir=config['data']['root_dir'],
        split=bc_config.get('split', 'train'),
        max_files=config['data'].get('max_bc_files'),
        observation_wind_override=bc_config.get('observation_wind_override'),
        cache_enabled=bool(bc_config.get('cache_enabled', True)),
        cache_dir=bc_config.get('cache_dir'),
        include_previous_action_observation=bool(env_config.get('include_previous_action_observation', False)),
        action_low=env_config.get('action_low'),
        action_high=env_config.get('action_high'),
        precomputed_action_target_columns=bc_config.get('precomputed_action_target_columns'),
        include_operating_state_observation=bool(
            env_config.get('include_operating_state_observation', False)
        ),
        state_action_artifact_path=env_config.get('state_action_artifact_path'),
    )
    dataloader = DataLoader(dataset, batch_size=int(bc_config.get('batch_size', 512)), shuffle=True, drop_last=True)
    return dataset, dataloader


def _build_offline_eval_dataset(config: dict) -> OfflineBCDataset:
    offline_eval_cfg = config.get('offline_eval', {})
    if not isinstance(offline_eval_cfg, dict):
        raise ValueError('offline_eval must be an object')
    data_cfg = config['data']
    bc_cfg = config.get('bc', {})
    if not isinstance(bc_cfg, dict):
        raise ValueError('bc must be an object')
    return OfflineBCDataset(
        root_dir=data_cfg['root_dir'],
        split=str(offline_eval_cfg.get('split', data_cfg.get('eval_split', 'val'))),
        max_files=data_cfg.get('max_bc_files'),
        observation_wind_override=offline_eval_cfg.get(
            'observation_wind_override',
            bc_cfg.get('observation_wind_override'),
        ),
        cache_enabled=bool(offline_eval_cfg.get('cache_enabled', bc_cfg.get('cache_enabled', True))),
        cache_dir=offline_eval_cfg.get('cache_dir', bc_cfg.get('cache_dir')),
        include_previous_action_observation=bool(config['env'].get('include_previous_action_observation', False)),
        action_low=config['env'].get('action_low'),
        action_high=config['env'].get('action_high'),
        include_operating_state_observation=bool(
            config['env'].get('include_operating_state_observation', False)
        ),
        state_action_artifact_path=config['env'].get('state_action_artifact_path'),
    )


def _next_bc_batch(
    iterator,
    controller: SafeResidualController,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = next(iterator)
    if len(batch) == 3:
        obs, actions_real, actions_target = batch
        return obs.to(device), actions_real.to(device), actions_target.to(device)
    obs, actions_real = batch
    obs = obs.to(device)
    actions_real = actions_real.to(device)
    actions_target = controller.action_target_tensor(
        actions_real,
        obs=obs,
    )
    return obs, actions_real, actions_target


def _get_bc_checkpoint_steps(config: dict) -> list[int]:
    bc_config = config.get('bc', {})
    raw_steps = bc_config.get('checkpoint_steps', [])
    if raw_steps is None:
        return []
    if not isinstance(raw_steps, (list, tuple)):
        raise ValueError('bc.checkpoint_steps must be a list of integers')
    total_steps = int(bc_config.get('steps', 0))
    checkpoint_steps = sorted({
        int(step)
        for step in raw_steps
        if int(step) > 0 and int(step) <= total_steps
    })
    return checkpoint_steps


def _resolve_bc_warmstart_checkpoint_path(config: dict) -> Path:
    bc_config = config.get('bc', {})
    configured_path = str(bc_config.get('resume_checkpoint', '')).strip()
    if configured_path:
        return normalize_user_path(configured_path)
    return PROJECT_ROOT / 'RL' / 'BC_checkpoint' / 'bc_step_20000.pt'


def _try_load_bc_warmstart_checkpoint(
    agent,
    checkpoint_path: Path,
    device: torch.device,
    data_schema_version: str,
    expected_control_fingerprint: str | None = None,
    expected_actor_control_fingerprint: str | None = None,
    expected_resolved_control_contract_fingerprint: str | None = None,
) -> bool:
    if not checkpoint_path.exists():
        return False
    checkpoint = torch.load(checkpoint_path, map_location=device)
    agent_state = None
    if isinstance(checkpoint, dict):
        if 'agent' in checkpoint:
            agent_state = checkpoint['agent']
        elif 'actor' in checkpoint:
            agent_state = checkpoint
    if agent_state is None:
        raise ValueError(f'Invalid BC checkpoint format: {checkpoint_path}')
    checkpoint_schema_version = str(checkpoint.get('data_schema_version', '')).strip()
    if checkpoint_schema_version != data_schema_version:
        raise ValueError(
            f'BC checkpoint schema is incompatible: {checkpoint_path} has {checkpoint_schema_version!r}, '
            f'but the current run requires {data_schema_version!r}. Re-run BC pretraining for the SI/My-only schema.'
        )
    if expected_resolved_control_contract_fingerprint is not None:
        checkpoint_resolved_fingerprint = str(
            checkpoint.get('control_contract_fingerprint', '')
        ).strip()
        if checkpoint_resolved_fingerprint != expected_resolved_control_contract_fingerprint:
            raise ValueError(
                f'BC checkpoint resolved control contract is incompatible: {checkpoint_path}. '
                'Rebuild B4, BC data, normalization, and the checkpoint for the resolved contract.'
            )
    checkpoint_actor_control_fingerprint = str(checkpoint.get('actor_control_fingerprint', '')).strip()
    if expected_actor_control_fingerprint is not None and checkpoint_actor_control_fingerprint:
        if checkpoint_actor_control_fingerprint != expected_actor_control_fingerprint:
            raise ValueError(
                f'BC checkpoint actor control contract is incompatible: {checkpoint_path}. '
                'Re-run BC pretraining for the configured baseline, residual limits, and safety envelope.'
            )
    elif expected_control_fingerprint is not None:
        checkpoint_control_fingerprint = str(checkpoint.get('control_fingerprint', '')).strip()
        if checkpoint_control_fingerprint != expected_control_fingerprint:
            raise ValueError(
                f'BC checkpoint control contract is incompatible: {checkpoint_path}. '
                'Re-run BC pretraining for the configured wind-speed baseline and safety envelope.'
            )
    checkpoint_actor_state = agent_state.get('actor')
    if not isinstance(checkpoint_actor_state, dict):
        raise ValueError(f'BC checkpoint actor state is missing or invalid: {checkpoint_path}')
    checkpoint_obs_median = checkpoint_actor_state.get('obs_median')
    checkpoint_obs_iqr = checkpoint_actor_state.get('obs_iqr')
    current_obs_dim = int(agent.actor.obs_median.numel())
    if not isinstance(checkpoint_obs_median, torch.Tensor) or not isinstance(checkpoint_obs_iqr, torch.Tensor):
        raise ValueError(
            f'BC checkpoint is missing actor observation normalization state: {checkpoint_path}. '
            'Re-run BC pretraining for the SI/My-only schema.'
        )
    if (
        int(checkpoint_obs_median.numel()) != current_obs_dim
        or int(checkpoint_obs_iqr.numel()) != current_obs_dim
    ):
        raise ValueError(
            f'BC checkpoint observation dimension is incompatible: {checkpoint_path} has '
            f'{int(checkpoint_obs_median.numel())}/{int(checkpoint_obs_iqr.numel())} median/IQR dimensions, '
            f'but the current data schema requires '
            f'{current_obs_dim}. Re-run BC pretraining with the current control observation schema.'
        )
    try:
        agent.load_state_dict(agent_state)
    except RuntimeError as exc:
        raise ValueError(
            f'BC checkpoint is incompatible with the current model/data schema: {checkpoint_path}. '
            'Re-run BC pretraining before online evaluation or RL training.'
        ) from exc
    return True


def _export_bc_checkpoint(
    agent,
    config: dict,
    checkpoint_path: Path,
    control_fingerprint: str | None = None,
    actor_control_fingerprint: str | None = None,
    online_training_fingerprint: str | None = None,
    control_contract_fingerprint: str | None = None,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        'agent': agent.state_dict(),
        'config': config,
        'data_schema_version': config['data_schema_version'],
        'control_fingerprint': control_fingerprint,
        'actor_control_fingerprint': actor_control_fingerprint,
        'online_training_fingerprint': online_training_fingerprint,
        'control_contract_fingerprint': control_contract_fingerprint,
        'stage': 'bc',
        'bc_total_steps': int(config['bc'].get('steps', 0)),
    }
    temporary_path = checkpoint_path.with_name(
        f'.{checkpoint_path.name}.{os.getpid()}.tmp'
    )
    try:
        torch.save(checkpoint, temporary_path)
        os.replace(temporary_path, checkpoint_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    print(f'[BC] exported checkpoint: {checkpoint_path}', flush=True)


def _save_bc_prediction_artifacts(
    agent,
    dataset: OfflineBCDataset,
    controller: SafeResidualController,
    logger: ExperimentLogger,
    config: dict,
    device: torch.device,
) -> None:
    preview_count = min(int(config['bc'].get('preview_samples', 512)), len(dataset))
    if preview_count <= 0:
        return
    preview_batch_size = max(1, int(config['bc'].get('preview_batch_size', 1024)))
    indices = np.linspace(0, len(dataset) - 1, num=preview_count, dtype=np.int64)
    obs_np = dataset.obs[indices]
    actions_true_np = dataset.actions[indices]
    action_targets_np = None if dataset.action_targets is None else dataset.action_targets[indices]

    pred_norm_chunks: list[np.ndarray] = []
    agent.actor.eval()
    with torch.inference_mode():
        for start in range(0, preview_count, preview_batch_size):
            end = min(start + preview_batch_size, preview_count)
            obs_batch = torch.from_numpy(obs_np[start:end]).to(device)
            pred_norm = agent.actor(obs_batch, agent.actor_residual_target_scale)
            pred_norm_chunks.append(pred_norm.cpu().numpy())
    agent.actor.train()
    pred_norm_np = np.concatenate(pred_norm_chunks, axis=0).astype(np.float32)
    if action_targets_np is not None:
        residual_mae = np.abs(pred_norm_np - action_targets_np).mean(axis=0)
        logger.save_summary(
            'bc_prediction_preview_summary.json',
            {
                'preview_samples': float(preview_count),
                'torque_residual_mae': float(residual_mae[0]),
                'pitch_residual_mae': float(residual_mae[1]),
            },
        )
        return

    actions_pred_np = controller.action_real_tensor(
        torch.from_numpy(pred_norm_np).to(device),
        torch.from_numpy(obs_np).to(device),
    ).cpu().numpy().astype(np.float32)

    torque_mae = float(np.abs(actions_pred_np[:, 0] - actions_true_np[:, 0]).mean())
    pitch_mae = float(np.abs(actions_pred_np[:, 1] - actions_true_np[:, 1]).mean())
    logger.save_summary(
        'bc_prediction_preview_summary.json',
        {
            'preview_samples': float(preview_count),
            'torque_mae_raw': torque_mae,
            'pitch_mae_raw': pitch_mae,
        },
    )

    preview_csv = logger.output_dir / 'bc_prediction_preview.csv'
    with preview_csv.open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                'sample_index',
                'torque_true',
                'torque_pred',
                'pitch_true',
                'pitch_pred',
            ],
        )
        writer.writeheader()
        for row_idx in range(preview_count):
            writer.writerow(
                {
                    'sample_index': int(indices[row_idx]),
                    'torque_true': float(actions_true_np[row_idx, 0]),
                    'torque_pred': float(actions_pred_np[row_idx, 0]),
                    'pitch_true': float(actions_true_np[row_idx, 1]),
                    'pitch_pred': float(actions_pred_np[row_idx, 1]),
                }
            )

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    xs = np.arange(preview_count)
    for action_idx, action_name in enumerate(['torque', 'pitch']):
        plt.figure(figsize=(10, 4))
        plt.plot(xs, actions_true_np[:, action_idx], label='true', linewidth=1.2)
        plt.plot(xs, actions_pred_np[:, action_idx], label='pred', linewidth=1.2)
        plt.title(f'bc {action_name} prediction vs true')
        plt.xlabel('preview sample')
        plt.ylabel(action_name)
        plt.legend()
        plt.tight_layout()
        plt.savefig(logger.figure_dir / f'bc_{action_name}_pred_vs_true.png')
        plt.close()


def _warm_up_gpu(
    agent,
    obs_dim: int,
    action_dim: int,
    batch_size: int,
    device: torch.device,
    updates: int,
    bc_obs: torch.Tensor | None = None,
    bc_actions: torch.Tensor | None = None,
) -> None:
    if device.type != 'cuda' or updates <= 0:
        return

    saved_state = copy.deepcopy(agent.state_dict())
    try:
        for _ in range(updates):
            warmup_batch = SimpleNamespace(
                obs=torch.randn(batch_size, obs_dim),
                actions=torch.empty(batch_size, action_dim).uniform_(-1.0, 1.0),
                rewards=torch.randn(batch_size, 1),
                next_obs=torch.randn(batch_size, obs_dim),
                dones=torch.zeros(batch_size, 1),
            )
            agent.update(warmup_batch, bc_obs=bc_obs, bc_actions=bc_actions)
        torch.cuda.synchronize(device)
    finally:
        agent.load_state_dict(saved_state)


def run_bc_warmstart(
    agent,
    controller: SafeResidualController,
    config: dict,
    logger: ExperimentLogger,
    device: torch.device,
    dataloader: DataLoader,
) -> None:
    bc_config = config['bc']
    iterator = itertools.cycle(dataloader)

    if bc_config.get('freeze_critic_during_bc', True):
        agent.set_critic_requires_grad(False)
    total_steps = int(bc_config.get('steps', 10000))
    loss_type = str(bc_config.get('loss_type', 'mse')).lower()
    log_every = int(bc_config.get('log_every', 500))
    checkpoint_steps = set(_get_bc_checkpoint_steps(config))
    print(f'[BC] start steps={total_steps} batch_size={dataloader.batch_size} loss={loss_type}', flush=True)
    for step in range(1, total_steps + 1):
        obs, _actions_real, actions_target = _next_bc_batch(iterator, controller, device)
        metrics = agent.bc_update(obs, actions_target, loss_type=loss_type)
        logger.log_metrics('bc', step, metrics)
        if step in checkpoint_steps:
            logger.save_checkpoint(
                f'bc_step_{step}.pt',
                {
                    'agent': agent.state_dict(),
                    'config': config,
                    'data_schema_version': config['data_schema_version'],
                    'control_fingerprint': controller.control_fingerprint,
                    'actor_control_fingerprint': controller.actor_control_fingerprint,
                    'online_training_fingerprint': controller.online_training_fingerprint,
                    'control_contract_fingerprint': controller.resolved_control_contract_fingerprint,
                    'stage': 'bc',
                    'bc_step': step,
                    'bc_total_steps': total_steps,
                },
            )
            logger.flush()
            print(f'[BC-Checkpoint] saved step={step}', flush=True)
        if step == 1 or step % log_every == 0 or step == total_steps:
            print(f'[BC] step={step}/{total_steps} {_format_metrics(metrics, ["bc_loss", "bc_mae", "bc_torque_mae", "bc_pitch_mae"])}', flush=True)
    if bc_config.get('freeze_critic_during_bc', True):
        agent.set_critic_requires_grad(True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='RL/configs/direct_rl.json')
    parser.add_argument('--data-root', type=str, default='', help='Override config.data.root_dir')
    parser.add_argument('--dynamics-ckpt', type=str, default='', help='Override config.env.dynamics_checkpoint')
    parser.add_argument('--device', type=str, default='', help='Override config.device (for example cpu or cuda:0).')
    parser.add_argument('--udp-host', type=str, default='', help='Override env.udp_host; offline BC can bind to loopback only.')
    parser.add_argument('--udp-port', type=int, default=None, help='Override env.udp_port; use 0 for an ephemeral local port.')
    parser.add_argument(
        '--queue-force-baseline',
        action='store_true',
        help='Diagnostic queue mode: send only the wind-speed baseline action and disable online updates.',
    )
    parser.add_argument(
        '--queue-force-baseline-hold-initial',
        action='store_true',
        help='Diagnostic queue mode: hold each condition at its first wind-speed baseline action.',
    )
    parser.add_argument(
        '--queue-force-bc-baseline',
        action='store_true',
        help='Diagnostic queue mode: use a frozen BC actor as the absolute baseline and disable online updates.',
    )
    parser.add_argument(
        '--queue-force-pid-history',
        type=str,
        default='',
        metavar='CSV',
        help='Diagnostic queue mode: inject CSV action history with safety bounds and rate limits.',
    )
    parser.add_argument(
        '--queue-pid-history-action-offset',
        type=int,
        default=0,
        metavar='STEPS',
        help='Advance CSV action history relative to the same-index state reference in PID-history diagnostic mode.',
    )
    parser.add_argument(
        '--queue-residual-scale',
        type=float,
        default=None,
        metavar='SCALE',
        help='Diagnostic queue mode: apply a deterministic actor residual scaled to [0, 1] without training.',
    )
    args = parser.parse_args()

    if args.queue_residual_scale is not None:
        if not math.isfinite(args.queue_residual_scale) or not 0.0 <= args.queue_residual_scale <= 1.0:
            parser.error('--queue-residual-scale must be a finite number between 0 and 1')
        if (
            args.queue_force_baseline
            or args.queue_force_baseline_hold_initial
            or args.queue_force_bc_baseline
            or args.queue_force_pid_history
        ):
            parser.error('--queue-residual-scale cannot be combined with a force-baseline option')
    if args.queue_force_bc_baseline and (args.queue_force_baseline or args.queue_force_baseline_hold_initial):
        parser.error('--queue-force-bc-baseline cannot be combined with a wind-speed force-baseline option')
    if args.queue_force_pid_history and (
        args.queue_force_baseline
        or args.queue_force_baseline_hold_initial
        or args.queue_force_bc_baseline
    ):
        parser.error('--queue-force-pid-history cannot be combined with another force-baseline option')
    if args.queue_pid_history_action_offset < 0:
        parser.error('--queue-pid-history-action-offset must be a non-negative integer')
    if args.queue_pid_history_action_offset and not args.queue_force_pid_history:
        parser.error('--queue-pid-history-action-offset requires --queue-force-pid-history')

    print(f'[Startup] loading config={args.config}', flush=True)
    config = load_config(args.config)
    config['_source_config_path'] = str(normalize_user_path(args.config).expanduser().resolve())

    # Allow overriding external artifact paths without editing config files.
    data_root_override = (args.data_root or os.environ.get('RL_DATA_ROOT', '')).strip()
    if data_root_override:
        config.setdefault('data', {})['root_dir'] = data_root_override
    dynamics_ckpt_override = (args.dynamics_ckpt or os.environ.get('RL_DYNAMICS_CHECKPOINT', '')).strip()
    if dynamics_ckpt_override:
        config.setdefault('env', {})['dynamics_checkpoint'] = dynamics_ckpt_override
    if args.udp_host.strip():
        config.setdefault('env', {})['udp_host'] = args.udp_host.strip()
    if args.udp_port is not None:
        if not 0 <= args.udp_port <= 65535:
            parser.error('--udp-port must be between 0 and 65535')
        config.setdefault('env', {})['udp_port'] = args.udp_port
    if args.device.strip():
        config['device'] = args.device.strip()
    if args.queue_force_baseline or args.queue_force_baseline_hold_initial:
        config.setdefault('queue', {})['force_baseline'] = True
        if args.queue_force_baseline_hold_initial:
            config['queue']['force_baseline_hold_initial'] = True
            config['experiment_name'] = f'{config["experiment_name"]}_baseline_hold_diagnosis'
        else:
            config['experiment_name'] = f'{config["experiment_name"]}_baseline_diagnosis'
    if args.queue_force_bc_baseline:
        config.setdefault('queue', {})['force_bc_baseline'] = True
        config['experiment_name'] = f'{config["experiment_name"]}_bc_baseline_diagnosis'
    if args.queue_force_pid_history:
        config.setdefault('queue', {})['pid_history_diagnostic_path'] = args.queue_force_pid_history
        if args.queue_pid_history_action_offset:
            config['queue']['pid_history_diagnostic_action_offset_steps'] = args.queue_pid_history_action_offset
            config['experiment_name'] = (
                f'{config["experiment_name"]}_pid_history_offset_{args.queue_pid_history_action_offset}_diagnosis'
            )
        else:
            config['experiment_name'] = f'{config["experiment_name"]}_pid_history_diagnosis'
    if args.queue_residual_scale is not None:
        config.setdefault('queue', {})['residual_scale_diagnostic'] = args.queue_residual_scale
        scale_label = f'{args.queue_residual_scale:.3f}'.replace('.', 'p')
        config['experiment_name'] = f'{config["experiment_name"]}_residual_scale_{scale_label}_diagnosis'

    data_root = config.get('data', {}).get('root_dir')
    if isinstance(data_root, str) and data_root.strip():
        config['data']['root_dir'] = str(normalize_user_path(data_root))
    dynamics_ckpt = config.get('env', {}).get('dynamics_checkpoint')
    if isinstance(dynamics_ckpt, str) and dynamics_ckpt.strip():
        config['env']['dynamics_checkpoint'] = str(normalize_user_path(dynamics_ckpt))
    output_dir = config.get('output_dir')
    if isinstance(output_dir, str) and output_dir.strip():
        config['output_dir'] = str(normalize_user_path(output_dir))

    configured_control_contract = resolve_configured_control_contract(
        config,
        project_root=PROJECT_ROOT,
    )
    resolved_control_contract_fingerprint = (
        None if configured_control_contract is None else configured_control_contract[1]
    )
    data_schema_version = _require_data_schema_version(config)

    set_seed(int(config.get('seed', 42)))
    device_name = config.get('device', 'cpu')
    if device_name.startswith('cuda') and not torch.cuda.is_available():
        device_name = 'cpu'
    device = torch.device(device_name)
    if physical_backend_selected(config):
        config['device'] = str(device)
        # Check the FINAL effective config, before UDP, exports or output files.
        if config.get('physical_shadow_launch', {}).get('route') in OPERATOR_TRAINING_ROUTES:
            require_fresh_operator_experiment(config)
        validate_physical_launch(config, project_root=PROJECT_ROOT)
        preflight_physical_config(config, project_root=PROJECT_ROOT)
    print(f'[Startup] config loaded algorithm={config["algorithm"]} mode={config.get("training_mode")} device={device}', flush=True)

    print('[Startup] building train environment...', flush=True)
    # Every initialization failure after binding UDP is inside cleanup scope.
    with ExitStack() as resources:
        env = WindTurbineEnv(config['env'], config['data'], device)
        resources.callback(env.close)
        _run_initialized_training(config, device, env, data_schema_version,
            resolved_control_contract_fingerprint, resources)


def require_fresh_operator_experiment(config: dict) -> Path:
    """Fail descriptively before any UDP/output initialization; preserve failed runs."""
    output_root = Path(config['output_dir'])
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    run_id = str(config['experiment_name'])
    if config.get('append_timestamp', True):
        matches = tuple(path for path in output_root.glob(f'{run_id}_*') if path.is_dir())
        output = output_root / f'{run_id}_<timestamp>'
    else:
        output = output_root / run_id
        matches = (output,) if output.exists() else ()
    if matches:
        raise FileExistsError(
            f'Experiment directory/prefix already exists: {matches[0]}. Old experiment preserved; '
            'prepare a fresh RUN_ID, regenerate the source-bound receipt, and synchronize '
            'Windows RUN_ID/state. Do not delete or resume it.'
        )
    return output


def _flush_training_logger_on_exit(logger: ExperimentLogger) -> None:
    try:
        logger.flush()
    except Exception as exc:
        print(f'[Cleanup] logger flush failed: {type(exc).__name__}', file=sys.stderr, flush=True)


def _close_physical_shadow_on_exit(shadow: PhysicalLoadShadow, logger: ExperimentLogger) -> None:
    try:
        summary = shadow.close()
    except Exception as exc:
        summary = {'backend': 'physical_fir_v2', 'finalized': False, 'faulted': True,
                   'summary_status': 'FINALIZATION_ERROR', 'last_error_type': type(exc).__name__,
                   'last_error_message': str(exc)}
    try:
        logger.save_summary('physical_load_shadow_summary.json', summary)
    except Exception as exc:
        print(f'[Cleanup] physical shadow summary failed: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)


def _run_initialized_training(config: dict, device: torch.device, env: WindTurbineEnv,
        data_schema_version: str, resolved_control_contract_fingerprint: str | None,
        resources: ExitStack) -> None:
    if getattr(env, 'requires_single_instance', False):
        print('[Startup] using shared train/eval environment instance', flush=True)
        eval_env = env
    else:
        print('[Startup] building eval environment...', flush=True)
        eval_env = WindTurbineEnv(config['env'], {**config['data'], 'train_split': config['data'].get('eval_split', 'val')}, device)
        resources.callback(eval_env.close)
    print('[Startup] building agent and replay buffer...', flush=True)
    agent = build_agent(config, env.obs_dim, env.action_dim, device)
    replay_buffer = _build_online_replay_buffer(env, config)
    controller = SafeResidualController(config['env'], config.get('control', {}))
    bc_dataset = None
    bc_dataloader = None
    bc_iterator = None
    if config.get('bc', {}).get('enabled', False):
        print('[Startup] loading BC dataset/statistics...', flush=True)
        bc_dataset, bc_dataloader = _build_bc_dataloader(config)
        bc_iterator = itertools.cycle(bc_dataloader)
        if hasattr(agent, 'set_actor_obs_normalizer'):
            agent.set_actor_obs_normalizer(
                torch.from_numpy(bc_dataset.obs_median),
                torch.from_numpy(bc_dataset.obs_iqr),
            )
        if hasattr(agent, 'set_critic_obs_normalizer'):
            agent.set_critic_obs_normalizer(
                torch.from_numpy(bc_dataset.obs_median),
                torch.from_numpy(bc_dataset.obs_iqr),
            )
        controller.set_observation_distribution(bc_dataset.obs_median, bc_dataset.obs_iqr)
        print('[Startup] actor and critic observation normalizers initialized from BC dataset', flush=True)

    offline_eval_cfg = config.get('offline_eval', {})
    if not isinstance(offline_eval_cfg, dict):
        raise ValueError('offline_eval must be an object')
    offline_eval_dataset = None
    if bool(offline_eval_cfg.get('enabled', False)):
        if bc_dataset is None:
            raise ValueError('offline_eval requires an enabled BC dataset for controller normalization')
        print('[Startup] loading offline validation dataset...', flush=True)
        offline_eval_dataset = _build_offline_eval_dataset(config)

    warmup_updates = int(config.get('train', {}).get('gpu_warmup_updates', 1))
    if device.type == 'cuda' and warmup_updates > 0:
        print(f'[Startup] warming up CUDA with {warmup_updates} TD3 update(s)...', flush=True)
        warmup_bc_obs = None
        warmup_bc_actions = None
        if bc_iterator is not None:
            warmup_bc_obs, _warmup_bc_actions_real, warmup_bc_actions = _next_bc_batch(bc_iterator, controller, device)
        _warm_up_gpu(
            agent,
            env.obs_dim,
            env.action_dim,
            batch_size=min(int(config['train'].get('batch_size', 256)), 256),
            device=device,
            updates=warmup_updates,
            bc_obs=warmup_bc_obs,
            bc_actions=warmup_bc_actions,
        )
        print('[Startup] CUDA warmup complete', flush=True)

    resolved_experiment_name = build_experiment_name(config)
    output_dir = Path(config.get('output_dir', 'RL/experiments')) / resolved_experiment_name
    logger = ExperimentLogger(output_dir, config.get('logging'))
    resources.callback(_flush_training_logger_on_exit, logger)
    try:
        from RL.run_meta import write_run_meta
        meta_path = write_run_meta(
            logger.output_dir,
            config,
            config_path=config.get('_source_config_path'),
            device=str(device),
            mode='legacy',
        )
        print(f'[Startup] run_meta: {meta_path}', flush=True)
    except Exception as exc:
        print(f'[Startup] run_meta failed: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
    config_to_save = dict(config)
    config_to_save['resolved_experiment_name'] = resolved_experiment_name
    logger.save_config(config_to_save)
    print(f'[Train] experiment={resolved_experiment_name} device={device} output={logger.output_dir}', flush=True)

    if config.get('training_mode') in {'bc_warmstart_rl', 'bc_pretrain_only'} and config['bc'].get('enabled', False):
        bc_checkpoint_path = _resolve_bc_warmstart_checkpoint_path(config)
        force_bc_retrain = config['bc'].get('force_retrain', False)
        if not isinstance(force_bc_retrain, bool):
            raise ValueError('bc.force_retrain must be a boolean')
        if force_bc_retrain and config.get('training_mode') != 'bc_pretrain_only':
            raise ValueError('bc.force_retrain is only supported for training_mode=bc_pretrain_only')
        checkpoint_contract_required = controller.checkpoint_contract_required()
        expected_control_fingerprint = (
            controller.control_fingerprint
            if checkpoint_contract_required
            else None
        )
        expected_actor_control_fingerprint = (
            controller.actor_control_fingerprint
            if checkpoint_contract_required
            else None
        )
        checkpoint_loaded = False if force_bc_retrain else _try_load_bc_warmstart_checkpoint(
            agent,
            bc_checkpoint_path,
            device,
            data_schema_version,
            expected_control_fingerprint,
            expected_actor_control_fingerprint,
            resolved_control_contract_fingerprint,
        )
        if checkpoint_loaded:
            print(f'[BC] found existing checkpoint, skip warmstart: {bc_checkpoint_path}', flush=True)
        else:
            if int(config['bc'].get('steps', 0)) <= 0:
                raise FileNotFoundError(
                    f'No compatible BC checkpoint found at {bc_checkpoint_path}. '
                    'Run the BC pretraining configuration for the current data schema first, '
                    'or configure bc.steps to a positive value.'
                )
            if force_bc_retrain:
                print('[Startup] bc.force_retrain=true; training a fresh BC actor and replacing the configured export checkpoint.', flush=True)
            else:
                print('[Startup] entering BC warmstart stage...', flush=True)
            run_bc_warmstart(agent, controller, config, logger, device, bc_dataloader)
        _save_bc_prediction_artifacts(agent, bc_dataset, controller, logger, config, device)
        print('[BC] saved prediction-vs-true preview artifacts', flush=True)
        export_path = str(config['bc'].get('export_checkpoint_path', '')).strip()
        if export_path:
            _export_bc_checkpoint(
                agent,
                config,
                normalize_user_path(export_path),
                controller.control_fingerprint,
                controller.actor_control_fingerprint,
                controller.online_training_fingerprint,
                controller.resolved_control_contract_fingerprint,
            )

    if config.get('training_mode') == 'bc_pretrain_only':
        if bool(offline_eval_cfg.get('enabled', False)):
            if offline_eval_dataset is None:
                raise RuntimeError('Offline validation dataset was not initialized')
            evaluation = evaluate_offline_policy(
                agent,
                offline_eval_dataset,
                controller,
                device,
                batch_size=int(offline_eval_cfg.get('batch_size', 1024)),
            )
            write_offline_policy_evaluation(logger.output_dir / 'offline_eval_final', evaluation)
            logger.save_summary('offline_eval_final_summary.json', evaluation.summary)
            logger.log_metrics('offline_eval_final', 0, evaluation.summary)
        logger.flush()
        print(f'[BC] pretraining finished output={logger.output_dir}', flush=True)
        return

    if env.queue_lifecycle_enabled:
        physical_load_shadow = build_physical_shadow(config, env,
            output_dir=logger.output_dir, project_root=PROJECT_ROOT)
        if physical_load_shadow is not None:
            resources.callback(_close_physical_shadow_on_exit, physical_load_shadow, logger)
        queue_summary = run_queue_training(
            agent,
            env,
            controller,
            replay_buffer,
            config,
            logger,
            device,
            bc_iterator=bc_iterator,
            bc_dataset=bc_dataset,
            offline_eval_dataset=offline_eval_dataset,
            physical_load_shadow=physical_load_shadow,
        )
        logger.save_summary('queue_training_summary.json', queue_summary)
        print(f'[Queue] finished output={logger.output_dir}', flush=True)
        return

    total_steps = 0
    train_cfg = config['train']
    eval_cfg = config.get('eval', {})
    if not isinstance(eval_cfg, dict):
        raise ValueError('eval must be an object')
    eval_episodes = int(eval_cfg.get('episodes', 3))
    if eval_episodes <= 0:
        raise ValueError('eval.episodes must be positive')
    save_best_eval_checkpoint = bool(eval_cfg.get('save_best_checkpoint', False))
    eval_checkpoint_metric = str(eval_cfg.get('checkpoint_metric', 'reward_mean')).strip()
    if save_best_eval_checkpoint and eval_checkpoint_metric not in {'reward_mean', 'reward_median'}:
        raise ValueError('eval.checkpoint_metric must be reward_mean or reward_median')
    best_checkpoint_name = str(eval_cfg.get('best_checkpoint_name', 'best_eval.pt')).strip()
    if save_best_eval_checkpoint and (not best_checkpoint_name or Path(best_checkpoint_name).name != best_checkpoint_name):
        raise ValueError('eval.best_checkpoint_name must be a checkpoint file name without directories')
    best_eval_score = -math.inf
    batch_size = int(train_cfg.get('batch_size', 256))
    warmup_steps = int(train_cfg.get('online_replay_warmup_steps', train_cfg.get('warmup_steps', 1000)))
    warmup_action_source = str(train_cfg.get('warmup_action_source', 'bc_policy')).lower()
    if warmup_action_source != 'bc_policy':
        raise ValueError('train.warmup_action_source must be bc_policy')
    warmup_exploration_noise = float(train_cfg.get('warmup_exploration_noise', 0.0))
    critic_update_after_steps = int(train_cfg.get('critic_update_after_steps', warmup_steps))
    actor_update_after_steps = int(train_cfg.get('actor_update_after_steps', critic_update_after_steps))
    online_rl_enabled = bool(train_cfg.get('online_rl_enabled', True))
    updates_per_step = int(train_cfg.get('updates_per_step', 1))
    update_every_steps = max(1, int(train_cfg.get('update_every_steps', 1)))
    exploration_noise = float(train_cfg.get('exploration_noise', 0.1))
    step_log_every = int(train_cfg.get('step_log_every', 100))
    timing_console_every = int(train_cfg.get('timing_console_every', 1))
    timing_console_overruns = bool(train_cfg.get('timing_console_overruns', False))
    logical_sim_step_ms = float(config['env'].get('bladed_step_duration_ms', 0.0))
    step_deadline_ms = float(config['env'].get('wall_clock_step_deadline_ms', 0.0))
    if logical_sim_step_ms < 0.0:
        raise ValueError('env.bladed_step_duration_ms must be non-negative')
    if step_deadline_ms < 0.0:
        raise ValueError('env.wall_clock_step_deadline_ms must be non-negative')
    print(
        f'[Startup] entering RL stage episodes={train_cfg.get("total_episodes", 10)} '
        f'warmup_steps={warmup_steps} warmup_action_source={warmup_action_source} '
        f'critic_update_after_steps={critic_update_after_steps} actor_update_after_steps={actor_update_after_steps} '
        f'online_rl_enabled={online_rl_enabled} batch_size={batch_size} update_every_steps={update_every_steps} '
        f'logical_sim_step_ms={logical_sim_step_ms:.3f} wall_clock_deadline_ms={step_deadline_ms:.3f}',
        flush=True,
    )
    all_step_times_ms: list[float] = []
    all_episode_times_ms: list[float] = []
    all_step_overrun_count = 0
    all_step_overrun_total_ms = 0.0
    all_step_overrun_max_ms = 0.0
    first_step_start_time: float | None = None
    last_step_end_time: float | None = None

    for episode in range(1, int(train_cfg.get('total_episodes', 10)) + 1):
        episode_start_time = time.perf_counter()
        episode_step_times_ms: list[float] = []
        current_td3_bc_weight = float(config['agent'].get('td3_bc_weight', 1.0))
        obs, _ = env.reset()
        controller.reset_condition(obs, previous_action_real=env.last_sent_action)
        episode_reward = 0.0
        episode_metrics: list[dict[str, float]] = []
        done = False
        step_in_episode = 0
        invalid_transition_count = 0
        replay_reject_count = 0
        update_skip_count = 0
        step_overrun_count = 0
        step_overrun_total_ms = 0.0
        last_update_metrics: dict[str, float] = {}

        while not done:
            step_start_time = time.perf_counter()
            if first_step_start_time is None:
                first_step_start_time = step_start_time
            obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device)
            if total_steps < warmup_steps:
                action_norm = agent.select_action(
                    obs_tensor,
                    noise_std=warmup_exploration_noise,
                    deterministic=warmup_exploration_noise <= 0.0,
                )
            else:
                action_norm = agent.select_action(
                    obs_tensor,
                    noise_std=exploration_noise if online_rl_enabled else 0.0,
                    deterministic=not online_rl_enabled,
                )
            policy_action_np = action_norm.squeeze(0).detach().cpu().numpy()
            safe_action = controller.action_from_policy(
                policy_action_np,
                obs,
                allow_policy=True,
            )
            action_np = safe_action.executed_action_norm
            action_ready_time = time.perf_counter()

            next_total_steps = total_steps + 1
            update_scheduled = (
                online_rl_enabled
                and len(replay_buffer) >= batch_size
                and next_total_steps >= critic_update_after_steps
                and next_total_steps % update_every_steps == 0
            )
            update_metrics_records: list[dict[str, float]] = []
            update_time_ms = 0.0

            def update_during_udp_wait(_action_sent_monotonic_s: float) -> None:
                nonlocal update_skip_count, update_time_ms, last_update_metrics
                if not update_scheduled:
                    return
                update_start_time = time.perf_counter()
                for _ in range(updates_per_step):
                    batch = replay_buffer.sample(batch_size)
                    if config['algorithm'].lower() == 'td3' and getattr(agent, 'td3_bc_enabled', False) and bc_iterator is not None:
                        bc_obs, _bc_actions_real, bc_actions_target = _next_bc_batch(bc_iterator, controller, device)
                        update_metrics = agent.update(
                            SimpleNamespace(**batch),
                            bc_obs=bc_obs,
                            bc_actions=bc_actions_target,
                            bc_weight_override=current_td3_bc_weight,
                            update_actor=next_total_steps >= actor_update_after_steps,
                        )
                    else:
                        update_metrics = agent.update(SimpleNamespace(**batch))
                    update_metrics.setdefault('td3_bc_weight_schedule', float(current_td3_bc_weight))
                    update_metrics_records.append(update_metrics)
                    last_update_metrics = update_metrics
                    if any(
                        update_metrics.get(key, 0.0) > 0.0
                        for key in ('update_skipped_nonfinite_batch', 'update_skipped_nonfinite_loss', 'update_skipped_q_guard')
                    ):
                        update_skip_count += 1
                update_time_ms = (time.perf_counter() - update_start_time) * 1000.0

            step_result = _step_environment(
                env,
                safe_action.action_norm,
                safe_action.action_real,
                on_action_sent=update_during_udp_wait if update_scheduled else None,
            )
            env_step_end_time = time.perf_counter()
            safety_stop = controller.observe_step(float(step_result.info.get('power', 0.0)))
            step_result.info.update(
                {
                    'control_ood_score': safe_action.ood_score,
                    'control_fallback_active': float(safe_action.fallback_active),
                    'control_above_rated_initial_pitch_floor_active': float(
                        safe_action.above_rated_initial_pitch_floor_active
                    ),
                    'control_above_rated_initial_pitch_floor_applied': float(
                        safe_action.above_rated_initial_pitch_floor_applied
                    ),
                    'control_action_torque': float(safe_action.action_real[0]),
                    'control_action_pitch': float(safe_action.action_real[1]),
                    'control_residual_torque': float(action_np[0]),
                    'control_residual_pitch': float(action_np[1]),
                    'control_policy_action_torque': float(safe_action.policy_action_norm[0]),
                    'control_policy_action_pitch': float(safe_action.policy_action_norm[1]),
                    'control_executed_action_torque': float(safe_action.executed_action_norm[0]),
                    'control_executed_action_pitch': float(safe_action.executed_action_norm[1]),
                    **safe_action_metrics(
                        safe_action,
                        sent_action_real=np.asarray(
                            [
                                step_result.info.get('action_torque', safe_action.action_real[0]),
                                step_result.info.get('action_pitch', safe_action.action_real[1]),
                            ],
                            dtype=np.float64,
                        ),
                    ),
                    'control_unsafe_stop': float(safety_stop),
                }
            )
            if _is_finite_transition(obs, action_np, step_result.reward, step_result.obs) and _should_store_transition(step_result.info, train_cfg):
                replay_buffer.add(obs, action_np, step_result.reward, step_result.obs, step_result.done)
            else:
                replay_reject_count += 1
            obs = step_result.obs
            done = step_result.done or safety_stop
            episode_reward += step_result.reward
            step_in_episode += 1
            total_steps += 1
            episode_metrics.append(step_result.info)
            logger.log_metrics('train_step', total_steps, step_result.info | {'reward_total': step_result.reward})
            for update_metrics in update_metrics_records:
                logger.log_metrics('train_update', total_steps, update_metrics)
            if step_result.info.get('env_invalid_prediction', 0.0) > 0.0 or step_result.info.get('env_invalid_reward', 0.0) > 0.0:
                invalid_transition_count += 1
            if step_in_episode == 1 or step_in_episode % step_log_every == 0 or done:
                print(
                    f'[Train] episode={episode} step={step_in_episode} total_steps={total_steps} '
                    f'reward={episode_reward:.4f} power={step_result.info.get("power", 0.0):.4f} '
                    f'invalid_steps={invalid_transition_count} replay_rejects={replay_reject_count}',
                    flush=True,
                )

            step_end_time = time.perf_counter()
            last_step_end_time = step_end_time
            step_time_ms = (step_end_time - step_start_time) * 1000.0
            action_time_ms = (action_ready_time - step_start_time) * 1000.0
            env_step_time_ms = (env_step_end_time - action_ready_time) * 1000.0
            step_overrun_ms = max(0.0, step_time_ms - step_deadline_ms) if step_deadline_ms > 0.0 else 0.0
            step_overrun = float(step_overrun_ms > 0.0)
            if step_overrun:
                step_overrun_count += 1
                step_overrun_total_ms += step_overrun_ms
                all_step_overrun_count += 1
                all_step_overrun_total_ms += step_overrun_ms
                all_step_overrun_max_ms = max(all_step_overrun_max_ms, step_overrun_ms)
            episode_step_times_ms.append(step_time_ms)
            all_step_times_ms.append(step_time_ms)
            timing_metrics = {
                'step_time_ms': step_time_ms,
                'action_time_ms': action_time_ms,
                'env_step_time_ms': env_step_time_ms,
                'update_time_ms': update_time_ms,
                'udp_receive_wait_ms': float(step_result.info.get('udp_receive_wait_ms', 0.0)),
                'logical_sim_step_ms': logical_sim_step_ms,
                'step_deadline_ms': step_deadline_ms,
                'step_overrun': step_overrun,
                'step_overrun_ms': step_overrun_ms,
                'update_scheduled': float(update_scheduled),
            }
            logger.log_metrics('train_timing', total_steps, timing_metrics)
            if step_overrun:
                logger.log_metrics('train_step_overrun', total_steps, timing_metrics | {'episode': float(episode), 'step_in_episode': float(step_in_episode)})
            if (timing_console_overruns and step_overrun) or (
                timing_console_every > 0 and step_in_episode % timing_console_every == 0
            ):
                print(
                    f'[Timing] episode={episode} step={step_in_episode} total_steps={total_steps} '
                    f'step_time_ms={step_time_ms:.3f} action_ms={action_time_ms:.3f} '
                    f'env_ms={env_step_time_ms:.3f} update_ms={update_time_ms:.3f} '
                    f'overrun_ms={step_overrun_ms:.3f}',
                    flush=True,
                )

        summary = summarize_metrics(episode_metrics)
        episode_time_ms = (time.perf_counter() - episode_start_time) * 1000.0
        all_episode_times_ms.append(episode_time_ms)
        if episode_step_times_ms:
            step_time_array = np.asarray(episode_step_times_ms, dtype=np.float64)
            summary['step_time_ms_mean'] = float(step_time_array.mean())
            summary['step_time_ms_min'] = float(step_time_array.min())
            summary['step_time_ms_max'] = float(step_time_array.max())
            summary['step_time_ms_p95'] = float(np.percentile(step_time_array, 95.0))
        summary['episode_time_ms'] = float(episode_time_ms)
        summary['episode_reward'] = episode_reward
        summary['episode_length'] = step_in_episode
        summary['invalid_transition_count'] = float(invalid_transition_count)
        summary['replay_reject_count'] = float(replay_reject_count)
        summary['update_skip_count'] = float(update_skip_count)
        summary['logical_sim_step_ms'] = logical_sim_step_ms
        summary['step_deadline_ms'] = step_deadline_ms
        summary['step_overrun_count'] = float(step_overrun_count)
        summary['step_overrun_total_ms'] = step_overrun_total_ms
        summary['td3_bc_weight_schedule'] = float(current_td3_bc_weight)
        logger.log_episode(episode, summary)
        print(
            f'[Episode] episode={episode} total_steps={total_steps} '
            f'{_format_metrics(summary, ["episode_reward", "power_mean", "reward_total_mean"])} '
            f'{_format_metrics(summary, ["episode_time_ms", "step_time_ms_mean", "step_time_ms_min", "step_time_ms_max", "step_time_ms_p95"])} '
            f'invalid_steps={invalid_transition_count} replay_rejects={replay_reject_count} update_skips={update_skip_count} '
            f'step_overruns={step_overrun_count} overrun_ms={step_overrun_total_ms:.3f} '
            f'{_format_metrics(last_update_metrics, ["actor_loss", "critic1_loss", "critic2_loss", "critic_loss", "actor_bc_weight"])}',
            flush=True,
        )

        if episode % int(train_cfg.get('save_every', 5)) == 0:
            logger.save_checkpoint(
                f'episode_{episode}.pt',
                {
                    'agent': agent.state_dict(),
                    'config': config,
                    'data_schema_version': data_schema_version,
                    'control_fingerprint': controller.control_fingerprint,
                    'episode': episode,
                    'total_steps': total_steps,
                },
            )
            logger.flush()
            print(f'[Checkpoint] saved episode={episode}', flush=True)

        if episode % int(train_cfg.get('eval_every', 5)) == 0:
            eval_summary = evaluate_policy(agent, eval_env, episodes=eval_episodes, logger=logger, prefix='eval', controller=controller)
            logger.save_summary(f'eval_episode_{episode}.json', eval_summary)
            if save_best_eval_checkpoint:
                eval_score = _get_eval_checkpoint_score(
                    eval_summary,
                    selection_metric=eval_checkpoint_metric,
                    expected_valid_episodes=eval_episodes,
                )
                if eval_score is not None and eval_score > best_eval_score:
                    best_eval_score = eval_score
                    logger.save_checkpoint(
                        best_checkpoint_name,
                        {
                            'agent': agent.state_dict(),
                            'config': config,
                            'data_schema_version': data_schema_version,
                            'control_fingerprint': controller.control_fingerprint,
                            'episode': episode,
                            'total_steps': total_steps,
                            'eval_checkpoint_metric': eval_checkpoint_metric,
                            'eval_checkpoint_score': eval_score,
                            'eval_summary': eval_summary,
                        },
                    )
                    logger.save_summary(
                        'best_eval_summary.json',
                        eval_summary
                        | {
                            'best_eval_episode': float(episode),
                            'best_eval_total_steps': float(total_steps),
                            'best_eval_selection_value': eval_score,
                        },
                    )
                    print(
                        f'[Best-Eval] checkpoint={best_checkpoint_name} episode={episode} '
                        f'{eval_checkpoint_metric}={eval_score:.4f}',
                        flush=True,
                    )
            logger.flush()
            print(
                f'[Eval] episode={episode} '
                f'{_format_metrics(eval_summary, ["reward_mean", "reward_std", "episode_invalid_mean", "power_mean_mean", "reward_total_mean_mean"])}',
                flush=True,
            )

    timing_overall_summary: dict[str, float] = {}
    if all_step_times_ms:
        all_step_array = np.asarray(all_step_times_ms, dtype=np.float64)
        timing_overall_summary['step_time_ms_mean'] = float(all_step_array.mean())
        timing_overall_summary['step_time_ms_min'] = float(all_step_array.min())
        timing_overall_summary['step_time_ms_max'] = float(all_step_array.max())
        timing_overall_summary['step_time_ms_range'] = float(all_step_array.max() - all_step_array.min())
    if all_episode_times_ms:
        all_episode_array = np.asarray(all_episode_times_ms, dtype=np.float64)
        timing_overall_summary['episode_time_ms_mean'] = float(all_episode_array.mean())
        timing_overall_summary['episode_time_ms_min'] = float(all_episode_array.min())
        timing_overall_summary['episode_time_ms_max'] = float(all_episode_array.max())
        timing_overall_summary['episode_time_ms_range'] = float(all_episode_array.max() - all_episode_array.min())
    if first_step_start_time is not None and last_step_end_time is not None and last_step_end_time >= first_step_start_time:
        timing_overall_summary['all_steps_wall_time_ms'] = float((last_step_end_time - first_step_start_time) * 1000.0)
    if step_deadline_ms > 0.0:
        timing_overall_summary['logical_sim_step_ms'] = logical_sim_step_ms
        timing_overall_summary['step_deadline_ms'] = step_deadline_ms
        timing_overall_summary['step_overrun_count'] = float(all_step_overrun_count)
        timing_overall_summary['step_overrun_total_ms'] = all_step_overrun_total_ms
        timing_overall_summary['step_overrun_max_ms'] = all_step_overrun_max_ms
    if timing_overall_summary:
        logger.save_summary('timing_overall_summary.json', timing_overall_summary)
        print(
            '[Timing-Summary] '
            f'{_format_metrics(timing_overall_summary, ["step_time_ms_mean", "step_time_ms_min", "step_time_ms_max", "step_time_ms_range", "episode_time_ms_mean", "episode_time_ms_min", "episode_time_ms_max", "episode_time_ms_range", "all_steps_wall_time_ms", "step_deadline_ms", "step_overrun_count", "step_overrun_total_ms", "step_overrun_max_ms"])}',
            flush=True,
        )

    logger.flush()
    print(f'[Train] finished output={logger.output_dir}', flush=True)


if __name__ == '__main__':
    main()
