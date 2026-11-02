"""Separate real-Bladed launch evidence for the development shadow backend.

Neither the D0-R result nor Linux fixture success satisfies verified gates.
Operator-confirmed development is a separate, training-only identity-bound route.
"""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import platform
import re
from typing import Any

import numpy as np
import torch

from RL.utils.physical_load_margin import document_sha256
from RL.utils.physical_load_shadow import physical_backend_selected, preflight_physical_config
from RL.utils.physical_load_shadow_artifact import artifact_path, file_sha256, strict_json


LAUNCH_SCHEMA = 'physical_load_shadow_bladed_launch_evidence_v1'
OPERATOR_SCHEMA = 'physical_shadow_operator_development_receipt_v2'
OPERATOR_ROUTE = 'operator_confirmed_4wind4round600_training_development_v1'
OPERATOR_TEMPLATE = ('RL/configs/physical_shadow_state20_v1_final_checked/'
                     'state20_physical_4wind_4round_600s_training.json')
FROZEN_R4_SCHEMA = 'physical_shadow_round4_frozen_actor_operator_development_receipt_v1'
FROZEN_R4_ROUTE = 'operator_confirmed_round4_frozen_actor_4wind1round600_shadow_inference_v1'
FROZEN_R4_TEMPLATE = ('RL/configs/physical_shadow_state20_v1_final_checked/'
                      'state20_physical_4wind_1round_600s_frozen_diagnostic.json')
OPERATOR_23WIND_SCHEMA = 'physical_shadow_operator_23wind_round600_training_receipt_v2'
OPERATOR_23WIND_ROUTE = 'operator_confirmed_23wind_round600_training_development_v2'
OPERATOR_23WIND_RELEASE_ID = 'WT15_STATE20_RL_20261023_R2'
OPERATOR_23WIND_TEMPLATE = (
    f'RL/configs/{OPERATOR_23WIND_RELEASE_ID}/state20_physical_23wind_round600_training.json'
)
OPERATOR_23WIND_PROFILE_BUILD_REPORT = (
    f'RL/configs/{OPERATOR_23WIND_RELEASE_ID}/profile_build_report.json'
)
OPERATOR_23WIND_RECEIPT_NAME = 'operator_physical_receipt.json'
OPERATOR_TRAINING_ROUTES = (OPERATOR_ROUTE, OPERATOR_23WIND_ROUTE)
OPERATOR_23WIND_IDENTITY_FILES = (
    'RL/build_state20_queue_configs.py',
    'RL/build_physical_load_shadow_profiles.py',
    'RL/evaluate_physical_load_shadow.py',
    'RL/prepare_physical_operator_run.py',
    'RL/finalize_physical_operator_run.py',
    'RL/utils/physical_load_shadow_launch.py',
    'RL/utils/physical_queue_schedule.py',
    'bladed_env/udptest/udptest/controller.cpp',
    'bladed_env/udptest/Release/udptest.dll',
)
OPERATOR_23WIND_SOURCE_PROFILE = (
    'RL/experiments/artifacts/state20_queue_profiles_operator_23wind_round600_v2/'
    'state20_23wind_round600_source.json'
)
OPERATOR_23WIND_DLL_IDENTITY_PATH = 'bladed_env/udptest/Release/udptest.dll'
FULL_23WIND_BASE_ORDER = (
    12, 11, 13, 10, 14, 9, 15, 8, 16, 7, 17, 6,
    18, 5, 19, 4, 20, 3, 21, 22, 23, 24, 25,
)
OPERATOR_23WIND_ROLE = {
    'stage': 'moving_actor_23wind_training_diagnostic',
    'independent_audit': False,
    'new_independent_seed': False,
    'policy_acceptance': False,
    'soft_enabled': False,
    'hard_enabled': False,
}
OPERATOR_ATTESTATION = {
    'dll_personally_confirmed': True, 'protocol_personally_confirmed': True,
    'evidence_kind': 'operator_attestation_not_independently_verified',
    'manual_project_and_wind_selection_required_each_slot': True,
    'project_wind_seed_personally_confirmed_matches_pid': True,
}
REQUIRED_GATES = (
    'offline_physical_shadow', 'windows_loaded_release_x64_dll',
    'queue_v2_dynamic_rounds_and_terminal_frame', 'engineering_and_action_provenance',
    'target_shadow_latency_and_resource_budget', 'overall_v1_permission',
)
RUNTIME_FILES = (
    'RL/utils/physical_load_shadow_artifact.py', 'RL/utils/physical_load_shadow.py',
    'RL/utils/physical_load_shadow_launch.py', 'RL/utils/physical_load_margin.py',
    'RL/utils/physical_load_margin_dynamic.py', 'RL/train_rl.py',
    'RL/infer/run_bladed_inference.py', 'RL/envs/wind_turbine_env.py',
    'RL/utils/operating_state.py', 'RL/utils/safe_residual_control.py',
    'RL/utils/paired_pid_reference.py', 'RL/utils/queue_case_identity.py',
    'RL/utils/control_contract.py', 'RL/utils/constants.py',
    'RL/agents/td3.py', 'RL/models/actor_mlp.py', 'RL/models/critic_mlp.py',
    'RL/utils/reward.py', 'RL/utils/region_aware_baseline.py',
    'RL/utils/state_action_artifact.py', 'RL/utils/observation.py',
    'RL/utils/normalize.py', 'RL/utils/power_fields.py',
    'RL/utils/control_execution.py', 'RL/utils/critic_q_health.py',
    'RL/utils/dual_constraint.py', 'RL/utils/condition_queue.py',
    'RL/utils/queue_manifest.py', 'RL/replay/replay_buffer.py',
    'RL/utils/config.py', 'RL/utils/set_seed.py', 'RL/utils/logger.py',
    'RL/utils/physical_queue_schedule.py',
    'RL/prepare_physical_shadow_development.py',
    'RL/build_physical_load_shadow_profiles.py',
    'RL/evaluate_b4_residual_bc.py', 'RL/data/offline_dataset.py',
)


def physical_target_identity(device_name: str) -> dict[str, Any]:
    """Local execution/resource identity, not a new measured control variable."""
    device = torch.device(device_name)
    gpu = None
    if device.type == 'cuda' and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(device)
        gpu = {'index': int(device.index or 0), 'name': props.name,
               'total_memory_bytes': int(props.total_memory),
               'capability': [int(props.major), int(props.minor)]}
    return {
        'host_identity_sha256': hashlib.sha256(platform.node().encode('utf-8')).hexdigest(),
        'system': platform.system(), 'release': platform.release(), 'machine': platform.machine(),
        'processor': platform.processor(), 'logical_cpu_count': os.cpu_count(),
        'python_version': platform.python_version(), 'numpy_version': np.__version__,
        'torch_version': str(torch.__version__), 'cuda_runtime': torch.version.cuda,
        'device': str(device), 'gpu': gpu,
        'torch_cpu_threads': torch.get_num_threads(),
        'thread_environment': {name: os.environ.get(name) for name in
                               ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')},
    }


def load_verified_physical_checkpoint(path: Path, expected_sha256: str) -> dict[str, Any]:
    """Load the exact authorized bytes; never authorize one file and run another dict."""
    if file_sha256(path) != expected_sha256:
        raise ValueError('Physical execution checkpoint SHA mismatch before load')
    checkpoint = torch.load(path, map_location='cpu')
    if file_sha256(path) != expected_sha256 or not isinstance(checkpoint, dict):
        raise ValueError('Physical execution checkpoint changed or has invalid format')
    return checkpoint


def _same_state(actual: Any, expected: Any) -> bool:
    if isinstance(actual, torch.Tensor) or isinstance(expected, torch.Tensor):
        return (isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor) and
                actual.dtype == expected.dtype and actual.shape == expected.shape and
                torch.equal(actual.detach().cpu(), expected.detach().cpu()))
    if isinstance(actual, dict) or isinstance(expected, dict):
        return (isinstance(actual, dict) and isinstance(expected, dict) and actual.keys() == expected.keys() and
                all(_same_state(actual[key], expected[key]) for key in actual))
    if isinstance(actual, (list, tuple)) or isinstance(expected, (list, tuple)):
        return (type(actual) is type(expected) and len(actual) == len(expected) and
                all(_same_state(a, b) for a, b in zip(actual, expected)))
    return type(actual) is type(expected) and actual == expected


def validate_physical_agent_parameters(agent: Any, config: dict[str, Any], device: torch.device) -> None:
    """TD3 runtime scalars are not contained in its checkpoint state_dict."""
    settings = config['agent']
    if agent.device != device:
        raise ValueError('Physical training actual agent device mismatch')
    defaults = {
        'gamma': 0.99, 'tau': 0.005, 'policy_noise': 0.2, 'noise_clip': 0.5,
        'policy_delay': 2, 'td3_bc_enabled': False, 'td3_bc_alpha': 2.5,
        'td3_bc_weight': 1.0, 'td3_bc_loss_type': 'smooth_l1', 'td3_bc_loss_scale': 1.0,
        'td3_bc_target_ratio': 0.0, 'actor_residual_target_scale': 1.0,
        'td3_bc_max_effective_weight': 1.0e6, 'max_target_q_abs_mean': 0.0,
        'max_actor_q_abs_mean': 0.0, 'critic_loss_type': 'mse',
        'critic_loss_delta': 1.0, 'critic_grad_norm': 10.0,
    }
    for name, default in defaults.items():
        expected = type(default)(settings.get(name, default))
        if isinstance(expected, str):
            expected = expected.lower()
        if getattr(agent, name, None) != expected:
            raise ValueError(f'Physical training actual agent runtime parameter mismatch: {name}')


def validate_physical_training_binding(config: dict[str, Any], env: Any, controller: Any,
                                       agent: Any, shadow: Any, device: torch.device, *,
                                       project_root: Path) -> None:
    """Bind the objects actually executed by the public training API."""
    if config.get('device') != str(device):
        raise ValueError('Physical training actual device/config mismatch')
    preflight = preflight_physical_config(config, project_root=project_root)
    if preflight is None or shadow is None:
        raise ValueError('Physical training requires its selected, preflighted worker')
    artifacts, reference = preflight
    settings = config['load_predictor']
    if (shadow.artifacts.bundle_file_sha256 != artifacts.bundle_file_sha256 or
            shadow.artifacts.dynamic_file_sha256 != artifacts.dynamic_file_sha256 or
            shadow.evaluation_reference is None or
            shadow.evaluation_reference.manifest_sha256 != reference.manifest_sha256 or
            document_sha256(shadow.artifacts.model.to_document()) != document_sha256(artifacts.model.to_document()) or
            shadow.core.stride != settings.get('origin_stride_frames', 5) or
            shadow._queue.maxsize != settings.get('worker_queue_capacity', 256) or
            shadow.core.records.maxlen != settings.get('condition_record_capacity', 16384) or
            shadow._limits != (settings.get('max_records', 250000), settings.get('max_record_bytes', 32768),
                               settings.get('max_total_bytes', 268435456))):
        raise ValueError('Physical training worker/bundle/reference/budget binding mismatch')
    if getattr(env, 'initialization_config_sha256', None) != document_sha256(config['env']):
        raise ValueError('Physical training actual environment/config binding mismatch')
    from RL.agents import TD3Agent
    from RL.utils.safe_residual_control import SafeResidualController
    if not isinstance(agent, TD3Agent):
        raise ValueError('Physical training requires the configured TD3 implementation')
    validate_physical_agent_parameters(agent, config, device)
    configured_controller = SafeResidualController(config['env'], config.get('control', {}))
    for name in ('control_fingerprint', 'actor_control_fingerprint', 'online_training_fingerprint'):
        if getattr(controller, name, None) != getattr(configured_controller, name):
            raise ValueError('Physical training actual controller/config binding mismatch')
    if (not np.array_equal(controller.obs_median, agent.actor.obs_median.detach().cpu().numpy()) or
            not np.array_equal(controller.obs_scale, agent.actor.obs_iqr.detach().cpu().numpy())):
        raise ValueError('Physical training actual controller/Actor observation normalization mismatch')
    checkpoint_path = artifact_path(config['bc']['resume_checkpoint'], project_root)
    checkpoint = load_verified_physical_checkpoint(checkpoint_path,
        config['state20_queue_contract']['bc_warmstart_checkpoint_sha256'])
    state = checkpoint.get('agent', checkpoint)
    if not isinstance(state, dict) or 'actor' not in state:
        raise ValueError('Physical training authorized checkpoint has no agent state')
    expected = dict(state)
    expected.setdefault('actor_execution', expected['actor'])
    expected.setdefault('update_step', 0)
    if not _same_state(agent.state_dict(), expected):
        raise ValueError('Physical training actual agent differs from authorized initial checkpoint')


def physical_runtime_identity(config: dict[str, Any], project_root: Path, *,
                              checkpoint_path: Path | None = None,
                              baseline_only: bool = False) -> dict[str, Any]:
    predictor = config['load_predictor']
    queue = config['queue']
    manifest = artifact_path(queue['manifest_path'], project_root)
    bc = artifact_path(config['bc']['resume_checkpoint'], project_root)
    paired = artifact_path(config['state20_queue_contract']['paired_reference_path'], project_root)
    if (file_sha256(bc) != config['state20_queue_contract']['bc_warmstart_checkpoint_sha256'] or
            file_sha256(paired) != config['state20_queue_contract']['paired_reference_sha256']):
        raise ValueError('Physical launch Actor/reference source SHA mismatch')
    runtime_config = {key: value for key, value in config.items()
                      if key not in ('physical_shadow_launch', 'output_dir', 'experiment_name',
                                     'resolved_experiment_name')}
    identity = {
        'bundle_sha256': predictor['bundle_sha256'],
        'resolved_control_contract_fingerprint': config['control_contract_fingerprint'],
        'paired_pid_reference_sha256': config['state20_queue_contract']['paired_reference_sha256'],
        'bc_checkpoint_sha256': config['state20_queue_contract']['bc_warmstart_checkpoint_sha256'],
        'queue_manifest_sha256': file_sha256(manifest),
        'effective_runtime_config_sha256': document_sha256(runtime_config),
        'execution_entry': 'inference' if checkpoint_path is not None else 'training',
        'actor_checkpoint_sha256': file_sha256(checkpoint_path) if checkpoint_path is not None else file_sha256(bc),
        'baseline_only': bool(baseline_only),
        'execution_target': physical_target_identity(str(config['device'])),
        'runtime_code_sha256': {name: file_sha256(project_root / name) for name in RUNTIME_FILES},
    }
    if config.get('physical_shadow_launch', {}).get('route') == OPERATOR_23WIND_ROUTE:
        release_id = config.get('release_id')
        if release_id != OPERATOR_23WIND_RELEASE_ID:
            raise ValueError('23-wind operator config is not part of the current unified release')
        source_profile_path = project_root / OPERATOR_23WIND_SOURCE_PROFILE
        if not source_profile_path.is_file():
            raise FileNotFoundError('23-wind source profile is missing')
        identity['release_id'] = release_id
        identity['operator_23wind_preparation_code_sha256'] = {
            name: file_sha256(project_root / name) for name in OPERATOR_23WIND_IDENTITY_FILES
        }
        identity['operator_23wind_source_profile_sha256'] = file_sha256(source_profile_path)
    return identity


def validate_physical_launch(config: dict[str, Any], *, project_root: Path,
                             checkpoint_path: Path | None = None,
                             baseline_only: bool = False) -> dict[str, Any] | None:
    if not physical_backend_selected(config):
        if config.get('physical_shadow_launch', {}).get('route') in OPERATOR_TRAINING_ROUTES:
            raise ValueError('Operator development requires physical shadow backend')
        return
    gate = config.get('physical_shadow_launch', {})
    if not isinstance(gate, dict) or gate.get('authorized') is not True:
        raise ValueError('Physical shadow Bladed launch is BLOCKED: missing offline/Windows/provenance/'
                         'target-resource/overall-V1 evidence. No NN fallback or automatic launch.')
    value, digest = gate.get('evidence_path'), gate.get('evidence_sha256')
    if not isinstance(value, str) or not value or not isinstance(digest, str):
        raise ValueError('Physical shadow launch requires an explicit hashed evidence document')
    path = artifact_path(value, project_root)
    if file_sha256(path) != digest:
        raise ValueError('Physical shadow launch evidence SHA mismatch')
    evidence = strict_json(path)
    identity = physical_runtime_identity(config, project_root, checkpoint_path=checkpoint_path,
                                         baseline_only=baseline_only)
    if gate.get('route') == FROZEN_R4_ROUTE:
        return validate_round4_frozen_actor_development(
            config, evidence, identity, project_root,
            checkpoint_path=checkpoint_path, baseline_only=baseline_only,
        )
    if gate.get('route') == OPERATOR_ROUTE:
        return validate_operator_development(config, evidence, identity, project_root,
                                             checkpoint_path=checkpoint_path, baseline_only=baseline_only)
    if gate.get('route') == OPERATOR_23WIND_ROUTE:
        validated = validate_operator_23wind_development(
            config, evidence, identity, project_root,
            checkpoint_path=checkpoint_path, baseline_only=baseline_only,
        )
        gate_path = artifact_path(config['physical_shadow_launch']['evidence_path'], project_root).resolve()
        gate_digest = config['physical_shadow_launch'].get('evidence_sha256')
        if (gate_digest != file_sha256(gate_path) or
                gate_path != (project_root / 'RL/configs' / str(config['experiment_name']) /
                                  OPERATOR_23WIND_RECEIPT_NAME).resolve()):
                raise ValueError('23-wind physical launch gate receipt path/hash is not bound to this RUN_ID')
        return validated
    if config.get('queue', {}).get('case_identity_validation', {}).get('wind_mismatch_policy', 'abort') != 'abort':
        raise ValueError('Verified launch cannot use development wind warnings')
    if (evidence.get('schema_version') != LAUNCH_SCHEMA or
            evidence.get('purpose') != 'real_bladed_development_shadow_only' or
            evidence.get('runtime_identity') != identity or
            evidence.get('policy_acceptance') is not False or
            set(evidence.get('gates', {})) != set(REQUIRED_GATES)):
        raise ValueError('Physical shadow launch evidence scope/identity mismatch')
    for name in REQUIRED_GATES:
        entry = evidence['gates'][name]
        if entry.get('status') != 'PASS' or entry.get('runtime_identity') != identity:
            raise ValueError(f'Physical shadow launch gate has not passed: {name}')
        report_path = artifact_path(entry['evidence_path'], project_root)
        if file_sha256(report_path) != entry['evidence_sha256']:
            raise ValueError(f'Physical shadow source evidence SHA mismatch: {name}')
        report = strict_json(report_path)
        if (report.get('gate_id') != name or report.get('status') != 'PASS' or
                report.get('runtime_identity') != identity):
            raise ValueError(f'Physical shadow source evidence does not substantiate PASS: {name}')
        if name != 'offline_physical_shadow' and report.get('evidence_kind') != 'verified_target_evidence':
            raise ValueError(f'Synthetic/Linux fixture evidence cannot satisfy target gate: {name}')
        if not report.get('verified_at_utc') or not report.get('verification_reference'):
            raise ValueError(f'Physical shadow source evidence lacks a verification receipt: {name}')
    return identity


def validate_round4_frozen_actor_development(
    config: dict[str, Any], receipt: dict[str, Any], identity: dict[str, Any], project_root: Path,
    *, checkpoint_path: Path | None, baseline_only: bool,
) -> dict[str, Any]:
    """Validate the narrow non-training, frozen round-4 physical-shadow route."""
    import copy

    from RL.utils.queue_manifest import expand_queue_manifest

    if checkpoint_path is None or baseline_only:
        raise ValueError('Frozen round-4 route requires a persisted policy checkpoint, not baseline-only')
    if any(os.environ.get(name) for name in ('RL_EXPERIMENT_SUFFIX', 'RL_DATA_ROOT', 'RL_DYNAMICS_CHECKPOINT')):
        raise ValueError('Frozen round-4 route forbids identity-changing RL_* environment overrides')
    if config.get('physical_shadow_run_role') != {
            'stage': 'round4_frozen_actor_event_window_validation',
            'source_run_id': receipt.get('source_run', {}).get('run_id'),
            'source_round': 4,
            'actor_frozen': True,
            'optimizer_updates': False,
            'independent_audit': False,
            'policy_acceptance': False,
            'soft_enabled': False,
            'hard_enabled': False,
    }:
        raise ValueError('Frozen round-4 run role metadata is not the exact development-only scope')
    if config.get('policy_acceptance') is not False or config.get('physical_shadow_run_role', {}).get(
            'policy_acceptance') is not False:
        raise ValueError('Frozen round-4 route cannot claim policy acceptance')
    if config.get('train', {}).get('online_rl_enabled') is not False:
        raise ValueError('Frozen round-4 inference route requires online training disabled')
    if config.get('queue', {}).get('case_identity_validation', {}).get('wind_mismatch_policy') != 'warn_development_only':
        raise ValueError('Frozen round-4 operator development requires wind warnings to remain unverified')
    scope = config.get('physical_shadow_execution_scope')
    if (not isinstance(scope, dict) or scope.get('entry') != 'inference' or
            scope.get('episodes') != 4 or scope.get('max_steps') is not None or
            scope.get('baseline_only') is not False):
        raise ValueError('Frozen round-4 launch is restricted to exactly four uncapped queue conditions')
    output_dir = Path(str(scope.get('output_dir', ''))).expanduser().resolve()
    run_id = config.get('experiment_name')
    if not isinstance(run_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', run_id):
        raise ValueError('Frozen round-4 inference requires a valid RUN_ID')
    config_dir = (project_root / 'RL/configs' / run_id).resolve()
    experiment_dir = (project_root / 'RL/experiments' / run_id).resolve()
    if (config_dir.parent != (project_root / 'RL/configs').resolve() or
            output_dir != experiment_dir or
            artifact_path(config.get('output_dir', ''), project_root).resolve() !=
            (project_root / 'RL/experiments').resolve() or
            config.get('append_timestamp') is not False):
        raise ValueError('Frozen round-4 route requires canonical RL/configs/<RUN_ID> and RL/experiments/<RUN_ID>')
    if experiment_dir.exists() and any(experiment_dir.iterdir()):
        raise FileExistsError(f'Frozen round-4 experiment output must be new and empty: {experiment_dir}')
    evidence_path = artifact_path(config['physical_shadow_launch']['evidence_path'], project_root).resolve()
    if evidence_path != config_dir / 'round4_inference_receipt.json':
        raise ValueError('Frozen round-4 receipt must be saved beside its inference config')
    inference_config_path = config_dir / 'inference.json'
    if not inference_config_path.is_file():
        raise FileNotFoundError('Frozen round-4 inference.json must exist before launch')
    persisted_config = strict_json(inference_config_path)
    config_for_disk = {key: value for key, value in config.items() if key != 'physical_shadow_execution_scope'}
    if persisted_config != config_for_disk:
        raise ValueError('Frozen round-4 effective inference config differs from the bound run config')

    template = strict_json(project_root / FROZEN_R4_TEMPLATE)
    expected = copy.deepcopy(template)
    actual = copy.deepcopy(config)
    for document in (expected, actual):
        for key in ('physical_shadow_launch', 'output_dir', 'experiment_name', 'append_timestamp',
                    'resolved_experiment_name', 'device', 'physical_shadow_execution_scope',
                    'physical_shadow_run_role', 'policy_acceptance'):
            document.pop(key, None)
    expected['queue']['case_identity_validation']['wind_mismatch_policy'] = 'warn_development_only'
    expected['logging'].update(
        queue_event_full_state_enabled=True,
        queue_event_pre_frames=50,
        queue_event_post_frames=25,
        queue_event_window_max_record_bytes=512 * 1024,
        queue_event_windows_max_total_bytes=512 * 1024 * 1024,
    )
    if actual != expected:
        raise ValueError('Frozen round-4 inference config differs from the exact no-training diagnostic scope')
    if config['logging'].get('queue_event_full_state_enabled') is not True:
        raise ValueError('Frozen round-4 validation requires bounded full existing-state event logging')
    slots = expand_queue_manifest(strict_json(artifact_path(config['queue']['manifest_path'], project_root)))
    if ([slot.condition_id for slot in slots] != [8, 11, 12, 16] or
            any(slot.round_id != 1 or slot.train_sim_s != 600.0 for slot in slots) or len(slots) != 4):
        raise ValueError('Frozen round-4 validation requires exactly one 600s round over winds 8/11/12/16')

    source = receipt.get('source_run')
    if not isinstance(source, dict):
        raise ValueError('Frozen round-4 receipt is missing completed source-run provenance')
    source_run_id = str(source.get('run_id', ''))
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', source_run_id):
        raise ValueError('Frozen round-4 receipt source RUN_ID is invalid')
    source_config_path = (project_root / 'RL/configs' / source_run_id / 'train.json').resolve()
    source_receipt_path = (project_root / 'RL/configs' / source_run_id / 'operator_development_receipt.json').resolve()
    source_experiment_dir = (project_root / 'RL/experiments' / source_run_id).resolve()
    if (source.get('config_path') != str(source_config_path) or
            source.get('operator_receipt_path') != str(source_receipt_path) or
            source.get('experiment_dir') != str(source_experiment_dir) or
            not source_config_path.is_file() or not source_receipt_path.is_file()):
        raise ValueError('Frozen round-4 source run paths are not canonical or do not exist')
    source_config_sha = file_sha256(source_config_path)
    source_receipt_sha = file_sha256(source_receipt_path)
    source_config = strict_json(source_config_path)
    source_receipt = strict_json(source_receipt_path)
    source_effective_config = {key: value for key, value in source_config.items()
                               if key not in ('physical_shadow_launch', 'output_dir', 'experiment_name',
                                              'append_timestamp', 'resolved_experiment_name')}
    historical_identity = source_receipt.get('runtime_identity', {})
    source_manifest = artifact_path(source_config['queue']['manifest_path'], project_root)
    source_bundle = artifact_path(source_config['load_predictor']['bundle_path'], project_root)
    if (source_receipt.get('schema_version') != OPERATOR_SCHEMA or
            source_receipt.get('route') != OPERATOR_ROUTE or
            source_receipt.get('independently_verified') is not False or
            source_receipt.get('policy_acceptance') is not False or
            source_receipt.get('run_id') != source_run_id or
            source_receipt.get('run_config') != str(source_config_path) or
            source_receipt.get('experiment_dir') != str(source_experiment_dir) or
            historical_identity.get('effective_runtime_config_sha256') != document_sha256(source_effective_config) or
            historical_identity.get('queue_manifest_sha256') != file_sha256(source_manifest) or
            historical_identity.get('bundle_sha256') != file_sha256(source_bundle) or
            source.get('config_sha256') != source_config_sha or
            source.get('operator_receipt_sha256') != source_receipt_sha):
        raise ValueError('Frozen round-4 source operator training receipt/config/bundle identity mismatch')
    source_training_summary_path = source_experiment_dir / 'queue_training_summary.json'
    source_round_summary_path = source_experiment_dir / 'queue_round_4_summary.json'
    round_checkpoint_path = source_experiment_dir / 'checkpoints' / 'queue_round_4.pt'
    for path in (source_training_summary_path, source_round_summary_path, round_checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(f'Frozen round-4 source evidence is missing: {path}')
    if (source.get('training_summary_sha256') != file_sha256(source_training_summary_path) or
            source.get('round_4_summary_sha256') != file_sha256(source_round_summary_path) or
            source.get('round_4_checkpoint_sha256') != file_sha256(round_checkpoint_path) or
            checkpoint_path.resolve() != round_checkpoint_path.resolve() or
            source.get('round_4_checkpoint_path') != str(round_checkpoint_path)):
        raise ValueError('Frozen round-4 checkpoint or training summary SHA/path mismatch')
    training_summary = strict_json(source_training_summary_path)
    round_summary = strict_json(source_round_summary_path)
    for summary_name, summary in (('queue_training_summary', training_summary),
                                  ('queue_round_4_summary', round_summary)):
        if (summary.get('queue_training_valid') != 1.0 or summary.get('queue_completed_rounds') != 4.0 or
                summary.get('queue_total_rounds') != 4.0 or summary.get('queue_skipped_condition_count') != 0.0 or
                summary.get('queue_retried_condition_attempt_count') != 0.0 or
                summary.get('queue_critic_q_health_passed') != 1.0 or
                summary.get('queue_critic_q_health_degraded') != 0.0):
            raise ValueError(f'Frozen round-4 source {summary_name} is not a completed valid four-round run')
    checkpoint = load_verified_physical_checkpoint(round_checkpoint_path, source['round_4_checkpoint_sha256'])
    queue_state = checkpoint.get('queue_state', {})
    agent_state = checkpoint.get('agent', {})
    checkpoint_config = checkpoint.get('config', {})
    if (checkpoint_config.get('control_contract_fingerprint') != config.get('control_contract_fingerprint') or
            checkpoint.get('data_schema_version') != config.get('data_schema_version') or
            checkpoint.get('control_fingerprint') != identity.get('control_fingerprint', checkpoint.get('control_fingerprint')) or
            queue_state.get('completed_rounds') != 4 or queue_state.get('total_rounds') != 4 or
            queue_state.get('training_valid') is not True or
            queue_state.get('skipped_condition_count', 0) != 0 or
            queue_state.get('retried_condition_attempt_count', 0) != 0 or
            not isinstance(agent_state, dict) or agent_state.get('update_step') != 12_000 or
            not isinstance(agent_state.get('actor'), dict) or
            not isinstance(agent_state.get('actor_execution'), dict) or
            not _same_state(agent_state['actor'], agent_state['actor_execution'])):
        raise ValueError('Round-4 checkpoint is incomplete, invalid, contract-incompatible, or not the executed frozen Actor')
    contract_controller = None
    from RL.utils.safe_residual_control import SafeResidualController
    contract_controller = SafeResidualController(config['env'], config.get('control', {}))
    if (checkpoint.get('control_fingerprint') != contract_controller.control_fingerprint or
            checkpoint.get('actor_control_fingerprint') != contract_controller.actor_control_fingerprint or
            checkpoint.get('control_contract_fingerprint') != contract_controller.resolved_control_contract_fingerprint):
        raise ValueError('Round-4 checkpoint control/Actor contract differs from frozen inference config')
    if (receipt.get('schema_version') != FROZEN_R4_SCHEMA or receipt.get('route') != FROZEN_R4_ROUTE or
            receipt.get('runtime_identity') != identity or receipt.get('source_run') != source or
            receipt.get('operator_attestation') != OPERATOR_ATTESTATION or
            receipt.get('independently_verified') is not False or
            receipt.get('policy_acceptance') is not False or
            receipt.get('optimizer_updates') is not False or
            receipt.get('actor_frozen') is not True or
            receipt.get('event_window_validation') != {
                'full_existing_state_enabled': True,
                'pre_frames': 50,
                'post_frames': 25,
                'max_record_bytes': 512 * 1024,
                'max_total_bytes': 512 * 1024 * 1024,
            } or
            receipt.get('run_id') != run_id or receipt.get('config_path') != str(inference_config_path) or
            receipt.get('experiment_dir') != str(experiment_dir) or
            receipt.get('created_at_utc') is None):
        raise ValueError('Frozen round-4 receipt scope/identity/attestation mismatch')
    return {**identity, 'route': FROZEN_R4_ROUTE, 'operator_development_authorized': True,
            'source_round_4_checkpoint_sha256': source['round_4_checkpoint_sha256']}


def validate_operator_scope(config: dict[str, Any], project_root: Path) -> None:
    """Only the prepared copy of the frozen 16-slot moving-Actor profile is allowed."""
    import copy
    from RL.utils.control_contract import resolve_configured_control_contract
    from RL.utils.queue_manifest import expand_queue_manifest
    template = strict_json(project_root / OPERATOR_TEMPLATE)
    effective = copy.deepcopy(config)
    # Sole development exception; tolerance, time alignment and all control
    # settings still have to equal the frozen template exactly.
    identity_settings = effective['queue']['case_identity_validation']
    policy = identity_settings.pop('wind_mismatch_policy', 'abort')
    if policy not in ('abort', 'warn_development_only'):
        raise ValueError('Operator development wind_mismatch_policy must be abort or warn_development_only')
    for key in ('physical_shadow_launch', 'output_dir', 'experiment_name', 'append_timestamp',
                'resolved_experiment_name', 'device'):
        effective.pop(key, None)
        template.pop(key, None)
    # The runner absolutizes data.root_dir. New bundle/manifest copies must have
    # exactly the same bytes as the frozen compatible dependencies.
    if artifact_path(effective['data']['root_dir'], project_root).resolve() != artifact_path(
            template['data']['root_dir'], project_root).resolve():
        raise ValueError('Operator development data source mismatch')
    effective['data']['root_dir'] = template['data']['root_dir']
    for section, field in (('queue', 'manifest_path'), ('load_predictor', 'bundle_path')):
        if file_sha256(artifact_path(effective[section][field], project_root)) != file_sha256(
                artifact_path(template[section][field], project_root)):
            raise ValueError('Operator development frozen bundle/queue mismatch')
        effective[section][field] = template[section][field]
    if effective != template:
        raise ValueError('Operator development scope is restricted to frozen 4wind4round600 shadow training')
    slots = expand_queue_manifest(strict_json(artifact_path(config['queue']['manifest_path'], project_root)))
    expected = [8, 11, 12, 16, 11, 12, 16, 8, 12, 16, 8, 11, 16, 8, 11, 12]
    if [slot.condition_id for slot in slots] != expected or any(slot.train_sim_s != 600 for slot in slots):
        raise ValueError('Operator development requires exactly the smoke_4x4 600s schedule')
    if resolve_configured_control_contract(config, project_root=project_root) is None:
        raise ValueError('Operator development requires resolved control contract')
    preflight_physical_config(config, project_root=project_root)


def _operator_23wind_schedule_contract(config: dict[str, Any], project_root: Path) -> dict[str, Any]:
    from RL.utils.queue_manifest import expand_queue_manifest
    from RL.utils.physical_queue_schedule import (
        FULL_23WIND_BASE_ORDER,
        FULL_23WIND_CONDITIONS,
        FULL_23WIND_DURATION_S,
        FULL_23WIND_PROFILE,
        PHYSICAL_SHADOW_MAX_TOTAL_BYTES,
        PHYSICAL_SHADOW_ORIGIN_STRIDE_FRAMES,
        PHYSICAL_SHADOW_RECORD_MAX_BYTES,
        validate_full_23wind_rounds,
    )

    queue = config.get('queue', {})
    predictor = config.get('load_predictor', {})
    logging = config.get('logging', {})
    contract = config.get('state20_queue_contract', {})
    manifest_path = artifact_path(queue.get('manifest_path', ''), project_root).resolve()
    manifest = strict_json(manifest_path)
    template_path = (project_root / OPERATOR_23WIND_TEMPLATE).resolve()
    template = strict_json(template_path)
    profile_report_path = (project_root / OPERATOR_23WIND_PROFILE_BUILD_REPORT).resolve()
    profile_report = strict_json(profile_report_path)
    source_profile_path = (project_root / OPERATOR_23WIND_SOURCE_PROFILE).resolve()
    profile_entry = profile_report.get('profiles', {}).get(FULL_23WIND_PROFILE, {})
    expected_dll_sha = file_sha256(project_root / OPERATOR_23WIND_DLL_IDENTITY_PATH)
    if (profile_report.get('schema_version') != 'physical_load_shadow_profile_build_v1' or
            profile_report.get('release_id') != OPERATOR_23WIND_RELEASE_ID or
            profile_report.get('decision') != 'DEVELOPMENT_PROFILES_BUILT_REAL_LAUNCH_BLOCKED' or
            profile_report.get('bladed_started') is not False or
            profile_report.get('source_profile_path') != OPERATOR_23WIND_SOURCE_PROFILE or
            profile_report.get('source_profile_sha256') != file_sha256(source_profile_path) or
            profile_report.get('expected_windows_release_dll_path') != OPERATOR_23WIND_DLL_IDENTITY_PATH or
            profile_report.get('expected_windows_release_dll_sha256') != expected_dll_sha or
            profile_report.get('windows_queue_source_path') != 'bladed_env/udptest/udptest/controller.cpp' or
            profile_report.get('windows_queue_source_sha256') != file_sha256(
                project_root / 'bladed_env/udptest/udptest/controller.cpp') or
            template.get('release_id') != OPERATOR_23WIND_RELEASE_ID or
            profile_entry.get('release_id') != OPERATOR_23WIND_RELEASE_ID or
            profile_entry.get('config_path') != OPERATOR_23WIND_TEMPLATE or
            profile_entry.get('config_sha256') != file_sha256(template_path) or
            profile_entry.get('manifest_path') != template['queue']['manifest_path'] or
            profile_entry.get('manifest_sha256') != file_sha256(
                artifact_path(template['queue']['manifest_path'], project_root)) or
            profile_entry.get('slots') != template['queue']['total_condition_rounds'] *
                len(template['queue']['condition_ids']) or
            profile_entry.get('bladed_launch_authorized') is not False or
            profile_entry.get('runtime_identity') != physical_runtime_identity(template, project_root)):
        raise ValueError('Unified 23-wind profile/report/source/runtime identities do not close')

    slots = expand_queue_manifest(manifest)
    conditions = list(FULL_23WIND_CONDITIONS)
    rounds = validate_full_23wind_rounds(int(queue.get('total_condition_rounds', 0)))
    duration_s = FULL_23WIND_DURATION_S
    stride = int(predictor.get('origin_stride_frames', PHYSICAL_SHADOW_ORIGIN_STRIDE_FRAMES))
    step_duration_s = float(config['env']['bladed_step_duration_ms']) / 1000.0
    steps_per_slot_float = duration_s / step_duration_s
    if not math.isfinite(steps_per_slot_float) or not math.isclose(
        steps_per_slot_float, round(steps_per_slot_float), rel_tol=0.0, abs_tol=1.0e-9
    ):
        raise ValueError('Unified 23-wind slot duration must close on an integer number of simulation steps')
    steps_per_slot = int(round(steps_per_slot_float))
    if stride <= 0:
        raise ValueError('Physical shadow origin stride must be positive')
    expected_shadow_records = len(slots) * 2 * (steps_per_slot // stride + 1)
    max_shadow_records = int(predictor.get('max_records', 0))
    max_shadow_record_bytes = int(predictor.get('max_record_bytes', 0))
    max_shadow_bytes = int(predictor.get('max_total_bytes', 0))
    event_capacity = int(logging.get('queue_event_window_capacity', 0))
    event_record_bytes = int(logging.get('queue_event_window_max_record_bytes', 0))
    event_total_bytes = int(logging.get('queue_event_windows_max_total_bytes', 0))
    summary_record_bytes = int(logging.get('queue_condition_max_record_bytes', 0))
    summary_total_bytes = int(logging.get('queue_condition_summaries_max_total_bytes', 0))
    event_worst_case_bytes = len(slots) * event_capacity * event_record_bytes
    update_every_steps = int(config['train'].get('update_every_steps', 0))
    if update_every_steps <= 0:
        raise ValueError('train.update_every_steps must be positive')
    expected_updates_float = ((rounds - 1) * len(conditions) * steps_per_slot / update_every_steps)
    if not math.isclose(expected_updates_float, round(expected_updates_float), rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError('Unified 23-wind update schedule must close on an integer number of updates')
    expected_updates = int(round(expected_updates_float))
    base = list(FULL_23WIND_BASE_ORDER)
    expected_order: list[int] = []
    for round_index in range(rounds):
        offset = (round_index * 7) % len(base)
        expected_order.extend(base[offset:] + base[:offset])
    schedule_document = {
        'protocol_version': manifest.get('protocol_version'),
        'execution_mode': manifest.get('execution_mode'),
        'base_condition_order': base,
        'rotation_stride': 7,
        'total_condition_rounds': rounds,
        'train_sim_s': duration_s,
        'expanded_condition_order': expected_order,
    }
    schedule_sha256 = document_sha256(schedule_document)
    manifest_order = [slot.condition_id for slot in slots]
    if (manifest.get('protocol_version') != 'bladed_udp_19_si_my_queue_v2' or
            manifest.get('execution_mode') != 'independent_case_reload' or
            manifest.get('total_condition_rounds') != rounds or
            manifest.get('base_condition_order') != base or manifest.get('rotation_stride') != 7 or
            manifest.get('manual_batch_profile') != 'full' or
            manifest.get('manual_batch_rounds') != rounds or
            manifest.get('expanded_slot_count') != len(slots) or
            len(slots) != len(conditions) * rounds or manifest_order != expected_order or
            any(slot.train_sim_s != duration_s for slot in slots)):
        raise ValueError('Unified 23-wind manifest does not match its configured rounds/order/duration')
    if (queue.get('condition_ids') != conditions or float(queue.get('train_sim_s', -1.0)) != duration_s or
            contract.get('condition_ids') != conditions or contract.get('rounds') != rounds or
            contract.get('train_sim_s') != duration_s or
            contract.get('expected_scheduled_updates') != expected_updates or
            contract.get('profile') != FULL_23WIND_PROFILE or
            contract.get('resolved_control_contract_fingerprint') != config.get('control_contract_fingerprint')):
        raise ValueError('Unified 23-wind effective config/schedule contract does not close')
    if (config.get('release_id') != OPERATOR_23WIND_RELEASE_ID or
            config.get('physical_shadow_run_role') != OPERATOR_23WIND_ROLE or
            config.get('policy_acceptance', False) is not False or
            queue.get('case_identity_validation', {}).get('wind_mismatch_policy') != 'warn_development_only'):
        raise ValueError('Unified 23-wind route must remain development-only with explicit operator wind attestation')
    launch_gate = config.get('physical_shadow_launch', {})
    if launch_gate.get('route') != OPERATOR_23WIND_ROUTE or launch_gate.get('authorized') not in (False, True):
        raise ValueError('Unified 23-wind profile must retain its explicit route and boolean launch status')
    if (config.get('train', {}).get('online_rl_enabled') is not True or
            config.get('train', {}).get('online_eval_enabled') is not False or
            queue.get('collect_exploration_noise') != 0.0 or
            queue.get('critic_exploration_noise') != 0.0 or
            queue.get('actor_exploration_noise') != 0.0):
        raise ValueError('Unified 23-wind route requires the frozen training/noise contract')
    required_slot_record_capacity = 2 * (steps_per_slot // stride + 1)
    if (max_shadow_records < expected_shadow_records or
            max_shadow_record_bytes < PHYSICAL_SHADOW_RECORD_MAX_BYTES or
            max_shadow_bytes > PHYSICAL_SHADOW_MAX_TOTAL_BYTES or
            max_shadow_bytes < max_shadow_records * max_shadow_record_bytes):
        raise ValueError('Unified 23-wind physical shadow record/byte budgets do not close')
    if (int(predictor.get('condition_record_capacity', 0)) < required_slot_record_capacity or
            int(predictor.get('worker_queue_capacity', 0)) < 256):
        raise ValueError('Unified 23-wind physical shadow per-slot/worker queue capacity is below its preflight floor')
    if (logging.get('mode') != 'compact_streaming' or logging.get('debug_long_form_enabled') is not False or
            event_capacity < 1 or event_record_bytes < 1 or event_record_bytes > 512 * 1024 or
            event_total_bytes < event_worst_case_bytes or
            summary_record_bytes < 1 or summary_record_bytes > 512 * 1024 or
            summary_total_bytes < len(slots) * summary_record_bytes or
            int(logging.get('queue_condition_ring_capacity', 0)) < len(slots)):
        raise ValueError('Unified 23-wind compact logger budget is below the configured slot/event bound')
    if config.get('load_predictor', {}).get('backend') != 'physical_fir_v2' or predictor.get('mode') != 'shadow' or predictor.get('nn_active') is not False:
        raise ValueError('Unified 23-wind physical route is read-only shadow; NN/soft/hard paths are forbidden')
    preflight_physical_config(config, project_root=project_root)
    return {
        'release_id': OPERATOR_23WIND_RELEASE_ID,
        'schema_version': 'physical_operator_23wind_round600_schedule_contract_v2',
        'condition_ids': conditions,
        'base_condition_order': base,
        'rotation_stride': 7,
        'rounds': rounds,
        'train_sim_s': duration_s,
        'slot_count': len(slots),
        'expected_expanded_condition_order': expected_order,
        'schedule_sha256': schedule_sha256,
        'expected_scheduled_updates': expected_updates,
        'manifest_sha256': file_sha256(manifest_path),
        'profile_build_report_path': OPERATOR_23WIND_PROFILE_BUILD_REPORT,
        'profile_build_report_sha256': file_sha256(profile_report_path),
        'source_profile_path': OPERATOR_23WIND_SOURCE_PROFILE,
        'source_profile_sha256': file_sha256(source_profile_path),
        'expected_windows_release_dll_path': OPERATOR_23WIND_DLL_IDENTITY_PATH,
        'expected_windows_release_dll_sha256': expected_dll_sha,
        'windows_queue_source_sha256': file_sha256(project_root / 'bladed_env/udptest/udptest/controller.cpp'),
        'shadow_expected_record_upper_bound': expected_shadow_records,
        'shadow_max_records': max_shadow_records,
        'shadow_max_record_bytes': max_shadow_record_bytes,
        'shadow_max_total_bytes': max_shadow_bytes,
        'event_window_capacity_per_slot': event_capacity,
        'event_window_max_record_bytes': event_record_bytes,
        'event_window_worst_case_bytes': event_worst_case_bytes,
        'event_windows_max_total_bytes': event_total_bytes,
        'policy_acceptance': False,
        'independent_audit': False,
        'soft_enabled': False,
        'hard_enabled': False,
    }


def validate_operator_23wind_scope(config: dict[str, Any], project_root: Path) -> dict[str, Any]:
    """Validate the one 23-wind × 600-second operator template for configured rounds."""
    import copy
    from RL.utils.control_contract import resolve_configured_control_contract

    template = strict_json(project_root / OPERATOR_23WIND_TEMPLATE)
    effective = copy.deepcopy(config)
    identity_policy = effective['queue']['case_identity_validation'].pop('wind_mismatch_policy', 'abort')
    if identity_policy != 'warn_development_only':
        raise ValueError('Unified 23-wind operator scope requires explicit development-only wind mismatch warnings')
    for document in (effective, template):
        for key in ('physical_shadow_launch', 'output_dir', 'experiment_name', 'append_timestamp',
                    'resolved_experiment_name', 'device'):
            document.pop(key, None)
    effective.pop('policy_acceptance', None)
    template.pop('policy_acceptance', None)
    if artifact_path(effective['data']['root_dir'], project_root).resolve() != artifact_path(
            template['data']['root_dir'], project_root).resolve():
        raise ValueError('Unified 23-wind operator development data source mismatch')
    effective['data']['root_dir'] = template['data']['root_dir']
    for section, field in (('queue', 'manifest_path'), ('load_predictor', 'bundle_path')):
        actual_path = artifact_path(effective[section][field], project_root)
        template_path = artifact_path(template[section][field], project_root)
        if section == 'load_predictor' and file_sha256(actual_path) != file_sha256(template_path):
            raise ValueError('Unified 23-wind operator development manifest/bundle mismatch')
        effective[section][field] = template[section][field]
    # Schedule-derived values are allowed to vary only because the strict
    # validator below recomputes slots, order, update count, and budgets.
    for section, field in (('queue', 'total_condition_rounds'),
                           ('state20_queue_contract', 'rounds'),
                           ('state20_queue_contract', 'expected_scheduled_updates')):
        effective[section][field] = template[section][field]
    for field in ('max_records', 'max_record_bytes', 'max_total_bytes'):
        effective['load_predictor'][field] = template['load_predictor'][field]
    if effective != template:
        raise ValueError('Unified 23-wind config differs from its profile/report template')
    route_contract = _operator_23wind_schedule_contract(config, project_root)
    if resolve_configured_control_contract(config, project_root=project_root) is None:
        raise ValueError('Unified 23-wind operator development requires the resolved state20 control contract')
    return route_contract


def _operator_23wind_confirmation(
    route_contract: dict[str, Any], identity: dict[str, Any]
) -> dict[str, Any]:
    return {
        **OPERATOR_ATTESTATION,
        'confirmation_source': 'user_confirmation_in_current_session',
        'manual_batch_profile_confirmed': 'full',
        'manual_batch_rounds_confirmed': route_contract['rounds'],
        'manual_batch_train_sim_s_confirmed': route_contract['train_sim_s'],
        'manual_batch_slot_count_confirmed': route_contract['slot_count'],
        'manual_batch_schedule_sha256_confirmed': route_contract['schedule_sha256'],
        'protocol_version_confirmed': 'bladed_udp_19_si_my_queue_v2',
        'controller_source_sha256_confirmed': route_contract['windows_queue_source_sha256'],
        'repository_release_dll_sha256': route_contract['expected_windows_release_dll_sha256'],
    }


def validate_operator_23wind_development(
    config: dict[str, Any], receipt: dict[str, Any], identity: dict[str, Any], project_root: Path,
    *, checkpoint_path: Path | None, baseline_only: bool,
) -> dict[str, Any]:
    if checkpoint_path is not None or baseline_only:
        raise ValueError('Unified 23-wind operator receipt is training-only, not inference/baseline')
    if any(os.environ.get(name) for name in
           ('RL_EXPERIMENT_SUFFIX', 'RL_DATA_ROOT', 'RL_DYNAMICS_CHECKPOINT')):
        raise ValueError('Unified 23-wind operator development forbids identity-changing environment overrides')
    route_contract = validate_operator_23wind_scope(config, project_root)
    run_id = config.get('experiment_name')
    if not isinstance(run_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', run_id):
        raise ValueError('Unified 23-wind operator development requires a valid RUN_ID')
    config_dir = (project_root / 'RL/configs' / run_id).resolve()
    experiment_prefix = (project_root / 'RL/experiments' / f'{run_id}_').resolve()
    output_dir = artifact_path(config['output_dir'], project_root).resolve()
    if (output_dir != (project_root / 'RL/experiments').resolve() or
            config_dir.parent != (project_root / 'RL/configs').resolve() or
            experiment_prefix.parent != output_dir or config.get('append_timestamp') is not True):
        raise ValueError('Unified operator development requires canonical configs/experiment layout')
    launch = config.get('physical_shadow_launch', {})
    if (launch.get('authorized') is not True or launch.get('route') != OPERATOR_23WIND_ROUTE or
            launch.get('decision') != 'OPERATOR_CONFIRMED_23WIND_DEVELOPMENT_ONLY_NOT_SIX_GATE_VERIFIED'):
        raise ValueError('Unified operator launch gate is not explicitly finalized')
    receipt_path = artifact_path(config['physical_shadow_launch'].get('evidence_path', ''), project_root).resolve()
    if receipt_path != config_dir / OPERATOR_23WIND_RECEIPT_NAME:
        raise ValueError('Unified operator receipt must be colocated with its RUN_ID config')
    expected_attestation = _operator_23wind_confirmation(route_contract, identity)
    if (receipt.get('schema_version') != OPERATOR_23WIND_SCHEMA or
            receipt.get('route') != OPERATOR_23WIND_ROUTE or
            receipt.get('release_id') != OPERATOR_23WIND_RELEASE_ID or
            receipt.get('rounds') != route_contract['rounds'] or
            receipt.get('slots') != route_contract['slot_count'] or
            receipt.get('runtime_identity') != identity or
            receipt.get('operator_attestation') != expected_attestation or
            receipt.get('profile_build_report_sha256') != route_contract['profile_build_report_sha256'] or
            receipt.get('source_profile_path') != route_contract['source_profile_path'] or
            receipt.get('source_profile_sha256') != route_contract['source_profile_sha256'] or
            receipt.get('operator_route_contract') != route_contract or
            receipt.get('independently_verified') is not False or
            receipt.get('policy_acceptance') is not False or
            receipt.get('online_latency_status') != 'unknown_measure_during_experiment' or
            receipt.get('template_sha256') != file_sha256(project_root / OPERATOR_23WIND_TEMPLATE) or
            receipt.get('resource_inventory') != operator_resource_inventory(config, project_root) or
            receipt.get('run_id') != run_id or receipt.get('config_dir') != str(config_dir) or
            receipt.get('run_config') != str(config_dir / 'train.json') or
            receipt.get('experiment_dir_prefix') != str(experiment_prefix) or
            not receipt.get('created_at_utc') or
            receipt.get('optimizer_updates') is not True or receipt.get('actor_frozen') is not False or
            receipt.get('independent_audit') is not False or
            receipt.get('counterfactual_validation') is not False or
            receipt.get('soft_authorized') is not False or receipt.get('hard_authorized') is not False or
            receipt.get('authorization') != {
                'mode': 'shadow_development_only',
                'policy_acceptance': False,
                'independent_audit': False,
                'counterfactual_validation': False,
                'soft_authorized': False,
                'hard_authorized': False,
                'rl_margin_coverage_authorized': False,
            }):
        raise ValueError('Unified 23-wind operator receipt scope/identity/attestation mismatch')
    from RL.prepare_physical_shadow_development import check_resources
    check_resources(config, Path(f'{experiment_prefix}<timestamp>'))
    return {**identity, 'route': OPERATOR_23WIND_ROUTE,
            'operator_development_authorized': True, 'operator_route_contract': route_contract}


def operator_resource_inventory(config: dict[str, Any], project_root: Path) -> dict[str, str]:
    """Hash actual BC data and control dependencies, not a claimed resource list."""
    from RL.utils.control_contract import resolve_configured_control_contract
    contract, _ = resolve_configured_control_contract(config, project_root=project_root)
    paths = [artifact_path(value, project_root) for value in contract['artifact_paths'].values()]
    paths += [artifact_path(config[key], project_root) for key in
              ('control_contract_static_path', 'resolved_control_contract_path')]
    data = artifact_path(config['data']['root_dir'], project_root)
    if file_sha256(data / 'manifest.json') != config['b4_residual_contract']['data_manifest_sha256']:
        raise ValueError('Operator development BC data manifest mismatch')
    paths.append(data / 'manifest.json')
    for split in ('train', 'val'):
        files = sorted((data / split).glob('*.csv'))
        if not files:
            raise ValueError(f'Operator development missing BC data: {split}')
        paths.extend(files)
    # The default loader can use mmap caches instead of CSV bytes.
    cache = data / '.rl_cache' / 'offline_bc'
    if cache.is_dir():
        paths.extend(sorted(path for path in cache.iterdir() if path.suffix in ('.npy', '.json')))
    # Keep historical profiles' declared source bound; the current R2 profile
    # omits this retired reward dependency entirely.
    rotor_speed_reference_path = config['env'].get('reward', {}).get('rotor_speed_reference_schedule_path')
    if rotor_speed_reference_path is not None:
        paths.append(artifact_path(rotor_speed_reference_path, project_root))
    return {str(path.resolve()): file_sha256(path) for path in paths}


def validate_operator_development(config: dict[str, Any], receipt: dict[str, Any],
        identity: dict[str, Any], project_root: Path, *, checkpoint_path: Path | None,
        baseline_only: bool) -> dict[str, Any]:
    if checkpoint_path is not None or baseline_only:
        raise ValueError('Operator development receipt is training-only, not inference/baseline')
    if any(os.environ.get(name) for name in
           ('RL_EXPERIMENT_SUFFIX', 'RL_DATA_ROOT', 'RL_DYNAMICS_CHECKPOINT')):
        raise ValueError('Operator development forbids identity-changing environment overrides')
    validate_operator_scope(config, project_root)
    run_id = config.get('experiment_name')
    if not isinstance(run_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', run_id):
        raise ValueError('Operator development requires a valid RUN_ID')
    config_dir = (project_root / 'RL/configs' / run_id).resolve()
    experiment_dir = (project_root / 'RL/experiments' / run_id).resolve()
    output_dir = artifact_path(config['output_dir'], project_root).resolve()
    if (output_dir != (project_root / 'RL/experiments').resolve() or
            config_dir.parent != (project_root / 'RL/configs').resolve() or
            experiment_dir.parent != output_dir):
        raise ValueError('Operator development requires canonical configs/experiment layout')
    if (receipt.get('schema_version') != OPERATOR_SCHEMA or receipt.get('route') != OPERATOR_ROUTE or
            receipt.get('runtime_identity') != identity or
            receipt.get('operator_attestation') != OPERATOR_ATTESTATION or
            receipt.get('independently_verified') is not False or
            receipt.get('policy_acceptance') is not False or
            receipt.get('online_latency_status') != 'unknown_measure_during_experiment' or
            receipt.get('template_sha256') != file_sha256(project_root / OPERATOR_TEMPLATE) or
            receipt.get('resource_inventory') != operator_resource_inventory(config, project_root) or
            receipt.get('run_id') != run_id or receipt.get('config_dir') != str(config_dir) or
            receipt.get('run_config') != str(config_dir / 'train.json') or
            receipt.get('experiment_dir') != str(experiment_dir) or 'run_root' in receipt or
            config.get('append_timestamp') is not False or
            not receipt.get('created_at_utc')):
        raise ValueError('Operator development receipt scope/identity/attestation mismatch')
    evidence = artifact_path(config['physical_shadow_launch']['evidence_path'], project_root).resolve()
    if evidence != config_dir / 'operator_development_receipt.json':
        raise ValueError('Operator development receipt must belong to its RL/configs batch')
    for section, field, filename in (
            ('queue', 'manifest_path', 'queue_manifest.json'),
            ('load_predictor', 'bundle_path', 'physical_load_shadow_bundle.json')):
        if artifact_path(config[section][field], project_root).resolve() != config_dir / filename:
            raise ValueError('Operator development bundle/queue must belong to its RL/configs batch')
    from RL.prepare_physical_shadow_development import check_resources
    check_resources(config, experiment_dir)
    return identity