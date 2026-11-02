"""Prepare a narrow frozen-round-4 physical-shadow inference config and receipt.

This command writes config/receipt files only. It never binds UDP, starts Bladed,
loads training data, changes an action, or enables an optimizer.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from RL.utils.physical_load_shadow_artifact import artifact_path, file_sha256, strict_json
from RL.utils.physical_load_shadow_launch import (
    FROZEN_R4_ROUTE,
    FROZEN_R4_SCHEMA,
    FROZEN_R4_TEMPLATE,
    OPERATOR_ATTESTATION,
    OPERATOR_ROUTE,
    OPERATOR_SCHEMA,
    _same_state,
    physical_runtime_identity,
)
from RL.utils.queue_manifest import expand_queue_manifest

DEFAULT_SOURCE_RUN_ID = 'physical_4x4x600_20261013T100824Z_20940_retry_075710_2626'
EVENT_WINDOW_VALIDATION = {
    'full_existing_state_enabled': True,
    'pre_frames': 50,
    'post_frames': 25,
    'max_record_bytes': 512 * 1024,
    'max_total_bytes': 512 * 1024 * 1024,
}


def _sha(path: Path) -> str:
    return file_sha256(path)


def _validate_source(source_run_id: str) -> tuple[Path, Path, dict[str, Any], dict[str, Any], Path]:
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', source_run_id):
        raise ValueError('Source RUN_ID must contain 1-80 ASCII letters, numbers, underscores, or hyphens')
    config_path = (PROJECT_ROOT / 'RL/configs' / source_run_id / 'train.json').resolve()
    receipt_path = (PROJECT_ROOT / 'RL/configs' / source_run_id / 'operator_development_receipt.json').resolve()
    experiment_dir = (PROJECT_ROOT / 'RL/experiments' / source_run_id).resolve()
    training_summary_path = experiment_dir / 'queue_training_summary.json'
    round_summary_path = experiment_dir / 'queue_round_4_summary.json'
    checkpoint_path = experiment_dir / 'checkpoints' / 'queue_round_4.pt'
    for path in (config_path, receipt_path, training_summary_path, round_summary_path, checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(f'Frozen round-4 source evidence is missing: {path}')
    config = strict_json(config_path)
    receipt = strict_json(receipt_path)
    if (receipt.get('schema_version') != OPERATOR_SCHEMA or receipt.get('route') != OPERATOR_ROUTE or
            receipt.get('run_id') != source_run_id or receipt.get('run_config') != str(config_path) or
            receipt.get('experiment_dir') != str(experiment_dir) or
            receipt.get('operator_attestation') != OPERATOR_ATTESTATION or
            receipt.get('independently_verified') is not False or receipt.get('policy_acceptance') is not False):
        raise ValueError('Frozen round-4 source does not have the expected completed operator-development receipt')
    runtime_identity = receipt.get('runtime_identity', {})
    effective_config = {key: value for key, value in config.items()
                        if key not in ('physical_shadow_launch', 'output_dir', 'experiment_name',
                                       'append_timestamp', 'resolved_experiment_name')}
    queue_path = Path(config['queue']['manifest_path'])
    if not queue_path.is_absolute():
        queue_path = PROJECT_ROOT / queue_path
    bundle_path = Path(config['load_predictor']['bundle_path'])
    if not bundle_path.is_absolute():
        bundle_path = PROJECT_ROOT / bundle_path
    if (runtime_identity.get('effective_runtime_config_sha256') != json_sha256(effective_config) or
            runtime_identity.get('queue_manifest_sha256') != _sha(queue_path) or
            runtime_identity.get('bundle_sha256') != _sha(bundle_path) or
            receipt.get('config_dir') != str(config_path.parent) or
            receipt.get('run_id') != source_run_id):
        raise ValueError('Frozen round-4 source config/receipt/queue/bundle identity mismatch')
    for name, path in (('queue_training_summary', training_summary_path),
                       ('queue_round_4_summary', round_summary_path)):
        summary = strict_json(path)
        if (summary.get('queue_training_valid') != 1.0 or
                summary.get('queue_completed_rounds') != 4.0 or
                summary.get('queue_total_rounds') != 4.0 or
                summary.get('queue_skipped_condition_count') != 0.0 or
                summary.get('queue_retried_condition_attempt_count') != 0.0 or
                summary.get('queue_critic_q_health_passed') != 1.0 or
                summary.get('queue_critic_q_health_degraded') != 0.0):
            raise ValueError(f'{name} does not attest a completed, valid 4-round source training run')
    if receipt.get('runtime_identity', {}).get('actor_checkpoint_sha256') != config.get(
            'state20_queue_contract', {}).get('bc_warmstart_checkpoint_sha256'):
        raise ValueError('Source operator receipt must bind the BC warm-start checkpoint as its training entry')
    return config_path, receipt_path, config, receipt, checkpoint_path


def json_sha256(value: dict[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    import hashlib
    return hashlib.sha256(encoded).hexdigest()


def _validate_round4_checkpoint(checkpoint_path: Path, source_config: dict[str, Any], template: dict[str, Any]) -> str:
    digest = _sha(checkpoint_path)
    expected = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if not isinstance(expected, dict):
        raise ValueError('Round-4 checkpoint must contain a mapping')
    queue_state = expected.get('queue_state', {})
    agent_state = expected.get('agent', {})
    if (expected.get('total_steps') != 480_016 or
            expected.get('data_schema_version') != template.get('data_schema_version') or
            expected.get('control_contract_fingerprint') != template.get('control_contract_fingerprint') or
            expected.get('actor_control_fingerprint') != template.get('control', {}).get(
                'actor_control_fingerprint', expected.get('actor_control_fingerprint')) or
            expected.get('config', {}).get('control_contract_fingerprint') !=
            source_config.get('control_contract_fingerprint') or
            queue_state.get('completed_rounds') != 4 or queue_state.get('total_rounds') != 4 or
            queue_state.get('training_valid') is not True or
            queue_state.get('skipped_condition_count', 0) != 0 or
            queue_state.get('retried_condition_attempt_count', 0) != 0 or
            not isinstance(agent_state, dict) or agent_state.get('update_step') != 12_000 or
            not isinstance(agent_state.get('actor'), dict) or
            not isinstance(agent_state.get('actor_execution'), dict) or
            not _same_state(agent_state['actor'], agent_state['actor_execution'])):
        raise ValueError('Round-4 checkpoint identity/completion or frozen actor_execution binding mismatch')
    return digest


def _write_new(path: Path, document: dict[str, Any]) -> None:
    with path.open('x', encoding='utf-8') as handle:
        json.dump(document, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def prepare_round4_frozen_inference(
    run_id: str,
    *,
    source_run_id: str = DEFAULT_SOURCE_RUN_ID,
    device: str = 'cuda:0',
) -> dict[str, Any]:
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', run_id):
        raise ValueError('RUN_ID must contain 1-80 ASCII letters, numbers, underscores, or hyphens')
    config_dir = (PROJECT_ROOT / 'RL/configs' / run_id).resolve()
    experiment_dir = (PROJECT_ROOT / 'RL/experiments' / run_id).resolve()
    if config_dir.exists() or experiment_dir.exists():
        raise FileExistsError('RUN_ID config or experiment directory already exists; choose a new RUN_ID')
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise ValueError('Requested CUDA device is unavailable; prepare with --device cpu on this host')
    source_config_path, source_receipt_path, source_config, source_receipt, checkpoint_path = _validate_source(
        source_run_id
    )
    template = strict_json(PROJECT_ROOT / FROZEN_R4_TEMPLATE)
    checkpoint_sha256 = _validate_round4_checkpoint(checkpoint_path, source_config, template)
    manifest_path = Path(template['queue']['manifest_path'])
    if not manifest_path.is_absolute():
        manifest_path = PROJECT_ROOT / manifest_path
    slots = expand_queue_manifest(strict_json(manifest_path))
    if [slot.condition_id for slot in slots] != [8, 11, 12, 16] or len(slots) != 4 or any(
            slot.train_sim_s != 600.0 for slot in slots):
        raise ValueError('Frozen R4 inference profile must contain exactly one 600s round over winds 8/11/12/16')

    config = deepcopy(template)
    config.update({
        'experiment_name': run_id,
        'output_dir': 'RL/experiments',
        'append_timestamp': False,
        'device': device,
        'policy_acceptance': False,
    })
    config['queue']['case_identity_validation']['wind_mismatch_policy'] = 'warn_development_only'
    config['train']['online_rl_enabled'] = False
    config['logging'].update(
        queue_event_full_state_enabled=True,
        queue_event_pre_frames=EVENT_WINDOW_VALIDATION['pre_frames'],
        queue_event_post_frames=EVENT_WINDOW_VALIDATION['post_frames'],
        queue_event_window_capacity=16,
        queue_event_window_max_record_bytes=EVENT_WINDOW_VALIDATION['max_record_bytes'],
        queue_event_windows_max_total_bytes=EVENT_WINDOW_VALIDATION['max_total_bytes'],
        debug_long_form_enabled=False,
        debug_max_rows=0,
        debug_max_bytes=0,
    )
    config['physical_shadow_run_role'] = {
        'stage': 'round4_frozen_actor_event_window_validation',
        'source_run_id': source_run_id,
        'source_round': 4,
        'actor_frozen': True,
        'optimizer_updates': False,
        'independent_audit': False,
        'policy_acceptance': False,
        'soft_enabled': False,
        'hard_enabled': False,
    }
    config_dir.mkdir(parents=True, exist_ok=False)
    inference_config_path = config_dir / 'inference.json'
    receipt_path = config_dir / 'round4_inference_receipt.json'
    config['physical_shadow_launch'] = {
        'authorized': True,
        'route': FROZEN_R4_ROUTE,
        'evidence_path': str(receipt_path),
        'evidence_sha256': '',
        'decision': 'OPERATOR_CONFIRMED_FROZEN_ACTOR_SHADOW_DEVELOPMENT_ONLY',
    }
    scope = {
        'entry': 'inference', 'episodes': 4, 'max_steps': None,
        'baseline_only': False, 'output_dir': str(experiment_dir),
    }
    runtime_config = deepcopy(config)
    runtime_config['physical_shadow_execution_scope'] = scope
    identity = physical_runtime_identity(runtime_config, PROJECT_ROOT,
                                         checkpoint_path=checkpoint_path, baseline_only=False)
    receipt = {
        'schema_version': FROZEN_R4_SCHEMA,
        'route': FROZEN_R4_ROUTE,
        'created_at_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'run_id': run_id,
        'config_path': str(inference_config_path),
        'experiment_dir': str(experiment_dir),
        'runtime_identity': identity,
        'source_run': {
            'run_id': source_run_id,
            'config_path': str(source_config_path),
            'operator_receipt_path': str(source_receipt_path),
            'experiment_dir': str((PROJECT_ROOT / 'RL/experiments' / source_run_id).resolve()),
            'config_sha256': _sha(source_config_path),
            'operator_receipt_sha256': _sha(source_receipt_path),
            'training_summary_sha256': _sha(
                PROJECT_ROOT / 'RL/experiments' / source_run_id / 'queue_training_summary.json'),
            'round_4_summary_sha256': _sha(
                PROJECT_ROOT / 'RL/experiments' / source_run_id / 'queue_round_4_summary.json'),
            'round_4_checkpoint_path': str(checkpoint_path.resolve()),
            'round_4_checkpoint_sha256': checkpoint_sha256,
        },
        'operator_attestation': dict(source_receipt['operator_attestation']),
        'independently_verified': False,
        'policy_acceptance': False,
        'optimizer_updates': False,
        'actor_frozen': True,
        'event_window_validation': dict(EVENT_WINDOW_VALIDATION),
        'authorization': {
            'physical_shadow': dict(config['load_predictor']['authorization']),
            'policy_acceptance': False,
            'counterfactual_validation': False,
        },
    }
    try:
        _write_new(receipt_path, receipt)
        config['physical_shadow_launch']['evidence_sha256'] = _sha(receipt_path)
        _write_new(inference_config_path, config)
    except BaseException:
        import shutil
        shutil.rmtree(config_dir, ignore_errors=True)
        raise
    return {
        'run_id': run_id,
        'source_run_id': source_run_id,
        'config_path': str(inference_config_path),
        'receipt_path': str(receipt_path),
        'round_4_checkpoint_path': str(checkpoint_path),
        'round_4_checkpoint_sha256': checkpoint_sha256,
        'experiment_dir': str(experiment_dir),
        'bladed_started': False,
        'policy_acceptance': False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True, help='New RUN_ID; both configs and experiment paths must be unused.')
    parser.add_argument('--source-run-id', default=DEFAULT_SOURCE_RUN_ID,
                        help='Completed 4x4x600 source run containing checkpoints/queue_round_4.pt.')
    parser.add_argument('--device', default='cuda:0', help='Exact inference device to bind in the receipt.')
    args = parser.parse_args()
    result = prepare_round4_frozen_inference(args.run_id, source_run_id=args.source_run_id, device=args.device)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
