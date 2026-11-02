from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import sys
import zlib
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from RL.utils.control_contract import ENGINEERING_DEMAND_HIGH_NM
from RL.utils.load_predictor_artifact import load_candidate_conditioned_predictor
from RL.utils.load_predictor_contract import (
    LoadPredictorInterface,
    PLAN_KIND_ZOH_CANDIDATE,
    make_zoh_candidate_plan,
    normalized_action_plan_mismatch,
)


SHADOW_DATASET_SCHEMA_VERSION = 'load_predictor_dataset_v2'


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def build_shadow_dataset(
    *,
    shadow_jsonl: Path,
    interface_path: Path,
    predictor_artifact_path: Path,
    output_root: Path,
    max_rows_per_shard: int = 4096,
    require_strict_identity: bool = True,
    require_actor_execution: bool = True,
) -> dict[str, Any]:
    if max_rows_per_shard <= 0:
        raise ValueError('max_rows_per_shard must be positive')
    shadow_jsonl = shadow_jsonl.expanduser().resolve()
    interface_path = interface_path.expanduser().resolve()
    interface = LoadPredictorInterface.from_path(interface_path)
    interface_sha256 = _sha256(interface_path)
    predictor_artifact_path = predictor_artifact_path.expanduser().resolve()
    predictor_sha256 = _sha256(predictor_artifact_path)
    expected_predictor_summary_path = predictor_artifact_path.with_name(
        f'{predictor_artifact_path.stem}.summary.json'
    )
    if not expected_predictor_summary_path.is_file():
        raise FileNotFoundError(f'Predictor artifact summary does not exist: {expected_predictor_summary_path}')
    predictor_summary = json.loads(expected_predictor_summary_path.read_text(encoding='utf-8'))
    if (
        predictor_summary.get('checkpoint_sha256') != predictor_sha256
        or predictor_summary.get('schema_version') != 'load_predictor_checkpoint_v4_candidate_conditioned'
    ):
        raise ValueError('Predictor artifact summary SHA/schema does not match the frozen v4 artifact')
    import torch
    predictor_checkpoint = torch.load(predictor_artifact_path, map_location='cpu', weights_only=False)
    if not isinstance(predictor_checkpoint, dict):
        raise ValueError('Predictor checkpoint does not contain a metadata mapping')
    paired_identity_by_condition = predictor_checkpoint.get('paired_identity_by_condition')
    if not isinstance(paired_identity_by_condition, dict):
        raise ValueError('Predictor checkpoint is missing its frozen paired PID identity table')
    load_candidate_conditioned_predictor(
        predictor_artifact_path,
        interface_path=interface_path,
        expected_checkpoint_sha256=predictor_sha256,
        action_low=[0.0, 0.0],
        action_high=[ENGINEERING_DEMAND_HIGH_NM, 0.43],
        device=torch.device('cpu'),
        expected_dataset_manifest_sha256=str(predictor_checkpoint.get('dataset_manifest_sha256', '')),
        expected_time_partitions_s=predictor_summary.get('time_partitions_s'),
    )
    source_dataset_root = Path(str(predictor_checkpoint.get('dataset_root', ''))).expanduser()
    if not source_dataset_root.is_absolute():
        source_dataset_root = PROJECT_ROOT / source_dataset_root
    source_dataset_manifest_path = source_dataset_root / 'manifest.json'
    if not source_dataset_manifest_path.is_file():
        raise FileNotFoundError(f'Predictor source dataset manifest does not exist: {source_dataset_manifest_path}')
    source_dataset_manifest = json.loads(source_dataset_manifest_path.read_text(encoding='utf-8'))
    if _sha256(source_dataset_manifest_path) != predictor_checkpoint.get('dataset_manifest_sha256'):
        raise ValueError('Predictor source dataset manifest SHA does not match the frozen checkpoint')
    output_root.mkdir(parents=True, exist_ok=True)
    if (output_root / 'manifest.json').exists() or any(output_root.glob('shard_*.npz')):
        raise FileExistsError(f'Refusing to overwrite existing RL-shadow predictor dataset: {output_root}')
    max_frames = max(interface.horizon_frames)
    shard_records: list[dict[str, Any]] = []
    shards: list[dict[str, Any]] = []
    sample_count_by_plan = {PLAN_KIND_ZOH_CANDIDATE: 0, 'realized_plan': 0}
    rejection_counts: dict[str, int] = {}
    source_conditions: set[int] = set()
    source_instances: set[tuple[int, int]] = set()
    source_pid_sha_by_condition: dict[str, str] = {}
    actor_policy_versions: set[int] = set()
    actor_policy_sha256s: set[str] = set()
    # Stream JSONL line by line; never load an entire run or raw table.
    def flush_shard() -> None:
        if not shard_records:
            return
        shard_path = output_root / f'shard_{len(shards):05d}.npz'
        np.savez_compressed(
            shard_path,
            history=np.stack([record['history'] for record in shard_records]).astype(np.float32),
            model_input_plan=np.stack([record['model_input_plan'] for record in shard_records]).astype(np.float32),
            actual_future_action_plan=np.stack([
                record['actual_future_action_plan'] for record in shard_records
            ]).astype(np.float32),
            future_actual_torque_pitch=np.asarray([
                record['future_actual_torque_pitch'] for record in shard_records
            ], dtype=np.float32),
            future_peak_ratio=np.asarray([
                record['future_peak_ratio'] for record in shard_records
            ], dtype=np.float32),
            timestamp_s=np.asarray([record['timestamp_s'] for record in shard_records], dtype=np.float64),
            condition_id=np.asarray([record['condition_id'] for record in shard_records], dtype=np.int32),
            condition_instance_id=np.asarray([
                record['condition_instance_id'] for record in shard_records
            ], dtype=np.int64),
            horizon_s=np.asarray([record['horizon_s'] for record in shard_records], dtype=np.float32),
            horizon_frames=np.asarray([record['horizon_frames'] for record in shard_records], dtype=np.int32),
            plan_kind=np.asarray([record['plan_kind'] for record in shard_records], dtype='U32'),
            optimization_state=np.asarray([
                record['optimization_state'] for record in shard_records
            ], dtype='U64'),
            plan_mismatch=np.asarray([record['plan_mismatch'] for record in shard_records], dtype=np.float32),
            zoh_calibration_eligible=np.asarray([
                record['zoh_calibration_eligible'] for record in shard_records
            ], dtype=np.bool_),
            strict_case_matched=np.asarray([
                record['strict_case_matched'] for record in shard_records
            ], dtype=np.bool_),
            pid_raw_sha256=np.asarray([record['pid_raw_sha256'] for record in shard_records], dtype='U64'),
            pid_raw_csv_path=np.asarray([record['pid_raw_csv_path'] for record in shard_records], dtype='U512'),
            raw_source_sha256=np.asarray([record['raw_source_sha256'] for record in shard_records], dtype='U64'),
            predictor_artifact_sha256=np.asarray([
                record['predictor_artifact_sha256'] for record in shard_records
            ], dtype='U64'),
            actor_policy_version=np.asarray([
                record['actor_policy_version'] for record in shard_records
            ], dtype=np.int64),
            actor_policy_sha256=np.asarray([
                record['actor_policy_sha256'] for record in shard_records
            ], dtype='U64'),
            actor_policy_window_frozen=np.asarray([
                record['actor_policy_window_frozen'] for record in shard_records
            ], dtype=np.bool_),
        )
        shards.append({
            'path': shard_path.name,
            'samples': len(shard_records),
            'sha256': _sha256(shard_path),
        })
        shard_records.clear()

    with shadow_jsonl.open('r', encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            if len(line.encode('utf-8')) > 64 * 1024:
                rejection_counts['record_too_large'] = rejection_counts.get('record_too_large', 0) + 1
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'Invalid shadow audit JSONL record at line {line_number}') from exc
            if not isinstance(row, dict) or row.get('schema_version') != 'load_predictor_shadow_audit_record_v1':
                rejection_counts['schema_mismatch'] = rejection_counts.get('schema_mismatch', 0) + 1
                continue
            if not bool(row.get('prediction_available', False)):
                rejection_counts['prediction_unavailable'] = rejection_counts.get('prediction_unavailable', 0) + 1
                continue
            if row.get('plan_kind') != PLAN_KIND_ZOH_CANDIDATE:
                rejection_counts['not_zoh_candidate'] = rejection_counts.get('not_zoh_candidate', 0) + 1
                continue
            if not bool(row.get('online_authorized', False)):
                rejection_counts['horizon_not_online_authorized'] = rejection_counts.get(
                    'horizon_not_online_authorized', 0
                ) + 1
                continue
            if require_strict_identity and (
                row.get('identity_status') != 'strict_case_matched'
                or not bool(row.get('strict_case_matched', False))
            ):
                rejection_counts['identity_not_strict'] = rejection_counts.get('identity_not_strict', 0) + 1
                continue
            if require_actor_execution and row.get('execution_mode') != 'actor':
                rejection_counts['execution_mode_not_actor'] = rejection_counts.get(
                    'execution_mode_not_actor',
                    0,
                ) + 1
                continue
            if row.get('outcome_tag') != 'complete':
                rejection_counts['outcome_not_complete'] = rejection_counts.get('outcome_not_complete', 0) + 1
                continue
            if not bool(row.get('actor_policy_window_frozen', False)):
                rejection_counts['actor_policy_changed_within_future_window'] = rejection_counts.get(
                    'actor_policy_changed_within_future_window',
                    0,
                ) + 1
                continue
            if not bool(row.get('actor_policy_window_frozen', False)):
                rejection_counts['actor_policy_changed_within_future_window'] = rejection_counts.get(
                    'actor_policy_changed_within_future_window',
                    0,
                ) + 1
                continue
            if row.get('plan_mismatch') is None or float(row['plan_mismatch']) > interface.max_plan_mismatch:
                rejection_counts['plan_mismatch'] = rejection_counts.get('plan_mismatch', 0) + 1
                continue
            if (
                float(row['plan_mismatch']) < 0.0
                or not np.isfinite(float(row['plan_mismatch']))
            ):
                rejection_counts['invalid_plan_mismatch'] = rejection_counts.get('invalid_plan_mismatch', 0) + 1
                continue
            if row.get('predictor_artifact_sha256') != predictor_sha256:
                rejection_counts['predictor_hash_mismatch'] = rejection_counts.get('predictor_hash_mismatch', 0) + 1
                continue
            if row.get('predictor_interface_sha256') != interface_sha256:
                rejection_counts['interface_hash_mismatch'] = rejection_counts.get('interface_hash_mismatch', 0) + 1
                continue
            row_pid_sha = str(row.get('pid_raw_sha256', '')).lower()
            if len(row_pid_sha) != 64 or any(character not in '0123456789abcdef' for character in row_pid_sha):
                rejection_counts['invalid_pid_raw_sha256'] = rejection_counts.get('invalid_pid_raw_sha256', 0) + 1
                continue
            if row.get('horizon_s') not in interface.soft_guard_horizons_s:
                rejection_counts['horizon_not_authorized'] = rejection_counts.get('horizon_not_authorized', 0) + 1
                continue
            history_shape = tuple(int(value) for value in row.get('history_shape', ()))
            if (
                int(row.get('feature_dim', -1)) != 31
                or row.get('history_schema_version') != 'load_predictor_shadow_feature_v1'
                or history_shape != (interface.history_frames, 31)
            ):
                rejection_counts['history_shape'] = rejection_counts.get('history_shape', 0) + 1
                continue
            try:
                if row.get('history_schema_version') != 'load_predictor_shadow_feature_v1':
                    raise ValueError('unsupported history schema')
                history_bytes = zlib.decompress(base64.b64decode(row['history_zlib_base64'], validate=True))
                history = np.frombuffer(history_bytes, dtype='<f4').reshape(history_shape).copy()
                model_plan = np.asarray(row['model_input_plan'], dtype=np.float32)
                actual_plan = np.asarray(row['actual_future_action_plan'], dtype=np.float32)
                candidate_action = np.asarray(row['candidate_action_real'], dtype=np.float64)
                response = np.asarray(row['actual_future_response'], dtype=np.float32)
                peaks = np.asarray(row['actual_future_peak_ratio_by_blade'], dtype=np.float32)
                horizon_s = float(row['horizon_s'])
                horizon_index = min(
                    range(len(interface.horizons_s)),
                    key=lambda index: abs(interface.horizons_s[index] - horizon_s),
                )
                expected_frames = interface.horizon_frames[horizon_index]
                condition_id = int(row['condition_id'])
                condition_instance_id = int(row['condition_instance_id'])
                timestamp_s = float(row['timestamp_s'])
                if (
                    model_plan.shape != (expected_frames, 2)
                    or actual_plan.shape != (expected_frames, 2)
                    or candidate_action.shape != (2,)
                    or response.shape != (2,)
                    or peaks.shape != (3,)
                    or not np.isfinite(history).all()
                    or not np.isfinite(model_plan).all()
                    or not np.isfinite(actual_plan).all()
                    or not np.isfinite(candidate_action).all()
                    or not np.isfinite(response).all()
                    or not np.isfinite(peaks).all()
                    or not np.isfinite(timestamp_s)
                ):
                    raise ValueError('invalid shadow audit array shape or value')
            except (KeyError, TypeError, ValueError, zlib.error) as exc:
                rejection_counts['invalid_record_payload'] = rejection_counts.get('invalid_record_payload', 0) + 1
                continue
            predictor_identity = paired_identity_by_condition.get(str(condition_id))
            observed_raw_path = Path(str(row.get('pid_raw_csv_path', ''))).expanduser()
            expected_raw_path = Path(
                str(predictor_identity.get('pid_raw_csv_path', ''))
                if isinstance(predictor_identity, dict)
                else ''
            ).expanduser()
            if not observed_raw_path.is_absolute():
                observed_raw_path = PROJECT_ROOT / observed_raw_path
            if not expected_raw_path.is_absolute():
                expected_raw_path = PROJECT_ROOT / expected_raw_path
            if (
                not isinstance(predictor_identity, dict)
                or str(row.get('pid_raw_sha256', '')).lower()
                != str(predictor_identity.get('pid_raw_sha256', '')).lower()
                or str(row.get('scenario_id', '')) != str(predictor_identity.get('scenario_id', ''))
                or str(row.get('expected_trace_signature', '')).lower()
                != str(predictor_identity.get('expected_trace_signature', '')).lower()
                or int(row.get('expected_trace_sample_count', -1))
                != int(predictor_identity.get('expected_trace_sample_count', -2))
                or observed_raw_path.resolve() != expected_raw_path.resolve()
            ):
                rejection_counts['predictor_reference_identity_mismatch'] = rejection_counts.get(
                    'predictor_reference_identity_mismatch',
                    0,
                ) + 1
                continue
            if not np.allclose(
                model_plan,
                make_zoh_candidate_plan(candidate_action, expected_frames),
                rtol=0.0,
                atol=1.0e-6,
            ):
                rejection_counts['model_input_not_zoh_candidate'] = rejection_counts.get(
                    'model_input_not_zoh_candidate',
                    0,
                ) + 1
                continue
            recomputed_plan_mismatch = normalized_action_plan_mismatch(
                actual_plan,
                model_plan,
                [0.0, 0.0],
                [21_586_451.33303, 0.43],
            )
            if abs(recomputed_plan_mismatch - float(row['plan_mismatch'])) > 1.0e-6:
                rejection_counts['plan_mismatch_recomputation_failed'] = rejection_counts.get(
                    'plan_mismatch_recomputation_failed',
                    0,
                ) + 1
                continue
            previous_pid_sha = source_pid_sha_by_condition.get(str(condition_id))
            if previous_pid_sha is not None and previous_pid_sha != row_pid_sha:
                rejection_counts['condition_pid_sha_mismatch'] = rejection_counts.get(
                    'condition_pid_sha_mismatch',
                    0,
                ) + 1
                continue
            source_conditions.add(condition_id)
            source_instances.add((condition_id, condition_instance_id))
            source_pid_sha_by_condition[str(condition_id)] = row_pid_sha
            actor_policy_version = int(row['actor_policy_version'])
            actor_policy_sha256 = str(row.get('actor_policy_sha256', '')).lower()
            actor_policy_versions.add(actor_policy_version)
            if actor_policy_sha256:
                actor_policy_sha256s.add(actor_policy_sha256)
            sample_count_by_plan[PLAN_KIND_ZOH_CANDIDATE] += 1
            shard_records.append({
                'history': history,
                'model_input_plan': np.pad(
                    model_plan,
                    ((0, max_frames - expected_frames), (0, 0)),
                ),
                'actual_future_action_plan': np.pad(
                    actual_plan,
                    ((0, max_frames - expected_frames), (0, 0)),
                ),
                'future_actual_torque_pitch': response,
                'future_peak_ratio': peaks,
                'timestamp_s': timestamp_s,
                'condition_id': condition_id,
                'condition_instance_id': condition_instance_id,
                'horizon_s': horizon_s,
                'horizon_frames': expected_frames,
                'plan_kind': PLAN_KIND_ZOH_CANDIDATE,
                'optimization_state': str(row['optimization_state']),
                'plan_mismatch': float(row['plan_mismatch']),
                'zoh_calibration_eligible': True,
                'strict_case_matched': bool(row['strict_case_matched']),
                'pid_raw_sha256': str(row['pid_raw_sha256']),
                'pid_raw_csv_path': str(predictor_identity['pid_raw_csv_path']),
                'raw_source_sha256': str(predictor_identity['raw_source_sha256']),
                'predictor_artifact_sha256': predictor_sha256,
                'actor_policy_version': actor_policy_version,
                'actor_policy_sha256': actor_policy_sha256,
                'actor_policy_window_frozen': bool(row['actor_policy_window_frozen']),
            })
            if len(shard_records) >= max_rows_per_shard:
                flush_shard()
    flush_shard()
    selected_identity_by_condition = {
        str(condition_id): paired_identity_by_condition[str(condition_id)]
        for condition_id in sorted(source_conditions)
    }
    selected_raw_sha_by_condition = {
        str(condition_id): str(selected_identity_by_condition[str(condition_id)]['raw_source_sha256'])
        for condition_id in sorted(source_conditions)
    }
    manifest = {
        'schema_version': SHADOW_DATASET_SCHEMA_VERSION,
        'dataset_role': 'strict_rl_shadow_zoh_candidate',
        'source_shadow_jsonl': str(shadow_jsonl.resolve()),
        'source_shadow_sha256': _sha256(shadow_jsonl),
        'source_conditions': sorted(source_conditions),
        'source_condition_instances': [
            {'condition_id': condition_id, 'condition_instance_id': instance_id}
            for condition_id, instance_id in sorted(source_instances)
        ],
        'source_pid_raw_sha256_by_condition': source_pid_sha_by_condition,
        'actor_policy_versions': sorted(actor_policy_versions),
        'actor_policy_sha256s': sorted(actor_policy_sha256s),
        'predictor_artifact_path': str(predictor_artifact_path.resolve()),
        'predictor_artifact_sha256': predictor_sha256,
        'predictor_training_dataset_manifest_sha256': predictor_summary.get('dataset_manifest_sha256'),
        'predictor_time_partitions_s': predictor_summary.get('time_partitions_s'),
        'predictor_interface_path': str(interface_path.resolve()),
        'predictor_interface_schema_version': interface.schema_version,
        'predictor_interface_sha256': interface_sha256,
        'paired_reference_path': str(
            source_dataset_manifest.get('paired_reference_path', predictor_checkpoint['paired_reference_path'])
        ),
        'paired_reference_sha256': str(
            source_dataset_manifest.get('paired_reference_sha256', predictor_checkpoint['paired_reference_sha256'])
        ),
        'state_action_artifact_path': str(source_dataset_manifest['state_action_artifact_path']),
        'state_action_artifact_sha256': str(source_dataset_manifest['state_action_artifact_sha256']),
        'paired_identity_by_condition': selected_identity_by_condition,
        'raw_source_sha256_by_condition': selected_raw_sha_by_condition,
        'source_data_root': str(source_dataset_manifest['source_data_root']),
        'time_window_start_s': float(source_dataset_manifest['time_window_start_s']),
        'time_window_end_s': float(source_dataset_manifest['time_window_end_s']),
        'history_frames': interface.history_frames,
        'horizons_s': list(interface.horizons_s),
        'feature_dim': 31,
        'max_horizon_frames': max_frames,
        'plan_counts': sample_count_by_plan,
        'shards': shards,
        'rejection_counts': rejection_counts,
        'identity_requirements': {
            'strict_identity_required': require_strict_identity,
            'actor_execution_required': require_actor_execution,
            'outcome_tag_required': 'complete',
            'plan_kind': PLAN_KIND_ZOH_CANDIDATE,
            'horizons_s': list(interface.soft_guard_horizons_s),
        },
    }
    manifest_path = output_root / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description='Stream strict-identity RL shadow JSONL into bounded ZOH predictor shards.')
    parser.add_argument('--shadow-jsonl', type=Path, required=True)
    parser.add_argument('--interface', type=Path, default=PROJECT_ROOT / 'RL/data/load_predictor_interface_v3.json')
    parser.add_argument('--predictor-artifact', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--max-rows-per-shard', type=int, default=4096)
    parser.add_argument('--allow-non-strict-identity', action='store_true')
    parser.add_argument('--include-non-actor', action='store_true')
    args = parser.parse_args()
    manifest = build_shadow_dataset(
        shadow_jsonl=args.shadow_jsonl,
        interface_path=args.interface,
        predictor_artifact_path=args.predictor_artifact,
        output_root=args.output_root,
        max_rows_per_shard=args.max_rows_per_shard,
        require_strict_identity=not args.allow_non_strict_identity,
        require_actor_execution=not args.include_non_actor,
    )
    print(json.dumps({
        'output_root': str(args.output_root),
        'schema_version': manifest['schema_version'],
        'sample_count': sum(int(shard['samples']) for shard in manifest['shards']),
        'source_conditions': manifest['source_conditions'],
        'rejection_counts': manifest['rejection_counts'],
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
