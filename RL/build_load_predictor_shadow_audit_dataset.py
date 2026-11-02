from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import sys
from typing import Any
import zlib

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from RL.utils.load_predictor_contract import LoadPredictorInterface


SHADOW_AUDIT_DATASET_SCHEMA_VERSION = 'load_predictor_rl_shadow_audit_dataset_v1'


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def build_shadow_audit_dataset(
    input_jsonl: Path,
    output_root: Path,
    *,
    interface_path: Path,
    max_records: int = 250_000,
    max_records_per_shard: int = 1024,
) -> dict[str, Any]:
    if max_records <= 0 or max_records_per_shard <= 0:
        raise ValueError('max_records and max_records_per_shard must be positive')
    interface = LoadPredictorInterface.from_path(interface_path)
    interface_sha256 = _sha256(interface_path)
    input_jsonl = input_jsonl.expanduser().resolve()
    if not input_jsonl.is_file():
        raise FileNotFoundError(f'RL shadow audit JSONL does not exist: {input_jsonl}')
    output_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    shards: list[dict[str, Any]] = []
    plan_counts = {'realized_plan': 0, 'zoh_candidate': 0}
    prediction_valid_count = 0
    complete_outcome_count = 0
    strict_identity_count = 0
    condition_ids: set[int] = set()
    eligible_count = 0
    input_count = 0
    artifact_sha_by_record: str | None = None
    interface_sha_by_record: str | None = None

    def flush_shard() -> None:
        if not records:
            return
        shard_path = output_root / f'shard_{len(shards):05d}.npz'
        np.savez_compressed(
            shard_path,
            history=np.stack([record['history'] for record in records]).astype(np.float32),
            model_input_plan=np.stack([record['model_input_plan'] for record in records]).astype(np.float32),
            actual_future_action_plan=np.stack([
                record['actual_future_action_plan'] for record in records
            ]).astype(np.float32),
            future_actual_torque_pitch=np.asarray([
                record['future_actual_torque_pitch'] for record in records
            ], dtype=np.float32),
            future_peak_ratio=np.asarray([
                record['future_peak_ratio'] for record in records
            ], dtype=np.float32),
            timestamp_s=np.asarray([record['timestamp_s'] for record in records], dtype=np.float64),
            condition_id=np.asarray([record['condition_id'] for record in records], dtype=np.int32),
            condition_instance_id=np.asarray([
                record['condition_instance_id'] for record in records
            ], dtype=np.int64),
            horizon_s=np.asarray([record['horizon_s'] for record in records], dtype=np.float32),
            horizon_frames=np.asarray([record['horizon_frames'] for record in records], dtype=np.int32),
            plan_kind=np.asarray([record['plan_kind'] for record in records], dtype='U32'),
            optimization_state=np.asarray([record['optimization_state'] for record in records], dtype='U64'),
            plan_mismatch=np.asarray([record['plan_mismatch'] for record in records], dtype=np.float32),
            calibration_eligible=np.asarray([
                record['calibration_eligible'] for record in records
            ], dtype=np.bool_),
            strict_case_matched=np.asarray([
                record['strict_case_matched'] for record in records
            ], dtype=np.bool_),
            pid_raw_sha256=np.asarray([record['pid_raw_sha256'] for record in records], dtype='U64'),
            execution_mode=np.asarray([record['execution_mode'] for record in records], dtype='U64'),
            outcome_tag=np.asarray([record['outcome_tag'] for record in records], dtype='U32'),
            prediction_available=np.asarray([
                record['prediction_available'] for record in records
            ], dtype=np.bool_),
            predicted_future_torque_pitch=np.asarray([
                record['predicted_future_torque_pitch'] for record in records
            ], dtype=np.float32),
            predicted_peak_ratio=np.asarray([
                record['predicted_peak_ratio'] for record in records
            ], dtype=np.float32),
        )
        shards.append({
            'path': shard_path.name,
            'samples': len(records),
            'sha256': _sha256(shard_path),
        })
        records.clear()

    with input_jsonl.open('r', encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            if len(line.encode('utf-8')) > 64 * 1024:
                raise ValueError(f'Shadow audit row exceeds 64 KiB at {input_jsonl}:{line_number}')
            input_count += 1
            if input_count > max_records:
                raise ValueError(f'Input audit JSONL exceeds max_records={max_records}: {input_jsonl}')
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'Invalid shadow audit JSON at {input_jsonl}:{line_number}') from exc
            if not isinstance(row, dict) or row.get('schema_version') != 'load_predictor_shadow_audit_record_v1':
                raise ValueError(f'Incompatible shadow audit row at {input_jsonl}:{line_number}')
            row_artifact_sha = str(row.get('predictor_artifact_sha256', '')).lower()
            row_interface_sha = str(row.get('predictor_interface_sha256', '')).lower()
            if (
                len(row_artifact_sha) != 64
                or any(character not in '0123456789abcdef' for character in row_artifact_sha)
                or row_interface_sha != interface_sha256
            ):
                raise ValueError(f'Shadow audit artifact/interface provenance is invalid at {input_jsonl}:{line_number}')
            if artifact_sha_by_record is None:
                artifact_sha_by_record = row_artifact_sha
                interface_sha_by_record = row_interface_sha
            elif row_artifact_sha != artifact_sha_by_record or row_interface_sha != interface_sha_by_record:
                raise ValueError('Shadow audit JSONL mixes predictor/interface artifact identities')
            history_shape = tuple(int(value) for value in row.get('history_shape', ()))
            if history_shape != (interface.history_frames, 31):
                raise ValueError(f'Shadow history has an incompatible shape at {input_jsonl}:{line_number}')
            if row.get('history_dtype') != '<f4':
                raise ValueError(f'Shadow history dtype is incompatible at {input_jsonl}:{line_number}')
            try:
                history_bytes = zlib.decompress(base64.b64decode(row['history_zlib_base64'], validate=True))
            except (KeyError, ValueError, zlib.error) as exc:
                raise ValueError(f'Shadow history payload is invalid at {input_jsonl}:{line_number}') from exc
            expected_bytes = int(np.prod(history_shape)) * np.dtype('<f4').itemsize
            if len(history_bytes) != expected_bytes:
                raise ValueError(f'Shadow history byte length mismatch at {input_jsonl}:{line_number}')
            history = np.frombuffer(history_bytes, dtype='<f4').reshape(history_shape).copy()
            try:
                candidate_plan = np.asarray(row['model_input_plan'], dtype=np.float32)
                actual_plan = np.asarray(row['actual_future_action_plan'], dtype=np.float32)
                horizon_s = float(row['horizon_s'])
                horizon_index = min(
                    range(len(interface.horizons_s)),
                    key=lambda index: abs(interface.horizons_s[index] - horizon_s),
                )
                horizon_frames = int(row['horizon_frames'])
                condition_id = int(row['condition_id'])
                condition_instance_id = int(row['condition_instance_id'])
                timestamp_s = float(row['timestamp_s'])
                mismatch = float(row['plan_mismatch'])
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f'Shadow audit metadata is invalid at {input_jsonl}:{line_number}') from exc
            if abs(interface.horizons_s[horizon_index] - horizon_s) > 1.0e-6:
                raise ValueError(f'Shadow audit contains unsupported horizon={horizon_s}')
            if horizon_frames != interface.horizon_frames[horizon_index]:
                raise ValueError(f'Shadow audit horizon frame count mismatch at {input_jsonl}:{line_number}')
            if (
                candidate_plan.shape != (horizon_frames, 2)
                or actual_plan.shape != (horizon_frames, 2)
                or candidate_plan.ndim != 2
            ):
                raise ValueError(f'Shadow audit action plan shapes differ at {input_jsonl}:{line_number}')
            plan_kind = str(row.get('plan_kind', ''))
            if plan_kind not in plan_counts:
                raise ValueError(f'Shadow audit plan kind is invalid: {plan_kind!r}')
            try:
                actual_response = np.asarray(row['actual_future_response'], dtype=np.float32)
                actual_peaks = np.asarray(row['actual_future_peak_ratio_by_blade'], dtype=np.float32)
                prediction_available = bool(row.get('predicted_response_available', False))
                if prediction_available:
                    predicted_response = np.asarray([
                        row['predicted_future_torque_nm'],
                        row['predicted_future_pitch_rad'],
                    ], dtype=np.float32)
                    predicted_peaks = np.asarray(row['predicted_peak_ratio_by_blade'], dtype=np.float32)
                else:
                    predicted_response = np.zeros(2, dtype=np.float32)
                    predicted_peaks = np.zeros(3, dtype=np.float32)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f'Shadow audit response labels are invalid at {input_jsonl}:{line_number}') from exc
            if (
                actual_response.shape != (2,)
                or actual_peaks.shape != (3,)
                or not np.isfinite(actual_response).all()
                or not np.isfinite(actual_peaks).all()
                or not np.isfinite(candidate_plan).all()
                or not np.isfinite(actual_plan).all()
                or not np.isfinite(history).all()
                or not np.isfinite(timestamp_s)
                or not np.isfinite(mismatch)
            ):
                raise ValueError('Shadow audit actual response labels have incompatible shape')
            if predicted_response.shape != (2,) or predicted_peaks.shape != (3,):
                raise ValueError('Shadow audit predicted response has incompatible shape')
            strict_case_matched = bool(row.get('strict_case_matched', False))
            pid_sha = str(row.get('pid_raw_sha256', '')).lower()
            if strict_case_matched and (len(pid_sha) != 64 or any(ch not in '0123456789abcdef' for ch in pid_sha)):
                raise ValueError('Strict shadow sample is missing a valid PID raw SHA-256')
            record = {
                'history': history,
                'model_input_plan': np.pad(
                    candidate_plan,
                    ((0, max(interface.horizon_frames) - horizon_frames), (0, 0)),
                ),
                'actual_future_action_plan': np.pad(
                    actual_plan,
                    ((0, max(interface.horizon_frames) - horizon_frames), (0, 0)),
                ),
                'future_actual_torque_pitch': actual_response,
                'future_peak_ratio': actual_peaks,
                'timestamp_s': timestamp_s,
                'condition_id': condition_id,
                'condition_instance_id': condition_instance_id,
                'horizon_s': horizon_s,
                'horizon_frames': horizon_frames,
                'plan_kind': plan_kind,
                'optimization_state': str(row['optimization_state']),
                'plan_mismatch': mismatch,
                'calibration_eligible': bool(row.get('calibration_eligible', False)),
                'strict_case_matched': strict_case_matched,
                'pid_raw_sha256': pid_sha,
                'execution_mode': str(row.get('execution_mode', 'unknown')),
                'outcome_tag': str(row.get('outcome_tag', 'unknown')),
                'prediction_available': prediction_available,
                'predicted_future_torque_pitch': predicted_response,
                'predicted_peak_ratio': predicted_peaks,
            }
            records.append(record)
            plan_counts[plan_kind] += 1
            prediction_valid_count += int(prediction_available)
            complete_outcome_count += int(record['outcome_tag'] == 'complete')
            strict_identity_count += int(strict_case_matched)
            condition_ids.add(condition_id)
            eligible_count += int(record['calibration_eligible'])
            if len(records) >= max_records_per_shard:
                flush_shard()
    flush_shard()
    manifest = {
        'schema_version': SHADOW_AUDIT_DATASET_SCHEMA_VERSION,
        'source_audit_jsonl_path': str(input_jsonl),
        'source_audit_jsonl_sha256': _sha256(input_jsonl),
        'predictor_interface_path': str(interface_path.resolve()),
        'predictor_interface_sha256': interface_sha256,
        'predictor_artifact_sha256': artifact_sha_by_record,
        'history_frames': interface.history_frames,
        'feature_dim': 31,
        'horizons_s': list(interface.horizons_s),
        'plan_counts': plan_counts,
        'prediction_available_count': prediction_valid_count,
        'complete_outcome_count': complete_outcome_count,
        'strict_identity_count': strict_identity_count,
        'calibration_eligible_count': eligible_count,
        'condition_ids': sorted(condition_ids),
        'sample_count': input_count,
        'max_records_per_shard': max_records_per_shard,
        'shards': shards,
        'split_status': 'raw_infer_holdout_required_before_predictor_training_or_calibration',
    }
    (output_root / 'manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description='Build bounded RL shadow predictor training/calibration NPZ shards.')
    parser.add_argument('--input-jsonl', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--interface', type=Path, default=PROJECT_ROOT / 'RL/data/load_predictor_interface_v3.json')
    parser.add_argument('--max-records', type=int, default=250_000)
    parser.add_argument('--max-records-per-shard', type=int, default=1_024)
    args = parser.parse_args()
    manifest = build_shadow_audit_dataset(
        args.input_jsonl,
        args.output_root,
        interface_path=args.interface,
        max_records=args.max_records,
        max_records_per_shard=args.max_records_per_shard,
    )
    print(json.dumps({
        'output_root': str(args.output_root.resolve()),
        'sample_count': manifest['sample_count'],
        'plan_counts': manifest['plan_counts'],
        'prediction_available_count': manifest['prediction_available_count'],
        'strict_identity_count': manifest['strict_identity_count'],
        'calibration_eligible_count': manifest['calibration_eligible_count'],
        'shard_count': len(manifest['shards']),
        'split_status': manifest['split_status'],
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
