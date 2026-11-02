from __future__ import annotations

import argparse
from collections import deque
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from RL.utils.load_predictor_contract import (
    PLAN_KIND_REALIZED,
    PLAN_KIND_ZOH_CANDIDATE,
    LoadPredictorInterface,
    normalized_action_plan_mismatch,
)
from RL.utils.load_predictor_shadow import (
    BLADE_ROOT_MY_FIELDS,
    FEATURE_DIM,
    PHYSICAL_FEATURE_FIELDS,
    STATE_ENCODING_WIDTH,
)
from RL.utils.operating_state import OperatingPoint, OperatingStateClassifier
from RL.utils.paired_pid_reference import PairedPidReference
from RL.utils.state_action_artifact import StateActionEnvelopeArtifact, encode_state_regime


DATASET_SCHEMA_VERSION = 'load_predictor_dataset_v2'
TIME_FIELD = 'Time'
ACTION_FIELDS = ('O_PcsTorqueDemand', 'O_PitchPositionDemand1')
REQUIRED_FIELDS = (
    TIME_FIELD,
    'I_RotorAzimuth',
    *PHYSICAL_FEATURE_FIELDS,
    *BLADE_ROOT_MY_FIELDS,
    *ACTION_FIELDS,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _finite(value: Any, *, path: Path, row_number: int, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{path.name} row {row_number} has invalid {field}') from exc
    if not math.isfinite(parsed):
        raise ValueError(f'{path.name} row {row_number} has non-finite {field}')
    return parsed


def _iter_feature_frames(
    path: Path,
    *,
    condition_id: int,
    state_artifact: StateActionEnvelopeArtifact,
    pid_peak_abs_nm: np.ndarray,
) -> Iterable[dict[str, Any]]:
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        missing = sorted(set(REQUIRED_FIELDS) - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f'{path.name} is missing load predictor fields: {missing}')
        previous_time: float | None = None
        previous_action: np.ndarray | None = None
        previous_blade_my: np.ndarray | None = None
        classifier = state_artifact.classifier()
        count = 0
        for row_number, row in enumerate(reader, start=2):
            parsed = {field: _finite(row[field], path=path, row_number=row_number, field=field) for field in REQUIRED_FIELDS}
            timestamp_s = parsed[TIME_FIELD]
            if previous_time is not None:
                delta_s = timestamp_s - previous_time
                if delta_s <= 0.0:
                    raise ValueError(f'{path.name} timestamps must be strictly increasing')
                if not math.isclose(delta_s, 0.02, rel_tol=0.0, abs_tol=1.0e-6):
                    raise ValueError(f'{path.name} must have a contiguous 20 ms sample interval')
            previous_time = parsed[TIME_FIELD]
            action = np.clip(
                np.asarray([parsed[field] for field in ACTION_FIELDS], dtype=np.float64),
                [0.0, 0.0],
                [21_586_451.33303, 0.43],
            )
            blade_my = np.asarray([parsed[field] for field in BLADE_ROOT_MY_FIELDS], dtype=np.float64)
            blade_my_rate = (
                np.zeros(3, dtype=np.float64)
                if previous_blade_my is None
                else (blade_my - previous_blade_my) / 0.02
            )
            if previous_action is None:
                previous_action = action.copy()
            classification = classifier.classify(OperatingPoint(
                timestamp_s=timestamp_s,
                wind_speed_mps=parsed['I_AverageWindSpeed'],
                rotor_speed_rad_s=parsed['I_RotorSpeed'],
                real_power_w=parsed['I_RealPower'],
                pitch_rad=parsed['I_PitchPosition1'],
                previous_torque_nm=float(previous_action[0]),
                previous_pitch_rad=float(previous_action[1]),
            ))
            state_encoding = np.asarray(encode_state_regime(classification), dtype=np.float64)
            azimuth = parsed['I_RotorAzimuth']
            physical = np.asarray([parsed[field] for field in PHYSICAL_FEATURE_FIELDS], dtype=np.float64)
            feature = np.concatenate((
                physical,
                [math.sin(azimuth), math.cos(azimuth)],
                blade_my,
                blade_my_rate,
                action,
                action,
                state_encoding,
                1.0 - np.abs(blade_my) / pid_peak_abs_nm,
            )).astype(np.float32)
            if feature.shape != (FEATURE_DIM,) or not np.isfinite(feature).all():
                raise ValueError(f'Invalid load predictor feature at condition={condition_id} row={row_number}')
            yield {
                'index': count,
                'timestamp_s': timestamp_s,
                'feature': feature,
                'action': action,
                'blade_root_my': blade_my,
                'electrical_torque_nm': parsed['I_ElectricalTorque'],
                'pitch_rad': parsed['I_PitchPosition1'],
                'optimization_state': classification.optimization_state.value,
            }
            count += 1
            previous_action = action
            previous_blade_my = blade_my
    if count == 0:
        raise ValueError(f'{path.name} has no rows')


def _condition_id(path: Path) -> int:
    stem = path.stem.lower()
    marker = 'windspeed'
    if marker not in stem:
        raise ValueError(f'Cannot infer condition ID from {path.name}')
    return int(stem.split(marker, 1)[1])


def _pad_plan(plan: np.ndarray, max_horizon_frames: int) -> np.ndarray:
    padded = np.zeros((max_horizon_frames, 2), dtype=np.float32)
    padded[:len(plan)] = plan.astype(np.float32)
    return padded


def _write_shard(path: Path, records: list[dict[str, Any]], max_horizon_frames: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        history=np.stack([record['history'] for record in records]).astype(np.float32),
        model_input_plan=np.stack([record['model_input_plan'] for record in records]).astype(np.float32),
        actual_future_action_plan=np.stack([record['actual_future_action_plan'] for record in records]).astype(np.float32),
        future_actual_torque_pitch=np.stack([record['future_actual_torque_pitch'] for record in records]).astype(np.float32),
        future_peak_ratio=np.stack([record['future_peak_ratio'] for record in records]).astype(np.float32),
        timestamp_s=np.asarray([record['timestamp_s'] for record in records], dtype=np.float64),
        condition_id=np.asarray([record['condition_id'] for record in records], dtype=np.int32),
        horizon_s=np.asarray([record['horizon_s'] for record in records], dtype=np.float32),
        horizon_frames=np.asarray([record['horizon_frames'] for record in records], dtype=np.int32),
        plan_kind=np.asarray([record['plan_kind'] for record in records], dtype='U32'),
        optimization_state=np.asarray([record['optimization_state'] for record in records], dtype='U64'),
        plan_mismatch=np.asarray([record['plan_mismatch'] for record in records], dtype=np.float32),
        zoh_calibration_eligible=np.asarray([record['zoh_calibration_eligible'] for record in records], dtype=np.bool_),
        pid_raw_sha256=np.asarray([record['pid_raw_sha256'] for record in records], dtype='U64'),
        pid_raw_csv_path=np.asarray([record['pid_raw_csv_path'] for record in records], dtype='U512'),
        raw_source_sha256=np.asarray([record['raw_source_sha256'] for record in records], dtype='U64'),
    )


def build_dataset(
    *,
    data_root: Path,
    paired_reference_path: Path,
    state_artifact_path: Path,
    interface_path: Path,
    output_root: Path,
    max_samples_per_shard: int = 4096,
    stride: int = 1,
    condition_ids: Iterable[int] | None = None,
) -> dict[str, Any]:
    if max_samples_per_shard <= 0 or stride <= 0:
        raise ValueError('max_samples_per_shard and stride must be positive')
    interface = LoadPredictorInterface.from_path(interface_path)
    state_artifact = StateActionEnvelopeArtifact.from_path(state_artifact_path)
    paired_reference = PairedPidReference.from_path(paired_reference_path, project_root=PROJECT_ROOT)
    selected = None if condition_ids is None else {int(value) for value in condition_ids}
    paths = sorted(data_root.glob('data_windspeed*.csv'), key=_condition_id)
    if selected is not None:
        paths = [path for path in paths if _condition_id(path) in selected]
    if not paths:
        raise FileNotFoundError(f'No load predictor CSV files found in {data_root}')
    if {_condition_id(path) for path in paths} != set(paired_reference.entries):
        raise ValueError('Predictor source conditions must exactly match the paired PID reference conditions')
    output_root.mkdir(parents=True, exist_ok=True)
    if (output_root / 'manifest.json').exists() or any(output_root.glob('shard_*.npz')):
        raise FileExistsError(f'Refusing to overwrite existing load predictor dataset: {output_root}')
    max_horizon_frames = max(interface.horizon_frames)
    records: list[dict[str, Any]] = []
    shards: list[dict[str, Any]] = []
    sample_count = 0
    source_time_start_s: float | None = None
    source_time_end_s: float | None = None
    plan_counts = {PLAN_KIND_REALIZED: 0, PLAN_KIND_ZOH_CANDIDATE: 0}
    raw_source_sha256_by_condition: dict[str, str] = {}
    paired_identity_by_condition: dict[str, dict[str, Any]] = {}

    def flush_shard() -> None:
        if not records:
            return
        shard_index = len(shards)
        shard_path = output_root / f'shard_{shard_index:05d}.npz'
        _write_shard(shard_path, records, max_horizon_frames)
        shards.append({'path': shard_path.name, 'samples': len(records), 'sha256': _sha256(shard_path)})
        records.clear()

    for path in paths:
        condition_id = _condition_id(path)
        raw_source_sha256 = _sha256(path)
        raw_source_sha256_by_condition[str(condition_id)] = raw_source_sha256
        entry = paired_reference.entry(condition_id)
        if entry.pid_raw_sha256 != raw_source_sha256:
            raise ValueError(
                f'PID reference raw SHA does not match dataset source for condition={condition_id}'
            )
        paired_identity_by_condition[str(condition_id)] = {
            'scenario_id': entry.scenario_id,
            'condition_id': condition_id,
            'pid_raw_csv_path': _portable_path(path),
            'pid_raw_sha256': entry.pid_raw_sha256,
            'raw_source_sha256': raw_source_sha256,
            'time_window_start_s': entry.time_window_start_s,
            'time_window_end_s': entry.time_window_end_s,
            'expected_trace_signature': entry.trace_signature,
            'expected_trace_sample_count': int(entry.timestamps_s.size),
            'time_signature_quantization_s': entry.time_signature_quantization_s,
            'wind_signature_quantization_mps': entry.wind_signature_quantization_mps,
        }
        horizon_frames = interface.horizon_frames
        required_buffer_frames = interface.history_frames + max_horizon_frames
        feature_ring: deque[dict[str, Any]] = deque(maxlen=required_buffer_frames)
        start_index = interface.history_frames - 1
        condition_start_s: float | None = None
        condition_end_s: float | None = None
        for frame in _iter_feature_frames(
            path,
            condition_id=condition_id,
            state_artifact=state_artifact,
            pid_peak_abs_nm=entry.blade_root_peak_abs_nm,
        ):
            frame_index = int(frame['index'])
            timestamp_s = float(frame['timestamp_s'])
            condition_start_s = timestamp_s if condition_start_s is None else condition_start_s
            condition_end_s = timestamp_s
            feature_ring.append(frame)
            if (
                len(feature_ring) == required_buffer_frames
                and frame_index >= start_index + max_horizon_frames
            ):
                start_frame_index = frame_index - max_horizon_frames
                if (
                    start_frame_index < start_index
                    or (start_frame_index - start_index) % stride != 0
                ):
                    continue
                history_start_index = start_frame_index - interface.history_frames + 1
                ring_start_index = frame_index - required_buffer_frames + 1
                if history_start_index < ring_start_index:
                    raise RuntimeError('Streaming predictor dataset ring lost required history context')
                history_start_offset = history_start_index - ring_start_index
                history = np.stack([
                    feature_ring[offset]['feature']
                    for offset in range(history_start_offset, history_start_offset + interface.history_frames)
                ]).astype(np.float32)
                future_start_offset = start_frame_index + 1 - ring_start_index
                actual_plan = np.stack([
                    feature_ring[future_start_offset + offset]['action']
                    for offset in range(max_horizon_frames)
                ]).astype(np.float32)
                current_frame = feature_ring[start_frame_index - ring_start_index]
                start_timestamp_s = float(current_frame['timestamp_s'])
                candidate_plan = np.repeat(
                    np.asarray(current_frame['action'], dtype=np.float64)[None, :],
                    max_horizon_frames,
                    axis=0,
                )
                start_my = np.asarray(current_frame['blade_root_my'], dtype=np.float64)
                states_at_start = str(current_frame['optimization_state'])
                response_by_horizon = {
                    horizon: np.asarray([
                        feature_ring[start_frame_index + horizon - ring_start_index]['electrical_torque_nm'],
                        feature_ring[start_frame_index + horizon - ring_start_index]['pitch_rad'],
                    ], dtype=np.float32)
                    for horizon in horizon_frames
                }
                future_absolute_peaks: dict[int, np.ndarray] = {}
                for horizon in horizon_frames:
                    window_my = np.stack([
                        feature_ring[
                            start_frame_index + offset + 1 - ring_start_index
                        ]['blade_root_my']
                        for offset in range(horizon)
                    ])
                    future_absolute_peaks[horizon] = np.max(
                        np.abs(window_my),
                        axis=0,
                    ) / entry.blade_root_peak_abs_nm
                for horizon_s, horizon in zip(interface.horizons_s, horizon_frames, strict=True):
                    future_peak_ratio = future_absolute_peaks[horizon].astype(np.float32)
                    response = response_by_horizon[horizon]
                    horizon_plan = actual_plan[:horizon]
                    horizon_candidate_plan = candidate_plan[:horizon]
                    horizon_mismatch = normalized_action_plan_mismatch(
                        horizon_plan,
                        horizon_candidate_plan,
                        [0.0, 0.0],
                        [21_586_451.33303, 0.43],
                    )
                    padded_plan = _pad_plan(horizon_plan, max_horizon_frames)
                    padded_candidate = _pad_plan(horizon_candidate_plan, max_horizon_frames)
                    base_record = {
                        'history': history,
                        'future_actual_torque_pitch': response,
                        'future_peak_ratio': future_peak_ratio,
                        'timestamp_s': start_timestamp_s,
                        'condition_id': condition_id,
                        'pid_raw_sha256': entry.pid_raw_sha256,
                        'pid_raw_csv_path': _portable_path(path),
                        'raw_source_sha256': raw_source_sha256,
                        'horizon_s': horizon_s,
                        'horizon_frames': horizon,
                        'optimization_state': states_at_start,
                        'plan_mismatch': horizon_mismatch,
                    }
                    sample_records = [{
                        **base_record,
                        'model_input_plan': padded_plan,
                        'actual_future_action_plan': padded_plan,
                        'plan_kind': PLAN_KIND_REALIZED,
                        'zoh_calibration_eligible': False,
                    }]
                    plan_counts[PLAN_KIND_REALIZED] += 1
                    if (
                        horizon_s in interface.soft_guard_horizons_s
                        and horizon_mismatch <= interface.max_plan_mismatch
                    ):
                        sample_records.append({
                            **base_record,
                            'model_input_plan': padded_candidate,
                            'actual_future_action_plan': padded_plan,
                            'plan_kind': PLAN_KIND_ZOH_CANDIDATE,
                            'zoh_calibration_eligible': True,
                        })
                        plan_counts[PLAN_KIND_ZOH_CANDIDATE] += 1
                    for sample_record in sample_records:
                        if len(records) >= max_samples_per_shard:
                            flush_shard()
                        records.append(sample_record)
                    sample_count += 1
        if condition_start_s is None or condition_end_s is None:
            raise ValueError(f'{path.name} has no samples')
        expected_start = entry.time_window_start_s
        expected_end = entry.time_window_end_s
        if (
            not math.isclose(condition_start_s, expected_start, rel_tol=0.0, abs_tol=1.0e-6)
            or not math.isclose(condition_end_s, expected_end, rel_tol=0.0, abs_tol=1.0e-6)
        ):
            raise ValueError(f'Predictor source condition={condition_id} does not exactly match paired reference window')
        source_time_start_s = condition_start_s if source_time_start_s is None else min(
            source_time_start_s,
            condition_start_s,
        )
        source_time_end_s = condition_end_s if source_time_end_s is None else max(
            source_time_end_s,
            condition_end_s,
        )
    flush_shard()
    manifest = {
        'schema_version': DATASET_SCHEMA_VERSION,
        'predictor_interface_schema_version': interface.schema_version,
        'source_data_root': _portable_path(data_root),
        'source_conditions': [_condition_id(path) for path in paths],
        'paired_reference_path': _portable_path(paired_reference_path),
        'paired_reference_sha256': _sha256(paired_reference_path),
        'raw_source_sha256_by_condition': raw_source_sha256_by_condition,
        'paired_identity_by_condition': paired_identity_by_condition,
        'state_action_artifact_path': _portable_path(state_artifact_path),
        'state_action_artifact_sha256': state_artifact.source_sha256,
        'predictor_interface_path': _portable_path(interface_path),
        'predictor_interface_sha256': _sha256(interface_path),
        'history_frames': interface.history_frames,
        'horizons_s': list(interface.horizons_s),
        'feature_dim': FEATURE_DIM,
        'time_window_start_s': source_time_start_s,
        'time_window_end_s': source_time_end_s,
        'max_horizon_frames': max_horizon_frames,
        'stride': stride,
        'max_plan_mismatch': interface.max_plan_mismatch,
        'plan_counts': plan_counts,
        'sample_count_before_plan_expansion': sample_count,
        'shards': shards,
        'plan_semantics': {
            PLAN_KIND_REALIZED: 'future actual actions are labels/diagnostics only',
            PLAN_KIND_ZOH_CANDIDATE: 'current action held over 0.10/0.20 s and only mismatch-qualified samples are calibration eligible',
        },
    }
    (output_root / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description='Build causal load-predictor realized/ZOH training shards.')
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--paired-reference', type=Path, required=True)
    parser.add_argument('--state-artifact', type=Path, required=True)
    parser.add_argument('--interface', type=Path, default=PROJECT_ROOT / 'RL/data/load_predictor_interface_v3.json')
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--max-samples-per-shard', type=int, default=4096)
    parser.add_argument('--stride', type=int, default=1)
    args = parser.parse_args()
    manifest = build_dataset(
        data_root=args.data_root,
        paired_reference_path=args.paired_reference,
        state_artifact_path=args.state_artifact,
        interface_path=args.interface,
        output_root=args.output_root,
        max_samples_per_shard=args.max_samples_per_shard,
        stride=args.stride,
    )
    print(json.dumps({
        'output_root': str(args.output_root),
        'schema_version': manifest['schema_version'],
        'sample_count_before_plan_expansion': manifest['sample_count_before_plan_expansion'],
        'plan_counts': manifest['plan_counts'],
        'shard_count': len(manifest['shards']),
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
