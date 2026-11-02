from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
from statistics import NormalDist
import sys
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from RL.data.load_predictor_dataset import LoadPredictorDataset, estimate_normalization
from RL.models.load_predictor_tcn import CausalTCNLoadPredictor
from RL.utils.load_predictor_contract import LoadPredictorInterface


PREDICTOR_SPLIT_SCHEMA_VERSION = 'load_predictor_condition_split_v2'


class _ShardBatchSampler:
    """Shuffle locally and across shards without thrashing the lazy one-shard cache."""

    def __init__(self, dataset: LoadPredictorDataset, batch_size: int, *, seed: int = 42) -> None:
        if batch_size <= 0:
            raise ValueError('batch_size must be positive')
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0
        self.indices_by_shard: dict[int, list[int]] = defaultdict(list)
        for dataset_index, (shard_index, _sample_index) in enumerate(dataset._indices):
            self.indices_by_shard[int(shard_index)].append(dataset_index)

    def __iter__(self):
        randomizer = random.Random(self.seed + self.epoch)
        self.epoch += 1
        shard_batches: list[list[list[int]]] = []
        shard_order = list(self.indices_by_shard)
        randomizer.shuffle(shard_order)
        for shard_index in shard_order:
            shard_indices = self.indices_by_shard[shard_index]
            indices = list(shard_indices)
            randomizer.shuffle(indices)
            batches = [
                indices[offset:offset + self.batch_size]
                for offset in range(0, len(indices), self.batch_size)
            ]
            randomizer.shuffle(batches)
            shard_batches.append(batches)
        for batches in shard_batches:
            yield from batches

    def __len__(self) -> int:
        return sum(
            math.ceil(len(indices) / self.batch_size)
            for indices in self.indices_by_shard.values()
        )


def build_condition_partitions(condition_ids: list[int] | tuple[int, ...]) -> dict[str, list[int]]:
    ordered = sorted({int(value) for value in condition_ids})
    if len(ordered) < 4:
        raise ValueError('At least four distinct conditions are required for train/val/calibration/audit splits')
    # Whole-condition holdout; the last three partitions are kept disjoint from
    # both optimizer fitting and each other. Coverage audit is never used to
    # tune either network weights or conformal margins.
    train_end = max(1, int(math.floor(len(ordered) * 0.65)))
    validation_end = max(train_end + 1, int(math.floor(len(ordered) * 0.78)))
    calibration_end = max(validation_end + 1, int(math.floor(len(ordered) * 0.87)))
    if calibration_end >= len(ordered):
        calibration_end = len(ordered) - 1
    if validation_end >= calibration_end:
        validation_end = calibration_end - 1
    partitions = {
        'train': ordered[:train_end],
        'validation': ordered[train_end:validation_end],
        'calibration': ordered[validation_end:calibration_end],
        'coverage_audit': ordered[calibration_end:],
    }
    if any(not values for values in partitions.values()):
        raise ValueError('Condition partitioning produced an empty train/validation/calibration/audit split')
    flattened = [condition for values in partitions.values() for condition in values]
    if len(flattened) != len(ordered) or set(flattened) != set(ordered):
        raise RuntimeError('Load predictor whole-condition partitions do not cover the source conditions')
    if sum(len(values) for values in partitions.values()) != len(set(flattened)):
        raise RuntimeError('Load predictor condition partitions overlap')
    return partitions


def build_time_partitions(
    start_s: float,
    end_s: float,
    *,
    interface: LoadPredictorInterface,
) -> dict[str, tuple[float, float]]:
    if not math.isfinite(start_s) or not math.isfinite(end_s) or end_s <= start_s:
        raise ValueError('Predictor source time window must be finite and increasing')
    duration_s = end_s - start_s
    purge_s = (
        interface.history_frames - 1 + max(interface.horizon_frames)
    ) * interface.step_duration_s
    train_end = start_s + 0.34 * duration_s
    validation_start = train_end + purge_s
    validation_end = start_s + 0.37 * duration_s
    calibration_start = validation_end + purge_s
    calibration_end = start_s + 0.65 * duration_s
    audit_start = calibration_end + purge_s
    audit_end = end_s + max(interface.step_duration_s, 1.0e-9)
    partitions = {
        'train': (start_s, train_end),
        'validation': (validation_start, validation_end),
        'calibration': (calibration_start, calibration_end),
        'coverage_audit': (audit_start, audit_end),
    }
    if any(end <= start for start, end in partitions.values()):
        raise ValueError('Source window is too short for purged predictor time partitions')
    ordered = [partitions[name] for name in ('train', 'validation', 'calibration', 'coverage_audit')]
    for previous, following in zip(ordered, ordered[1:]):
        if previous[1] + 1.0e-12 > following[0]:
            raise RuntimeError('Purged predictor time partitions overlap')
    return partitions


def _resolve_time_partitions(
    dataset: LoadPredictorDataset,
    interface: LoadPredictorInterface,
    override: Mapping[str, tuple[float, float]] | None,
) -> dict[str, tuple[float, float]]:
    names = ('train', 'validation', 'calibration', 'coverage_audit')
    purge_s = (
        interface.history_frames - 1 + max(interface.horizon_frames)
    ) * interface.step_duration_s
    if override is not None:
        if set(override) != set(names):
            raise ValueError(f'time_partitions_s must define exactly {names}')
        result = {
            name: (float(override[name][0]), float(override[name][1]))
            for name in names
        }
        if any(
            not math.isfinite(start) or not math.isfinite(end) or end <= start
            for start, end in result.values()
        ):
            raise ValueError('time_partitions_s values must be finite increasing ranges')
    else:
        try:
            start_s = float(dataset.manifest['time_window_start_s'])
            end_s = float(dataset.manifest['time_window_end_s'])
        except (KeyError, TypeError, ValueError):
            reference_path = Path(str(dataset.manifest.get('paired_reference_path', ''))).expanduser()
            if not reference_path.is_file():
                raise ValueError('Dataset needs source time bounds or a readable paired reference manifest')
            with reference_path.open('r', encoding='utf-8') as handle:
                prefix = handle.read(1024 * 1024)
            metadata_key = re.search(r'"metadata"\s*:', prefix)
            if metadata_key is None:
                raise ValueError('Paired reference metadata is not present in the bounded file prefix')
            object_start = prefix.find('{', metadata_key.end())
            if object_start < 0:
                raise ValueError('Paired reference metadata object is invalid')
            try:
                metadata, _end = json.JSONDecoder().raw_decode(prefix[object_start:])
                start_s = float(metadata['time_window_start_s'])
                end_s = float(metadata['time_window_end_s'])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError('Paired reference metadata is missing finite time-window bounds') from exc
        result = build_time_partitions(start_s, end_s, interface=interface)
    for previous, following in zip(names, names[1:]):
        gap = result[following][0] - result[previous][1]
        if gap + 1.0e-12 < purge_s:
            raise ValueError(
                f'Time partitions {previous} and {following} need a {purge_s:.6f}s history/future purge'
            )
    return result


def build_coverage_audit_report(
    observations: list[dict[str, Any]],
    *,
    calibration_metadata: Mapping[str, Any],
    predictor_artifact_sha256: str,
    interface_sha256: str,
    interface: LoadPredictorInterface,
    minimum_sample_spacing_s: float | None = None,
) -> dict[str, Any]:
    if predictor_artifact_sha256 != calibration_metadata.get('predictor_artifact_sha256'):
        raise ValueError('Coverage audit predictor SHA does not match the frozen calibration artifact')
    if interface_sha256 != calibration_metadata.get('predictor_interface_sha256'):
        raise ValueError('Coverage audit interface SHA does not match the frozen calibration artifact')
    coverage_target = float(calibration_metadata['coverage_target'])
    confidence_level = float(calibration_metadata['confidence_level'])
    minimum_samples = int(calibration_metadata['minimum_samples_per_group'])
    max_plan_mismatch = float(calibration_metadata['max_plan_mismatch'])
    spacing = minimum_sample_spacing_s
    if spacing is None:
        spacing = (
            interface.history_frames - 1 + max(interface.horizon_frames)
        ) * interface.step_duration_s
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise ValueError('Coverage audit sample spacing must be positive and finite')
    groups: dict[tuple[int, str, float], list[dict[str, Any]]] = defaultdict(list)
    seen_condition_ids: set[int] = set()
    timestamps: list[float] = []
    previous_time_by_group: dict[tuple[int, str, float], float] = {}
    dropped_overlap = 0
    dropped_ineligible = 0
    for observation in sorted(observations, key=lambda row: (
        int(row['condition_id']),
        str(row['optimization_state']),
        float(row['horizon_s']),
        float(row['timestamp_s']),
    )):
        if observation.get('plan_kind') != PLAN_KIND_ZOH_CANDIDATE:
            dropped_ineligible += 1
            continue
        if float(observation.get('plan_mismatch', math.inf)) > max_plan_mismatch:
            dropped_ineligible += 1
            continue
        condition_id = int(observation['condition_id'])
        state = str(observation['optimization_state'])
        horizon_s = float(observation['horizon_s'])
        timestamp_s = float(observation['timestamp_s'])
        group = (condition_id, state, horizon_s)
        previous = previous_time_by_group.get(group)
        if previous is not None and timestamp_s - previous < spacing - 1.0e-9:
            dropped_overlap += 1
            continue
        previous_time_by_group[group] = timestamp_s
        groups[group].append(observation)
        seen_condition_ids.add(condition_id)
        timestamps.append(timestamp_s)

    calibration_groups = calibration_metadata.get('groups')
    if not isinstance(calibration_groups, list):
        raise ValueError('Calibration artifact is missing its group table')
    group_reports: list[dict[str, Any]] = []
    all_pass = True
    z = NormalDist().inv_cdf(confidence_level)
    for calibration_group in calibration_groups:
        blade = int(calibration_group['blade'])
        state = str(calibration_group['optimization_state'])
        horizon_s = float(calibration_group['horizon_s'])
        key = (blade, state, horizon_s)
        rows = groups.get(key, [])
        quantile = float(calibration_group['conformal_quantile'])
        residuals = np.asarray([
            max(float(row['actual_ratio']) - float(row['predicted_ratio']), 0.0)
            for row in rows
        ], dtype=np.float64)
        successes = int(np.count_nonzero(residuals <= quantile))
        total = int(residuals.size)
        if total:
            proportion = successes / total
            denominator = 1.0 + z * z / total
            center = proportion + z * z / (2.0 * total)
            spread = z * math.sqrt(
                proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
            )
            lower_bound = max(0.0, (center - spread) / denominator)
        else:
            proportion = 0.0
            lower_bound = 0.0
        passed = (
            calibration_group.get('plan_kind') == PLAN_KIND_ZOH_CANDIDATE
            and total >= minimum_samples
            and lower_bound >= coverage_target
        )
        group_reports.append({
            'group_id': calibration_group['group_id'],
            'blade': blade,
            'optimization_state': state,
            'horizon_s': horizon_s,
            'plan_kind': calibration_group.get('plan_kind'),
            'sample_count': total,
            'covered_count': successes,
            'coverage': proportion,
            'coverage_lower_confidence_bound': lower_bound,
            'coverage_target': coverage_target,
            'pass': passed,
        })
        if calibration_group.get('plan_kind') == PLAN_KIND_ZOH_CANDIDATE:
            all_pass = all_pass and passed
    return {
        'schema_version': 'load_predictor_independent_coverage_audit_v1',
        'predictor_artifact_sha256': predictor_artifact_sha256,
        'predictor_interface_sha256': interface_sha256,
        'coverage_target': coverage_target,
        'confidence_level': confidence_level,
        'minimum_samples_per_group': minimum_samples,
        'max_plan_mismatch': max_plan_mismatch,
        'minimum_nonoverlap_sample_spacing_s': spacing,
        'condition_ids': sorted(seen_condition_ids),
        'audit_start_s': min(timestamps) if timestamps else None,
        'audit_end_s': max(timestamps) if timestamps else None,
        'input_sample_count': len(observations),
        'eligible_nonoverlap_sample_count': sum(len(rows) for rows in groups.values()),
        'dropped_overlap_sample_count': dropped_overlap,
        'dropped_ineligible_sample_count': dropped_ineligible,
        'all_zoh_groups_pass': bool(all_pass and bool(groups)),
        'groups': group_reports,
    }


def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        'history': torch.from_numpy(np.stack([item['history'] for item in batch])).float(),
        'candidate_plan': torch.from_numpy(np.stack([item['candidate_plan'] for item in batch])).float(),
        'horizon_index': torch.from_numpy(np.asarray([item['horizon_index'] for item in batch], dtype=np.int64)),
        'plan_kind_index': torch.from_numpy(np.asarray([item['plan_kind_index'] for item in batch], dtype=np.int64)),
        'future_actual_torque_pitch': torch.from_numpy(
            np.stack([item['future_actual_torque_pitch'] for item in batch])
        ).float(),
        'future_peak_ratio': torch.from_numpy(
            np.stack([item['future_peak_ratio'] for item in batch])
        ).float(),
        'condition_id': torch.from_numpy(np.asarray([item['condition_id'] for item in batch], dtype=np.int64)),
        'timestamp_s': torch.from_numpy(np.asarray([item['timestamp_s'] for item in batch], dtype=np.float64)),
        'horizon_s': torch.from_numpy(np.asarray([item['horizon_s'] for item in batch], dtype=np.float32)),
        'optimization_state': [str(item['optimization_state']) for item in batch],
    }


def evaluate_predictor_split(
    model: CausalTCNLoadPredictor,
    dataset: LoadPredictorDataset,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    if len(dataset) == 0:
        raise ValueError('Cannot evaluate a load predictor on an empty split')
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=_collate,
    )
    total = 0
    response_abs_error = np.zeros(2, dtype=np.float64)
    response_squared_error = np.zeros(2, dtype=np.float64)
    peak_abs_error = np.zeros(3, dtype=np.float64)
    peak_squared_error = np.zeros(3, dtype=np.float64)
    predicted_peak_above_reference_count = np.zeros(3, dtype=np.int64)
    groups: dict[str, dict[str, Any]] = {}
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            history = batch['history'].to(device)
            prediction = model.predict_candidate(
                history,
                batch['candidate_plan'].to(device),
                batch['horizon_index'].to(device),
                batch['plan_kind_index'].to(device),
            ).detach().cpu().numpy().astype(np.float64)
            actual_response = batch['future_actual_torque_pitch'].numpy().astype(np.float64)
            actual_peaks = batch['future_peak_ratio'].numpy().astype(np.float64)
            if (
                prediction.shape != (len(history), 5)
                or not np.isfinite(prediction).all()
                or not np.isfinite(actual_response).all()
                or not np.isfinite(actual_peaks).all()
            ):
                raise FloatingPointError('Load predictor independent split contains non-finite inputs/outputs')
            response_error = prediction[:, :2] - actual_response
            peak_error = prediction[:, 2:] - actual_peaks
            response_abs_error += np.abs(response_error).sum(axis=0)
            response_squared_error += np.square(response_error).sum(axis=0)
            peak_abs_error += np.abs(peak_error).sum(axis=0)
            peak_squared_error += np.square(peak_error).sum(axis=0)
            predicted_peak_above_reference_count += np.count_nonzero(prediction[:, 2:] > 1.0, axis=0)
            horizon_indices = batch['horizon_index'].numpy().astype(np.int64)
            for row_index, (state, horizon_index) in enumerate(zip(
                batch['optimization_state'],
                horizon_indices,
                strict=True,
            )):
                horizon_s = float(dataset.interface.horizons_s[int(horizon_index)])
                group_id = f'{state}|h{int(round(horizon_s * 1000)):03d}'
                group = groups.setdefault(group_id, {
                    'optimization_state': str(state),
                    'horizon_s': horizon_s,
                    'sample_count': 0,
                    'response_abs_error_sum': [0.0, 0.0],
                    'peak_ratio_abs_error_sum_by_blade': [0.0, 0.0, 0.0],
                    'predicted_peak_ratio_above_one_count_by_blade': [0, 0, 0],
                })
                group['sample_count'] += 1
                group['response_abs_error_sum'] = (
                    np.asarray(group['response_abs_error_sum'], dtype=np.float64)
                    + np.abs(response_error[row_index])
                ).tolist()
                group['peak_ratio_abs_error_sum_by_blade'] = (
                    np.asarray(group['peak_ratio_abs_error_sum_by_blade'], dtype=np.float64)
                    + np.abs(peak_error[row_index])
                ).tolist()
                group['predicted_peak_ratio_above_one_count_by_blade'] = (
                    np.asarray(group['predicted_peak_ratio_above_one_count_by_blade'], dtype=np.int64)
                    + (prediction[row_index, 2:] > 1.0).astype(np.int64)
                ).tolist()
            total += int(len(history))
    if total <= 0:
        raise RuntimeError('Load predictor split evaluation produced no predictions')
    for group in groups.values():
        count = int(group['sample_count'])
        group['response_mae'] = (
            np.asarray(group.pop('response_abs_error_sum'), dtype=np.float64) / count
        ).tolist()
        group['peak_ratio_mae_by_blade'] = (
            np.asarray(group.pop('peak_ratio_abs_error_sum_by_blade'), dtype=np.float64) / count
        ).tolist()
    return {
        'schema_version': 'load_predictor_independent_prediction_audit_v1',
        'sample_count': total,
        'plan_counts': dataset.plan_counts,
        'response_mae': (response_abs_error / total).tolist(),
        'response_rmse': np.sqrt(response_squared_error / total).tolist(),
        'peak_ratio_mae_by_blade': (peak_abs_error / total).tolist(),
        'peak_ratio_rmse_by_blade': np.sqrt(peak_squared_error / total).tolist(),
        'predicted_peak_ratio_above_one_fraction_by_blade': (
            predicted_peak_above_reference_count / total
        ).tolist(),
        'groups': groups,
        'guard_calibration_authorized': False,
        'note': 'Independent PID-distribution predictions; diagnostic error summary only, not RL coverage authorization.',
    }


def train_predictor(
    *,
    dataset_root: Path,
    interface_path: Path,
    output: Path,
    epochs: int = 1,
    batch_size: int = 256,
    learning_rate: float = 1.0e-3,
    device: str = 'cpu',
    seed: int = 42,
    max_samples: int | None = None,
    normalization_max_samples: int = 8_192,
    time_partitions_s: Mapping[str, tuple[float, float]] | None = None,
) -> dict[str, Any]:
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_path = output.with_name(f'{output.stem}.summary.json')
    lock_path = output.with_name(f'{output.stem}.training.lock')
    lock_fd: int | None = None
    lock_owned = False
    staging_output: Path | None = None
    staging_summary: Path | None = None
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        lock_owned = True
        os.write(lock_fd, json.dumps({
            'schema_version': 'load_predictor_training_lock_v1',
            'pid': os.getpid(),
            'dataset_root': str(dataset_root.resolve()),
            'interface_path': str(interface_path.resolve()),
        }, sort_keys=True).encode('utf-8'))
        os.fsync(lock_fd)
        if output.exists() or summary_path.exists():
            raise FileExistsError(f'Refusing to overwrite predictor artifact or summary: {output}')
        staging_output = output.with_name(f'.{output.stem}.{os.getpid()}.staging.pt')
        staging_summary = staging_output.with_name(f'{staging_output.stem}.summary.json')
        result = _train_predictor_impl(
            dataset_root=dataset_root,
            interface_path=interface_path,
            output=staging_output,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
            seed=seed,
            max_samples=max_samples,
            normalization_max_samples=normalization_max_samples,
            time_partitions_s=time_partitions_s,
        )
        result['output'] = str(output)
        staging_summary.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
        os.link(staging_output, output)
        try:
            os.link(staging_summary, summary_path)
        except BaseException:
            output.unlink(missing_ok=True)
            raise
        return result
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
        if lock_owned:
            lock_path.unlink(missing_ok=True)
        if staging_output is not None:
            staging_output.unlink(missing_ok=True)
        if staging_summary is not None:
            staging_summary.unlink(missing_ok=True)


def _train_predictor_impl(
    *,
    dataset_root: Path,
    interface_path: Path,
    output: Path,
    epochs: int = 1,
    batch_size: int = 256,
    learning_rate: float = 1.0e-3,
    device: str = 'cpu',
    seed: int = 42,
    max_samples: int | None = None,
    normalization_max_samples: int = 8_192,
    time_partitions_s: Mapping[str, tuple[float, float]] | None = None,
) -> dict[str, Any]:
    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0.0:
        raise ValueError('epochs, batch_size, and learning_rate must be positive')
    if seed < 0:
        raise ValueError('seed must be non-negative')
    if torch.device(device).type != 'cpu':
        raise ValueError('W5 v3 predictor training is CPU-only until the configured CUDA runtime is validated')
    random.seed(seed)
    np.random.seed(seed)
    # Seed the CPU generator directly. torch.manual_seed() also initializes
    # CUDA generators in some builds; this workstation's CUDA capability query
    # is known to SIGSEGV, so never touch CUDA for offline predictor fitting.
    torch.random.default_generator.manual_seed(seed)
    interface = LoadPredictorInterface.from_path(interface_path)
    if interface.schema_version != 'causal_load_predictor_interface_v3':
        raise ValueError('Predictor training requires the candidate-conditioned interface v3')
    full_dataset = LoadPredictorDataset(dataset_root, interface=interface)
    if full_dataset.manifest.get('schema_version') != 'load_predictor_dataset_v2':
        raise ValueError('Predictor training requires the provenance-complete load_predictor_dataset_v2 schema')
    if full_dataset.manifest.get('predictor_interface_sha256') != hashlib.sha256(
        interface_path.read_bytes()
    ).hexdigest():
        raise ValueError('Predictor dataset manifest interface SHA does not match the training interface file')
    split_conditions = build_condition_partitions(list(full_dataset.manifest.get('source_conditions', ())))
    split_times = _resolve_time_partitions(full_dataset, interface, time_partitions_s)
    train_dataset = LoadPredictorDataset(
        dataset_root,
        interface=interface,
        condition_ids=split_conditions['train'],
        calibration_eligible_only=True,
        time_range_s=split_times['train'],
    )
    validation_dataset = LoadPredictorDataset(
        dataset_root,
        interface=interface,
        condition_ids=split_conditions['validation'],
        calibration_eligible_only=True,
        time_range_s=split_times['validation'],
    )
    calibration_dataset = LoadPredictorDataset(
        dataset_root,
        interface=interface,
        condition_ids=split_conditions['calibration'],
        plan_kind='zoh_candidate',
        calibration_eligible_only=True,
        time_range_s=split_times['calibration'],
    )
    coverage_audit_dataset = LoadPredictorDataset(
        dataset_root,
        interface=interface,
        condition_ids=split_conditions['coverage_audit'],
        plan_kind='zoh_candidate',
        calibration_eligible_only=True,
        time_range_s=split_times['coverage_audit'],
    )
    normalization_estimate = estimate_normalization(
        train_dataset,
        max_samples=normalization_max_samples,
        seed=seed,
    )
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError('max_samples must be positive when configured')
        train_dataset._indices = train_dataset._indices[:max_samples]
    if any(len(dataset) == 0 for dataset in (
        train_dataset,
        validation_dataset,
        calibration_dataset,
        coverage_audit_dataset,
    )):
        raise ValueError('Purged load predictor train/validation/calibration/audit split has an empty plan-eligible partition')
    normalization = normalization_estimate
    runtime_device = torch.device(device)
    model = CausalTCNLoadPredictor(train_dataset.feature_dim, interface).to(runtime_device)
    model.set_normalization(normalization)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    train_batch_sampler = _ShardBatchSampler(train_dataset, batch_size, seed=seed)
    loader = DataLoader(train_dataset, batch_sampler=train_batch_sampler, collate_fn=_collate)
    loss_total = 0.0
    update_count = 0
    epoch_history: list[dict[str, float]] = []
    best_validation_score = math.inf
    best_epoch = 0
    best_model_state: dict[str, torch.Tensor] | None = None
    best_optimizer_state: dict[str, Any] | None = None
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=_collate,
    )

    def normalized_targets(batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        response_median = torch.as_tensor(normalization['response_median'], dtype=torch.float32, device=runtime_device)
        response_iqr = torch.as_tensor(normalization['response_iqr'], dtype=torch.float32, device=runtime_device)
        ratio_median = torch.as_tensor(normalization['peak_ratio_median'], dtype=torch.float32, device=runtime_device)
        ratio_iqr = torch.as_tensor(normalization['peak_ratio_iqr'], dtype=torch.float32, device=runtime_device)
        response = (batch['future_actual_torque_pitch'].to(runtime_device) - response_median) / response_iqr
        ratios = (batch['future_peak_ratio'].to(runtime_device) - ratio_median) / ratio_iqr
        return response, ratios

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss_total = 0.0
        epoch_update_count = 0
        for batch in loader:
            history = batch['history'].to(runtime_device)
            horizon_index = batch['horizon_index'].to(runtime_device)
            candidate_plan = batch['candidate_plan'].to(runtime_device)
            plan_kind_index = batch['plan_kind_index'].to(runtime_device)
            response_target, peak_target = normalized_targets(batch)
            prediction = model(history, candidate_plan, horizon_index, plan_kind_index)
            response_prediction = prediction[:, :2]
            peak_prediction = prediction[:, 2:]
            loss = torch.nn.functional.smooth_l1_loss(response_prediction, response_target)
            loss = loss + torch.nn.functional.smooth_l1_loss(peak_prediction, peak_target)
            if not torch.isfinite(loss):
                raise FloatingPointError('Load predictor training loss became non-finite')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError('Load predictor gradient norm became non-finite')
            optimizer.step()
            loss_total += float(loss.detach().cpu())
            epoch_loss_total += float(loss.detach().cpu())
            update_count += 1
            epoch_update_count += 1
        model.eval()
        epoch_response_validation_sum = 0.0
        epoch_peak_validation_sum = 0.0
        epoch_validation_count = 0
        with torch.inference_mode():
            for batch in validation_loader:
                history = batch['history'].to(runtime_device)
                prediction = model(
                    history,
                    batch['candidate_plan'].to(runtime_device),
                    batch['horizon_index'].to(runtime_device),
                    batch['plan_kind_index'].to(runtime_device),
                )
                response_target, peak_target = normalized_targets(batch)
                epoch_response_validation_sum += float(torch.nn.functional.l1_loss(
                    prediction[:, :2], response_target, reduction='sum'
                ).cpu())
                epoch_peak_validation_sum += float(torch.nn.functional.l1_loss(
                    prediction[:, 2:], peak_target, reduction='sum'
                ).cpu())
                epoch_validation_count += int(history.shape[0])
        response_validation_mae = epoch_response_validation_sum / max(epoch_validation_count * 2, 1)
        peak_validation_mae = epoch_peak_validation_sum / max(epoch_validation_count * 3, 1)
        validation_score = response_validation_mae + peak_validation_mae
        epoch_metrics = {
            'epoch': float(epoch),
            'train_mean_loss': epoch_loss_total / max(epoch_update_count, 1),
            'updates': float(epoch_update_count),
            'validation_sample_count': float(epoch_validation_count),
            'validation_response_normalized_mae': response_validation_mae,
            'validation_peak_ratio_normalized_mae': peak_validation_mae,
            'validation_selection_score': validation_score,
        }
        epoch_history.append(epoch_metrics)
        if not all(math.isfinite(value) for value in epoch_metrics.values()):
            raise FloatingPointError('Load predictor validation produced non-finite metrics')
        if validation_score < best_validation_score:
            best_validation_score = validation_score
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            best_optimizer_state = copy.deepcopy(optimizer.state_dict())
    if best_model_state is None or best_optimizer_state is None:
        raise RuntimeError('Load predictor training did not produce a valid validation checkpoint')
    model.load_state_dict(best_model_state, strict=True)
    response_validation_mae = epoch_history[best_epoch - 1]['validation_response_normalized_mae']
    peak_validation_mae = epoch_history[best_epoch - 1]['validation_peak_ratio_normalized_mae']
    validation_count = int(epoch_history[best_epoch - 1]['validation_sample_count'])
    calibration_prediction_audit = evaluate_predictor_split(
        model,
        calibration_dataset,
        batch_size=batch_size,
        device=runtime_device,
    )
    coverage_audit_prediction_audit = evaluate_predictor_split(
        model,
        coverage_audit_dataset,
        batch_size=batch_size,
        device=runtime_device,
    )
    if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise FloatingPointError('Load predictor training produced non-finite model parameters')
    output.parent.mkdir(parents=True, exist_ok=True)
    plan_counts = train_dataset.plan_counts
    checkpoint = {
        'schema_version': 'load_predictor_checkpoint_v4_candidate_conditioned',
        'interface_schema_version': interface.schema_version,
        'interface_path': str(interface_path),
        'dataset_root': str(dataset_root),
        'dataset_manifest_sha256': train_dataset.dataset_manifest_sha256,
        'paired_reference_path': full_dataset.manifest['paired_reference_path'],
        'paired_reference_sha256': full_dataset.manifest['paired_reference_sha256'],
        'state_action_artifact_path': full_dataset.manifest['state_action_artifact_path'],
        'state_action_artifact_sha256': full_dataset.manifest['state_action_artifact_sha256'],
        'raw_source_sha256_by_condition': full_dataset.manifest['raw_source_sha256_by_condition'],
        'paired_identity_by_condition': full_dataset.manifest['paired_identity_by_condition'],
        'interface_sha256': hashlib.sha256(interface_path.read_bytes()).hexdigest(),
        'interface_identity': {
            'schema_version': interface.schema_version,
            'history_frames': interface.history_frames,
            'step_duration_s': interface.step_duration_s,
            'horizons_s': list(interface.horizons_s),
            'soft_guard_horizons_s': list(interface.soft_guard_horizons_s),
            'online_candidate_plan_kinds': list(interface.online_candidate_plan_kinds),
            'candidate_plan_encoding': interface.candidate_plan_encoding,
            'candidate_plan_padding': interface.candidate_plan_padding,
            'horizon_conditioning': interface.horizon_conditioning,
            'plan_kind_conditioning': interface.plan_kind_conditioning,
            'model_input_contract': interface.model_input_contract,
            'response_label_semantics': interface.response_label_semantics,
            'peak_label_semantics': interface.peak_label_semantics,
        },
        'split_schema_version': PREDICTOR_SPLIT_SCHEMA_VERSION,
        'condition_partitions': split_conditions,
        'time_partitions_s': {name: list(bounds) for name, bounds in split_times.items()},
        'condition_partitions': split_conditions,
        'temporal_purge_s': (
            interface.history_frames - 1 + max(interface.horizon_frames)
        ) * interface.step_duration_s,
        'split_sample_counts': {
            'train': len(train_dataset),
            'validation': len(validation_dataset),
            'calibration': len(calibration_dataset),
            'coverage_audit': len(coverage_audit_dataset),
        },
        'predictor_training_end_s': split_times['train'][1],
        'calibration_start_s': split_times['calibration'][0],
        'coverage_audit_start_s': split_times['coverage_audit'][0],
        'plan_counts': plan_counts,
        'plan_counts_by_split': {
            'train': train_dataset.plan_counts,
            'validation': validation_dataset.plan_counts,
            'calibration': calibration_dataset.plan_counts,
            'coverage_audit': coverage_audit_dataset.plan_counts,
        },
        'feature_dim': train_dataset.feature_dim,
        'model_architecture': {
            'model_class': 'CausalTCNLoadPredictor',
            'input_dim': train_dataset.feature_dim,
            'hidden_dim': model.hidden_dim,
            'dilations': [1, 2, 4, 8],
            'plan_encoder': 'GRU(input_size=2, hidden_size=64, batch_first=True)',
            'horizon_embedding_dim': 8,
            'plan_kind_embedding_dim': 8,
            'output_width': model.OUTPUT_WIDTH,
        },
        'normalization': normalization,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': best_optimizer_state,
        'epochs': epochs,
        'best_validation_epoch': best_epoch,
        'validation_epoch_history': epoch_history,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'seed': seed,
        'updates': update_count,
        'mean_loss': loss_total / max(update_count, 1),
        'validation_sample_count': validation_count,
        'validation_response_normalized_mae': response_validation_mae,
        'validation_peak_ratio_normalized_mae': peak_validation_mae,
        'calibration_prediction_audit': calibration_prediction_audit,
        'coverage_audit_prediction_audit': coverage_audit_prediction_audit,
        'guard_calibration_authorized': False,
        'guard_calibration_note': 'PID-only held-out coverage does not authorize RL-distribution intervention.',
    }
    torch.save(checkpoint, output)
    checkpoint_digest = hashlib.sha256()
    with output.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            checkpoint_digest.update(chunk)
    summary = {
        'output': str(output),
        'checkpoint_sha256': checkpoint_digest.hexdigest(),
        'schema_version': checkpoint['schema_version'],
        'interface_schema_version': checkpoint['interface_schema_version'],
        'interface_sha256': checkpoint['interface_sha256'],
        'dataset_root': str(dataset_root.expanduser().resolve()),
        'dataset_schema_version': full_dataset.manifest['schema_version'],
        'dataset_manifest_sha256': checkpoint['dataset_manifest_sha256'],
        'feature_dim': train_dataset.feature_dim,
        'updates': update_count,
        'mean_loss': checkpoint['mean_loss'],
        'plan_counts': checkpoint['plan_counts'],
        'plan_counts_by_split': checkpoint['plan_counts_by_split'],
        'condition_partitions': split_conditions,
        'time_partitions_s': {name: list(bounds) for name, bounds in split_times.items()},
        'split_sample_counts': checkpoint['split_sample_counts'],
        'validation_response_normalized_mae': checkpoint['validation_response_normalized_mae'],
        'validation_peak_ratio_normalized_mae': checkpoint['validation_peak_ratio_normalized_mae'],
        'calibration_prediction_audit': calibration_prediction_audit,
        'coverage_audit_prediction_audit': coverage_audit_prediction_audit,
        'best_validation_epoch': best_epoch,
        'validation_epoch_history': epoch_history,
        'guard_calibration_authorized': False,
    }
    output.with_name(f'{output.stem}.summary.json').write_text(
        json.dumps(summary, indent=2) + '\n',
        encoding='utf-8',
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description='Train the causal W5 load predictor offline.')
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--interface', type=Path, default=PROJECT_ROOT / 'RL/data/load_predictor_interface_v3.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=1.0e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--max-samples', type=int, default=0)
    parser.add_argument('--normalization-max-samples', type=int, default=8192)
    args = parser.parse_args()
    summary = train_predictor(
        dataset_root=args.dataset_root,
        interface_path=args.interface,
        output=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        seed=args.seed,
        max_samples=None if args.max_samples <= 0 else args.max_samples,
        normalization_max_samples=args.normalization_max_samples,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
