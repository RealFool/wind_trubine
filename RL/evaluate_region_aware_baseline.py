from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from RL.utils.constants import ACTION_VARIABLES, OBSERVATION_VARIABLES
from RL.utils.observation import build_observation_from_mapping
from RL.utils.region_aware_baseline import RegionAwareJointBaseline


_CONDITION_FILE_PATTERN = re.compile(r'^data_windspeed(\d+)\.csv$')
_DELTA_LIMIT = np.asarray([15_000.0, 0.0003], dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Evaluate a region-aware baseline on independent raw_infer action traces.'
    )
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--step-duration-s', type=float, default=0.02)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(row: dict[str, str], field: str, path: Path, row_number: int) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f'{path.name} row {row_number} has invalid {field}') from exc
    if not math.isfinite(value):
        raise ValueError(f'{path.name} row {row_number} has non-finite {field}')
    return value


def _condition_files(data_root: Path) -> list[tuple[int, Path]]:
    root = data_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f'Independent action holdout directory does not exist: {root}')
    parsed: list[tuple[int, Path]] = []
    unexpected: list[str] = []
    for path in root.glob('*.csv'):
        match = _CONDITION_FILE_PATTERN.fullmatch(path.name)
        if match is None:
            unexpected.append(path.name)
        else:
            parsed.append((int(match.group(1)), path))
    if unexpected:
        raise ValueError(f'Independent action holdout has unexpected CSV files: {sorted(unexpected)}')
    if not parsed:
        raise ValueError('Independent action holdout contains no data_windspeed*.csv files')
    parsed.sort(key=lambda item: item[0])
    condition_ids = [condition_id for condition_id, _ in parsed]
    if len(condition_ids) != len(set(condition_ids)):
        raise ValueError('Independent action holdout contains duplicate condition IDs')
    return parsed


def _metric_summary(predictions: np.ndarray, targets: np.ndarray, previous: np.ndarray) -> dict[str, Any]:
    errors = predictions - targets
    predicted_delta = predictions - previous
    target_delta = targets - previous
    abs_errors = np.abs(errors)
    abs_predicted_delta = np.abs(predicted_delta)
    abs_target_delta = np.abs(target_delta)
    return {
        'sample_count': int(len(predictions)),
        'action_mae': np.mean(abs_errors, axis=0).astype(float).tolist(),
        'action_rmse': np.sqrt(np.mean(np.square(errors), axis=0)).astype(float).tolist(),
        'action_abs_error_p95': np.quantile(abs_errors, 0.95, axis=0).astype(float).tolist(),
        'predicted_action_mean': np.mean(predictions, axis=0).astype(float).tolist(),
        'target_action_mean': np.mean(targets, axis=0).astype(float).tolist(),
        'predicted_delta_abs_p50': np.quantile(abs_predicted_delta, 0.50, axis=0).astype(float).tolist(),
        'predicted_delta_abs_p95': np.quantile(abs_predicted_delta, 0.95, axis=0).astype(float).tolist(),
        'predicted_delta_abs_p99': np.quantile(abs_predicted_delta, 0.99, axis=0).astype(float).tolist(),
        'target_delta_abs_p50': np.quantile(abs_target_delta, 0.50, axis=0).astype(float).tolist(),
        'target_delta_abs_p95': np.quantile(abs_target_delta, 0.95, axis=0).astype(float).tolist(),
        'target_delta_abs_p99': np.quantile(abs_target_delta, 0.99, axis=0).astype(float).tolist(),
        'predicted_delta_limit_fraction': np.mean(
            abs_predicted_delta >= _DELTA_LIMIT * 0.999,
            axis=0,
        ).astype(float).tolist(),
        'target_delta_over_limit_fraction': np.mean(
            abs_target_delta > _DELTA_LIMIT + 1.0e-12,
            axis=0,
        ).astype(float).tolist(),
        'predicted_positive_pitch_fraction': float(np.mean(predictions[:, 1] > 1.0e-8)),
        'target_positive_pitch_fraction': float(np.mean(targets[:, 1] > 1.0e-8)),
    }


def _evaluate_condition(
    baseline_path: Path,
    condition_id: int,
    path: Path,
    *,
    step_duration_s: float,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    baseline = RegionAwareJointBaseline.from_path(
        baseline_path,
        step_duration_s=step_duration_s,
    )
    required = {'Time', *OBSERVATION_VARIABLES, *ACTION_VARIABLES}
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    times: list[float] = []
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f'{path.name} is missing action holdout fields: {missing}')
        for row_number, row in enumerate(reader, start=2):
            timestamp = _finite(row, 'Time', path, row_number)
            if times and timestamp <= times[-1]:
                raise ValueError(f'{path.name} timestamps must be strictly increasing')
            payload = {name: _finite(row, name, path, row_number) for name in OBSERVATION_VARIABLES}
            observations.append(build_observation_from_mapping(payload))
            actions.append(np.asarray([
                _finite(row, ACTION_VARIABLES[0], path, row_number),
                _finite(row, ACTION_VARIABLES[1], path, row_number),
            ], dtype=np.float64))
            times.append(timestamp)
    if len(observations) < 2:
        raise ValueError(f'{path.name} action holdout needs at least two rows')
    action_array = np.stack(actions)
    predictions = np.stack([
        baseline.predict(observation, previous_action_real=action_array[index])
        for index, observation in enumerate(observations[:-1])
    ]).astype(np.float64)
    previous = action_array[:-1]
    targets = action_array[1:]
    summary = _metric_summary(predictions, targets, previous)
    summary.update({
        'condition_id': condition_id,
        'filename': path.name,
        'sha256': _sha256(path),
        'start_time_s': times[0],
        'end_time_s': times[-1],
    })
    return summary, predictions, targets, previous


def evaluate_region_aware_baseline(
    baseline_path: Path,
    data_root: Path,
    *,
    step_duration_s: float,
) -> dict[str, Any]:
    if not math.isfinite(step_duration_s) or step_duration_s <= 0.0:
        raise ValueError('step_duration_s must be positive and finite')
    resolved_baseline = baseline_path.expanduser().resolve()
    files = _condition_files(data_root)
    condition_summaries: list[dict[str, Any]] = []
    all_predictions: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_previous: list[np.ndarray] = []
    for condition_id, path in files:
        summary, predictions, targets, previous = _evaluate_condition(
            resolved_baseline,
            condition_id,
            path,
            step_duration_s=step_duration_s,
        )
        condition_summaries.append(summary)
        all_predictions.append(predictions)
        all_targets.append(targets)
        all_previous.append(previous)
    aggregate = _metric_summary(
        np.concatenate(all_predictions, axis=0),
        np.concatenate(all_targets, axis=0),
        np.concatenate(all_previous, axis=0),
    )
    return {
        'schema_version': 'region_aware_baseline_holdout_evaluation_v1',
        'validation_scope': 'teacher_forced_one_step_action_holdout',
        'validation_status': 'diagnostic_only',
        'physical_gate_status': 'not_evaluated_requires_closed_loop_bladed',
        'limitations': [
            'Holdout state and previous realized action are supplied from the recorded PID trace.',
            'This report does not measure closed-loop energy, rotor speed, safety projection, floor, or unload.',
        ],
        'baseline_path': str(resolved_baseline),
        'baseline_sha256': _sha256(resolved_baseline),
        'data_root': str(data_root.expanduser().resolve()),
        'step_duration_s': step_duration_s,
        'action_names': list(ACTION_VARIABLES),
        'action_units': ['N m', 'rad'],
        'deployment_action_delta_limit': _DELTA_LIMIT.astype(float).tolist(),
        'condition_ids': [condition_id for condition_id, _ in files],
        'aggregate': aggregate,
        'conditions': condition_summaries,
    }


def main() -> None:
    args = parse_args()
    document = evaluate_region_aware_baseline(
        args.baseline,
        args.data_root,
        step_duration_s=float(args.step_duration_s),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({
        'output': str(args.output),
        'condition_ids': document['condition_ids'],
        'sample_count': document['aggregate']['sample_count'],
        'action_mae': document['aggregate']['action_mae'],
        'predicted_delta_abs_p95': document['aggregate']['predicted_delta_abs_p95'],
        'target_delta_abs_p95': document['aggregate']['target_delta_abs_p95'],
        'physical_gate_status': document['physical_gate_status'],
    }, indent=2))


if __name__ == '__main__':
    main()