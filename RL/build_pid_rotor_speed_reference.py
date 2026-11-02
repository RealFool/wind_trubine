from __future__ import annotations

import argparse
from collections import Counter
import csv
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

from RL.utils.raw_dataset_contract import (
    EXPECTED_CONDITION_IDS,
    EXPECTED_END_TIME_S,
    EXPECTED_START_TIME_S,
    TIME_TOLERANCE_S,
    manifest_sha256,
    sha256_file,
    validate_raw_source_root,
    write_json,
)
from RL.utils.constants import DATA_SCHEMA_VERSION, REAL_POWER_VARIABLE
from RL.utils.operating_state import (
    OPERATING_STATE_CONTRACT_VERSION,
    OperatingPoint,
    OperatingState,
    OperatingStateClassifier,
)


SCHEMA_VERSION = 'pid_rotor_speed_reference_schedule_v1'
HOLDOUT_REPORT_SCHEMA_VERSION = 'pid_rotor_speed_reference_holdout_v1'
TIME_FIELD = 'Time'
WIND_FIELD = 'I_AverageWindSpeed'
ROTOR_SPEED_FIELD = 'I_RotorSpeed'
_HOLDOUT_FILE_PATTERN = re.compile(r'^data_windspeed(\d+)\.csv$')
_ELIGIBLE_STATES = {
    OperatingState.NORMAL_GENERATION,
    OperatingState.TRANSITION,
    OperatingState.ABOVE_RATED_REGULATION,
}


def _portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def _read_trace(path: Path) -> tuple[list[dict[str, float | str]], dict[str, int]]:
    required = {TIME_FIELD, WIND_FIELD, ROTOR_SPEED_FIELD, REAL_POWER_VARIABLE}
    classifier = OperatingStateClassifier()
    rows: list[dict[str, float | str]] = []
    state_counts = {state.value: 0 for state in OperatingState}
    previous_time: float | None = None
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f'{path.name} is missing rotor-speed reference fields: {missing}')
        for row_number, row in enumerate(reader, start=2):
            try:
                timestamp = float(row[TIME_FIELD])
                wind_speed = float(row[WIND_FIELD])
                rotor_speed = float(row[ROTOR_SPEED_FIELD])
                real_power = float(row[REAL_POWER_VARIABLE])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f'{path.name} row {row_number} has invalid rotor-speed data') from exc
            if not all(math.isfinite(value) for value in (timestamp, wind_speed, rotor_speed, real_power)):
                raise ValueError(f'{path.name} row {row_number} has non-finite rotor-speed data')
            if previous_time is not None and timestamp <= previous_time:
                raise ValueError(f'{path.name} timestamps must be strictly increasing')
            previous_time = timestamp
            classification = classifier.classify(OperatingPoint(
                timestamp_s=timestamp,
                wind_speed_mps=wind_speed,
                rotor_speed_rad_s=rotor_speed,
                real_power_w=real_power,
            ))
            state_counts[classification.operating_state.value] += 1
            rows.append({
                'timestamp_s': timestamp,
                'wind_speed_mps': wind_speed,
                'rotor_speed_rad_s': rotor_speed,
                'operating_state': classification.operating_state.value,
                'control_region': classification.control_region.value,
            })
    if len(rows) < 2:
        raise ValueError(f'{path.name} needs at least two rotor-speed rows')
    if not (
        math.isclose(float(rows[0]['timestamp_s']), EXPECTED_START_TIME_S, rel_tol=0.0, abs_tol=TIME_TOLERANCE_S)
        and math.isclose(float(rows[-1]['timestamp_s']), EXPECTED_END_TIME_S, rel_tol=0.0, abs_tol=TIME_TOLERANCE_S)
    ):
        raise ValueError(f'{path.name} rotor-speed reference does not cover 0-600 s')
    return rows, state_counts


def build_pid_rotor_speed_reference(
    data_root: Path,
    *,
    minimum_deadband_rad_s: float = 0.01,
    deadband_quantile: float = 0.95,
    min_eligible_rows: int = 100,
    max_adjacent_target_jump_rad_s: float = 0.20,
    max_adjacent_deadband_jump_rad_s: float = 0.10,
) -> dict[str, Any]:
    if not math.isfinite(minimum_deadband_rad_s) or minimum_deadband_rad_s <= 0.0:
        raise ValueError('minimum_deadband_rad_s must be positive and finite')
    if not 0.5 <= deadband_quantile < 1.0:
        raise ValueError('deadband_quantile must be in [0.5, 1.0)')
    if min_eligible_rows <= 0:
        raise ValueError('min_eligible_rows must be positive')
    resolved_root, source_files = validate_raw_source_root(data_root)
    schedule: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for condition_id, path in source_files:
        rows, state_counts = _read_trace(path)
        eligible = np.asarray([
            float(row['rotor_speed_rad_s'])
            for row in rows
            if OperatingState(str(row['operating_state'])) in _ELIGIBLE_STATES
        ], dtype=np.float64)
        if eligible.size < min_eligible_rows:
            raise ValueError(
                f'{path.name} has {eligible.size} eligible generating rows; '
                f'at least {min_eligible_rows} are required'
            )
        target = float(np.median(eligible))
        absolute_deviation = np.abs(eligible - target)
        observed_deadband = float(np.quantile(absolute_deviation, deadband_quantile))
        deadband = max(minimum_deadband_rad_s, observed_deadband)
        eligible_region_counts = Counter(
            str(row['control_region'])
            for row in rows
            if OperatingState(str(row['operating_state'])) in _ELIGIBLE_STATES
        )
        dominant_region = eligible_region_counts.most_common(1)[0][0]
        schedule.append({
            'condition_id': condition_id,
            'wind_speed': float(condition_id),
            'target_rotor_speed': target,
            'deadband': deadband,
            'eligible_rows': int(eligible.size),
            'dominant_control_region': dominant_region,
            'rotor_speed_p05': float(np.quantile(eligible, 0.05)),
            'rotor_speed_p95': float(np.quantile(eligible, 0.95)),
        })
        sources.append({
            'condition_id': condition_id,
            'path': _portable_path(path),
            'filename': path.name,
            'sha256': sha256_file(path),
            'rows': len(rows),
            'eligible_rows': int(eligible.size),
            'operating_state_counts': state_counts,
        })
    targets = np.asarray([row['target_rotor_speed'] for row in schedule], dtype=np.float64)
    deadbands = np.asarray([row['deadband'] for row in schedule], dtype=np.float64)
    target_jumps = np.abs(np.diff(targets))
    deadband_jumps = np.abs(np.diff(deadbands))
    continuity = {
        'interpolation': 'piecewise_linear_continuous',
        'max_adjacent_target_jump_rad_s': float(target_jumps.max()),
        'max_adjacent_deadband_jump_rad_s': float(deadband_jumps.max()),
        'target_jump_limit_rad_s': max_adjacent_target_jump_rad_s,
        'deadband_jump_limit_rad_s': max_adjacent_deadband_jump_rad_s,
        'target_jump_passed': bool(np.all(target_jumps <= max_adjacent_target_jump_rad_s)),
        'deadband_jump_passed': bool(np.all(deadband_jumps <= max_adjacent_deadband_jump_rad_s)),
    }
    if not continuity['target_jump_passed'] or not continuity['deadband_jump_passed']:
        raise ValueError(f'PID rotor-speed schedule continuity check failed: {continuity}')
    return {
        'schema_version': SCHEMA_VERSION,
        'metadata': {
            'data_schema_version': DATA_SCHEMA_VERSION,
            'source_role': 'raw_pid_calibration',
            'source_data_root': str(resolved_root),
            'source_file_pattern': 'data_windspeed{condition_id}.csv',
            'input_manifest_sha256': manifest_sha256(sources),
            'input_manifest_format': 'condition_id:filename:file_sha256\\n sorted by condition_id',
            'condition_ids': list(EXPECTED_CONDITION_IDS),
            'condition_count': len(schedule),
            'rows_total': sum(int(source['rows']) for source in sources),
            'eligible_rows_total': sum(int(source['eligible_rows']) for source in sources),
            'operating_state_contract_version': OPERATING_STATE_CONTRACT_VERSION,
            'eligible_operating_states': sorted(state.value for state in _ELIGIBLE_STATES),
            'target_statistic': 'median',
            'deadband_statistic': f'quantile_{deadband_quantile:.6f}_absolute_deviation_from_median',
            'minimum_deadband_rad_s': minimum_deadband_rad_s,
            'validation_data_policy': 'raw_infer_and_new_seed_are_excluded_from_calibration',
        },
        'continuity': continuity,
        'schedule': schedule,
        'sources': sources,
    }


def evaluate_holdout(schedule_path: Path, holdout_root: Path) -> dict[str, Any]:
    document = json.loads(schedule_path.read_text(encoding='utf-8'))
    if document.get('schema_version') != SCHEMA_VERSION:
        raise ValueError(f'Rotor-speed schedule schema_version must be {SCHEMA_VERSION}')
    root = holdout_root.resolve()
    if root.name != 'raw_infer' or not root.is_dir():
        raise ValueError("--holdout-root must be an existing directory named 'raw_infer'")
    files: list[tuple[int, Path]] = []
    for path in sorted(root.glob('data_windspeed*.csv')):
        match = _HOLDOUT_FILE_PATTERN.fullmatch(path.name)
        if match:
            files.append((int(match.group(1)), path))
    if not files:
        raise ValueError('raw_infer contains no data_windspeed*.csv files')
    calibration_hashes = {str(source['sha256']) for source in document['sources']}
    holdout_hashes = {sha256_file(path) for _condition_id, path in files}
    overlap = sorted(calibration_hashes & holdout_hashes)
    if overlap:
        raise ValueError('Holdout files overlap calibration files by SHA-256')
    wind = np.asarray([row['wind_speed'] for row in document['schedule']], dtype=np.float64)
    targets = np.asarray([row['target_rotor_speed'] for row in document['schedule']], dtype=np.float64)
    deadbands = np.asarray([row['deadband'] for row in document['schedule']], dtype=np.float64)
    reports: list[dict[str, Any]] = []
    for condition_id, path in files:
        rows, _state_counts = _read_trace(path)
        eligible = [row for row in rows if OperatingState(str(row['operating_state'])) in _ELIGIBLE_STATES]
        if not eligible:
            raise ValueError(f'{path.name} has no eligible generating holdout rows')
        normalized_excess: list[float] = []
        within = 0
        for row in eligible:
            query_wind = float(row['wind_speed_mps'])
            target = float(np.interp(query_wind, wind, targets))
            deadband = float(np.interp(query_wind, wind, deadbands))
            error = abs(float(row['rotor_speed_rad_s']) - target)
            within += int(error <= deadband)
            normalized_excess.append(max(error - deadband, 0.0) / deadband)
        reports.append({
            'condition_id': condition_id,
            'path': _portable_path(path),
            'sha256': sha256_file(path),
            'eligible_rows': len(eligible),
            'within_deadband_fraction': within / len(eligible),
            'normalized_band_excess_p95': float(np.quantile(normalized_excess, 0.95)),
        })
    return {
        'schema_version': HOLDOUT_REPORT_SCHEMA_VERSION,
        'validation_scope': 'independent_raw_infer_rotor_speed_band_diagnostic',
        'schedule_path': _portable_path(schedule_path),
            'schedule_sha256': sha256_file(schedule_path),
        'holdout_root': str(root),
        'calibration_holdout_sha256_overlap_count': 0,
        'condition_count': len(reports),
        'conditions': reports,
        'aggregate': {
            'eligible_rows': sum(row['eligible_rows'] for row in reports),
            'minimum_condition_within_deadband_fraction': min(row['within_deadband_fraction'] for row in reports),
            'maximum_condition_normalized_band_excess_p95': max(row['normalized_band_excess_p95'] for row in reports),
        },
        'formal_acceptance': False,
        'limitation': 'Recorded PID holdout is diagnostic and does not replace closed-loop Bladed validation.',
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Build and evaluate the PID rotor-speed reference schedule.')
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--holdout-root', type=Path)
    parser.add_argument('--holdout-output', type=Path)
    args = parser.parse_args()
    document = build_pid_rotor_speed_reference(args.data_root)
    write_json(args.output, document)
    summary: dict[str, Any] = {
        'output': str(args.output),
        'condition_count': document['metadata']['condition_count'],
        'eligible_rows_total': document['metadata']['eligible_rows_total'],
        'input_manifest_sha256': document['metadata']['input_manifest_sha256'],
        'continuity': document['continuity'],
    }
    if args.holdout_root is not None or args.holdout_output is not None:
        if args.holdout_root is None or args.holdout_output is None:
            raise ValueError('--holdout-root and --holdout-output must be provided together')
        report = evaluate_holdout(args.output, args.holdout_root)
        write_json(args.holdout_output, report)
        summary['holdout_output'] = str(args.holdout_output)
        summary['holdout_aggregate'] = report['aggregate']
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()