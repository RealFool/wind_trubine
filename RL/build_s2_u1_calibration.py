from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from RL.utils.raw_dataset_contract import (
    EXPECTED_END_TIME_S,
    EXPECTED_START_TIME_S,
    TIME_TOLERANCE_S,
    manifest_sha256,
    sha256_file,
    validate_raw_source_root,
    write_json,
)
from RL.utils.constants import DATA_SCHEMA_VERSION
from RL.utils.operating_state import (
    OPERATING_STATE_CONTRACT_VERSION,
    OperatingPoint,
    OperatingState,
    OperatingStateClassifier,
)


SCHEMA_VERSION = 's2_u1_pid_envelope_v1'
REPORT_SCHEMA_VERSION = 's2_u1_pid_replay_v1'
REQUIRED_FIELDS = (
    'Time',
    'I_AverageWindSpeed',
    'I_RotorSpeed',
    'I_RealPower',
    'I_PitchPosition1',
    'O_PcsTorqueDemand',
    'Blade root 1 My (Nm)',
    'Blade root 2 My (Nm)',
    'Blade root 3 My (Nm)',
)
ELIGIBLE_STATES = {
    OperatingState.NORMAL_GENERATION,
    OperatingState.TRANSITION,
    OperatingState.ABOVE_RATED_REGULATION,
}


def _read_trace(path: Path) -> dict[str, np.ndarray]:
    values = {field: [] for field in REQUIRED_FIELDS}
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        missing = sorted(set(REQUIRED_FIELDS) - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f'{path.name} is missing S2/U1 fields: {missing}')
        for row_number, row in enumerate(reader, start=2):
            try:
                parsed = {field: float(row[field]) for field in REQUIRED_FIELDS}
            except (TypeError, ValueError) as exc:
                raise ValueError(f'{path.name} row {row_number} has invalid S2/U1 data') from exc
            if not all(math.isfinite(value) for value in parsed.values()):
                raise ValueError(f'{path.name} row {row_number} has non-finite S2/U1 data')
            for field, value in parsed.items():
                values[field].append(value)
    arrays = {field: np.asarray(field_values, dtype=np.float64) for field, field_values in values.items()}
    timestamps = arrays['Time']
    if timestamps.size < 2 or np.any(np.diff(timestamps) <= 0.0):
        raise ValueError(f'{path.name} timestamps must be strictly increasing')
    if not (
        math.isclose(float(timestamps[0]), EXPECTED_START_TIME_S, abs_tol=TIME_TOLERANCE_S)
        and math.isclose(float(timestamps[-1]), EXPECTED_END_TIME_S, abs_tol=TIME_TOLERANCE_S)
    ):
        raise ValueError(f'{path.name} does not cover 0-600 s')
    return arrays


def _eligible_mask(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, int]]:
    classifier = OperatingStateClassifier()
    states: list[OperatingState] = []
    counts = {state.value: 0 for state in OperatingState}
    for timestamp, wind, rotor_speed, power in zip(
        arrays['Time'],
        arrays['I_AverageWindSpeed'],
        arrays['I_RotorSpeed'],
        arrays['I_RealPower'],
        strict=True,
    ):
        state = classifier.classify(OperatingPoint(
            timestamp_s=float(timestamp),
            wind_speed_mps=float(wind),
            rotor_speed_rad_s=float(rotor_speed),
            real_power_w=float(power),
        )).operating_state
        states.append(state)
        counts[state.value] += 1
    return np.asarray([state in ELIGIBLE_STATES for state in states], dtype=bool), counts


def _causal_acceleration(arrays: dict[str, np.ndarray]) -> np.ndarray:
    acceleration = np.zeros_like(arrays['I_RotorSpeed'])
    acceleration[1:] = np.diff(arrays['I_RotorSpeed']) / np.diff(arrays['Time'])
    return acceleration


def build_calibration(data_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved_root, source_files = validate_raw_source_root(data_root)
    schedule: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    replay_conditions: list[dict[str, Any]] = []
    total_eligible = 0
    total_emergency = 0
    for condition_id, path in source_files:
        arrays = _read_trace(path)
        eligible_mask, state_counts = _eligible_mask(arrays)
        acceleration = _causal_acceleration(arrays)
        eligible_mask[0] = False
        eligible_count = int(np.sum(eligible_mask))
        if eligible_count < 100:
            raise ValueError(f'{path.name} needs at least 100 stable generating rows')
        total_eligible += eligible_count
        rotor = arrays['I_RotorSpeed'][eligible_mask]
        power = arrays['I_RealPower'][eligible_mask]
        accel = acceleration[eligible_mask]
        pitch = arrays['I_PitchPosition1'][eligible_mask]
        torque = arrays['O_PcsTorqueDemand'][eligible_mask]
        loads = np.max(np.abs(np.stack([
            arrays['Blade root 1 My (Nm)'][eligible_mask],
            arrays['Blade root 2 My (Nm)'][eligible_mask],
            arrays['Blade root 3 My (Nm)'][eligible_mask],
        ], axis=1)), axis=1)
        row = {
            'condition_id': condition_id,
            'wind_speed': float(condition_id),
            'eligible_rows': eligible_count,
            's2': {
                'rotor_speed_enter_rad_s': float(np.quantile(rotor, 0.995)),
                'rotor_speed_release_rad_s': float(np.quantile(rotor, 0.95)),
                'power_enter_w': float(np.quantile(power, 0.995)),
                'power_release_w': float(np.quantile(power, 0.95)),
                'acceleration_enter_rad_s2': float(np.quantile(accel, 0.995)),
                'acceleration_release_rad_s2': float(np.quantile(accel, 0.75)),
                'pitch_p05_rad': float(np.quantile(pitch, 0.05)),
                'blade_root_abs_p995_nm': float(np.quantile(loads, 0.995)),
            },
            'u1': {
                'warning_speed_rad_s': float(np.quantile(rotor, 0.005)),
                'protection_speed_rad_s': float(np.quantile(rotor, 0.001)),
                'recovery_speed_rad_s': float(np.quantile(rotor, 0.02)),
                'warning_acceleration_rad_s2': float(np.quantile(accel, 0.005)),
                'protection_acceleration_rad_s2': float(np.quantile(accel, 0.001)),
                'pid_torque_p05_nm': float(np.quantile(torque, 0.05)),
            },
        }
        if not (
            row['u1']['protection_speed_rad_s']
            < row['u1']['warning_speed_rad_s']
            < row['u1']['recovery_speed_rad_s']
        ):
            raise ValueError(f'{path.name} produced unordered U1 speed thresholds')
        schedule.append(row)
        sources.append({
            'condition_id': condition_id,
            'path': str(path.resolve().relative_to(PROJECT_ROOT)),
            'filename': path.name,
            'sha256': sha256_file(path),
            'rows': int(arrays['Time'].size),
            'eligible_rows': eligible_count,
            'operating_state_counts': state_counts,
        })

        warning = eligible_mask & (
            (arrays['I_RotorSpeed'] <= row['u1']['warning_speed_rad_s'])
            & (acceleration <= row['u1']['warning_acceleration_rad_s2'])
        )
        protection = eligible_mask & (
            (arrays['I_RotorSpeed'] <= row['u1']['protection_speed_rad_s'])
            & (acceleration <= row['u1']['protection_acceleration_rad_s2'])
        )
        emergency = eligible_mask & (arrays['I_RotorSpeed'] < 0.4)
        s2_enter = eligible_mask & (
            (arrays['I_RotorSpeed'] >= row['s2']['rotor_speed_enter_rad_s'])
            | (arrays['I_RealPower'] >= row['s2']['power_enter_w'])
            | (acceleration >= row['s2']['acceleration_enter_rad_s2'])
        )
        emergency_count = int(np.sum(emergency))
        total_emergency += emergency_count
        replay_conditions.append({
            'condition_id': condition_id,
            'eligible_rows': eligible_count,
            's2_enter_rows': int(np.sum(s2_enter)),
            'u1_warning_rows': int(np.sum(warning & ~protection)),
            'u1_protection_rows': int(np.sum(protection)),
            'u1_emergency_rows': emergency_count,
        })

    manifest_hash = manifest_sha256(sources)
    calibration = {
        'schema_version': SCHEMA_VERSION,
        'metadata': {
            'data_schema_version': DATA_SCHEMA_VERSION,
            'source_role': 'raw_pid_calibration',
            'source_data_root': str(resolved_root),
            'input_manifest_sha256': manifest_hash,
            'condition_ids': [condition_id for condition_id, _ in source_files],
            'condition_count': len(source_files),
            'rows_total': int(sum(source['rows'] for source in sources)),
            'eligible_rows_total': total_eligible,
            'operating_state_contract_version': OPERATING_STATE_CONTRACT_VERSION,
            'causal_acceleration': 'backward_difference_current_and_previous_sample',
            'online_load_signal_available': False,
            'load_role': 'offline_envelope_audit_only',
            'emergency_threshold_rad_s': 0.4,
            'emergency_threshold_source': 'engineering_prior_not_observed_in_eligible_pid_rows',
            'formal_closed_loop_acceptance': False,
        },
        'control_defaults': {
            'pitch_protection_mode': 'dynamic_pid_envelope',
            'low_speed_protection_mode': 'pid_envelope_staged',
            's2_u1_calibration_path': 'RL/data/s2_u1_pid_envelope_3_to_25.json',
            'low_speed_warning_torque_scale': 0.9,
            'low_speed_protection_torque_scale': 0.7,
            'low_rotor_speed_unload_torque': 0.0,
        },
        'schedule': schedule,
        'sources': sources,
    }
    report = {
        'schema_version': REPORT_SCHEMA_VERSION,
        'calibration_schema_version': SCHEMA_VERSION,
        'input_manifest_sha256': manifest_hash,
        'condition_count': len(source_files),
        'eligible_rows_total': total_eligible,
        'conditions': replay_conditions,
        'checks': {
            'all_conditions_present': len(source_files) == 23,
            'all_thresholds_ordered': True,
            'all_metrics_finite': True,
            'emergency_activation_count': total_emergency,
            'emergency_unobserved_in_pid': total_emergency == 0,
        },
        'validation_status': 'passed_offline',
        'formal_closed_loop_acceptance': False,
        'limitations': [
            'observational PID replay cannot estimate counterfactual response to modified pitch or torque',
            'no eligible PID row reaches the engineering-prior emergency threshold',
            'blade-root loads are not available in the current controller observation contract',
        ],
    }
    return calibration, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build hash-locked S2/U1 PID envelopes and replay report.')
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report-output', type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    calibration, report = build_calibration(args.data_root)
    write_json(args.output, calibration)
    report['calibration_sha256'] = sha256_file(args.output)
    write_json(args.report_output, report)
    print(f'Wrote S2/U1 calibration: {args.output}')
    print(f'Wrote S2/U1 replay report: {args.report_output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())