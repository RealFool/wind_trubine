from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from RL.envs.wind_turbine_env import BLADED_CSV_HEADER


POWER_FIELD = 'Electrical power'
TIME_FIELD = 'Time'
WIND_SPEED_FIELD = 'I_AverageWindSpeed'
PID_TORQUE_DEMAND_FIELD = 'O_PcsTorqueDemand'
PID_PITCH_DEMAND_FIELD = 'O_PitchPositionDemand1'
MODEL_SENT_TORQUE_FIELD = 'sent_action_torque'
MODEL_SENT_PITCH_FIELD = 'sent_action_pitch'
BLADE_ROOT_MY_FIELDS = [
    'Blade root 1 My (Nm)',
    'Blade root 2 My (Nm)',
    'Blade root 3 My (Nm)',
]
INFERENCE_REQUIRED_FIELDS = {
    'record_type',
    'inference_episode',
    'inference_global_control_step',
    'info_control_fallback_active',
    'info_control_ood_score',
    MODEL_SENT_TORQUE_FIELD,
    MODEL_SENT_PITCH_FIELD,
    TIME_FIELD,
    WIND_SPEED_FIELD,
    POWER_FIELD,
    *BLADE_ROOT_MY_FIELDS,
}
PID_REQUIRED_FIELDS = {
    TIME_FIELD,
    WIND_SPEED_FIELD,
    POWER_FIELD,
    PID_TORQUE_DEMAND_FIELD,
    PID_PITCH_DEMAND_FIELD,
    *BLADE_ROOT_MY_FIELDS,
}
POWER_GUARDRAIL_PERCENT = 0.0
POWER_IMPROVEMENT_PERCENT = 1.0
LOAD_GUARDRAIL_PERCENT = 0.0
LOAD_IMPROVEMENT_PERCENT = 1.0
FORMAL_TIME_ALIGNMENT_TOLERANCE_S = 1.0e-9

PLOT_FONT_FAMILY = 'DejaVu Sans'
RL_COLOR = '#0000FF'
PID_COLOR = '#FF0000'
WIND_COLOR = '#808080'
WIND_REFERENCE_COLOR = '#FFA500'
TITLE_COLOR = '#000000'
GRID_COLOR = '#B0B0B0'
POSITIVE_DELTA_COLOR = '#2E8B57'
NEGATIVE_DELTA_COLOR = '#AC2E2A'


@dataclass(frozen=True)
class Trace:
    source_rows: list[dict[str, str]]
    raw_time_s: np.ndarray
    elapsed_time_s: np.ndarray
    wind_speed_mps: np.ndarray
    power_w: np.ndarray
    blade_root_my_abs_nm: np.ndarray
    torque_demand_nm: np.ndarray | None = None
    pitch_demand_rad: np.ndarray | None = None

    @property
    def blade_root_my_rms_nm(self) -> np.ndarray:
        return np.sqrt(np.mean(np.square(self.blade_root_my_abs_nm), axis=1))


@dataclass(frozen=True)
class InferenceTrace:
    trace: Trace
    episodes: np.ndarray
    global_steps: np.ndarray
    fallback_active: np.ndarray
    ood_score: np.ndarray
    sent_action_torque_nm: np.ndarray
    sent_action_pitch_rad: np.ndarray
    ignored_trailing_control_rows: int = 0


def _resolve_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if path.is_file() or path.is_dir():
        return path.resolve()
    candidate = PROJECT_ROOT / path
    if candidate.is_file() or candidate.is_dir():
        return candidate.resolve()
    raise FileNotFoundError(f'Path does not exist: {path_text}')


def _resolve_output_dir(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    return path.resolve()


def _finite_float(row: dict[str, str], field: str, row_number: int, source_name: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f'{source_name} row {row_number} has an invalid {field!r} value') from exc
    if not math.isfinite(value):
        raise ValueError(f'{source_name} row {row_number} has a non-finite {field!r} value')
    return value


def _validate_header(fieldnames: list[str] | None, required: set[str], source_name: str) -> None:
    if fieldnames is None:
        raise ValueError(f'{source_name} has no CSV header')
    missing = sorted(required - set(fieldnames))
    if missing:
        raise ValueError(f'{source_name} is missing required columns: {missing}')


def _has_complete_model_action(row: dict[str, str]) -> bool:
    for field in (MODEL_SENT_TORQUE_FIELD, MODEL_SENT_PITCH_FIELD):
        value = (row.get(field) or '').strip()
        if not value:
            return False
        try:
            numeric = float(value)
        except ValueError:
            return False
        if not math.isfinite(numeric):
            return False
    return True


def _nominal_step_s(raw_times: np.ndarray) -> float:
    if len(raw_times) < 2:
        return 0.0
    deltas = np.diff(raw_times)
    positive_deltas = deltas[(deltas > 0.0) & np.isfinite(deltas)]
    if positive_deltas.size == 0:
        return 0.0
    return float(statistics.median(positive_deltas.tolist()))


def _monotonic_elapsed_time(raw_times: np.ndarray) -> np.ndarray:
    if raw_times.size == 0:
        return np.empty(0, dtype=np.float64)
    nominal_step_s = _nominal_step_s(raw_times)
    elapsed = np.zeros(len(raw_times), dtype=np.float64)
    for index in range(1, len(raw_times)):
        delta_s = raw_times[index] - raw_times[index - 1]
        elapsed[index] = elapsed[index - 1] + (delta_s if delta_s > 0.0 else nominal_step_s)
    return elapsed


def _build_trace(rows: list[dict[str, str]], source_name: str) -> Trace:
    raw_time_s = np.asarray(
        [_finite_float(row, TIME_FIELD, index + 2, source_name) for index, row in enumerate(rows)],
        dtype=np.float64,
    )
    power_w = np.asarray(
        [_finite_float(row, POWER_FIELD, index + 2, source_name) for index, row in enumerate(rows)],
        dtype=np.float64,
    )
    wind_speed_mps = np.asarray(
        [_finite_float(row, WIND_SPEED_FIELD, index + 2, source_name) for index, row in enumerate(rows)],
        dtype=np.float64,
    )
    blade_root_my_abs_nm = np.asarray(
        [
            [
                abs(_finite_float(row, my_field, index + 2, source_name))
                for my_field in BLADE_ROOT_MY_FIELDS
            ]
            for index, row in enumerate(rows)
        ],
        dtype=np.float64,
    )
    return Trace(
        source_rows=rows,
        raw_time_s=raw_time_s,
        elapsed_time_s=_monotonic_elapsed_time(raw_time_s),
        wind_speed_mps=wind_speed_mps,
        power_w=power_w,
        blade_root_my_abs_nm=blade_root_my_abs_nm,
    )


def _load_inference_trace(
    path: Path,
    *,
    include_terminal_observation: bool = False,
    expected_sample_count: int | None = None,
) -> InferenceTrace:
    with path.open('r', encoding='utf-8-sig', newline='') as file:
        reader = csv.DictReader(file)
        _validate_header(reader.fieldnames, INFERENCE_REQUIRED_FIELDS, str(path))
        control_rows: list[tuple[int, dict[str, str]]] = []
        terminal_rows: list[tuple[int, dict[str, str]]] = []
        for row_number, row in enumerate(reader, start=2):
            if row.get('record_type') == 'control_step':
                control_rows.append((row_number, row))
            elif row.get('record_type') == 'terminal_observation':
                terminal_rows.append((row_number, row))
    control_row_count = len(control_rows)
    while control_rows and not _has_complete_model_action(control_rows[-1][1]):
        control_rows.pop()
    ignored_trailing_control_rows = control_row_count - len(control_rows)
    rows = [row for _row_number, row in control_rows]
    row_numbers = [row_number for row_number, _row in control_rows]
    if not rows:
        raise ValueError(f'No control_step records found in {path}')
    if (
        include_terminal_observation
        and terminal_rows
        and terminal_rows[-1][0] > row_numbers[-1]
        and (expected_sample_count is None or len(rows) < expected_sample_count)
    ):
        terminal_row_number, raw_terminal_row = terminal_rows[-1]
        terminal_row = dict(raw_terminal_row)
        for field in (
            'inference_episode',
            'inference_global_control_step',
            'info_control_fallback_active',
            'info_control_ood_score',
            MODEL_SENT_TORQUE_FIELD,
            MODEL_SENT_PITCH_FIELD,
        ):
            if not (terminal_row.get(field) or '').strip():
                terminal_row[field] = rows[-1][field]
        rows.append(terminal_row)
        row_numbers.append(terminal_row_number)
    trace = _build_trace(rows, str(path))
    episodes = np.asarray(
        [_finite_float(row, 'inference_episode', row_numbers[index], str(path)) for index, row in enumerate(rows)],
        dtype=np.int64,
    )
    global_steps = np.asarray(
        [_finite_float(row, 'inference_global_control_step', row_numbers[index], str(path)) for index, row in enumerate(rows)],
        dtype=np.int64,
    )
    fallback_active = np.asarray(
        [_finite_float(row, 'info_control_fallback_active', row_numbers[index], str(path)) for index, row in enumerate(rows)],
        dtype=np.float64,
    )
    ood_score = np.asarray(
        [_finite_float(row, 'info_control_ood_score', row_numbers[index], str(path)) for index, row in enumerate(rows)],
        dtype=np.float64,
    )
    sent_action_torque_nm = np.asarray(
        [_finite_float(row, MODEL_SENT_TORQUE_FIELD, row_numbers[index], str(path)) for index, row in enumerate(rows)],
        dtype=np.float64,
    )
    sent_action_pitch_rad = np.asarray(
        [_finite_float(row, MODEL_SENT_PITCH_FIELD, row_numbers[index], str(path)) for index, row in enumerate(rows)],
        dtype=np.float64,
    )
    return InferenceTrace(
        trace=trace,
        episodes=episodes,
        global_steps=global_steps,
        fallback_active=fallback_active,
        ood_score=ood_score,
        sent_action_torque_nm=sent_action_torque_nm,
        sent_action_pitch_rad=sent_action_pitch_rad,
        ignored_trailing_control_rows=ignored_trailing_control_rows,
    )


def _load_pid_trace(path: Path, start_row: int, row_count: int) -> Trace:
    if start_row < 0:
        raise ValueError('pid-start-row must be non-negative')
    selected_rows: list[dict[str, str]] = []
    with path.open('r', encoding='utf-8-sig', newline='') as file:
        reader = csv.DictReader(file)
        _validate_header(reader.fieldnames, PID_REQUIRED_FIELDS, str(path))
        for row_index, row in enumerate(reader):
            if row_index < start_row:
                continue
            if len(selected_rows) >= row_count:
                break
            selected_rows.append(row)
    if len(selected_rows) != row_count:
        raise ValueError(
            f'PID source has only {len(selected_rows)} rows after offset {start_row}; '
            f'{row_count} rows are required to match the inference duration.'
        )
    base_trace = _build_trace(selected_rows, str(path))
    return Trace(
        source_rows=base_trace.source_rows,
        raw_time_s=base_trace.raw_time_s,
        elapsed_time_s=base_trace.elapsed_time_s,
        wind_speed_mps=base_trace.wind_speed_mps,
        power_w=base_trace.power_w,
        blade_root_my_abs_nm=base_trace.blade_root_my_abs_nm,
        torque_demand_nm=np.asarray(
            [_finite_float(row, PID_TORQUE_DEMAND_FIELD, index + 2, str(path)) for index, row in enumerate(selected_rows)],
            dtype=np.float64,
        ),
        pitch_demand_rad=np.asarray(
            [_finite_float(row, PID_PITCH_DEMAND_FIELD, index + 2, str(path)) for index, row in enumerate(selected_rows)],
            dtype=np.float64,
        ),
    )


def _pid_action_demands(pid: Trace) -> tuple[np.ndarray, np.ndarray]:
    if pid.torque_demand_nm is None or pid.pitch_demand_rad is None:
        raise ValueError('PID action-demand fields are unavailable.')
    return pid.torque_demand_nm, pid.pitch_demand_rad


def _cumulative_energy_mwh(power_w: np.ndarray, elapsed_time_s: np.ndarray) -> np.ndarray:
    cumulative = np.zeros(len(power_w), dtype=np.float64)
    for index in range(1, len(power_w)):
        delta_s = max(0.0, float(elapsed_time_s[index] - elapsed_time_s[index - 1]))
        cumulative[index] = cumulative[index - 1] + 0.5 * (power_w[index - 1] + power_w[index]) * delta_s / 3.6e9
    return cumulative


def _strict_total_energy_mwh(power_w: np.ndarray, raw_time_s: np.ndarray) -> float:
    delta_time_s = np.diff(raw_time_s)
    return float(np.sum(0.5 * (power_w[:-1] + power_w[1:]) * delta_time_s) / 3.6e9)


def _formal_acceptance_invalid(reason: str) -> dict[str, Any]:
    return {
        'mode': 'formal_acceptance_v1',
        'valid': False,
        'reason': reason,
        'criteria': {
            'P0_delta_energy_percent_min': POWER_GUARDRAIL_PERCENT,
            'P1_delta_energy_percent_min': POWER_IMPROVEMENT_PERCENT,
            'L0_average_delta_load_percent_min': LOAD_GUARDRAIL_PERCENT,
            'L1_average_delta_load_percent_min': LOAD_IMPROVEMENT_PERCENT,
            'final_rule': '(P1 and L0) or (L1 and P0)',
        },
        'official_acceptance_pass': False,
        'strict_per_blade_pass': False,
        'strict_per_blade_comparison_tolerance_nm': 0.0,
    }


def build_formal_acceptance_report(model: InferenceTrace, pid: Trace) -> dict[str, Any]:
    sample_count = len(model.global_steps)
    if sample_count < 2:
        return _formal_acceptance_invalid('At least two aligned control steps are required for trapezoidal energy.')
    if len(pid.raw_time_s) != sample_count:
        return _formal_acceptance_invalid('RL and PID traces have different aligned control-step counts.')
    if len(np.unique(model.episodes)) != 1:
        return _formal_acceptance_invalid(
            'Formal acceptance requires one continuous RL episode; multi-episode diagnostic traces are not valid.'
        )
    if np.any(np.diff(model.trace.raw_time_s) <= 0.0):
        return _formal_acceptance_invalid('RL simulation timestamps must be strictly increasing.')
    if np.any(np.diff(pid.raw_time_s) <= 0.0):
        return _formal_acceptance_invalid('PID simulation timestamps must be strictly increasing.')
    model_relative_time_s = model.trace.raw_time_s - model.trace.raw_time_s[0]
    pid_relative_time_s = pid.raw_time_s - pid.raw_time_s[0]
    if not np.allclose(
        model_relative_time_s,
        pid_relative_time_s,
        rtol=0.0,
        atol=FORMAL_TIME_ALIGNMENT_TOLERANCE_S,
    ):
        return _formal_acceptance_invalid(
            'RL and PID simulation timestamps are not aligned on the same relative time window.'
        )

    model_energy_mwh = _strict_total_energy_mwh(model.trace.power_w, model.trace.raw_time_s)
    pid_energy_mwh = _strict_total_energy_mwh(pid.power_w, pid.raw_time_s)
    if not math.isfinite(model_energy_mwh) or not math.isfinite(pid_energy_mwh) or pid_energy_mwh <= 0.0:
        return _formal_acceptance_invalid('Energy is invalid because PID trapezoidal energy must be finite and positive.')

    model_peaks_nm = np.max(model.trace.blade_root_my_abs_nm, axis=0)
    pid_peaks_nm = np.max(pid.blade_root_my_abs_nm, axis=0)
    if not np.isfinite(model_peaks_nm).all() or not np.isfinite(pid_peaks_nm).all() or np.any(pid_peaks_nm <= 0.0):
        return _formal_acceptance_invalid('Blade-root peaks are invalid because every finite positive PID channel is required.')

    delta_energy_percent = (model_energy_mwh / pid_energy_mwh - 1.0) * 100.0
    delta_load_percent = (1.0 - model_peaks_nm / pid_peaks_nm) * 100.0
    average_delta_load_percent = float(np.mean(delta_load_percent))
    comparison_tolerance_nm = 0.0
    blade_load_increase_nm = model_peaks_nm - pid_peaks_nm
    strict_per_blade_pass = bool(np.all(blade_load_increase_nm <= comparison_tolerance_nm))
    p0 = bool(delta_energy_percent >= POWER_GUARDRAIL_PERCENT)
    p1 = bool(delta_energy_percent >= POWER_IMPROVEMENT_PERCENT)
    l0 = bool(average_delta_load_percent >= LOAD_GUARDRAIL_PERCENT)
    l1 = bool(average_delta_load_percent >= LOAD_IMPROVEMENT_PERCENT)
    path_1 = bool(p1 and l0)
    path_2 = bool(l1 and p0)
    channels = []
    for blade_index, field in enumerate(BLADE_ROOT_MY_FIELDS):
        channels.append(
            {
                'field': field,
                'rl_peak_abs_nm': float(model_peaks_nm[blade_index]),
                'pid_peak_abs_nm': float(pid_peaks_nm[blade_index]),
                'delta_load_percent': float(delta_load_percent[blade_index]),
            }
        )
    return {
        'mode': 'formal_acceptance_v1',
        'valid': True,
        'reason': '',
        'sample_count': int(sample_count),
        'model_episode': int(model.episodes[0]),
        'time_alignment_tolerance_s': FORMAL_TIME_ALIGNMENT_TOLERANCE_S,
        'model_duration_s': float(model_relative_time_s[-1]),
        'pid_duration_s': float(pid_relative_time_s[-1]),
        'energy': {
            'rl_mwh': model_energy_mwh,
            'pid_mwh': pid_energy_mwh,
            'delta_energy_percent': float(delta_energy_percent),
        },
        'blade_root_my': {
            'channels': channels,
            'average_delta_load_percent': average_delta_load_percent,
            'strict_per_blade': [
                {
                    'blade': blade_index + 1,
                    'rl_peak_abs_nm': float(model_peaks_nm[blade_index]),
                    'pid_peak_abs_nm': float(pid_peaks_nm[blade_index]),
                    'increase_nm': float(blade_load_increase_nm[blade_index]),
                    'comparison_tolerance_nm': comparison_tolerance_nm,
                    'pass': bool(blade_load_increase_nm[blade_index] <= comparison_tolerance_nm),
                }
                for blade_index in range(3)
            ],
        },
        'criteria': {
            'P0_delta_energy_percent_min': POWER_GUARDRAIL_PERCENT,
            'P1_delta_energy_percent_min': POWER_IMPROVEMENT_PERCENT,
            'L0_average_delta_load_percent_min': LOAD_GUARDRAIL_PERCENT,
            'L1_average_delta_load_percent_min': LOAD_IMPROVEMENT_PERCENT,
            'final_rule': '(P1 and L0) or (L1 and P0)',
        },
        'P0': p0,
        'P1': p1,
        'L0': l0,
        'L1': l1,
        'path_1_pass': path_1,
        'path_2_pass': path_2,
        'final_pass': bool(path_1 or path_2),
        'official_acceptance_pass': bool(path_1 or path_2),
        'strict_per_blade_pass': strict_per_blade_pass,
        'strict_per_blade_comparison_tolerance_nm': comparison_tolerance_nm,
    }


def evaluate_streaming_scenario(
    inference_csv: Path,
    pid_csv: Path,
    *,
    pid_start_row: int = 0,
    expected_pid_window_s: tuple[float, float] | None = None,
    expected_trace_sample_count: int | None = None,
) -> dict[str, Any]:
    """Build one bounded-memory scenario report from the existing streaming CSV readers."""
    model = _load_inference_trace(
        inference_csv,
        include_terminal_observation=True,
        expected_sample_count=expected_trace_sample_count,
    )
    pid = _load_pid_trace(pid_csv, start_row=pid_start_row, row_count=len(model.global_steps))
    if expected_pid_window_s is not None:
        expected_start_s, expected_end_s = map(float, expected_pid_window_s)
        if (
            not math.isfinite(expected_start_s)
            or not math.isfinite(expected_end_s)
            or expected_end_s <= expected_start_s
        ):
            return _formal_acceptance_invalid('Paired PID identity has an invalid frozen time window.')
        if (
            abs(float(pid.raw_time_s[0]) - expected_start_s) > FORMAL_TIME_ALIGNMENT_TOLERANCE_S
            or abs(float(pid.raw_time_s[-1]) - expected_end_s) > FORMAL_TIME_ALIGNMENT_TOLERANCE_S
        ):
            return _formal_acceptance_invalid(
                'Selected PID rows do not span the strict identity paired-reference time window.'
            )
    return build_formal_acceptance_report(model, pid)


def build_multi_scenario_acceptance_report(
    scenarios: list[dict[str, Any]],
    *,
    expected_scenario_ids: list[str] | tuple[str, ...],
    comparison_tolerance_nm: float = 0.0,
) -> dict[str, Any]:
    if not math.isfinite(comparison_tolerance_nm) or comparison_tolerance_nm < 0.0:
        raise ValueError('comparison_tolerance_nm must be finite and non-negative')
    expected = tuple(str(value) for value in expected_scenario_ids)
    if not expected or len(expected) != len(set(expected)):
        raise ValueError('expected_scenario_ids must be a non-empty unique sequence')
    by_id: dict[str, dict[str, Any]] = {}
    for scenario in scenarios:
        if not isinstance(scenario, dict):
            raise ValueError('Each multi-scenario acceptance entry must be an object')
        scenario_id = str(scenario.get('scenario_id', '')).strip()
        if not scenario_id or scenario_id in by_id:
            raise ValueError(f'Duplicate or empty scenario_id in acceptance inputs: {scenario_id!r}')
        report = scenario.get('report')
        if not isinstance(report, dict):
            raise ValueError(f'Scenario {scenario_id} is missing its formal report')
        by_id[scenario_id] = report

    results: list[dict[str, Any]] = []
    for scenario_id in expected:
        report = by_id.get(scenario_id)
        if report is None:
            results.append({
                'scenario_id': scenario_id,
                'valid': False,
                'official_acceptance_pass': False,
                'strict_per_blade_pass': False,
                'strict_case_matched': False,
                'reason': 'missing expected scenario',
            })
            continue
        if not report.get('valid', False):
            results.append({
                'scenario_id': scenario_id,
                'valid': False,
                'official_acceptance_pass': False,
                'strict_per_blade_pass': False,
                'strict_case_matched': bool(scenario.get('strict_case_matched', False)),
                'reason': report.get('reason', 'invalid scenario report'),
            })
            continue
        blade_rows = report.get('blade_root_my', {}).get('strict_per_blade', ())
        if len(blade_rows) != 3:
            results.append({
                'scenario_id': scenario_id,
                'valid': False,
                'official_acceptance_pass': False,
                'strict_per_blade_pass': False,
                'strict_case_matched': bool(scenario.get('strict_case_matched', False)),
                'reason': 'strict per-blade evidence is incomplete',
            })
            continue
        strict_pass = all(
            float(row['rl_peak_abs_nm']) - float(row['pid_peak_abs_nm']) <= comparison_tolerance_nm
            for row in blade_rows
        )
        results.append({
            'scenario_id': scenario_id,
            'valid': True,
            'official_acceptance_pass': bool(report.get('official_acceptance_pass', report.get('final_pass', False))),
            'strict_per_blade_pass': strict_pass,
            'strict_case_matched': bool(scenario.get('strict_case_matched', False)),
            'strict_per_blade_comparison_tolerance_nm': comparison_tolerance_nm,
            'delta_energy_percent': report['energy']['delta_energy_percent'],
            'average_delta_load_percent': report['blade_root_my']['average_delta_load_percent'],
            'blade_peak_delta_percent': [row['delta_load_percent'] for row in report['blade_root_my']['channels']],
            'reason': '',
        })
    unexpected = sorted(set(by_id) - set(expected))
    complete = len(results) == len(expected) and all(item['valid'] for item in results) and not unexpected
    all_official = complete and all(item['official_acceptance_pass'] for item in results)
    all_strict = complete and all(item['strict_per_blade_pass'] for item in results)
    all_identity = complete and all(item['strict_case_matched'] for item in results)
    return {
        'schema_version': 'multi_scenario_acceptance_v1',
        'expected_scenario_ids': list(expected),
        'observed_scenario_ids': sorted(by_id),
        'unexpected_scenario_ids': unexpected,
        'scenario_count': len(results),
        'valid_scenario_count': sum(bool(item['valid']) for item in results),
        'all_expected_scenarios_valid': bool(complete),
        'official_acceptance_pass': bool(all_official),
        'strict_per_blade_pass': bool(all_strict),
        'strict_case_identity_pass': bool(all_identity),
        'release_candidate_pass': bool(complete and all_official and all_strict and all_identity),
        'aggregation_semantics': (
            'all_expected_scenarios_must_be_valid_strictly_identified_and_each_must_pass_both_official_and_strict_gates'
        ),
        'comparison_tolerance_nm': comparison_tolerance_nm,
        'scenarios': results,
    }


def _finite_statistics(values: np.ndarray) -> dict[str, float]:
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return {'count': 0.0, 'mean': 0.0, 'min': 0.0, 'median': 0.0, 'p95': 0.0, 'max': 0.0}
    return {
        'count': float(finite_values.size),
        'mean': float(np.mean(finite_values)),
        'min': float(np.min(finite_values)),
        'median': float(np.median(finite_values)),
        'p95': float(np.percentile(finite_values, 95.0)),
        'max': float(np.max(finite_values)),
    }


def _episode_boundaries(trace: InferenceTrace) -> list[int]:
    boundaries: list[int] = []
    for index in range(1, len(trace.episodes)):
        if trace.episodes[index] != trace.episodes[index - 1]:
            boundaries.append(index)
    return boundaries


def _draw_episode_boundaries(axis, step_values: np.ndarray, boundaries: list[int]) -> None:
    for index in boundaries:
        axis.axvline(step_values[index], color='#64748b', linestyle=':', linewidth=0.75, alpha=0.7)


def _draw_fallback_spans(
    axis,
    step_values: np.ndarray,
    fallback_active: np.ndarray,
    min_span_steps: int = 32,
) -> None:
    active_start: int | None = None
    for index, active in enumerate(fallback_active > 0.5):
        if active and active_start is None:
            active_start = index
        if active_start is None or (active and index != len(fallback_active) - 1):
            continue
        end_index = index if not active else index + 1
        if end_index - active_start >= min_span_steps:
            left = step_values[active_start]
            right = step_values[min(end_index, len(step_values) - 1)]
            axis.axvspan(left, right, color=WIND_REFERENCE_COLOR, alpha=0.14)
        active_start = None


def _style_axis(
    axis,
    *,
    title_size: float = 16.0,
    label_size: float = 13.0,
    tick_size: float = 11.0,
    grid_alpha: float = 0.65,
) -> None:
    axis.set_axisbelow(True)
    axis.tick_params(axis='both', labelsize=tick_size)
    axis.grid(
        color=GRID_COLOR,
        linestyle='-',
        linewidth=0.8,
        alpha=grid_alpha,
    )
    axis.title.set_fontfamily(PLOT_FONT_FAMILY)
    axis.title.set_fontsize(title_size)
    axis.title.set_fontweight('bold')
    axis.title.set_color(TITLE_COLOR)
    for label in (axis.xaxis.label, axis.yaxis.label):
        label.set_fontfamily(PLOT_FONT_FAMILY)
        label.set_fontsize(label_size)
    legend = axis.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontfamily(PLOT_FONT_FAMILY)
            text.set_fontsize(tick_size)


def _plot_power(
    output_path: Path,
    model: InferenceTrace,
    pid: Trace,
    episode_boundaries: list[int],
    show_fallback: bool,
) -> None:
    import matplotlib.pyplot as plt

    if show_fallback:
        figure, (power_axis, fallback_axis) = plt.subplots(
            2,
            1,
            figsize=(16, 8),
            sharex=True,
            gridspec_kw={'height_ratios': [4, 1]},
        )
    else:
        figure, power_axis = plt.subplots(figsize=(16, 5))
        fallback_axis = None
    elapsed_time_s = model.trace.elapsed_time_s
    if show_fallback:
        _draw_fallback_spans(power_axis, elapsed_time_s, model.fallback_active)
    power_axis.plot(elapsed_time_s, model.trace.power_w / 1e6, color=RL_COLOR, linewidth=1.0, label='RL deployment')
    power_axis.plot(elapsed_time_s, pid.power_w / 1e6, color=PID_COLOR, linewidth=1.0, label='PID validation')
    _draw_episode_boundaries(power_axis, elapsed_time_s, episode_boundaries)
    power_axis.set_title('Electrical Power Comparison')
    power_axis.set_ylabel('Electrical power (MW)')
    if not show_fallback:
        power_axis.set_xlabel('Time (s)')
    power_axis.legend(loc='best')
    _style_axis(power_axis, title_size=17.0)

    if fallback_axis is not None:
        fallback_axis.step(
            elapsed_time_s,
            model.fallback_active,
            where='post',
            color=WIND_REFERENCE_COLOR,
            linewidth=1.0,
            label='OOD fallback active',
        )
        _draw_episode_boundaries(fallback_axis, elapsed_time_s, episode_boundaries)
        fallback_axis.set_ylim(-0.1, 1.1)
        fallback_axis.set_yticks([0.0, 1.0])
        fallback_axis.set_ylabel('Fallback')
        fallback_axis.set_xlabel('Time (s)')
        fallback_axis.legend(loc='best')
        _style_axis(fallback_axis, title_size=14.0, label_size=12.0, tick_size=10.0)
    figure.subplots_adjust(left=0.08, right=0.98, top=0.94, bottom=0.09, hspace=0.22)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _plot_wind_speed(
    output_path: Path,
    model: InferenceTrace,
    pid: Trace,
    episode_boundaries: list[int],
    show_fallback: bool,
) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(16, 5))
    elapsed_time_s = model.trace.elapsed_time_s
    if show_fallback:
        _draw_fallback_spans(axis, elapsed_time_s, model.fallback_active)
    axis.plot(
        elapsed_time_s,
        model.trace.wind_speed_mps,
        color=WIND_COLOR,
        linestyle='-',
        linewidth=1.0,
        label='RL wind speed',
    )
    axis.plot(
        elapsed_time_s,
        pid.wind_speed_mps,
        color=WIND_REFERENCE_COLOR,
        linestyle='--',
        linewidth=1.0,
        label='PID wind speed',
    )
    _draw_episode_boundaries(axis, elapsed_time_s, episode_boundaries)
    axis.set_title('Wind Speed Comparison')
    axis.set_ylabel('Wind speed (m/s)')
    axis.set_xlabel('Time (s)')
    axis.legend(loc='best')
    _style_axis(axis, title_size=17.0)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _format_average_power_annotation(model_mean_power_w: float, pid_mean_power_w: float) -> tuple[str, str]:
    delta_power_w = model_mean_power_w - pid_mean_power_w
    if delta_power_w > 0.0:
        direction = 'Increase'
        color = '#047857'
    elif delta_power_w < 0.0:
        direction = 'Decrease'
        color = '#b91c1c'
    else:
        direction = 'No change'
        color = '#475569'

    if pid_mean_power_w == 0.0:
        relative_delta = 'n/a'
    else:
        relative_delta = f'{abs(delta_power_w / pid_mean_power_w) * 100.0:.2f}%'
    annotation = (
        'Average electrical power vs PID\n'
        f'RL: {model_mean_power_w / 1e6:.3f} MW | PID: {pid_mean_power_w / 1e6:.3f} MW\n'
        f'{direction}: {abs(delta_power_w) / 1e3:.1f} kW ({relative_delta})'
    )
    return annotation, color


def _plot_cumulative_energy(
    output_path: Path,
    model: InferenceTrace,
    pid: Trace,
    model_energy_mwh: np.ndarray,
    pid_energy_mwh: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    cumulative_delta_mwh = model_energy_mwh - pid_energy_mwh
    cumulative_delta_kwh = cumulative_delta_mwh * 1.0e3
    difference_limit_kwh = max(0.1, float(np.max(np.abs(cumulative_delta_kwh))) * 1.15)
    figure, (energy_axis, difference_axis) = plt.subplots(
        2,
        1,
        figsize=(16, 8),
        sharex=True,
        gridspec_kw={'height_ratios': [3, 2]},
    )
    elapsed_time_s = model.trace.elapsed_time_s
    energy_axis.plot(elapsed_time_s, model_energy_mwh, color=RL_COLOR, linewidth=2.2, label='RL deployment')
    energy_axis.plot(elapsed_time_s, pid_energy_mwh, color=PID_COLOR, linewidth=2.2, label='PID validation')
    energy_axis.fill_between(
        elapsed_time_s,
        model_energy_mwh,
        pid_energy_mwh,
        where=cumulative_delta_mwh >= 0.0,
        color=POSITIVE_DELTA_COLOR,
        alpha=0.10,
        interpolate=True,
    )
    energy_axis.fill_between(
        elapsed_time_s,
        model_energy_mwh,
        pid_energy_mwh,
        where=cumulative_delta_mwh < 0.0,
        color=NEGATIVE_DELTA_COLOR,
        alpha=0.10,
        interpolate=True,
    )
    energy_axis.set_title('Cumulative Electrical Energy Over Matched Inference Duration')
    energy_axis.set_ylabel('Cumulative energy (MWh)')
    energy_axis.legend(loc='best')
    final_delta_percent = (
        float(cumulative_delta_mwh[-1] / pid_energy_mwh[-1] * 100.0)
        if abs(float(pid_energy_mwh[-1])) > 1.0e-12
        else 0.0
    )
    energy_axis.text(
        0.98,
        0.04,
        f'Final RL-PID: {cumulative_delta_kwh[-1]:+.3f} kWh ({final_delta_percent:+.4f}%)',
        transform=energy_axis.transAxes,
        ha='right',
        va='bottom',
        color=POSITIVE_DELTA_COLOR if cumulative_delta_mwh[-1] >= 0.0 else NEGATIVE_DELTA_COLOR,
        fontsize=10,
        bbox={
            'boxstyle': 'round,pad=0.35',
            'facecolor': '#ffffff',
            'edgecolor': '#64748b',
            'alpha': 0.92,
        },
    )

    # Keep the main axis physically honest in MWh, then compare the two
    # cumulative-growth lines over a short terminal window on a kWh scale.
    tail_duration_s = min(5.0, max(float(elapsed_time_s[-1] - elapsed_time_s[0]) * 0.10, 1.0e-6))
    tail_start_s = max(float(elapsed_time_s[0]), float(elapsed_time_s[-1]) - tail_duration_s)
    tail_indices = np.flatnonzero(elapsed_time_s >= tail_start_s)
    if tail_indices.size < 2:
        tail_indices = np.arange(max(0, len(elapsed_time_s) - 2), len(elapsed_time_s))
    tail_time_s = elapsed_time_s[tail_indices]
    tail_model_kwh = model_energy_mwh[tail_indices] * 1.0e3
    tail_pid_kwh = pid_energy_mwh[tail_indices] * 1.0e3
    tail_delta_kwh = tail_model_kwh - tail_pid_kwh
    tail_reference_kwh = float(tail_pid_kwh[0])
    tail_model_growth_kwh = tail_model_kwh - tail_reference_kwh
    tail_pid_growth_kwh = tail_pid_kwh - tail_reference_kwh
    tail_min_kwh = float(min(np.min(tail_model_growth_kwh), np.min(tail_pid_growth_kwh)))
    tail_max_kwh = float(max(np.max(tail_model_growth_kwh), np.max(tail_pid_growth_kwh)))
    tail_span_kwh = max(tail_max_kwh - tail_min_kwh, 1.0e-6)
    tail_padding_kwh = max(tail_span_kwh * 0.12, 0.01)
    energy_axis.axvspan(
        float(tail_time_s[0]),
        float(tail_time_s[-1]),
        color='#94a3b8',
        alpha=0.10,
        label='Tail zoom window',
    )
    zoom_axis = inset_axes(energy_axis, width='48%', height='52%', loc='upper right', borderpad=1.1)
    zoom_axis.plot(
        tail_time_s,
        tail_model_growth_kwh,
        color='#2563eb',
        linewidth=1.35,
        label='RL cumulative growth',
    )
    zoom_axis.plot(
        tail_time_s,
        tail_pid_growth_kwh,
        color='#dc2626',
        linewidth=1.35,
        label='PID cumulative growth',
    )
    zoom_axis.fill_between(
        tail_time_s,
        tail_model_growth_kwh,
        tail_pid_growth_kwh,
        where=tail_delta_kwh >= 0.0,
        color=POSITIVE_DELTA_COLOR,
        alpha=0.14,
        interpolate=True,
    )
    zoom_axis.fill_between(
        tail_time_s,
        tail_model_growth_kwh,
        tail_pid_growth_kwh,
        where=tail_delta_kwh < 0.0,
        color=NEGATIVE_DELTA_COLOR,
        alpha=0.14,
        interpolate=True,
    )
    zoom_axis.axhline(0.0, color='#475569', linewidth=0.8)
    zoom_axis.scatter(
        [tail_time_s[-1], tail_time_s[-1]],
        [tail_model_growth_kwh[-1], tail_pid_growth_kwh[-1]],
        color=['#2563eb', '#dc2626'],
        s=16,
        zorder=4,
    )
    zoom_axis.annotate(
        f'RL={tail_model_growth_kwh[-1]:.2f} kWh\nPID={tail_pid_growth_kwh[-1]:.2f} kWh\nΔ={tail_delta_kwh[-1]:+.2f} kWh',
        xy=(tail_time_s[-1], max(tail_model_growth_kwh[-1], tail_pid_growth_kwh[-1])),
        xytext=(-8, 10),
        textcoords='offset points',
        ha='right',
        va='bottom',
        fontsize=8,
        color='#334155',
        bbox={'boxstyle': 'round,pad=0.2', 'facecolor': '#ffffff', 'edgecolor': '#94a3b8', 'alpha': 0.9},
    )
    zoom_axis.set_xlim(float(tail_time_s[0]), float(tail_time_s[-1]))
    zoom_axis.set_ylim(tail_min_kwh - tail_padding_kwh, tail_max_kwh + tail_padding_kwh)
    zoom_axis.set_title(f'Tail cumulative growth: last {tail_time_s[-1] - tail_time_s[0]:.1f} s', fontsize=9)
    zoom_axis.set_ylabel('Growth from tail start (kWh)', fontsize=8)
    zoom_axis.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    zoom_axis.tick_params(labelsize=7)
    zoom_axis.legend(loc='upper left', fontsize=7, framealpha=0.85)
    _style_axis(energy_axis, title_size=17.0)
    _style_axis(zoom_axis, title_size=10.0, label_size=8.0, tick_size=7.0)

    power_annotation, annotation_color = _format_average_power_annotation(
        float(np.mean(model.trace.power_w)),
        float(np.mean(pid.power_w)),
    )
    energy_axis.text(
        0.98,
        0.17,
        power_annotation,
        transform=energy_axis.transAxes,
        ha='right',
        va='bottom',
        color=annotation_color,
        fontsize=10,
        bbox={
            'boxstyle': 'round,pad=0.35',
            'facecolor': '#ffffff',
            'edgecolor': annotation_color,
            'alpha': 0.92,
        },
    )

    difference_axis.plot(
        model.trace.elapsed_time_s,
        cumulative_delta_kwh,
        color=POSITIVE_DELTA_COLOR,
        linewidth=2.0,
        label='RL deployment - PID',
    )
    difference_axis.fill_between(
        model.trace.elapsed_time_s,
        cumulative_delta_kwh,
        0.0,
        where=cumulative_delta_kwh >= 0.0,
        color=POSITIVE_DELTA_COLOR,
        alpha=0.12,
        interpolate=True,
    )
    difference_axis.fill_between(
        model.trace.elapsed_time_s,
        cumulative_delta_kwh,
        0.0,
        where=cumulative_delta_kwh < 0.0,
        color=NEGATIVE_DELTA_COLOR,
        alpha=0.12,
        interpolate=True,
    )
    difference_axis.axhline(0.0, color='#475569', linewidth=1.0)
    difference_axis.set_ylim(-difference_limit_kwh, difference_limit_kwh)
    difference_axis.set_ylabel('RL-PID (kWh)')
    difference_axis.legend(loc='best')
    difference_axis.set_xlabel('Time (s)')
    _style_axis(difference_axis, title_size=16.0)
    figure.subplots_adjust(left=0.08, right=0.98, top=0.94, bottom=0.09, hspace=0.22)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _plot_blade_root_load(
    output_path: Path,
    model: InferenceTrace,
    pid: Trace,
    episode_boundaries: list[int],
    show_fallback: bool,
) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    elapsed_time_s = model.trace.elapsed_time_s
    load_summary = _blade_root_load_summary(model, pid)
    model_peaks_nm = np.asarray(load_summary['rl_peak_nm'], dtype=np.float64)
    pid_peaks_nm = np.asarray(load_summary['pid_peak_nm'], dtype=np.float64)
    peak_delta_nm = np.asarray(load_summary['peak_delta_nm'], dtype=np.float64)
    peak_delta_percent = np.asarray(load_summary['peak_delta_percent'], dtype=np.float64)

    def _signed_value(value: float, scale: float = 1.0, digits: int = 3) -> str:
        if not math.isfinite(value):
            return 'n/a'
        return f'{value / scale:+.{digits}f}'

    labels = ['Blade 1 |My|', 'Blade 2 |My|', 'Blade 3 |My|']
    for blade_index, axis in enumerate(axes):
        if show_fallback:
            _draw_fallback_spans(axis, elapsed_time_s, model.fallback_active)
        axis.plot(
            elapsed_time_s,
            model.trace.blade_root_my_abs_nm[:, blade_index] / 1e6,
            color=RL_COLOR,
            linewidth=0.85,
            label='RL deployment',
        )
        axis.plot(
            elapsed_time_s,
            pid.blade_root_my_abs_nm[:, blade_index] / 1e6,
            color=PID_COLOR,
            linewidth=0.85,
            label='PID validation',
        )
        _draw_episode_boundaries(axis, elapsed_time_s, episode_boundaries)
        axis.set_title(
            f'{labels[blade_index]} | RL peak {model_peaks_nm[blade_index] / 1e6:.3f} MNm | '
            f'PID peak {pid_peaks_nm[blade_index] / 1e6:.3f} MNm\n'
            f'RL - PID: {_signed_value(peak_delta_nm[blade_index], 1e6)} MNm '
            f'({_signed_value(peak_delta_percent[blade_index], 1.0, 2)}%)',
            fontsize=10,
            pad=8,
        )
        axis.set_ylabel('|My| (MNm)')
        axis.legend(loc='best')
        if blade_index == len(axes) - 1:
            axis.set_xlabel('Time (s)')
        _style_axis(axis, title_size=13.0, label_size=11.0, tick_size=10.0)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(figure)


def _plot_actions(
    output_path: Path,
    model: InferenceTrace,
    pid: Trace,
    episode_boundaries: list[int],
    show_fallback: bool,
) -> None:
    import matplotlib.pyplot as plt

    pid_torque_demand_nm, pid_pitch_demand_rad = _pid_action_demands(pid)
    figure, (wind_axis, torque_axis, pitch_axis) = plt.subplots(3, 1, figsize=(16, 11), sharex=True)
    elapsed_time_s = model.trace.elapsed_time_s

    if show_fallback:
        _draw_fallback_spans(wind_axis, elapsed_time_s, model.fallback_active)
    wind_axis.plot(
        elapsed_time_s,
        pid.wind_speed_mps,
        color=WIND_COLOR,
        linestyle='-',
        linewidth=1.0,
    )
    _draw_episode_boundaries(wind_axis, elapsed_time_s, episode_boundaries)
    wind_axis.set_title('Wind Speed')
    wind_axis.set_ylabel('Wind speed (m/s)')
    _style_axis(wind_axis, title_size=17.0)

    if show_fallback:
        _draw_fallback_spans(torque_axis, elapsed_time_s, model.fallback_active)
    torque_axis.plot(elapsed_time_s, model.sent_action_torque_nm / 1e3, color=PID_COLOR, linewidth=1.0, label='RL sent command')
    torque_axis.plot(elapsed_time_s, pid_torque_demand_nm / 1e3, color=RL_COLOR, linewidth=1.0, label='PID torque demand')
    _draw_episode_boundaries(torque_axis, elapsed_time_s, episode_boundaries)
    torque_axis.set_title('Generator Torque Command (kNm)')
    torque_axis.set_ylabel('Torque (kNm)')
    torque_axis.legend(loc='best')
    _style_axis(torque_axis, title_size=17.0)

    if show_fallback:
        _draw_fallback_spans(pitch_axis, elapsed_time_s, model.fallback_active)
    pitch_axis.plot(
        elapsed_time_s,
        np.rad2deg(model.sent_action_pitch_rad),
        color=PID_COLOR,
        linewidth=1.0,
        label='RL sent command',
    )
    pitch_axis.plot(
        elapsed_time_s,
        np.rad2deg(pid_pitch_demand_rad),
        color=RL_COLOR,
        linewidth=1.0,
        label='PID pitch demand',
    )
    _draw_episode_boundaries(pitch_axis, elapsed_time_s, episode_boundaries)
    pitch_axis.set_title('Pitch Position Demand (°)')
    pitch_axis.set_ylabel('Pitch angle (°)')
    pitch_axis.set_xlabel('Time (s)')
    pitch_axis.legend(loc='best')
    pitch_axis.set_xlabel('Time (s)')
    _style_axis(pitch_axis, title_size=17.0)
    figure.subplots_adjust(left=0.08, right=0.98, top=0.94, bottom=0.09, hspace=0.20)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _write_comparison_csv(
    output_path: Path,
    model: InferenceTrace,
    pid: Trace,
    model_energy_mwh: np.ndarray,
    pid_energy_mwh: np.ndarray,
) -> None:
    pid_torque_demand_nm, pid_pitch_demand_rad = _pid_action_demands(pid)
    fieldnames = [
        'matched_step',
        'model_episode',
        'model_global_step',
        'model_time_s',
        'model_elapsed_time_s',
        'pid_time_s',
        'pid_elapsed_time_s',
        'model_wind_speed_mps',
        'pid_wind_speed_mps',
        'wind_speed_delta_mps',
        'model_power_w',
        'pid_power_w',
        'power_delta_w',
        'model_cumulative_energy_mwh',
        'pid_cumulative_energy_mwh',
        'cumulative_energy_delta_mwh',
        'model_fallback_active',
        'model_ood_score',
        'model_sent_action_torque_nm',
        'pid_torque_demand_nm',
        'torque_command_delta_nm',
        'model_sent_action_pitch_rad',
        'pid_pitch_demand_rad',
        'pitch_command_delta_rad',
        'model_blade_root_1_my_abs_nm',
        'pid_blade_root_1_my_abs_nm',
        'model_blade_root_2_my_abs_nm',
        'pid_blade_root_2_my_abs_nm',
        'model_blade_root_3_my_abs_nm',
        'pid_blade_root_3_my_abs_nm',
        'model_blade_root_my_rms_nm',
        'pid_blade_root_my_rms_nm',
        'blade_root_my_rms_delta_nm',
    ]
    with output_path.open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(len(model.global_steps)):
            row = {
                'matched_step': index + 1,
                'model_episode': int(model.episodes[index]),
                'model_global_step': int(model.global_steps[index]),
                'model_time_s': float(model.trace.raw_time_s[index]),
                'model_elapsed_time_s': float(model.trace.elapsed_time_s[index]),
                'pid_time_s': float(pid.raw_time_s[index]),
                'pid_elapsed_time_s': float(pid.elapsed_time_s[index]),
                'model_wind_speed_mps': float(model.trace.wind_speed_mps[index]),
                'pid_wind_speed_mps': float(pid.wind_speed_mps[index]),
                'wind_speed_delta_mps': float(model.trace.wind_speed_mps[index] - pid.wind_speed_mps[index]),
                'model_power_w': float(model.trace.power_w[index]),
                'pid_power_w': float(pid.power_w[index]),
                'power_delta_w': float(model.trace.power_w[index] - pid.power_w[index]),
                'model_cumulative_energy_mwh': float(model_energy_mwh[index]),
                'pid_cumulative_energy_mwh': float(pid_energy_mwh[index]),
                'cumulative_energy_delta_mwh': float(model_energy_mwh[index] - pid_energy_mwh[index]),
                'model_fallback_active': float(model.fallback_active[index]),
                'model_ood_score': float(model.ood_score[index]),
                'model_sent_action_torque_nm': float(model.sent_action_torque_nm[index]),
                'pid_torque_demand_nm': float(pid_torque_demand_nm[index]),
                'torque_command_delta_nm': float(model.sent_action_torque_nm[index] - pid_torque_demand_nm[index]),
                'model_sent_action_pitch_rad': float(model.sent_action_pitch_rad[index]),
                'pid_pitch_demand_rad': float(pid_pitch_demand_rad[index]),
                'pitch_command_delta_rad': float(model.sent_action_pitch_rad[index] - pid_pitch_demand_rad[index]),
                'model_blade_root_1_my_abs_nm': float(model.trace.blade_root_my_abs_nm[index, 0]),
                'pid_blade_root_1_my_abs_nm': float(pid.blade_root_my_abs_nm[index, 0]),
                'model_blade_root_2_my_abs_nm': float(model.trace.blade_root_my_abs_nm[index, 1]),
                'pid_blade_root_2_my_abs_nm': float(pid.blade_root_my_abs_nm[index, 1]),
                'model_blade_root_3_my_abs_nm': float(model.trace.blade_root_my_abs_nm[index, 2]),
                'pid_blade_root_3_my_abs_nm': float(pid.blade_root_my_abs_nm[index, 2]),
                'model_blade_root_my_rms_nm': float(model.trace.blade_root_my_rms_nm[index]),
                'pid_blade_root_my_rms_nm': float(pid.blade_root_my_rms_nm[index]),
                'blade_root_my_rms_delta_nm': float(model.trace.blade_root_my_rms_nm[index] - pid.blade_root_my_rms_nm[index]),
            }
            writer.writerow(row)


def _episode_summaries(model: InferenceTrace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for episode in np.unique(model.episodes):
        mask = model.episodes == episode
        rows.append(
            {
                'episode': int(episode),
                'steps': int(np.count_nonzero(mask)),
                'power_mean_w': float(np.mean(model.trace.power_w[mask])),
                'blade_root_my_rms_mean_nm': float(np.mean(model.trace.blade_root_my_rms_nm[mask])),
                'fallback_fraction': float(np.mean(model.fallback_active[mask])),
                'ood_score_mean': float(np.mean(model.ood_score[mask])),
            }
        )
    return rows


def _blade_root_load_summary(model: InferenceTrace, pid: Trace) -> dict[str, Any]:
    model_peaks_nm = np.max(model.trace.blade_root_my_abs_nm, axis=0)
    pid_peaks_nm = np.max(pid.blade_root_my_abs_nm, axis=0)
    peak_delta_nm = model_peaks_nm - pid_peaks_nm
    peak_delta_percent = np.divide(
        peak_delta_nm,
        pid_peaks_nm,
        out=np.zeros_like(peak_delta_nm),
        where=pid_peaks_nm > 0.0,
    ) * 100.0
    load_reduction_percent = -peak_delta_percent
    overall_rl_peak_nm = float(np.max(model_peaks_nm))
    overall_pid_peak_nm = float(np.max(pid_peaks_nm))
    overall_peak_delta_nm = overall_rl_peak_nm - overall_pid_peak_nm
    overall_peak_delta_percent = (
        overall_peak_delta_nm / overall_pid_peak_nm * 100.0
        if overall_pid_peak_nm > 0.0
        else 0.0
    )
    overall_rl_rms_nm = float(np.mean(model.trace.blade_root_my_rms_nm))
    overall_pid_rms_nm = float(np.mean(pid.blade_root_my_rms_nm))
    overall_rms_delta_nm = overall_rl_rms_nm - overall_pid_rms_nm
    overall_rms_delta_percent = (
        overall_rms_delta_nm / overall_pid_rms_nm * 100.0
        if overall_pid_rms_nm > 0.0
        else 0.0
    )
    return {
        'rl_peak_nm': model_peaks_nm.astype(float).tolist(),
        'pid_peak_nm': pid_peaks_nm.astype(float).tolist(),
        'peak_delta_nm': peak_delta_nm.astype(float).tolist(),
        'peak_delta_percent': peak_delta_percent.astype(float).tolist(),
        'load_reduction_percent': load_reduction_percent.astype(float).tolist(),
        'formal_min_load_reduction_percent': float(np.min(load_reduction_percent)),
        'overall_peak_envelope': {
            'rl_nm': overall_rl_peak_nm,
            'pid_nm': overall_pid_peak_nm,
            'delta_nm': float(overall_peak_delta_nm),
            'delta_percent': float(overall_peak_delta_percent),
        },
        'overall_rms_mean': {
            'rl_nm': overall_rl_rms_nm,
            'pid_nm': overall_pid_rms_nm,
            'delta_nm': float(overall_rms_delta_nm),
            'delta_percent': float(overall_rms_delta_percent),
        },
    }


def build_comparison(
    inference_dir: Path,
    pid_csv: Path,
    output_dir: Path,
    pid_start_row: int = 0,
    show_fallback: bool = True,
    formal_acceptance: bool = False,
) -> dict[str, Any]:
    inference_csv = inference_dir / 'bladed_inference_steps.csv'
    if not inference_csv.is_file():
        raise FileNotFoundError(f'Missing inference record file: {inference_csv}')
    model = _load_inference_trace(inference_csv)
    pid = _load_pid_trace(pid_csv, start_row=pid_start_row, row_count=len(model.global_steps))
    output_dir.mkdir(parents=True, exist_ok=True)

    model_energy_mwh = _cumulative_energy_mwh(model.trace.power_w, model.trace.elapsed_time_s)
    pid_energy_mwh = _cumulative_energy_mwh(pid.power_w, pid.elapsed_time_s)
    episode_boundaries = _episode_boundaries(model)
    pid_torque_demand_nm, pid_pitch_demand_rad = _pid_action_demands(pid)
    blade_root_load_summary = _blade_root_load_summary(model, pid)

    comparison_csv = output_dir / 'inference_pid_matched_steps.csv'
    _write_comparison_csv(comparison_csv, model, pid, model_energy_mwh, pid_energy_mwh)
    power_plot = output_dir / 'inference_vs_pid_power_step.png'
    wind_speed_plot = output_dir / 'inference_vs_pid_wind_speed.png'
    energy_plot = output_dir / 'inference_vs_pid_cumulative_energy.png'
    blade_plot = output_dir / 'inference_vs_pid_blade_root_my.png'
    action_plot = output_dir / 'inference_vs_pid_actions.png'
    _plot_power(power_plot, model, pid, episode_boundaries, show_fallback=show_fallback)
    _plot_wind_speed(wind_speed_plot, model, pid, episode_boundaries, show_fallback=show_fallback)
    _plot_cumulative_energy(energy_plot, model, pid, model_energy_mwh, pid_energy_mwh)
    _plot_blade_root_load(blade_plot, model, pid, episode_boundaries, show_fallback=show_fallback)
    _plot_actions(action_plot, model, pid, episode_boundaries, show_fallback=show_fallback)

    formal_report: dict[str, Any] | None = None
    formal_report_path: Path | None = None
    if formal_acceptance:
        formal_report = build_formal_acceptance_report(model, pid)
        formal_report_path = output_dir / 'formal_acceptance_report.json'
        with formal_report_path.open('w', encoding='utf-8') as file:
            json.dump(formal_report, file, ensure_ascii=False, indent=2, allow_nan=False)

    fallback_fraction = float(np.mean(model.fallback_active))
    summary = {
        'inference_dir': str(inference_dir),
        'inference_csv': str(inference_csv),
        'pid_csv': str(pid_csv),
        'pid_start_row_zero_based': int(pid_start_row),
        'ignored_trailing_control_rows': int(model.ignored_trailing_control_rows),
        'show_fallback_markers': bool(show_fallback),
        'matched_control_steps': int(len(model.global_steps)),
        'model_elapsed_duration_s': float(model.trace.elapsed_time_s[-1]),
        'pid_elapsed_duration_s': float(pid.elapsed_time_s[-1]),
        'model_power_w': _finite_statistics(model.trace.power_w),
        'pid_power_w': _finite_statistics(pid.power_w),
        'power_delta_w': _finite_statistics(model.trace.power_w - pid.power_w),
        'model_wind_speed_mps': _finite_statistics(model.trace.wind_speed_mps),
        'pid_wind_speed_mps': _finite_statistics(pid.wind_speed_mps),
        'wind_speed_delta_mps': _finite_statistics(model.trace.wind_speed_mps - pid.wind_speed_mps),
        'model_sent_action_torque_nm': _finite_statistics(model.sent_action_torque_nm),
        'pid_torque_demand_nm': _finite_statistics(pid_torque_demand_nm),
        'torque_command_delta_nm': _finite_statistics(model.sent_action_torque_nm - pid_torque_demand_nm),
        'model_sent_action_pitch_rad': _finite_statistics(model.sent_action_pitch_rad),
        'pid_pitch_demand_rad': _finite_statistics(pid_pitch_demand_rad),
        'pitch_command_delta_rad': _finite_statistics(model.sent_action_pitch_rad - pid_pitch_demand_rad),
        'model_blade_root_my_rms_nm': _finite_statistics(model.trace.blade_root_my_rms_nm),
        'pid_blade_root_my_rms_nm': _finite_statistics(pid.blade_root_my_rms_nm),
        'blade_root_my_rms_delta_nm': _finite_statistics(model.trace.blade_root_my_rms_nm - pid.blade_root_my_rms_nm),
        'blade_root_limit_load': blade_root_load_summary,
        'model_cumulative_energy_mwh': float(model_energy_mwh[-1]),
        'pid_cumulative_energy_mwh': float(pid_energy_mwh[-1]),
        'cumulative_energy_delta_mwh': float(model_energy_mwh[-1] - pid_energy_mwh[-1]),
        'cumulative_energy_delta_kwh': float((model_energy_mwh[-1] - pid_energy_mwh[-1]) * 1.0e3),
        'cumulative_energy_delta_percent': float(
            (model_energy_mwh[-1] / pid_energy_mwh[-1] - 1.0) * 100.0
            if abs(float(pid_energy_mwh[-1])) > 1.0e-12
            else 0.0
        ),
        'fallback_step_count': int(np.count_nonzero(model.fallback_active > 0.5)),
        'fallback_fraction': fallback_fraction,
        'policy_active_fraction': float(1.0 - fallback_fraction),
        'ood_score': _finite_statistics(model.ood_score),
        'episode_boundaries_after_step_indices': [int(model.global_steps[index - 1]) for index in episode_boundaries],
        'per_episode': _episode_summaries(model),
        'comparison_csv': str(comparison_csv),
        'power_plot': str(power_plot),
        'wind_speed_plot': str(wind_speed_plot),
        'cumulative_energy_plot': str(energy_plot),
        'blade_root_plot': str(blade_plot),
        'action_plot': str(action_plot),
    }
    if formal_report is not None and formal_report_path is not None:
        summary['formal_acceptance'] = formal_report
        summary['formal_acceptance_report_json'] = str(formal_report_path)
    summary_path = output_dir / 'inference_pid_comparison_summary.json'
    with summary_path.open('w', encoding='utf-8') as file:
        json.dump(summary, file, ensure_ascii=False, indent=2, allow_nan=False)
    summary['summary_json'] = str(summary_path)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Compare live TD3+BC Bladed inference records with a matched PID raw Bladed trajectory.',
    )
    parser.add_argument('--inference-dir', required=True, help='Directory containing bladed_inference_steps.csv.')
    parser.add_argument(
        '--pid-csv',
        default='bladed-data-new/raw/val.csv',
        help='PID raw Bladed CSV. The first matched rows are used by default.',
    )
    parser.add_argument('--pid-start-row', type=int, default=0, help='Zero-based PID row offset before the matched window.')
    parser.add_argument(
        '--output-dir',
        default='',
        help='Defaults to <inference-dir>/figures/inference_vs_pid.',
    )
    parser.add_argument(
        '--hide-fallback',
        action='store_true',
        help='Hide fallback shading and the fallback panel; uses a separate default output directory.',
    )
    parser.add_argument(
        '--formal-acceptance',
        action='store_true',
        help='Write a strict energy-and-per-blade-peak acceptance report using the fixed P0/P1/L0/L1 criteria.',
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    inference_dir = _resolve_path(args.inference_dir)
    if not inference_dir.is_dir():
        raise NotADirectoryError(f'Inference path is not a directory: {inference_dir}')
    pid_csv = _resolve_path(args.pid_csv)
    if not pid_csv.is_file():
        raise FileNotFoundError(f'PID CSV is not a file: {pid_csv}')
    if args.output_dir.strip():
        output_dir = _resolve_output_dir(args.output_dir)
    elif args.hide_fallback:
        output_dir = inference_dir / 'figures' / 'inference_vs_pid_no_fallback'
    else:
        output_dir = inference_dir / 'figures' / 'inference_vs_pid'
    summary = build_comparison(
        inference_dir=inference_dir,
        pid_csv=pid_csv,
        output_dir=output_dir,
        pid_start_row=args.pid_start_row,
        show_fallback=not args.hide_fallback,
        formal_acceptance=args.formal_acceptance,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()