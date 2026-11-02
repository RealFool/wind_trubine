"""Exploratory, read-only action-to-My lag diagnostics for a frozen R4 run.

This analyzes an already completed inference CSV. It does not start Bladed,
inject actions, or estimate a causal treatment effect. The full inference CSV
is parsed one row at a time; only the four compact 600 s slot arrays are kept.
"""
from __future__ import annotations

import argparse
from array import array
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DT = 0.02
EXPECTED_CONDITIONS = (8, 11, 12, 16)
LAG_FRAMES = (0, 1, 2, 3, 5, 10, 15, 25, 50, 75, 100)
LOW_LOAD_QUANTILE = 0.25
MY_FIELDS = (
    'Blade root 1 My (Nm)',
    'Blade root 2 My (Nm)',
    'Blade root 3 My (Nm)',
)
ARRAY_FIELDS = (
    'Time', 'I_RotorAzimuth', 'I_RotorSpeed', 'I_AverageWindSpeed',
    *MY_FIELDS,
    'sent_action_torque', 'sent_action_pitch',
    'policy_delta_torque', 'policy_delta_pitch',
    'baseline_action_torque', 'baseline_action_pitch',
    'requested_action_torque', 'requested_action_pitch',
    'info_udp_sim_time', 'info_udp_next_sim_time',
    'info_udp_frame_id', 'info_udp_next_frame_id',
    'info_udp_handshake_complete', 'info_udp_inter_frame_timeout',
    'info_udp_frame_lag', 'info_udp_dropped_stale_frames',
    'info_env_invalid_prediction', 'info_env_invalid_reward',
    'info_control_unsafe_stop', 'info_control_fallback_active',
    'info_control_action_projection_applied', 'info_control_action_rate_limited',
    'info_control_pitch_protection_applied', 'info_control_low_speed_stage',
    'info_condition_end', 'step_invalid',
)
REQUIRED_FIELDS = {
    'record_type', 'source_frame_role', 'inference_episode', 'inference_step',
    'inference_global_control_step', 'info_udp_condition_id',
    'info_udp_condition_instance_id', 'info_udp_queue_epoch', *ARRAY_FIELDS,
}
SIGNALS = {
    'sent_torque_nm': ('sent_action_torque', 'Nm'),
    'sent_pitch_rad': ('sent_action_pitch', 'rad'),
    'actor_delta_torque_nm': ('policy_delta_torque', 'Nm'),
    'actor_delta_pitch_rad': ('policy_delta_pitch', 'rad'),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError(f'Expected a JSON object: {path}')
    return value


def _number(row: dict[str, str], name: str, row_number: int, *, default: float | None = None) -> float:
    text = row.get(name, '')
    if text == '' and default is not None:
        return default
    try:
        value = float(text)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'Invalid numeric field {name!r} at CSV row {row_number}') from exc
    if not math.isfinite(value):
        raise ValueError(f'Non-finite field {name!r} at CSV row {row_number}')
    return value


def _read_contract(run_dir: Path) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    required_paths = (
        run_dir / 'run_summary.json', run_dir / 'effective_config.json',
        run_dir / 'episode_summary.csv',
    )
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(f'Missing frozen-run evidence: {path}')
    run_summary = _read_json(run_dir / 'run_summary.json')
    config = _read_json(run_dir / 'effective_config.json')
    config_dir = run_dir.parents[1] / 'configs' / run_dir.name
    receipt_path = config_dir / 'round4_inference_receipt.json'
    if not receipt_path.is_file():
        raise FileNotFoundError(f'Missing frozen Actor receipt: {receipt_path}')
    receipt = _read_json(receipt_path)

    if (run_summary.get('status') != 'completed' or run_summary.get('queue_mode') is not True or
            run_summary.get('episodes_requested') != 4 or run_summary.get('episodes_completed') != 4):
        raise ValueError('Input is not a completed four-slot frozen queue inference')
    role = config.get('physical_shadow_run_role', {})
    if (role.get('actor_frozen') is not True or role.get('optimizer_updates') is not False or
            config.get('train', {}).get('online_rl_enabled') is not False):
        raise ValueError('Effective config does not bind a frozen, optimizer-disabled Actor run')
    authorization = receipt.get('authorization', {}).get('physical_shadow', {})
    if (receipt.get('route') != 'operator_confirmed_round4_frozen_actor_4wind1round600_shadow_inference_v1' or
            receipt.get('run_id') != run_dir.name or
            receipt.get('actor_frozen') is not True or receipt.get('optimizer_updates') is not False or
            receipt.get('policy_acceptance') is not False or
            authorization.get('counterfactual_validation') is not False or
            authorization.get('soft_authorized') is not False or
            authorization.get('hard_authorized') is not False):
        raise ValueError('Round4 receipt does not attest the expected development-only scope')

    episodes: dict[int, dict[str, Any]] = {}
    with (run_dir / 'episode_summary.csv').open('r', encoding='utf-8-sig', newline='') as handle:
        for row_number, row in enumerate(csv.DictReader(handle), start=2):
            episode = int(row['episode'])
            if episode in episodes:
                raise ValueError(f'Duplicate episode summary at row {row_number}')
            episodes[episode] = {
                'episode': episode,
                'condition_id': int(float(row['queue_condition_id'])),
                'condition_instance_id': int(float(row['queue_condition_instance_id'])),
                'queue_epoch': int(float(row['queue_epoch'])),
                'expected_control_steps': int(row['control_steps']),
                'identity_status': row.get('case_identity_status', 'unknown'),
                'strict_case_matched': bool(int(float(row.get('case_identity_strict_case_matched') or 0))),
            }
    if len(episodes) != 4 or tuple(episodes[i]['condition_id'] for i in sorted(episodes)) != EXPECTED_CONDITIONS:
        raise ValueError('Episode summaries do not match the expected [8, 11, 12, 16] wind slots')
    return episodes, {
        'run_summary': run_summary,
        'effective_config': config,
        'receipt': receipt,
        'receipt_path': receipt_path,
        'actor_checkpoint_sha256': receipt.get('runtime_identity', {}).get('actor_checkpoint_sha256'),
    }


def _read_inference_csv(run_dir: Path, episodes: dict[int, dict[str, Any]]) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    path = run_dir / 'bladed_inference_steps.csv'
    if not path.is_file():
        raise FileNotFoundError(f'Missing inference steps CSV: {path}')

    slots: dict[int, dict[str, Any]] = {}
    for episode, metadata in episodes.items():
        slots[episode] = {
            'metadata': metadata,
            'columns': {name: array('d') for name in ARRAY_FIELDS},
            'valid': array('b'),
            'steps': array('q'),
            'global_steps': array('q'),
            'non_control_records': 0,
        }

    record_types: dict[str, int] = {}
    total_rows = 0
    global_step_previous = 0
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if len(fieldnames) != len(set(fieldnames)):
            raise ValueError('Inference CSV has duplicate column names')
        missing = REQUIRED_FIELDS - set(fieldnames)
        if missing:
            raise ValueError(f'Inference CSV is missing required columns: {sorted(missing)}')
        for row_number, row in enumerate(reader, start=2):
            total_rows += 1
            record_type = row['record_type']
            record_types[record_type] = record_types.get(record_type, 0) + 1
            if record_type != 'control_step':
                try:
                    episode = int(row['inference_episode'])
                except (TypeError, ValueError):
                    continue
                if episode in slots:
                    slots[episode]['non_control_records'] += 1
                continue
            if row['source_frame_role'] != 'control_input':
                raise ValueError(f'Unexpected control_step source role at CSV row {row_number}')
            episode = int(row['inference_episode'])
            if episode not in slots:
                raise ValueError(f'Control row references unknown episode {episode} at CSV row {row_number}')
            metadata = slots[episode]['metadata']
            if (int(float(row['info_udp_condition_id'])) != metadata['condition_id'] or
                    int(float(row['info_udp_condition_instance_id'])) != metadata['condition_instance_id'] or
                    int(float(row['info_udp_queue_epoch'])) != metadata['queue_epoch']):
                raise ValueError(f'Queue slot identity mismatch at CSV row {row_number}')

            step = int(row['inference_step'])
            expected_step = len(slots[episode]['steps']) + 1
            if step != expected_step:
                raise ValueError(f'Non-contiguous inference_step for episode {episode}: {step} != {expected_step}')
            global_step = int(row['inference_global_control_step'])
            if global_step <= global_step_previous:
                raise ValueError(f'Global control step is not strictly increasing at CSV row {row_number}')
            global_step_previous = global_step
            slots[episode]['steps'].append(step)
            slots[episode]['global_steps'].append(global_step)

            values = {name: _number(row, name, row_number) for name in ARRAY_FIELDS}
            for name, value in values.items():
                slots[episode]['columns'][name].append(value)
            valid = (
                values['step_invalid'] == 0.0 and
                values['info_udp_handshake_complete'] == 1.0 and
                values['info_udp_inter_frame_timeout'] == 0.0 and
                values['info_udp_frame_lag'] == 0.0 and
                values['info_udp_dropped_stale_frames'] == 0.0 and
                values['info_env_invalid_prediction'] == 0.0 and
                values['info_env_invalid_reward'] == 0.0 and
                values['info_control_unsafe_stop'] == 0.0
            )
            slots[episode]['valid'].append(int(valid))

    for episode, slot in slots.items():
        expected_steps = slot['metadata']['expected_control_steps']
        if len(slot['steps']) != expected_steps:
            raise ValueError(f'Episode {episode} control-row count mismatch: {len(slot["steps"])} != {expected_steps}')
        if expected_steps != 30_001:
            raise ValueError(f'Episode {episode} is not the expected 600 s / 20 ms trace')
        slot['columns'] = {name: np.frombuffer(values, dtype=np.float64) for name, values in slot['columns'].items()}
        slot['valid'] = np.frombuffer(slot['valid'], dtype=np.int8).astype(bool)
        slot['steps'] = np.frombuffer(slot['steps'], dtype=np.int64)
        slot['global_steps'] = np.frombuffer(slot['global_steps'], dtype=np.int64)
        columns = slot['columns']
        times = columns['Time']
        expected_times = np.arange(expected_steps, dtype=np.float64) * DT
        if not np.allclose(times, expected_times, rtol=0.0, atol=1e-6):
            raise ValueError(f'Episode {episode} raw Time is not the expected 0..600 s grid')
        if not np.allclose(columns['info_udp_sim_time'], times, rtol=0.0, atol=1e-6):
            raise ValueError(f'Episode {episode} Time and current UDP simulation time differ')
        if not np.allclose(columns['info_udp_frame_id'], np.arange(1, expected_steps + 1), rtol=0.0, atol=0.0):
            raise ValueError(f'Episode {episode} frame IDs are not a contiguous 1-based sequence')
        if not np.allclose(columns['info_udp_next_frame_id'], columns['info_udp_frame_id'] + 1,
                           rtol=0.0, atol=0.0):
            raise ValueError(f'Episode {episode} next-frame IDs are not consecutive')
        next_delta = columns['info_udp_next_sim_time'] - times
        if (not np.allclose(next_delta[:-1], DT, rtol=0.0, atol=1e-6) or
                not math.isclose(float(next_delta[-1]), 0.01, rel_tol=0.0, abs_tol=1e-6) or
                columns['info_condition_end'][-1] != 1.0):
            raise ValueError(f'Episode {episode} terminal END delimiter timing did not close as expected')
        slot['valid_bad_prefix'] = np.concatenate(([0], np.cumsum(~slot['valid'], dtype=np.int64)))
        slot['max_abs_my_nm_by_frame'] = np.max(np.abs(np.column_stack(
            [columns[field] for field in MY_FIELDS])), axis=1)
        slot['low_load_threshold_nm'] = float(np.quantile(
            slot['max_abs_my_nm_by_frame'][slot['valid']], LOW_LOAD_QUANTILE))

    return slots, {
        'path': path,
        'sha256': _sha256(path),
        'bytes': path.stat().st_size,
        'csv_rows': total_rows,
        'record_type_counts': record_types,
    }


def _pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2 or len(y) != len(x):
        return None
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    denominator = float(np.sqrt(np.dot(x_centered, x_centered) * np.dot(y_centered, y_centered)))
    if denominator <= 1e-20:
        return None
    return float(np.dot(x_centered, y_centered) / denominator)


def _window_is_valid(slot: dict[str, Any], start: np.ndarray, end: np.ndarray) -> np.ndarray:
    prefix = slot['valid_bad_prefix']
    return prefix[end + 1] - prefix[start] == 0


def _append_correlation(
    rows: list[dict[str, Any]],
    pooled: dict[tuple[str, int, int, str, str], dict[str, Any]],
    *, episode: int, condition_id: int, stratum: str, blade: int, lag: int,
    signal: str, metric: str, x: np.ndarray, y: np.ndarray,
) -> None:
    r = _pearson(x, y)
    rows.append({
        'grouping': 'per_wind_slot', 'condition_id': condition_id, 'episode': episode,
        'stratum': stratum, 'blade': blade, 'lag_frames': lag, 'lag_s': round(lag * DT, 6),
        'lag_semantics': 'contemporaneous_noncausal_baseline' if lag == 0 else 'action_time_precedes_measurement',
        'signal': signal, 'metric': metric, 'n': int(len(x)), 'r': r, 'slot_count': 1,
    })
    key = (stratum, blade, lag, signal, metric)
    stats = pooled.setdefault(key, {'n': 0, 'cov': 0.0, 'var_x': 0.0, 'var_y': 0.0, 'slot_rs': []})
    stats['slot_rs'].append(r)
    if len(x) < 2:
        return
    x0 = x - np.mean(x)
    y0 = y - np.mean(y)
    stats['n'] += int(len(x))
    stats['cov'] += float(np.dot(x0, y0))
    stats['var_x'] += float(np.dot(x0, x0))
    stats['var_y'] += float(np.dot(y0, y0))


def _correlation_rows(slots: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pooled: dict[tuple[str, int, int, str, str], dict[str, Any]] = {}
    for episode in sorted(slots):
        slot = slots[episode]
        c = slot['columns']
        n = len(c['Time'])
        abs_my = np.abs(np.column_stack([c[field] for field in MY_FIELDS]))
        actor_active = c['info_control_fallback_active'] == 0.0
        for lag in LAG_FRAMES:
            count = n - lag
            start = np.arange(count, dtype=np.int64)
            end = start + lag
            valid = _window_is_valid(slot, start, end)
            for stratum in ('all_valid', 'relative_low_load_q25'):
                select = valid.copy()
                if stratum == 'relative_low_load_q25':
                    select &= slot['max_abs_my_nm_by_frame'][start] <= slot['low_load_threshold_nm']
                for blade in range(1, 4):
                    outcome = abs_my[:, blade - 1]
                    for signal_name, (signal_field, _unit) in SIGNALS.items():
                        signal_select = select.copy()
                        if signal_name.startswith('actor_delta_'):
                            signal_select &= actor_active[start]
                        x = c[signal_field][start][signal_select]
                        y = outcome[end][signal_select]
                        _append_correlation(
                            rows, pooled, episode=episode, condition_id=slot['metadata']['condition_id'],
                            stratum=stratum, blade=blade, lag=lag, signal=signal_name,
                            metric='action_level_vs_future_abs_my', x=x, y=y,
                        )
                        if lag == 0:
                            continue
                        delta_start = np.arange(1, n - lag, dtype=np.int64)
                        delta_end = delta_start + lag
                        delta_valid = _window_is_valid(slot, delta_start - 1, delta_end)
                        if stratum == 'relative_low_load_q25':
                            delta_valid &= (slot['max_abs_my_nm_by_frame'][delta_start] <=
                                            slot['low_load_threshold_nm'])
                        if signal_name.startswith('actor_delta_'):
                            delta_valid &= actor_active[delta_start] & actor_active[delta_start - 1]
                        x_change = c[signal_field][delta_start] - c[signal_field][delta_start - 1]
                        # Compare against the one-frame My increment at the
                        # selected lag endpoint; do not let response-window
                        # duration grow with lag.
                        y_change = outcome[delta_end] - outcome[delta_end - 1]
                        _append_correlation(
                            rows, pooled, episode=episode, condition_id=slot['metadata']['condition_id'],
                            stratum=stratum, blade=blade, lag=lag, signal=signal_name,
                            metric='action_step_vs_abs_my_change',
                            x=x_change[delta_valid], y=y_change[delta_valid],
                        )

    # Weighted pooled correlation after centering within each wind slot, and an
    # equal-wind mean of the per-slot r values are both descriptive summaries.
    for key, stats in pooled.items():
        stratum, blade, lag, signal, metric = key
        n = stats['n']
        denom = math.sqrt(stats['var_x'] * stats['var_y'])
        r = None if denom <= 1e-20 else stats['cov'] / denom
        rows.append({
            'grouping': 'pooled_within_slot_variance_weighted', 'condition_id': '',
            'stratum': stratum, 'blade': blade, 'lag_frames': lag, 'lag_s': round(lag * DT, 6),
            'lag_semantics': 'contemporaneous_noncausal_baseline' if lag == 0 else 'action_time_precedes_measurement',
            'signal': signal, 'metric': metric, 'n': n, 'r': None if r is None else float(r),
            'slot_count': len(stats['slot_rs']),
        })
        rs = [value for value in stats['slot_rs'] if value is not None]
        rows.append({
            'grouping': 'equal_slot_mean_r', 'condition_id': '', 'stratum': stratum,
            'blade': blade, 'lag_frames': lag, 'lag_s': round(lag * DT, 6),
            'lag_semantics': 'contemporaneous_noncausal_baseline' if lag == 0 else 'action_time_precedes_measurement',
            'signal': signal, 'metric': metric, 'n': n,
            'r': None if not rs else float(np.mean(rs)), 'slot_count': len(rs),
        })
    return rows


def _standardized_ols(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, float | None, int, float | None]:
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.shape[0] or len(y) < x.shape[1] + 5:
        return np.full(x.shape[1] if x.ndim == 2 else 0, np.nan), None, 0, None
    y_std = float(np.std(y))
    x_std = np.std(x, axis=0)
    active = x_std > 1e-12
    if y_std <= 1e-12 or int(np.count_nonzero(active)) == 0:
        return np.full(x.shape[1], np.nan), None, int(np.count_nonzero(active)), None
    xz = (x[:, active] - np.mean(x[:, active], axis=0)) / x_std[active]
    yz = (y - np.mean(y)) / y_std
    beta, _, rank, singular = np.linalg.lstsq(xz, yz, rcond=None)
    prediction = xz @ beta
    denominator = float(np.dot(yz, yz))
    r2 = None if denominator <= 1e-20 else float(1.0 - np.dot(yz - prediction, yz - prediction) / denominator)
    full_beta = np.full(x.shape[1], np.nan)
    full_beta[active] = beta
    condition = None
    if len(singular) and singular[-1] > 1e-12:
        condition = float(singular[0] / singular[-1])
    return full_beta, r2, int(rank), condition


def _regression_rows(slots: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for episode in sorted(slots):
        slot = slots[episode]
        c = slot['columns']
        n = len(c['Time'])
        my = np.abs(np.column_stack([c[field] for field in MY_FIELDS]))
        azimuth = c['I_RotorAzimuth']
        actor_active = c['info_control_fallback_active'] == 0.0
        for stratum in ('all_valid', 'relative_low_load_q25'):
            for lag in LAG_FRAMES:
                # u_t is sent after My[t] was observed, so lag zero cannot
                # estimate a directional action-to-response relation.
                if lag == 0:
                    continue
                # For each My increment ending at t, compare an earlier action
                # change at t-lag; choose low-load status at that action origin.
                t = np.arange(max(1, lag + 1), n, dtype=np.int64)
                action_index = t - lag
                start = action_index - 1
                end = t
                control_index = action_index
                valid = _window_is_valid(slot, start, end)
                if stratum == 'relative_low_load_q25':
                    valid &= (slot['max_abs_my_nm_by_frame'][control_index] <=
                              slot['low_load_threshold_nm'])
                t = t[valid]
                action_index = action_index[valid]
                control_index = control_index[valid]
                if len(t) < 100:
                    continue

                wind_delta = c['I_AverageWindSpeed'][t] - c['I_AverageWindSpeed'][control_index]
                rotor_delta = c['I_RotorSpeed'][t] - c['I_RotorSpeed'][control_index]
                controls = np.column_stack((
                    my[control_index, 0],  # replaced per blade below
                    c['I_AverageWindSpeed'][control_index], wind_delta,
                    c['I_RotorSpeed'][control_index], rotor_delta,
                    np.sin(azimuth[control_index]), np.cos(azimuth[control_index]),
                    c['baseline_action_torque'][control_index], c['baseline_action_pitch'][control_index],
                ))
                action_families = {
                    'sent_action_change': ('sent_action_torque', 'sent_action_pitch'),
                    'actor_policy_residual_change': ('policy_delta_torque', 'policy_delta_pitch'),
                }
                for family, (torque_field, pitch_field) in action_families.items():
                    torque_change = c[torque_field][action_index] - c[torque_field][action_index - 1]
                    pitch_change = c[pitch_field][action_index] - c[pitch_field][action_index - 1]
                    family_valid = np.ones(len(t), dtype=bool)
                    if family == 'actor_policy_residual_change':
                        family_valid &= actor_active[action_index] & actor_active[action_index - 1]
                    for blade in range(1, 4):
                        controls[:, 0] = my[control_index, blade - 1]
                        x = np.column_stack((torque_change, pitch_change, controls))
                        y = my[t, blade - 1] - my[t - 1, blade - 1]
                        beta, r2, rank, condition = _standardized_ols(
                            y[family_valid], x[family_valid])
                        rows.append({
                            'grouping': 'per_wind_slot',
                            'episode': episode,
                            'condition_id': slot['metadata']['condition_id'],
                            'stratum': stratum,
                            'blade': blade,
                            'lag_frames': lag,
                            'lag_s': round(lag * DT, 6),
                            'lag_semantics': 'earlier_action_change_vs_current_my_increment',
                            'action_family': family,
                            'n': int(np.count_nonzero(family_valid)),
                            'beta_torque_standardized': None if not math.isfinite(float(beta[0])) else float(beta[0]),
                            'beta_pitch_standardized': None if not math.isfinite(float(beta[1])) else float(beta[1]),
                            'r2_in_sample': r2,
                            'design_rank': rank,
                            'design_condition_number': condition,
                        })
    return rows


def _slot_quality_rows(slots: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for episode in sorted(slots):
        slot = slots[episode]
        c = slot['columns']
        my = np.abs(np.column_stack([c[field] for field in MY_FIELDS]))
        action_torque = c['sent_action_torque']
        action_pitch = c['sent_action_pitch']
        actor_active = c['info_control_fallback_active'] == 0.0
        rows.append({
            'episode': episode,
            'condition_id': slot['metadata']['condition_id'],
            'condition_instance_id': slot['metadata']['condition_instance_id'],
            'queue_epoch': slot['metadata']['queue_epoch'],
            'identity_status': slot['metadata']['identity_status'],
            'strict_case_matched': slot['metadata']['strict_case_matched'],
            'control_steps': len(c['Time']),
            'analysis_valid_frames': int(np.count_nonzero(slot['valid'])),
            'invalid_or_unconfirmed_frames': int(len(slot['valid']) - np.count_nonzero(slot['valid'])),
            'terminal_end_delimiter_rows': int(np.count_nonzero(c['info_condition_end'])),
            'relative_low_load_definition': 'max(abs(My blade 1..3)) <= this slot empirical q25; not a safety limit',
            'relative_low_load_q25_max_abs_my_mnm': slot['low_load_threshold_nm'] / 1e6,
            'relative_low_load_frames': int(np.count_nonzero(
                slot['valid'] & (slot['max_abs_my_nm_by_frame'] <= slot['low_load_threshold_nm']))),
            'max_abs_my_q50_mnm': float(np.quantile(slot['max_abs_my_nm_by_frame'], 0.50) / 1e6),
            'max_abs_my_q90_mnm': float(np.quantile(slot['max_abs_my_nm_by_frame'], 0.90) / 1e6),
            'max_abs_my_peak_mnm': float(np.max(my) / 1e6),
            'sent_torque_abs_step_p95_knm': float(np.quantile(np.abs(np.diff(action_torque)), 0.95) / 1000.0),
            'sent_pitch_abs_step_p95_mrad': float(np.quantile(np.abs(np.diff(action_pitch)), 0.95) * 1000.0),
            'actor_active_frames': int(np.count_nonzero(actor_active)),
            'actor_delta_torque_abs_p95_knm': float(np.quantile(np.abs(c['policy_delta_torque'][actor_active]), 0.95) / 1000.0),
            'actor_delta_pitch_abs_p95_mrad': float(np.quantile(np.abs(c['policy_delta_pitch'][actor_active]), 0.95) * 1000.0),
            'fallback_frames': int(np.count_nonzero(c['info_control_fallback_active'] > 0.0)),
            'projection_frames': int(np.count_nonzero(c['info_control_action_projection_applied'] > 0.0)),
            'rate_limit_frames': int(np.count_nonzero(c['info_control_action_rate_limited'] > 0.0)),
            'pitch_protection_frames': int(np.count_nonzero(c['info_control_pitch_protection_applied'] > 0.0)),
            'low_speed_stage_frames': int(np.count_nonzero(c['info_control_low_speed_stage'] > 0.0)),
        })
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f'Refusing to write an empty analysis table: {path.name}')
    fields = list(rows[0])
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='raise')
        writer.writeheader()
        writer.writerows(rows)


def _write_plots(output_dir: Path, correlation_rows: list[dict[str, Any]]) -> list[str]:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    pooled = [row for row in correlation_rows
              if row['grouping'] == 'pooled_within_slot_variance_weighted' and
              row['metric'] == 'action_step_vs_abs_my_change' and int(row['lag_frames']) > 0]
    filenames = []
    for stratum, suffix in (('all_valid', 'all_valid'), ('relative_low_load_q25', 'relative_low_load_q25')):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
        for ax, signal, title in zip(
            axes.flat,
            ('sent_torque_nm', 'sent_pitch_rad', 'actor_delta_torque_nm', 'actor_delta_pitch_rad'),
            ('Sent torque command change', 'Sent pitch command change',
             'Actor torque residual change', 'Actor pitch residual change'),
            strict=True,
        ):
            for blade in (1, 2, 3):
                selected = sorted(
                    (row for row in pooled if row['stratum'] == stratum and row['signal'] == signal and
                     int(row['blade']) == blade), key=lambda row: float(row['lag_s']))
                ax.plot([float(row['lag_s']) for row in selected],
                        [np.nan if row['r'] is None else float(row['r']) for row in selected],
                        marker='o', linewidth=1.4, label=f'Blade {blade}')
            ax.axhline(0.0, color='#666666', linewidth=0.8)
            ax.set_title(title)
            ax.set_ylim(-1.0, 1.0)
            ax.grid(True, alpha=0.25)
        axes[1, 0].set_xlabel('Lag from action send to My measurement (s)')
        axes[1, 1].set_xlabel('Lag from action send to My measurement (s)')
        axes[0, 0].set_ylabel('Within-slot pooled Pearson r')
        axes[1, 0].set_ylabel('Within-slot pooled Pearson r')
        axes[0, 0].legend(loc='best', fontsize=8)
        axes[0, 1].legend(loc='best', fontsize=8)
        fig.suptitle(f'Frozen Actor trace: action change vs future delta abs(My)\n{stratum}; descriptive, not causal')
        fig.tight_layout()
        filename = f'lagged_change_correlations_{suffix}.png'
        fig.savefig(output_dir / filename, dpi=160)
        plt.close(fig)
        filenames.append(filename)
    return filenames


def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
    lines.extend('| ' + ' | '.join(str(value) for value in row) + ' |' for row in rows)
    return '\n'.join(lines)


def _write_report(
    path: Path,
    *,
    run_id: str,
    source: dict[str, Any],
    slots: dict[int, dict[str, Any]],
    quality_rows: list[dict[str, Any]],
    correlation_rows: list[dict[str, Any]],
    regression_rows: list[dict[str, Any]],
    plot_files: list[str],
) -> None:
    pooled = [row for row in correlation_rows
              if row['grouping'] == 'pooled_within_slot_variance_weighted' and
              row['metric'] == 'action_step_vs_abs_my_change']
    pooled_index = {
        (row['stratum'], row['signal'], int(row['blade']), int(row['lag_frames'])): row['r']
        for row in pooled
    }
    selected_lags = (1, 5, 10, 25, 50, 100)
    metric_tables = []
    for stratum, label in (('all_valid', '全轨迹有效样本'),
                           ('relative_low_load_q25', '本槽相对低载荷 Q25 子集')):
        table_rows = []
        for lag in selected_lags:
            def display(signal: str, blade: int) -> str:
                value = pooled_index.get((stratum, signal, blade, lag))
                return 'NA' if value is None else f'{float(value):+.3f}'
            table_rows.append([
                f'{lag * DT:.2f}',
                *(display('sent_torque_nm', blade) for blade in (1, 2, 3)),
                *(display('sent_pitch_rad', blade) for blade in (1, 2, 3)),
            ])
        metric_tables.append(f'### {label}\n\n' + _md_table(
            ['滞后 s', 'Torque→叶片1', 'Torque→叶片2', 'Torque→叶片3',
             'Pitch→叶片1', 'Pitch→叶片2', 'Pitch→叶片3'], table_rows))

    q_rows = [[
        row['condition_id'], row['identity_status'], row['analysis_valid_frames'],
        row['relative_low_load_frames'], f"{row['relative_low_load_q25_max_abs_my_mnm']:.3f}",
        row['projection_frames'], row['rate_limit_frames'], row['pitch_protection_frames'],
        row['fallback_frames'],
    ] for row in quality_rows]
    n_regression = len(regression_rows)
    regression_tables = []
    for stratum, label in (('all_valid', '全轨迹有效样本'),
                           ('relative_low_load_q25', '本槽相对低载荷 Q25 子集')):
        table_rows = []
        for lag in (1, 5, 10, 25, 50, 100):
            selected = [row for row in regression_rows
                        if row['stratum'] == stratum and row['action_family'] == 'sent_action_change' and
                        row['lag_frames'] == lag]
            values = {}
            for name in ('torque', 'pitch'):
                key = f'beta_{name}_standardized'
                beta_values = [float(row[key]) for row in selected if row[key] is not None]
                if beta_values:
                    positive = sum(value > 0.0 for value in beta_values)
                    negative = sum(value < 0.0 for value in beta_values)
                    values[name] = (float(np.mean(beta_values)), positive, negative, len(beta_values))
                else:
                    values[name] = (math.nan, 0, 0, 0)
            table_rows.append([
                f'{lag * DT:.2f}',
                f'{values["torque"][0]:+.3f}' if values['torque'][3] else 'NA',
                f'{values["torque"][1]}/{values["torque"][2]}',
                f'{values["pitch"][0]:+.3f}' if values['pitch'][3] else 'NA',
                f'{values["pitch"][1]}/{values["pitch"][2]}',
            ])
        regression_tables.append(f'### {label}\n\n' + _md_table(
            ['滞后 s', 'Torque 标准化 β 均值', 'Torque 正/负槽叶片数',
             'Pitch 标准化 β 均值', 'Pitch 正/负槽叶片数'], table_rows))
    lines = [
        f'# {run_id}：冻结 Actor 动作—My 滞后探索性分析',
        '',
        '> **结论级别：仅观察性诊断，不是随机对照或因果效应。** 未注入 torque/pitch 扰动，未分 treatment/control；结果不授权载荷保护、策略验收、soft/hard 或安全保证。',
        '',
        '## 数据和身份审计',
        '',
        f"- 输入逐步 CSV：[bladed_inference_steps.csv](../../bladed_inference_steps.csv)：{source['bytes']:,} bytes；SHA-256 `{source['sha256']}`。",
        f"- 记录类型：`{json.dumps(source['record_type_counts'], ensure_ascii=False, sort_keys=True)}`；控制样本 `{sum(len(slots[e]['columns']['Time']) for e in slots):,}` 行，20 ms 网格。",
        f"- Actor checkpoint SHA-256：`{source['actor_checkpoint_sha256']}`；receipt 证明冻结 Actor/优化关闭，但不是独立核验。",
        f"- 4 个风况的 strict case match：{sum(bool(slots[e]['metadata']['strict_case_matched']) for e in slots)}/4；风速/END identity 仍不严格匹配。",
        '- 不跨 episode 拼接；无效/超时/帧跳跃样本及其跨越窗口不用于相关分析。投影、rate limit、pitch protection 和 fallback 样本保留在主分析中，并单独计数。',
        '',
        '## 时序和统计定义',
        '',
        '- 每条 control row 的 `Time` 是动作计算/发送前读到的当前 UDP 帧；该行 `sent_action_*` 是控制器处理后报告的发送动作，按当前帧发出并作用于后续区间。逐步 CSV 未保存 6 位小数序列化后的 wire 字节，因此用其报告的 sent action 作为近似命令值。',
        '- 相关曲线的滞后 0 是“同一时间戳动作 vs 已观测 My”的非因果水平关联；滞后 0.02 s 起才是动作时间早于 My 测量的配对。方向性动作变化分析只采用正滞后，比较动作差与该滞后末端单个 20 ms 采样间隔的 $|My|$ 增量。',
        '- 方向性回归也只拟合正滞后：该行命令在当前 My 帧读取后才发送，无法解释同一时刻已观测到的 My 增量。',
        '- `actor_delta_*` 仅在 `fallback_active=0` 的 Actor 活跃步参与关联与回归；sent-action 分析保留实际执行的保护、fallback、投影和限幅结果。',
        '- 主响应为每片叶片 $|My|$。Pearson r 仅作描述；池化 r 先在各风况内去均值，再合并协方差。报告不计算逐帧 IID p 值或置信区间。',
        '- “相对低载荷 Q25”是用整槽轨迹事后计算的阈值；分层依据为动作发送前同一控制帧已观测的三叶片最大 $|My|$。它不是在线试验筛选规则或工程安全限值，也不能据此宣称安全。',
        f'- 另生成 {n_regression} 条标准化逐点滞后回归：响应为对应正滞后末端单个 20 ms 采样间隔内的 My 幅值增量，torque/pitch 同时入模，并控制动作原点已观测的 My、风速/转速及其滞后内变化、方位角和 B4 baseline；每个滞后单独拟合，非联合分布滞后模型，不作因果解释。',
        '',
        '## 动作变化与后续载荷变化的池化相关',
        '',
        '以下 r 为“池化的槽内去均值相关”，不是处理效应。选择展示滞后不用于显著性筛选。',
        '',
        *metric_tables,
        '',
        '## 状态调整后的逐点回归（探索性）',
        '',
        '每个滞后单独拟合；正/负计数按 4 个风况 × 3 片叶片统计，不是独立样本数或显著性检验。系数方向不一致时不总结为稳定的动作响应。',
        '',
        *regression_tables,
        '',
        '## 每槽低载荷分布与控制介入计数',
        '',
        _md_table(['风况', '身份', '有效帧', 'Q25 子集帧', 'Q25 max abs(My) MNm',
                   'projection 帧', 'rate-limit 帧', 'pitch-protection 帧', 'fallback 帧'], q_rows),
        '',
        '## 产物',
        '',
        '- [lag_correlations.csv](lag_correlations.csv)：各槽、槽内去均值池化、等权槽均值的 lag 相关。',
        '- [adjusted_lag_regression.csv](adjusted_lag_regression.csv)：每个槽/叶片/滞后/负载分层的逐点标准化回归系数。',
        '- [slot_quality.csv](slot_quality.csv)：数据质量、低载荷经验分位数、动作和保护介入摘要。',
        '- [summary.json](summary.json)：输入 hash、分析口径、汇总指标和授权标志。',
        *[f'- [{filename}]({filename})：按叶片绘制 torque/pitch 动作变化与后续 $\\Delta|My|$ 的滞后曲线。' for filename in plot_files],
        '',
        '## 限制与解释',
        '',
        '- 当前 run 每个风况只有一条冻结 Actor 轨迹；存在共同风况驱动、闭环反馈、My 惯性和自相关。时间先后和回归控制均不能替代随机化。',
        '- 报告中的动作是实际控制器报告的合成 sent action（含 B4、Actor 和现有保护/投影结果），不是独立随机 torque-only/pitch-only 激励。Actor residual 关联也不是扰动效应。',
        '- 003 的 case identity strict match 为 0/4，且输出频率 25.96 Hz 未达到 50 Hz T1。不得重跑该 run；下一步只可据本报告规划独立、经工程审批的新试验。',
        '- **禁止外推：** 不将这些相关系数或回归系数解释为 torque/pitch 对 My 的因果响应，不据此选择物理扰动幅度或解除任何保护。',
        '',
        '授权：`randomized_treatment_present=false`、`causal_effect_estimated=false`、`policy_acceptance=false`、`independent_audit=false`、`soft_authorized=false`、`hard_authorized=false`。',
        '',
    ]
    path.write_text('\n'.join(lines), encoding='utf-8')


def analyze_frozen_round4(run_dir: Path, output_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f'Refusing to overwrite existing diagnostic output: {output_dir}')
    episodes, contract = _read_contract(run_dir)
    slots, input_meta = _read_inference_csv(run_dir, episodes)
    quality_rows = _slot_quality_rows(slots)
    correlation_rows = _correlation_rows(slots)
    regression_rows = _regression_rows(slots)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f'.{output_dir.name}.tmp_', dir=output_dir.parent))
    try:
        _write_csv(temporary / 'slot_quality.csv', quality_rows)
        _write_csv(temporary / 'lag_correlations.csv', correlation_rows)
        _write_csv(temporary / 'adjusted_lag_regression.csv', regression_rows)
        plot_files = _write_plots(temporary, correlation_rows)
        summary = {
            'schema_version': 'frozen_round4_action_my_lag_diagnostic_v1',
            'evidence_role': 'retrospective_observational_lag_association_only',
            'run_id': run_dir.name,
            'input': {
                'path': str(input_meta['path']), 'sha256': input_meta['sha256'],
                'bytes': input_meta['bytes'], 'csv_rows': input_meta['csv_rows'],
                'record_type_counts': input_meta['record_type_counts'],
            },
            'actor_checkpoint_sha256': contract['actor_checkpoint_sha256'],
            'actor_frozen': True,
            'optimizer_updates': False,
            'slot_count': len(slots),
            'control_sample_count': sum(len(slots[e]['columns']['Time']) for e in slots),
            'sample_interval_s': DT,
            'lag_frames': list(LAG_FRAMES),
            'lag_s': [round(frame * DT, 6) for frame in LAG_FRAMES],
            'low_load_quantile': LOW_LOAD_QUANTILE,
            'low_load_definition': 'within-slot q25 of max over three blade abs(My), classified using the payload observed at the action origin before sending; retrospective exploratory cohort only, not a safety limit',
            'strict_case_match_slots': sum(bool(slots[e]['metadata']['strict_case_matched']) for e in slots),
            'randomized_treatment_present': False,
            'causal_effect_estimated': False,
            'policy_acceptance': False,
            'independent_audit': False,
            'soft_authorized': False,
            'hard_authorized': False,
            'slot_quality': quality_rows,
            'outputs': [
                'slot_quality.csv', 'lag_correlations.csv', 'adjusted_lag_regression.csv',
                *plot_files, 'lag_diagnostic_report.md',
            ],
            'limitations': [
                'Four wind slots are single frozen-Actor trajectories; rows are serially correlated and closed-loop actions are endogenous.',
                'The inference CSV stores reported sent actions, not exact serialized wire bytes; wire serialization rounds to six decimals.',
                'A lagged observational association is not a randomized treatment effect.',
                'The within-slot lower-quartile My cohort is not an engineering-approved safety envelope.',
                'Strict PID case identity is not established by the source run.',
            ],
        }
        (temporary / 'summary.json').write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        _write_report(
            temporary / 'lag_diagnostic_report.md', run_id=run_dir.name,
            source=input_meta | {'actor_checkpoint_sha256': contract['actor_checkpoint_sha256']},
            slots=slots, quality_rows=quality_rows, correlation_rows=correlation_rows,
            regression_rows=regression_rows, plot_files=plot_files,
        )
        temporary.rename(output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    summary = analyze_frozen_round4(args.run_dir, args.output_dir)
    print(json.dumps({
        'output_dir': str(args.output_dir.expanduser().resolve()),
        'source_sha256': summary['input']['sha256'],
        'control_sample_count': summary['control_sample_count'],
        'strict_case_match_slots': summary['strict_case_match_slots'],
        'causal_effect_estimated': summary['causal_effect_estimated'],
        'policy_acceptance': summary['policy_acceptance'],
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()