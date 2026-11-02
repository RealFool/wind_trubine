"""Run the pre-registered PID-only physics-first greybox feasibility demo.

One case at a time is read into bounded NumPy arrays.  Fitting, model
selection, empirical-margin calibration and held-out audit are separated.
The output is not a deployable controller or a load-guard authorization.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import resource
import sys
import time
from typing import Any, Callable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from RL.utils.physical_load_margin import (
    METHODS,
    REGIONS,
    GreyboxFitter,
    GreyboxModel,
    PidCase,
    PredictionBatch,
    document_sha256,
    future_labels,
    load_protocol,
    margin_status,
    read_pid_case,
    verify_static_project,
)
from RL.utils.raw_dataset_contract import manifest_sha256, sha256_file, validate_raw_source_root


def save_json(path: Path, value: Any) -> None:
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def portable(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def horizon_key(horizon_s: float) -> str:
    return f'{horizon_s:.2f}'


def prediction_origins(case: PidCase, protocol: dict[str, Any]) -> np.ndarray:
    return np.arange(0, len(case.time) - 1, protocol['origin_stride_frames'], dtype=np.int64)


def freeze_model_selection(model: GreyboxModel, paths: dict[int, Path], protocol: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    candidates = protocol['model']['residual_decay_candidates_s']
    sums = np.zeros(len(candidates), dtype=np.float64)
    counts = np.zeros(len(candidates), dtype=np.int64)
    per_case: dict[str, Any] = {}
    for cid in protocol['split']['validation']:
        case = read_pid_case(paths[cid], cid, protocol)
        origins = prediction_origins(case, protocol)
        labels = future_labels(case, origins, protocol)
        case_metrics = []
        for index, decay in enumerate(candidates):
            predictions = model.predict(case, origins, decay)
            selected = predictions.supported & labels['approximate_zoh']
            error = np.abs(predictions.peaks_nm['physical_greybox'] - labels['peak_nm']) / case.pid_peak_nm
            count = int(selected.sum()) * 3
            total = float(error[selected].sum())
            sums[index] += total
            counts[index] += count
            case_metrics.append({'decay_s': decay, 'blade_origin_horizon_count': count,
                                 'peak_normalized_mae': total / count if count else None})
        per_case[str(cid)] = case_metrics
    if not np.any(counts > 0):
        raise ValueError('Validation has no supported approximate-ZOH origins; selection cannot proceed')
    scores = np.divide(sums, counts, out=np.full_like(sums, np.inf), where=counts > 0)
    best = int(np.argmin(scores))
    return float(candidates[best]), {
        'selection_role': 'validation only; frozen before calibration and audit',
        'candidates': [{'decay_s': value, 'blade_origin_horizon_count': int(counts[i]),
                        'peak_normalized_mae': float(scores[i]) if counts[i] else None}
                       for i, value in enumerate(candidates)],
        'selected_residual_decay_s': float(candidates[best]), 'per_case': per_case,
    }


def calibrate_empirical_margins(model: GreyboxModel, paths: dict[int, Path], protocol: dict[str, Any],
                                decay_s: float) -> dict[str, Any]:
    buffers: dict[tuple[str, int, int], list[np.ndarray]] = {}
    case_counts: dict[tuple[int, int], dict[str, int]] = {}
    for cid in protocol['split']['calibration']:
        case = read_pid_case(paths[cid], cid, protocol)
        origins = prediction_origins(case, protocol)
        predictions = model.predict(case, origins, decay_s)
        labels = future_labels(case, origins, protocol)
        for region_index in range(len(REGIONS)):
            for hi in range(len(protocol['horizons_s'])):
                mask = predictions.supported[:, hi] & labels['approximate_zoh'][:, hi] & (case.region[origins] == region_index)
                if not mask.any():
                    continue
                case_counts.setdefault((region_index, hi), {})[str(cid)] = int(mask.sum())
                for method in METHODS:
                    error = np.maximum(labels['peak_nm'][mask, hi] - predictions.peaks_nm[method][mask, hi], 0.0)
                    buffers.setdefault((method, region_index, hi), []).append(error)
    result: dict[str, Any] = {
        'role': 'calibration only; positive-part one-sided empirical residual quantiles in Nm',
        'not_a_deterministic_bound': True, 'origins_are_overlapping_not_independent': True,
        'plan_qualification': 'approximate-ZOH recorded PID only; no unexecuted candidate calibration',
        'groups': {},
    }
    for region_index, region_name in enumerate(REGIONS):
        for hi, horizon in enumerate(protocol['horizons_s']):
            key = f'{region_name}|{horizon_key(horizon)}'
            group: dict[str, Any] = {'region': region_name, 'horizon_s': horizon,
                                     'case_origin_counts': case_counts.get((region_index, hi), {}), 'methods': {}}
            for method in METHODS:
                parts = buffers.get((method, region_index, hi), [])
                errors = np.concatenate(parts, axis=0) if parts else np.empty((0, 3))
                eligible = len(errors) >= protocol['evaluation']['minimum_calibration_origins_per_region_horizon']
                group['methods'][method] = {
                    'origin_count': len(errors), 'supported': eligible,
                    'error_nm': np.quantile(errors, protocol['evaluation']['empirical_error_quantile'], axis=0,
                                            method='higher').tolist() if eligible else None,
                }
            result['groups'][key] = group
    return result


def margin_for(calibration: dict[str, Any], method: str, region: int, horizon: float) -> np.ndarray | None:
    if region < 0:
        return None
    entry = calibration['groups'][f'{REGIONS[region]}|{horizon_key(horizon)}']['methods'][method]
    return np.asarray(entry['error_nm'], dtype=np.float64) if entry['supported'] else None


def point_metrics(parts: dict[str, list[np.ndarray]], protocol: dict[str, Any]) -> dict[str, Any]:
    if not parts.get('actual'):
        return {'origin_count': 0}
    actual = np.concatenate(parts['actual'], axis=0)
    predicted = np.concatenate(parts['predicted'], axis=0)
    reference = np.concatenate(parts['reference'], axis=0)
    margins = np.concatenate(parts['margin'], axis=0)
    signed_error = actual - predicted
    under = np.maximum(signed_error, 0.0)
    absolute = np.abs(signed_error)
    result: dict[str, Any] = {
        'origin_count': len(actual), 'blade_sample_count': int(actual.size),
        'mae_nm': float(absolute.mean()), 'normalized_mae': float((absolute / reference).mean()),
        'per_blade_mae_nm': absolute.mean(axis=0).tolist(),
        'underestimate_p95_nm': np.quantile(under, 0.95, axis=0).tolist(),
        'underestimate_p99_nm': np.quantile(under, 0.99, axis=0).tolist(),
        'maximum_underestimate_nm': np.max(under, axis=0).tolist(),
    }
    usable_margin = np.isfinite(margins).all(axis=1)
    upper = predicted + margins
    coverage = actual <= upper
    result['empirical_upper'] = {
        'origin_count': int(usable_margin.sum()),
        'per_blade_coverage': coverage[usable_margin].mean(axis=0).tolist() if usable_margin.any() else None,
        'simultaneous_three_blade_coverage': float(coverage[usable_margin].all(axis=1).mean()) if usable_margin.any() else None,
    }
    result['near_peak'] = {}
    for ratio in protocol['evaluation']['near_peak_ratios']:
        positive = actual >= ratio * reference
        actual_positive = int(positive.sum())
        alert = predicted >= ratio * reference
        true_positive = int((alert & positive).sum())
        negative = ~positive
        near_under = under[positive]
        calibrated_positive = positive & usable_margin[:, None]
        calibrated_negative = negative & usable_margin[:, None]
        upper_alert = upper >= ratio * reference
        result['near_peak'][str(ratio)] = {
            'positive_blade_origins': actual_positive,
            'point_recall': true_positive / actual_positive if actual_positive else None,
            'near_peak_underestimate_p95_nm': float(np.quantile(near_under, 0.95)) if len(near_under) else None,
            'point_false_positive_fraction': float((alert & negative).sum() / negative.sum()) if negative.any() else None,
            'upper_recall': float((upper_alert & calibrated_positive).sum() / calibrated_positive.sum()) if calibrated_positive.any() else None,
            'upper_false_positive_fraction': float((upper_alert & calibrated_negative).sum() / calibrated_negative.sum()) if calibrated_negative.any() else None,
        }
    result['model_upper_crosses_pid_reference_fraction'] = float(np.any(upper[usable_margin] > reference[usable_margin], axis=1).mean()) if usable_margin.any() else None
    return result


def load_episodes(times: np.ndarray, absolute_load: np.ndarray, threshold: float,
                  merge_gap_s: float) -> list[tuple[float, float]]:
    indices = np.flatnonzero(absolute_load >= threshold)
    if not len(indices):
        return []
    breaks = np.flatnonzero(np.diff(times[indices]) > merge_gap_s + 1e-9) + 1
    return [(float(times[group[0]]), float(times[group[-1]])) for group in np.split(indices, breaks)]


def episode_metrics(case: PidCase, predictions: PredictionBatch, calibration: dict[str, Any],
                    method: str, hi: int, ratio: float, protocol: dict[str, Any]) -> dict[str, Any]:
    horizon = protocol['horizons_s'][hi]
    times = case.time[predictions.origins]
    result: dict[str, Any] = {'total_events': 0, 'opportunity_events': 0, 'timely_events': 0,
                              'late_events': 0, 'no_model_opportunity_events': 0, 'lead_times_s': []}
    for blade in range(3):
        threshold = ratio * case.pid_peak_nm[blade]
        upper_alert = np.zeros(len(times), dtype=bool)
        usable = np.zeros(len(times), dtype=bool)
        for region_index in range(len(REGIONS)):
            selected = (case.region[predictions.origins] == region_index) & predictions.supported[:, hi]
            margin = margin_for(calibration, method, region_index, horizon)
            if margin is None:
                continue
            usable |= selected
            upper_alert[selected] = predictions.peaks_nm[method][selected, hi, blade] + margin[blade] >= threshold
        episodes = load_episodes(case.time, np.abs(case.my_nm[:, blade]), threshold,
                                 protocol['evaluation']['event_merge_gap_s'])
        for onset, end in episodes:
            result['total_events'] += 1
            opportunities = (times >= onset - horizon - 1e-9) & (times < onset - 1e-9) & usable
            if opportunities.any():
                result['opportunity_events'] += 1
            else:
                result['no_model_opportunity_events'] += 1
            early = opportunities & upper_alert
            if early.any():
                result['timely_events'] += 1
                result['lead_times_s'].append(float(onset - times[early][0]))
            elif np.any((times >= onset - 1e-9) & (times <= end + 1e-9) & upper_alert):
                result['late_events'] += 1
    return result


def summarize_episodes(items: list[dict[str, Any]]) -> dict[str, Any]:
    result = {name: sum(item[name] for item in items) for name in (
        'total_events', 'opportunity_events', 'timely_events', 'late_events', 'no_model_opportunity_events',
    )}
    leads = [value for item in items for value in item['lead_times_s']]
    result['timely_recall_all_events'] = result['timely_events'] / result['total_events'] if result['total_events'] else None
    result['timely_recall_with_model_opportunity'] = result['timely_events'] / result['opportunity_events'] if result['opportunity_events'] else None
    result['median_lead_time_s'] = float(np.median(leads)) if leads else None
    result['minimum_lead_time_s'] = float(np.min(leads)) if leads else None
    result['qualification'] = 'all recorded PID events, not counterfactual action validation; events and origins may be dependent'
    return result


def append_parts(target: dict[str, list[np.ndarray]], actual: np.ndarray, predicted: np.ndarray,
                 reference: np.ndarray, margin: np.ndarray) -> None:
    for name, values in (('actual', actual), ('predicted', predicted), ('reference', reference), ('margin', margin)):
        target.setdefault(name, []).append(values)


def sensitivity_records(model: GreyboxModel, case: PidCase, calibration: dict[str, Any],
                        decay_s: float, protocol: dict[str, Any]) -> list[dict[str, Any]]:
    """Deliberately no call to future_labels(): unexecuted actions have no truth."""
    spec = protocol['evaluation']['candidate_sensitivity']
    records = []
    for timestamp in spec['audit_times_s']:
        origin = int(np.argmin(np.abs(case.time - timestamp)))
        if abs(case.time[origin] - timestamp) > 1e-6:
            continue
        if not case.valid[origin] or not np.isfinite(case.actions[origin]).all():
            records.append({'condition_id': case.condition_id, 'time_s': float(case.time[origin]),
                            'candidate_safety_status': 'unknown_invalid_input', 'skip_reason': 'invalid_origin_no_candidate_constructed',
                            'action': None, 'predicted_peak_nm': None, 'has_physical_counterfactual_label': False})
            continue
        demand = case.actions[origin].copy()
        deltas = ((0.0, 0.0), (spec['torque_delta_nm'], 0.0), (-spec['torque_delta_nm'], 0.0),
                  (0.0, spec['pitch_delta_rad']), (0.0, -spec['pitch_delta_rad']))
        names = ('pid_zoh', 'torque_plus', 'torque_minus', 'pitch_plus', 'pitch_minus')
        candidates = np.asarray([demand + delta for delta in deltas])
        candidates = np.clip(candidates, [0.0, 0.0], [protocol['static_parameters']['request_torque_max_nm'], 0.43])
        batch = model.predict(case, np.full(len(candidates), origin), decay_s, candidates)
        for ci, name in enumerate(names):
            for hi, horizon in enumerate(protocol['horizons_s']):
                margin = margin_for(calibration, 'physical_greybox', int(case.region[origin]), horizon)
                state, headroom = margin_status(case.pid_peak_nm, case.cumulative_peak_nm[origin],
                                               batch.peaks_nm['physical_greybox'][ci, hi], margin,
                                               bool(batch.supported[ci, hi]))
                records.append({
                    'condition_id': case.condition_id, 'time_s': float(case.time[origin]), 'candidate': name,
                    'action': candidates[ci].tolist(), 'horizon_s': horizon,
                    'model_supported': bool(batch.supported[ci, hi]), 'support_reason': str(batch.reasons[ci, hi]),
                    'predicted_peak_nm': batch.peaks_nm['physical_greybox'][ci, hi].tolist()
                    if batch.supported[ci, hi] and np.isfinite(batch.peaks_nm['physical_greybox'][ci, hi]).all() else None,
                    'delta_predicted_peak_nm_vs_pid_zoh': (batch.peaks_nm['physical_greybox'][ci, hi] - batch.peaks_nm['physical_greybox'][0, hi]).tolist()
                    if batch.supported[ci, hi] and batch.supported[0, hi] else None,
                    'margin_state': state, 'empirical_headroom_nm': headroom.tolist() if headroom is not None else None,
                    'margin_qualification': 'PID empirical margin reused for sensitivity illustration only; alternative-action coverage is unknown',
                    'candidate_safety_status': 'unknown_unexecuted_action' if ci else 'PID_ZOH_diagnostic_only',
                    'has_physical_counterfactual_label': False,
                })
    return records


def evaluate_audit(model: GreyboxModel, paths: dict[int, Path], protocol: dict[str, Any],
                   calibration: dict[str, Any], decay_s: float, output: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    buffers: dict[tuple[str, str, int, str], dict[str, list[np.ndarray]]] = {}
    episodes: dict[tuple[str, int, float], list[dict[str, Any]]] = {}
    support_counts = np.zeros((len(protocol['horizons_s']), 3), dtype=np.int64)
    response_parts: dict[int, list[np.ndarray]] = {}
    candidate_results = []
    per_case: dict[str, Any] = {}
    plot_data: dict[str, Any] = {}
    with (output / 'audit_samples.csv').open('x', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(('condition_id', 'time_s', 'horizon_s', 'blade', 'region', 'model_supported',
                         'support_reason', 'future_label_valid', 'normalized_plan_mismatch', 'approximate_zoh',
                         'actual_future_peak_nm', *[f'{method}_peak_nm' for method in METHODS],
                         'physical_empirical_error_nm', 'physical_empirical_upper_nm', 'pid_reference_peak_nm',
                         'cumulative_peak_nm', 'empirical_headroom_nm', 'margin_state'))
        for cid in protocol['split']['audit']:
            print(f'audit wind{cid}', flush=True)
            case = read_pid_case(paths[cid], cid, protocol)
            origins = prediction_origins(case, protocol)
            predictions = model.predict(case, origins, decay_s)
            labels = future_labels(case, origins, protocol)
            case_record: dict[str, Any] = {'source': case.audit, 'horizons': {}}
            candidate_results.extend(sensitivity_records(model, case, calibration, decay_s, protocol))
            for hi, horizon in enumerate(protocol['horizons_s']):
                valid = labels['valid'][:, hi]
                known = predictions.supported[:, hi] & valid
                matched = known & labels['approximate_zoh'][:, hi]
                support_counts[hi] += [int(valid.sum()), int(known.sum()), int(matched.sum())]
                case_hi: dict[str, Any] = {
                    'total_origins': len(origins), 'label_valid_origins': int(valid.sum()),
                    'model_known_origins': int(known.sum()), 'approximate_zoh_known_origins': int(matched.sum()),
                    'support_reason_counts': dict(Counter(str(value) for value in predictions.reasons[:, hi])),
                    'plan_mismatch_p95': float(np.quantile(labels['plan_mismatch'][valid, hi], 0.95)) if valid.any() else None,
                    'metrics': {},
                }
                if matched.any():
                    error = np.abs(predictions.actual_response_prediction[matched, hi] - labels['response'][matched, hi])
                    response_parts.setdefault(hi, []).append(error)
                    case_hi['actual_response_mae'] = {'torque_nm': float(error[:, 0].mean()), 'pitch_rad': float(error[:, 1].mean())}
                static_known = known & np.isfinite(predictions.static_current_my_nm).all(axis=1)
                case_hi['unanchored_static_current_mae_nm'] = float(np.abs(predictions.static_current_my_nm[static_known] - case.my_nm[origins[static_known]]).mean()) if static_known.any() else None
                for method in METHODS:
                    margin_array = np.full((len(origins), 3), np.nan)
                    for region_index in range(len(REGIONS)):
                        margin = margin_for(calibration, method, region_index, horizon)
                        if margin is not None:
                            margin_array[case.region[origins] == region_index] = margin
                    case_metrics: dict[str, Any] = {}
                    for role, selection in (('all_pid', known), ('approximate_zoh', matched)):
                        for region_index, region_name in [(-1, 'all'), *enumerate(REGIONS)]:
                            mask = selection if region_index < 0 else selection & (case.region[origins] == region_index)
                            if not mask.any():
                                continue
                            parts: dict[str, list[np.ndarray]] = {}
                            references = np.broadcast_to(case.pid_peak_nm, (int(mask.sum()), 3)).copy()
                            append_parts(parts, labels['peak_nm'][mask, hi], predictions.peaks_nm[method][mask, hi], references, margin_array[mask])
                            if region_index < 0:
                                case_metrics[role] = point_metrics(parts, protocol)
                            pooled = buffers.setdefault((method, role, hi, region_name), {})
                            append_parts(pooled, labels['peak_nm'][mask, hi], predictions.peaks_nm[method][mask, hi], references, margin_array[mask])
                    case_hi['metrics'][method] = case_metrics
                    for ratio in protocol['evaluation']['near_peak_ratios']:
                        event_result = episode_metrics(case, predictions, calibration, method, hi, ratio, protocol)
                        episodes.setdefault((method, hi, ratio), []).append(event_result)
                for oi, origin in enumerate(origins):
                    region_index = int(case.region[origin])
                    error_margin = margin_for(calibration, 'physical_greybox', region_index, horizon)
                    status, headroom = margin_status(case.pid_peak_nm, case.cumulative_peak_nm[origin],
                                                    predictions.peaks_nm['physical_greybox'][oi, hi], error_margin,
                                                    bool(predictions.supported[oi, hi]))
                    for blade in range(3):
                        writer.writerow((cid, f'{case.time[origin]:.8f}', horizon, blade + 1,
                                         REGIONS[region_index] if region_index >= 0 else 'unsupported',
                                         int(predictions.supported[oi, hi]), predictions.reasons[oi, hi], int(valid[oi]),
                                         labels['plan_mismatch'][oi, hi], int(labels['approximate_zoh'][oi, hi]),
                                         labels['peak_nm'][oi, hi, blade] if valid[oi] else '',
                                         *[predictions.peaks_nm[method][oi, hi, blade] if predictions.supported[oi, hi] else '' for method in METHODS],
                                         error_margin[blade] if error_margin is not None else '',
                                         predictions.peaks_nm['physical_greybox'][oi, hi, blade] + error_margin[blade]
                                         if error_margin is not None and predictions.supported[oi, hi] else '',
                                         case.pid_peak_nm[blade], case.cumulative_peak_nm[origin, blade],
                                         headroom[blade] if headroom is not None else '', status))
                case_record['horizons'][horizon_key(horizon)] = case_hi
            per_case[str(cid)] = case_record
            if cid == protocol['split']['audit'][0]:
                peak_time = case.time[int(np.argmax(np.abs(case.my_nm[:, 0])))]
                keep = (case.time[origins] >= peak_time - 4.0) & (case.time[origins] <= peak_time + 4.0)
                hi = 1
                upper = np.full(int(keep.sum()), np.nan)
                for pi, oi in enumerate(np.flatnonzero(keep)):
                    margin = margin_for(calibration, 'physical_greybox', int(case.region[origins[oi]]), protocol['horizons_s'][hi])
                    if margin is not None and predictions.supported[oi, hi]:
                        upper[pi] = predictions.peaks_nm['physical_greybox'][oi, hi, 0] + margin[0]
                plot_data = {'time': case.time[origins][keep], 'actual': labels['peak_nm'][keep, hi, 0],
                             'physical': predictions.peaks_nm['physical_greybox'][keep, hi, 0],
                             'trend': predictions.peaks_nm['causal_trend'][keep, hi, 0],
                             'upper': upper, 'reference': float(case.pid_peak_nm[0]), 'condition_id': cid,
                             'horizon': protocol['horizons_s'][hi]}
    summary: dict[str, Any] = {'role': 'held-out audit; never used to select model or calibrate margins',
                               'per_case': per_case, 'support': {}, 'metrics': {}, 'events': {}, 'response': {},
                               'strict_pid_exceedance': {'positive_events': 0, 'recall': None,
                                                         'reason': 'each PID trajectory defines its own complete-window maximum'}}
    for hi, horizon in enumerate(protocol['horizons_s']):
        label_count, known_count, match_count = (int(value) for value in support_counts[hi])
        summary['support'][horizon_key(horizon)] = {
            'label_valid_origins': label_count, 'known_origins': known_count, 'approximate_zoh_known_origins': match_count,
            'known_fraction': known_count / label_count if label_count else None,
            'matched_fraction_of_known': match_count / known_count if known_count else None,
        }
        if hi in response_parts:
            error = np.concatenate(response_parts[hi], axis=0)
            summary['response'][horizon_key(horizon)] = {'approximate_zoh_origin_count': len(error),
                                                        'torque_mae_nm': float(error[:, 0].mean()), 'pitch_mae_rad': float(error[:, 1].mean())}
        for method in METHODS:
            for role in ('all_pid', 'approximate_zoh'):
                for region_name in ('all', *REGIONS):
                    summary['metrics'][f'{method}|{role}|{horizon_key(horizon)}|{region_name}'] = point_metrics(
                        buffers.get((method, role, hi, region_name), {}), protocol)
            for ratio in protocol['evaluation']['near_peak_ratios']:
                summary['events'][f'{method}|{horizon_key(horizon)}|{ratio}'] = summarize_episodes(episodes[(method, hi, ratio)])
    return summary, plot_data, candidate_results


def feasibility_gate(audit: dict[str, Any], calibration: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    gates = protocol['feasibility_gate']
    checks = []

    def check(name: str, value: float | None, threshold: float, compare: Callable[[float, float], bool], scope: str) -> None:
        checks.append({'name': name, 'scope': scope, 'value': value, 'threshold': threshold,
                       'status': 'INCONCLUSIVE' if value is None else ('PASS' if compare(value, threshold) else 'FAIL')})

    for horizon in protocol['online_scope_horizons_s']:
        hs = horizon_key(horizon)
        check('known_origin_fraction', audit['support'][hs]['known_fraction'], gates['minimum_known_origin_fraction'],
              lambda x, y: x >= y, hs)
        physical = audit['metrics'][f'physical_greybox|approximate_zoh|{hs}|all'].get('normalized_mae')
        baselines = [audit['metrics'][f'{method}|approximate_zoh|{hs}|all'].get('normalized_mae')
                     for method in ('persistence', 'causal_trend')]
        valid_baselines = [value for value in baselines if value is not None and value > 0.0]
        ratio = physical / min(valid_baselines) if physical is not None and valid_baselines else None
        check('mae_vs_best_non_nn_baseline', ratio, gates['maximum_mae_ratio_to_best_non_nn_baseline'],
              lambda x, y: x <= y, hs)
        for region in REGIONS:
            metrics = audit['metrics'][f'physical_greybox|approximate_zoh|{hs}|{region}']
            calibrated = calibration['groups'][f'{region}|{hs}']['methods']['physical_greybox']['supported']
            upper = metrics.get('empirical_upper', {})
            enough = upper.get('origin_count', 0) >= protocol['evaluation']['minimum_audit_origins_per_region_horizon']
            coverage = upper.get('simultaneous_three_blade_coverage') if calibrated and enough else None
            check('simultaneous_three_blade_empirical_coverage', coverage,
                  gates['minimum_empirical_upper_coverage_per_supported_group'], lambda x, y: x >= y, f'{region}|{hs}')
        event = audit['events'][f'physical_greybox|{hs}|{protocol["evaluation"]["primary_near_peak_ratio"]}']
        enough_events = event['total_events'] >= gates['minimum_near_peak_events_per_online_horizon']
        check('timely_near_peak_event_recall', event['timely_recall_all_events'] if enough_events else None,
              gates['minimum_timely_event_recall'], lambda x, y: x >= y, hs)
        fp = audit['metrics'][f'physical_greybox|approximate_zoh|{hs}|all'].get('near_peak', {}).get(
            str(protocol['evaluation']['primary_near_peak_ratio']), {}).get('upper_false_positive_fraction')
        check('upper_near_peak_false_positive_fraction', fp, gates['maximum_false_positive_fraction'], lambda x, y: x <= y, hs)
    passed = all(item['status'] == 'PASS' for item in checks)
    return {'decision': 'PASS_OFFLINE_FEASIBILITY' if passed else 'REVIEW_DO_NOT_INTEGRATE', 'checks': checks,
            'authorization': dict(protocol['authorization']),
            'reason': 'pre-registered held-out tests only; passing would still not validate RL alternative actions'}


def make_plot(audit: dict[str, Any], plot_data: dict[str, Any], protocol: dict[str, Any], output: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.1), constrained_layout=True)
    positions = np.arange(len(protocol['horizons_s']))
    for mi, method in enumerate(METHODS):
        values = [audit['metrics'][f'{method}|approximate_zoh|{horizon_key(h)}|all'].get('normalized_mae', np.nan)
                  for h in protocol['horizons_s']]
        axes[0].bar(positions + (mi - 1.5) * 0.18, values, width=0.18, label=method)
    axes[0].set_xticks(positions, [f'{h:.2f}s' for h in protocol['horizons_s']])
    axes[0].set_ylabel('MAE / same-case PID peak')
    axes[0].set_title('Held-out PID audit: approximate ZOH')
    axes[0].legend(fontsize=8)
    if plot_data:
        for name, label, style in (('actual', 'actual future peak', '-'), ('physical', 'greybox', '-'),
                                   ('trend', 'causal trend', '--'), ('upper', 'empirical upper', ':')):
            axes[1].plot(plot_data['time'], plot_data[name] / 1e6, style, label=label)
        axes[1].axhline(plot_data['reference'] / 1e6, color='black', alpha=0.6, label='PID 600s maximum')
        axes[1].set_xlabel('Origin time (s)')
        axes[1].set_ylabel('Blade 1 |My| (MNm)')
        axes[1].set_title(f'wind{plot_data["condition_id"]}, H={plot_data["horizon"]:.2f}s near peak')
        axes[1].legend(fontsize=8)
    fig.savefig(output / 'diagnostic.png', dpi=160)
    plt.close(fig)


def report_markdown(report: dict[str, Any]) -> str:
    protocol = report['protocol']
    audit = report['audit']
    lines = [
        '# 物理优先载荷裕量评估器：PID 离线 Demo v1 结果', '',
        f'- 冻结协议：SHA-256 `{report["protocol_sha256"]}`。',
        f'- 原始数据：{len(report["sources"])} 个 PID CSV，合计 {sum(item["rows"] for item in report["sources"]):,} 行；'
        f'逐文件读取，准确窗口 {protocol["reference_window_s"]} s。',
        f'- 数据清单 SHA-256：`{report["input_manifest_sha256"]}`。',
        '- 协议：[RL/configs/physical_load_margin_demo_v1.json](RL/configs/physical_load_margin_demo_v1.json)。',
        '- 脚本：[RL/demo_physical_load_margin.py](RL/demo_physical_load_margin.py)。',
        '- 模型不是 BEM 或完整气弹性求解器；是量纲化准稳态曲面＋执行器/转子运动＋实测 My 锚定＋因果残差趋势。无神经网络。',
        '- 不增加 UDP/传感器/Actor 状态变量；仅从现有白名单生成内部导数、相位、累计峰值等。',
        f'- 动作来源限制：{protocol["recorded_action_provenance"]}。',
        f'- 结论：**{report["gate"]["decision"]}**；`policy_acceptance=false`，soft/hard 未授权。', '',
        '## 1. 冻结拆分和模型选择', '',
    ]
    for role, ids in protocol['split'].items():
        lines.append(f'- {role}：wind {ids}，整文件隔离。')
    lines += [
        f'- 仅 validation 选择残差衰减时间：{report["selection"]["selected_residual_decay_s"]:.2f} s；'
        'calibration 只估计经验单侧误差余量；audit 不参与选择和余量估计。',
        '- 600 s PID 最大值仅作为预冻结同窗评价参考，不进入拟合特征，不调低为近峰阈值。',
        '- 当前一条风轨/工况；相邻 origins 和相邻叶片事件相关，不能冒充独立重复 seed 或独立覆盖样本。', '',
        '## 2. 独立 Audit 点预测', '',
        '下表在所有方法相同的模型支持域与近似 ZOH origin 上比较；0.50 s 仅诊断。', '',
        '| H | 已知比例 | 近似 ZOH / 已知 | 灰箱 NMAE | 仅物理增量 NMAE | 持续值 NMAE | 因果趋势 NMAE |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for horizon in protocol['horizons_s']:
        hs = horizon_key(horizon)
        support = audit['support'][hs]
        values = [audit['metrics'][f'{method}|approximate_zoh|{hs}|all'].get('normalized_mae') for method in METHODS]
        cells = [f'{horizon:.2f}s', f'{support["known_fraction"]:.2%}', f'{support["matched_fraction_of_known"]:.2%}']
        cells += [f'{value:.6f}' if value is not None else '不可估计' for value in values]
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines += ['', 'NMAE 为每片绝对误差除以对应 case/600 s PID 峰值后取平均，不是材料利用率。', '',
              '### 执行器响应诊断（录制 demand 来源假设成立时）', '',
              '| H | 转矩 MAE (Nm) | 实际桨距 MAE (rad) |', '|---|---:|---:|']
    for hs, response in audit['response'].items():
        lines.append(f'| {hs}s | {response["torque_mae_nm"]:.2f} | {response["pitch_mae_rad"]:.6g} |')
    lines += ['', '## 3. 经验余量覆盖和近峰事件', '',
              '- 误差余量是 calibration 上正部低估残差的逐片 95% 分位数，不是确定性上界。',
              '- 门禁核查三片同时覆盖；逐片各 95% 不自动等于三片联合 95%。',
              '- 近峰用 0.90/0.95 倍 PID 极值，正式极限仍为 1.00；严格超 PID 正例为 0，recall **不可估计**。', '',
              '| H | 区域 | 校准 origin | audit origin | 三片同时经验覆盖 |', '|---|---|---:|---:|---:|']
    for horizon in protocol['horizons_s']:
        hs = horizon_key(horizon)
        for region in REGIONS:
            cal = report['calibration']['groups'][f'{region}|{hs}']['methods']['physical_greybox']
            metrics = audit['metrics'][f'physical_greybox|approximate_zoh|{hs}|{region}']
            upper = metrics.get('empirical_upper', {})
            coverage = upper.get('simultaneous_three_blade_coverage')
            lines.append(f'| {hs}s | {region} | {cal["origin_count"]} | {metrics["origin_count"]} | '
                         + (f'{coverage:.2%}' if coverage is not None else '不可估计') + ' |')
    lines += ['', '| H | 0.95 近峰事件 | 按时预警 | 晚报 | 无支持机会 | 全事件按时 recall | 中位提前量 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for horizon in protocol['horizons_s']:
        hs = horizon_key(horizon)
        event = audit['events'][f'physical_greybox|{hs}|0.95']
        recall = event['timely_recall_all_events']
        lead = event['median_lead_time_s']
        lines.append(f'| {hs}s | {event["total_events"]} | {event["timely_events"]} | {event["late_events"]} | '
                     f'{event["no_model_opportunity_events"]} | ' + (f'{recall:.2%}' if recall is not None else '不可估计')
                     + ' | ' + (f'{lead:.3f}s' if lead is not None else '—') + ' |')
    lines += ['', '按时指事件起点前 H 内已有预警；事件开始后报出不能计为预警成功。'
              '事件表覆盖全部 PID 近峰片段，未来 PID 计划不一定保持，是运行诊断而非严格候选因果验证。', '',
              '## 4. 预注册继续实施门禁', '', '| 检查 | 范围 | 实测 | 门槛 | 状态 |', '|---|---|---:|---:|---|']
    for check in report['gate']['checks']:
        value = f'{check["value"]:.6f}' if check['value'] is not None else '不可估计'
        scope = check['scope'].replace('|', ' / ')
        lines.append(f'| {check["name"]} | {scope} | {value} | {check["threshold"]} | {check["status"]} |')
    lines += ['', '未通过或缺组时停止生产接线；不得利用本次 audit 调参后仍称其独立验证。'
              '即便全部通过，也只支持 PID 离线可行性，不允许 soft/hard。', '',
              '## 5. 模型敏感性与尚未证明的内容', '',
              '- audit 固定时点测试 ±20 kNm、±0.00035 rad 等候选，只报告模型响应与裕量变化。',
              '- 未执行动作没有真值；不把 PID 后续 My 复制成候选标签，不计算候选降峰收益/反事实准确率。',
              '- 平均风速不能确定逐叶片局部风、诱导、模态运动；未来 wind/yaw/加速度保持只是短窗假设。',
              '- 示例工程常量绑定了哈希及数值，但不能证明全部历史 case 使用完全相同工程。'
              '0.67R 是明确的等效半径近似；未装载外部 Ct/Cp/Cq 曲面。',
              '- demand 不能当实际执行器状态；实际响应单独诊断，当前 DLL 的 O_* 测量别名不能用于生产候选计划。', '',
              '## 6. 工件与资源', '',
              f'- 墙钟耗时 {report["resources"]["elapsed_s"]:.2f} s；Linux 进程峰值 RSS '
              f'{report["resources"]["peak_rss_mib"]:.2f} MiB；非目标 Windows 硬件 T1。',
              '- 输出：机器可读报告、冻结模型/协议/选择/校准、逐 origin audit CSV、候选敏感性 JSON、图。',
              '- 未启动 Bladed/UDP，未更改 Actor/BC/TCN v7/生产 guard 或既有变量合同。', '']
    return '\n'.join(lines)


def run_demo(data_root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    started = time.perf_counter()
    protocol = load_protocol(protocol_path)
    root, sources = validate_raw_source_root(data_root)
    if output.exists():
        raise FileExistsError('Output directory already exists; use a new isolated output to preserve audit identity')
    output.mkdir(parents=True)
    static = verify_static_project(PROJECT_ROOT / protocol['static_parameters']['source_project'], protocol)
    save_json(output / 'frozen_protocol.json', protocol)
    paths = dict(sources)
    audits = []
    fitter = GreyboxFitter(protocol)
    for cid, path in sources:
        print(f'input audit wind{cid}' + (' + coefficient fit' if cid in protocol['split']['train'] else ''), flush=True)
        case = read_pid_case(path, cid, protocol)
        audits.append(case.audit)
        if cid in protocol['split']['train']:
            fitter.add_case(case)
    del case
    model = fitter.finish()
    save_json(output / 'input_audit.json', {'sources': audits, 'input_manifest_sha256': manifest_sha256(audits)})
    decay_s, selection = freeze_model_selection(model, paths, protocol)
    save_json(output / 'frozen_selection.json', selection)
    model_document = model.to_document()
    model_document['selected_residual_decay_s'] = decay_s
    model_document['input_manifest_sha256'] = manifest_sha256(audits)
    model_document['static_project'] = static
    save_json(output / 'frozen_model.json', model_document)
    calibration = calibrate_empirical_margins(model, paths, protocol, decay_s)
    calibration['frozen_model_sha256'] = document_sha256(model_document)
    save_json(output / 'frozen_calibration.json', calibration)
    audit, plot_data, candidates = evaluate_audit(model, paths, protocol, calibration, decay_s, output)
    save_json(output / 'candidate_sensitivity.json', {'records': candidates,
              'qualification': protocol['evaluation']['candidate_sensitivity']['qualification'],
              'has_physical_counterfactual_labels': False})
    gate = feasibility_gate(audit, calibration, protocol)
    make_plot(audit, plot_data, protocol, output)
    report = {
        'schema_version': 'physical_load_margin_demo_report_v1', 'protocol': protocol,
        'protocol_sha256': document_sha256(protocol), 'protocol_file_sha256': sha256_file(protocol_path),
        'data_root': portable(root), 'sources': audits, 'input_manifest_sha256': manifest_sha256(audits),
        'code_sha256': {'runner': sha256_file(Path(__file__)), 'model_module': sha256_file(PROJECT_ROOT / 'RL/utils/physical_load_margin.py')},
        'static_project': static, 'model_sha256': document_sha256(model_document),
        'selection': selection, 'calibration': calibration, 'audit': audit, 'gate': gate,
        'resources': {'elapsed_s': time.perf_counter() - started,
                      'peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
                      'qualification': 'Linux offline process including report/figure, not production per-step latency'},
        'authorization': dict(protocol['authorization']),
    }
    save_json(output / 'report.json', report)
    with (output / 'README.md').open('x', encoding='utf-8') as handle:
        handle.write(report_markdown(report))
    print(json.dumps({'decision': gate['decision'], 'output': portable(output),
                      'elapsed_s': report['resources']['elapsed_s'], 'peak_rss_mib': report['resources']['peak_rss_mib']}, ensure_ascii=False), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=PROJECT_ROOT / 'bladed-data-new/raw')
    parser.add_argument('--protocol', type=Path, default=PROJECT_ROOT / 'RL/configs/physical_load_margin_demo_v1.json')
    parser.add_argument('--output-dir', type=Path, default=PROJECT_ROOT / 'RL/diagnostics/physical_load_margin_demo_v1')
    args = parser.parse_args()
    run_demo(args.data_root.resolve(), args.protocol.resolve(), args.output_dir.resolve())


if __name__ == '__main__':
    main()