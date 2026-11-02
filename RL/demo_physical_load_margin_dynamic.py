"""D0-R: non-NN dynamic increments, with old audit blocked before access.

Train fits physics/FIR/error-scale; validation selects the finite candidate
grid; calibration fixes joint empirical margins. Post-selection validation
coverage is explicitly reused development evidence, NOT independent audit.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path
import resource
import sys
import time
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from RL.demo_physical_load_margin import append_parts, horizon_key, load_episodes, point_metrics, portable, save_json
from RL.utils.physical_load_margin import REGIONS, GreyboxFitter, PidCase, document_sha256, verify_static_project
from RL.utils.physical_load_margin_dynamic import (
    AUTHORIZATION, MARGIN_METHODS, POINT_METHODS, DynamicFitter,
    DynamicModel, ErrorScaleFitter, ErrorScaleModel, PreparedCase,
    joint_margin_arrays, load_development_protocol, make_joint_calibration,
    prepare_case, read_development_case,
)
from RL.utils.raw_dataset_contract import manifest_sha256, sha256_file, validate_raw_source_root


class DevelopmentReader:
    def __init__(self, data_root: Path, spec: dict[str, Any], base: dict[str, Any]) -> None:
        self.root, pairs = validate_raw_source_root(data_root)
        self.paths = dict(pairs)
        self.spec, self.base = spec, base
        self.sources: dict[int, dict[str, Any]] = {}
        self.access_log: list[dict[str, Any]] = []

    def read(self, cid: int, role: str, stage: str) -> PidCase:
        case = read_development_case(self.paths[cid], cid, role, self.spec, self.base)
        if cid in self.sources and case.audit['sha256'] != self.sources[cid]['sha256']:
            raise ValueError('Source file changed during the frozen experiment')
        self.sources[cid] = {**case.audit, 'role': role, 'path': portable(self.paths[cid])}
        self.access_log.append({'condition_id': cid, 'role': role, 'stage': stage})
        return case


def select_dynamic_model(models: list[DynamicModel], reader: DevelopmentReader,
                         spec: dict[str, Any], base: dict[str, Any]) -> tuple[DynamicModel | None, dict[str, Any]]:
    scores: dict[str, list[float]] = {model.name: [] for model in models}
    per_case: dict[str, Any] = {}
    eligibility: dict[str, list[float]] = {model.name: [] for model in models}
    comparison_fractions = []
    primary = [base['horizons_s'].index(h) for h in spec['selection']['primary_horizons_s']]
    for cid in base['split']['validation']:
        case = reader.read(cid, 'validation', 'candidate_selection')
        prepared = prepare_case(models[0].physics, case, base['origin_stride_frames'])
        batches = [model.predict(case, prepared.origins, prepared=prepared) for model in models]
        # All candidates use the SAME origins for scores; no apparent gain by
        # dropping difficult rows from one candidate's support population.
        common = np.logical_and.reduce([batch.supported for batch in batches]) & prepared.labels['approximate_zoh']
        for hi in primary:
            denominator = int(prepared.labels['approximate_zoh'][:, hi].sum())
            comparison_fractions.append(float(common[:, hi].sum() / denominator) if denominator else 0.0)
        diagnostics: dict[str, Any] = {}
        for model, batch in zip(models, batches):
            values = {}
            for hi, horizon in enumerate(base['horizons_s']):
                valid = prepared.labels['valid'][:, hi]
                own_known = batch.supported[:, hi] & valid
                fraction = float(own_known.sum() / valid.sum()) if valid.any() else 0.0
                if hi in primary:
                    eligibility[model.name].append(fraction)
                mask = common[:, hi]
                if not mask.any():
                    values[horizon_key(horizon)] = {'origin_count': 0, 'known_fraction': fraction}
                    if hi in primary:
                        scores[model.name].append(float('inf'))
                    continue
                signed = (prepared.labels['peak_nm'][mask, hi] - batch.peaks_nm['dynamic_fir'][mask, hi]) / case.pid_peak_nm
                mae = float(np.abs(signed).mean())
                under95 = float(np.quantile(np.maximum(signed, 0.0), 0.95))
                score = mae + spec['selection']['tail_weight'] * under95
                values[horizon_key(horizon)] = {'origin_count': int(mask.sum()), 'known_fraction': fraction,
                                                'normalized_peak_mae': mae, 'normalized_underestimate_p95': under95,
                                                'selection_score': score}
                if hi in primary:
                    scores[model.name].append(score)
            diagnostics[model.name] = values
        per_case[str(cid)] = diagnostics
    frozen = []
    for model in models:
        enough = bool(eligibility[model.name]) and min(eligibility[model.name]) >= spec['selection']['minimum_known_origin_fraction']
        enough &= bool(comparison_fractions) and min(comparison_fractions) >= spec['selection']['minimum_common_comparison_fraction']
        finite = bool(scores[model.name]) and bool(np.isfinite(scores[model.name]).all())
        score = float(np.mean(scores[model.name])) if finite else None
        frozen.append({'name': model.name, 'lags_frames': list(model.lags), 'ridge_fraction': model.ridge_fraction,
                   'eligible': bool(enough and finite), 'equal_case_horizon_score': score,
                       'minimum_known_fraction': min(eligibility[model.name]) if eligibility[model.name] else None})
    allowed = [i for i, row in enumerate(frozen) if row['eligible']]
    best = min(allowed, key=lambda i: frozen[i]['equal_case_horizon_score']) if allowed else None
    return models[best] if best is not None else None, {'role': 'reused validation selection, not independent audit',
                          'common_support_for_all_candidates': True,
                          'minimum_common_comparison_fraction': min(comparison_fractions) if comparison_fractions else None,
                          'rule': spec['selection']['score'], 'candidates': frozen,
                          'status': 'SELECTED' if best is not None else 'INCONCLUSIVE_NO_ELIGIBLE_CANDIDATE',
                          'selected_model': models[best].name if best is not None else None, 'per_case': per_case}


def calibration_records(model: DynamicModel, reader: DevelopmentReader,
                        scales: dict[str, ErrorScaleModel]) -> dict[str, list[dict[str, Any]]]:
    p = model.physics.protocol
    primary = [p['horizons_s'].index(h) for h in model.spec['joint_margin']['scope_horizons_s']]
    records = {method: [] for method in scales}
    for cid in p['split']['calibration']:
        case = reader.read(cid, 'calibration', 'joint_margin_calibration')
        prepared = prepare_case(model.physics, case, p['origin_stride_frames'])
        predictions = model.predict(case, prepared.origins, prepared=prepared)
        scale_predictions = {method: scale_model.predict(case, prepared.origins) for method, scale_model in scales.items()}
        common_scale_known = np.logical_and.reduce([known[:, primary].all(axis=1) for _, known in scale_predictions.values()])
        for method, scale_model in scales.items():
            scale, scale_known = scale_predictions[method]
            eligible = (predictions.supported[:, primary] & prepared.labels['approximate_zoh'][:, primary] &
                        scale_known[:, primary]).all(axis=1)
            eligible &= common_scale_known
            records[method].append({'condition_id': cid, 'region': case.region[prepared.origins],
                                    'actual': prepared.labels['peak_nm'], 'predicted': predictions.peaks_nm[method],
                                    'scale': scale, 'eligible': eligible})
    return records


def tight_plan_match(prepared: PreparedCase, base: dict[str, Any], spec: dict[str, Any]) -> np.ndarray:
    steps = np.rint(np.asarray(base['horizons_s']) / base['step_duration_s']).astype(int)
    indices = prepared.origins[:, None] + np.arange(1, steps[-1] + 1)[None, :]
    bounded = np.minimum(indices, len(prepared.case.time) - 1)
    tolerance = spec['evaluation']['tight_pid_plan_tolerance']
    delta = np.abs(prepared.case.actions[bounded] - prepared.case.actions[prepared.origins, None])
    match = (delta[:, :, 0] <= tolerance['torque_nm']) & (delta[:, :, 1] <= tolerance['pitch_rad'])
    return np.logical_and.accumulate(match, axis=1)[:, steps - 1] & prepared.labels['valid']


def joint_metrics(entries: list[dict[str, np.ndarray]], base: dict[str, Any]) -> dict[str, Any]:
    if not entries:
        return {'origin_count': 0, 'joint_coverage': None}
    actual = np.concatenate([item['actual'] for item in entries])
    predicted = np.concatenate([item['predicted'] for item in entries])
    margin = np.concatenate([item['margin'] for item in entries])
    reference = np.concatenate([item['reference'] for item in entries])
    covered = actual <= predicted + margin
    upper = predicted + margin
    ratio = base['evaluation']['primary_near_peak_ratio']
    positive = actual >= ratio * reference[:, None, :]
    negative = ~positive
    alert = upper >= ratio * reference[:, None, :]
    return {'origin_count': len(actual), 'joint_coverage': float(covered.all(axis=(1, 2)).mean()),
            'per_horizon_three_blade_coverage': covered.all(axis=2).mean(axis=0).tolist(),
            'per_horizon_per_blade_coverage': covered.mean(axis=0).tolist(),
            'mean_margin_nm': float(margin.mean()), 'per_horizon_mean_margin_nm': margin.mean(axis=(0, 2)).tolist(),
            'mean_margin_fraction_of_pid_peak': float((margin / reference[:, None, :]).mean()),
            'near_peak_positive_blade_horizon_origins': int(positive.sum()),
            'near_peak_upper_recall': float((positive & alert).sum() / positive.sum()) if positive.any() else None,
            'near_peak_upper_false_positive_fraction': float((negative & alert).sum() / negative.sum()) if negative.any() else None,
            'model_upper_crosses_pid_reference_fraction': float((upper > reference[:, None, :]).any(axis=(1, 2)).mean()),
            'qualification': 'reused validation, simultaneous 3 blades x 2 horizons per origin; not independent/trajectory guarantee'}


def validation_events(case: PidCase, origins: np.ndarray, peaks: np.ndarray,
                      margins: np.ndarray, usable: np.ndarray, base: dict[str, Any],
                      primary: list[int], approximate_zoh: np.ndarray | None = None) -> dict[str, Any]:
    result = {}
    times = case.time[origins]
    for pi, hi in enumerate(primary):
        horizon = base['horizons_s'][hi]
        counts = Counter(detected_episodes=0, total_events=0, censored_onset_events=0, timely_events=0,
                 late_events=0, no_model_opportunity_events=0, no_zoh_matched_opportunity_events=0)
        leads = []
        for blade in range(3):
            threshold = base['evaluation']['primary_near_peak_ratio'] * case.pid_peak_nm[blade]
            alerts = usable & (peaks[:, hi, blade] + margins[:, pi, blade] >= threshold)
            for segment in np.unique(case.segment[case.valid]):
                rows = np.flatnonzero(case.valid & (case.segment == segment))
                episodes = load_episodes(case.time[rows], np.abs(case.my_nm[rows, blade]), threshold,
                                         base['evaluation']['event_merge_gap_s'])
                for onset, end in episodes:
                    counts['detected_episodes'] += 1
                    if abs(onset - case.time[rows[0]]) < 1e-9:
                        counts['censored_onset_events'] += 1
                        continue
                    counts['total_events'] += 1
                    same_segment = case.segment[origins] == segment
                    opportunities = usable & same_segment & (times >= onset - horizon - 1e-9) & (times < onset - 1e-9)
                    counts['no_model_opportunity_events'] += int(not opportunities.any())
                    if approximate_zoh is not None:
                        counts['no_zoh_matched_opportunity_events'] += int(not np.any(opportunities & approximate_zoh[:, hi]))
                    early = opportunities & alerts
                    if early.any():
                        counts['timely_events'] += 1
                        leads.append(float(onset - times[early][0]))
                    elif np.any(alerts & same_segment & (times >= onset - 1e-9) & (times <= end + 1e-9)):
                        counts['late_events'] += 1
        result[horizon_key(horizon)] = {**dict(counts), 'lead_times_s': leads,
                                        'timely_recall': counts['timely_events'] / counts['total_events'] if counts['total_events'] else None}
    return result


def evaluate_validation(model: DynamicModel, reader: DevelopmentReader, spec: dict[str, Any],
                        base: dict[str, Any], scales: dict[str, ErrorScaleModel],
                        calibration: dict[str, dict[str, Any]], output: Path) -> dict[str, Any]:
    primary = [base['horizons_s'].index(h) for h in spec['joint_margin']['scope_horizons_s']]
    points: dict[tuple[str, int, str, str], dict[str, list[np.ndarray]]] = {}
    joint: dict[tuple[str, str, str], list[dict[str, np.ndarray]]] = {}
    per_case, support, margin_support = {}, {}, {}
    episode_reports = []
    with (output / 'validation_samples.csv').open('x', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(('condition_id', 'time_s', 'horizon_s', 'blade', 'region', 'label_valid', 'model_supported',
                         'support_reason', 'approximate_zoh', 'tight_pid_plan_match', 'actual_future_peak_nm',
                         *[f'{method}_peak_nm' for method in POINT_METHODS], 'pid_reference_peak_nm',
                         'cumulative_peak_nm', 'common_margin_known',
                         *[f'{method}_{margin_method}_error_nm' for method in scales for margin_method in MARGIN_METHODS]))
        for cid in base['split']['validation']:
            print(f'reused validation wind{cid}', flush=True)
            case = reader.read(cid, 'validation', 'post_selection_development_evaluation')
            prepared = prepare_case(model.physics, case, base['origin_stride_frames'])
            batch = model.predict(case, prepared.origins, prepared=prepared)
            tight = tight_plan_match(prepared, base, spec)
            margin_results = {method: joint_margin_arrays(calibration[method], scale_model, case, prepared.origins)
                              for method, scale_model in scales.items()}
            common_margin_known = np.logical_and.reduce([known for _, known in margin_results.values()])
            case_report = {'horizons': {}}
            for hi, horizon in enumerate(base['horizons_s']):
                hs = horizon_key(horizon)
                labels_valid = prepared.labels['valid'][:, hi]
                known = labels_valid & batch.supported[:, hi]
                eligible = known & prepared.labels['approximate_zoh'][:, hi]
                counts = support.setdefault(hs, Counter())
                counts.update(label_valid=int(labels_valid.sum()), known=int(known.sum()), approximate_zoh=int(eligible.sum()),
                              tight_pid_plan=int((known & tight[:, hi]).sum()))
                case_horizon = {'origin_count': int(eligible.sum()), 'known_fraction': float(known.sum() / labels_valid.sum()) if labels_valid.any() else None,
                                'support_reason_counts': dict(Counter(str(x) for x in batch.reasons[:, hi])), 'metrics': {}}
                for method in POINT_METHODS:
                    for role, mask in (('approximate_zoh', eligible), ('tight_pid_plan', known & tight[:, hi])):
                        for ri, region in [(-1, 'all'), *enumerate(REGIONS)]:
                            selected = mask if ri < 0 else mask & (case.region[prepared.origins] == ri)
                            if not selected.any():
                                continue
                            target = points.setdefault((method, hi, region, role), {})
                            reference = np.broadcast_to(case.pid_peak_nm, (int(selected.sum()), 3)).copy()
                            append_parts(target, prepared.labels['peak_nm'][selected, hi], batch.peaks_nm[method][selected, hi],
                                         reference, np.full((int(selected.sum()), 3), np.nan))
                        if role == 'approximate_zoh' and mask.any():
                            parts = {}
                            append_parts(parts, prepared.labels['peak_nm'][mask, hi], batch.peaks_nm[method][mask, hi],
                                         np.broadcast_to(case.pid_peak_nm, (int(mask.sum()), 3)).copy(), np.full((int(mask.sum()), 3), np.nan))
                            case_horizon['metrics'][method] = point_metrics(parts, base)
                for oi, origin in enumerate(prepared.origins):
                    ri = int(case.region[origin])
                    for blade in range(3):
                        pi = primary.index(hi) if hi in primary else None
                        writer.writerow((cid, f'{case.time[origin]:.8f}', horizon, blade + 1,
                                         REGIONS[ri] if ri >= 0 else 'unsupported', int(labels_valid[oi]), int(batch.supported[oi, hi]),
                                         batch.reasons[oi, hi], int(prepared.labels['approximate_zoh'][oi, hi]), int(tight[oi, hi]),
                                         prepared.labels['peak_nm'][oi, hi, blade] if labels_valid[oi] else '',
                                                                                 *[batch.peaks_nm[method][oi, hi, blade] if batch.supported[oi, hi] else '' for method in POINT_METHODS],
                                                                                 case.pid_peak_nm[blade], case.cumulative_peak_nm[origin, blade],
                                                                                 int(common_margin_known[oi]) if pi is not None else '',
                                                                                 *[margin_results[method][0][margin_method][oi, pi, blade]
                                                                                     if pi is not None and common_margin_known[oi] else ''
                                                                                     for method in scales for margin_method in MARGIN_METHODS]))
                case_report['horizons'][hs] = case_horizon
            joint_labels_eligible = (prepared.labels['approximate_zoh'][:, primary] & batch.supported[:, primary]).all(axis=1)
            for method, scale_model in scales.items():
                arrays, scale_known = margin_results[method]
                eligible = joint_labels_eligible & common_margin_known
                counts = margin_support.setdefault(method, Counter())
                counts.update(point_eligible=int(joint_labels_eligible.sum()), margin_known=int(eligible.sum()),
                              own_margin_known=int((joint_labels_eligible & scale_known).sum()))
                for margin_method, margins in arrays.items():
                    for ri, region in [(-1, 'all'), *enumerate(REGIONS)]:
                        selected = eligible if ri < 0 else eligible & (case.region[prepared.origins] == ri)
                        if selected.any():
                            joint.setdefault((method, margin_method, region), []).append({
                                'actual': prepared.labels['peak_nm'][selected][:, primary],
                                'predicted': batch.peaks_nm[method][selected][:, primary], 'margin': margins[selected],
                                'reference': np.broadcast_to(case.pid_peak_nm, (int(selected.sum()), 3)).copy()})
                if method == 'dynamic_fir':
                    causal_usable = batch.supported[:, primary].all(axis=1) & scale_known
                    episode_reports.append(validation_events(case, prepared.origins, batch.peaks_nm[method],
                                                             arrays['joint_conditional95'], causal_usable, base, primary,
                                                             prepared.labels['approximate_zoh']))
            per_case[str(cid)] = case_report
    metrics = {f'{method}|{horizon_key(h)}|{region}|{role}': point_metrics(points.get((method, hi, region, role), {}), base)
               for hi, h in enumerate(base['horizons_s']) for method in POINT_METHODS
               for region in ('all', *REGIONS) for role in ('approximate_zoh', 'tight_pid_plan')}
    joint_report = {f'{method}|{margin_method}|{region}': joint_metrics(joint.get((method, margin_method, region), []), base)
                    for method in scales for margin_method in MARGIN_METHODS for region in ('all', *REGIONS)}
    events = {}
    for horizon in spec['joint_margin']['scope_horizons_s']:
        hs = horizon_key(horizon)
        rows = [row[hs] for row in episode_reports]
        event = {name: sum(row[name] for row in rows) for name in ('detected_episodes', 'total_events', 'censored_onset_events',
             'timely_events', 'late_events', 'no_model_opportunity_events', 'no_zoh_matched_opportunity_events')}
        leads = [lead for row in rows for lead in row['lead_times_s']]
        event['timely_recall'] = event['timely_events'] / event['total_events'] if event['total_events'] else None
        event['median_lead_s'] = float(np.median(leads)) if leads else None
        event['qualification'] = 'reused validation recorded-PID proxy, NOT restricted to future plan match; censored onsets excluded, same valid segment; ideal offline timing, not strict exceedance/control evidence'
        events[hs] = event
    return {'role': 'reused validation development evidence only', 'fresh_independent_audit': False,
            'support': {hs: {**dict(count), 'known_fraction': count['known'] / count['label_valid'] if count['label_valid'] else None}
                        for hs, count in support.items()},
            'margin_comparison_domain': 'common point support AND both causal scale/calibration-group supports; calibration and evaluation cross-model comparisons use this intersection',
            'margin_support': {method: {**dict(count), 'known_fraction': count['margin_known'] / count['point_eligible'] if count['point_eligible'] else None,
                                       'own_known_fraction': count['own_margin_known'] / count['point_eligible'] if count['point_eligible'] else None}
                               for method, count in margin_support.items()},
            'metrics': metrics, 'joint_margins': joint_report, 'events': events, 'per_case': per_case,
            'strict_pid_exceedance': {'positive_events': 0, 'recall': None, 'reason': 'own full-window PID maximum cannot be strictly exceeded by itself'}}


def development_gate(validation: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    checks = []
    gate = spec['development_gate']

    def append(name: str, scope: str, value: float | None, threshold: float, minimum: bool = True, kind: str = 'point') -> None:
        passed = value is not None and (value >= threshold if minimum else value <= threshold)
        checks.append({'name': name, 'scope': scope, 'value': value, 'threshold': threshold,
                       'status': 'INCONCLUSIVE' if value is None else ('PASS' if passed else 'FAIL'), 'kind': kind})

    for horizon in spec['selection']['primary_horizons_s']:
        hs = horizon_key(horizon)
        old = validation['metrics'][f'v1_greybox|{hs}|all|approximate_zoh']
        new = validation['metrics'][f'dynamic_fir|{hs}|all|approximate_zoh']
        reduction = 1.0 - new['mae_nm'] / old['mae_nm'] if old.get('mae_nm', 0.0) > 0.0 and new.get('origin_count') else None
        append('peak_MAE_reduction_vs_v1', hs, reduction, gate['minimum_peak_mae_reduction_vs_v1_each_primary_horizon'])
        old95 = float(np.mean(old['underestimate_p95_nm'])) if old.get('origin_count') else 0.0
        new95 = float(np.mean(new['underestimate_p95_nm'])) if new.get('origin_count') else 0.0
        tail_reduction = 1.0 - new95 / old95 if old95 > 0.0 else None
        append('underestimate_p95_reduction_vs_v1', hs, tail_reduction, gate['minimum_p95_underestimate_reduction_vs_v1_each_primary_horizon'])
        append('known_origin_fraction', hs, validation['support'][hs]['known_fraction'], spec['selection']['minimum_known_origin_fraction'])
        for cid, case in validation['per_case'].items():
            values = case['horizons'][hs]['metrics']
            ratio = values['dynamic_fir']['mae_nm'] / values['v1_greybox']['mae_nm'] if values.get('v1_greybox', {}).get('mae_nm', 0.0) else None
            append('case_MAE_ratio_to_v1', f'wind{cid}/{hs}', ratio, gate['maximum_case_mae_ratio_to_v1'], minimum=False)
    for region in REGIONS:
        values = validation['joint_margins'][f'dynamic_fir|joint_conditional95|{region}']
        coverage = values['joint_coverage'] if values['origin_count'] >= spec['evaluation']['minimum_joint_evaluation_origins_per_region'] else None
        append('3_blades_2_windows_empirical_coverage', region, coverage, gate['minimum_joint_empirical_coverage_each_region'], kind='margin')
    append('margin_known_fraction', 'all', validation['margin_support']['dynamic_fir']['known_fraction'],
           gate['minimum_margin_known_fraction'], kind='margin')
    append('near_peak_upper_false_positive_fraction', 'all',
           validation['joint_margins']['dynamic_fir|joint_conditional95|all'].get('near_peak_upper_false_positive_fraction'),
           gate['maximum_near_peak_upper_false_positive_fraction'], minimum=False, kind='margin')
    point_passed = all(row['status'] == 'PASS' for row in checks if row['kind'] == 'point')
    margin_passed = all(row['status'] == 'PASS' for row in checks if row['kind'] == 'margin')
    return {'point_decision': gate['on_point_success'] if point_passed else gate['on_point_failure'],
            'margin_decision': 'PROMISING_REUSED_VALIDATION_MARGIN_ONLY' if margin_passed else 'REVIEW_JOINT_MARGIN',
            'integration_decision': 'BLOCKED_NO_FRESH_INDEPENDENT_AUDIT', 'checks': checks,
            'authorization': dict(AUTHORIZATION)}


def draw_plot(validation: dict[str, Any], base: dict[str, Any], output: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.2), constrained_layout=True)
    positions = np.arange(len(base['horizons_s']))
    for mi, method in enumerate(POINT_METHODS):
        y = [validation['metrics'][f'{method}|{horizon_key(h)}|all|approximate_zoh'].get('normalized_mae', np.nan) for h in base['horizons_s']]
        axes[0].bar(positions + (mi - 2) * 0.15, y, width=0.15, label=method)
    axes[0].set_xticks(positions, [f'{h:.2f}s' for h in base['horizons_s']])
    axes[0].set_ylabel('Peak MAE / same-case PID maximum')
    axes[0].set_title('REUSED validation, not independent audit')
    axes[0].legend(fontsize=8)
    x = np.arange(len(REGIONS))
    for mi, method in enumerate(MARGIN_METHODS):
        y = [validation['joint_margins'][f'dynamic_fir|{method}|{region}']['joint_coverage'] for region in REGIONS]
        axes[1].bar(x + (mi - 1) * 0.22, [np.nan if value is None else value for value in y], width=0.22, label=method)
    axes[1].axhline(0.95, color='black', linestyle='--', label='development target .95')
    axes[1].set_xticks(x, REGIONS)
    axes[1].set_ylim(0, 1.04)
    axes[1].set_ylabel('Empirical joint coverage: 3 blades x 2 windows')
    axes[1].set_title('Conditional margin comparison')
    axes[1].legend(fontsize=8)
    fig.savefig(output / 'diagnostic.png', dpi=160)
    plt.close(fig)


def report_markdown(report: dict[str, Any]) -> str:
    base, spec, validation = report['base_protocol'], report['protocol'], report['validation']
    if validation is None:
        lines = ['# D0-R：物理载荷裕量动态改进验证 v2', '',
                 '**INCONCLUSIVE：没有候选达到冻结的共同支持域/可用比例，未选模型，不进入余量校准。**', '',
                 '- 六组候选分数和各 case/H 的样本数量已保存；未扩网格、未放宽门槛。',
                 '- 本轮只读取 train/validation；旧 audit 未打开/未哈希，calibration 未使用。',
                 '- 仅复用 validation 开发结果；生产授权、未执行动作验证和新独立 audit 均为 false。',
                 '- 接线结论：**BLOCKED_NO_FRESH_INDEPENDENT_AUDIT**。', '',
                 '| 候选 | 等权开发分数 | 最小已知比例 | 可选 |', '|---|---:|---:|---|']
        for row in report['selection']['candidates']:
            score = f'{row["equal_case_horizon_score"]:.6f}' if row['equal_case_horizon_score'] is not None else '不可估计'
            known = f'{row["minimum_known_fraction"]:.2%}' if row['minimum_known_fraction'] is not None else '不可估计'
            lines.append(f'| {row["name"]} | {score} | {known} | {row["eligible"]} |')
        lines += ['', f'- 实际读取 {len(report["sources"])} 个 case / {report["unique_input_rows"]} 行。',
                  f'- 协议 SHA-256：`{report["protocol_sha256"]}`。',
                  f'- 清单 SHA-256：`{report["input_manifest_sha256"]}`。',
                  '- 应先解释支持域/缺组失败，不能把未选模型当成已验证模型接线。', '']
        return '\n'.join(lines)
    lines = ['# D0-R：物理载荷裕量动态改进验证 v2', '',
             '**范围：仅开发验证，不是新的独立 audit，不授权生产接线。**', '',
             f'- 动态候选：{report["selection"]["selected_model"]}；有限窗线性 FIR，无神经网络、无递归模态求解。',
             f'- 数据：train {base["split"]["train"]}；validation {base["split"]["validation"]}；calibration {base["split"]["calibration"]}。',
             f'- 旧 audit {spec["data_policy"]["blocked_old_audit_ids"]} **未打开/未哈希/未参与评价**。',
             '- validation 用于选择历史长度/岭参数，随后指标和覆盖只能称“复用验证集开发结果”。',
             '- train 拟合物理曲面、每个未来帧的有符号载荷增量残差、条件误差尺度；calibration 只冻结余量。',
             '- 输入仍为现有白名单；使用历史差分/相位/加速度等内部派生量，不增加 UDP/Actor/观测变量。',
             '- 原始 demand 来源及工程身份仍待核定；只验证已有 PID 轨迹，不验证未执行 RL 候选因果效果。',
             f'- 点预测决策：**{report["gate"]["point_decision"]}**。',
             f'- 联合余量决策：**{report["gate"]["margin_decision"]}**。',
             f'- 接线决策：**{report["gate"]["integration_decision"]}**。', '',
             '## 1. 因果模型与选择规则', '',
             '- 保持 v1 实测 My 锚定物理 rollout，回归未来 20 ms 各帧的有符号残差增量，最后取 max_abs。',
             '- 历史只读到 origin；未来 My 仅作为监督目标，未来 wind/actual/action 不进入特征。',
             '- 所有候选在同一支持域 origins 上比较，避免通过排除难样本制造改善。',
             '- 选择分数为逐 case/短窗等权的 NMAE + 0.25×归一化低估 p95；只从六个冻结候选中选。',
             '- 错误尺度是 train 残差的线性 log-scale 拟合，校准样本再决定分位数；不是新实测变量或硬界。', '',
             '## 2. 复用验证集的点预测', '',
             '| H | v1 NMAE | 动态 FIR NMAE | 改善 | 动态 MAE (MNm) | 动态低估 p95 (MNm，三片平均) |',
             '|---|---:|---:|---:|---:|---:|']
    for horizon in base['horizons_s']:
        hs = horizon_key(horizon)
        old = validation['metrics'][f'v1_greybox|{hs}|all|approximate_zoh']
        new = validation['metrics'][f'dynamic_fir|{hs}|all|approximate_zoh']
        if not new['origin_count'] or not old['origin_count']:
            lines.append(f'| {hs}s | 不可估计 | 不可估计 | — | — | — |')
            continue
        improvement = 1.0 - new['normalized_mae'] / old['normalized_mae'] if old['normalized_mae'] else 0.0
        lines.append(f'| {hs}s | {old["normalized_mae"]:.6f} | {new["normalized_mae"]:.6f} | {improvement:.2%} | '
                     f'{new["mae_nm"]/1e6:.4f} | {np.mean(new["underestimate_p95_nm"])/1e6:.4f} |')
    lines += ['', 'v1 是本轮相同 train/validation、相同 origin 上重算的基线；不是与旧 audit 数字交叉比较。0.50 s 仅诊断。', '',
              '| case | H | v1 NMAE | 动态 NMAE |', '|---|---|---:|---:|']
    for cid, case in validation['per_case'].items():
        for horizon in spec['selection']['primary_horizons_s']:
            hs = horizon_key(horizon)
            row = case['horizons'][hs]['metrics']
            if 'dynamic_fir' in row:
                lines.append(f'| wind{cid} | {hs}s | {row["v1_greybox"]["normalized_mae"]:.6f} | {row["dynamic_fir"]["normalized_mae"]:.6f} |')
    lines += ['', '## 3. 三片×双窗联合余量', '',
              '在相同可用误差尺度/模型域内，比较逐片 95%、全六维固定联合余量、条件化全六维联合余量。'
              '数值是在复用 validation 上观察的经验覆盖，不具有独立置信保证。', '',
              '| 模型 | 余量方法 | 区域 | origin | 六维联合覆盖 | 平均余量 (MNm) |',
              '|---|---|---|---:|---:|---:|']
    for method in ('v1_greybox', 'dynamic_fir'):
        for margin_method in MARGIN_METHODS:
            for region in REGIONS:
                values = validation['joint_margins'][f'{method}|{margin_method}|{region}']
                coverage = f'{values["joint_coverage"]:.2%}' if values['joint_coverage'] is not None else '不可估计'
                width = f'{values["mean_margin_nm"]/1e6:.4f}' if values.get('mean_margin_nm') is not None else '—'
                lines.append(f'| {method} | {margin_method} | {region} | {values["origin_count"]} | {coverage} | {width} |')
    lines += ['', '未知/排除比例和逐窗逐片覆盖详见机器报告；没有覆盖组不能用零余量填补。'
              '加大余量不等于计算精度提高，需同时检查误报及平均宽度。', '',
              '## 4. 近峰与开发门禁', '', '| H | 0.95 近峰事件 | 按时 | 晚报 | 无模型机会 | 按时 recall |',
              '|---|---:|---:|---:|---:|---:|']
    for hs, event in validation['events'].items():
        recall = f'{event["timely_recall"]:.2%}' if event['timely_recall'] is not None else '不可估计'
        lines.append(f'| {hs}s | {event["total_events"]} | {event["timely_events"]} | {event["late_events"]} | '
                     f'{event["no_model_opportunity_events"]} | {recall} |')
    lines += ['', '这是未限制未来计划匹配的 recorded-PID 近峰 proxy 和理想离线提前量；跨无效段机会不计，起点删失事件单列并排除。'
              '不是严格超 PID recall；严格正例为 0，recall 不可估计。', '',
              '| 检查 | 范围 | 实测 | 门槛 | 状态 |', '|---|---|---:|---:|---|']
    for row in report['gate']['checks']:
        value = f'{row["value"]:.6f}' if row['value'] is not None else '不可估计'
        lines.append(f'| {row["name"]} | {row["scope"]} | {value} | {row["threshold"]} | {row["status"]} |')
    lines += ['', '## 5. 身份与下一步边界', '',
              f'- 冻结开发协议 SHA-256：`{report["protocol_sha256"]}`。',
              f'- 实际读取 {len(report["sources"])} 个 case / {report["unique_input_rows"]} 行，清单 SHA-256：`{report["input_manifest_sha256"]}`。',
              f'- 动态模型 SHA-256：`{report["model_sha256"]}`；各文件哈希、读取阶段记录在报告。',
              f'- 运行 {report["resources"]["elapsed_s"]:.2f} s，Linux 峰值 RSS {report["resources"]["peak_rss_mib"]:.2f} MiB；不是 Windows T1。',
              '- 旧 v1 协议/工件/报告不覆盖；本轮无 Bladed/生产 UDP/Actor/BC/TCN 更新。',
              '- 没有新独立数据证据，不能把“validation 选出来的最好模型”当作已达可用精度。',
              '- 保留全部候选分数/失败门禁；若动态点预测改善但覆盖失败，继续检查条件尺度/校准域，不调到旧 audit 通过。',
              '- 即使开发点预测和余量均达标，仍需动作/工程来源核定、未执行动作局部验证及获准后的独立真实 evidence，soft/hard 仍关闭。', '']
    return '\n'.join(lines)


def run_experiment(data_root: Path, protocol_path: Path, output: Path,
                   project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    started = time.perf_counter()
    spec, base = load_development_protocol(protocol_path, project_root)
    if output.exists():
        raise FileExistsError('Development output already exists; preserve frozen artifacts with a new directory')
    output.mkdir(parents=True)
    save_json(output / 'frozen_protocol.json', spec)
    save_json(output / 'frozen_base_protocol.json', base)
    static = verify_static_project(project_root / base['static_parameters']['source_project'], base)
    reader = DevelopmentReader(data_root, spec, base)
    physics_fitter = GreyboxFitter(base)
    for cid in base['split']['train']:
        print(f'physics fit wind{cid}', flush=True)
        physics_fitter.add_case(reader.read(cid, 'train', 'physics_fit'))
    physics = physics_fitter.finish()
    physics_document = physics.to_document()
    save_json(output / 'frozen_physics_model.json', physics_document)
    fitters = [DynamicFitter(physics, spec, tuple(lags)) for lags in spec['dynamic_model']['history_lag_candidates_frames']]
    for cid in base['split']['train']:
        print(f'causal FIR increment fit wind{cid}', flush=True)
        prepared = prepare_case(physics, reader.read(cid, 'train', 'dynamic_fit'), spec['dynamic_model']['fit_stride_frames'])
        for fitter in fitters:
            fitter.add_case(prepared)
    models = [fitter.finish(fraction) for fitter in fitters for fraction in spec['dynamic_model']['ridge_fraction_candidates']]
    save_json(output / 'frozen_candidate_models.json', [model.to_document() for model in models])
    selected, selection = select_dynamic_model(models, reader, spec, base)
    save_json(output / 'frozen_selection.json', selection)
    model_document, validation, calibration = None, None, {}
    if selected is None:
        gate = {'point_decision': 'INCONCLUSIVE_NO_ELIGIBLE_CANDIDATE', 'margin_decision': 'NOT_RUN_NO_SELECTED_MODEL',
                'integration_decision': 'BLOCKED_NO_FRESH_INDEPENDENT_AUDIT', 'checks': [], 'authorization': dict(AUTHORIZATION)}
    else:
        print(f'frozen selection {selected.name}', flush=True)
        model_document = selected.to_document()
        save_json(output / 'frozen_dynamic_model.json', model_document)
        scale_fitters = {method: ErrorScaleFitter(selected, method) for method in ('v1_greybox', 'dynamic_fir')}
        for cid in base['split']['train']:
            case = reader.read(cid, 'train', 'error_scale_fit')
            prepared = prepare_case(physics, case, base['origin_stride_frames'])
            batch = selected.predict(case, prepared.origins, prepared=prepared)
            for fitter in scale_fitters.values():
                fitter.add_case(prepared, batch)
        scales = {method: fitter.finish() for method, fitter in scale_fitters.items()}
        save_json(output / 'frozen_error_scales.json', {method: model.to_document() for method, model in scales.items()})
        records = calibration_records(selected, reader, scales)
        calibration = {method: make_joint_calibration(records[method], selected, scale) for method, scale in scales.items()}
        save_json(output / 'frozen_joint_calibration.json', calibration)
        validation = evaluate_validation(selected, reader, spec, base, scales, calibration, output)
        gate = development_gate(validation, spec)
        draw_plot(validation, base, output)
    sources = [reader.sources[cid] for cid in sorted(reader.sources)]
    read_ids = sorted({row['condition_id'] for row in reader.access_log})
    if set(read_ids) & set(spec['data_policy']['blocked_old_audit_ids']):
        raise AssertionError('Old audit was accessed')
    report = {'schema_version': 'physical_load_margin_dynamic_development_report_v2',
              'protocol': spec, 'base_protocol': base, 'protocol_sha256': document_sha256(spec),
              'protocol_file_sha256': sha256_file(protocol_path), 'static_project': static,
              'sources': sources, 'input_manifest_sha256': manifest_sha256(sources), 'read_access_log': reader.access_log,
              'source_role_manifest_sha256': {role: manifest_sha256([row for row in sources if row['role'] == role])
                                             for role in spec['data_policy']['allowed_roles']},
              'unique_input_rows': sum(row['rows'] for row in sources),
              'read_condition_ids': read_ids, 'blocked_audit_access_count': 0,
              'selection': selection, 'model_sha256': document_sha256(model_document) if model_document is not None else None,
              'physics_model_sha256': document_sha256(physics_document),
              'calibration': calibration, 'validation': validation, 'gate': gate,
              'code_sha256': {'runner': sha256_file(Path(__file__)),
                              'dynamic_module': sha256_file(PROJECT_ROOT / 'RL/utils/physical_load_margin_dynamic.py'),
                              'base_module': sha256_file(PROJECT_ROOT / 'RL/utils/physical_load_margin.py'),
                              'shared_metrics': sha256_file(PROJECT_ROOT / 'RL/demo_physical_load_margin.py'),
                              'raw_contract': sha256_file(PROJECT_ROOT / 'RL/utils/raw_dataset_contract.py'),
                              'constants': sha256_file(PROJECT_ROOT / 'RL/utils/constants.py'),
                              'state_classifier': sha256_file(PROJECT_ROOT / 'RL/utils/operating_state.py')},
              'artifact_file_sha256': {path.name: sha256_file(path) for path in sorted(output.iterdir()) if path.is_file()},
              'resources': {'elapsed_s': time.perf_counter() - started,
                            'peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
                            'python_version': sys.version.split()[0], 'numpy_version': np.__version__,
                            'blas_thread_environment': {key: os.environ.get(key) for key in (
                                'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')}},
              'authorization': dict(AUTHORIZATION)}
    save_json(output / 'report.json', report)
    with (output / 'README.md').open('x', encoding='utf-8') as handle:
        handle.write(report_markdown(report))
    print(json.dumps({'point_decision': gate['point_decision'], 'margin_decision': gate['margin_decision'],
                      'integration_decision': gate['integration_decision'], 'selected_model': selected.name if selected is not None else None,
                      'output': portable(output), 'elapsed_s': report['resources']['elapsed_s']}, ensure_ascii=False), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=PROJECT_ROOT / 'bladed-data-new/raw')
    parser.add_argument('--protocol', type=Path, default=PROJECT_ROOT / 'RL/configs/physical_load_margin_dynamic_demo_v2.json')
    parser.add_argument('--output-dir', type=Path, default=PROJECT_ROOT / 'RL/diagnostics/physical_load_margin_dynamic_v2')
    args = parser.parse_args()
    run_experiment(args.data_root.resolve(), args.protocol.resolve(), args.output_dir.resolve())


if __name__ == '__main__':
    main()