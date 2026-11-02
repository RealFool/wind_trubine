"""Plot frozen D0-R PID validation outputs; never refits or opens raw/audit data."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIAGNOSTICS = PROJECT_ROOT / 'RL/diagnostics/physical_load_margin_dynamic_v2'
DEFAULT_OUTPUT = PROJECT_ROOT / 'RL/diagnostics/physical_load_margin_dynamic_v2_visualizations'
HORIZONS = ('0.10', '0.20', '0.50')
PRIMARY_HORIZONS = ('0.10', '0.20')
VALIDATION_IDS = (5, 13, 21)
METHODS = ('v1_greybox', 'dynamic_fir', 'persistence')
METHOD_LABELS = {
    'v1_greybox': 'v1 物理灰箱',
    'dynamic_fir': 'D0-R 动态 FIR',
    'persistence': 'Persistence 基线',
}
MARGIN_METHODS = ('legacy_per_blade95_nm', 'joint_constant95_nm', 'joint_conditional95')
MARGIN_LABELS = {
    'legacy_per_blade95_nm': '逐片 95% 余量',
    'joint_constant95_nm': '固定六维联合余量',
    'joint_conditional95': '条件六维联合余量',
}
COLORS = {'v1_greybox': '#64748b', 'dynamic_fir': '#0f766e', 'persistence': '#d97706'}
REGIONS = ('below_rated', 'transition', 'above_rated')
REGION_LABELS = {'below_rated': '额定以下', 'transition': '过渡区', 'above_rated': '额定以上'}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _number(value: str | None) -> float:
    if value is None or value == '':
        return math.nan
    result = float(value)
    return result if math.isfinite(result) else math.nan


def _read_sources(report_path: Path, samples_path: Path) -> tuple[dict[str, Any], dict[tuple[int, int, str], list[dict[str, Any]]]]:
    report = json.loads(report_path.read_text(encoding='utf-8'))
    protocol = report.get('protocol', {})
    policy = protocol.get('data_policy', {})
    if (report.get('schema_version') != 'physical_load_margin_dynamic_development_report_v2' or
            policy.get('read_old_audit') is not False or
            report.get('blocked_audit_access_count') != 0 or
            set(report.get('read_condition_ids', ())) & set(policy.get('blocked_old_audit_ids', ())) or
            report.get('selection', {}).get('selected_model') != 'fir_lag40_ridge0.0001' or
            report.get('selection', {}).get('role') != 'reused validation selection, not independent audit'):
        raise ValueError('Expected the frozen D0-R reused-validation report; refusing other evidence scope')
    frozen_sha = report.get('artifact_file_sha256', {}).get(samples_path.name)
    if not isinstance(frozen_sha, str) or _sha256(samples_path) != frozen_sha:
        raise ValueError('Frozen validation sample CSV SHA mismatch; original diagnostics remain unchanged')
    sources = report.get('sources', [])
    validation_sources = [item for item in sources if item.get('role') == 'validation']
    validation_ids = tuple(sorted(int(item['condition_id']) for item in validation_sources))
    if validation_ids != VALIDATION_IDS:
        raise ValueError(f'Unexpected validation cases: {validation_ids}')
    cases = {int(item['condition_id']): item for item in validation_sources}
    grouped: dict[tuple[int, int, str], list[dict[str, Any]]] = {}
    with samples_path.open('r', encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        required = {'condition_id', 'time_s', 'horizon_s', 'blade', 'region', 'label_valid',
                    'model_supported', 'approximate_zoh', 'actual_future_peak_nm',
                    'dynamic_fir_peak_nm', 'v1_greybox_peak_nm', 'persistence_peak_nm',
                    'pid_reference_peak_nm', 'cumulative_peak_nm',
                    'dynamic_fir_joint_conditional95_error_nm'}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError('Frozen validation sample table is missing required columns')
        for raw in reader:
            cid, blade, horizon = int(raw['condition_id']), int(raw['blade']), f"{float(raw['horizon_s']):.2f}"
            if cid not in VALIDATION_IDS or blade not in (1, 2, 3) or horizon not in HORIZONS:
                raise ValueError('Unexpected case/blade/horizon in frozen validation table')
            row = {
                'condition_id': cid, 'blade': blade, 'horizon_s': horizon,
                'time_s': _number(raw['time_s']), 'region': raw['region'],
                'label_valid': raw['label_valid'] == '1',
                'model_supported': raw['model_supported'] == '1',
                'approximate_zoh': raw['approximate_zoh'] == '1',
                'tight_pid_plan_match': raw['tight_pid_plan_match'] == '1',
                'actual_future_peak_nm': _number(raw['actual_future_peak_nm']),
                'dynamic_fir_peak_nm': _number(raw['dynamic_fir_peak_nm']),
                'v1_greybox_peak_nm': _number(raw['v1_greybox_peak_nm']),
                'persistence_peak_nm': _number(raw['persistence_peak_nm']),
                'pid_reference_peak_nm': _number(raw['pid_reference_peak_nm']),
                'cumulative_peak_nm': _number(raw['cumulative_peak_nm']),
                'dynamic_fir_margin_nm': _number(raw['dynamic_fir_joint_conditional95_error_nm']),
            }
            grouped.setdefault((cid, blade, horizon), []).append(row)
    # The CSV intentionally writes every sampled origin/blade/horizon, including
    # unsupported or right-censored labels. Its size therefore closes against
    # the frozen case row counts and the frozen origin stride, not valid labels.
    stride = int(report.get('base_protocol', {}).get('origin_stride_frames', 0))
    if stride < 1:
        raise ValueError('Frozen report has no valid validation origin stride')
    expected_validation_rows = sum(max(0, (int(case['rows']) - 2) // stride + 1) *
                                   3 * len(HORIZONS) for case in validation_sources)
    actual_rows = sum(len(rows) for rows in grouped.values())
    if actual_rows != expected_validation_rows or actual_rows <= 0:
        raise ValueError(f'Frozen validation table row count mismatch: {actual_rows} != {expected_validation_rows}')
    for key, rows in grouped.items():
        times = [row['time_s'] for row in rows]
        if not all(math.isfinite(value) for value in times) or times != sorted(times):
            raise ValueError(f'Validation origins must have finite sorted timestamps: {key}')
    report['_validation_sources'] = cases
    report['_validation_csv_sha256'] = frozen_sha
    report['_validation_row_count'] = actual_rows
    return report, grouped


def _configure_matplotlib():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Noto Sans CJK JP', 'DejaVu Sans'],
        'axes.unicode_minus': False,
        'figure.facecolor': 'white',
        'axes.facecolor': '#fbfdff',
        'axes.grid': True,
        'grid.alpha': 0.23,
        'grid.linestyle': '-',
    })
    return plt


def _plot_metrics(report: dict[str, Any], output: Path, plt) -> None:
    validation = report['validation']
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2), constrained_layout=True)
    positions = np.arange(len(HORIZONS), dtype=np.float64)
    width = 0.23
    for index, method in enumerate(METHODS):
        mae, under95 = [], []
        for horizon in HORIZONS:
            metrics = validation['metrics'][f'{method}|{horizon}|all|approximate_zoh']
            mae.append(metrics['mae_nm'] / 1e6)
            under95.append(float(np.mean(metrics['underestimate_p95_nm'])) / 1e6)
        offset = (index - (len(METHODS) - 1) / 2) * width
        axes[0].bar(positions + offset, mae, width, color=COLORS[method], label=METHOD_LABELS[method])
        axes[1].bar(positions + offset, under95, width, color=COLORS[method], label=METHOD_LABELS[method])
    for ax, title, ylabel in (
        (axes[0], '未来峰值点预测误差', '三片平均 MAE（MNm）'),
        (axes[1], '低估尾部（正部误差 p95）', '三片 p95 平均（MNm）'),
    ):
        ax.set_title(title, fontsize=13, weight='bold')
        ax.set_ylabel(ylabel)
        ax.set_xticks(positions, [f'{float(h):.2f} s' for h in HORIZONS])
        ax.legend(frameon=True, fontsize=9)
    axes[0].text(0.02, 0.97, '0.50 s 仅作诊断', transform=axes[0].transAxes,
                 va='top', fontsize=9, color='#475569')
    fig.suptitle('离线 D0-R validation：点预测精度与低估尾部', fontsize=15, weight='bold')
    fig.savefig(output / 'prediction_metrics.png', dpi=180)
    plt.close(fig)


def _plot_margins(report: dict[str, Any], output: Path, plt) -> None:
    joint = report['validation']['joint_margins']
    x = np.arange(len(REGIONS), dtype=np.float64)
    width = 0.24
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.1), constrained_layout=True)
    margin_colors = ('#64748b', '#0f766e', '#7c3aed')
    for index, method in enumerate(MARGIN_METHODS):
        coverage, margin_width = [], []
        for region in REGIONS:
            item = joint[f'dynamic_fir|{method}|{region}']
            coverage.append(np.nan if item['joint_coverage'] is None else item['joint_coverage'])
            margin_width.append(np.nan if item.get('mean_margin_nm') is None else item['mean_margin_nm'] / 1e6)
        offset = (index - (len(MARGIN_METHODS) - 1) / 2) * width
        axes[0].bar(x + offset, coverage, width, color=margin_colors[index], label=MARGIN_LABELS[method])
        axes[1].bar(x + offset, margin_width, width, color=margin_colors[index], label=MARGIN_LABELS[method])
        for xpos, value in zip(x + offset, coverage):
            if math.isfinite(float(value)):
                axes[0].text(xpos, value + 0.0015, f'{value:.1%}', ha='center', va='bottom', fontsize=7, rotation=90)
    axes[0].axhline(0.95, color='#b91c1c', linestyle='--', linewidth=1.5, label='开发参考线 95%')
    axes[0].set_ylim(0.75, 1.015)
    axes[0].set_ylabel('逐 origin 六维联合经验覆盖率')
    axes[0].set_title('三叶片 × 0.10/0.20 s 同时覆盖', fontsize=13, weight='bold')
    axes[1].set_ylabel('平均经验余量（MNm）')
    axes[1].set_title('余量宽度与覆盖方法对照', fontsize=13, weight='bold')
    for ax in axes:
        ax.set_xticks(x, [REGION_LABELS[name] for name in REGIONS])
        ax.legend(frameon=True, fontsize=8)
    fig.suptitle('D0-R 条件误差余量：复用 validation 的经验结果（非独立保证）', fontsize=14, weight='bold')
    fig.savefig(output / 'joint_margin_validation.png', dpi=180)
    plt.close(fig)


def _plot_scatter(grouped: dict[tuple[int, int, str], list[dict[str, Any]]], output: Path, plt) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.9), constrained_layout=True, sharex=True, sharey=True)
    for ax, horizon in zip(axes, HORIZONS):
        actual, predicted = [], []
        for cid in VALIDATION_IDS:
            for blade in (1, 2, 3):
                for row in grouped[(cid, blade, horizon)]:
                    if not (row['label_valid'] and row['model_supported'] and row['approximate_zoh']):
                        continue
                    if not all(math.isfinite(row[key]) for key in (
                            'actual_future_peak_nm', 'dynamic_fir_peak_nm', 'pid_reference_peak_nm')):
                        continue
                    actual.append(row['actual_future_peak_nm'] / row['pid_reference_peak_nm'])
                    predicted.append(row['dynamic_fir_peak_nm'] / row['pid_reference_peak_nm'])
        actual_array, predicted_array = np.asarray(actual), np.asarray(predicted)
        near = actual_array >= 0.95
        ax.scatter(actual_array[~near], predicted_array[~near], s=5, alpha=0.10,
                   color='#64748b', linewidths=0, rasterized=True, label='实际 < 0.95L')
        ax.scatter(actual_array[near], predicted_array[near], s=16, alpha=0.62,
                   color='#dc2626', linewidths=0, rasterized=True, label='实际 ≥ 0.95L')
        ax.plot([0, 1.02], [0, 1.02], color='#0f172a', linestyle='--', linewidth=1, label='预测 = 实际')
        ax.axvline(0.95, color='#f59e0b', linestyle=':', linewidth=1.2)
        ax.axhline(0.95, color='#f59e0b', linestyle=':', linewidth=1.2)
        ax.axvline(1.0, color='#b91c1c', linestyle='-.', linewidth=1.1)
        ax.axhline(1.0, color='#b91c1c', linestyle='-.', linewidth=1.1)
        ax.set_xlim(0.0, 1.03)
        ax.set_ylim(0.0, 1.18)
        ax.set_aspect('auto')
        ax.set_title(f'H = {float(horizon):.2f} s | n={len(actual_array):,} 叶片-origin')
        ax.set_xlabel('实际未来窗 max |My| / 同 case PID 峰值 L')
        ax.legend(loc='upper left', fontsize=7, framealpha=0.92)
    axes[0].set_ylabel('动态 FIR 预测未来峰值 / PID 峰值 L')
    fig.suptitle('预测锚点散点：每个点是一片叶片的 origin 与未来窗标签', fontsize=14, weight='bold')
    fig.savefig(output / 'prediction_vs_actual_scatter.png', dpi=180)
    plt.close(fig)


def _latest_peak_anchor(rows: list[dict[str, Any]], *, peak_time_s: float,
                        horizon: str, reference_nm: float) -> dict[str, Any] | None:
    span = float(horizon)
    candidates = [row for row in rows if row['label_valid'] and row['model_supported'] and row['approximate_zoh'] and
                  math.isfinite(row['time_s']) and row['time_s'] < peak_time_s - 1e-9 and
                  row['time_s'] + span >= peak_time_s - 1e-9 and
                  math.isclose(row['actual_future_peak_nm'], reference_nm, rel_tol=1e-8, abs_tol=1.0) and
                  math.isfinite(row['dynamic_fir_peak_nm'])]
    return max(candidates, key=lambda row: row['time_s']) if candidates else None


def _anchor_row(source: dict[str, Any], blade: int, horizon: str,
                anchor: dict[str, Any] | None, reference_nm: float) -> dict[str, Any]:
    result = {
        'condition_id': int(source['condition_id']), 'blade': blade, 'horizon_s': float(horizon),
        'pid_peak_time_s': float(source['pid_peak_times_s'][blade - 1]),
        'pid_peak_reference_nm': reference_nm, 'anchor_status': 'matched' if anchor else 'no_eligible_anchor',
    }
    if anchor is None:
        return result
    prediction = anchor['dynamic_fir_peak_nm']
    margin = anchor['dynamic_fir_margin_nm']
    result.update({
        'origin_time_s': anchor['time_s'], 'label_endpoint_time_s': anchor['time_s'] + float(horizon),
        'lead_to_pid_peak_s': float(source['pid_peak_times_s'][blade - 1]) - anchor['time_s'],
        'actual_future_peak_nm': anchor['actual_future_peak_nm'],
        'predicted_future_peak_nm': prediction,
        'conditional_margin_nm_illustrative': margin,
        'predicted_upper_nm_illustrative': prediction + margin if math.isfinite(margin) else math.nan,
        'actual_over_pid_ratio': anchor['actual_future_peak_nm'] / reference_nm,
        'prediction_over_pid_ratio': prediction / reference_nm,
        'illustrative_upper_over_pid_ratio': (prediction + margin) / reference_nm if math.isfinite(margin) else math.nan,
        'observed_cumulative_anchor_over_pid_ratio': anchor['cumulative_peak_nm'] / reference_nm,
        'region': anchor['region'], 'approximate_zoh': anchor['approximate_zoh'],
    })
    return result


def _plot_peak_anchors(report: dict[str, Any], grouped: dict[tuple[int, int, str], list[dict[str, Any]]],
                       output: Path, plt) -> list[dict[str, Any]]:
    fig, axes = plt.subplots(3, 3, figsize=(17.2, 12.6), sharex=True, sharey=True)
    anchors: list[dict[str, Any]] = []
    for column, cid in enumerate(VALIDATION_IDS):
        source = report['_validation_sources'][cid]
        for blade in (1, 2, 3):
            ax = axes[blade - 1, column]
            peak_time = float(source['pid_peak_times_s'][blade - 1])
            reference = float(source['pid_peak_abs_nm'][blade - 1])
            short_rows = grouped[(cid, blade, '0.10')]
            long_rows = grouped[(cid, blade, '0.20')]
            for horizon, rows in (('0.10', short_rows), ('0.20', long_rows)):
                chosen = _latest_peak_anchor(rows, peak_time_s=peak_time,
                    horizon=horizon, reference_nm=reference)
                anchors.append(_anchor_row(source, blade, horizon, chosen, reference))

            left, right = peak_time - 1.5, peak_time + 0.25
            view_short = [row for row in short_rows if left <= row['time_s'] <= right]
            view_long = [row for row in long_rows if left <= row['time_s'] <= right]
            ax.axvspan(-0.20, 0.0, color='#f59e0b', alpha=0.11, linewidth=0,
                       label='0.20 s 峰值前锚点窗')
            ax.axhline(1.0, color='#b91c1c', linestyle='--', linewidth=1.15, label='PID 历史最大值 L')
            ax.axhline(0.95, color='#f59e0b', linestyle=':', linewidth=1.0, label='近峰参考 0.95L')
            ax.axvline(0.0, color='#334155', linestyle=':', linewidth=1.1)

            cumulative_rows = [row for row in view_short if row['label_valid'] and math.isfinite(row['cumulative_peak_nm'])]
            if cumulative_rows:
                ax.plot([row['time_s'] - peak_time for row in cumulative_rows],
                        [row['cumulative_peak_nm'] / reference for row in cumulative_rows],
                        color='#111827', linewidth=1.6, label='锚点时已观测累计峰值')

            for horizon, rows, color, marker in (
                    ('0.10', view_short, '#ea580c', 'o'), ('0.20', view_long, '#2563eb', 's')):
                selected = [row for row in rows if row['label_valid'] and row['model_supported'] and
                            row['approximate_zoh'] and math.isfinite(row['dynamic_fir_peak_nm'])]
                if not selected:
                    continue
                x = np.asarray([row['time_s'] - peak_time for row in selected])
                predicted = np.asarray([row['dynamic_fir_peak_nm'] / reference for row in selected])
                actual = np.asarray([row['actual_future_peak_nm'] / reference for row in selected])
                ax.plot(x, predicted, color=color, linewidth=1.8,
                        label=f'D0-R 预测未来峰值 H={float(horizon):.2f}s')
                ax.scatter(x, actual, color=color, marker=marker, facecolors='none', s=21,
                           alpha=0.72, linewidths=0.9,
                           label=f'实际未来窗峰值 H={float(horizon):.2f}s')
                if horizon == '0.20':
                    margin_rows = [row for row in selected if math.isfinite(row['dynamic_fir_margin_nm'])]
                    if margin_rows:
                        margin_x = np.asarray([row['time_s'] - peak_time for row in margin_rows])
                        upper = np.asarray([(row['dynamic_fir_peak_nm'] + row['dynamic_fir_margin_nm']) / reference
                                            for row in margin_rows])
                        ax.plot(margin_x, upper, color='#16a34a', linestyle='-.', linewidth=1.45,
                                label='预测 + PID 条件经验余量（仅示意）')

            local_anchors = [row for row in anchors if row['condition_id'] == cid and row['blade'] == blade]
            for anchor in local_anchors:
                if anchor['anchor_status'] != 'matched':
                    continue
                x = anchor['origin_time_s'] - peak_time
                color = '#ea580c' if anchor['horizon_s'] == 0.10 else '#2563eb'
                marker = '*' if anchor['horizon_s'] == 0.10 else 'D'
                ax.scatter([x], [anchor['prediction_over_pid_ratio']], color=color, marker=marker,
                           s=90, edgecolors='white', linewidths=0.7, zorder=6)
                ax.scatter([x], [anchor['actual_over_pid_ratio']], color='#111827', marker='x',
                           s=48, linewidths=1.7, zorder=7)
                ax.vlines(x, min(anchor['prediction_over_pid_ratio'], anchor['actual_over_pid_ratio']),
                          max(anchor['prediction_over_pid_ratio'], anchor['actual_over_pid_ratio']),
                          color=color, alpha=0.68, linewidth=1.0, zorder=5)

            ax.set_xlim(-1.5, 0.25)
            ax.set_ylim(0.0, 1.30)
            ax.set_title(f'wind{cid} · 叶片{blade} | PID 峰值 t={peak_time:.2f}s', fontsize=10.5, weight='bold')
            if column == 0:
                ax.set_ylabel('|My| 峰值 / 同叶片 PID 历史峰值 L')
            if blade == 3:
                ax.set_xlabel('origin 相对 PID 全窗最大峰时刻（s）')

    handles, labels = axes[0, 0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(unique.values(), unique.keys(), loc='lower center', ncol=4,
               bbox_to_anchor=(0.5, 0.018), frameon=True, fontsize=8.5)
    fig.suptitle('预测锚点放大：每片叶片 PID 全窗历史最大载荷附近', fontsize=16, weight='bold', y=0.99)
    fig.text(0.5, 0.956,
             '星形/菱形：距真实 PID 峰值最近且未来窗包含该峰的有效预测 origin；黑色 ×：该 origin 的实际未来窗峰值。'
             '绿线为 PID 校准集条件经验余量，不是 RL 安全界。',
             ha='center', va='top', fontsize=9.5, color='#475569')
    fig.tight_layout(rect=[0.015, 0.09, 0.99, 0.94])
    fig.savefig(output / 'pid_peak_prediction_anchors.png', dpi=190)
    plt.close(fig)
    return anchors


def _write_outputs(report: dict[str, Any], anchors: list[dict[str, Any]], output: Path,
                   report_path: Path, samples_path: Path, script_path: Path) -> None:
    with (output / 'pid_peak_anchor_examples.csv').open('x', encoding='utf-8', newline='') as handle:
        fields = list(dict.fromkeys(key for row in anchors for key in row))
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(anchors)
    matched = sum(row['anchor_status'] == 'matched' for row in anchors)
    if matched != len(VALIDATION_IDS) * 3 * len(PRIMARY_HORIZONS):
        raise ValueError(f'Expected an eligible final pre-peak anchor for all primary windows; got {matched}/{len(anchors)}')
    manifest = {
        'schema_version': 'physical_load_margin_dynamic_visualization_v1',
        'evidence_scope': 'frozen D0-R reused PID validation; development visualization only',
        'selection': report['selection']['selected_model'],
        'validation_condition_ids': list(VALIDATION_IDS),
        'validation_origin_rows': report['_validation_row_count'],
        'validation_samples_sha256': report['_validation_csv_sha256'],
        'input_report_sha256': _sha256(report_path),
        'plot_script_sha256': _sha256(script_path),
        'blocked_audit_access_count': 0,
        'old_audit_opened_or_rehashed_by_this_plotter': False,
        'model_refit_or_recalibrated': False,
        'pid_peak_anchor_definition': 'latest supported approximate-ZOH origin t < historical PID peak time <= t+H; actual future-window maximum equals same-blade PID reference',
        'matched_pid_peak_anchors': matched,
        'counterfactual_or_control_authorization': False,
        'figures': ['prediction_metrics.png', 'joint_margin_validation.png',
                    'prediction_vs_actual_scatter.png', 'pid_peak_prediction_anchors.png'],
    }
    (output / 'plot_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    lines = [
        '# D0-R 离线载荷预测可视化', '',
        '**仅展示冻结报告中 wind5/13/21 的复用 validation；没有重拟合、重校准或访问旧 audit。不是独立验证、实时预警或控制安全证据。**', '',
        '## 图表', '',
        '- `prediction_metrics.png`：v1 灰箱、动态 FIR、Persistence 的短窗/0.50 s 点误差与低估尾部。',
        '- `joint_margin_validation.png`：逐片余量、固定六维联合余量、条件六维联合余量的区域覆盖和宽度。',
        '- `prediction_vs_actual_scatter.png`：逐叶片 origin 的预测未来峰值与实际未来窗峰值；红点仅表示实际值达到0.95倍 PID 全窗峰值。',
        '- `pid_peak_prediction_anchors.png`：wind5/13/21 三片叶片各一格，在全窗 PID 最大峰附近显示累计观测峰值、0.10/0.20 s 预测锚点、实际未来窗标签及条件经验上界示意。',
        '- `pid_peak_anchor_examples.csv`：18个短窗锚点的 origin、窗口端点、预测、实际标签、经验余量和归一化值。', '',
        '## 锚点定义与限制', '',
        '预测锚点是历史 origin 时刻 t。每个锚点只在模型支持、近似 ZOH 匹配，且 t 严格早于已知 PID 全窗峰时刻、未来窗 (t,t+H] 包含该峰时才展示。图中的峰时刻和 future My 是离线标签；实测 PID demand 轨迹不是 RL 未执行候选的反事实结果。重复/重叠预测 origin 不独立。', '',
        '绿线/CSV 条件经验余量来自 PID calibration，对 RL 或新动作分布没有授权；其上界不能称为安全边界。图的目的在于直观看模型能否在历史极值附近给出接近未来实测峰值的预测，以及短窗预测提前量有多短。', '',
        f"- 选中模型：`{manifest['selection']}`。",
        f"- 样本 SHA-256：`{manifest['validation_samples_sha256']}`。",
        f"- 输入报告 SHA-256：`{manifest['input_report_sha256']}`。",
        f"- 有效 PID 峰值锚点：{matched}/{len(anchors)}。",
        '- 旧 audit 访问数：0；模型重拟合/重校准：否；控制/反事实授权：否。', '',
    ]
    (output / 'README.md').write_text('\n'.join(lines), encoding='utf-8')


def plot_visualizations(report_path: Path, samples_path: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError('Refusing to overwrite visualization output; choose a new isolated directory')
    report, grouped = _read_sources(report_path, samples_path)
    output.mkdir(parents=True)
    plt = _configure_matplotlib()
    _plot_metrics(report, output, plt)
    _plot_margins(report, output, plt)
    _plot_scatter(grouped, output, plt)
    anchors = _plot_peak_anchors(report, grouped, output, plt)
    _write_outputs(report, anchors, output, report_path, samples_path, Path(__file__).resolve())
    return json.loads((output / 'plot_manifest.json').read_text(encoding='utf-8'))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=DEFAULT_DIAGNOSTICS / 'report.json')
    parser.add_argument('--samples', type=Path, default=DEFAULT_DIAGNOSTICS / 'validation_samples.csv')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = plot_visualizations(args.report.resolve(), args.samples.resolve(), args.output_dir.resolve())
    print(json.dumps({'output_dir': str(args.output_dir.resolve()),
                      'matched_pid_peak_anchors': result['matched_pid_peak_anchors'],
                      'figures': result['figures'], 'development_only': True}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()