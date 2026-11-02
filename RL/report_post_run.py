"""Summarize a finished queue run from compact logs.

This is deliberately a *summary* tool: it reads the small per-condition JSONL
and optional summaries, never the multi-GB long-form tables.  It reports the
pooled energy change and per-blade peak changes, but it is not a formal
acceptance report.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 'post_run_report_v1'
CONDITION_SUMMARIES = 'compact_queue_condition_summaries.jsonl'
METRICS_SUMMARY = 'compact_metrics_summary.json'
SHADOW_SUMMARY = 'physical_load_shadow_summary.json'
RUN_META = 'run_meta.json'


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z')


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def load_conditions(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    warnings: list[str] = []
    with path.open('r', encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                warnings.append(f'line {line_number}: invalid JSON ({exc.msg})')
                continue
            if isinstance(record, dict):
                records.append(record)
            else:
                warnings.append(f'line {line_number}: not a JSON object')
    return records, warnings


def _energy_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    trapezoid = [
        record for record in records
        if _finite(record.get('queue_condition_trapezoid_energy_valid')) == 1.0
        and (_finite(record.get('queue_condition_pid_trapezoid_energy_j')) or 0.0) > 0.0
    ]
    if trapezoid:
        basis = 'trapezoid'
        rl = sum(_finite(record.get('queue_condition_rl_trapezoid_energy_j')) or 0.0 for record in trapezoid)
        pid = sum(_finite(record.get('queue_condition_pid_trapezoid_energy_j')) or 0.0 for record in trapezoid)
        used = trapezoid
    else:
        basis = 'pdt_fallback'
        used = [
            record for record in records
            if (_finite(record.get('condition_pid_energy_j')) or 0.0) > 0.0
        ]
        rl = sum(_finite(record.get('condition_rl_energy_j')) or 0.0 for record in used)
        pid = sum(_finite(record.get('condition_pid_energy_j')) or 0.0 for record in used)
    delta_percent = (rl / pid - 1.0) * 100.0 if pid > 0.0 else None
    return {
        'basis': basis,
        'conditions_used': len(used),
        'rl_energy_j': rl,
        'pid_energy_j': pid,
        'delta_energy_percent': delta_percent,
        'trapezoid_invalid_conditions': len(records) - len(trapezoid) if basis == 'trapezoid' else len(records),
    }


def _blade_summary(records: list[dict[str, Any]], blade: int) -> dict[str, Any]:
    deltas = []
    ratios = []
    peaks = []
    for record in records:
        delta = _finite(record.get(f'queue_condition_blade_{blade}_delta_load_percent'))
        ratio = _finite(record.get(f'queue_condition_blade_{blade}_peak_ratio_to_pid'))
        rl_peak = _finite(record.get(f'queue_condition_blade_{blade}_peak_abs_nm'))
        pid_peak = _finite(record.get(f'queue_condition_pid_peak_blade_{blade}_nm'))
        if ratio is None and rl_peak is not None and pid_peak:
            ratio = rl_peak / pid_peak
            delta = (ratio - 1.0) * 100.0
        if delta is not None:
            deltas.append(delta)
        if ratio is not None:
            ratios.append(ratio)
        if rl_peak is not None:
            peaks.append(rl_peak)
    mean_delta = sum(deltas) / len(deltas) if deltas else None
    return {
        'conditions_used': len(deltas),
        'mean_load_change_percent': mean_delta,
        'mean_reduction_percent': -mean_delta if mean_delta is not None else None,
        'mean_peak_ratio_to_pid': sum(ratios) / len(ratios) if ratios else None,
        'worst_peak_ratio_to_pid': max(ratios) if ratios else None,
        'max_rl_peak_nm': max(peaks) if peaks else None,
    }


def _worst_conditions(records: list[dict[str, Any]], limit: int = 5) -> dict[str, Any]:
    energy = []
    loads = {1: [], 2: [], 3: []}
    for record in records:
        condition_id = _finite(record.get('condition_id'))
        round_index = _finite(record.get('queue_condition_round_index'))
        delta_energy = _finite(record.get('queue_condition_trapezoid_delta_energy_percent'))
        if delta_energy is None:
            delta_energy = _finite(record.get('condition_energy_deficit_percent'))
        if delta_energy is not None:
            energy.append({
                'condition_id': condition_id,
                'round': round_index,
                'delta_energy_percent': delta_energy,
            })
        for blade in (1, 2, 3):
            delta_load = _finite(record.get(f'queue_condition_blade_{blade}_delta_load_percent'))
            if delta_load is not None:
                loads[blade].append({
                    'condition_id': condition_id,
                    'round': round_index,
                    'delta_load_percent': delta_load,
                })
    energy.sort(key=lambda item: item['delta_energy_percent'])
    for blade in loads:
        loads[blade].sort(key=lambda item: item['delta_load_percent'], reverse=True)
    return {
        'worst_energy': energy[:limit],
        'worst_load_blade_1': loads[1][:limit],
        'worst_load_blade_2': loads[2][:limit],
        'worst_load_blade_3': loads[3][:limit],
    }


def _shadow_summary(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        document = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        return {'error': f'{type(exc).__name__}: {exc}'}
    counters = document.get('counters') if isinstance(document, dict) else None
    core = document.get('core_counters') if isinstance(document, dict) else None
    return {
        'summary_status': document.get('summary_status') if isinstance(document, dict) else None,
        'faulted': document.get('faulted') if isinstance(document, dict) else None,
        'worker_alive': document.get('worker_alive') if isinstance(document, dict) else None,
        'counters': counters if isinstance(counters, dict) else {},
        'core_counters': core if isinstance(core, dict) else {},
    }


def summarize(run_dir: Path) -> dict[str, Any]:
    conditions_path = run_dir / CONDITION_SUMMARIES
    if not conditions_path.is_file():
        raise FileNotFoundError(f'missing {CONDITION_SUMMARIES} in {run_dir}')
    records, warnings = load_conditions(conditions_path)
    if not records:
        raise ValueError(f'no condition summaries in {conditions_path}')

    energy = _energy_summary(records)
    blades = {str(blade): _blade_summary(records, blade) for blade in (1, 2, 3)}
    deltas = [blades[key]['mean_load_change_percent'] for key in ('1', '2', '3')]
    if all(value is not None for value in deltas):
        mean_load_change = sum(value for value in deltas if value is not None) / 3.0
    else:
        mean_load_change = None

    rounds = sorted({int(value) for value in (
        _finite(record.get('queue_condition_round_index')) for record in records
    ) if value is not None})
    instances = [int(value) for value in (
        _finite(record.get('condition_instance_id')) for record in records
    ) if value is not None]

    identity_complete = sum(
        1 for record in records
        if _finite(record.get('queue_case_identity_full_trace_complete')) == 1.0
    )
    trapezoid_invalid = sum(
        1 for record in records
        if _finite(record.get('queue_condition_trapezoid_energy_valid')) != 1.0
    )
    dual_incomplete = sum(
        1 for record in records
        if _finite(record.get('dual_constraint_condition_complete')) != 1.0
    )
    if identity_complete < len(records):
        warnings.append(
            f'strict full-trace identity complete for {identity_complete}/{len(records)} conditions'
        )
    if trapezoid_invalid:
        warnings.append(f'trapezoid energy invalid for {trapezoid_invalid}/{len(records)} conditions')
    if dual_incomplete:
        warnings.append(f'dual constraint incomplete for {dual_incomplete}/{len(records)} conditions')
    if len(instances) != len(set(instances)):
        warnings.append('duplicate condition_instance_id detected')

    shadow = _shadow_summary(run_dir / SHADOW_SUMMARY)
    if shadow and isinstance(shadow.get('counters'), dict):
        strict_complete = shadow['counters'].get('strict_complete_conditions')
        if strict_complete == 0:
            warnings.append('physical shadow strict_complete_conditions=0')
        if shadow.get('faulted') is True:
            warnings.append('physical shadow worker faulted')
    if shadow is None:
        warnings.append('physical_load_shadow_summary.json not found')

    meta = None
    if (run_dir / RUN_META).is_file():
        try:
            meta = json.loads((run_dir / RUN_META).read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            meta = None

    return {
        'schema_version': SCHEMA_VERSION,
        'generated_at_utc': _now_utc(),
        'run_dir': str(run_dir),
        'run_meta': meta,
        'counts': {
            'conditions': len(records),
            'rounds': rounds,
            'unique_instances': len(set(instances)),
        },
        'energy': energy,
        'loads': {
            'blade_1': blades['1'],
            'blade_2': blades['2'],
            'blade_3': blades['3'],
            'mean_load_change_percent': mean_load_change,
            'mean_reduction_percent': -mean_load_change if mean_load_change is not None else None,
        },
        'worst': _worst_conditions(records),
        'identity': {
            'full_trace_complete': identity_complete,
            'conditions': len(records),
        },
        'shadow': shadow,
        'warnings': warnings,
        'acceptance': False,
        'note': 'Summary only; not a formal policy/life acceptance report.',
    }


def render_markdown(document: dict[str, Any]) -> str:
    energy = document['energy']
    loads = document['loads']
    lines = [
        '# 运行后摘要（非正式验收）',
        '',
        f"- run_dir: `{document['run_dir']}`",
        f"- conditions: {document['counts']['conditions']}；rounds: {document['counts']['rounds']}",
        '',
        '## 电量',
        '',
        f"- basis: `{energy['basis']}`（{energy['conditions_used']} 条）",
        f"- RL: {energy['rl_energy_j'] / 3.6e9:.3f} MWh",
        f"- PID: {energy['pid_energy_j'] / 3.6e9:.3f} MWh",
    ]
    if energy['delta_energy_percent'] is None:
        lines.append('- ΔE: N/A')
    else:
        lines.append(f"- ΔE: {energy['delta_energy_percent']:+.4f}%")
    lines.extend([
        '',
        '## 三叶根 My（逐 case 峰值 vs PID）',
        '',
        '| 叶片 | 平均载荷变化 | 平均降低率 | 平均峰值比 | 最差峰值比 |',
        '|---|---:|---:|---:|---:|',
    ])
    for name, key in (('Blade 1', 'blade_1'), ('Blade 2', 'blade_2'), ('Blade 3', 'blade_3')):
        blade = loads[key]
        change = blade['mean_load_change_percent']
        ratio = blade['mean_peak_ratio_to_pid']
        worst = blade['worst_peak_ratio_to_pid']
        if change is None or ratio is None or worst is None:
            lines.append(f'| {name} | N/A | N/A | N/A | N/A |')
        else:
            lines.append(
                f'| {name} | {change:+.3f}% | {-change:+.3f}% | {ratio:.4f} | {worst:.4f} |'
            )
    mean_change = loads['mean_load_change_percent']
    if mean_change is not None:
        lines.append('')
        lines.append(
            f'- 三叶平均载荷变化：{mean_change:+.3f}%；平均降低率：{-mean_change:+.3f}%'
            '（正降低率 = RL 峰值低于 PID）'
        )
    lines.extend([
        '',
        '## 警告',
        '',
    ])
    lines.extend([f'- {warning}' for warning in document['warnings']] or ['- 无'])
    lines.extend([
        '',
        '## 边界',
        '',
        '- 本摘要只读取 compact 小表，不替代正式验收。',
        '- 载荷变化为正表示 RL 峰值更高；降低率为正表示 RL 峰值更低。',
        '- 正式结论仍需同工况、同时间窗、冻结策略与人工签认。',
        '',
    ])
    return '\n'.join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Summarize a finished queue run from compact logs.')
    parser.add_argument('--run-dir', required=True, help='Experiment/run directory')
    parser.add_argument('--output-dir', default='', help='Default: <run-dir>/post_run')
    parser.add_argument('--require-complete', action='store_true', help='Exit 1 when warnings are present')
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = Path(args.run_dir).expanduser()
    if not run_dir.is_absolute():
        run_dir = (PROJECT_ROOT / run_dir).resolve()
    document = summarize(run_dir)
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else run_dir / 'post_run'
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'post_run_report.json').write_text(
        json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
        encoding='utf-8',
    )
    (output_dir / 'post_run_report.md').write_text(render_markdown(document), encoding='utf-8')
    print(render_markdown(document))
    print(f'wrote {output_dir / "post_run_report.json"}')
    if args.require_complete and document['warnings']:
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
