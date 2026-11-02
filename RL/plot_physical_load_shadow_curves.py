"""Stream a completed physical-shadow run into RL/PID load and classification plots.

Reads the large physical shadow JSONL one bounded line at a time. It retains
only the 16 fixed 600 s curves, a bounded set of classification markers, and
integer confusion counters; it never materializes the JSONL in memory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import resource
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from RL.utils.paired_pid_reference import (
    BLADE_ROOT_MY_FIELDS, CanonicalTraceHasher,
)
from RL.utils.physical_load_shadow_artifact import file_sha256, strict_json
from RL.utils.queue_manifest import expand_queue_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_WINDS = (8, 11, 12, 16)
DT = 0.02
SIMULATION_END_S = 600.0
FRAME_COUNT = round(SIMULATION_END_S / DT) + 1
MAX_RECORD_BYTES = 32 * 1024
MAX_RECORDS = 250_000
TN_MARKER_EVERY_FRAMES = 500  # correctly classified negatives: one dot per 10 s
ROUND_COLORS = {1: '#2878B5', 2: '#E17C05', 3: '#2A9D55', 4: '#8A62A8'}
CLASS_COLORS = {'TP': '#159447', 'FP': '#E88900', 'FN': '#D62728', 'TN': '#777777'}
CLASS_LABELS = {
    'TP': '正确报超限 TP', 'FP': '误报 FP', 'FN': '漏报 FN',
    'TN': f'正确未报 TN（图中每{TN_MARKER_EVERY_FRAMES * DT:g}s抽样）',
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _read_pid_csv(path: Path, *, expected_sha256: str,
                  expected_trace_signature: str) -> tuple[np.ndarray, np.ndarray, str]:
    if _sha256(path) != expected_sha256:
        raise ValueError(f'PID raw CSV SHA mismatch: {path}')
    times = np.empty(FRAME_COUNT, dtype=np.float64)
    signed_my = np.empty((FRAME_COUNT, 3), dtype=np.float32)
    hasher = CanonicalTraceHasher()
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f'PID raw CSV is empty: {path}') from exc
        wanted = ('Time', *BLADE_ROOT_MY_FIELDS, 'I_AverageWindSpeed')
        missing = [name for name in wanted if name not in header]
        if missing:
            raise ValueError(f'PID raw CSV is missing fields {missing}: {path}')
        index = {name: header.index(name) for name in wanted}
        count = 0
        for row_number, row in enumerate(reader, start=2):
            if count >= FRAME_COUNT:
                raise ValueError(f'PID raw CSV has excess rows: {path}')
            try:
                timestamp = float(row[index['Time']])
                wind = float(row[index['I_AverageWindSpeed']])
                loads = [float(row[index[name]]) for name in BLADE_ROOT_MY_FIELDS]
            except (IndexError, ValueError) as exc:
                raise ValueError(f'Invalid PID time/wind/My at {path}:{row_number}') from exc
            if not all(math.isfinite(value) for value in (timestamp, wind, *loads)):
                raise ValueError(f'Non-finite PID time/wind/My at {path}:{row_number}')
            expected_time = count * DT
            if not math.isclose(timestamp, expected_time, rel_tol=0.0, abs_tol=1e-6):
                raise ValueError(f'PID trace is off the 0.02 s grid at {path}:{row_number}')
            times[count] = timestamp
            signed_my[count] = loads
            hasher.update(timestamp, wind)
            count += 1
    if count != FRAME_COUNT:
        raise ValueError(f'PID trace row count mismatch for {path}: {count} != {FRAME_COUNT}')
    signature = hasher.hexdigest()
    if signature != expected_trace_signature:
        raise ValueError(f'PID time/wind trace signature mismatch: {path}')
    return times, signed_my, _sha256(path)


def _read_slot_references(run_dir: Path, slots: list[Any]) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    slot_files = sorted(run_dir.glob('queue_case_identity_condition_*_instance_*.json'))
    if len(slot_files) != 16:
        raise ValueError(f'Expected 16 per-slot identity summaries, found {len(slot_files)}')
    manifest_slot_by_identity = {
        (int(slot.condition_id), int(slot.condition_instance_id)): int(slot.queue_epoch)
        for slot in slots
    }
    by_wind: dict[int, dict[str, Any]] = {}
    per_slot = []
    seen_slots = set()
    for path in slot_files:
        entry = strict_json(path)
        wind = int(entry['condition_id'])
        instance = int(entry['condition_instance_id'])
        identity = (wind, instance)
        if identity not in manifest_slot_by_identity:
            raise ValueError(f'Identity summary does not match manifest: {identity}')
        epoch = manifest_slot_by_identity[identity]
        slot_key = (wind, instance, epoch)
        if slot_key in seen_slots:
            raise ValueError(f'Duplicate slot identity summary: {slot_key}')
        seen_slots.add(slot_key)
        current = {
            'path': str(Path(entry['pid_raw_csv_path']).resolve()),
            'sha256': entry['pid_raw_sha256'],
            'trace_signature': entry['expected_trace_signature'],
        }
        if wind in by_wind and by_wind[wind] != current:
            raise ValueError(f'PID reference source changed between slots for wind{wind}')
        by_wind[wind] = current
        per_slot.append({
            'wind': wind, 'instance': instance, 'epoch': int(entry.get('queue_epoch', instance)),
            'identity_status': entry.get('identity_status', 'missing'),
            'trace_signature_match': bool(entry.get('trace_signature_match', 0.0)),
            'wind_mismatch_frames': int(entry.get('queue_case_identity_wind_mismatch_count', 0)),
            'wind_max_abs_error_mps': float(entry.get('queue_case_identity_wind_max_abs_error_mps', 0.0)),
            'observed_frames': int(entry.get('observed_trace_sample_count', 0)),
            'expected_frames': int(entry.get('expected_trace_sample_count', 0)),
            'end_marker_time_s': entry.get('end_marker_time_s'),
        })
    if set(by_wind) != set(EXPECTED_WINDS):
        raise ValueError(f'Unexpected PID source winds: {sorted(by_wind)}')
    return by_wind, per_slot


def _confusion_counts() -> dict[str, int]:
    return {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}


def _classify(predicted_peak_nm: float, actual_peak_nm: float, pid_peak_nm: float) -> str:
    predicted_exceed = predicted_peak_nm > pid_peak_nm
    actual_exceed = actual_peak_nm > pid_peak_nm
    if predicted_exceed and actual_exceed:
        return 'TP'
    if predicted_exceed:
        return 'FP'
    if actual_exceed:
        return 'FN'
    return 'TN'


def _put_curve_sample(curve: np.ndarray, counts: np.ndarray, index: int,
                      value: np.ndarray, *, slot: tuple[int, int, int]) -> None:
    if index < 0 or index >= FRAME_COUNT:
        raise ValueError(f'RL load sample index outside 0..600 s: slot={slot}, index={index}')
    sample = np.asarray(value, dtype=np.float32)
    if sample.shape != (3,) or not np.isfinite(sample).all():
        raise ValueError(f'Invalid RL blade My sample: slot={slot}, index={index}')
    if counts[index]:
        if not np.allclose(curve[index], sample, rtol=0.0, atol=1e-5):
            raise ValueError(f'Overlapping shadow labels disagree: slot={slot}, index={index}')
    else:
        curve[index] = sample
    counts[index] = min(int(counts[index]) + 1, 255)


def _classification_rows(counts: dict[tuple[Any, ...], dict[str, int]]) -> list[dict[str, Any]]:
    rows = []
    for (wind, horizon, blade), values in sorted(counts.items(), key=lambda item: (
            str(item[0][0]), float(item[0][1]), str(item[0][2]))):
        tp, fp, fn, tn = (values[key] for key in ('TP', 'FP', 'FN', 'TN'))
        n = tp + fp + fn + tn
        rows.append({
            'wind': wind if wind == 'ALL' else int(wind),
            'horizon_s': float(horizon),
            'blade': blade if blade == 'ALL' else int(blade),
            'classified_points': n, 'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
            'accuracy_pct': None if not n else 100.0 * (tp + tn) / n,
            'precision_pct': None if tp + fp == 0 else 100.0 * tp / (tp + fp),
            'recall_pct': None if tp + fn == 0 else 100.0 * tp / (tp + fn),
            'specificity_pct': None if tn + fp == 0 else 100.0 * tn / (tn + fp),
            'actual_exceedance_pct': None if not n else 100.0 * (tp + fn) / n,
        })
    return rows


def _make_plot(path: Path, *, horizon: float, times: np.ndarray,
               pid_curves: dict[int, np.ndarray], rl_curves: dict[tuple[int, int, int], np.ndarray],
               pid_peaks: dict[int, np.ndarray], slots_by_wind: dict[int, list[tuple[int, int, int, int]]],
               selected_rounds: tuple[int, ...],
               marker_data: dict[tuple[int, int, float, str], tuple[list[float], list[float]]],
               confusion: dict[tuple[Any, ...], dict[str, int]], summary_note: str) -> None:
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Noto Sans CJK JP', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(3, 4, figsize=(22, 13), sharex=True)
    for column, wind in enumerate(EXPECTED_WINDS):
        for blade in range(1, 4):
            ax = axes[blade - 1, column]
            pid = np.abs(pid_curves[wind][:, blade - 1]) / 1e6
            ax.plot(times, pid, color='#151515', linewidth=1.0, alpha=0.72, zorder=2)
            threshold = float(pid_peaks[wind][blade - 1]) / 1e6
            ax.axhline(threshold, color='#5A2020', linestyle='--', linewidth=1.0, alpha=0.9, zorder=4)
            for slot in slots_by_wind[wind]:
                round_id = slot[1]
                curve_key = (slot[0], slot[2], slot[3])
                values = np.abs(rl_curves[curve_key][:, blade - 1]) / 1e6
                ax.plot(times, values, color=ROUND_COLORS[round_id], linewidth=0.82,
                    alpha=0.78, zorder=3, rasterized=True)
            for category in ('TN', 'TP', 'FP', 'FN'):
                x, y = marker_data.get((wind, blade, horizon, category), ([], []))
                if not x:
                    continue
                kwargs = {'s': 8 if category == 'TN' else 23,
                          'alpha': 0.28 if category == 'TN' else 0.86,
                          'color': CLASS_COLORS[category], 'linewidths': 0.55,
                          'rasterized': True, 'zorder': 6}
                if category == 'TP':
                    kwargs['marker'] = '^'
                elif category == 'FP':
                    kwargs['marker'] = 'x'
                    kwargs['linewidths'] = 0.9
                elif category == 'FN':
                    kwargs['marker'] = 'v'
                else:
                    kwargs['marker'] = '.'
                ax.scatter(x, y, **kwargs)
            counts = confusion.get((wind, horizon, blade), _confusion_counts())
            n = sum(counts.values())
            ax.text(0.015, 0.985,
                    f"n={n:,}  TP={counts['TP']} FP={counts['FP']} FN={counts['FN']} TN={counts['TN']:,}",
                    transform=ax.transAxes, va='top', ha='left', fontsize=7.1,
                    bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.78, 'pad': 2})
            ax.set_xlim(0.0, SIMULATION_END_S)
            ax.set_ylim(bottom=0.0)
            ax.grid(True, linewidth=0.35, alpha=0.3)
            if blade == 1:
                ax.set_title(f'wind{wind}', fontsize=11, pad=7)
            if column == 0:
                ax.set_ylabel(f'叶片{blade} |My| (MNm)')
            if blade == 3:
                ax.set_xlabel('仿真时间 (s)')

    line_handles = [Line2D([0], [0], color='#151515', lw=1.4, label='PID 实测 |My|')]
    line_handles += [Line2D([0], [0], color=ROUND_COLORS[i], lw=1.4, label=f'RL 第{i}轮实测 |My|')
                     for i in selected_rounds]
    line_handles.append(Line2D([0], [0], color='#5A2020', lw=1.2, ls='--', label='PID 全程最大值阈值'))
    marker_handles = [
        Line2D([0], [0], marker='^', color='w', markerfacecolor=CLASS_COLORS['TP'],
               markeredgecolor=CLASS_COLORS['TP'], markersize=7, label=CLASS_LABELS['TP']),
        Line2D([0], [0], marker='x', color=CLASS_COLORS['FP'], markersize=7, label=CLASS_LABELS['FP']),
        Line2D([0], [0], marker='v', color='w', markerfacecolor=CLASS_COLORS['FN'],
               markeredgecolor=CLASS_COLORS['FN'], markersize=7, label=CLASS_LABELS['FN']),
        Line2D([0], [0], marker='.', color=CLASS_COLORS['TN'], markersize=7, label=CLASS_LABELS['TN']),
    ]
    rounds_text = '、'.join(str(round_id) for round_id in selected_rounds)
    fig.suptitle(f'RL 第{rounds_text}轮与 PID 实际叶根载荷及超 PID 峰值点位分类（H={horizon:.2f} s）',
                 fontsize=16, y=0.985)
    fig.text(0.5, 0.952,
             '分类点落在该预测窗口内的 RL 实际 |My| 峰值时刻；预测是否超限按“预测窗口峰值 > 对应 PID 叶片全程最大值”判定。',
             ha='center', va='center', fontsize=9)
    fig.legend(handles=line_handles + marker_handles, loc='upper center', ncol=9,
               bbox_to_anchor=(0.5, 0.925), frameon=False, fontsize=8.2)
    fig.text(0.5, 0.014,
             summary_note + '  预测筛选：支持域内 + approximate ZOH + 窗口内 Actor policy version 不变；TN 点为显示抽样，表格计数不抽样。',
             ha='center', va='bottom', fontsize=8.2, wrap=True)
    fig.tight_layout(rect=(0.025, 0.045, 0.995, 0.89), h_pad=1.35, w_pad=0.85)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=135, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_run(run_dir: Path, output_dir: Path, selected_rounds: tuple[int, ...] = (4,)) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    output_dir = output_dir.resolve()
    selected_rounds = tuple(sorted(set(int(value) for value in selected_rounds)))
    if not selected_rounds or any(value not in (1, 2, 3, 4) for value in selected_rounds):
        raise ValueError('Plot rounds must be a non-empty subset of 1, 2, 3, 4')
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f'Refusing to overwrite plot output directory: {output_dir}')
    config_path = run_dir / 'config.json'
    records_path = run_dir / 'physical_load_shadow_development.jsonl'
    shadow_summary_path = run_dir / 'physical_load_shadow_summary.json'
    queue_summary_path = run_dir / 'queue_training_summary.json'
    for path in (config_path, records_path, shadow_summary_path, queue_summary_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    config = strict_json(config_path)
    shadow_summary = strict_json(shadow_summary_path)
    queue_summary = strict_json(queue_summary_path)
    if config.get('experiment_name') != run_dir.name:
        raise ValueError('Run directory name/config experiment_name mismatch')
    if records_path.stat().st_size > int(config['load_predictor']['max_total_bytes']):
        raise ValueError('Physical shadow JSONL exceeds configured total-byte budget')
    if queue_summary.get('queue_completed_rounds') != 4.0 or queue_summary.get('queue_total_steps') != 480016.0:
        raise ValueError('This plotter requires the completed 4x4x600 queue run')

    manifest_path = Path(config['queue']['manifest_path'])
    if not manifest_path.is_absolute():
        manifest_path = PROJECT_ROOT / manifest_path
    slots = expand_queue_manifest(strict_json(manifest_path))
    if len(slots) != 16 or {int(slot.condition_id) for slot in slots} != set(EXPECTED_WINDS):
        raise ValueError('Expected exact four-wind, four-round manifest')
    slot_by_key = {}
    slots_by_wind: dict[int, list[tuple[int, int, int, int]]] = {wind: [] for wind in EXPECTED_WINDS}
    for slot in slots:
        key = (int(slot.condition_id), int(slot.condition_instance_id), int(slot.queue_epoch))
        slot_by_key[key] = int(slot.round_id)
        if int(slot.round_id) in selected_rounds:
            slots_by_wind[key[0]].append((key[0], int(slot.round_id), key[1], key[2]))

    references, slot_identity = _read_slot_references(run_dir, slots)
    pid_times: dict[int, np.ndarray] = {}
    pid_signed: dict[int, np.ndarray] = {}
    pid_peaks: dict[int, np.ndarray] = {}
    pid_sha = {}
    for wind, source in references.items():
        pid_path = Path(source['path'])
        times, signed_my, digest = _read_pid_csv(pid_path, expected_sha256=source['sha256'],
                                                expected_trace_signature=source['trace_signature'])
        pid_times[wind] = times
        pid_signed[wind] = signed_my
        pid_peaks[wind] = np.max(np.abs(signed_my), axis=0).astype(np.float64)
        pid_sha[wind] = digest

    selected_slot_keys = {key for key, round_id in slot_by_key.items() if round_id in selected_rounds}
    curves = {key: np.full((FRAME_COUNT, 3), np.nan, dtype=np.float32) for key in selected_slot_keys}
    curve_counts = {key: np.zeros(FRAME_COUNT, dtype=np.uint8) for key in slot_by_key}
    confusion: dict[tuple[Any, ...], dict[str, int]] = defaultdict(_confusion_counts)
    markers: dict[tuple[int, int, float, str], tuple[list[float], list[float]]] = {}
    exclusion_counts: dict[tuple[int, float], dict[str, int]] = defaultdict(
        lambda: {'records': 0, 'unsupported': 0, 'non_zoh': 0, 'actor_not_frozen': 0,
                 'classified_records': 0, 'tn_markers_plotted': 0})
    record_identity = Counter()
    record_rows = 0
    max_record_bytes = 0
    stream_sha = hashlib.sha256()
    actual_pid_reference_mismatches = 0

    with records_path.open('rb') as handle:
        while True:
            encoded = handle.readline(MAX_RECORD_BYTES + 1)
            if not encoded:
                break
            if len(encoded) > MAX_RECORD_BYTES:
                raise ValueError(f'Oversized physical shadow row at record {record_rows + 1}')
            stream_sha.update(encoded)
            max_record_bytes = max(max_record_bytes, len(encoded))
            record_rows += 1
            if record_rows > MAX_RECORDS:
                raise ValueError('Physical shadow row count exceeds the configured cap')
            row = json.loads(encoded)
            wind = int(row['condition_id'])
            instance = int(row['condition_instance_id'])
            epoch = int(row['queue_epoch'])
            slot = (wind, instance, epoch)
            if slot not in slot_by_key:
                raise ValueError(f'Physical shadow record belongs to non-manifest slot: {slot}')
            round_id = slot_by_key[slot]
            h = float(row['horizon_s'])
            if h not in (0.1, 0.2):
                raise ValueError(f'Unexpected horizon in run: {h}')
            record_identity[str(row.get('identity_status', 'missing'))] += 1
            origin = float(row['origin_time_s'])
            origin_index_float = origin / DT
            origin_index = int(round(origin_index_float))
            if abs(origin_index_float - origin_index) > 1e-5:
                raise ValueError(f'RL origin is off the fixed time grid: {origin}')
            endpoint = float(row['label_endpoint_time_s'])
            frames = int(row['horizon_frames'])
            expected_frames = int(round(h / DT))
            if frames != expected_frames or not math.isclose(endpoint - origin, h, rel_tol=0.0, abs_tol=1e-6):
                raise ValueError('Shadow label horizon/time/frame count is inconsistent')
            signed = np.asarray(row['actual_future_signed_my_nm_by_blade'], dtype=np.float32)
            if signed.shape != (frames, 3) or not np.isfinite(signed).all():
                raise ValueError(f'Invalid future My trace in slot={slot} origin={origin}')
            actual_peak = np.max(np.abs(signed), axis=0)
            reported_actual = np.asarray(row['actual_future_peak_nm_by_blade'], dtype=np.float64)
            if reported_actual.shape != (3,) or not np.allclose(actual_peak, reported_actual, rtol=0.0, atol=1e-4):
                raise ValueError('Actual future My series does not reproduce its stored window peak')
            pid_peak = np.asarray(row['pid_peak_abs_nm'], dtype=np.float64)
            if pid_peak.shape != (3,) or not np.allclose(pid_peak, pid_peaks[wind], rtol=0.0, atol=1e-3):
                actual_pid_reference_mismatches += 1
                raise ValueError(f'Physical record PID peak does not match the hashed raw curve for wind{wind}')

            # Rebuild each RL curve from its actual origin sample plus contiguous
            # 0.10 s future labels. Shared endpoints are checked, never averaged.
            if h == 0.1 and round_id in selected_rounds:
                curve = curves[slot]
                seen = curve_counts[slot]
                _put_curve_sample(curve, seen, origin_index,
                                  np.asarray(row['origin_signed_my_nm_by_blade'], dtype=np.float32), slot=slot)
                for offset, sample in enumerate(signed, start=1):
                    _put_curve_sample(curve, seen, origin_index + offset, sample, slot=slot)

            if round_id not in selected_rounds:
                continue
            breakdown = exclusion_counts[(wind, h)]
            breakdown['records'] += 1
            supported = row.get('prediction_supported') is True
            approximate_zoh = row.get('approximate_zoh') is True
            actor_frozen = row.get('actor_policy_window_frozen') is True
            breakdown['unsupported'] += int(not supported)
            breakdown['non_zoh'] += int(not approximate_zoh)
            breakdown['actor_not_frozen'] += int(not actor_frozen)
            predicted = row.get('predicted_peak_nm_by_blade')
            if not (supported and approximate_zoh and actor_frozen):
                continue
            if not isinstance(predicted, list) or len(predicted) != 3:
                raise ValueError('A classified physical point has no three-blade prediction')
            predicted = np.asarray(predicted, dtype=np.float64)
            if not np.isfinite(predicted).all():
                raise ValueError('Non-finite predicted peak in a supported record')
            breakdown['classified_records'] += 1
            frame_id = int(row.get('origin_frame_id', origin_index))
            for blade in range(3):
                category = _classify(float(predicted[blade]), float(actual_peak[blade]),
                                     float(pid_peaks[wind][blade]))
                blade_id = blade + 1
                local_counts = confusion[(wind, h, blade_id)]
                allwind_counts = confusion[('ALL', h, blade_id)]
                allblade_counts = confusion[(wind, h, 'ALL')]
                overall_counts = confusion[('ALL', h, 'ALL')]
                for target in (local_counts, allwind_counts, allblade_counts, overall_counts):
                    target[category] += 1
                # Actual label window is (t, t+H]; place the marker on the RL
                # absolute-My curve at the measured maximum's sample time.
                peak_offset = int(np.argmax(np.abs(signed[:, blade])))
                marker_time = origin + (peak_offset + 1) * DT
                marker_value = float(abs(signed[peak_offset, blade])) / 1e6
                draw_marker = category != 'TN' or frame_id % TN_MARKER_EVERY_FRAMES == 0
                if draw_marker:
                    x, y = markers.setdefault((wind, blade_id, h, category), ([], []))
                    x.append(marker_time)
                    y.append(marker_value)
                    if category == 'TN':
                        breakdown['tn_markers_plotted'] += 1

    if record_rows != 191664:
        raise ValueError(f'Expected 191664 physical rows for this completed run, found {record_rows}')
    if stream_sha.hexdigest() != shadow_summary.get('records_file_sha256'):
        raise ValueError('Streamed physical records SHA does not match finalized worker summary')
    if shadow_summary.get('counters', {}).get('records_written') != record_rows:
        raise ValueError('Streamed row count does not match finalized worker summary')

    slot_curve_report = []
    for wind, round_id, instance in sorted((key[0], round_id, key[1])
                                           for key, round_id in slot_by_key.items()
                                           if round_id in selected_rounds):
        key = next(key for key, rid in slot_by_key.items() if key[0] == wind and key[1] == instance and rid == round_id)
        count = curve_counts[key]
        present = int(np.count_nonzero(count))
        missing_indices = np.flatnonzero(count == 0)
        slot_curve_report.append({
            'wind': wind, 'round': round_id, 'instance': instance, 'epoch': key[2],
            'curve_samples_present': present, 'curve_samples_expected': FRAME_COUNT,
            'coverage_pct': 100.0 * present / FRAME_COUNT,
            'first_missing_time_s': None if not len(missing_indices) else float(missing_indices[0] * DT),
            'last_missing_time_s': None if not len(missing_indices) else float(missing_indices[-1] * DT),
        })

    summary_note = (
        'RL曲线由影子origin与未来My标签重建，预热期0–0.98s未记录；PID raw CSV SHA及其Time+风速签名已核验。'
        '但16个RL槽位均development_unverified（风速轨迹不满足逐帧身份容差），点位比较仅为经验性结果，不是正式配对验收。'
    )
    plot_files = []
    times = np.arange(FRAME_COUNT, dtype=np.float32) * DT
    output_dir.mkdir(parents=True, exist_ok=True)
    for horizon in (0.1, 0.2):
        round_suffix = '_'.join(str(round_id) for round_id in selected_rounds)
        filename = f'rl_pid_round{round_suffix}_actual_load_and_point_classification_H{int(horizon * 1000):03d}ms.png'
        plot_path = output_dir / filename
        _make_plot(plot_path, horizon=horizon, times=times, pid_curves=pid_signed,
                   rl_curves=curves, pid_peaks=pid_peaks, slots_by_wind=slots_by_wind,
                   selected_rounds=selected_rounds,
                   marker_data=markers, confusion=confusion, summary_note=summary_note)
        plot_files.append({'file': filename, 'sha256': file_sha256(plot_path),
                           'bytes': plot_path.stat().st_size, 'horizon_s': horizon})

    csv_path = output_dir / 'point_classification_metrics.csv'
    csv_rows = _classification_rows(confusion)
    fields = ['wind', 'horizon_s', 'blade', 'classified_points', 'TP', 'FP', 'FN', 'TN',
              'accuracy_pct', 'precision_pct', 'recall_pct', 'specificity_pct', 'actual_exceedance_pct']
    with csv_path.open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)

    plot_manifest = {
        'schema_version': 'physical_shadow_rl_pid_curve_plot_v1',
        'run_id': run_dir.name,
        'selected_rl_rounds': list(selected_rounds),
        'training_summary': {
            'queue_training_valid': bool(queue_summary.get('queue_training_valid')),
            'completed_rounds': int(queue_summary['queue_completed_rounds']),
            'total_steps': int(queue_summary['queue_total_steps']),
        },
        'inputs': {
            'physical_records_path': str(records_path), 'physical_records_sha256': stream_sha.hexdigest(),
            'physical_records_bytes': records_path.stat().st_size, 'physical_record_count': record_rows,
            'physical_worker_summary_sha256': file_sha256(shadow_summary_path),
            'queue_manifest_sha256': file_sha256(manifest_path),
            'pid_raw_csvs': {str(wind): {'path': references[wind]['path'], 'sha256': pid_sha[wind],
                                          'trace_signature': references[wind]['trace_signature']}
                             for wind in EXPECTED_WINDS},
        },
        'identity': {
            'strict_identity_slot_count': sum(item['identity_status'] == 'strict_case_matched' for item in slot_identity),
            'total_slots': len(slot_identity), 'per_slot': slot_identity,
            'interpretation': 'PID source CSV hash and its time/wind signature were checked; RL slots remain operator-unverified because their measured wind traces differ from PID.',
        },
        'curves': slot_curve_report,
        'classification': {
            'threshold': 'strictly greater than each blade PID full-window maximum absolute My',
            'positive_label': 'actual_future_peak_nm_by_blade > PID full-window peak',
            'prediction_label': 'predicted_peak_nm_by_blade > PID full-window peak',
            'point_population': f'round in {list(selected_rounds)} AND prediction_supported AND approximate_zoh AND actor_policy_window_frozen',
            'correct_and_error_codes': {'TP': 'predicted exceed and actual exceed', 'FP': 'predicted exceed, actual not exceed',
                                        'FN': 'predicted not exceed, actual exceed', 'TN': 'neither exceed'},
            'tn_markers': {'all_tn_counted': True, 'figure_tn_marker_sample_every_origin_frame_id': TN_MARKER_EVERY_FRAMES},
            'metrics_file': csv_path.name,
            'identity_qualification': 'descriptive only; not strict paired-PID acceptance, calibration, event recall or safety guarantee',
            'excluded_counts_by_wind_horizon': {f'{wind}_H{h:.2f}': values for (wind, h), values in exclusion_counts.items()},
        },
        'plots': plot_files,
        'analyzer_max_rss_kib': int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        'source_was_streamed_line_by_line': True,
        'source_rows_retained_in_memory': False,
    }
    manifest_path_out = output_dir / 'plot_manifest.json'
    with manifest_path_out.open('x', encoding='utf-8') as handle:
        json.dump(plot_manifest, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    readme = output_dir / 'README.md'
    with readme.open('x', encoding='utf-8') as handle:
        handle.write(
            f"# RL 第{','.join(map(str, selected_rounds))}轮 / PID 实际载荷曲线与影子超限点位\n\n"
            '图中黑线为同风况 PID raw CSV 实测 |My|；彩线为对应 wind、RL round 的实测 |My|；'
            '虚线为该 PID 轨迹每片叶片的全窗最大 |My|。点位表示影子在预测 origin 后，'
            'H=0.10/0.20 s 窗口内 RL 实测 |My| 最大的采样。\n\n'
            '- 绿三角 TP：预测超 PID 峰值，窗口实测也超。\n'
            '- 橙叉 FP：预测超，窗口实测未超。\n'
            '- 红倒三角 FN：预测未超，窗口实测超。\n'
            f'- 灰点 TN：预测与实测均未超；图上每 {TN_MARKER_EVERY_FRAMES * DT:g} s 确定性抽样显示，CSV 仍统计全部 TN。\n\n'
            '点位分类总体仅纳入模型支持、近似 ZOH、预测区间 Actor policy version 不变的窗口；'
            '但全批 16 个 RL 槽位仍未通过严格 PID 风速轨迹身份核验。因此这些图和混淆矩阵是'
            '本次试验的经验性点位比较，不是独立验收、校准覆盖或安全保证。预测发生于动作已发送之后，'
            '不能解释为发送前保护。\n\n'
            '每张图每个面板标题下的 TP/FP/FN/TN 是完整计数，图上的 TN marker 为抽样。'
            '严格逐叶片指标及来源身份见 `plot_manifest.json` 与 `point_classification_metrics.csv`。\n'
        )
    return {
        'output_dir': str(output_dir), 'plots': plot_files,
        'classification_csv': str(csv_path), 'manifest': str(manifest_path_out),
        'records_rows': record_rows, 'records_sha256': stream_sha.hexdigest(),
        'slot_curve_samples': slot_curve_report,
        'strict_identity_slots': plot_manifest['identity']['strict_identity_slot_count'],
        'point_rows': len(csv_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True,
                        help='Completed RL/experiments/<RUN_ID> directory')
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='New, non-existing output directory; existing output is never overwritten')
    parser.add_argument('--rounds', type=int, nargs='+', default=[4], choices=(1, 2, 3, 4),
                        help='RL rounds to plot and classify; default: round 4 only')
    args = parser.parse_args()
    print(json.dumps(plot_run(args.run_dir, args.output_dir, tuple(args.rounds)),
                     ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()