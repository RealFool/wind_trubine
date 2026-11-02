from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from RL.infer.plot_inference_vs_pid import (
    BLADE_ROOT_MY_FIELDS,
    _load_inference_trace,
    _load_pid_trace,
    _strict_total_energy_mwh,
)


@dataclass(frozen=True)
class RoundMetrics:
    round_id: int
    inference_dir: Path
    matched_steps: int
    duration_s: float
    rl_mean_power_mw: float
    pid_mean_power_mw: float
    rl_energy_mwh: float
    pid_energy_mwh: float
    rl_mean_blade_peak_mnm: float
    pid_mean_blade_peak_mnm: float
    rl_blade_peak_mnm: tuple[float, float, float]
    pid_blade_peak_mnm: tuple[float, float, float]
    fallback_fraction: float
    ood_mean: float

    @property
    def power_delta_percent(self) -> float:
        if abs(self.pid_mean_power_mw) <= 1.0e-12:
            return 0.0
        return (self.rl_mean_power_mw / self.pid_mean_power_mw - 1.0) * 100.0

    @property
    def energy_delta_percent(self) -> float:
        if abs(self.pid_energy_mwh) <= 1.0e-12:
            return 0.0
        return (self.rl_energy_mwh / self.pid_energy_mwh - 1.0) * 100.0

    @property
    def load_reduction_percent(self) -> float:
        if abs(self.pid_mean_blade_peak_mnm) <= 1.0e-12:
            return 0.0
        return (1.0 - self.rl_mean_blade_peak_mnm / self.pid_mean_blade_peak_mnm) * 100.0

    @property
    def blade_load_reduction_percent(self) -> tuple[float, float, float]:
        return tuple(
            (1.0 - rl / pid) * 100.0 if abs(pid) > 1.0e-12 else 0.0
            for rl, pid in zip(self.rl_blade_peak_mnm, self.pid_blade_peak_mnm)
        )


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
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _finite(value: Any, name: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f'{name} must be finite')
    return parsed


def _metrics_for_round(
    round_id: int,
    inference_dir: Path,
    pid_csv: Path,
    pid_start_row: int,
) -> RoundMetrics:
    inference_csv = inference_dir / 'bladed_inference_steps.csv'
    if not inference_csv.is_file():
        raise FileNotFoundError(f'Missing inference record file: {inference_csv}')
    model = _load_inference_trace(inference_csv)
    pid = _load_pid_trace(pid_csv, start_row=pid_start_row, row_count=len(model.global_steps))
    rl_energy_mwh = _strict_total_energy_mwh(model.trace.power_w, model.trace.raw_time_s)
    pid_energy_mwh = _strict_total_energy_mwh(pid.power_w, pid.raw_time_s)
    rl_blade_peak_mnm = tuple(np.max(model.trace.blade_root_my_abs_nm, axis=0) / 1.0e6)
    pid_blade_peak_mnm = tuple(np.max(pid.blade_root_my_abs_nm, axis=0) / 1.0e6)
    return RoundMetrics(
        round_id=round_id,
        inference_dir=inference_dir,
        matched_steps=len(model.global_steps),
        duration_s=float(model.trace.elapsed_time_s[-1]),
        rl_mean_power_mw=float(np.mean(model.trace.power_w) / 1.0e6),
        pid_mean_power_mw=float(np.mean(pid.power_w) / 1.0e6),
        rl_energy_mwh=rl_energy_mwh,
        pid_energy_mwh=pid_energy_mwh,
        rl_mean_blade_peak_mnm=float(np.mean(rl_blade_peak_mnm)),
        pid_mean_blade_peak_mnm=float(np.mean(pid_blade_peak_mnm)),
        rl_blade_peak_mnm=tuple(float(value) for value in rl_blade_peak_mnm),
        pid_blade_peak_mnm=tuple(float(value) for value in pid_blade_peak_mnm),
        fallback_fraction=float(np.mean(model.fallback_active)),
        ood_mean=float(np.mean(model.ood_score)),
    )


def _annotation(axis, x: int, y: float, text: str, color: str, offset: int = 8) -> None:
    axis.annotate(
        text,
        (x, y),
        xytext=(0, offset),
        textcoords='offset points',
        ha='center',
        va='bottom' if offset >= 0 else 'top',
        color=color,
        fontsize=8,
    )


def _round_axis(axis, round_ids: list[int]) -> None:
    axis.set_xticks(round_ids)
    axis.set_xlim(min(round_ids) - 0.25, max(round_ids) + 0.25)
    axis.grid(alpha=0.25)


def _plot_absolute_metrics(output_path: Path, metrics: list[RoundMetrics]) -> None:
    import matplotlib.pyplot as plt

    round_ids = [item.round_id for item in metrics]
    rl_power = [item.rl_mean_power_mw for item in metrics]
    pid_power = [item.pid_mean_power_mw for item in metrics]
    rl_load = [item.rl_mean_blade_peak_mnm for item in metrics]
    pid_load = [item.pid_mean_blade_peak_mnm for item in metrics]
    figure, (power_axis, load_axis) = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

    power_axis.plot(round_ids, rl_power, color='#2563eb', marker='o', linewidth=2.2, label='RL mean power')
    power_axis.axhline(pid_power[0], color='#dc2626', linestyle='--', linewidth=1.8, label='PID reference')
    for round_id, value in zip(round_ids, rl_power):
        _annotation(power_axis, round_id, value, f'{value:.4f}', '#1d4ed8')
    power_axis.set_title('Round Progress: Mean Electrical Power')
    power_axis.set_ylabel('Mean power (MW)')
    power_axis.ticklabel_format(axis='y', style='plain', useOffset=False)
    power_axis.legend(loc='best')
    _round_axis(power_axis, round_ids)

    load_axis.plot(round_ids, rl_load, color='#0f766e', marker='o', linewidth=2.2, label='RL overall max |My|')
    load_axis.axhline(pid_load[0], color='#dc2626', linestyle='--', linewidth=1.8, label='PID reference')
    for round_id, value in zip(round_ids, rl_load):
        _annotation(load_axis, round_id, value, f'{value:.3f}', '#0f766e', offset=8)
    load_axis.set_title('Round Progress: Mean Blade-root My Peak')
    load_axis.set_ylabel('Mean peak |My| (MNm)')
    load_axis.set_xlabel('Training round')
    load_axis.legend(loc='best')
    _round_axis(load_axis, round_ids)

    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(figure)


def _plot_relative_metrics(output_path: Path, metrics: list[RoundMetrics]) -> None:
    import matplotlib.pyplot as plt

    round_ids = [item.round_id for item in metrics]
    power_delta = [item.power_delta_percent for item in metrics]
    load_reduction = [item.load_reduction_percent for item in metrics]
    figure, (power_axis, load_axis) = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

    power_axis.plot(round_ids, power_delta, color='#2563eb', marker='o', linewidth=2.2, label='RL vs PID power')
    power_axis.axhline(0.0, color='#dc2626', linestyle='--', linewidth=1.6, label='PID reference (0%)')
    for round_id, value in zip(round_ids, power_delta):
        _annotation(power_axis, round_id, value, f'{value:+.3f}%', '#1d4ed8', offset=8 if value >= 0.0 else -8)
    power_axis.set_title('Round Progress Relative to PID: Mean Power')
    power_axis.set_ylabel('RL-PID / PID (%)')
    power_axis.legend(loc='best')
    _round_axis(power_axis, round_ids)

    load_axis.plot(round_ids, load_reduction, color='#0f766e', marker='o', linewidth=2.2, label='RL load reduction')
    load_axis.axhline(0.0, color='#dc2626', linestyle='--', linewidth=1.6, label='PID reference (0%)')
    for round_id, value in zip(round_ids, load_reduction):
        _annotation(load_axis, round_id, value, f'{value:+.3f}%', '#0f766e', offset=8 if value >= 0.0 else -8)
    load_axis.set_title('Round Progress Relative to PID: Mean Blade-root My')
    load_axis.set_ylabel('Load reduction (%)')
    load_axis.set_xlabel('Training round')
    load_axis.legend(loc='best')
    _round_axis(load_axis, round_ids)

    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(figure)


def _plot_blade_loads(output_path: Path, metrics: list[RoundMetrics]) -> None:
    import matplotlib.pyplot as plt

    round_ids = [item.round_id for item in metrics]
    figure, axis = plt.subplots(figsize=(13, 6))
    colors = ('#2563eb', '#7c3aed', '#f59e0b')
    for blade_index, color in enumerate(colors):
        rl_values = [item.rl_blade_peak_mnm[blade_index] for item in metrics]
        pid_value = metrics[0].pid_blade_peak_mnm[blade_index]
        axis.plot(
            round_ids,
            rl_values,
            color=color,
            marker='o',
            linewidth=2.0,
            label=f'RL Blade {blade_index + 1}',
        )
        axis.axhline(
            pid_value,
            color=color,
            linestyle=':',
            linewidth=1.2,
            alpha=0.9,
            label=f'PID Blade {blade_index + 1}',
        )
        for round_id, value in zip(round_ids, rl_values):
            _annotation(axis, round_id, value, f'{value:.3f}', color, offset=7 + blade_index * 9)
    axis.set_title('Round Progress: Per-blade Blade-root My Peaks')
    axis.set_ylabel('Peak |My| (MNm)')
    axis.set_xlabel('Training round')
    axis.legend(ncol=3, loc='best')
    _round_axis(axis, round_ids)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(figure)


def _write_summary_csv(output_path: Path, metrics: list[RoundMetrics]) -> None:
    fieldnames = [
        'round',
        'inference_dir',
        'matched_steps',
        'duration_s',
        'rl_mean_power_mw',
        'pid_mean_power_mw',
        'power_delta_percent',
        'rl_energy_mwh',
        'pid_energy_mwh',
        'energy_delta_percent',
        'rl_mean_blade_peak_mnm',
        'pid_mean_blade_peak_mnm',
        'load_reduction_percent',
        'rl_blade_1_peak_mnm',
        'rl_blade_2_peak_mnm',
        'rl_blade_3_peak_mnm',
        'pid_blade_1_peak_mnm',
        'pid_blade_2_peak_mnm',
        'pid_blade_3_peak_mnm',
        'blade_1_load_reduction_percent',
        'blade_2_load_reduction_percent',
        'blade_3_load_reduction_percent',
        'fallback_fraction',
        'ood_mean',
    ]
    with output_path.open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for item in metrics:
            blade_reduction = item.blade_load_reduction_percent
            writer.writerow({
                'round': item.round_id,
                'inference_dir': str(item.inference_dir),
                'matched_steps': item.matched_steps,
                'duration_s': item.duration_s,
                'rl_mean_power_mw': item.rl_mean_power_mw,
                'pid_mean_power_mw': item.pid_mean_power_mw,
                'power_delta_percent': item.power_delta_percent,
                'rl_energy_mwh': item.rl_energy_mwh,
                'pid_energy_mwh': item.pid_energy_mwh,
                'energy_delta_percent': item.energy_delta_percent,
                'rl_mean_blade_peak_mnm': item.rl_mean_blade_peak_mnm,
                'pid_mean_blade_peak_mnm': item.pid_mean_blade_peak_mnm,
                'load_reduction_percent': item.load_reduction_percent,
                'rl_blade_1_peak_mnm': item.rl_blade_peak_mnm[0],
                'rl_blade_2_peak_mnm': item.rl_blade_peak_mnm[1],
                'rl_blade_3_peak_mnm': item.rl_blade_peak_mnm[2],
                'pid_blade_1_peak_mnm': item.pid_blade_peak_mnm[0],
                'pid_blade_2_peak_mnm': item.pid_blade_peak_mnm[1],
                'pid_blade_3_peak_mnm': item.pid_blade_peak_mnm[2],
                'blade_1_load_reduction_percent': blade_reduction[0],
                'blade_2_load_reduction_percent': blade_reduction[1],
                'blade_3_load_reduction_percent': blade_reduction[2],
                'fallback_fraction': item.fallback_fraction,
                'ood_mean': item.ood_mean,
            })


def _summary_payload(metrics: list[RoundMetrics], pid_csv: Path, pid_start_row: int) -> dict[str, Any]:
    return {
        'mode': 'round_progress_v1',
        'pid_csv': str(pid_csv),
        'pid_start_row_zero_based': int(pid_start_row),
        'rounds': [
            {
                'round': item.round_id,
                'inference_dir': str(item.inference_dir),
                'matched_steps': item.matched_steps,
                'duration_s': item.duration_s,
                'rl_mean_power_mw': item.rl_mean_power_mw,
                'pid_mean_power_mw': item.pid_mean_power_mw,
                'power_delta_percent': item.power_delta_percent,
                'rl_energy_mwh': item.rl_energy_mwh,
                'pid_energy_mwh': item.pid_energy_mwh,
                'energy_delta_percent': item.energy_delta_percent,
                'rl_mean_blade_peak_mnm': item.rl_mean_blade_peak_mnm,
                'pid_mean_blade_peak_mnm': item.pid_mean_blade_peak_mnm,
                'load_reduction_percent': item.load_reduction_percent,
                'rl_blade_peak_mnm': list(item.rl_blade_peak_mnm),
                'pid_blade_peak_mnm': list(item.pid_blade_peak_mnm),
                'blade_load_reduction_percent': list(item.blade_load_reduction_percent),
                'fallback_fraction': item.fallback_fraction,
                'ood_mean': item.ood_mean,
            }
            for item in metrics
        ],
    }


def build_round_progress(
    round_dirs: dict[int, Path],
    pid_csv: Path,
    output_dir: Path,
    pid_start_row: int = 0,
) -> dict[str, Any]:
    if not round_dirs:
        raise ValueError('At least one round directory is required')
    if pid_start_row < 0:
        raise ValueError('pid_start_row must be non-negative')
    if not pid_csv.is_file():
        raise FileNotFoundError(f'PID CSV is not a file: {pid_csv}')
    metrics = [
        _metrics_for_round(round_id, round_dirs[round_id], pid_csv, pid_start_row)
        for round_id in sorted(round_dirs)
    ]
    pid_power_values = [item.pid_mean_power_mw for item in metrics]
    pid_load_values = [item.pid_mean_blade_peak_mnm for item in metrics]
    if not np.allclose(pid_power_values, pid_power_values[0], rtol=0.0, atol=1.0e-9):
        raise ValueError('PID mean power changed across rounds; use one matching PID holdout.')
    if not np.allclose(pid_load_values, pid_load_values[0], rtol=0.0, atol=1.0e-9):
        raise ValueError('PID load reference changed across rounds; use one matching PID holdout.')

    output_dir.mkdir(parents=True, exist_ok=True)
    absolute_plot = output_dir / 'round_progress_power_load.png'
    relative_plot = output_dir / 'round_progress_relative_to_pid.png'
    blade_plot = output_dir / 'round_progress_blade_root_peaks.png'
    summary_csv = output_dir / 'round_progress_summary.csv'
    summary_json = output_dir / 'round_progress_summary.json'
    _plot_absolute_metrics(absolute_plot, metrics)
    _plot_relative_metrics(relative_plot, metrics)
    _plot_blade_loads(blade_plot, metrics)
    _write_summary_csv(summary_csv, metrics)
    payload = _summary_payload(metrics, pid_csv, pid_start_row)
    payload.update({
        'absolute_plot': str(absolute_plot),
        'relative_plot': str(relative_plot),
        'blade_plot': str(blade_plot),
        'summary_csv': str(summary_csv),
        'summary_json': str(summary_json),
    })
    with summary_json.open('w', encoding='utf-8') as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, allow_nan=False)
    return payload


def _parse_round_dirs(values: list[str]) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for value in values:
        try:
            round_text, path_text = value.split('=', 1)
            round_id = int(round_text)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'Round directory must use ROUND=PATH syntax: {value!r}') from exc
        if round_id <= 0:
            raise ValueError(f'Round must be positive: {round_id}')
        if round_id in result:
            raise ValueError(f'Duplicate round directory: {round_id}')
        result[round_id] = _resolve_path(path_text)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Plot round-by-round RL power and blade-root load progress against one PID holdout.',
    )
    parser.add_argument(
        '--round-dir',
        action='append',
        required=True,
        metavar='ROUND=PATH',
        help='Inference directory in ROUND=PATH form; repeat for each round.',
    )
    parser.add_argument('--pid-csv', required=True, help='Matching PID raw_infer CSV.')
    parser.add_argument('--pid-start-row', type=int, default=0, help='Zero-based PID row offset.')
    parser.add_argument('--output-dir', required=True, help='Directory for round progress plots and summaries.')
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    round_dirs = _parse_round_dirs(args.round_dir)
    pid_csv = _resolve_path(args.pid_csv)
    output_dir = _resolve_output_dir(args.output_dir)
    summary = build_round_progress(
        round_dirs=round_dirs,
        pid_csv=pid_csv,
        output_dir=output_dir,
        pid_start_row=args.pid_start_row,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
