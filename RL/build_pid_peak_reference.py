from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import statistics
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from RL.utils.constants import DATA_SCHEMA_VERSION


WIND_SPEED_FIELD = 'I_AverageWindSpeed'
SIM_TIME_FIELD = 'Time'
BLADE_ROOT_MY_FIELDS = (
    'Blade root 1 My (Nm)',
    'Blade root 2 My (Nm)',
    'Blade root 3 My (Nm)',
)
_NOMINAL_WIND_PATTERN = re.compile(r'data_windspeed(-?\d+(?:\.\d+)?)', re.IGNORECASE)


def _finite(row: dict[str, str], field: str) -> float:
    value = float(row[field])
    if not math.isfinite(value):
        raise ValueError(field)
    return value


def _quantile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError('Cannot calculate a quantile from no values')
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * weight)


def _nominal_wind_speed(path: Path, observed_winds: list[float]) -> float:
    match = _NOMINAL_WIND_PATTERN.search(path.stem)
    if match is not None:
        return float(match.group(1))
    if not observed_winds:
        raise ValueError(f'No valid wind speeds in {path}')
    return float(statistics.median(observed_winds))


def _window_peaks(
    path: Path,
    target_train_sim_s: float | None,
) -> list[tuple[float, list[float], int, float | None, float | None]]:
    required = {WIND_SPEED_FIELD, *BLADE_ROOT_MY_FIELDS}
    if target_train_sim_s is not None:
        required.add(SIM_TIME_FIELD)
    rows: list[tuple[float | None, float, list[float]]] = []
    with path.open('r', encoding='utf-8-sig', newline='') as file:
        reader = csv.DictReader(file)
        if not required.issubset(set(reader.fieldnames or [])):
            return []
        for row in reader:
            try:
                wind_speed = _finite(row, WIND_SPEED_FIELD)
                values = [abs(_finite(row, field)) for field in BLADE_ROOT_MY_FIELDS]
                sim_time = _finite(row, SIM_TIME_FIELD) if target_train_sim_s is not None else None
            except (KeyError, TypeError, ValueError):
                continue
            rows.append((sim_time, wind_speed, values))
    if not rows:
        return []

    if target_train_sim_s is None:
        peak_abs_nm = [0.0, 0.0, 0.0]
        observed_winds = [row[1] for row in rows]
        for _sim_time, _wind_speed, values in rows:
            peak_abs_nm = [max(peak, value) for peak, value in zip(peak_abs_nm, values, strict=True)]
        return [(_nominal_wind_speed(path, observed_winds), peak_abs_nm, len(rows), None, None)]

    sim_times = [float(row[0]) for row in rows]
    if any(next_time <= current_time for current_time, next_time in zip(sim_times, sim_times[1:])):
        raise ValueError(f'{path} has non-increasing {SIM_TIME_FIELD} values')

    windows: list[tuple[float, list[float], int, float, float]] = []
    start_index = 0
    while start_index < len(rows):
        start_time = sim_times[start_index]
        target_end_time = start_time + target_train_sim_s
        end_index = start_index
        while end_index < len(rows) and sim_times[end_index] < target_end_time:
            end_index += 1
        if end_index >= len(rows):
            break
        segment = rows[start_index:end_index + 1]
        peak_abs_nm = [0.0, 0.0, 0.0]
        observed_winds = [row[1] for row in segment]
        for _sim_time, _wind_speed, values in segment:
            peak_abs_nm = [max(peak, value) for peak, value in zip(peak_abs_nm, values, strict=True)]
        windows.append(
            (
                _nominal_wind_speed(path, observed_winds),
                peak_abs_nm,
                len(segment),
                start_time,
                sim_times[end_index],
            )
        )
        start_index = end_index + 1
    return windows


def build_peak_reference(
    data_root: Path,
    splits: list[str],
    wind_bin_width: float = 0.5,
    peak_quantile: float = 1.0,
    min_windows: int = 1,
    target_train_sim_s: float | None = None,
) -> dict[str, Any]:
    if wind_bin_width <= 0.0 or not math.isfinite(wind_bin_width):
        raise ValueError('wind_bin_width must be positive and finite')
    if not 0.0 <= peak_quantile <= 1.0:
        raise ValueError('peak_quantile must be between 0 and 1')
    if min_windows <= 0:
        raise ValueError('min_windows must be positive')
    if target_train_sim_s is not None and (
        not math.isfinite(target_train_sim_s) or target_train_sim_s <= 0.0
    ):
        raise ValueError('target_train_sim_s must be positive and finite when configured')

    grouped_windows: dict[int, list[tuple[float, list[float]]]] = {}
    files_seen = 0
    files_used = 0
    rows_used = 0
    fixed_duration_window_count = 0
    fixed_duration_source_windows: list[dict[str, Any]] = []
    for split in splits:
        split_dir = data_root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f'Dataset split directory does not exist: {split_dir}')
        for path in sorted(split_dir.glob('*.csv')):
            files_seen += 1
            windows = _window_peaks(path, target_train_sim_s)
            if not windows:
                continue
            files_used += 1
            for window_index, (wind_speed, peak_abs_nm, valid_rows, start_time, end_time) in enumerate(windows):
                bin_key = int(round(wind_speed / wind_bin_width))
                grouped_windows.setdefault(bin_key, []).append((wind_speed, peak_abs_nm))
                rows_used += valid_rows
                if target_train_sim_s is not None:
                    fixed_duration_window_count += 1
                    fixed_duration_source_windows.append(
                        {
                            'split': split,
                            'source_file': path.name,
                            'window_index': window_index,
                            'nominal_wind_speed': wind_speed,
                            'start_sim_time_s': start_time,
                            'end_sim_time_s': end_time,
                            'duration_s': float(end_time - start_time),
                            'rows_used': valid_rows,
                        }
                    )

    rows: list[dict[str, Any]] = []
    for bin_key in sorted(grouped_windows):
        windows = grouped_windows[bin_key]
        if len(windows) < min_windows:
            continue
        wind_speed = float(statistics.median(window[0] for window in windows))
        peak_abs_nm = [
            _quantile([window[1][blade_index] for window in windows], peak_quantile)
            for blade_index in range(3)
        ]
        rows.append(
            {
                'wind_speed': wind_speed,
                'peak_abs_nm': peak_abs_nm,
                'window_count': len(windows),
            }
        )
    if len(rows) < 2:
        if target_train_sim_s is not None and fixed_duration_window_count == 0:
            raise RuntimeError(
                f'No CSV contains a contiguous {target_train_sim_s:g}-second {SIM_TIME_FIELD} window. '
                'Provide real PID traces matching the target TRAIN duration.'
            )
        raise RuntimeError('Fewer than two qualified wind conditions; provide a broader PID dataset.')
    if any(rows[index]['wind_speed'] >= rows[index + 1]['wind_speed'] for index in range(len(rows) - 1)):
        raise RuntimeError('Wind-speed reference rows must be strictly increasing')

    return {
        'reward': {'blade_root_peak_reference': rows},
        'metadata': {
            'data_schema_version': DATA_SCHEMA_VERSION,
            'source_data_root': str(data_root),
            'splits': list(splits),
            'wind_bin_width': wind_bin_width,
            'peak_quantile': peak_quantile,
            'min_windows': min_windows,
            'aggregation': 'per_window_abs_peak_then_per_condition_quantile',
            'wind_condition_source': 'data_windspeed<value> filename when available, otherwise median observed wind',
            'blade_root_fields': list(BLADE_ROOT_MY_FIELDS),
            'target_train_sim_s': target_train_sim_s,
            'window_duration_mode': (
                'full_csv_window' if target_train_sim_s is None else 'non_overlapping_contiguous_sim_time_windows'
            ),
            'simulation_time_field': SIM_TIME_FIELD if target_train_sim_s is not None else None,
            'files_seen': files_seen,
            'files_used': files_used,
            'rows_used': rows_used,
            'fixed_duration_window_count': fixed_duration_window_count,
            'fixed_duration_source_windows': fixed_duration_source_windows,
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Build wind-conditioned PID Blade root My peak references from processed CSV windows.',
    )
    parser.add_argument('--data-root', type=Path, required=True, help='Dataset root containing processed train/ or val/ CSVs.')
    parser.add_argument('--splits', nargs='+', default=['train'], help='Splits used to build the training proxy. Default: train.')
    parser.add_argument('--wind-bin-width', type=float, default=0.5, help='Wind grouping width in m/s. Default: 0.5.')
    parser.add_argument(
        '--peak-quantile',
        type=float,
        default=1.0,
        help='Quantile across per-window PID peaks for each condition. Default: 1.0.',
    )
    parser.add_argument('--min-windows', type=int, default=1, help='Minimum windows per wind condition. Default: 1.')
    parser.add_argument(
        '--target-train-s',
        type=float,
        default=None,
        help='Require non-overlapping contiguous Time windows of this duration before building the reference.',
    )
    parser.add_argument('--output', type=Path, required=True, help='Output JSON path.')
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    document = build_peak_reference(
        data_root=args.data_root,
        splits=args.splits,
        wind_bin_width=args.wind_bin_width,
        peak_quantile=args.peak_quantile,
        min_windows=args.min_windows,
        target_train_sim_s=args.target_train_s,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding='utf-8')
    reference_rows = document['reward']['blade_root_peak_reference']
    print(json.dumps({
        'output': str(args.output),
        'wind_conditions': len(reference_rows),
        'wind_speed_range': [reference_rows[0]['wind_speed'], reference_rows[-1]['wind_speed']],
        'files_used': document['metadata']['files_used'],
        'rows_used': document['metadata']['rows_used'],
        'target_train_sim_s': document['metadata']['target_train_sim_s'],
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()