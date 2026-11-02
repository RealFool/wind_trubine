from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


MINIMUM_SAMPLE_COUNT = 5_000
P99_LIMIT_MS = 20.0
MINIMUM_OUTPUT_FREQUENCY_HZ = 50.0
REQUIRED_FIELDS = {'record_type', 't_rx_monotonic_s', 't_tx_monotonic_s'}


def _finite_float(row: dict[str, str], field: str) -> float:
    value = float(row[field])
    if not math.isfinite(value):
        raise ValueError(field)
    return value


def analyze_edge_performance(records_csv: Path) -> dict[str, Any]:
    with records_csv.open('r', encoding='utf-8-sig', newline='') as file:
        reader = csv.DictReader(file)
        missing = sorted(REQUIRED_FIELDS - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f'Inference records are missing required timing columns: {missing}')
        control_step_count = 0
        invalid_timing_count = 0
        receive_times: list[float] = []
        send_times: list[float] = []
        latency_ms: list[float] = []
        for row in reader:
            if row.get('record_type') != 'control_step':
                continue
            control_step_count += 1
            try:
                receive_time = _finite_float(row, 't_rx_monotonic_s')
                send_time = _finite_float(row, 't_tx_monotonic_s')
            except (KeyError, TypeError, ValueError):
                invalid_timing_count += 1
                continue
            if send_time < receive_time:
                invalid_timing_count += 1
                continue
            receive_times.append(receive_time)
            send_times.append(send_time)
            latency_ms.append((send_time - receive_time) * 1000.0)

    valid_sample_count = len(latency_ms)
    reasons: list[str] = []
    timing_order_valid = True
    output_frequency_hz = 0.0
    if invalid_timing_count > 0:
        reasons.append(f'{invalid_timing_count} control steps have invalid receive/send timestamps.')
    if valid_sample_count < 2:
        timing_order_valid = False
        reasons.append('At least two valid control-step timestamps are required for output frequency.')
    else:
        send_deltas = np.diff(np.asarray(send_times, dtype=np.float64))
        if np.any(send_deltas <= 0.0):
            timing_order_valid = False
            reasons.append('Action send timestamps must be strictly increasing.')
        else:
            output_frequency_hz = float((valid_sample_count - 1) / (send_times[-1] - send_times[0]))
    if valid_sample_count < MINIMUM_SAMPLE_COUNT:
        reasons.append(f'At least {MINIMUM_SAMPLE_COUNT} valid control steps are required for T1.')

    latency = np.asarray(latency_ms, dtype=np.float64)
    latency_summary = {
        'min': float(np.min(latency)) if valid_sample_count else 0.0,
        'p50': float(np.percentile(latency, 50.0)) if valid_sample_count else 0.0,
        'p95': float(np.percentile(latency, 95.0)) if valid_sample_count else 0.0,
        'p99': float(np.percentile(latency, 99.0)) if valid_sample_count else 0.0,
        'max': float(np.max(latency)) if valid_sample_count else 0.0,
    }
    metrics_valid = bool(invalid_timing_count == 0 and timing_order_valid)
    sample_count_pass = valid_sample_count >= MINIMUM_SAMPLE_COUNT
    latency_pass = bool(valid_sample_count > 0 and latency_summary['p99'] <= P99_LIMIT_MS)
    frequency_pass = bool(output_frequency_hz >= MINIMUM_OUTPUT_FREQUENCY_HZ)
    t1_pass = bool(metrics_valid and sample_count_pass and latency_pass and frequency_pass)
    return {
        'mode': 'edge_performance_v1',
        'records_csv': str(records_csv),
        'valid': metrics_valid,
        'reason': ' '.join(reasons),
        'control_step_count': control_step_count,
        'valid_sample_count': valid_sample_count,
        'invalid_timing_count': invalid_timing_count,
        'latency_ms': latency_summary,
        'output_frequency_hz': output_frequency_hz,
        'criteria': {
            'minimum_sample_count': MINIMUM_SAMPLE_COUNT,
            'p99_e2e_ms_max': P99_LIMIT_MS,
            'output_frequency_hz_min': MINIMUM_OUTPUT_FREQUENCY_HZ,
            'T1_rule': 'P99(T_e2e) <= 20 ms and f_out >= 50 Hz and valid samples >= 5000',
        },
        'sample_count_pass': sample_count_pass,
        'p99_latency_pass': latency_pass,
        'output_frequency_pass': frequency_pass,
        'T1_pass': t1_pass,
    }


def write_edge_performance_report(records_csv: Path, output_path: Path) -> dict[str, Any]:
    report = analyze_edge_performance(records_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as file:
        json.dump(report, file, ensure_ascii=False, indent=2, allow_nan=False)
    report['summary_json'] = str(output_path)
    return report


def _resolve_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if path.exists():
        return path.resolve()
    candidate = PROJECT_ROOT / path
    if candidate.exists():
        return candidate.resolve()
    raise FileNotFoundError(f'Path does not exist: {path_text}')


def _resolve_output_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Calculate the formal Bladed edge-server P99 receive-to-send latency and output-frequency report.',
    )
    parser.add_argument('--inference-dir', required=True, help='Directory containing bladed_inference_steps.csv.')
    parser.add_argument('--output', default='', help='Defaults to <inference-dir>/edge_performance_summary.json.')
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    inference_dir = _resolve_path(args.inference_dir)
    if not inference_dir.is_dir():
        raise NotADirectoryError(f'Inference path is not a directory: {inference_dir}')
    records_csv = inference_dir / 'bladed_inference_steps.csv'
    if not records_csv.is_file():
        raise FileNotFoundError(f'Missing inference records: {records_csv}')
    output_path = _resolve_output_path(args.output) if args.output.strip() else inference_dir / 'edge_performance_summary.json'
    report = write_edge_performance_report(records_csv, output_path)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()