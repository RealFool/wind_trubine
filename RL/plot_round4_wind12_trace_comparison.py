"""Plot the PID and round-4 condition-12 wind-speed traces, streaming CSV input.

The raw frame file has no condition-id column, so TRAIN slots are segmented by
Time resets and selected by the documented queue order [8, 11, 12, 16]. Only the
selected 30,001 time/wind pairs are retained in memory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from decimal import Decimal, InvalidOperation
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RUN_ID = "physical_round4_frozen_events_20261015_003"
QUEUE_ORDER = (8, 11, 12, 16)
DT_S = 0.02
END_TIME_S = 600.0
END_MARKER_TIME_S = 600.01
MISMATCH_THRESHOLD_MPS = 0.01


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fractional_digits(value: str) -> int:
    try:
        exponent = Decimal(value.strip()).as_tuple().exponent
    except InvalidOperation as exc:
        raise ValueError(f"Invalid decimal field: {value!r}") from exc
    return max(0, -exponent)


def _load_pid_trace(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    times: list[float] = []
    winds: list[float] = []
    max_digits = 0
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or "Time" not in reader.fieldnames or "I_AverageWindSpeed" not in reader.fieldnames:
            raise ValueError(f"PID CSV lacks Time/I_AverageWindSpeed columns: {path}")
        for row_number, row in enumerate(reader, start=2):
            t = float(row["Time"])
            wind_text = row["I_AverageWindSpeed"]
            wind = float(wind_text)
            if not math.isfinite(t) or not math.isfinite(wind):
                raise ValueError(f"Non-finite PID time/wind at CSV row {row_number}")
            times.append(t)
            winds.append(wind)
            max_digits = max(max_digits, _fractional_digits(wind_text))
    return np.asarray(times, dtype=np.float64), np.asarray(winds, dtype=np.float64), max_digits


def _load_r4_condition_12(path: Path) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    target_segment = QUEUE_ORDER.index(12)
    target_times: list[float] = []
    target_winds: list[float] = []
    target_digits = 0
    segment = -1
    previous_time: float | None = None
    segment_row_counts = [0] * len(QUEUE_ORDER)
    end_marker_count = [0] * len(QUEUE_ORDER)

    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or "Time" not in reader.fieldnames or "I_AverageWindSpeed" not in reader.fieldnames:
            raise ValueError(f"R4 raw frames lack Time/I_AverageWindSpeed columns: {path}")
        for row_number, row in enumerate(reader, start=2):
            t = float(row["Time"])
            if not math.isfinite(t):
                raise ValueError(f"Non-finite R4 time at CSV row {row_number}")
            if previous_time is None or t < previous_time - 1e-9:
                segment += 1
                if segment >= len(QUEUE_ORDER):
                    raise ValueError(f"More than {len(QUEUE_ORDER)} queue segments detected")
            previous_time = t
            if segment < 0:
                continue
            if abs(t - END_MARKER_TIME_S) <= 1e-9:
                end_marker_count[segment] += 1
            if segment != target_segment or t < -1e-9 or t > END_TIME_S + 1e-9:
                continue
            wind_text = row["I_AverageWindSpeed"]
            wind = float(wind_text)
            if not math.isfinite(wind):
                raise ValueError(f"Non-finite R4 wind at CSV row {row_number}")
            target_times.append(t)
            target_winds.append(wind)
            target_digits = max(target_digits, _fractional_digits(wind_text))
            segment_row_counts[segment] += 1

    if segment != len(QUEUE_ORDER) - 1:
        raise ValueError(f"Expected {len(QUEUE_ORDER)} queue segments, found {segment + 1}")
    return (
        np.asarray(target_times, dtype=np.float64),
        np.asarray(target_winds, dtype=np.float64),
        target_digits,
        segment_row_counts[target_segment],
        end_marker_count[target_segment],
    )


def _validate_trace(times: np.ndarray, winds: np.ndarray, label: str) -> None:
    expected_count = round(END_TIME_S / DT_S) + 1
    if len(times) != expected_count or len(winds) != expected_count:
        raise ValueError(f"{label}: expected {expected_count} TRAIN samples, found {len(times)}")
    expected_times = np.arange(expected_count, dtype=np.float64) * DT_S
    errors = np.abs(times - expected_times)
    if float(errors.max(initial=0.0)) > 1e-8:
        raise ValueError(f"{label}: time grid mismatch (max abs error {errors.max():.12g} s)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    run_dir = project_root / "RL" / "experiments" / args.run_id
    pid_path = project_root / "bladed-data-new" / "raw" / "data_windspeed12.csv"
    r4_path = run_dir / "bladed_raw_frames.csv"
    output_path = args.output or run_dir / "physical_evaluation" / "wind12_pid_vs_r4_wind_trace.png"
    output_path = output_path.resolve()

    pid_time, pid_wind, pid_digits = _load_pid_trace(pid_path)
    r4_time, r4_wind, r4_digits, r4_rows, end_markers = _load_r4_condition_12(r4_path)
    _validate_trace(pid_time, pid_wind, "PID wind12")
    _validate_trace(r4_time, r4_wind, "R4 condition12")

    time_error = np.abs(r4_time - pid_time)
    if float(time_error.max(initial=0.0)) > 1e-8:
        raise ValueError(f"PID/R4 sample times are not aligned (max abs error {time_error.max():.12g} s)")

    delta = r4_wind - pid_wind
    abs_delta = np.abs(delta)
    exceed_count = int(np.count_nonzero(abs_delta > MISMATCH_THRESHOLD_MPS))
    n = len(delta)
    max_rounding_bound = 0.5 * (10.0 ** (-pid_digits) + 10.0 ** (-r4_digits))
    level_correlation = float(np.corrcoef(pid_wind, r4_wind)[0, 1])
    increment_correlation = float(np.corrcoef(np.diff(pid_wind), np.diff(r4_wind))[0, 1])
    lag_correlations = []
    for lag_samples in range(-5, 6):
        if lag_samples < 0:
            pid_slice = pid_wind[-lag_samples:]
            r4_slice = r4_wind[:lag_samples]
        elif lag_samples > 0:
            pid_slice = pid_wind[:-lag_samples]
            r4_slice = r4_wind[lag_samples:]
        else:
            pid_slice = pid_wind
            r4_slice = r4_wind
        lag_correlations.append((float(np.corrcoef(pid_slice, r4_slice)[0, 1]), lag_samples))
    best_lag_correlation, best_lag_samples = max(lag_correlations)
    summary = {
        "run_id": args.run_id,
        "condition_id": 12,
        "r4_slot_index_zero_based": QUEUE_ORDER.index(12),
        "pid_file": str(pid_path.relative_to(project_root)),
        "pid_sha256": _sha256(pid_path),
        "r4_raw_frames_file": str(r4_path.relative_to(project_root)),
        "r4_raw_frames_sha256": _sha256(r4_path),
        "r4_slot_train_sample_count": r4_rows,
        "r4_slot_end_marker_count": end_markers,
        "sample_count": n,
        "sample_interval_s": DT_S,
        "pid_time_start_end_s": [float(pid_time[0]), float(pid_time[-1])],
        "r4_time_start_end_s": [float(r4_time[0]), float(r4_time[-1])],
        "max_time_grid_error_s": float(time_error.max(initial=0.0)),
        "wind_difference_definition": "R4 condition-12 I_AverageWindSpeed minus PID data_windspeed12.csv I_AverageWindSpeed",
        "r4_decimal_places_max": r4_digits,
        "pid_decimal_places_max": pid_digits,
        "rounding_only_difference_upper_bound_mps": max_rounding_bound,
        "absolute_difference_threshold_mps": MISMATCH_THRESHOLD_MPS,
        "samples_above_threshold": exceed_count,
        "fraction_above_threshold": exceed_count / n,
        "mae_mps": float(abs_delta.mean()),
        "mean_signed_difference_mps": float(delta.mean()),
        "p95_absolute_difference_mps": float(np.quantile(abs_delta, 0.95)),
        "max_absolute_difference_mps": float(abs_delta.max()),
        "pearson_correlation_wind_level": level_correlation,
        "pearson_correlation_first_difference": increment_correlation,
        "best_level_correlation_lag_within_plus_minus_5_samples": {
            "lag_samples_r4_relative_to_pid": best_lag_samples,
            "lag_s": best_lag_samples * DT_S,
            "pearson_r": best_lag_correlation,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, (ax_wind, ax_diff) = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    ax_wind.plot(pid_time, pid_wind, color="#2369a1", linewidth=0.8, alpha=0.9, label="PID raw: data_windspeed12.csv")
    ax_wind.plot(r4_time, r4_wind, color="#e87522", linewidth=0.8, alpha=0.82, label="R4 raw: condition 12 (slot 3)")
    ax_wind.set_ylabel("I_AverageWindSpeed (m/s)")
    ax_wind.set_title("Wind-speed trace comparison: PID wind12 vs frozen round-4 run")
    ax_wind.grid(True, alpha=0.25)
    ax_wind.legend(loc="upper right")

    ax_diff.plot(r4_time, delta, color="#543c9b", linewidth=0.8, label="R4 - PID")
    ax_diff.axhline(0.0, color="black", linewidth=0.8)
    ax_diff.axhline(MISMATCH_THRESHOLD_MPS, color="#d62728", linewidth=0.9, linestyle="--", label="±0.01 m/s threshold")
    ax_diff.axhline(-MISMATCH_THRESHOLD_MPS, color="#d62728", linewidth=0.9, linestyle="--")
    ax_diff.fill_between(r4_time, -MISMATCH_THRESHOLD_MPS, MISMATCH_THRESHOLD_MPS, color="#2ca02c", alpha=0.12)
    ax_diff.set_xlabel("Simulation time (s)")
    ax_diff.set_ylabel("Wind difference (m/s)")
    ax_diff.grid(True, alpha=0.25)
    ax_diff.legend(loc="upper right")
    annotation = (
        f"|Δ| > 0.01 m/s: {exceed_count:,}/{n:,} ({100 * exceed_count / n:.2f}%)\n"
        f"MAE={abs_delta.mean():.5f} m/s; max |Δ|={abs_delta.max():.5f} m/s\n"
        f"Pearson r: level={level_correlation:.5f}; step change={increment_correlation:.5f}\n"
        f"Decimal serialization rounding bound ≤ {max_rounding_bound:.2g} m/s"
    )
    ax_diff.text(
        0.012, 0.96, annotation, transform=ax_diff.transAxes, va="top", ha="left",
        fontsize=9, bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.88, "edgecolor": "#999999"}
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(figure)

    summary_path = output_path.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"plot": str(output_path), "summary": str(summary_path), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
