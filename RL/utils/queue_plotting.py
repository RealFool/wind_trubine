from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import re
from typing import Any


QUEUE_STEP_METRICS = (
    'wind_t',
    'power_reward_metric',
    'power_state_real',
    'reward_total',
    'reward_energy_increment_j',
    'reward_blade_root_peak_excess',
    'control_ood_score',
    'control_fallback_active',
    'control_action_projection_applied',
    'control_action_rate_limited',
    'transition_store_eligible',
    'condition_train_active',
    'queue_stage_code',
)

QUEUE_UPDATE_METRICS = (
    'critic1_loss',
    'critic2_loss',
    'critic1_td_error_abs_mean',
    'critic2_td_error_abs_mean',
    'critic_q_abs_gap_mean',
    'q1_mean',
    'q2_mean',
    'target_q_mean',
    'actor_loss',
    'actor_rl_loss',
    'actor_bc_loss',
    'actor_bc_effective_weight',
    'actor_bc_to_rl_ratio',
)

_Q_HEALTH_VALUE_METRICS = (
    'target_q_abs_mean',
    'actor_q_abs_mean',
    'q1_mean',
    'q2_mean',
    'target_q_mean',
    'critic_q_abs_gap_mean',
)

_MAX_METRICS = {'control_ood_score'}
_CONDITION_METRICS = {
    'condition_id',
    'condition_instance_id',
    'queue_completed_round',
    'queue_completed_rounds',
    'queue_condition_raw_frame_count',
    'queue_condition_train_frame_count',
    'queue_condition_eligible_transition_count',
    'queue_condition_stored_transition_count',
    'queue_condition_rejected_transition_count',
    'queue_condition_rejected_boundary_count',
    'queue_condition_rejected_non_train_count',
    'queue_condition_rejected_fallback_count',
    'queue_condition_rejected_unsafe_count',
    'queue_condition_rejected_negative_power_count',
    'queue_condition_rejected_invalid_count',
    'queue_condition_rejected_other_count',
    'queue_condition_replay_count',
    'queue_condition_replay_raw_proxy_count',
    'queue_condition_replay_online_bladed_count',
    'queue_train_elapsed_s',
}
_CONDITION_OUTPUT_FIELDS = (
    'step',
    'round',
    'condition_id',
    'condition_instance_id',
    'train_elapsed_s',
    'raw_frame_count',
    'train_frame_count',
    'eligible_transition_count',
    'stored_transition_count',
    'rejected_transition_count',
    'rejected_boundary_count',
    'rejected_non_train_count',
    'rejected_fallback_count',
    'rejected_unsafe_count',
    'rejected_negative_power_count',
    'rejected_invalid_count',
    'rejected_other_count',
    'replay_count',
    'replay_raw_proxy_count',
    'replay_online_bladed_count',
)


@dataclass
class _MetricBucket:
    sums: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    maxima: dict[str, float] = field(default_factory=dict)
    sample_count: int = 0

    def add(self, name: str, value: float) -> None:
        self.sums[name] = self.sums.get(name, 0.0) + value
        self.counts[name] = self.counts.get(name, 0) + 1
        self.maxima[name] = max(self.maxima.get(name, value), value)
        if name == 'wind_t':
            self.sample_count += 1

    def to_row(self, bucket_index: int, bin_steps: int, metric_names: tuple[str, ...]) -> dict[str, Any]:
        row: dict[str, Any] = {
            'step_bin': bucket_index,
            'step_center': (bucket_index + 0.5) * bin_steps,
            'sample_count': self.sample_count or max(self.counts.values(), default=0),
        }
        for name in metric_names:
            count = self.counts.get(name, 0)
            if count <= 0:
                continue
            row[f'{name}_mean'] = self.sums[name] / count
            if name in _MAX_METRICS:
                row[f'{name}_max'] = self.maxima[name]
        return row


def _parse_float(value: str | None) -> float | None:
    try:
        parsed = float(value) if value is not None else float('nan')
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _stream_metric_csv(
    source_path: Path,
    metric_names: tuple[str, ...],
    bin_steps: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    wanted = set(metric_names)
    buckets: dict[int, _MetricBucket] = {}
    source_row_count = 0
    selected_row_count = 0
    with source_path.open('r', encoding='utf-8', newline='') as file:
        reader = csv.reader(file)
        header = next(reader, None)
        if header is None:
            return [], {'source_row_count': 0, 'selected_row_count': 0, 'step_count': 0}
        indexes = {name: index for index, name in enumerate(header)}
        required = {'step', 'name', 'value'}
        if not required.issubset(indexes):
            raise ValueError(f'{source_path} must contain columns: {", ".join(sorted(required))}')
        for row in reader:
            source_row_count += 1
            if source_row_count % 10_000_000 == 0:
                print(f'[QueuePlot] scanned {source_row_count:,} rows from {source_path.name}', flush=True)
            if max(indexes['step'], indexes['name'], indexes['value']) >= len(row):
                continue
            name = row[indexes['name']]
            if name not in wanted:
                continue
            try:
                step = int(float(row[indexes['step']]))
            except (TypeError, ValueError):
                continue
            value = _parse_float(row[indexes['value']])
            if value is None:
                continue
            bucket = buckets.setdefault(max(0, (step - 1) // bin_steps), _MetricBucket())
            bucket.add(name, value)
            selected_row_count += 1
    rows = [
        buckets[index].to_row(index, bin_steps, metric_names)
        for index in sorted(buckets)
    ]
    return rows, {
        'source_row_count': source_row_count,
        'selected_row_count': selected_row_count,
        'step_count': sum(int(row['sample_count']) for row in rows),
    }


def _read_aggregate_csv(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open('r', encoding='utf-8', newline='') as file:
        for raw_row in csv.DictReader(file):
            row: dict[str, float] = {}
            for key, value in raw_row.items():
                parsed = _parse_float(value)
                if parsed is not None:
                    row[key] = parsed
            rows.append(row)
    return rows


def _write_aggregate_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    fieldnames = list(rows[0])
    for row in rows[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_or_build_aggregate(
    source_path: Path,
    output_dir: Path,
    stem: str,
    metric_names: tuple[str, ...],
    bin_steps: int,
) -> list[dict[str, float]]:
    if bin_steps <= 0:
        raise ValueError('bin_steps must be positive')
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate_path = output_dir / f'{stem}.csv'
    metadata_path = output_dir / f'{stem}_meta.json'
    source_stat = source_path.stat()
    signature = {
        'source_size_bytes': source_stat.st_size,
        'source_mtime_ns': source_stat.st_mtime_ns,
        'bin_steps': bin_steps,
        'metric_names': list(metric_names),
    }
    if aggregate_path.is_file() and metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            metadata = None
        if isinstance(metadata, dict) and all(metadata.get(key) == value for key, value in signature.items()):
            return _read_aggregate_csv(aggregate_path)

    rows, stats = _stream_metric_csv(source_path, metric_names, bin_steps)
    _write_aggregate_csv(aggregate_path, rows)
    metadata_path.write_text(
        json.dumps({**signature, **stats, 'bucket_count': len(rows)}, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    return _read_aggregate_csv(aggregate_path)


def aggregate_queue_step_metrics(
    source_path: str | Path,
    output_dir: str | Path,
    bin_steps: int = 10_000,
) -> list[dict[str, float]]:
    return _load_or_build_aggregate(
        Path(source_path),
        Path(output_dir),
        'queue_train_step_aggregated',
        QUEUE_STEP_METRICS,
        bin_steps,
    )


def aggregate_queue_update_metrics(
    source_path: str | Path,
    output_dir: str | Path,
    bin_steps: int = 10_000,
) -> list[dict[str, float]]:
    return _load_or_build_aggregate(
        Path(source_path),
        Path(output_dir),
        'queue_train_update_aggregated',
        QUEUE_UPDATE_METRICS,
        bin_steps,
    )


def _stream_q_health_metrics(source_path: Path, bin_steps: int) -> list[dict[str, float]]:
    values: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    stage_values: dict[int, list[float]] = defaultdict(list)
    with source_path.open('r', encoding='utf-8', newline='') as file:
        reader = csv.reader(file)
        header = next(reader, None)
        if header is None:
            return []
        indexes = {name: index for index, name in enumerate(header)}
        if not {'step', 'name', 'value'}.issubset(indexes):
            raise ValueError(f'{source_path} must contain columns: name, step, value')
        for row in reader:
            if max(indexes['step'], indexes['name'], indexes['value']) >= len(row):
                continue
            name = row[indexes['name']]
            if name != 'queue_stage_code' and name not in {
                'update_skipped_q_guard',
                'actor_update_skipped_q_guard',
                'actor_update_deferred',
                'critic1_loss',
                'actor_rl_loss',
                'actor_loss',
                *_Q_HEALTH_VALUE_METRICS,
            }:
                continue
            try:
                step = int(float(row[indexes['step']]))
                value = float(row[indexes['value']])
            except (TypeError, ValueError):
                continue
            if not math.isfinite(value):
                continue
            bucket = max(0, (step - 1) // bin_steps)
            if name == 'queue_stage_code':
                stage_values[bucket].append(value)
            else:
                values[bucket][name].append(value)

    rows: list[dict[str, float]] = []
    bucket_indexes = sorted(set(stage_values) | set(values))
    for bucket in bucket_indexes:
        bucket_values = values[bucket]
        stages = stage_values[bucket]
        update_opportunities = len(stages)
        critic_stage_opportunities = sum(1 for stage in stages if stage == 2.0)
        actor_stage_opportunities = sum(1 for stage in stages if stage == 3.0)
        critic_update_count = len(bucket_values['critic1_loss'])
        actor_attempt_count = len(bucket_values['actor_rl_loss'])
        actor_update_count = len(bucket_values['actor_loss'])
        critic_q_guard_count = sum(
            1 for value in bucket_values['update_skipped_q_guard'] if value > 0.0
        )
        actor_q_guard_count = sum(
            1 for value in bucket_values['actor_update_skipped_q_guard'] if value > 0.0
        )
        actor_update_deferred_count = sum(
            1 for value in bucket_values['actor_update_deferred'] if value > 0.0
        )
        row: dict[str, float] = {
            'step_bin': float(bucket),
            'step_center': (bucket + 0.5) * bin_steps,
            'update_opportunity_count': float(update_opportunities),
            'critic_stage_opportunity_count': float(critic_stage_opportunities),
            'actor_stage_opportunity_count': float(actor_stage_opportunities),
            'critic_update_count': float(critic_update_count),
            'actor_attempt_count': float(actor_attempt_count),
            'actor_update_count': float(actor_update_count),
            'critic_q_guard_count': float(critic_q_guard_count),
            'actor_q_guard_count': float(actor_q_guard_count),
            'actor_update_deferred_count': float(actor_update_deferred_count),
            'critic_update_rate': (
                critic_update_count / update_opportunities if update_opportunities else 0.0
            ),
            'actor_attempt_rate': (
                actor_attempt_count / actor_stage_opportunities if actor_stage_opportunities else 0.0
            ),
            'actor_update_rate': actor_update_count / actor_attempt_count if actor_attempt_count else 0.0,
            'critic_q_guard_rate': (
                critic_q_guard_count / update_opportunities if update_opportunities else 0.0
            ),
            'actor_q_guard_rate': actor_q_guard_count / actor_attempt_count if actor_attempt_count else 0.0,
            'actor_deferred_rate': (
                actor_update_deferred_count / critic_stage_opportunities
                if critic_stage_opportunities
                else 0.0
            ),
        }
        for name in _Q_HEALTH_VALUE_METRICS:
            metric_values = bucket_values[name]
            if not metric_values:
                continue
            row[f'{name}_mean'] = sum(metric_values) / len(metric_values)
            row[f'{name}_max'] = max(metric_values)
        rows.append(row)
    return rows


def aggregate_queue_q_health_metrics(
    source_path: str | Path,
    output_dir: str | Path,
    bin_steps: int = 10_000,
) -> list[dict[str, float]]:
    if bin_steps <= 0:
        raise ValueError('bin_steps must be positive')
    source = Path(source_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    aggregate_path = destination / 'queue_q_health_aggregated.csv'
    metadata_path = destination / 'queue_q_health_aggregated_meta.json'
    source_stat = source.stat()
    signature = {
        'source_size_bytes': source_stat.st_size,
        'source_mtime_ns': source_stat.st_mtime_ns,
        'bin_steps': bin_steps,
    }
    if aggregate_path.is_file() and metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            metadata = None
        if isinstance(metadata, dict) and all(metadata.get(key) == value for key, value in signature.items()):
            return _read_aggregate_csv(aggregate_path)
    rows = _stream_q_health_metrics(source, bin_steps)
    _write_aggregate_csv(aggregate_path, rows)
    metadata_path.write_text(
        json.dumps({**signature, 'bucket_count': len(rows)}, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    return rows


def read_queue_condition_metrics(source_path: str | Path) -> list[dict[str, float]]:
    records: dict[int, dict[str, float]] = {}
    source = Path(source_path)
    with source.open('r', encoding='utf-8', newline='') as file:
        reader = csv.reader(file)
        header = next(reader, None)
        if header is None:
            return []
        indexes = {name: index for index, name in enumerate(header)}
        if not {'step', 'name', 'value'}.issubset(indexes):
            return []
        for row in reader:
            if max(indexes['step'], indexes['name'], indexes['value']) >= len(row):
                continue
            name = row[indexes['name']]
            if name not in _CONDITION_METRICS:
                continue
            try:
                step = int(float(row[indexes['step']]))
            except (TypeError, ValueError):
                continue
            value = _parse_float(row[indexes['value']])
            if value is not None:
                records.setdefault(step, {})[name] = value

    rows: list[dict[str, float]] = []
    for step, record in sorted(records.items()):
        if 'condition_id' not in record:
            continue
        rows.append(
            {
                'step': float(step),
                'round': record.get('queue_completed_rounds', 0.0),
                'condition_id': record['condition_id'],
                'condition_instance_id': record.get('condition_instance_id', 0.0),
                'train_elapsed_s': record.get('queue_train_elapsed_s', 0.0),
                'raw_frame_count': record.get('queue_condition_raw_frame_count', 0.0),
                'train_frame_count': record.get('queue_condition_train_frame_count', 0.0),
                'eligible_transition_count': record.get('queue_condition_eligible_transition_count', 0.0),
                'stored_transition_count': record.get('queue_condition_stored_transition_count', 0.0),
                'rejected_transition_count': record.get('queue_condition_rejected_transition_count', 0.0),
                'rejected_boundary_count': record.get('queue_condition_rejected_boundary_count', 0.0),
                'rejected_non_train_count': record.get('queue_condition_rejected_non_train_count', 0.0),
                'rejected_fallback_count': record.get('queue_condition_rejected_fallback_count', 0.0),
                'rejected_unsafe_count': record.get('queue_condition_rejected_unsafe_count', 0.0),
                'rejected_negative_power_count': record.get('queue_condition_rejected_negative_power_count', 0.0),
                'rejected_invalid_count': record.get('queue_condition_rejected_invalid_count', 0.0),
                'rejected_other_count': record.get('queue_condition_rejected_other_count', 0.0),
                'replay_count': record.get('queue_condition_replay_count', 0.0),
                'replay_raw_proxy_count': record.get('queue_condition_replay_raw_proxy_count', 0.0),
                'replay_online_bladed_count': record.get('queue_condition_replay_online_bladed_count', 0.0),
            }
        )
    condition_count = len({int(row['condition_id']) for row in rows})
    if condition_count > 0:
        for index, row in enumerate(rows):
            row['round'] = float(index // condition_count + 1)
    return rows


def read_queue_round_summaries(output_dir: str | Path) -> list[dict[str, float]]:
    output_path = Path(output_dir)
    summaries: list[dict[str, float]] = []
    for path in sorted(output_path.glob('queue_round_*_summary.json')):
        match = re.fullmatch(r'queue_round_(\d+)_summary\.json', path.name)
        if match is None:
            continue
        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue

        def number(name: str, default: float = 0.0) -> float:
            value = payload.get(name, default)
            return float(value) if isinstance(value, (int, float)) else default

        summaries.append(
            {
                'round': number('queue_completed_round', float(match.group(1))),
                'total_steps': number('queue_total_steps'),
                'stored_transitions': number('queue_total_stored_transitions'),
                'rejected_transitions': number('queue_total_rejected_transitions'),
                'update_count': number('queue_update_count'),
                'skipped_conditions': number('queue_skipped_condition_count'),
                'retried_conditions': number('queue_retried_condition_attempt_count'),
                'training_valid': number('queue_training_valid'),
                'all_conditions_covered': number('queue_all_conditions_covered'),
            }
        )
    return sorted(summaries, key=lambda row: row['round'])


def _read_offline_rows(output_dir: Path) -> list[dict[str, float]]:
    summary_path = output_dir / 'offline_eval_final' / 'offline_policy_eval_summary.json'
    if summary_path.is_file():
        try:
            payload = json.loads(summary_path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            payload = None
        if isinstance(payload, dict):
            condition_ids = sorted(
                int(match.group(1))
                for key in payload
                if (match := re.fullmatch(r'condition_(-?\d+)_sample_count', key)) is not None
            )
            rows = []
            for condition_id in condition_ids:
                prefix = f'condition_{condition_id}'
                rows.append(
                    {
                        'condition_id': float(condition_id),
                        'sample_count': float(payload.get(f'{prefix}_sample_count', 0.0)),
                        'torque_mae': float(payload.get(f'{prefix}_torque_mae', 0.0)),
                        'torque_p95': float(payload.get(f'{prefix}_torque_abs_error_p95', 0.0)),
                        'pitch_mae': float(payload.get(f'{prefix}_pitch_mae', 0.0)),
                        'pitch_p95': float(payload.get(f'{prefix}_pitch_abs_error_p95', 0.0)),
                        'ood_max': float(payload.get(f'{prefix}_ood_max', 0.0)),
                        'fallback_fraction': float(payload.get(f'{prefix}_fallback_fraction', 0.0)),
                        'projection_fraction': float(payload.get(f'{prefix}_action_projection_fraction', 0.0)),
                        'rate_limited_fraction': float(payload.get(f'{prefix}_action_rate_limited_fraction', 0.0)),
                        'saturation_fraction': float(payload.get(f'{prefix}_residual_saturation_fraction', 0.0)),
                    }
                )
            if rows:
                return rows

    records_path = output_dir / 'offline_eval_final' / 'offline_policy_eval.csv'
    if not records_path.is_file():
        return []
    grouped: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    with records_path.open('r', encoding='utf-8', newline='') as file:
        for raw_row in csv.DictReader(file):
            try:
                condition_id = int(float(raw_row['condition_id']))
            except (KeyError, TypeError, ValueError):
                continue
            for name in (
                'torque_abs_error',
                'pitch_abs_error',
                'ood_score',
                'fallback_active',
                'action_projection_applied',
                'action_rate_limited',
                'residual_saturated',
            ):
                value = _parse_float(raw_row.get(name))
                if value is not None:
                    grouped[condition_id][name].append(value)

    def percentile(values: list[float], fraction: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        index = (len(ordered) - 1) * fraction
        lower = int(index)
        upper = min(lower + 1, len(ordered) - 1)
        weight = index - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    rows = []
    for condition_id, values in sorted(grouped.items()):
        torque = values['torque_abs_error']
        pitch = values['pitch_abs_error']
        rows.append(
            {
                'condition_id': float(condition_id),
                'sample_count': float(len(torque)),
                'torque_mae': sum(torque) / len(torque) if torque else 0.0,
                'torque_p95': percentile(torque, 0.95),
                'pitch_mae': sum(pitch) / len(pitch) if pitch else 0.0,
                'pitch_p95': percentile(pitch, 0.95),
                'ood_max': max(values['ood_score'], default=0.0),
                'fallback_fraction': sum(values['fallback_active']) / len(values['fallback_active']) if values['fallback_active'] else 0.0,
                'projection_fraction': sum(values['action_projection_applied']) / len(values['action_projection_applied']) if values['action_projection_applied'] else 0.0,
                'rate_limited_fraction': sum(values['action_rate_limited']) / len(values['action_rate_limited']) if values['action_rate_limited'] else 0.0,
                'saturation_fraction': sum(values['residual_saturated']) / len(values['residual_saturated']) if values['residual_saturated'] else 0.0,
            }
        )
    return rows


def _pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    return plt


def _value(row: dict[str, float], key: str) -> float | None:
    value = row.get(key)
    if value is None or not math.isfinite(value):
        return None
    return value


def _plot_series(
    axis,
    rows: list[dict[str, float]],
    field: str,
    label: str,
    color: str,
    scale: float = 1.0,
    **kwargs: Any,
) -> bool:
    points = [
        (_value(row, 'step_center'), _value(row, field))
        for row in rows
    ]
    points = [(x, y * scale) for x, y in points if x is not None and y is not None]
    if not points:
        return False
    axis.plot(
        [point[0] for point in points],
        [point[1] for point in points],
        label=label,
        color=color,
        **kwargs,
    )
    return True


def _draw_round_boundaries(axis, round_summaries: list[dict[str, float]]) -> None:
    for row in round_summaries[:-1]:
        axis.axvline(
            row['total_steps'],
            color='#9ca3af',
            linestyle='--',
            linewidth=0.8,
            alpha=0.8,
        )


def _save_figure(plt, figure, path: Path) -> Path:
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def _plot_queue_step_diagnostics(
    rows: list[dict[str, float]],
    round_summaries: list[dict[str, float]],
    figure_dir: Path,
) -> Path | None:
    if not rows:
        return None
    plt = _pyplot()
    if plt is None:
        return None
    figure, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
    for axis in axes:
        _draw_round_boundaries(axis, round_summaries)

    power_axis = axes[0]
    _plot_series(
        power_axis,
        rows,
        'power_state_real_mean',
        'Power state',
        '#2563eb',
        scale=1e-6,
        linewidth=1.35,
    )
    _plot_series(
        power_axis,
        rows,
        'power_reward_metric_mean',
        'Power reward metric',
        '#ea580c',
        scale=1e-6,
        linewidth=1.25,
    )
    wind_axis = power_axis.twinx()
    _plot_series(wind_axis, rows, 'wind_t_mean', 'Wind speed', '#059669', linewidth=1.15)
    power_axis.set_title('Queue Training: Power and Wind')
    power_axis.set_ylabel('Power (MW)')
    wind_axis.set_ylabel('Wind (m/s)')
    power_axis.grid(alpha=0.25)
    power_handles, power_labels = power_axis.get_legend_handles_labels()
    wind_handles, wind_labels = wind_axis.get_legend_handles_labels()
    power_axis.legend(power_handles + wind_handles, power_labels + wind_labels, loc='best', ncol=3, fontsize=8)

    reward_axis = axes[1]
    _plot_series(reward_axis, rows, 'reward_total_mean', 'Total reward', '#7c3aed', linewidth=1.25)
    _plot_series(
        reward_axis,
        rows,
        'reward_blade_root_peak_excess_mean',
        'Blade-root peak excess',
        '#dc2626',
        scale=1e-6,
        linewidth=1.25,
    )
    _plot_series(
        reward_axis,
        rows,
        'reward_energy_increment_j_mean',
        'Energy increment',
        '#0891b2',
        scale=1e-6,
        linewidth=1.15,
    )
    reward_axis.set_title('Queue Training: Reward Components')
    reward_axis.set_ylabel('Reward / scaled physical value')
    reward_axis.grid(alpha=0.25)
    reward_axis.legend(loc='best', ncol=3, fontsize=8)

    safety_axis = axes[2]
    safety_fields = (
        ('control_fallback_active_mean', 'Fallback', '#dc2626'),
        ('control_action_projection_applied_mean', 'Projection', '#ea580c'),
        ('control_action_rate_limited_mean', 'Rate limited', '#7c3aed'),
        ('transition_store_eligible_mean', 'Eligible transition', '#059669'),
    )
    for field, label, color in safety_fields:
        _plot_series(safety_axis, rows, field, label, color, scale=100.0, linewidth=1.25)
    safety_axis.set_title('Queue Training: Safety and Replay Eligibility')
    safety_axis.set_ylabel('Fraction (%)')
    safety_axis.set_ylim(bottom=0.0)
    safety_axis.grid(alpha=0.25)
    safety_axis.legend(loc='best', ncol=4, fontsize=8)

    ood_axis = axes[3]
    _plot_series(ood_axis, rows, 'control_ood_score_mean', 'OOD score mean', '#2563eb', linewidth=1.25)
    _plot_series(ood_axis, rows, 'control_ood_score_max', 'OOD score max', '#b91c1c', linewidth=1.15)
    _plot_series(ood_axis, rows, 'condition_train_active_mean', 'TRAIN-active fraction', '#059669', scale=100.0, linewidth=1.15)
    ood_axis.set_title('Queue Training: OOD Control Signal')
    ood_axis.set_ylabel('Score / fraction (%)')
    ood_axis.set_xlabel('Online environment step')
    ood_axis.grid(alpha=0.25)
    ood_axis.legend(loc='best', ncol=3, fontsize=8)

    figure.suptitle('Queue Training Step Diagnostics', y=0.995)
    return _save_figure(plt, figure, figure_dir / 'queue_train_step_diagnostics.png')


def _plot_update_diagnostics(
    rows: list[dict[str, float]],
    round_summaries: list[dict[str, float]],
    figure_dir: Path,
) -> Path | None:
    if not rows:
        return None
    plt = _pyplot()
    if plt is None:
        return None
    figure, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
    for axis in axes:
        _draw_round_boundaries(axis, round_summaries)

    loss_axis = axes[0]
    _plot_series(loss_axis, rows, 'critic1_loss_mean', 'Critic 1 loss', '#2563eb', linewidth=1.25)
    _plot_series(loss_axis, rows, 'critic2_loss_mean', 'Critic 2 loss', '#ea580c', linewidth=1.25)
    loss_axis.set_yscale('log')
    loss_axis.set_title('Queue Updates: Critic Loss')
    loss_axis.set_ylabel('Loss (log scale)')
    loss_axis.grid(alpha=0.25, which='both')
    loss_axis.legend(loc='best', ncol=2, fontsize=8)

    td_axis = axes[1]
    _plot_series(td_axis, rows, 'critic1_td_error_abs_mean_mean', 'Critic 1 |TD error|', '#1d4ed8', linewidth=1.25)
    _plot_series(td_axis, rows, 'critic2_td_error_abs_mean_mean', 'Critic 2 |TD error|', '#c2410c', linewidth=1.25)
    _plot_series(td_axis, rows, 'critic_q_abs_gap_mean_mean', 'Twin critic gap', '#059669', linewidth=1.25)
    td_axis.set_yscale('log')
    td_axis.set_title('Queue Updates: TD Error and Twin-Critic Gap')
    td_axis.set_ylabel('Absolute value (log scale)')
    td_axis.grid(alpha=0.25, which='both')
    td_axis.legend(loc='best', ncol=3, fontsize=8)

    q_axis = axes[2]
    _plot_series(q_axis, rows, 'q1_mean_mean', 'Q1 mean', '#2563eb', linewidth=1.25)
    _plot_series(q_axis, rows, 'q2_mean_mean', 'Q2 mean', '#ea580c', linewidth=1.25)
    _plot_series(q_axis, rows, 'target_q_mean_mean', 'Target Q mean', '#7c3aed', linewidth=1.25)
    q_axis.set_title('Queue Updates: Q Estimates')
    q_axis.set_ylabel('Q value')
    q_axis.grid(alpha=0.25)
    q_axis.legend(loc='best', ncol=3, fontsize=8)

    actor_axis = axes[3]
    _plot_series(actor_axis, rows, 'actor_loss_mean', 'Actor loss', '#0f766e', linewidth=1.25)
    _plot_series(actor_axis, rows, 'actor_rl_loss_mean', 'Actor RL loss', '#0891b2', linewidth=1.25)
    _plot_series(actor_axis, rows, 'actor_bc_loss_mean', 'Actor BC loss', '#9333ea', linewidth=1.25)
    weight_axis = actor_axis.twinx()
    _plot_series(weight_axis, rows, 'actor_bc_effective_weight_mean', 'BC effective weight', '#dc2626', linewidth=1.15)
    actor_axis.set_title('Queue Updates: TD3+BC Actor Terms')
    actor_axis.set_ylabel('Loss')
    weight_axis.set_ylabel('BC weight')
    actor_axis.set_xlabel('Online environment step')
    actor_axis.grid(alpha=0.25)
    actor_handles, actor_labels = actor_axis.get_legend_handles_labels()
    weight_handles, weight_labels = weight_axis.get_legend_handles_labels()
    actor_axis.legend(actor_handles + weight_handles, actor_labels + weight_labels, loc='best', ncol=4, fontsize=8)

    figure.suptitle('Queue Training Update Diagnostics', y=0.995)
    return _save_figure(plt, figure, figure_dir / 'queue_training_update_diagnostics.png')


def _plot_q_health_diagnostics(
    rows: list[dict[str, float]],
    round_summaries: list[dict[str, float]],
    figure_dir: Path,
    target_q_limit: float | None = None,
    actor_q_limit: float | None = None,
) -> Path | None:
    if not rows:
        return None
    plt = _pyplot()
    if plt is None:
        return None

    figure, axes = plt.subplots(3, 1, figsize=(14, 14), sharex=True)
    for axis in axes:
        _draw_round_boundaries(axis, round_summaries)

    q_magnitude_axis = axes[0]
    _plot_series(
        q_magnitude_axis,
        rows,
        'target_q_abs_mean_mean',
        'Target Q absolute mean',
        '#7c3aed',
        linewidth=1.45,
    )
    _plot_series(
        q_magnitude_axis,
        rows,
        'actor_q_abs_mean_mean',
        'Actor Q absolute mean',
        '#0891b2',
        linewidth=1.45,
    )
    if target_q_limit is not None and target_q_limit > 0.0:
        q_magnitude_axis.axhline(
            target_q_limit,
            color='#7c3aed',
            linestyle='--',
            linewidth=1.1,
            label=f'Target Q guard = {target_q_limit:g}',
        )
    if actor_q_limit is not None and actor_q_limit > 0.0:
        q_magnitude_axis.axhline(
            actor_q_limit,
            color='#0891b2',
            linestyle=':',
            linewidth=1.1,
            label=f'Actor Q guard = {actor_q_limit:g}',
        )
    q_magnitude_axis.set_title('Q Health: Absolute Magnitude and Guard Thresholds')
    q_magnitude_axis.set_ylabel('|Q| mean')
    q_magnitude_axis.grid(alpha=0.25)
    q_magnitude_axis.legend(loc='best', ncol=4, fontsize=8)

    estimate_axis = axes[1]
    _plot_series(estimate_axis, rows, 'q1_mean_mean', 'Q1 mean', '#2563eb', linewidth=1.25)
    _plot_series(estimate_axis, rows, 'q2_mean_mean', 'Q2 mean', '#ea580c', linewidth=1.25)
    _plot_series(estimate_axis, rows, 'target_q_mean_mean', 'Target Q mean', '#7c3aed', linewidth=1.25)
    estimate_axis.set_title('Q Health: Twin-Critic Estimates and Disagreement')
    estimate_axis.set_ylabel('Q value')
    estimate_axis.grid(alpha=0.25)
    gap_axis = estimate_axis.twinx()
    _plot_series(
        gap_axis,
        rows,
        'critic_q_abs_gap_mean_mean',
        'Mean |Q1-Q2|',
        '#059669',
        linewidth=1.25,
    )
    gap_axis.set_ylabel('|Q1-Q2|')
    estimate_handles, estimate_labels = estimate_axis.get_legend_handles_labels()
    gap_handles, gap_labels = gap_axis.get_legend_handles_labels()
    estimate_axis.legend(
        estimate_handles + gap_handles,
        estimate_labels + gap_labels,
        loc='best',
        ncol=4,
        fontsize=8,
    )

    outcome_axis = axes[2]
    outcome_fields = (
        ('critic_update_rate', 'Critic update rate', '#2563eb'),
        ('critic_q_guard_rate', 'Critic Q guard rate', '#dc2626'),
        ('actor_update_rate', 'Actor update / candidate', '#059669'),
        ('actor_q_guard_rate', 'Actor Q guard / candidate', '#9333ea'),
        ('actor_deferred_rate', 'Actor deferred / critic update', '#f59e0b'),
    )
    for field, label, color in outcome_fields:
        _plot_series(outcome_axis, rows, field, label, color, scale=100.0, linewidth=1.35)
    outcome_axis.set_title('Q Health: Effective Update and Guard Rates')
    outcome_axis.set_ylabel('Rate (%)')
    outcome_axis.set_xlabel('Online environment step')
    outcome_axis.set_ylim(bottom=0.0, top=105.0)
    outcome_axis.grid(alpha=0.25)
    outcome_axis.legend(loc='best', ncol=3, fontsize=8)

    figure.suptitle('Queue Q Health and Training Availability', y=0.995)
    return _save_figure(plt, figure, figure_dir / 'queue_q_health_diagnostics.png')


def _plot_condition_metrics(rows: list[dict[str, float]], figure_dir: Path) -> Path | None:
    if not rows:
        return None
    plt = _pyplot()
    if plt is None:
        return None
    conditions = sorted({int(row['condition_id']) for row in rows})
    rounds = sorted({int(row['round']) for row in rows})
    x = list(range(len(conditions)))
    by_round_condition = {(int(row['round']), int(row['condition_id'])): row for row in rows}
    figure, axes = plt.subplots(2, 2, figsize=(15, 10))

    coverage_axis = axes[0][0]
    for round_index in rounds:
        values = [
            by_round_condition.get((round_index, condition), {}).get('stored_transition_count', 0.0)
            for condition in conditions
        ]
        coverage_axis.plot(x, values, marker='o', markersize=3, linewidth=1.15, label=f'Round {round_index}')
    coverage_axis.set_title('Stored Transitions by Condition and Round')
    coverage_axis.set_ylabel('Stored transitions')
    coverage_axis.grid(alpha=0.25)
    coverage_axis.legend(loc='best', ncol=2, fontsize=8)

    rejection_fields = (
        ('rejected_fallback_count', 'Fallback', '#dc2626'),
        ('rejected_unsafe_count', 'Unsafe', '#ea580c'),
        ('rejected_negative_power_count', 'Negative power', '#ca8a04'),
        ('rejected_invalid_count', 'Invalid', '#7c3aed'),
        ('rejected_boundary_count', 'Boundary', '#64748b'),
        ('rejected_non_train_count', 'Non-TRAIN', '#0891b2'),
        ('rejected_other_count', 'Other', '#059669'),
    )
    rejection_axis = axes[0][1]
    bottom = [0.0] * len(conditions)
    for field, label, color in rejection_fields:
        values = [
            sum(row.get(field, 0.0) for row in rows if int(row['condition_id']) == condition)
            for condition in conditions
        ]
        rejection_axis.bar(x, values, bottom=bottom, color=color, label=label, width=0.82)
        bottom = [old + new for old, new in zip(bottom, values, strict=True)]
    rejection_axis.set_title('Rejected Transition Categories')
    rejection_axis.set_ylabel('Rejected transitions')
    rejection_axis.grid(axis='y', alpha=0.25)
    rejection_axis.legend(loc='best', fontsize=7, ncol=2)

    source_axis = axes[1][0]
    raw_values = [
        sum(row.get('replay_raw_proxy_count', 0.0) for row in rows if int(row['condition_id']) == condition)
        for condition in conditions
    ]
    online_values = [
        sum(row.get('replay_online_bladed_count', 0.0) for row in rows if int(row['condition_id']) == condition)
        for condition in conditions
    ]
    source_axis.bar(x, raw_values, color='#f59e0b', label='Raw realized-action proxy', width=0.82)
    source_axis.bar(x, online_values, bottom=raw_values, color='#2563eb', label='Online Bladed', width=0.82)
    source_axis.set_title('Replay Source Composition')
    source_axis.set_ylabel('Replay transitions')
    source_axis.grid(axis='y', alpha=0.25)
    source_axis.legend(loc='best', fontsize=8)

    heatmap_axis = axes[1][1]
    matrix = [
        [
            by_round_condition.get((round_index, condition), {}).get('stored_transition_count', 0.0)
            for condition in conditions
        ]
        for round_index in rounds
    ]
    image = heatmap_axis.imshow(matrix, aspect='auto', interpolation='nearest', cmap='YlGn')
    heatmap_axis.set_title('Stored Transition Coverage Heatmap')
    heatmap_axis.set_xlabel('Condition / wind ID')
    heatmap_axis.set_ylabel('Completed round')
    heatmap_axis.set_xticks(x)
    heatmap_axis.set_xticklabels([str(condition) for condition in conditions], rotation=45, ha='right')
    heatmap_axis.set_yticks(list(range(len(rounds))))
    heatmap_axis.set_yticklabels([str(round_index) for round_index in rounds])
    figure.colorbar(image, ax=heatmap_axis, label='Stored transitions')

    for axis in axes.flat:
        axis.set_xticks(x)
        axis.set_xticklabels([str(condition) for condition in conditions], rotation=45, ha='right')
    figure.suptitle('Queue Condition Metrics', y=0.995)
    return _save_figure(plt, figure, figure_dir / 'queue_condition_metrics.png')


def _plot_round_summary(rows: list[dict[str, float]], figure_dir: Path) -> Path | None:
    if not rows:
        return None
    plt = _pyplot()
    if plt is None:
        return None
    rounds = [int(row['round']) for row in rows]
    x = list(range(len(rounds)))
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    stored = [row['stored_transitions'] for row in rows]
    axes[0][0].bar(x, stored, color='#2563eb', label='Stored')
    axes[0][0].bar(x, [row['rejected_transitions'] for row in rows], bottom=stored, color='#f97316', label='Rejected')
    axes[0][0].set_title('Cumulative Transition Accounting')
    axes[0][0].set_ylabel('Transitions')
    axes[0][0].legend(loc='best')
    axes[0][0].grid(axis='y', alpha=0.25)

    axes[0][1].plot(x, [row['update_count'] for row in rows], marker='o', color='#7c3aed')
    axes[0][1].set_title('Cumulative Update Count')
    axes[0][1].set_ylabel('Updates')
    axes[0][1].grid(alpha=0.25)

    axes[1][0].plot(x, [row['total_steps'] for row in rows], marker='o', color='#0891b2')
    axes[1][0].set_title('Cumulative Online Steps')
    axes[1][0].set_ylabel('Environment steps')
    axes[1][0].grid(alpha=0.25)

    axes[1][1].plot(x, [row['all_conditions_covered'] * 100.0 for row in rows], marker='o', label='All conditions covered', color='#059669')
    axes[1][1].plot(x, [row['training_valid'] * 100.0 for row in rows], marker='o', label='Training valid', color='#dc2626')
    axes[1][1].plot(x, [row['skipped_conditions'] for row in rows], marker='o', label='Skipped conditions', color='#f97316')
    axes[1][1].set_title('Queue Lifecycle Status')
    axes[1][1].set_ylabel('Percent / count')
    axes[1][1].set_ylim(bottom=0.0)
    axes[1][1].legend(loc='best', fontsize=8)
    axes[1][1].grid(alpha=0.25)

    for axis in axes.flat:
        axis.set_xticks(x)
        axis.set_xticklabels([str(round_index) for round_index in rounds])
        axis.set_xlabel('Completed round')
    figure.suptitle('Queue Round Summary', y=0.995)
    return _save_figure(plt, figure, figure_dir / 'queue_round_summary.png')


def _plot_offline_action_error(rows: list[dict[str, float]], figure_dir: Path) -> Path | None:
    if not rows:
        return None
    plt = _pyplot()
    if plt is None:
        return None
    conditions = [int(row['condition_id']) for row in rows]
    x = list(range(len(conditions)))
    width = 0.38
    figure, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    axes[0].bar([value - width / 2 for value in x], [row['torque_mae'] / 1e6 for row in rows], width=width, label='MAE', color='#2563eb')
    axes[0].bar([value + width / 2 for value in x], [row['torque_p95'] / 1e6 for row in rows], width=width, label='P95', color='#93c5fd')
    axes[0].set_title('Offline Policy Torque Error by Condition / Wind ID')
    axes[0].set_ylabel('Torque error (MNm)')
    axes[0].grid(axis='y', alpha=0.25)
    axes[0].legend(loc='best')

    axes[1].bar([value - width / 2 for value in x], [row['pitch_mae'] * 1e3 for row in rows], width=width, label='MAE', color='#ea580c')
    axes[1].bar([value + width / 2 for value in x], [row['pitch_p95'] * 1e3 for row in rows], width=width, label='P95', color='#fdba74')
    axes[1].set_title('Offline Policy Pitch Error by Condition / Wind ID')
    axes[1].set_ylabel('Pitch error (mrad)')
    axes[1].grid(axis='y', alpha=0.25)
    axes[1].legend(loc='best')
    axes[1].set_xlabel('Condition / wind ID (m/s)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(condition) for condition in conditions])
    figure.suptitle('Offline Policy Action Error', y=0.995)
    return _save_figure(plt, figure, figure_dir / 'queue_offline_action_error_by_wind.png')


def _plot_offline_safety(rows: list[dict[str, float]], figure_dir: Path) -> Path | None:
    if not rows:
        return None
    plt = _pyplot()
    if plt is None:
        return None
    conditions = [int(row['condition_id']) for row in rows]
    x = list(range(len(conditions)))
    figure, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    fields = (
        ('fallback_fraction', 'Fallback', '#dc2626'),
        ('projection_fraction', 'Projection', '#ea580c'),
        ('rate_limited_fraction', 'Rate limited', '#7c3aed'),
        ('saturation_fraction', 'Residual saturated', '#0891b2'),
    )
    for field, label, color in fields:
        axes[0].plot(x, [row[field] * 100.0 for row in rows], marker='o', markersize=3, linewidth=1.2, label=label, color=color)
    axes[0].set_title('Offline Policy Safety Interventions by Condition / Wind ID')
    axes[0].set_ylabel('Fraction (%)')
    axes[0].set_ylim(bottom=0.0)
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc='best', ncol=4, fontsize=8)
    axes[1].plot(x, [row['ood_max'] for row in rows], marker='o', color='#b91c1c', linewidth=1.3, label='OOD score max')
    axes[1].set_title('Offline Policy Maximum OOD Score')
    axes[1].set_ylabel('OOD score')
    axes[1].set_xlabel('Condition / wind ID (m/s)')
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc='best')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(condition) for condition in conditions])
    figure.suptitle('Offline Policy Safety Diagnostics', y=0.995)
    return _save_figure(plt, figure, figure_dir / 'queue_offline_safety_by_wind.png')


def generate_queue_plots(
    output_dir: str | Path,
    figure_dir: str | Path | None = None,
    bin_steps: int = 10_000,
) -> dict[str, str]:
    output_path = Path(output_dir)
    destination = Path(figure_dir) if figure_dir is not None else output_path / 'figures'
    destination.mkdir(parents=True, exist_ok=True)
    if _pyplot() is None:
        return {}

    generated: dict[str, str] = {}
    round_summaries = read_queue_round_summaries(output_path)
    step_path = output_path / 'queue_train_step_metrics.csv'
    if step_path.is_file():
        step_rows = aggregate_queue_step_metrics(step_path, output_path, bin_steps)
        step_figure = _plot_queue_step_diagnostics(step_rows, round_summaries, destination)
        if step_figure is not None:
            generated[step_figure.name] = str(step_figure)

    update_path = output_path / 'queue_train_update_metrics.csv'
    if update_path.is_file():
        update_rows = aggregate_queue_update_metrics(update_path, output_path, bin_steps)
        update_figure = _plot_update_diagnostics(update_rows, round_summaries, destination)
        if update_figure is not None:
            generated[update_figure.name] = str(update_figure)
        q_health_rows = aggregate_queue_q_health_metrics(update_path, output_path, bin_steps)
        target_q_limit = None
        actor_q_limit = None
        config_path = output_path / 'config.json'
        if config_path.is_file():
            try:
                config = json.loads(config_path.read_text(encoding='utf-8'))
                agent_config = config.get('agent', {})
                if isinstance(agent_config, dict):
                    target_q_limit = float(agent_config.get('max_target_q_abs_mean', 0.0))
                    actor_q_limit = float(agent_config.get('max_actor_q_abs_mean', 0.0))
                    if not math.isfinite(target_q_limit) or target_q_limit <= 0.0:
                        target_q_limit = None
                    if not math.isfinite(actor_q_limit) or actor_q_limit <= 0.0:
                        actor_q_limit = None
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                target_q_limit = None
                actor_q_limit = None
        q_health_figure = _plot_q_health_diagnostics(
            q_health_rows,
            round_summaries,
            destination,
            target_q_limit=target_q_limit,
            actor_q_limit=actor_q_limit,
        )
        if q_health_figure is not None:
            generated[q_health_figure.name] = str(q_health_figure)

    condition_path = output_path / 'queue_condition_metrics.csv'
    if condition_path.is_file():
        condition_rows = read_queue_condition_metrics(condition_path)
        if condition_rows:
            _write_aggregate_csv(output_path / 'queue_condition_summary.csv', condition_rows)
        condition_figure = _plot_condition_metrics(condition_rows, destination)
        if condition_figure is not None:
            generated[condition_figure.name] = str(condition_figure)

    round_figure = _plot_round_summary(round_summaries, destination)
    if round_figure is not None:
        generated[round_figure.name] = str(round_figure)

    offline_rows = _read_offline_rows(output_path)
    offline_error_figure = _plot_offline_action_error(offline_rows, destination)
    if offline_error_figure is not None:
        generated[offline_error_figure.name] = str(offline_error_figure)
    offline_safety_figure = _plot_offline_safety(offline_rows, destination)
    if offline_safety_figure is not None:
        generated[offline_safety_figure.name] = str(offline_safety_figure)

    summary = {
        'output_dir': str(output_path),
        'bin_steps': int(bin_steps),
        'queue_train_step_source': str(step_path) if step_path.is_file() else None,
        'queue_train_update_source': str(update_path) if update_path.is_file() else None,
        'queue_q_health_source': str(output_path / 'queue_q_health_aggregated.csv') if update_path.is_file() else None,
        'offline_policy_summary_source': str(output_path / 'offline_eval_final' / 'offline_policy_eval_summary.json'),
        'generated_figures': sorted(generated),
        'aggregate_files': sorted(path.name for path in output_path.glob('queue_*_aggregated.csv')),
    }
    (output_path / 'queue_plot_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    return generated