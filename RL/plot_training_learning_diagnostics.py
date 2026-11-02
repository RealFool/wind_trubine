from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
import statistics
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


UPDATE_METRICS = {
    'critic1_loss',
    'critic2_loss',
    'critic1_td_error_abs_mean',
    'critic2_td_error_abs_mean',
    'critic_q_abs_gap_mean',
    'q1_mean',
    'q2_mean',
    'target_q_mean',
    'update_skipped_q_guard',
    'actor_update_skipped_q_guard',
    'td3_bc_weight_schedule',
    'actor_rl_loss',
}


def _resolve_experiment_dir(experiment: str) -> Path:
    path = Path(experiment).expanduser()
    if path.is_dir():
        return path.resolve()
    path = PROJECT_ROOT / 'RL' / 'experiments' / experiment
    if path.is_dir():
        return path.resolve()
    raise FileNotFoundError(f'Experiment directory not found: {experiment}')


def _load_config(experiment_dir: Path) -> dict:
    config_path = experiment_dir / 'config.json'
    if not config_path.is_file():
        raise FileNotFoundError(f'Missing experiment config: {config_path}')
    return json.loads(config_path.read_text(encoding='utf-8'))


def _read_metric_series(csv_path: Path, wanted: set[str]) -> dict[str, list[tuple[int, float]]]:
    result: dict[str, list[tuple[int, float]]] = defaultdict(list)
    with csv_path.open('r', encoding='utf-8', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            metric_name = row.get('name', '')
            if metric_name not in wanted:
                continue
            try:
                step = int(float(row['step']))
                value = float(row['value'])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(value):
                result[metric_name].append((step, value))
    return dict(result)


def _mean_by_bucket(points: list[tuple[int, float]], bin_steps: int) -> tuple[list[float], list[float]]:
    sums: dict[int, float] = defaultdict(float)
    counts: dict[int, int] = defaultdict(int)
    for step, value in points:
        bucket = step // bin_steps
        sums[bucket] += value
        counts[bucket] += 1
    buckets = sorted(sums)
    return (
        [(bucket + 0.5) * bin_steps for bucket in buckets],
        [sums[bucket] / counts[bucket] for bucket in buckets],
    )


def _rate_by_bucket(
    event_points: list[tuple[int, float]],
    opportunity_points: list[tuple[int, float]],
    bin_steps: int,
) -> tuple[list[float], list[float]]:
    events: dict[int, float] = defaultdict(float)
    opportunities: dict[int, int] = defaultdict(int)
    for step, value in event_points:
        events[step // bin_steps] += max(0.0, value)
    for step, _value in opportunity_points:
        opportunities[step // bin_steps] += 1
    buckets = sorted(opportunities)
    return (
        [(bucket + 0.5) * bin_steps for bucket in buckets],
        [events[bucket] / opportunities[bucket] for bucket in buckets],
    )


def _rolling_median(values: list[float], window: int) -> list[float]:
    return [statistics.median(values[max(0, index - window + 1):index + 1]) for index in range(len(values))]


def _ema(values: list[float], alpha: float) -> list[float]:
    if not values:
        return []
    result = [values[0]]
    for value in values[1:]:
        result.append(alpha * value + (1.0 - alpha) * result[-1])
    return result


def _positive(values: list[float]) -> list[float]:
    return [max(value, 1e-8) for value in values]


def _shade_stages(axis, critic_start: int, actor_start: int, end_step: int, show_labels: bool) -> None:
    axis.axvspan(0, critic_start, color='#dbeafe', alpha=0.42, label='BC replay warmup' if show_labels else None)
    axis.axvspan(critic_start, actor_start, color='#fef3c7', alpha=0.42, label='Critic-only updates' if show_labels else None)
    axis.axvspan(actor_start, end_step, color='#dcfce7', alpha=0.35, label='TD3+BC actor updates' if show_labels else None)
    axis.axvline(critic_start, color='#b45309', linestyle='--', linewidth=1.0)
    axis.axvline(actor_start, color='#15803d', linestyle='--', linewidth=1.0)


def _plot_smoothed(axis, steps: list[float], values: list[float], label: str, color: str, window: int, ema_alpha: float) -> None:
    axis.plot(steps, values, color=color, alpha=0.20, linewidth=0.75)
    axis.plot(
        steps,
        _rolling_median(values, window),
        color=color,
        linewidth=1.75,
        label=f'{label} rolling median ({window} bins)',
    )
    axis.plot(
        steps,
        _ema(values, ema_alpha),
        color=color,
        linestyle='--',
        linewidth=1.1,
        label=f'{label} EMA ({ema_alpha:.2f})',
    )


def _align_q_gap(
    q1: tuple[list[float], list[float]],
    q2: tuple[list[float], list[float]],
) -> tuple[list[float], list[float]]:
    q1_by_step = dict(zip(q1[0], q1[1]))
    q2_by_step = dict(zip(q2[0], q2[1]))
    steps = sorted(set(q1_by_step) & set(q2_by_step))
    return steps, [abs(q1_by_step[step] - q2_by_step[step]) for step in steps]


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Plot streamed TD3+BC critic, Q guard, and safety-fallback diagnostics.',
    )
    parser.add_argument('--experiment', required=True, help='Experiment path or name under RL/experiments.')
    parser.add_argument('--bin-steps', type=int, default=10_000, help='Aggregation window in online environment steps.')
    parser.add_argument('--smooth-window', type=int, default=5, help='Rolling-median window in aggregated bins.')
    parser.add_argument('--ema-alpha', type=float, default=0.30, help='EMA smoothing factor in the range (0, 1].')
    parser.add_argument('--output-dir', default='', help='Defaults to <experiment>/figures/training_learning_diagnostics.')
    args = parser.parse_args()
    if args.bin_steps <= 0:
        raise ValueError('--bin-steps must be positive')
    if args.smooth_window <= 0:
        raise ValueError('--smooth-window must be positive')
    if not 0.0 < args.ema_alpha <= 1.0:
        raise ValueError('--ema-alpha must be in the range (0, 1]')

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError('Plotting requires matplotlib. Install it in the active Python environment.') from exc

    experiment_dir = _resolve_experiment_dir(args.experiment)
    config = _load_config(experiment_dir)
    update_csv_path = experiment_dir / 'train_update_metrics.csv'
    step_csv_path = experiment_dir / 'train_step_metrics.csv'
    if not update_csv_path.is_file():
        raise FileNotFoundError(f'Missing update log: {update_csv_path}')
    if not step_csv_path.is_file():
        raise FileNotFoundError(f'Missing step log: {step_csv_path}')

    update_series = _read_metric_series(update_csv_path, UPDATE_METRICS)
    step_series = _read_metric_series(step_csv_path, {'control_fallback_active'})
    required_update_metrics = {
        'critic1_loss',
        'critic2_loss',
        'critic1_td_error_abs_mean',
        'critic2_td_error_abs_mean',
        'q1_mean',
        'q2_mean',
        'target_q_mean',
    }
    missing_update_metrics = sorted(name for name in required_update_metrics if name not in update_series)
    if missing_update_metrics:
        raise ValueError(f'The update log lacks: {", ".join(missing_update_metrics)}')
    if 'control_fallback_active' not in step_series:
        raise ValueError('The step log lacks control_fallback_active.')

    binned = {
        name: _mean_by_bucket(points, args.bin_steps)
        for name, points in update_series.items()
    }
    if 'critic_q_abs_gap_mean' in binned:
        q_gap = binned['critic_q_abs_gap_mean']
        q_gap_label = 'Mean |Q1-Q2|'
        q_gap_source = 'critic_q_abs_gap_mean'
    else:
        q_gap = _align_q_gap(binned['q1_mean'], binned['q2_mean'])
        q_gap_label = '|mean(Q1)-mean(Q2)| (fallback)'
        q_gap_source = 'absolute difference of q1_mean and q2_mean'

    update_opportunities = update_series.get('td3_bc_weight_schedule', update_series['critic1_loss'])
    critic_guard_rate = _rate_by_bucket(
        update_series.get('update_skipped_q_guard', []),
        update_opportunities,
        args.bin_steps,
    )
    actor_guard_rate = _rate_by_bucket(
        update_series.get('actor_update_skipped_q_guard', []),
        update_series.get('actor_rl_loss', []),
        args.bin_steps,
    )
    fallback_rate = _mean_by_bucket(step_series['control_fallback_active'], args.bin_steps)

    train_config = config['train']
    critic_start = int(train_config.get('critic_update_after_steps', train_config.get('online_replay_warmup_steps', 0)))
    actor_start = int(train_config.get('actor_update_after_steps', critic_start))
    end_step = int(max(binned['q1_mean'][0]))
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else experiment_dir / 'figures' / 'training_learning_diagnostics'
    output_dir.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(4, 1, figsize=(15, 15), sharex=True)

    loss_axis = axes[0]
    _shade_stages(loss_axis, critic_start, actor_start, end_step, show_labels=True)
    _plot_smoothed(loss_axis, binned['critic1_loss'][0], _positive(binned['critic1_loss'][1]), 'Critic 1 loss', '#2563eb', args.smooth_window, args.ema_alpha)
    _plot_smoothed(loss_axis, binned['critic2_loss'][0], _positive(binned['critic2_loss'][1]), 'Critic 2 loss', '#ea580c', args.smooth_window, args.ema_alpha)
    loss_axis.set_yscale('log')
    loss_axis.set_title('Critic Loss (log scale)')
    loss_axis.set_ylabel('Smooth L1 loss')
    loss_axis.grid(alpha=0.25, which='both')
    loss_axis.legend(loc='best', ncol=2, fontsize=8)

    td_axis = axes[1]
    _shade_stages(td_axis, critic_start, actor_start, end_step, show_labels=False)
    _plot_smoothed(td_axis, binned['critic1_td_error_abs_mean'][0], _positive(binned['critic1_td_error_abs_mean'][1]), 'Critic 1 |Q-y|', '#1d4ed8', args.smooth_window, args.ema_alpha)
    _plot_smoothed(td_axis, binned['critic2_td_error_abs_mean'][0], _positive(binned['critic2_td_error_abs_mean'][1]), 'Critic 2 |Q-y|', '#c2410c', args.smooth_window, args.ema_alpha)
    td_axis.set_yscale('log')
    td_axis.set_title('Mean Absolute TD Error (log scale)')
    td_axis.set_ylabel('|Q-y|')
    td_axis.grid(alpha=0.25, which='both')
    td_axis.legend(loc='best', ncol=2, fontsize=8)

    q_axis = axes[2]
    _shade_stages(q_axis, critic_start, actor_start, end_step, show_labels=False)
    q_axis.plot(*binned['q1_mean'], color='#2563eb', linewidth=1.45, label='Q1 mean')
    q_axis.plot(*binned['q2_mean'], color='#ea580c', linewidth=1.45, label='Q2 mean')
    q_axis.plot(*binned['target_q_mean'], color='#7c3aed', linewidth=1.45, label='Target Q mean')
    q_axis.set_title('Q Estimates and Twin-Critic Gap')
    q_axis.set_ylabel('Q value')
    q_axis.grid(alpha=0.25)
    gap_axis = q_axis.twinx()
    gap_axis.plot(*q_gap, color='#059669', linewidth=1.45, label=q_gap_label)
    gap_axis.set_ylabel('|Q1-Q2|')
    q_handles, q_labels = q_axis.get_legend_handles_labels()
    gap_handles, gap_labels = gap_axis.get_legend_handles_labels()
    q_axis.legend(q_handles + gap_handles, q_labels + gap_labels, loc='best', ncol=2, fontsize=8)

    rate_axis = axes[3]
    _shade_stages(rate_axis, critic_start, actor_start, end_step, show_labels=False)
    rate_axis.plot(critic_guard_rate[0], [value * 100.0 for value in critic_guard_rate[1]], color='#dc2626', linewidth=1.45, label='Critic Q guard rate')
    if actor_guard_rate[0]:
        rate_axis.plot(actor_guard_rate[0], [value * 100.0 for value in actor_guard_rate[1]], color='#9333ea', linewidth=1.45, label='Actor Q guard rate')
    rate_axis.plot(fallback_rate[0], [value * 100.0 for value in fallback_rate[1]], color='#0891b2', linewidth=1.45, label='Safety fallback rate')
    rate_axis.set_title('Q Guard and Safety Fallback Rates')
    rate_axis.set_ylabel('Rate (%)')
    rate_axis.set_ylim(bottom=0.0)
    rate_axis.grid(alpha=0.25)
    rate_axis.legend(loc='best', ncol=3, fontsize=8)
    rate_axis.set_xlabel(f'Online environment step (mean/rate over {args.bin_steps} steps)')

    figure.suptitle('TD3+BC Training Diagnostics', y=0.995)
    figure.tight_layout()
    figure_path = output_dir / 'training_critic_q_guard_fallback_diagnostics.png'
    figure.savefig(figure_path, dpi=170)
    plt.close(figure)

    summary = {
        'experiment_dir': str(experiment_dir),
        'update_source_csv': str(update_csv_path),
        'step_source_csv': str(step_csv_path),
        'bin_steps': int(args.bin_steps),
        'smooth_window_bins': int(args.smooth_window),
        'ema_alpha': float(args.ema_alpha),
        'critic_update_start_step': critic_start,
        'actor_update_start_step': actor_start,
        'last_plotted_step': end_step,
        'q_gap_source': q_gap_source,
        'output_figure': str(figure_path),
    }
    (output_dir / 'training_learning_diagnostics_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()