from __future__ import annotations

import csv
from collections import defaultdict
import json
from pathlib import Path
import re


def _extract_eval_episode(path: Path) -> int | None:
    match = re.fullmatch(r'eval_episode_(\d+)\.json', path.name)
    if match is None:
        return None
    return int(match.group(1))


def _generate_eval_summary(output_path: Path) -> Path | None:
    eval_files = []
    for path in sorted(output_path.glob('eval_episode_*.json')):
        episode = _extract_eval_episode(path)
        if episode is None:
            continue
        eval_files.append((episode, path))
    if not eval_files:
        return None

    rows: list[dict[str, float | int]] = []
    fieldnames = {'episode'}
    for episode, path in eval_files:
        with path.open('r', encoding='utf-8') as file:
            payload = json.load(file)
        row: dict[str, float | int] = {'episode': episode}
        for key, value in payload.items():
            row[key] = float(value)
            fieldnames.add(key)
        rows.append(row)

    summary_path = output_path / 'eval_summary.csv'
    ordered_fieldnames = ['episode', *sorted(name for name in fieldnames if name != 'episode')]
    with summary_path.open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=ordered_fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return summary_path


def _generate_eval_plots(summary_path: Path, figure_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    grouped: dict[str, list[tuple[int, float]]] = defaultdict(list)
    with summary_path.open('r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            episode = int(row['episode'])
            for key, value in row.items():
                if key == 'episode' or value in (None, ''):
                    continue
                grouped[key].append((episode, float(value)))

    for name, points in grouped.items():
        points.sort(key=lambda item: item[0])
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        plt.figure(figsize=(8, 4))
        plt.plot(xs, ys, marker='o', markersize=3)
        plt.title(f'eval:{name}')
        plt.xlabel('episode')
        plt.ylabel(name)
        safe_name = f'eval_{name}'.replace('/', '_').replace(' ', '_')
        plt.tight_layout()
        plt.savefig(figure_dir / f'{safe_name}.png')
        plt.close()


def maybe_generate_plots(output_dir: str | Path, queue_bin_steps: int = 10_000) -> None:
    output_path = Path(output_dir)
    if queue_bin_steps <= 0:
        raise ValueError('queue_bin_steps must be positive')
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    figure_dir = output_path / 'figures'
    figure_dir.mkdir(exist_ok=True)
    eval_summary_path = _generate_eval_summary(output_path)
    if eval_summary_path is not None:
        _generate_eval_plots(eval_summary_path, figure_dir)
    queue_experiment = any(
        (output_path / name).is_file()
        for name in (
            'queue_train_step_metrics.csv',
            'queue_train_update_metrics.csv',
            'queue_condition_metrics.csv',
            'queue_training_summary.json',
        )
    )
    if queue_experiment:
        from RL.utils.queue_plotting import generate_queue_plots

        generate_queue_plots(output_path, figure_dir, queue_bin_steps)
    for metrics_path in sorted(output_path.glob('*_metrics.csv')):
        if queue_experiment and (
            metrics_path.name.startswith('queue_')
            or metrics_path.name == 'offline_eval_final_metrics.csv'
        ):
            continue
        grouped: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
        with metrics_path.open('r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                grouped[(row['stage'], row['name'])].append((int(row['step']), float(row['value'])))

        for (stage, name), points in grouped.items():
            points.sort(key=lambda item: item[0])
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            plt.figure(figsize=(8, 4))
            plt.plot(xs, ys)
            plt.title(f'{stage}:{name}')
            plt.xlabel('step')
            plt.ylabel(name)
            safe_name = f'{stage}_{name}'.replace('/', '_').replace(' ', '_')
            plt.tight_layout()
            plt.savefig(figure_dir / f'{safe_name}.png')
            plt.close()
