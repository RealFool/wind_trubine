from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from RL.utils.plotting import maybe_generate_plots


def _resolve_experiment_dir(experiment: str) -> Path:
    raw = Path(experiment).expanduser()
    if raw.is_dir():
        return raw
    candidate = PROJECT_ROOT / 'RL' / 'experiments' / experiment
    if candidate.is_dir():
        return candidate
    raise FileNotFoundError(
        f'Experiment directory not found: {experiment}. '
        f'Expected absolute path or name under {PROJECT_ROOT / "RL" / "experiments"}'
    )


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate figures for an existing RL experiment directory.')
    parser.add_argument(
        '--experiment',
        type=str,
        required=True,
        help='Experiment directory path, or experiment name under RL/experiments',
    )
    parser.add_argument(
        '--queue-bin-steps',
        type=int,
        default=10_000,
        help='Online environment steps per bin for queue training plots.',
    )
    args = parser.parse_args()
    if args.queue_bin_steps <= 0:
        parser.error('--queue-bin-steps must be positive')

    experiment_dir = _resolve_experiment_dir(args.experiment)
    figure_dir = experiment_dir / 'figures'
    existing_figures = set(figure_dir.glob('*.png')) if figure_dir.exists() else set()

    print(f'[PostProcess] generating plots for: {experiment_dir}', flush=True)
    maybe_generate_plots(experiment_dir, queue_bin_steps=args.queue_bin_steps)

    new_figures = sorted((set(figure_dir.glob('*.png')) - existing_figures), key=lambda path: path.name)
    all_figures = sorted(figure_dir.glob('*.png'), key=lambda path: path.name)
    print(f'[PostProcess] figures total={len(all_figures)} newly_created={len(new_figures)}', flush=True)
    if new_figures:
        for path in new_figures:
            print(f'[PostProcess] + {path.name}', flush=True)


if __name__ == '__main__':
    main()
