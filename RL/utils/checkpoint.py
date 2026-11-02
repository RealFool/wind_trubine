from __future__ import annotations

import os
from pathlib import Path


def resolve_dynamics_checkpoint(configured_path: str | None) -> str:
    candidates: list[Path] = []
    env_path = os.environ.get('RL_DYNAMICS_CHECKPOINT', '').strip()
    if env_path:
        candidates.append(Path(env_path).expanduser())
    if configured_path:
        configured = Path(configured_path).expanduser()
        candidates.append(configured)
        if not configured.is_absolute():
            candidates.append((Path.cwd() / configured).resolve())

    project_root = Path(__file__).resolve().parents[2]
    candidates.extend([
        project_root / 'simulation-environment' / 'checkpoint' / 'model.ckpt',
        project_root / 'simulation-environment' / 'checkpoints' / 'model.ckpt',
    ])

    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())

    checkpoint_dir = project_root / 'simulation-environment' / 'checkpoint'
    checkpoints_dir = project_root / 'simulation-environment' / 'checkpoints'
    for base_dir in (checkpoint_dir, checkpoints_dir):
        if not base_dir.exists():
            continue
        for candidate in sorted(base_dir.rglob('*.ckpt')):
            return str(candidate.resolve())

    searched_paths = ', '.join(str(path) for path in candidates)
    raise FileNotFoundError(
        'No dynamics checkpoint found. '
        f'Searched: {searched_paths}. '
        'Set RL_DYNAMICS_CHECKPOINT to a local model.ckpt path if needed.'
    )
