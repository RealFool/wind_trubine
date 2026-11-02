from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from RL.agents import TD3Agent
from RL.data.offline_dataset import OfflineBCDataset
from RL.utils.config import load_config, normalize_user_path
from RL.utils.constants import ACTION_DIM
from RL.utils.control_contract import resolve_configured_control_contract
from RL.utils.safe_residual_control import SafeResidualController


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET_COLUMNS = ['B4 residual torque target', 'B4 residual pitch target']


def _load_dataset(config: dict[str, Any], split: str) -> OfflineBCDataset:
    return OfflineBCDataset(
        root_dir=config['data']['root_dir'],
        split=split,
        max_files=config['data'].get('max_bc_files'),
        cache_enabled=bool(config['bc'].get('cache_enabled', True)),
        cache_dir=config['bc'].get('cache_dir'),
        include_previous_action_observation=bool(config['env']['include_previous_action_observation']),
        action_low=config['env']['action_low'],
        action_high=config['env']['action_high'],
        precomputed_action_target_columns=TARGET_COLUMNS,
        include_operating_state_observation=bool(
            config['env'].get('include_operating_state_observation', False)
        ),
        state_action_artifact_path=config['env'].get('state_action_artifact_path'),
    )


def evaluate(config_path: Path, checkpoint_path: Path, split: str, batch_size: int) -> dict[str, Any]:
    config = load_config(str(config_path))
    dataset = _load_dataset(config, split)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    configured_control_contract = resolve_configured_control_contract(
        config,
        project_root=PROJECT_ROOT,
    )
    controller = SafeResidualController(config['env'], config['control'])
    if (
        configured_control_contract is not None
        and checkpoint.get('control_contract_fingerprint') != configured_control_contract[1]
    ):
        raise ValueError('BC checkpoint resolved control contract is incompatible with the evaluation config')
    if checkpoint.get('actor_control_fingerprint') != controller.actor_control_fingerprint:
        raise ValueError('BC checkpoint actor control contract is incompatible with the evaluation config')
    agent = TD3Agent(int(dataset.obs.shape[1]), ACTION_DIM, config['agent'], torch.device('cpu'))
    agent.load_state_dict(checkpoint['agent'])
    predictions: list[np.ndarray] = []
    agent.actor.eval()
    with torch.inference_mode():
        for start in range(0, len(dataset), batch_size):
            observations = torch.from_numpy(dataset.obs[start:start + batch_size])
            predictions.append(
                agent.actor(observations, agent.actor_residual_target_scale).cpu().numpy()
            )
    predicted = np.concatenate(predictions, axis=0)
    targets = dataset.action_targets
    if targets is None:
        raise ValueError('B4 residual validation data must contain precomputed targets')
    absolute_error = np.abs(predicted - targets)
    return {
        'schema_version': 'b4_residual_bc_holdout_v1',
        'split': split,
        'samples': int(len(dataset)),
        'normalized_residual_mae': np.mean(absolute_error, axis=0).astype(float).tolist(),
        'normalized_residual_rmse': np.sqrt(np.mean(np.square(predicted - targets), axis=0)).astype(float).tolist(),
        'normalized_residual_abs_error_p95': np.quantile(absolute_error, 0.95, axis=0).astype(float).tolist(),
        'predicted_abs_p99': np.quantile(np.abs(predicted), 0.99, axis=0).astype(float).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate B4-centered residual BC on precomputed holdout targets.')
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--split', default='val')
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = evaluate(
        normalize_user_path(str(args.config)),
        normalize_user_path(str(args.checkpoint)),
        args.split,
        args.batch_size,
    )
    output_path = normalize_user_path(str(args.output))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()