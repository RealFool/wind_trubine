from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from RL.agents import TD3Agent
from RL.data.offline_dataset import OfflineBCDataset
from RL.utils.config import load_config, normalize_user_path
from RL.utils.control_contract import resolve_configured_control_contract
from RL.utils.constants import ACTION_DIM, DATA_SCHEMA_VERSION
from RL.utils.offline_policy_eval import evaluate_offline_policy, write_offline_policy_evaluation
from RL.utils.safe_residual_control import SafeResidualController


def _checkpoint_agent_state(checkpoint: Any) -> dict[str, Any]:
    if not isinstance(checkpoint, dict):
        raise ValueError('Checkpoint must be a mapping.')
    state = checkpoint.get('agent', checkpoint)
    if not isinstance(state, dict) or 'actor' not in state:
        raise ValueError('Checkpoint does not contain a TD3 agent state.')
    return state


def _resolve_device(config: dict[str, Any], requested_device: str = '') -> torch.device:
    device_name = requested_device.strip() or str(config.get('device', 'cpu'))
    if device_name.startswith('cuda') and not torch.cuda.is_available():
        device_name = 'cpu'
    return torch.device(device_name)


def _config_from_checkpoint(checkpoint: dict[str, Any], config_path: str = '') -> dict[str, Any]:
    if config_path.strip():
        config = load_config(config_path)
    else:
        config = copy.deepcopy(checkpoint.get('config'))
    if not isinstance(config, dict) or not isinstance(config.get('data'), dict):
        raise ValueError('Provide --config or use a checkpoint with an embedded data configuration.')
    if not isinstance(config.get('env'), dict) or not isinstance(config.get('control'), dict):
        raise ValueError('Offline evaluation config must contain env and control objects.')
    if not isinstance(config.get('agent'), dict):
        raise ValueError('Offline evaluation config must contain an agent object.')
    if str(config.get('data_schema_version', '')).strip() != DATA_SCHEMA_VERSION:
        raise ValueError(f'Config data_schema_version must be {DATA_SCHEMA_VERSION!r}.')
    return config


def _dataset(config: dict[str, Any], split: str) -> OfflineBCDataset:
    data_cfg = config['data']
    offline_cfg = config.get('offline_eval', {})
    bc_cfg = config.get('bc', {})
    if not isinstance(offline_cfg, dict) or not isinstance(bc_cfg, dict):
        raise ValueError('offline_eval and bc must be objects when configured.')
    return OfflineBCDataset(
        root_dir=str(data_cfg['root_dir']),
        split=split,
        max_files=data_cfg.get('max_bc_files'),
        observation_wind_override=offline_cfg.get(
            'observation_wind_override',
            bc_cfg.get('observation_wind_override'),
        ),
        cache_enabled=bool(offline_cfg.get('cache_enabled', bc_cfg.get('cache_enabled', True))),
        cache_dir=offline_cfg.get('cache_dir', bc_cfg.get('cache_dir')),
        include_previous_action_observation=bool(
            config['env'].get('include_previous_action_observation', False)
        ),
        action_low=config['env'].get('action_low'),
        action_high=config['env'].get('action_high'),
        include_operating_state_observation=bool(
            config['env'].get('include_operating_state_observation', False)
        ),
        state_action_artifact_path=config['env'].get('state_action_artifact_path'),
    )


def _validate_checkpoint_contract(
    checkpoint: dict[str, Any],
    controller: SafeResidualController,
    expected_resolved_control_contract_fingerprint: str | None = None,
) -> None:
    if str(checkpoint.get('data_schema_version', '')).strip() != DATA_SCHEMA_VERSION:
        raise ValueError('Checkpoint data schema is incompatible with the offline evaluator.')
    if expected_resolved_control_contract_fingerprint is not None:
        if checkpoint.get('control_contract_fingerprint') != expected_resolved_control_contract_fingerprint:
            raise ValueError('Checkpoint resolved control contract is incompatible with the offline evaluator.')
    checkpoint_actor_fingerprint = str(checkpoint.get('actor_control_fingerprint', '')).strip()
    if checkpoint_actor_fingerprint:
        if checkpoint_actor_fingerprint != controller.actor_control_fingerprint:
            raise ValueError('Checkpoint actor control contract is incompatible with the configured safety envelope.')
        return
    checkpoint_control_fingerprint = str(checkpoint.get('control_fingerprint', '')).strip()
    if checkpoint_control_fingerprint and checkpoint_control_fingerprint != controller.control_fingerprint:
        raise ValueError('Checkpoint control contract is incompatible with the configured control contract.')


def _default_output_dir(checkpoint_path: Path) -> Path:
    base_dir = checkpoint_path.parent.parent if checkpoint_path.parent.name == 'checkpoints' else checkpoint_path.parent
    return base_dir / 'offline_eval' / checkpoint_path.stem


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    config: dict[str, Any],
    output_dir: str | Path,
    *,
    device: torch.device | None = None,
    batch_size: int | None = None,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if not isinstance(checkpoint, dict):
        raise ValueError('Checkpoint must deserialize to a mapping.')
    config = _config_from_checkpoint(checkpoint) if config is None else config
    runtime_device = _resolve_device(config) if device is None else device
    configured_control_contract = resolve_configured_control_contract(
        config,
        project_root=PROJECT_ROOT,
    )
    controller = SafeResidualController(config['env'], config['control'])
    _validate_checkpoint_contract(
        checkpoint,
        controller,
        None if configured_control_contract is None else configured_control_contract[1],
    )

    offline_cfg = config.get('offline_eval', {})
    if not isinstance(offline_cfg, dict):
        raise ValueError('offline_eval must be an object when configured.')
    normalizer_split = str(offline_cfg.get('normalizer_split', config.get('bc', {}).get('split', 'train')))
    evaluation_split = str(offline_cfg.get('split', config['data'].get('eval_split', 'val')))
    normalizer_dataset = _dataset(config, normalizer_split)
    evaluation_dataset = _dataset(config, evaluation_split)
    controller.set_observation_distribution(normalizer_dataset.obs_median, normalizer_dataset.obs_iqr)

    agent = TD3Agent(int(evaluation_dataset.obs.shape[1]), ACTION_DIM, config['agent'], runtime_device)
    agent.load_state_dict(_checkpoint_agent_state(checkpoint))
    evaluation = evaluate_offline_policy(
        agent,
        evaluation_dataset,
        controller,
        runtime_device,
        batch_size=int(batch_size or offline_cfg.get('batch_size', 1024)),
    )

    destination = Path(output_dir).expanduser().resolve()
    write_offline_policy_evaluation(destination, evaluation)
    result = {
        'mode': 'offline_behavior_safety_v1',
        'checkpoint_path': str(checkpoint_path),
        'output_dir': str(destination),
        'normalizer_split': normalizer_split,
        'evaluation_split': evaluation_split,
        'device': str(runtime_device),
        'summary': evaluation.summary,
    }
    with (destination / 'offline_policy_eval_run.json').open('w', encoding='utf-8') as file:
        json.dump(result, file, ensure_ascii=False, indent=2, allow_nan=False)
    return result


def _load_job_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open('r', encoding='utf-8') as file:
        for line_number, line in enumerate(file, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f'Invalid JSON in {path} line {line_number}') from exc
            if not isinstance(row, dict) or not str(row.get('checkpoint_path', '')).strip():
                raise ValueError(f'{path} line {line_number} requires checkpoint_path')
            rows.append(row)
    return rows


def _resolve_job_checkpoint(path_text: str, jobs_path: Path) -> Path:
    candidate = normalize_user_path(path_text)
    if candidate.is_absolute() or candidate.is_file():
        return candidate
    return jobs_path.parent / candidate


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run behavior-and-safety-only offline evaluation for frozen TD3 checkpoints.')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--checkpoint', default='', help='One checkpoint to evaluate.')
    source.add_argument('--jobs-file', default='', help='JSONL jobs emitted by queue training after completed rounds.')
    parser.add_argument('--config', default='', help='Optional config; otherwise use the checkpoint embedded config.')
    parser.add_argument('--output-dir', default='', help='Output directory for one checkpoint or a parent directory for jobs.')
    parser.add_argument('--device', default='', help='Override device, for example cpu or cuda:0.')
    parser.add_argument('--batch-size', type=int, default=0, help='Override offline evaluation batch size.')
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.jobs_file:
        jobs_path = normalize_user_path(args.jobs_file).resolve()
        rows = _load_job_rows(jobs_path)
        output_root = normalize_user_path(args.output_dir).resolve() if args.output_dir.strip() else None
        results: list[dict[str, Any]] = []
        for row in rows:
            checkpoint_path = _resolve_job_checkpoint(str(row['checkpoint_path']), jobs_path).resolve()
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if not isinstance(checkpoint, dict):
                raise ValueError(f'Checkpoint must deserialize to a mapping: {checkpoint_path}')
            config = _config_from_checkpoint(checkpoint, args.config)
            output_dir = output_root / checkpoint_path.stem if output_root is not None else _default_output_dir(checkpoint_path)
            results.append(
                evaluate_checkpoint(
                    checkpoint_path,
                    config,
                    output_dir,
                    device=_resolve_device(config, args.device),
                    batch_size=args.batch_size or None,
                )
            )
        print(json.dumps({'job_count': len(results), 'results': results}, ensure_ascii=False, indent=2, allow_nan=False))
        return

    checkpoint_path = normalize_user_path(args.checkpoint).resolve()
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if not isinstance(checkpoint, dict):
        raise ValueError('Checkpoint must deserialize to a mapping.')
    config = _config_from_checkpoint(checkpoint, args.config)
    output_dir = normalize_user_path(args.output_dir).resolve() if args.output_dir.strip() else _default_output_dir(checkpoint_path)
    result = evaluate_checkpoint(
        checkpoint_path,
        config,
        output_dir,
        device=_resolve_device(config, args.device),
        batch_size=args.batch_size or None,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()