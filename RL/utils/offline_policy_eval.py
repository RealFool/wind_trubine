from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from RL.data.offline_dataset import OfflineBCDataset, OfflineWindow
from RL.utils.safe_residual_control import SafeResidualController


@dataclass(frozen=True)
class OfflinePolicyEvaluation:
    summary: dict[str, float]
    records: list[dict[str, float]]


def _policy_residuals(agent: Any, observations: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    residual_chunks: list[np.ndarray] = []
    for start_index in range(0, len(observations), batch_size):
        stop_index = min(start_index + batch_size, len(observations))
        obs_tensor = torch.from_numpy(np.asarray(observations[start_index:stop_index], dtype=np.float32)).to(device)
        residual = agent.select_action(obs_tensor, deterministic=True)
        residual_chunks.append(residual.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(residual_chunks, axis=0)


def _window_records(
    agent: Any,
    dataset: OfflineBCDataset,
    window: OfflineWindow,
    controller: SafeResidualController,
    device: torch.device,
    batch_size: int,
) -> list[dict[str, float]]:
    observations = np.asarray(dataset.obs[window.start_index:window.end_index], dtype=np.float32)
    target_actions = np.asarray(dataset.actions[window.start_index:window.end_index], dtype=np.float32)
    if len(observations) == 0:
        return []
    controller.reset_condition(observations[0])
    records: list[dict[str, float]] = []
    condition_id = -1 if window.condition_id is None else int(window.condition_id)
    residuals = _policy_residuals(agent, observations, device, batch_size)
    for local_index, (obs, target_action, residual) in enumerate(
        zip(observations, target_actions, residuals, strict=True)
    ):
        safe_action = controller.action_from_policy(
            residual,
            obs,
            allow_policy=True,
        )
        action_error = np.abs(safe_action.action_real - target_action)
        records.append(
            {
                'window_index': float(window.window_index),
                'condition_id': float(condition_id),
                'sample_index': float(window.start_index + local_index),
                'torque_abs_error': float(action_error[0]),
                'pitch_abs_error': float(action_error[1]),
                'ood_score': float(safe_action.ood_score),
                'fallback_active': float(safe_action.fallback_active),
                'action_projection_applied': float(safe_action.action_projection_applied),
                'action_rate_limited': float(safe_action.action_rate_limited),
                'residual_saturated': float(np.any(np.abs(safe_action.residual_norm) >= 0.999)),
                'sent_action_torque': float(safe_action.action_real[0]),
                'sent_action_pitch': float(safe_action.action_real[1]),
                'pid_action_torque': float(target_action[0]),
                'pid_action_pitch': float(target_action[1]),
            }
        )
    return records


def _summarize_records(records: list[dict[str, float]]) -> dict[str, float]:
    if not records:
        return {'sample_count': 0.0}
    summary: dict[str, float] = {'sample_count': float(len(records))}
    grouped: dict[int, list[dict[str, float]]] = defaultdict(list)
    for record in records:
        grouped[int(record['condition_id'])].append(record)

    for prefix, rows in [('overall', records), *[(f'condition_{condition_id}', values) for condition_id, values in sorted(grouped.items())]]:
        torque_errors = np.asarray([row['torque_abs_error'] for row in rows], dtype=np.float64)
        pitch_errors = np.asarray([row['pitch_abs_error'] for row in rows], dtype=np.float64)
        ood_scores = np.asarray([row['ood_score'] for row in rows], dtype=np.float64)
        summary.update(
            {
                f'{prefix}_sample_count': float(len(rows)),
                f'{prefix}_torque_mae': float(torque_errors.mean()),
                f'{prefix}_torque_abs_error_p95': float(np.percentile(torque_errors, 95)),
                f'{prefix}_pitch_mae': float(pitch_errors.mean()),
                f'{prefix}_pitch_abs_error_p95': float(np.percentile(pitch_errors, 95)),
                f'{prefix}_ood_max': float(ood_scores.max()),
                f'{prefix}_fallback_fraction': float(np.mean([row['fallback_active'] for row in rows])),
                f'{prefix}_action_projection_fraction': float(
                    np.mean([row['action_projection_applied'] for row in rows])
                ),
                f'{prefix}_action_rate_limited_fraction': float(
                    np.mean([row['action_rate_limited'] for row in rows])
                ),
                f'{prefix}_residual_saturation_fraction': float(
                    np.mean([row['residual_saturated'] for row in rows])
                ),
            }
        )
    return summary


def evaluate_offline_policy(
    agent: Any,
    dataset: OfflineBCDataset,
    controller: SafeResidualController,
    device: torch.device,
    batch_size: int = 1024,
) -> OfflinePolicyEvaluation:
    if batch_size <= 0:
        raise ValueError('batch_size must be positive')
    records: list[dict[str, float]] = []
    for window in dataset.windows:
        records.extend(_window_records(agent, dataset, window, controller, device, batch_size))
    return OfflinePolicyEvaluation(summary=_summarize_records(records), records=records)


def write_offline_policy_evaluation(output_dir: str | Path, evaluation: OfflinePolicyEvaluation) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    records_path = destination / 'offline_policy_eval.csv'
    fields = list(evaluation.records[0].keys()) if evaluation.records else []
    with records_path.open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(evaluation.records)
    with (destination / 'offline_policy_eval_summary.json').open('w', encoding='utf-8') as file:
        json.dump(evaluation.summary, file, ensure_ascii=False, indent=2)