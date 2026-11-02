from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from RL.utils.reward import RewardFunction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Replay paired PID power references and verify relative energy reward is zero-centered.'
    )
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--energy-reference-j', type=float, default=300_000.0)
    parser.add_argument('--relative-energy-clip', type=float, default=2.0)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reward_function(
    manifest_path: Path,
    *,
    energy_reference_j: float,
    relative_energy_clip: float,
) -> RewardFunction:
    return RewardFunction({
        'rated_power': 15_000_000.0,
        'bladed_step_duration_ms': 20.0,
        'reward': {
            'power_reward_mode': 'relative_pid_energy',
            'pid_power_reference_manifest_path': str(manifest_path),
            'energy_reference_j': energy_reference_j,
            'relative_energy_clip': relative_energy_clip,
            'power_weight': 1.0,
            'blade_root_weight': 0.0,
            'acc_weight': 0.0,
            'shaft_weight': 0.0,
            'pitch_rate_weight': 0.0,
            'torque_rate_weight': 0.0,
            'command_excess_weight': 0.0,
            'power_tracking_weight': 0.0,
            'power_floor_weight': 0.0,
            'blade_root_continuous_weight': 0.0,
        },
    })


def _reference_rows(path: Path) -> tuple[np.ndarray, np.ndarray]:
    times: list[float] = []
    powers: list[float] = []
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        missing = {'Time', 'Electrical power'} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f'{path.name} is missing relative reward fields: {sorted(missing)}')
        for row_number, row in enumerate(reader, start=2):
            try:
                timestamp = float(row['Time'])
                power = float(row['Electrical power'])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f'{path.name} row {row_number} has invalid relative reward data') from exc
            if not math.isfinite(timestamp) or not math.isfinite(power):
                raise ValueError(f'{path.name} row {row_number} has non-finite relative reward data')
            if times and timestamp <= times[-1]:
                raise ValueError(f'{path.name} timestamps must be strictly increasing')
            times.append(timestamp)
            powers.append(power)
    if len(times) < 2:
        raise ValueError(f'{path.name} relative reward replay needs at least two rows')
    return np.asarray(times, dtype=np.float64), np.asarray(powers, dtype=np.float64)


def evaluate_relative_pid_reward(
    manifest_path: Path,
    *,
    energy_reference_j: float,
    relative_energy_clip: float,
) -> dict[str, Any]:
    if not math.isfinite(energy_reference_j) or energy_reference_j <= 0.0:
        raise ValueError('energy_reference_j must be positive and finite')
    if not math.isfinite(relative_energy_clip) or relative_energy_clip <= 0.0:
        raise ValueError('relative_energy_clip must be positive and finite')
    resolved_manifest = manifest_path.expanduser().resolve()
    reward_function = _reward_function(
        resolved_manifest,
        energy_reference_j=energy_reference_j,
        relative_energy_clip=relative_energy_clip,
    )
    try:
        manifest = json.loads(resolved_manifest.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f'Unable to read paired PID power reference manifest: {resolved_manifest}') from exc
    references = manifest.get('references') if isinstance(manifest, dict) else None
    if not isinstance(references, list) or not references:
        raise ValueError('Paired PID power reference manifest contains no references')

    condition_summaries: list[dict[str, Any]] = []
    all_relative_power: list[np.ndarray] = []
    all_relative_energy: list[np.ndarray] = []
    all_reward_terms: list[np.ndarray] = []
    for descriptor in references:
        condition_id = int(descriptor['condition_id'])
        path = Path(str(descriptor['path'])).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        times, powers = _reference_rows(path)
        reference_powers = np.asarray([
            reward_function.reference_power(condition_id, timestamp)
            for timestamp in times[1:]
        ], dtype=np.float64)
        relative_power = powers[1:] - reference_powers
        delta_time = np.diff(times)
        relative_energy = relative_power * delta_time
        reward_terms = np.clip(
            relative_energy / energy_reference_j,
            -relative_energy_clip,
            relative_energy_clip,
        )
        condition_summaries.append({
            'condition_id': condition_id,
            'filename': path.name,
            'sha256': _sha256(path),
            'transition_count': int(len(relative_power)),
            'max_abs_relative_power_w': float(np.max(np.abs(relative_power))),
            'mean_relative_power_w': float(np.mean(relative_power)),
            'cumulative_relative_energy_j': float(np.sum(relative_energy)),
            'mean_relative_reward_term': float(np.mean(reward_terms)),
            'max_abs_relative_reward_term': float(np.max(np.abs(reward_terms))),
        })
        all_relative_power.append(relative_power)
        all_relative_energy.append(relative_energy)
        all_reward_terms.append(reward_terms)

    relative_power = np.concatenate(all_relative_power)
    relative_energy = np.concatenate(all_relative_energy)
    reward_terms = np.concatenate(all_reward_terms)
    max_abs_relative_power = float(np.max(np.abs(relative_power)))
    max_abs_reward = float(np.max(np.abs(reward_terms)))
    return {
        'schema_version': 'relative_pid_reward_replay_evaluation_v1',
        'validation_scope': 'exact_time_same_reference_raw_replay',
        'validation_status': 'passed' if max_abs_relative_power <= 1.0e-9 and max_abs_reward <= 1.0e-12 else 'failed',
        'manifest_path': str(resolved_manifest),
        'manifest_sha256': _sha256(resolved_manifest),
        'input_manifest_sha256': manifest['metadata']['input_manifest_sha256'],
        'energy_reference_j': energy_reference_j,
        'relative_energy_clip': relative_energy_clip,
        'condition_count': len(condition_summaries),
        'transition_count': int(len(relative_power)),
        'aggregate': {
            'max_abs_relative_power_w': max_abs_relative_power,
            'mean_relative_power_w': float(np.mean(relative_power)),
            'cumulative_relative_energy_j': float(np.sum(relative_energy)),
            'mean_relative_reward_term': float(np.mean(reward_terms)),
            'max_abs_relative_reward_term': max_abs_reward,
        },
        'conditions': condition_summaries,
    }


def main() -> None:
    args = parse_args()
    document = evaluate_relative_pid_reward(
        args.manifest,
        energy_reference_j=float(args.energy_reference_j),
        relative_energy_clip=float(args.relative_energy_clip),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({
        'output': str(args.output),
        'validation_status': document['validation_status'],
        'condition_count': document['condition_count'],
        'transition_count': document['transition_count'],
        **document['aggregate'],
    }, indent=2))


if __name__ == '__main__':
    main()