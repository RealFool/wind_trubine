from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from RL.utils.critic_q_health import ActorQHealthGate, ActorQHealthGateConfig


SCHEMA_VERSION = 'actor_q_health_gate_offline_replay_v1'


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_updates(path: Path) -> list[dict[str, float]]:
    required = {
        'update',
        'critic1_loss',
        'critic1_td_error_abs_mean',
        'critic2_td_error_abs_mean',
        'critic_q_abs_gap_mean',
    }
    updates: list[dict[str, float]] = []
    with path.open('r', encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f'{path.name} is missing actor gate fields: {missing}')
        for row_number, row in enumerate(reader, start=2):
            try:
                updates.append({key: float(value) for key, value in row.items() if value != ''})
            except (TypeError, ValueError) as exc:
                raise ValueError(f'{path.name} row {row_number} has invalid actor gate data') from exc
    if not updates:
        raise ValueError('Actor gate replay needs at least one critic update')
    return updates


def _replay(updates: list[dict[str, float]], physical_gate_passed: bool) -> dict[str, Any]:
    gate = ActorQHealthGate(ActorQHealthGateConfig(
        enabled=True,
        physical_gate_passed=physical_gate_passed,
        min_successful_critic_updates=1000,
        window_updates=200,
        max_guard_rate=0.05,
        min_success_rate=0.90,
        max_td_error_relative_increase=0.10,
        max_q_gap_relative_increase=0.10,
    ))
    first_open_update: int | None = None
    final_metrics: dict[str, float] = gate.metrics()
    for row in updates:
        update_metrics = {
            'critic1_loss': row['critic1_loss'],
            'critic1_td_error_abs_mean': row['critic1_td_error_abs_mean'],
            'critic2_td_error_abs_mean': row['critic2_td_error_abs_mean'],
            'critic_q_abs_gap_mean': row['critic_q_abs_gap_mean'],
        }
        if row.get('update_skipped_q_guard', 0.0) > 0.0:
            update_metrics = {'update_skipped_q_guard': 1.0}
        final_metrics = gate.observe(update_metrics)
        if first_open_update is None and gate.allows_actor_update:
            first_open_update = int(row['update'])
    return {
        'physical_gate_passed': physical_gate_passed,
        'first_open_update': first_open_update,
        'open_after_final_update': gate.allows_actor_update,
        'final_metrics': final_metrics,
    }


def evaluate_actor_q_health_gate(updates_path: Path, expected_sha256: str) -> dict[str, Any]:
    updates_path = updates_path.expanduser().resolve()
    actual_sha256 = _sha256(updates_path)
    if actual_sha256 != expected_sha256:
        raise ValueError('Actor gate updates CSV SHA-256 mismatch')
    updates = _read_updates(updates_path)
    locked = _replay(updates, physical_gate_passed=False)
    counterfactual = _replay(updates, physical_gate_passed=True)
    checks = {
        'physical_gate_false_never_opens': locked['first_open_update'] is None,
        'physical_gate_true_opens_after_minimum_updates': (
            counterfactual['first_open_update'] is not None
            and counterfactual['first_open_update'] >= 1000
        ),
        'full_q_health_window_ready': counterfactual['final_metrics']['actor_q_health_window_ready'] == 1.0,
        'success_rate_above_threshold': counterfactual['final_metrics']['actor_q_health_window_success_rate'] > 0.90,
        'guard_rate_below_threshold': counterfactual['final_metrics']['actor_q_health_window_guard_rate'] < 0.05,
    }
    return {
        'schema_version': SCHEMA_VERSION,
        'validation_scope': 'offline_replay_of_formal_q3_critic_updates',
        'validation_status': 'passed' if all(checks.values()) else 'failed',
        'formal_actor_enablement': False,
        'updates_path': str(updates_path),
        'updates_sha256': actual_sha256,
        'scheduled_updates': len(updates),
        'gate_config': {
            'min_successful_critic_updates': 1000,
            'window_updates': 200,
            'max_guard_rate': 0.05,
            'min_success_rate': 0.90,
            'max_td_error_relative_increase': 0.10,
            'max_q_gap_relative_increase': 0.10,
        },
        'checks': checks,
        'physical_gate_locked_replay': locked,
        'physical_gate_passed_counterfactual': counterfactual,
        'limitation': (
            'The physical-gate-passed branch is counterfactual. Actor updates remain forbidden until '
            'a real Bladed physical-gate artifact is supplied.'
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Replay the A1 actor Q-health gate offline.')
    parser.add_argument('--updates', type=Path, required=True)
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_actor_q_health_gate(args.updates, args.expected_sha256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({
        'output': str(args.output),
        'validation_status': report['validation_status'],
        'scheduled_updates': report['scheduled_updates'],
        'checks': report['checks'],
        'locked_first_open_update': report['physical_gate_locked_replay']['first_open_update'],
        'counterfactual_first_open_update': report['physical_gate_passed_counterfactual']['first_open_update'],
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()