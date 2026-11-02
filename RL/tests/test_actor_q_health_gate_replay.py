from __future__ import annotations

import csv
import hashlib
from pathlib import Path
import tempfile
import unittest

from RL.evaluate_actor_q_health_gate import evaluate_actor_q_health_gate


class TestActorQHealthGateReplay(unittest.TestCase):
    def test_physical_gate_remains_authoritative(self) -> None:
        with tempfile.TemporaryDirectory(prefix='actor_gate_replay_') as tmpdir:
            path = Path(tmpdir) / 'updates.csv'
            with path.open('w', encoding='utf-8', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=[
                    'update', 'critic1_loss', 'critic1_td_error_abs_mean',
                    'critic2_td_error_abs_mean', 'critic_q_abs_gap_mean',
                ])
                writer.writeheader()
                for update in range(1, 1001):
                    writer.writerow({
                        'update': update,
                        'critic1_loss': 1.0,
                        'critic1_td_error_abs_mean': 1.0,
                        'critic2_td_error_abs_mean': 1.0,
                        'critic_q_abs_gap_mean': 1.0,
                    })
            report = evaluate_actor_q_health_gate(
                path,
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )
            self.assertEqual(report['validation_status'], 'passed')
            self.assertIsNone(report['physical_gate_locked_replay']['first_open_update'])
            self.assertEqual(report['physical_gate_passed_counterfactual']['first_open_update'], 1000)
            self.assertFalse(report['formal_actor_enablement'])


if __name__ == '__main__':
    unittest.main()