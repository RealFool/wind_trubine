from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from RL.evaluate_persistent_power_deficit_reward import evaluate_persistent_power_deficit


class TestPersistentPowerDeficitAblation(unittest.TestCase):
    @staticmethod
    def _fixture(root: Path) -> Path:
        reference = root / 'reference.csv'
        with reference.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['Time', 'Electrical power'])
            writer.writeheader()
            for index in range(101):
                writer.writerow({'Time': float(index), 'Electrical power': 10_000_000.0})
        manifest = root / 'manifest.json'
        manifest.write_text(json.dumps({
            'metadata': {'input_manifest_sha256': 'a' * 64},
            'references': [{
                'condition_id': 8,
                'path': str(reference),
                'sha256': hashlib.sha256(reference.read_bytes()).hexdigest(),
            }],
        }), encoding='utf-8')
        return manifest

    def test_ablation_is_zero_centered_and_distinguishes_duration(self) -> None:
        with tempfile.TemporaryDirectory(prefix='r1_ablation_') as tmpdir:
            report = evaluate_persistent_power_deficit(
                self._fixture(Path(tmpdir)),
                rated_power_w=10_000_000.0,
                weight=0.1,
                deadband_fraction=0.01,
                tau_s=2.0,
                clip=1.0,
            )
            self.assertEqual(report['validation_status'], 'passed')
            self.assertTrue(all(report['checks'].values()))
            self.assertFalse(report['formal_closed_loop_acceptance'])
            self.assertEqual(report['scenarios']['exact_pid']['term_max'], 0.0)
            self.assertLess(
                report['scenarios']['sustained_5pct_10s']['term_cumulative'],
                report['scenarios']['sustained_10pct_60s']['term_cumulative'],
            )

    def test_rejects_reference_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory(prefix='r1_ablation_') as tmpdir:
            manifest_path = self._fixture(Path(tmpdir))
            document = json.loads(manifest_path.read_text())
            document['references'][0]['sha256'] = '0' * 64
            manifest_path.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, 'SHA-256'):
                evaluate_persistent_power_deficit(
                    manifest_path,
                    rated_power_w=10_000_000.0,
                    weight=0.1,
                    deadband_fraction=0.01,
                    tau_s=2.0,
                    clip=1.0,
                )


if __name__ == '__main__':
    unittest.main()