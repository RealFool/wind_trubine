from __future__ import annotations

from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from RL.build_paired_pid_power_reference import main
from RL.evaluate_relative_pid_reward import evaluate_relative_pid_reward
from RL.utils.constants import DATA_SCHEMA_VERSION
from RL.utils.reward import PAIRED_PID_POWER_REFERENCE_SCHEMA_VERSION, RewardFunction


class TestBuildPairedPidPowerReference(unittest.TestCase):
    @staticmethod
    def _write_raw_files(raw_dir: Path) -> None:
        raw_dir.mkdir()
        for condition_id in range(3, 26):
            path = raw_dir / f'data_windspeed{condition_id}.csv'
            with path.open('w', encoding='utf-8', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=['Time', 'Electrical power'])
                writer.writeheader()
                writer.writerows([
                    {'Time': 0.0, 'Electrical power': condition_id * 1_000_000.0},
                    {'Time': 0.02, 'Electrical power': condition_id * 1_000_000.0 + 10_000.0},
                    {'Time': 600.0, 'Electrical power': condition_id * 1_000_000.0 + 20_000.0},
                ])

    @staticmethod
    def _run(raw_dir: Path, output: Path) -> None:
        argv = [
            'build_paired_pid_power_reference.py',
            '--data-root', str(raw_dir),
            '--output', str(output),
        ]
        with patch.object(sys, 'argv', argv), redirect_stdout(io.StringIO()):
            main()

    def test_builds_hashed_exact_time_reference_for_all_conditions(self) -> None:
        with tempfile.TemporaryDirectory(prefix='paired_pid_reference_') as tmpdir:
            root = Path(tmpdir)
            raw_dir = root / 'raw'
            output = root / 'paired_pid_reference.json'
            self._write_raw_files(raw_dir)

            self._run(raw_dir, output)

            document = json.loads(output.read_text(encoding='utf-8'))
            self.assertEqual(document['schema_version'], PAIRED_PID_POWER_REFERENCE_SCHEMA_VERSION)
            metadata = document['metadata']
            self.assertEqual(metadata['data_schema_version'], DATA_SCHEMA_VERSION)
            self.assertEqual(metadata['source_role'], 'paired_pid_outcome_reference')
            self.assertEqual(metadata['reference_signal'], 'Electrical power')
            self.assertEqual(metadata['condition_count'], 23)
            self.assertEqual(metadata['expected_time_range_s'], [0.0, 600.0])
            self.assertEqual(metadata['pairing_contract'], {
                'case_identity': 'same_bladed_case',
                'turbulence_seed_identity': 'same_seed',
                'time_window_identity': 'same_simulation_time',
                'time_alignment': 'exact_timestamp_no_interpolation',
            })
            self.assertEqual(len(metadata['input_manifest_sha256']), 64)
            self.assertEqual(len(document['references']), 23)
            self.assertTrue(all(len(item['sha256']) == 64 for item in document['references']))

            reward = RewardFunction({
                'rated_power': 15_000_000.0,
                'reward': {
                    'power_reward_mode': 'relative_pid_energy',
                    'pid_power_reference_manifest_path': str(output),
                },
            })
            self.assertEqual(reward.reference_power(8, 0.02), 8_010_000.0)

            report = evaluate_relative_pid_reward(
                output,
                energy_reference_j=300_000.0,
                relative_energy_clip=2.0,
            )
            self.assertEqual(report['validation_status'], 'passed')
            self.assertEqual(report['condition_count'], 23)
            self.assertEqual(report['transition_count'], 46)
            self.assertEqual(report['aggregate']['max_abs_relative_power_w'], 0.0)
            self.assertEqual(report['aggregate']['cumulative_relative_energy_j'], 0.0)
            self.assertEqual(report['aggregate']['max_abs_relative_reward_term'], 0.0)

    def test_rejects_processed_split_directory(self) -> None:
        with tempfile.TemporaryDirectory(prefix='paired_pid_reference_') as tmpdir:
            root = Path(tmpdir)
            train_dir = root / 'train'
            self._write_raw_files(train_dir)

            with self.assertRaisesRegex(ValueError, "must be named 'raw'"):
                self._run(train_dir, root / 'paired_pid_reference.json')


if __name__ == '__main__':
    unittest.main()