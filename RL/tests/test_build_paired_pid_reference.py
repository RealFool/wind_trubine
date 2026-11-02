from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from RL.utils.constants import DATA_SCHEMA_VERSION, STATE_DIM
from RL.utils.paired_pid_reference import (
    PAIRED_PID_REFERENCE_SCHEMA_VERSION,
    PairedPidReference,
    build_paired_pid_reference,
)
from RL.utils.reward import RewardFunction


class TestBuildPairedPidReference(unittest.TestCase):
    @staticmethod
    def _write_raw_files(raw_dir: Path) -> None:
        raw_dir.mkdir()
        fields = [
            'Time',
            'I_AverageWindSpeed',
            'Electrical power',
            'Blade root 1 My (Nm)',
            'Blade root 2 My (Nm)',
            'Blade root 3 My (Nm)',
        ]
        for condition_id in range(3, 26):
            with (raw_dir / f'data_windspeed{condition_id}.csv').open('w', encoding='utf-8', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for time_s in (0.0, 0.02, 180.0, 600.0):
                    writer.writerow({
                        'Time': time_s,
                        'I_AverageWindSpeed': condition_id + time_s / 10_000.0,
                        'Electrical power': condition_id * 1_000_000.0 + time_s,
                        'Blade root 1 My (Nm)': condition_id * 10_000_000.0 + time_s,
                        'Blade root 2 My (Nm)': condition_id * 10_000_000.0 + 2.0 * time_s,
                        'Blade root 3 My (Nm)': condition_id * 10_000_000.0 + 3.0 * time_s,
                    })

    def test_builds_one_fixed_window_with_paired_power_load_and_trace_identity(self) -> None:
        with tempfile.TemporaryDirectory(prefix='paired_pid_reference_v2_') as tmpdir:
            root = Path(tmpdir)
            raw_dir = root / 'raw'
            self._write_raw_files(raw_dir)
            document = build_paired_pid_reference(
                raw_dir,
                time_window_start_s=0.0,
                time_window_end_s=180.0,
                project_root=root,
            )
            output = root / 'paired_pid_reference_180s.json'
            output.write_text(json.dumps(document), encoding='utf-8')
            reference = PairedPidReference.from_path(output, project_root=root)

        self.assertEqual(document['schema_version'], PAIRED_PID_REFERENCE_SCHEMA_VERSION)
        self.assertEqual(document['metadata']['data_schema_version'], DATA_SCHEMA_VERSION)
        self.assertEqual(document['metadata']['time_window_start_s'], 0.0)
        self.assertEqual(document['metadata']['time_window_end_s'], 180.0)
        self.assertEqual(len(document['references']), 23)
        entry = document['references'][5]
        self.assertEqual(entry['condition_id'], 8)
        self.assertEqual(entry['scenario_id'], 'wind8')
        self.assertEqual(entry['exact_timestamps_s'], [0.0, 0.02, 180.0])
        self.assertEqual(entry['pid_electrical_power_w'][1], 8_000_000.02)
        self.assertEqual(entry['pid_blade_root_peak_abs_nm'], [80_000_180.0, 80_000_360.0, 80_000_540.0])
        self.assertEqual(entry['canonical_wind_trace']['sample_count'], 3)
        self.assertEqual(len(entry['canonical_wind_trace']['sha256']), 64)
        self.assertEqual(reference.power_at(8, 0.02), 8_000_000.02)
        self.assertEqual(reference.blade_root_peak(8).tolist(), entry['pid_blade_root_peak_abs_nm'])
        with self.assertRaisesRegex(ValueError, 'no exact timestamp'):
            reference.power_at(8, 600.0)

    def test_windows_are_distinct_and_cannot_be_substituted(self) -> None:
        with tempfile.TemporaryDirectory(prefix='paired_pid_reference_v2_windows_') as tmpdir:
            root = Path(tmpdir)
            raw_dir = root / 'raw'
            self._write_raw_files(raw_dir)
            reference_180 = build_paired_pid_reference(
                raw_dir,
                time_window_start_s=0.0,
                time_window_end_s=180.0,
                project_root=root,
            )
            reference_600 = build_paired_pid_reference(
                raw_dir,
                time_window_start_s=0.0,
                time_window_end_s=600.0,
                project_root=root,
            )

        entry_180 = reference_180['references'][0]
        entry_600 = reference_600['references'][0]
        self.assertNotEqual(entry_180['time_window_end_s'], entry_600['time_window_end_s'])
        self.assertNotEqual(
            entry_180['canonical_wind_trace']['sha256'],
            entry_600['canonical_wind_trace']['sha256'],
        )
        self.assertNotEqual(
            entry_180['pid_blade_root_peak_abs_nm'],
            entry_600['pid_blade_root_peak_abs_nm'],
        )

    def test_reward_uses_one_condition_entry_for_power_and_per_blade_peak(self) -> None:
        with tempfile.TemporaryDirectory(prefix='paired_pid_reference_v2_reward_') as tmpdir:
            root = Path(tmpdir)
            raw_dir = root / 'raw'
            self._write_raw_files(raw_dir)
            document = build_paired_pid_reference(
                raw_dir,
                time_window_start_s=0.0,
                time_window_end_s=180.0,
                project_root=root,
            )
            output = root / 'paired_pid_reference_180s.json'
            output.write_text(json.dumps(document), encoding='utf-8')
            reward = RewardFunction({
                'rated_power': 15_000_000.0,
                'bladed_step_duration_ms': 20.0,
                'reward': {
                    'power_reward_mode': 'relative_pid_energy',
                    'paired_pid_reference_path': str(output),
                    'energy_reference_j': 20_000.0,
                    'blade_root_load_mode': 'my_peak_increment',
                    'blade_root_weight': 1.0,
                    'blade_root_soft_limit': 1.0,
                    'power_weight': 0.0,
                    'acc_weight': 0.0,
                    'shaft_weight': 0.0,
                    'pitch_rate_weight': 0.0,
                    'torque_rate_weight': 0.0,
                    'command_excess_weight': 0.0,
                    'power_tracking_weight': 0.0,
                },
            })
            reward.reset_condition(8)
            state = np.zeros(STATE_DIM, dtype=np.float32)
            state_next = state.copy()
            value, breakdown = reward.compute(
                state_t=state,
                state_next=state_next,
                reward_metrics={
                    'Electrical power': 8_000_000.02,
                    'Blade root 1 My (Nm)': 80_000_180.0,
                    'Blade root 2 My (Nm)': 80_000_360.0,
                    'Blade root 3 My (Nm)': 80_000_540.0,
                },
                action_t=np.zeros(2),
                prev_action_t=np.zeros(2),
                wind_t=999.0,
                condition_id=8,
                power_t=8_000_000.0,
                sim_time_t=0.0,
                sim_time_next=0.02,
                power_reference_raw=reward.reference_power(8, 0.02),
            )

        self.assertEqual(value, 0.0)
        self.assertEqual(breakdown['reward_pid_power_reference_raw'], 8_000_000.02)
        self.assertEqual(breakdown['reward_blade_root_1_peak_reference_nm'], 80_000_180.0)
        self.assertEqual(breakdown['reward_blade_root_2_peak_reference_nm'], 80_000_360.0)
        self.assertEqual(breakdown['reward_blade_root_3_peak_reference_nm'], 80_000_540.0)


if __name__ == '__main__':
    unittest.main()