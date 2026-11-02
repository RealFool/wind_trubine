from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

from RL.infer.plot_round_progress import build_round_progress


INFERENCE_FIELDS = [
    'record_type',
    'inference_episode',
    'inference_global_control_step',
    'info_control_fallback_active',
    'info_control_ood_score',
    'sent_action_torque',
    'sent_action_pitch',
    'Time',
    'I_AverageWindSpeed',
    'Electrical power',
    'Blade root 1 My (Nm)',
    'Blade root 2 My (Nm)',
    'Blade root 3 My (Nm)',
]
PID_FIELDS = [
    'Time',
    'I_AverageWindSpeed',
    'Electrical power',
    'O_PcsTorqueDemand',
    'O_PitchPositionDemand1',
    'Blade root 1 My (Nm)',
    'Blade root 2 My (Nm)',
    'Blade root 3 My (Nm)',
]


class TestPlotRoundProgress(unittest.TestCase):
    @staticmethod
    def _write_inference(path: Path, power: float, blade_peak: float) -> None:
        rows = []
        for index in range(3):
            rows.append({
                'record_type': 'control_step',
                'inference_episode': '1',
                'inference_global_control_step': str(index + 1),
                'info_control_fallback_active': '0',
                'info_control_ood_score': '1.0',
                'sent_action_torque': '20000000',
                'sent_action_pitch': '0.1',
                'Time': f'{index * 0.02:.2f}',
                'I_AverageWindSpeed': '16.0',
                'Electrical power': str(power),
                'Blade root 1 My (Nm)': str(blade_peak),
                'Blade root 2 My (Nm)': str(blade_peak * 0.95),
                'Blade root 3 My (Nm)': str(blade_peak * 0.9),
            })
        with (path / 'bladed_inference_steps.csv').open('w', encoding='utf-8', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=INFERENCE_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _write_pid(path: Path) -> None:
        rows = []
        for index in range(3):
            rows.append({
                'Time': f'{index * 0.02:.2f}',
                'I_AverageWindSpeed': '16.0',
                'Electrical power': '10000000',
                'O_PcsTorqueDemand': '20000000',
                'O_PitchPositionDemand1': '0.1',
                'Blade root 1 My (Nm)': '10000000',
                'Blade root 2 My (Nm)': '9500000',
                'Blade root 3 My (Nm)': '9000000',
            })
        with path.open('w', encoding='utf-8', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=PID_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    def test_builds_round_summary_and_plots(self) -> None:
        with tempfile.TemporaryDirectory(prefix='rl_round_progress_') as temporary_directory:
            root = Path(temporary_directory)
            round_one = root / 'round1'
            round_two = root / 'round2'
            round_one.mkdir()
            round_two.mkdir()
            self._write_inference(round_one, power=9_900_000.0, blade_peak=10_500_000.0)
            self._write_inference(round_two, power=10_100_000.0, blade_peak=10_300_000.0)
            pid_csv = root / 'pid.csv'
            self._write_pid(pid_csv)

            summary = build_round_progress(
                round_dirs={1: round_one, 2: round_two},
                pid_csv=pid_csv,
                output_dir=root / 'output',
            )

            self.assertEqual(summary['mode'], 'round_progress_v1')
            self.assertEqual([item['round'] for item in summary['rounds']], [1, 2])
            self.assertLess(summary['rounds'][0]['power_delta_percent'], 0.0)
            self.assertGreater(summary['rounds'][1]['power_delta_percent'], 0.0)
            self.assertLess(summary['rounds'][0]['load_reduction_percent'], 0.0)
            self.assertAlmostEqual(summary['rounds'][0]['rl_mean_blade_peak_mnm'], 9.975)
            self.assertAlmostEqual(summary['rounds'][0]['pid_mean_blade_peak_mnm'], 9.5)
            self.assertTrue((root / 'output/round_progress_power_load.png').is_file())
            self.assertTrue((root / 'output/round_progress_relative_to_pid.png').is_file())
            self.assertTrue((root / 'output/round_progress_blade_root_peaks.png').is_file())
            self.assertTrue((root / 'output/round_progress_summary.csv').is_file())
            self.assertTrue((root / 'output/round_progress_summary.json').is_file())


if __name__ == '__main__':
    unittest.main()
