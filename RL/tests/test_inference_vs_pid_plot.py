from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np

from RL.infer.plot_inference_vs_pid import (
    InferenceTrace,
    Trace,
    _format_average_power_annotation,
    build_comparison,
    build_formal_acceptance_report,
    build_multi_scenario_acceptance_report,
)


REQUIRED_FIELDS = [
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


class TestInferenceVsPidPlot(unittest.TestCase):
    @staticmethod
    def _inference_trace(raw_time_s: list[float], power_w: list[float], peaks_nm: list[list[float]]) -> InferenceTrace:
        raw_time = np.asarray(raw_time_s, dtype=np.float64)
        trace = Trace(
            source_rows=[],
            raw_time_s=raw_time,
            elapsed_time_s=raw_time - raw_time[0],
            wind_speed_mps=np.full(len(raw_time), 12.0, dtype=np.float64),
            power_w=np.asarray(power_w, dtype=np.float64),
            blade_root_my_abs_nm=np.asarray(peaks_nm, dtype=np.float64),
        )
        return InferenceTrace(
            trace=trace,
            episodes=np.ones(len(raw_time), dtype=np.int64),
            global_steps=np.arange(1, len(raw_time) + 1, dtype=np.int64),
            fallback_active=np.zeros(len(raw_time), dtype=np.float64),
            ood_score=np.zeros(len(raw_time), dtype=np.float64),
            sent_action_torque_nm=np.zeros(len(raw_time), dtype=np.float64),
            sent_action_pitch_rad=np.zeros(len(raw_time), dtype=np.float64),
        )

    @staticmethod
    def _pid_trace(raw_time_s: list[float], power_w: list[float], peaks_nm: list[list[float]]) -> Trace:
        raw_time = np.asarray(raw_time_s, dtype=np.float64)
        return Trace(
            source_rows=[],
            raw_time_s=raw_time,
            elapsed_time_s=raw_time - raw_time[0],
            wind_speed_mps=np.full(len(raw_time), 12.0, dtype=np.float64),
            power_w=np.asarray(power_w, dtype=np.float64),
            blade_root_my_abs_nm=np.asarray(peaks_nm, dtype=np.float64),
        )

    def test_formats_average_power_increase_and_decrease(self) -> None:
        increase_text, increase_color = _format_average_power_annotation(4_800_000.0, 4_000_000.0)
        decrease_text, decrease_color = _format_average_power_annotation(3_800_000.0, 4_000_000.0)

        self.assertIn('Increase: 800.0 kW (20.00%)', increase_text)
        self.assertEqual(increase_color, '#047857')
        self.assertIn('Decrease: 200.0 kW (5.00%)', decrease_text)
        self.assertEqual(decrease_color, '#b91c1c')

    def test_formal_acceptance_uses_energy_and_average_blade_peak_reduction(self) -> None:
        model = self._inference_trace(
            [10.0, 10.02, 10.04],
            [10_000_000.0, 12_000_000.0, 12_000_000.0],
            [[9_000_000.0, 9_200_000.0, 10_500_000.0], [8_000_000.0, 9_500_000.0, 8_500_000.0], [8_500_000.0, 9_100_000.0, 8_000_000.0]],
        )
        pid = self._pid_trace(
            [3.0, 3.02, 3.04],
            [10_000_000.0, 10_000_000.0, 10_000_000.0],
            [[10_000_000.0, 10_000_000.0, 10_000_000.0], [9_000_000.0, 9_000_000.0, 9_000_000.0], [8_000_000.0, 8_000_000.0, 8_000_000.0]],
        )

        report = build_formal_acceptance_report(model, pid)

        self.assertTrue(report['valid'])
        self.assertAlmostEqual(report['energy']['delta_energy_percent'], 15.0)
        self.assertAlmostEqual(report['blade_root_my']['channels'][1]['delta_load_percent'], 5.0)
        self.assertAlmostEqual(report['blade_root_my']['channels'][2]['delta_load_percent'], -5.0)
        self.assertAlmostEqual(report['blade_root_my']['average_delta_load_percent'], 10.0 / 3.0)
        self.assertTrue(report['P0'])
        self.assertTrue(report['P1'])
        self.assertTrue(report['L0'])
        self.assertTrue(report['L1'])
        self.assertTrue(report['path_1_pass'])
        self.assertTrue(report['path_2_pass'])
        self.assertTrue(report['final_pass'])
        self.assertTrue(report['official_acceptance_pass'])
        self.assertFalse(report['strict_per_blade_pass'])
        self.assertEqual(report['strict_per_blade_comparison_tolerance_nm'], 0.0)

    def test_multi_scenario_acceptance_requires_every_expected_slot_and_both_gates(self) -> None:
        pid = self._pid_trace(
            [0.0, 0.02, 0.04],
            [10_000_000.0, 10_000_000.0, 10_000_000.0],
            [[100.0, 100.0, 100.0]] * 3,
        )
        strict_good = self._inference_trace(
            [1.0, 1.02, 1.04],
            [10_200_000.0, 10_200_000.0, 10_200_000.0],
            [[90.0, 90.0, 90.0]] * 3,
        )
        strict_bad = self._inference_trace(
            [2.0, 2.02, 2.04],
            [10_200_000.0, 10_200_000.0, 10_200_000.0],
            [[90.0, 90.0, 101.0]] * 3,
        )
        first_report = build_formal_acceptance_report(strict_good, pid)
        second_report = build_formal_acceptance_report(strict_bad, pid)

        missing = build_multi_scenario_acceptance_report(
            [{'scenario_id': 'case-3', 'report': first_report, 'strict_case_matched': True}],
            expected_scenario_ids=['case-3', 'case-4'],
        )
        self.assertFalse(missing['all_expected_scenarios_valid'])
        self.assertFalse(missing['official_acceptance_pass'])
        self.assertFalse(missing['strict_per_blade_pass'])
        self.assertFalse(missing['release_candidate_pass'])

        complete = build_multi_scenario_acceptance_report(
            [
                {'scenario_id': 'case-3', 'report': first_report, 'strict_case_matched': True},
                {'scenario_id': 'case-4', 'report': second_report, 'strict_case_matched': True},
            ],
            expected_scenario_ids=['case-3', 'case-4'],
        )
        self.assertTrue(complete['all_expected_scenarios_valid'])
        self.assertTrue(complete['official_acceptance_pass'])
        self.assertFalse(complete['strict_per_blade_pass'])
        self.assertTrue(complete['strict_case_identity_pass'])
        self.assertFalse(complete['release_candidate_pass'])

        identity_missing = build_multi_scenario_acceptance_report(
            [
                {'scenario_id': 'case-3', 'report': first_report, 'strict_case_matched': True},
                {'scenario_id': 'case-4', 'report': first_report, 'strict_case_matched': False},
            ],
            expected_scenario_ids=['case-3', 'case-4'],
        )
        self.assertFalse(identity_missing['strict_case_identity_pass'])
        self.assertFalse(identity_missing['release_candidate_pass'])

    def test_formal_acceptance_rejects_nonmonotonic_simulation_time(self) -> None:
        model = self._inference_trace(
            [0.0, 0.02, 0.01],
            [10_000_000.0, 10_000_000.0, 10_000_000.0],
            [[10_000_000.0, 10_000_000.0, 10_000_000.0]] * 3,
        )
        pid = self._pid_trace(
            [0.0, 0.02, 0.04],
            [10_000_000.0, 10_000_000.0, 10_000_000.0],
            [[10_000_000.0, 10_000_000.0, 10_000_000.0]] * 3,
        )

        report = build_formal_acceptance_report(model, pid)

        self.assertFalse(report['valid'])
        self.assertIn('strictly increasing', report['reason'])

    def test_matches_control_steps_across_episode_time_reset(self) -> None:
        with tempfile.TemporaryDirectory(prefix='rl_infer_plot_') as tmpdir:
            root = Path(tmpdir)
            inference_dir = root / 'inference'
            inference_dir.mkdir()
            pid_csv = root / 'pid.csv'
            output_dir = root / 'new_output_directory'

            inference_rows = [
                {
                    'record_type': 'control_step',
                    'inference_episode': str(1 if index < 2 else 2),
                    'inference_global_control_step': str(index + 1),
                    'info_control_fallback_active': '1' if index < 2 else '0',
                    'info_control_ood_score': str(10 - index),
                    'sent_action_torque': str(20_000_000 + index * 100),
                    'sent_action_pitch': str(0.01 + index * 0.001),
                    'Time': str((index % 2 + 1) * 0.02),
                    'I_AverageWindSpeed': str(12.0 + index * 0.1),
                    'Electrical power': str(15_000_000 + index * 100_000),
                    'Blade root 1 My (Nm)': str(45_000_000 + index * 10_000),
                    'Blade root 2 My (Nm)': str(45_100_000 + index * 10_000),
                    'Blade root 3 My (Nm)': str(45_200_000 + index * 10_000),
                }
                for index in range(4)
            ]
            inference_rows.append(
                {
                    'record_type': 'control_step',
                    'inference_episode': '1',
                    'inference_global_control_step': '5',
                    'info_control_fallback_active': '0',
                    'info_control_ood_score': '0',
                    'sent_action_torque': '',
                    'sent_action_pitch': '',
                    'Time': '0.10',
                    'I_AverageWindSpeed': '12.4',
                    'Electrical power': '15_000_000',
                    'Blade root 1 My (Nm)': '45_000_000',
                    'Blade root 2 My (Nm)': '45_100_000',
                    'Blade root 3 My (Nm)': '45_200_000',
                }
            )
            with (inference_dir / 'bladed_inference_steps.csv').open('w', encoding='utf-8', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=REQUIRED_FIELDS)
                writer.writeheader()
                writer.writerows(inference_rows)

            pid_rows = [
                {
                    'Time': str((index + 1) * 0.02),
                    'I_AverageWindSpeed': str(11.8 + index * 0.1),
                    'Electrical power': str(14_900_000 + index * 100_000),
                    'O_PcsTorqueDemand': str(20_000_000 + index * 100),
                    'O_PitchPositionDemand1': str(0.009 + index * 0.001),
                    'Blade root 1 My (Nm)': str(44_900_000 + index * 10_000),
                    'Blade root 2 My (Nm)': str(45_000_000 + index * 10_000),
                    'Blade root 3 My (Nm)': str(45_100_000 + index * 10_000),
                }
                for index in range(6)
            ]
            with pid_csv.open('w', encoding='utf-8', newline='') as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=[
                        'Time',
                        'I_AverageWindSpeed',
                        'Electrical power',
                        'O_PcsTorqueDemand',
                        'O_PitchPositionDemand1',
                        'Blade root 1 My (Nm)',
                        'Blade root 2 My (Nm)',
                        'Blade root 3 My (Nm)',
                    ],
                )
                writer.writeheader()
                writer.writerows(pid_rows)

            summary = build_comparison(
                inference_dir=inference_dir,
                pid_csv=pid_csv,
                output_dir=output_dir,
                pid_start_row=1,
            )

            self.assertEqual(summary['matched_control_steps'], 4)
            self.assertEqual(summary['ignored_trailing_control_rows'], 1)
            self.assertAlmostEqual(summary['model_elapsed_duration_s'], 0.06)
            self.assertAlmostEqual(summary['pid_elapsed_duration_s'], 0.06)
            self.assertEqual(summary['fallback_step_count'], 2)
            self.assertTrue((output_dir / 'inference_pid_matched_steps.csv').is_file())
            self.assertTrue((output_dir / 'inference_vs_pid_power_step.png').is_file())
            self.assertTrue((output_dir / 'inference_vs_pid_wind_speed.png').is_file())
            self.assertTrue((output_dir / 'inference_vs_pid_cumulative_energy.png').is_file())
            self.assertTrue((output_dir / 'inference_vs_pid_blade_root_my.png').is_file())
            self.assertTrue((output_dir / 'inference_vs_pid_actions.png').is_file())

            with (output_dir / 'inference_pid_matched_steps.csv').open('r', encoding='utf-8', newline='') as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 4)
            self.assertEqual(rows[0]['pid_power_w'], '15000000.0')
            self.assertEqual(rows[0]['model_wind_speed_mps'], '12.0')
            self.assertEqual(rows[0]['pid_wind_speed_mps'], '11.9')
            self.assertEqual(rows[0]['model_sent_action_torque_nm'], '20000000.0')
            self.assertEqual(rows[0]['pid_torque_demand_nm'], '20000100.0')
            self.assertEqual(rows[-1]['model_episode'], '2')

            hidden_fallback_output_dir = root / 'no_fallback_output_directory'
            no_fallback_summary = build_comparison(
                inference_dir=inference_dir,
                pid_csv=pid_csv,
                output_dir=hidden_fallback_output_dir,
                pid_start_row=1,
                show_fallback=False,
            )
            self.assertFalse(no_fallback_summary['show_fallback_markers'])
            self.assertTrue((hidden_fallback_output_dir / 'inference_vs_pid_power_step.png').is_file())
            self.assertTrue((hidden_fallback_output_dir / 'inference_vs_pid_wind_speed.png').is_file())
            self.assertTrue((hidden_fallback_output_dir / 'inference_vs_pid_blade_root_my.png').is_file())
            self.assertTrue((hidden_fallback_output_dir / 'inference_vs_pid_actions.png').is_file())

            formal_output_dir = root / 'formal_output_directory'
            formal_summary = build_comparison(
                inference_dir=inference_dir,
                pid_csv=pid_csv,
                output_dir=formal_output_dir,
                pid_start_row=1,
                formal_acceptance=True,
            )
            self.assertFalse(formal_summary['formal_acceptance']['valid'])
            self.assertIn('continuous RL episode', formal_summary['formal_acceptance']['reason'])
            self.assertTrue((formal_output_dir / 'formal_acceptance_report.json').is_file())


if __name__ == '__main__':
    unittest.main()