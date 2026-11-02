from __future__ import annotations

import unittest

import numpy as np

from RL.utils.dual_constraint import (
    DUAL_CONSTRAINT_SCHEMA_VERSION,
    ConditionConstraintAccumulator,
    DualConstraintConfig,
    DualConstraintController,
)
from RL.train_rl import _finalize_queue_condition_constraints


class TestDualConstraintController(unittest.TestCase):
    @staticmethod
    def _config() -> DualConstraintConfig:
        return DualConstraintConfig.from_reward_config({
            'dual_constraint': {
                'schema_version': DUAL_CONSTRAINT_SCHEMA_VERSION,
                'eta': {
                    'load_blade_1': 2.0,
                    'load_blade_2': 2.0,
                    'load_blade_3': 2.0,
                    'energy': 1.0,
                },
                'lambda_max': {
                    'load_blade_1': 1.0,
                    'load_blade_2': 1.0,
                    'load_blade_3': 1.0,
                    'energy': 1.0,
                },
                'epsilon_load_per_blade': [0.0, 0.0, 0.0],
                'epsilon_energy_percent': 0.0,
                'initial_lambda': {
                    'load_blade_1': 0.0,
                    'load_blade_2': 0.0,
                    'load_blade_3': 0.0,
                    'energy': 0.0,
                },
                'aggregation': 'mean_complete_conditions_at_round_boundary',
            }
        })

    def test_complete_condition_uses_per_blade_peaks_and_energy_deficit(self) -> None:
        config = self._config()
        accumulator = ConditionConstraintAccumulator()
        accumulator.observe(
            pid_power_w=100.0,
            rl_power_w=90.0,
            blade_root_my_nm=np.asarray([100.0, 95.0, 80.0]),
            delta_s=1.0,
            pid_peak_abs_nm=np.asarray([100.0, 100.0, 100.0]),
        )
        accumulator.observe(
            pid_power_w=100.0,
            rl_power_w=90.0,
            blade_root_my_nm=np.asarray([90.0, 105.0, 80.0]),
            delta_s=1.0,
            pid_peak_abs_nm=np.asarray([100.0, 100.0, 100.0]),
        )
        metrics = accumulator.finalize(config)

        self.assertEqual(metrics['condition_delta_energy_j'], -20.0)
        self.assertEqual(metrics['condition_g_load_blade_1'], 0.0)
        self.assertAlmostEqual(metrics['condition_g_load_blade_2'], 0.05)
        self.assertEqual(metrics['condition_g_load_blade_3'], 0.0)
        self.assertEqual(metrics['condition_g_energy'], 10.0)

    def test_updates_only_at_round_boundary_using_complete_condition_mean(self) -> None:
        controller = DualConstraintController(self._config())
        controller.complete_condition({
            'condition_g_load_blade_1': 0.1,
            'condition_g_load_blade_2': 0.0,
            'condition_g_load_blade_3': 0.0,
            'condition_g_energy': 0.2,
        })
        controller.complete_condition({
            'condition_g_load_blade_1': 0.3,
            'condition_g_load_blade_2': 0.1,
            'condition_g_load_blade_3': 0.0,
            'condition_g_energy': 0.4,
        })
        metrics = controller.update_at_round_boundary()

        self.assertEqual(metrics['dual_constraint_complete_conditions'], 2.0)
        self.assertAlmostEqual(metrics['dual_mean_g_load_blade_1'], 0.2)
        self.assertAlmostEqual(metrics['dual_lambda_load_blade_1'], 0.4)
        self.assertAlmostEqual(metrics['dual_lambda_load_blade_2'], 0.1)
        self.assertAlmostEqual(metrics['dual_lambda_energy'], 0.3)

    def test_skipped_or_retried_condition_cannot_leak_into_next_condition(self) -> None:
        config = self._config()
        for failure_path in ('skip', 'retry'):
            with self.subTest(failure_path=failure_path):
                controller = DualConstraintController(config)
                failed = ConditionConstraintAccumulator()
                failed.observe(
                    pid_power_w=100.0,
                    rl_power_w=0.0,
                    blade_root_my_nm=np.asarray([200.0, 200.0, 200.0]),
                    delta_s=1.0,
                    pid_peak_abs_nm=np.asarray([100.0, 100.0, 100.0]),
                )
                fresh, incomplete_metrics = _finalize_queue_condition_constraints(
                    failed,
                    controller,
                    config,
                    complete=False,
                )

                self.assertIsNot(fresh, failed)
                self.assertEqual(incomplete_metrics, {'dual_constraint_condition_complete': 0.0})
                self.assertEqual(controller.update_at_round_boundary()['dual_constraint_updated'], 0.0)

                fresh.observe(
                    pid_power_w=100.0,
                    rl_power_w=100.0,
                    blade_root_my_nm=np.asarray([90.0, 95.0, 80.0]),
                    delta_s=1.0,
                    pid_peak_abs_nm=np.asarray([100.0, 100.0, 100.0]),
                )
                _, complete_metrics = _finalize_queue_condition_constraints(
                    fresh,
                    controller,
                    config,
                    complete=True,
                )
                round_metrics = controller.update_at_round_boundary()

                self.assertEqual(complete_metrics['condition_pid_energy_j'], 100.0)
                self.assertEqual(complete_metrics['condition_rl_energy_j'], 100.0)
                self.assertEqual(complete_metrics['condition_g_load_blade_1'], 0.0)
                self.assertEqual(round_metrics['dual_constraint_complete_conditions'], 1.0)
                self.assertEqual(round_metrics['dual_mean_g_energy'], 0.0)


if __name__ == '__main__':
    unittest.main()