from __future__ import annotations

import unittest

import numpy as np

from RL.analyze_frozen_round4_action_my_lag import (
    MY_FIELDS,
    _correlation_rows,
    _regression_rows,
    _standardized_ols,
)


class FrozenRound4ActionMyLagTests(unittest.TestCase):
    def test_one_frame_action_precedes_corresponding_my_increment(self):
        rng = np.random.default_rng(20261020)
        sample_count = 2_000
        torque = rng.normal(size=sample_count).cumsum()
        torque_delta = np.diff(torque, prepend=torque[0])
        my_magnitude = np.full(sample_count, 50_000_000.0, dtype=np.float64)
        noise = rng.normal(scale=100.0, size=sample_count - 1)
        for index in range(1, sample_count):
            action_index = index - 1
            response = noise[action_index]
            if action_index % 2 == 0:
                response += 1_000.0 * torque_delta[action_index]
            my_magnitude[index] = my_magnitude[index - 1] + response

        fallback_active = np.zeros(sample_count, dtype=np.float64)
        fallback_active[:10] = 1.0
        columns = {
            'Time': np.arange(sample_count, dtype=np.float64) * 0.02,
            'I_RotorAzimuth': np.zeros(sample_count, dtype=np.float64),
            'I_RotorSpeed': np.ones(sample_count, dtype=np.float64),
            'I_AverageWindSpeed': np.full(sample_count, 8.0, dtype=np.float64),
            MY_FIELDS[0]: my_magnitude,
            MY_FIELDS[1]: my_magnitude * 0.8,
            MY_FIELDS[2]: my_magnitude * 0.6,
            'sent_action_torque': torque,
            'sent_action_pitch': rng.normal(size=sample_count),
            'policy_delta_torque': rng.normal(size=sample_count),
            'policy_delta_pitch': rng.normal(size=sample_count),
            'baseline_action_torque': np.zeros(sample_count, dtype=np.float64),
            'baseline_action_pitch': np.zeros(sample_count, dtype=np.float64),
            'info_control_fallback_active': fallback_active,
        }
        slot = {
            'metadata': {'condition_id': 8},
            'columns': columns,
            'valid': np.ones(sample_count, dtype=bool),
            'valid_bad_prefix': np.zeros(sample_count + 1, dtype=np.int64),
            # Alternate low/high load-origin labels to detect an off-by-one
            # classification of the action change.
            'max_abs_my_nm_by_frame': np.where(
                np.arange(sample_count) % 2 == 0, 30_000_000.0, 50_000_000.0),
            'low_load_threshold_nm': 40_000_000.0,
        }

        rows = _correlation_rows({1: slot})
        paired = next(row for row in rows if
                      row['grouping'] == 'per_wind_slot' and row['stratum'] == 'all_valid' and
                      row['blade'] == 1 and row['lag_frames'] == 1 and
                      row['signal'] == 'sent_torque_nm' and
                      row['metric'] == 'action_step_vs_abs_my_change')
        two_frames = next(row for row in rows if
                  row['grouping'] == 'per_wind_slot' and row['stratum'] == 'all_valid' and
                  row['blade'] == 1 and row['lag_frames'] == 2 and
                  row['signal'] == 'sent_torque_nm' and
                  row['metric'] == 'action_step_vs_abs_my_change')
        self.assertEqual(paired['lag_semantics'], 'action_time_precedes_measurement')
        contemporaneous = next(row for row in rows if
                               row['grouping'] == 'per_wind_slot' and row['stratum'] == 'all_valid' and
                               row['blade'] == 1 and row['lag_frames'] == 0 and
                               row['signal'] == 'sent_torque_nm' and
                               row['metric'] == 'action_level_vs_future_abs_my')
        self.assertEqual(contemporaneous['lag_semantics'], 'contemporaneous_noncausal_baseline')
        self.assertGreater(paired['r'], 0.60)
        self.assertLess(abs(two_frames['r']), 0.15)
        low_load_paired = next(row for row in rows if
                       row['grouping'] == 'per_wind_slot' and
                       row['stratum'] == 'relative_low_load_q25' and
                       row['blade'] == 1 and row['lag_frames'] == 1 and
                       row['signal'] == 'sent_torque_nm' and
                       row['metric'] == 'action_step_vs_abs_my_change')
        self.assertEqual(low_load_paired['n'], 999)
        self.assertGreater(low_load_paired['r'], 0.95)
        actor_level = next(row for row in rows if
                   row['grouping'] == 'per_wind_slot' and
                   row['stratum'] == 'all_valid' and row['blade'] == 1 and
                   row['lag_frames'] == 1 and row['signal'] == 'actor_delta_torque_nm' and
                   row['metric'] == 'action_level_vs_future_abs_my')
        actor_change = next(row for row in rows if
                    row['grouping'] == 'per_wind_slot' and
                    row['stratum'] == 'all_valid' and row['blade'] == 1 and
                    row['lag_frames'] == 1 and row['signal'] == 'actor_delta_torque_nm' and
                    row['metric'] == 'action_step_vs_abs_my_change')
        sent_change = next(row for row in rows if
                   row['grouping'] == 'per_wind_slot' and
                   row['stratum'] == 'all_valid' and row['blade'] == 1 and
                   row['lag_frames'] == 1 and row['signal'] == 'sent_torque_nm' and
                   row['metric'] == 'action_step_vs_abs_my_change')
        self.assertEqual(actor_level['n'], sample_count - 11)
        self.assertEqual(actor_change['n'], sample_count - 12)
        self.assertEqual(sent_change['n'], sample_count - 2)

        regression_rows = _regression_rows({1: slot})
        actor_regression = next(row for row in regression_rows if
                    row['stratum'] == 'all_valid' and row['blade'] == 1 and
                    row['lag_frames'] == 1 and
                    row['action_family'] == 'actor_policy_residual_change')
        sent_regression = next(row for row in regression_rows if
                       row['stratum'] == 'all_valid' and row['blade'] == 1 and
                       row['lag_frames'] == 1 and
                       row['action_family'] == 'sent_action_change')
        self.assertEqual(actor_regression['n'], sample_count - 12)
        self.assertEqual(sent_regression['n'], sample_count - 2)
        self.assertFalse(any(
            row['grouping'] == 'per_wind_slot' and row['stratum'] == 'all_valid' and
            row['blade'] == 1 and row['lag_frames'] == 0 and
            row['signal'] == 'sent_torque_nm' and
            row['metric'] == 'action_step_vs_abs_my_change'
            for row in rows
        ))

    def test_standardized_ols_recovers_signed_associations(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(5_000, 3))
        y = 0.7 * x[:, 0] - 0.4 * x[:, 1] + 0.1 * x[:, 2] + rng.normal(scale=0.15, size=5_000)

        beta, r2, rank, condition = _standardized_ols(y, x)

        expected_beta = np.asarray([0.7, -0.4, 0.1]) * np.std(x, axis=0) / np.std(y)
        np.testing.assert_allclose(beta, expected_beta, atol=0.02, rtol=0.0)
        self.assertIsNotNone(r2)
        self.assertGreater(float(r2), 0.9)
        self.assertEqual(rank, 3)
        self.assertIsNotNone(condition)


if __name__ == '__main__':
    unittest.main()