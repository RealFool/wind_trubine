from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

from RL.utils.queue_plotting import aggregate_queue_q_health_metrics, aggregate_queue_step_metrics


class TestQueuePlotting(unittest.TestCase):
    def test_streams_selected_metrics_into_step_bins(self) -> None:
        with tempfile.TemporaryDirectory(prefix='queue_plotting_') as temporary_directory:
            root = Path(temporary_directory)
            source = root / 'queue_train_step_metrics.csv'
            rows = [
                ('queue_train_step', 1, 'wind_t', 10.0),
                ('queue_train_step', 1, 'power_state_real', 1_000_000.0),
                ('queue_train_step', 1, 'control_fallback_active', 0.0),
                ('queue_train_step', 2, 'wind_t', 12.0),
                ('queue_train_step', 2, 'power_state_real', 3_000_000.0),
                ('queue_train_step', 2, 'control_fallback_active', 1.0),
                ('queue_train_step', 3, 'wind_t', 14.0),
                ('queue_train_step', 3, 'power_state_real', 5_000_000.0),
                ('queue_train_step', 3, 'control_fallback_active', 0.0),
                ('queue_train_step', 1, 'unselected_metric', 999.0),
            ]
            with source.open('w', encoding='utf-8', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['stage', 'step', 'name', 'value'])
                writer.writerows(rows)

            aggregated = aggregate_queue_step_metrics(source, root, bin_steps=2)

            self.assertEqual(len(aggregated), 2)
            first = aggregated[0]
            self.assertEqual(first['sample_count'], 2.0)
            self.assertAlmostEqual(first['wind_t_mean'], 11.0)
            self.assertAlmostEqual(first['power_state_real_mean'], 2_000_000.0)
            self.assertAlmostEqual(first['control_fallback_active_mean'], 0.5)
            self.assertNotIn('unselected_metric_mean', first)
            self.assertTrue((root / 'queue_train_step_aggregated.csv').is_file())
            self.assertTrue((root / 'queue_train_step_aggregated_meta.json').is_file())

            cached = aggregate_queue_step_metrics(source, root, bin_steps=2)
            self.assertEqual(cached, aggregated)

    def test_aggregates_q_health_guard_and_effective_update_rates(self) -> None:
        with tempfile.TemporaryDirectory(prefix='queue_q_health_') as temporary_directory:
            root = Path(temporary_directory)
            source = root / 'queue_train_update_metrics.csv'
            rows = [
                ('queue_train_update', 1, 'queue_stage_code', 2.0),
                ('queue_train_update', 1, 'critic1_loss', 1.0),
                ('queue_train_update', 1, 'target_q_abs_mean', 10.0),
                ('queue_train_update', 1, 'actor_update_deferred', 1.0),
                ('queue_train_update', 2, 'queue_stage_code', 2.0),
                ('queue_train_update', 2, 'critic1_loss', 0.5),
                ('queue_train_update', 2, 'target_q_abs_mean', 20.0),
                ('queue_train_update', 2, 'actor_update_deferred', 1.0),
                ('queue_train_update', 3, 'queue_stage_code', 3.0),
                ('queue_train_update', 3, 'critic1_loss', 0.4),
                ('queue_train_update', 3, 'target_q_abs_mean', 121.0),
                ('queue_train_update', 3, 'update_skipped_q_guard', 1.0),
                ('queue_train_update', 4, 'queue_stage_code', 3.0),
                ('queue_train_update', 4, 'critic1_loss', 0.3),
                ('queue_train_update', 4, 'target_q_abs_mean', 122.0),
                ('queue_train_update', 4, 'actor_rl_loss', -2.0),
                ('queue_train_update', 4, 'actor_update_skipped_q_guard', 1.0),
                ('queue_train_update', 5, 'queue_stage_code', 3.0),
                ('queue_train_update', 5, 'target_q_abs_mean', 123.0),
                ('queue_train_update', 5, 'actor_rl_loss', -1.5),
                ('queue_train_update', 5, 'actor_loss', -0.7),
            ]
            with source.open('w', encoding='utf-8', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['stage', 'step', 'name', 'value'])
                writer.writerows(rows)

            aggregated = aggregate_queue_q_health_metrics(source, root, bin_steps=4)

            self.assertEqual(len(aggregated), 2)
            first = aggregated[0]
            self.assertEqual(first['update_opportunity_count'], 4.0)
            self.assertEqual(first['critic_stage_opportunity_count'], 2.0)
            self.assertEqual(first['actor_stage_opportunity_count'], 2.0)
            self.assertEqual(first['critic_update_count'], 4.0)
            self.assertEqual(first['critic_q_guard_count'], 1.0)
            self.assertEqual(first['actor_attempt_count'], 1.0)
            self.assertEqual(first['actor_update_count'], 0.0)
            self.assertAlmostEqual(first['critic_update_rate'], 1.0)
            self.assertAlmostEqual(first['critic_q_guard_rate'], 0.25)
            self.assertAlmostEqual(first['actor_update_rate'], 0.0)
            self.assertAlmostEqual(first['actor_q_guard_rate'], 1.0)
            self.assertAlmostEqual(first['actor_deferred_rate'], 1.0)
            self.assertAlmostEqual(first['target_q_abs_mean_mean'], (10.0 + 20.0 + 121.0 + 122.0) / 4.0)
            self.assertTrue((root / 'queue_q_health_aggregated.csv').is_file())
            self.assertTrue((root / 'queue_q_health_aggregated_meta.json').is_file())


if __name__ == '__main__':
    unittest.main()