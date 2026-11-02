from __future__ import annotations

import unittest

from RL.benchmark_condition_replay import benchmark_condition_replay


class TestConditionReplayBenchmark(unittest.TestCase):
    def test_synthetic_no_udp_benchmark_closes_counts_and_enforces_source_quota(self) -> None:
        report = benchmark_condition_replay(
            condition_ids=(3, 4, 5),
            capacity_per_condition=32,
            transitions_per_condition=32,
            batch_size=8,
            sample_batches=12,
            seed=3,
        )

        self.assertEqual(report['mode'], 'synthetic_no_udp')
        self.assertEqual(report['stored_transition_count'], 96)
        self.assertEqual(report['sampled_transition_count'], 96)
        self.assertTrue(report['batch_count_closes'])
        self.assertTrue(report['quota_pass'])
        self.assertLessEqual(report['recovery_sample_fraction'], 0.25)
        self.assertGreater(report['replay_numpy_array_bytes'], 0)
        self.assertGreaterEqual(report['batch_latency_ms']['p99'], 0.0)
        self.assertGreater(report['process_rss_after_fill_bytes'], 0)


if __name__ == '__main__':
    unittest.main()
