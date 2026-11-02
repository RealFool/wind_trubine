from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from RL.benchmark_compact_logger import benchmark_compact_logger


class TestCompactLoggerBenchmark(unittest.TestCase):
    def test_small_synthetic_run_closes_slots_and_event_budgets(self) -> None:
        with tempfile.TemporaryDirectory(prefix='compact_logger_benchmark_test_') as temporary_directory:
            report = benchmark_compact_logger(
                Path(temporary_directory),
                rounds=2,
                conditions_per_round=2,
                steps_per_condition=100,
                event_pre_frames=5,
                event_post_frames=5,
                event_capacity_per_condition=16,
            )

        self.assertEqual(report['slot_count'], 4)
        self.assertEqual(report['step_rows_emitted'], 400)
        self.assertEqual(report['condition_rows_seen'], 4)
        self.assertTrue(report['pass'])
        self.assertLessEqual(report['total_artifact_bytes'], report['total_artifact_bytes_budget'])
        self.assertLessEqual(report['python_heap_peak_bytes'], report['python_heap_peak_budget_bytes'])
        self.assertLessEqual(report['event_windows_written'], 4 * 16)


if __name__ == '__main__':
    unittest.main()
