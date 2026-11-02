from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

from RL.infer.analyze_edge_performance import (
    _resolve_output_path,
    analyze_edge_performance,
    write_edge_performance_report,
)


class TestEdgePerformance(unittest.TestCase):
    def _write_records(self, path: Path, sample_count: int, latency_s: float = 0.005) -> None:
        with path.open('w', encoding='utf-8', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['record_type', 't_rx_monotonic_s', 't_tx_monotonic_s'])
            writer.writeheader()
            for index in range(sample_count):
                receive_time = index * 0.02
                writer.writerow(
                    {
                        'record_type': 'control_step',
                        't_rx_monotonic_s': receive_time,
                        't_tx_monotonic_s': receive_time + latency_s,
                    }
                )

    def test_t1_passes_with_5000_measured_receive_to_send_samples(self) -> None:
        with tempfile.TemporaryDirectory(prefix='edge_performance_') as tmpdir:
            records_path = Path(tmpdir) / 'bladed_inference_steps.csv'
            report_path = Path(tmpdir) / 'edge_performance_summary.json'
            self._write_records(records_path, sample_count=5_000)

            report = write_edge_performance_report(records_path, report_path)
            self.assertTrue(report_path.exists())

        self.assertTrue(report['valid'])
        self.assertEqual(report['valid_sample_count'], 5_000)
        self.assertAlmostEqual(report['latency_ms']['p99'], 5.0)
        self.assertGreaterEqual(report['output_frequency_hz'], 50.0)
        self.assertTrue(report['T1_pass'])

    def test_t1_rejects_insufficient_sample_count(self) -> None:
        with tempfile.TemporaryDirectory(prefix='edge_performance_') as tmpdir:
            records_path = Path(tmpdir) / 'bladed_inference_steps.csv'
            self._write_records(records_path, sample_count=3)

            report = analyze_edge_performance(records_path)

        self.assertTrue(report['valid'])
        self.assertFalse(report['sample_count_pass'])
        self.assertFalse(report['T1_pass'])
        self.assertIn('5000', report['reason'])

    def test_explicit_output_path_does_not_need_to_exist(self) -> None:
        with tempfile.TemporaryDirectory(prefix='edge_performance_') as tmpdir:
            output_path = _resolve_output_path(str(Path(tmpdir) / 'new_report.json'))

            self.assertEqual(output_path, (Path(tmpdir) / 'new_report.json').resolve())
            self.assertFalse(output_path.exists())


if __name__ == '__main__':
    unittest.main()