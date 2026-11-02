from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from RL.report_post_run import render_markdown, summarize


def _record(condition: int, *, rl: float, pid: float, deltas: tuple[float, float, float]) -> dict:
    return {
        'condition_id': float(condition),
        'condition_instance_id': float(condition),
        'queue_condition_round_index': float(condition),
        'queue_condition_trapezoid_energy_valid': 1.0,
        'queue_condition_rl_trapezoid_energy_j': rl,
        'queue_condition_pid_trapezoid_energy_j': pid,
        'queue_condition_trapezoid_delta_energy_percent': (rl / pid - 1.0) * 100.0,
        'dual_constraint_condition_complete': 1.0,
        'queue_case_identity_full_trace_complete': 1.0,
        'queue_condition_blade_1_delta_load_percent': deltas[0],
        'queue_condition_blade_2_delta_load_percent': deltas[1],
        'queue_condition_blade_3_delta_load_percent': deltas[2],
        'queue_condition_blade_1_peak_ratio_to_pid': 1.0 + deltas[0] / 100.0,
        'queue_condition_blade_2_peak_ratio_to_pid': 1.0 + deltas[1] / 100.0,
        'queue_condition_blade_3_peak_ratio_to_pid': 1.0 + deltas[2] / 100.0,
    }


class TestPostRunReport(unittest.TestCase):
    def _run_dir(self, tmp: str) -> Path:
        run_dir = Path(tmp)
        records = [
            _record(1, rl=110.0, pid=100.0, deltas=(10.0, 0.0, -5.0)),
            _record(2, rl=200.0, pid=200.0, deltas=(0.0, 0.0, 0.0)),
        ]
        (run_dir / 'compact_queue_condition_summaries.jsonl').write_text(
            '\n'.join(json.dumps(record) for record in records) + '\n',
            encoding='utf-8',
        )
        (run_dir / 'physical_load_shadow_summary.json').write_text(
            json.dumps({
                'summary_status': 'FINALIZED',
                'faulted': False,
                'counters': {'strict_complete_conditions': 2},
                'core_counters': {},
            }),
            encoding='utf-8',
        )
        return run_dir

    def test_summary_math(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            document = summarize(self._run_dir(tmp))
        self.assertAlmostEqual(document['energy']['delta_energy_percent'], 10.0 / 300.0 * 100.0, places=6)
        self.assertAlmostEqual(document['loads']['blade_1']['mean_load_change_percent'], 5.0, places=6)
        self.assertAlmostEqual(document['loads']['blade_3']['mean_load_change_percent'], -2.5, places=6)
        self.assertAlmostEqual(document['loads']['mean_reduction_percent'], -(5.0 + 0.0 - 2.5) / 3.0, places=6)
        self.assertEqual(document['counts']['conditions'], 2)
        self.assertEqual(document['warnings'], [])
        self.assertFalse(document['acceptance'])

    def test_markdown_marks_sign_convention(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            text = render_markdown(summarize(self._run_dir(tmp)))
        self.assertIn('平均降低率', text)
        self.assertIn('正降低率 = RL 峰值低于 PID', text)
        self.assertIn('非正式验收', text)

    def test_missing_conditions_file_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                summarize(Path(tmp))


if __name__ == '__main__':
    unittest.main()
