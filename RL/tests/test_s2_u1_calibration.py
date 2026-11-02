from __future__ import annotations

import json
from pathlib import Path
import unittest

from RL.build_s2_u1_calibration import REPORT_SCHEMA_VERSION, SCHEMA_VERSION


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class TestS2U1Calibration(unittest.TestCase):
    def test_locked_artifact_and_replay_report_keep_physical_gate_closed(self) -> None:
        calibration = json.loads(
            (PROJECT_ROOT / 'RL/data/s2_u1_pid_envelope_3_to_25.json').read_text(encoding='utf-8')
        )
        report = json.loads(
            (PROJECT_ROOT / 'RL/data/s2_u1_pid_replay_3_to_25.json').read_text(encoding='utf-8')
        )

        self.assertEqual(calibration['schema_version'], SCHEMA_VERSION)
        self.assertEqual(report['schema_version'], REPORT_SCHEMA_VERSION)
        self.assertEqual(calibration['metadata']['condition_count'], 23)
        self.assertEqual(calibration['metadata']['rows_total'], 690_023)
        self.assertEqual(len(calibration['metadata']['input_manifest_sha256']), 64)
        self.assertEqual(len(report['calibration_sha256']), 64)
        self.assertTrue(report['checks']['all_thresholds_ordered'])
        self.assertTrue(report['checks']['emergency_unobserved_in_pid'])
        self.assertEqual(report['checks']['emergency_activation_count'], 0)
        self.assertFalse(calibration['metadata']['formal_closed_loop_acceptance'])
        self.assertFalse(report['formal_closed_loop_acceptance'])
        self.assertFalse(calibration['metadata']['online_load_signal_available'])


if __name__ == '__main__':
    unittest.main()