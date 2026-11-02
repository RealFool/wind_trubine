from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

from RL.build_pid_rotor_speed_reference import (
    SCHEMA_VERSION,
    build_pid_rotor_speed_reference,
    evaluate_holdout,
)


class TestPidRotorSpeedReference(unittest.TestCase):
    @staticmethod
    def _write_trace(path: Path, condition_id: int, offset: float = 0.0) -> None:
        with path.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                'Time', 'I_AverageWindSpeed', 'I_RotorSpeed', 'I_RealPower',
            ])
            writer.writeheader()
            for index, timestamp in enumerate((0.0, 1.0, 2.0, 3.0, 600.0)):
                writer.writerow({
                    'Time': timestamp,
                    'I_AverageWindSpeed': float(condition_id) + offset,
                    'I_RotorSpeed': 0.6 + 0.02 * condition_id + 0.001 * index + offset,
                    'I_RealPower': 1_000_000.0,
                })

    def _raw_root(self, root: Path) -> Path:
        raw = root / 'raw'
        raw.mkdir()
        for condition_id in range(3, 26):
            self._write_trace(raw / f'data_windspeed{condition_id}.csv', condition_id)
        return raw

    def test_builds_23_condition_hash_locked_continuous_schedule(self) -> None:
        with tempfile.TemporaryDirectory(prefix='rotor_schedule_') as tmpdir:
            document = build_pid_rotor_speed_reference(
                self._raw_root(Path(tmpdir)),
                min_eligible_rows=2,
            )
            self.assertEqual(document['schema_version'], SCHEMA_VERSION)
            self.assertEqual(document['metadata']['condition_count'], 23)
            self.assertEqual(len(document['metadata']['input_manifest_sha256']), 64)
            self.assertEqual(len(document['schedule']), 23)
            self.assertTrue(document['continuity']['target_jump_passed'])
            self.assertTrue(document['continuity']['deadband_jump_passed'])
            self.assertEqual(document['metadata']['validation_data_policy'], 'raw_infer_and_new_seed_are_excluded_from_calibration')

    def test_holdout_has_zero_hash_overlap_and_is_diagnostic_only(self) -> None:
        with tempfile.TemporaryDirectory(prefix='rotor_schedule_') as tmpdir:
            root = Path(tmpdir)
            document = build_pid_rotor_speed_reference(self._raw_root(root), min_eligible_rows=2)
            schedule_path = root / 'schedule.json'
            schedule_path.write_text(json.dumps(document), encoding='utf-8')
            holdout = root / 'raw_infer'
            holdout.mkdir()
            self._write_trace(holdout / 'data_windspeed8.csv', 8, offset=0.1)
            report = evaluate_holdout(schedule_path, holdout)
            self.assertEqual(report['calibration_holdout_sha256_overlap_count'], 0)
            self.assertEqual(report['condition_count'], 1)
            self.assertFalse(report['formal_acceptance'])

    def test_rejects_non_raw_calibration_directory(self) -> None:
        with tempfile.TemporaryDirectory(prefix='rotor_schedule_') as tmpdir:
            root = Path(tmpdir)
            train = root / 'train'
            train.mkdir()
            with self.assertRaisesRegex(ValueError, "must be named 'raw'"):
                build_pid_rotor_speed_reference(train)


if __name__ == '__main__':
    unittest.main()