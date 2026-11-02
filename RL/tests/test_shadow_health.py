from __future__ import annotations

import unittest

from RL.utils.physical_load_shadow import shadow_health


class TestShadowHealth(unittest.TestCase):
    def test_ok_when_finalized_and_clean(self) -> None:
        health = shadow_health(
            {'submitted_frames': 10, 'invalid_submissions': 0, 'queue_full_dropped': 0,
             'worker_faults': 0, 'records_written': 8, 'finished_conditions': 1,
             'strict_complete_conditions': 1},
            faulted=False, finalized=True, worker_alive=False,
        )
        self.assertEqual(health['status'], 'OK')
        self.assertEqual(health['reasons'], [])
        self.assertAlmostEqual(health['submission_success_rate'], 1.0)

    def test_degraded_on_invalid_submission(self) -> None:
        health = shadow_health(
            {'submitted_frames': 9, 'invalid_submissions': 1},
            faulted=False, finalized=True, worker_alive=False,
        )
        self.assertEqual(health['status'], 'DEGRADED')
        self.assertIn('invalid_submissions=1', health['reasons'])
        self.assertAlmostEqual(health['submission_success_rate'], 0.9)

    def test_faulted_takes_precedence(self) -> None:
        health = shadow_health(
            {'submitted_frames': 0, 'invalid_submissions': 3, 'worker_faults': 1},
            faulted=True, finalized=False, worker_alive=True,
        )
        self.assertEqual(health['status'], 'FAULTED')
        self.assertIn('worker_faulted', health['reasons'])
        self.assertAlmostEqual(health['submission_success_rate'], 0.0)


if __name__ == '__main__':
    unittest.main()
