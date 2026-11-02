from __future__ import annotations

import unittest

from RL.build_slot_checklist import build_checklist


def _manifest() -> dict:
    return {
        'protocol_version': 'bladed_udp_19_si_my_queue_v2',
        'execution_mode': 'independent_case_reload',
        'total_condition_rounds': 2,
        'base_condition_order': [8, 11],
        'rotation_stride': 1,
        'queue_epoch_start': 1,
        'condition_instance_start': 1,
        'post_load_discard_sim_s': 0.0,
        'train_sim_s': 600.0,
        'manual_batch_profile': 'smoke_4x4',
        'state20_profile': 'test_profile',
    }


class TestSlotChecklist(unittest.TestCase):
    def test_builds_one_row_per_slot(self) -> None:
        text = build_checklist(_manifest())
        slot_rows = [
            line for line in text.splitlines()
            if line.startswith('| ') and line.count('|') == 12
            and line.split('|')[1].strip().isdigit()
        ]
        self.assertEqual(len(slot_rows), 4)
        self.assertIn('rounds: 2；slots: 4；conditions: 2', text)
        self.assertIn('| 1 | 1 | 1 | 8 |', text)
        self.assertIn('| 2 | 1 | 2 | 11 |', text)
        self.assertIn('| 3 | 2 | 3 | 11 |', text)
        self.assertIn('| 4 | 2 | 4 | 8 |', text)

    def test_empty_case_path_is_marked_manual(self) -> None:
        text = build_checklist(_manifest())
        self.assertIn('（手工加载）', text)

    def test_invalid_manifest_raises(self) -> None:
        with self.assertRaises(ValueError):
            build_checklist({'protocol_version': 'bad'})


if __name__ == '__main__':
    unittest.main()
