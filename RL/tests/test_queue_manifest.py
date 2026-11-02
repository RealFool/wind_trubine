from __future__ import annotations

import unittest

from RL.utils.queue_manifest import DEFAULT_DIAGNOSTIC_ORDER, expand_queue_manifest


class TestQueueManifest(unittest.TestCase):
    def test_independent_reload_manifest_covers_every_condition_once_per_round(self) -> None:
        document = {
            'protocol_version': 'bladed_udp_19_si_my_queue_v1',
            'execution_mode': 'independent_case_reload',
            'total_condition_rounds': 2,
            'base_condition_order': list(DEFAULT_DIAGNOSTIC_ORDER),
            'rotation_stride': 7,
            'queue_epoch_start': 10,
            'condition_instance_start': 100,
            'case_path_template': 'C:/bladed/wind{condition_id}.prj',
            'post_load_discard_sim_s': 30.0,
            'train_sim_s': 600.0,
        }

        slots = expand_queue_manifest(document)

        self.assertEqual(len(slots), 46)
        self.assertEqual([slot.condition_id for slot in slots[:23]], list(DEFAULT_DIAGNOSTIC_ORDER))
        self.assertEqual([slot.condition_id for slot in slots[23:]], list(DEFAULT_DIAGNOSTIC_ORDER[7:] + DEFAULT_DIAGNOSTIC_ORDER[:7]))
        self.assertEqual(sorted(slot.condition_id for slot in slots[:23]), list(range(3, 26)))
        self.assertEqual([slot.queue_epoch for slot in slots], list(range(10, 56)))
        self.assertEqual([slot.condition_instance_id for slot in slots], list(range(100, 146)))
        self.assertEqual(slots[0].case_path, 'C:/bladed/wind12.prj')
        self.assertEqual(slots[0].post_load_discard_sim_s, 30.0)
        self.assertEqual(slots[0].train_sim_s, 600.0)

    def test_full_23wind_six_round_manifest_expands_to_138_closed_slots(self) -> None:
        document = {
            'protocol_version': 'bladed_udp_19_si_my_queue_v2',
            'execution_mode': 'independent_case_reload',
            'total_condition_rounds': 6,
            'base_condition_order': [
                12, 11, 13, 10, 14, 9, 15, 8, 16, 7, 17, 6,
                18, 5, 19, 4, 20, 3, 21, 22, 23, 24, 25,
            ],
            'rotation_stride': 7,
            'queue_epoch_start': 1,
            'condition_instance_start': 1,
            'train_sim_s': 600.0,
        }

        slots = expand_queue_manifest(document)

        self.assertEqual(len(slots), 138)
        self.assertEqual([slot.round_id for slot in slots], [round_id for round_id in range(1, 7) for _ in range(23)])
        self.assertEqual([slot.condition_id for slot in slots[:23]], document['base_condition_order'])
        for round_id in range(1, 7):
            round_slots = [slot for slot in slots if slot.round_id == round_id]
            self.assertEqual(len(round_slots), 23)
            self.assertEqual(set(slot.condition_id for slot in round_slots), set(range(3, 26)))
            self.assertTrue(all(slot.train_sim_s == 600.0 for slot in round_slots))
        self.assertEqual([slot.queue_epoch for slot in slots], list(range(1, 139)))
        self.assertEqual([slot.condition_instance_id for slot in slots], list(range(1, 139)))


if __name__ == '__main__':
    unittest.main()