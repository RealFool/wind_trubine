from __future__ import annotations

import copy
from pathlib import Path
import unittest

from RL.train_rl import _validate_queue_manifest_config, _validate_queue_train_duration
from RL.utils.config import load_config
from RL.utils.physical_load_shadow_launch import OPERATOR_23WIND_TEMPLATE
from RL.utils.queue_manifest import expand_queue_manifest
from RL.utils.safe_residual_control import SafeResidualController


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_CONFIG = PROJECT_ROOT / OPERATOR_23WIND_TEMPLATE
QUEUE_MANIFEST = PROJECT_ROOT / load_config(str(TRAIN_CONFIG))['queue']['manifest_path']
BC_CONFIG = PROJECT_ROOT / 'RL/configs/b4_residual_smoke/bc_pretrain_b4_state20_residual_600s_v3.json'


class TestQueueConfigContract(unittest.TestCase):
    def test_current_unified_manifest_has_92_slots(self) -> None:
        manifest = load_config(str(QUEUE_MANIFEST))
        slots = expand_queue_manifest(manifest)

        self.assertEqual(len(slots), 92)
        self.assertEqual(manifest['total_condition_rounds'], 4)
        self.assertEqual(manifest['expanded_slot_count'], 92)
        self.assertTrue(all(slot.train_sim_s == 600.0 for slot in slots))
        for round_id in range(1, 5):
            self.assertEqual(
                sorted(slot.condition_id for slot in slots if slot.round_id == round_id),
                list(range(3, 26)),
            )

    def test_current_bc_and_training_share_actor_control_contract(self) -> None:
        bc = load_config(str(BC_CONFIG))
        training = load_config(str(TRAIN_CONFIG))
        bc_controller = SafeResidualController(bc['env'], bc['control'])
        training_controller = SafeResidualController(training['env'], training['control'])

        self.assertFalse(bc['env']['queue_lifecycle_enabled'])
        self.assertTrue(training['env']['queue_lifecycle_enabled'])
        self.assertEqual(bc['control']['action_representation'], 'residual')
        self.assertEqual(training['control']['action_representation'], 'residual')
        self.assertEqual(bc_controller.actor_control_fingerprint, training_controller.actor_control_fingerprint)
        self.assertTrue(training['agent']['td3_bc_enabled'])
        self.assertEqual(training['agent']['td3_bc_target_ratio'], 0.5)
        self.assertEqual(training['bc']['steps'], 0)
        self.assertTrue(training['train']['online_rl_enabled'])
        self.assertEqual(training['bc']['resume_checkpoint'], 'RL/BC_checkpoint/bladed_pid_b4_state20_residual_v3.pt')
        self.assertEqual(training['bc']['export_checkpoint_path'], '')
        self.assertTrue(training['env']['include_operating_state_observation'])
        self.assertEqual(training['env']['action_high'][0], 21_586_451.33303)
        self.assertIs(training['physical_shadow_launch']['authorized'], False)

    def test_current_queue_rejects_manifest_condition_mismatch(self) -> None:
        training = load_config(str(TRAIN_CONFIG))
        slots = _validate_queue_manifest_config(training, training['queue'])
        self.assertEqual(len(slots or []), 92)

        mismatched = copy.deepcopy(training)
        mismatched['queue']['condition_ids'] = list(range(2, 25))
        with self.assertRaisesRegex(ValueError, 'round 1'):
            _validate_queue_manifest_config(mismatched, mismatched['queue'])

    def test_manual_batch_controller_uses_one_full_profile_and_rounds_setting(self) -> None:
        source_path = PROJECT_ROOT / 'bladed_env/udptest/udptest/controller.cpp'
        source = source_path.read_text(encoding='utf-8')

        self.assertIn('MANUAL_BATCH_PROFILE_FULL_DYNAMIC = "full"', source)
        self.assertIn('BLADED_RL_MANUAL_BATCH_ROUNDS', source)
        self.assertEqual(load_config(str(QUEUE_MANIFEST))['manual_batch_profile'], 'full')

    def test_current_manifest_matches_dll_rotation(self) -> None:
        training = load_config(str(TRAIN_CONFIG))
        manifest = load_config(str(QUEUE_MANIFEST))
        slots = expand_queue_manifest(manifest)
        base_order = list(manifest['base_condition_order'])

        self.assertEqual(training['queue']['condition_ids'], list(range(3, 26)))
        self.assertEqual(training['queue']['total_condition_rounds'], 4)
        self.assertEqual(training['queue']['train_sim_s'], 600.0)
        self.assertEqual(manifest['manual_batch_profile'], 'full')
        self.assertEqual(manifest['manual_batch_rounds'], 4)
        self.assertEqual(manifest['rotation_stride'], 7)
        self.assertEqual(len(slots), 92)
        for round_index in range(4):
            offset = (round_index * 7) % len(base_order)
            expected = base_order[offset:] + base_order[:offset]
            actual = [slot.condition_id for slot in slots if slot.round_id == round_index + 1]
            self.assertEqual(actual, expected)

    def test_train_duration_requires_full_window_with_step_tolerance(self) -> None:
        queue_config = {'train_sim_s': 180.0, 'train_sim_tolerance_s': 0.1}
        elapsed_s = _validate_queue_train_duration(
            queue_config,
            condition_id=12,
            condition_instance_id=8,
            phase_elapsed_start_s=0.0,
            phase_elapsed_end_s=179.98,
        )
        self.assertAlmostEqual(elapsed_s, 179.98)
        with self.assertRaisesRegex(RuntimeError, 'too short'):
            _validate_queue_train_duration(
                queue_config,
                condition_id=12,
                condition_instance_id=8,
                phase_elapsed_start_s=0.0,
                phase_elapsed_end_s=179.0,
            )


if __name__ == '__main__':
    unittest.main()
