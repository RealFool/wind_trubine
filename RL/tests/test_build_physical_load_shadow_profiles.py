from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from RL.build_physical_load_shadow_profiles import (
    PROJECT_ROOT,
    SOURCE_PROFILE,
    SOURCE_PROFILE_23WIND,
    build_profiles,
)
from RL.utils.queue_manifest import expand_queue_manifest
from RL.utils.physical_queue_schedule import FULL_23WIND_PROFILE


class BuildPhysicalLoadShadowProfilesTests(unittest.TestCase):
    def test_one_23wind_profile_builds_dynamic_rounds_with_closed_identity_and_budget(self) -> None:
        for rounds, expected_updates in ((4, 69_000), (6, 115_000)):
            with self.subTest(rounds=rounds), tempfile.TemporaryDirectory(
                prefix='physical_shadow_unified_', dir=PROJECT_ROOT / 'RL/configs'
            ) as tmp:
                output_dir = Path(tmp) / 'profiles'
                report = build_profiles(
                    output_dir,
                    project_root=PROJECT_ROOT,
                    source_profile_path=PROJECT_ROOT / SOURCE_PROFILE_23WIND,
                    profile_names=(FULL_23WIND_PROFILE,),
                    rounds=rounds,
                )
                config = json.loads(
                    (output_dir / f'state20_physical_{FULL_23WIND_PROFILE}.json').read_text(encoding='utf-8')
                )
                manifest = json.loads(
                    (output_dir / f'state20_physical_{FULL_23WIND_PROFILE}_manifest.json').read_text(encoding='utf-8')
                )
                slots = expand_queue_manifest(manifest)

            expected_slots = 23 * rounds
            self.assertEqual(len(slots), expected_slots)
            self.assertEqual(manifest['total_condition_rounds'], rounds)
            self.assertEqual(manifest['manual_batch_profile'], 'full')
            self.assertEqual(manifest['expanded_slot_count'], expected_slots)
            self.assertEqual(config['queue']['condition_ids'], list(range(3, 26)))
            self.assertEqual(config['queue']['total_condition_rounds'], rounds)
            self.assertEqual(config['queue']['train_sim_s'], 600.0)
            self.assertEqual(config['state20_queue_contract']['profile'], FULL_23WIND_PROFILE)
            self.assertEqual(config['state20_queue_contract']['expected_scheduled_updates'], expected_updates)
            self.assertEqual(config['control']['resolved_control_contract_fingerprint'],
                             '578179ea2f324e7522014a930c5df8497dec6d5d819c2703358548891295ba24')
            self.assertEqual(config['load_predictor']['backend'], 'physical_fir_v2')
            self.assertEqual(config['load_predictor']['mode'], 'shadow')
            self.assertIs(config['load_predictor']['nn_active'], False)
            self.assertEqual(
                config['load_predictor']['max_records'],
                expected_slots * 2 * (30_000 // config['load_predictor']['origin_stride_frames'] + 1),
            )
            self.assertEqual(config['load_predictor']['max_record_bytes'], 8 * 1024)
            self.assertEqual(config['load_predictor']['max_total_bytes'], 18 * 1024 ** 3)
            self.assertEqual(config['logging']['queue_event_windows_max_total_bytes'], 2 * 1024 ** 3)
            self.assertIs(config['physical_shadow_launch']['authorized'], False)
            self.assertIs(config['physical_shadow_run_role']['policy_acceptance'], False)
            self.assertIs(config['physical_shadow_run_role']['soft_enabled'], False)
            self.assertIs(config['physical_shadow_run_role']['hard_enabled'], False)
            self.assertFalse(report['profiles'][FULL_23WIND_PROFILE]['bladed_launch_authorized'])
            self.assertEqual(report['profiles'][FULL_23WIND_PROFILE]['slots'], expected_slots)
            self.assertFalse(report['bladed_started'])

    def test_unified_23wind_profile_rejects_a_4wind_source(self) -> None:
        with tempfile.TemporaryDirectory(prefix='physical_shadow_unified_reject_', dir=PROJECT_ROOT / 'RL/configs') as tmp:
            with self.assertRaisesRegex(ValueError, 'Frozen source profile condition IDs|canonical 23-condition 600-second source contract'):
                build_profiles(
                    Path(tmp) / 'profiles',
                    project_root=PROJECT_ROOT,
                    source_profile_path=PROJECT_ROOT / SOURCE_PROFILE,
                    profile_names=(FULL_23WIND_PROFILE,),
                    rounds=4,
                )


if __name__ == '__main__':
    unittest.main()
