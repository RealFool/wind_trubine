from __future__ import annotations

from pathlib import Path
import os
import unittest

from RL.utils.physical_load_shadow_artifact import file_sha256, strict_json
from RL.utils.physical_load_shadow_launch import (
    OPERATOR_23WIND_PROFILE_BUILD_REPORT,
    OPERATOR_23WIND_RELEASE_ID,
    OPERATOR_23WIND_TEMPLATE,
)
from RL.utils.physical_queue_schedule import FULL_23WIND_PROFILE
from RL.utils.physical_queue_schedule import MAX_FULL_23WIND_ROUNDS


ROOT = Path(__file__).resolve().parents[2]
RELEASE_DIR = ROOT / 'RL/configs' / OPERATOR_23WIND_RELEASE_ID
RELEASE_MANIFEST = RELEASE_DIR / 'release_manifest.json'
PROFILE_REPORT = ROOT / OPERATOR_23WIND_PROFILE_BUILD_REPORT


@unittest.skipIf(
    os.environ.get('WT15_M6_TRANSITION') == '1',
    'M6 transition: pinned R2 runtime identity is intentionally invalidated; removed by M6B',
)
class TestUnifiedReleaseManifest(unittest.TestCase):
    def setUp(self) -> None:
        if not RELEASE_MANIFEST.is_file():
            self.skipTest('R2 release manifest has not been finalized')

    def test_one_configurable_rounds_profile_closes_across_template_and_report(self) -> None:
        release = strict_json(RELEASE_MANIFEST)
        config = strict_json(ROOT / OPERATOR_23WIND_TEMPLATE)
        report = strict_json(PROFILE_REPORT)
        profile = report['profiles'][FULL_23WIND_PROFILE]
        manifest = strict_json(ROOT / config['queue']['manifest_path'])

        self.assertEqual(release['manifest_format'], 'wind_turbine_unified_release_manifest')
        self.assertEqual(release['release_id'], OPERATOR_23WIND_RELEASE_ID)
        self.assertEqual(release['scope']['rounds_default'], 4)
        self.assertEqual(release['scope']['rounds_supported'], list(range(1, MAX_FULL_23WIND_ROUNDS + 1)))
        self.assertEqual(config['release_id'], OPERATOR_23WIND_RELEASE_ID)
        self.assertEqual(report['release_id'], OPERATOR_23WIND_RELEASE_ID)
        self.assertEqual(profile['config_path'], OPERATOR_23WIND_TEMPLATE)
        self.assertEqual(profile['slots'], release['scope']['default_slots'])
        self.assertEqual(config['state20_queue_contract']['profile'], FULL_23WIND_PROFILE)
        self.assertEqual(config['queue']['total_condition_rounds'], 4)
        self.assertEqual(manifest['expanded_slot_count'], release['scope']['default_slots'])
        self.assertEqual(config['queue']['train_sim_s'], 600.0)

    def test_all_pinned_files_and_runtime_sources_match(self) -> None:
        release = strict_json(RELEASE_MANIFEST)
        for relative_path, expected_sha in release['pinned_files'].items():
            with self.subTest(path=relative_path):
                self.assertEqual(file_sha256(ROOT / relative_path), expected_sha)

        report = strict_json(PROFILE_REPORT)
        runtime_identity = report['profiles'][FULL_23WIND_PROFILE]['runtime_identity']
        for relative_path, expected_sha in runtime_identity['runtime_code_sha256'].items():
            with self.subTest(runtime=relative_path):
                self.assertEqual(file_sha256(ROOT / relative_path), expected_sha)
        for relative_path, expected_sha in runtime_identity['operator_23wind_preparation_code_sha256'].items():
            with self.subTest(preparation=relative_path):
                self.assertEqual(file_sha256(ROOT / relative_path), expected_sha)

    def test_old_r1_is_archived_and_all_authorizations_remain_false(self) -> None:
        release = strict_json(RELEASE_MANIFEST)
        archive = release['historical_archive']
        r1_path = ROOT / archive['release_manifest']['path']
        self.assertEqual(file_sha256(r1_path), archive['release_manifest']['sha256'])
        self.assertEqual(release['historical_archive']['release_id'], 'WT15_STATE20_RL_20261022_R1')
        self.assertEqual(release['historical_archive']['profile']['rounds'], 6)
        self.assertEqual(release['historical_archive']['profile']['slots'], 138)
        self.assertIs(release['release_ready'], False)
        for value in release['authorization'].values():
            self.assertIs(value, False)
        config = strict_json(ROOT / OPERATOR_23WIND_TEMPLATE)
        self.assertIs(config['physical_shadow_launch']['authorized'], False)


if __name__ == '__main__':
    unittest.main()
