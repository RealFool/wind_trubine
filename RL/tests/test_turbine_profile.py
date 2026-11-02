from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from RL.turbine_profile import TurbineProfile, load_turbine_profile_for_config


ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / 'configs/turbines/wt15.json'
R2_CONFIG = ROOT / 'RL/configs/WT15_STATE20_RL_20261023_R2/state20_physical_23wind_round600_training.json'


class TestTurbineProfile(unittest.TestCase):
    def test_load_wt15(self) -> None:
        profile = TurbineProfile.from_path(PROFILE)
        self.assertEqual(profile.turbine_id, 'wt15')
        self.assertEqual(profile.rated_power_w, 15_000_000.0)
        self.assertEqual(profile.gear_ratio, 97.0)
        self.assertEqual(profile.action_high[0], 21_586_451.33303)
        self.assertEqual(profile.pitch_rate_limit_rad, 0.0003)

    def test_matches_r2_config(self) -> None:
        profile = TurbineProfile.from_path(PROFILE)
        config = json.loads(R2_CONFIG.read_text(encoding='utf-8'))
        profile.assert_config_matches(config, project_root=ROOT)

    def test_detects_power_mismatch(self) -> None:
        profile = TurbineProfile.from_path(PROFILE)
        config = json.loads(R2_CONFIG.read_text(encoding='utf-8'))
        altered = copy.deepcopy(config)
        altered['env']['rated_power'] = 22_000_000.0
        with self.assertRaisesRegex(ValueError, 'env.rated_power'):
            profile.assert_config_matches(altered, project_root=ROOT)

    def test_detects_asset_path_mismatch(self) -> None:
        profile = TurbineProfile.from_path(PROFILE)
        config = json.loads(R2_CONFIG.read_text(encoding='utf-8'))
        altered = copy.deepcopy(config)
        altered['control']['region_aware_baseline_path'] = 'RL/data/other.json'
        with self.assertRaisesRegex(ValueError, 'region_aware_baseline_path'):
            profile.assert_config_matches(altered, project_root=ROOT)

    def test_load_for_config_optional(self) -> None:
        self.assertIsNone(load_turbine_profile_for_config({}, project_root=ROOT))
        profile = load_turbine_profile_for_config(
            {'turbine_profile_path': 'configs/turbines/wt15.json'}, project_root=ROOT
        )
        self.assertIsNotNone(profile)
        self.assertEqual(profile.turbine_id, 'wt15')

    def test_invalid_schema_rejected(self) -> None:
        document = json.loads(PROFILE.read_text(encoding='utf-8'))
        document['schema_version'] = 'bad'
        with self.assertRaises(ValueError):
            TurbineProfile.from_document(document)


if __name__ == '__main__':
    unittest.main()
