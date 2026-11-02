from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from RL.build_state20_control_config import BC_CONFIG_PATH, build_bc_config
from RL.utils.control_contract import ENGINEERING_DEMAND_HIGH_NM, resolve_configured_control_contract
from RL.utils.safe_residual_control import SafeResidualController


class TestState20ControlConfig(unittest.TestCase):
    def test_valid_state_regime_one_hot_is_not_continuous_ood(self) -> None:
        with tempfile.TemporaryDirectory(prefix='state20_ood_') as tmpdir:
            config = build_bc_config(output_path=Path(tmpdir) / 'bc.json')
            controller = SafeResidualController(config['env'], config['control'])

        controller.set_observation_distribution(
            np.zeros(20, dtype=np.float32),
            np.ones(20, dtype=np.float32),
        )
        valid = np.zeros(20, dtype=np.float32)
        valid[12] = 1.0
        valid[17] = 1.0
        self.assertLess(controller.ood_score(valid), controller.ood_threshold)

        invalid = valid.copy()
        invalid[12:20] = 0.0
        self.assertTrue(np.isinf(controller.ood_score(invalid)))

    def test_builder_emits_self_consistent_state20_bc_contract(self) -> None:
        with tempfile.TemporaryDirectory(prefix='state20_control_config_') as tmpdir:
            output_path = Path(tmpdir) / 'bc.json'
            config = build_bc_config(output_path=output_path)
            resolved = resolve_configured_control_contract(config, project_root=Path(__file__).resolve().parents[2])
            controller = SafeResidualController(config['env'], config['control'])
            parsed_seed = json.loads(output_path.read_text(encoding='utf-8'))

        assert resolved is not None
        self.assertEqual(config['env']['action_high'][0], ENGINEERING_DEMAND_HIGH_NM)
        self.assertEqual(config['env']['observation_action_high'][0], ENGINEERING_DEMAND_HIGH_NM)
        self.assertTrue(config['env']['include_operating_state_observation'])
        self.assertFalse(config['bc']['force_retrain'])
        self.assertEqual(config['b4_residual_contract']['schema_version'], 'b4_centered_residual_control_v3')
        self.assertEqual(
            config['load_predictor']['interface_schema_version'],
            'causal_load_predictor_interface_v3',
        )
        self.assertEqual(
            controller.resolved_control_contract_fingerprint,
            config['control_contract_fingerprint'],
        )
        self.assertEqual(
            resolved[1],
            config['control_contract_fingerprint'],
        )
        self.assertEqual(
            parsed_seed['load_predictor']['interface_path'],
            config['load_predictor']['interface_path'],
        )
        self.assertEqual(
            parsed_seed['load_predictor']['dataset_manifest_sha256'],
            config['load_predictor']['dataset_manifest_sha256'],
        )
        self.assertEqual(
            parsed_seed['load_predictor']['shadow_audit_max_total_bytes'],
            256 * 1024 * 1024,
        )

    def test_generated_default_config_is_valid_json(self) -> None:
        with tempfile.TemporaryDirectory(prefix='state20_default_config_') as tmpdir:
            output_path = Path(tmpdir) / 'bc_config.json'
            config = build_bc_config(output_path=output_path)
            parsed = json.loads(output_path.read_text(encoding='utf-8'))

        self.assertEqual(parsed['control_contract_fingerprint'], config['control_contract_fingerprint'])

    def test_fixed_pretraining_config_can_be_refreshed_and_force_retrain_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory(prefix='state20_refresh_config_') as tmpdir:
            output_path = Path(tmpdir) / 'bc_config.json'
            output_path.write_text('stale config', encoding='utf-8')
            config = build_bc_config(output_path=output_path, overwrite=True, force_retrain=True)
            parsed = json.loads(output_path.read_text(encoding='utf-8'))

        self.assertTrue(parsed['bc']['force_retrain'])
        self.assertEqual(parsed['control_contract_fingerprint'], config['control_contract_fingerprint'])


if __name__ == '__main__':
    unittest.main()