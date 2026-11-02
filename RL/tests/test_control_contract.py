from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from RL.utils.control_contract import (
    ENGINEERING_DEMAND_HIGH_NM,
    PhysicalActionTransform,
    resolve_configured_control_contract,
    decode_model_action,
    load_static_control_contract,
    normalize_action_for_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class TestControlContract(unittest.TestCase):
    def test_static_contract_is_valid_and_fingerprinted(self) -> None:
        document, fingerprint = load_static_control_contract(
            PROJECT_ROOT / 'RL/configs/b4_residual_smoke/control_contract_static_v2.json'
        )

        self.assertEqual(document['action']['engineering_demand_high_nm'], ENGINEERING_DEMAND_HIGH_NM)
        self.assertEqual(len(fingerprint), 64)

    def test_physical_projection_preserves_exact_engineering_torque_cap(self) -> None:
        transform = PhysicalActionTransform(
            action_low=np.asarray([0.0, 0.0]),
            action_high=np.asarray([ENGINEERING_DEMAND_HIGH_NM, 0.43]),
            action_delta_low=np.asarray([-90_000.0, -0.000698]),
            action_delta_high=np.asarray([90_000.0, 0.000698]),
        )

        projection = transform.project(
            np.asarray([ENGINEERING_DEMAND_HIGH_NM + 1_000.0, 0.1]),
            apply_rate_limit=False,
        )

        self.assertEqual(projection.action_real.dtype, np.dtype(np.float64))
        self.assertEqual(float(projection.action_real[0]), ENGINEERING_DEMAND_HIGH_NM)
        self.assertLessEqual(float(projection.action_real[0]), ENGINEERING_DEMAND_HIGH_NM)
        self.assertEqual(transform.serialize_for_wire(projection.action_real)[0], ENGINEERING_DEMAND_HIGH_NM)

    def test_previous_action_model_round_trip_preserves_engineering_cap(self) -> None:
        action_low = np.asarray([0.0, 0.0])
        action_high = np.asarray([ENGINEERING_DEMAND_HIGH_NM, 0.43])

        normalized = normalize_action_for_model(action_high, action_low, action_high)
        decoded = decode_model_action(normalized, action_low, action_high)

        self.assertEqual(normalized.dtype, np.dtype(np.float32))
        self.assertEqual(float(decoded[0]), ENGINEERING_DEMAND_HIGH_NM)

    def test_configured_contract_requires_all_paths_and_matching_fingerprint(self) -> None:
        with self.assertRaisesRegex(ValueError, 'must define'):
            resolve_configured_control_contract(
                {'control_contract_static_path': 'RL/configs/b4_residual_smoke/control_contract_static_v2.json'},
                project_root=PROJECT_ROOT,
            )


if __name__ == '__main__':
    unittest.main()