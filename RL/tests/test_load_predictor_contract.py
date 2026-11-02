from __future__ import annotations

import unittest

import numpy as np

from RL.utils.load_predictor_contract import (
    PLAN_KIND_REALIZED,
    PLAN_KIND_ZOH_CANDIDATE,
    LoadPredictorInterface,
    make_zoh_candidate_plan,
    normalized_action_plan_mismatch,
)


class TestLoadPredictorContract(unittest.TestCase):
    def test_v2_interface_freezes_causal_horizons_and_plan_permissions(self) -> None:
        interface = LoadPredictorInterface.from_path('RL/data/load_predictor_interface_v2.json')

        self.assertEqual(interface.history_frames, 50)
        self.assertEqual(interface.horizon_frames, (5, 10, 25))
        self.assertEqual(interface.validate_plan_kind(PLAN_KIND_ZOH_CANDIDATE, online=True), PLAN_KIND_ZOH_CANDIDATE)
        with self.assertRaisesRegex(ValueError, 'not authorized online'):
            interface.validate_plan_kind(PLAN_KIND_REALIZED, online=True)
        self.assertEqual(interface.validate_plan_kind(PLAN_KIND_REALIZED, online=False), PLAN_KIND_REALIZED)

    def test_zoh_plan_and_mismatch_use_normalized_action_space(self) -> None:
        plan = make_zoh_candidate_plan([10.0, 0.2], 5)
        actual = plan.copy()
        actual[-1, 0] = 20.0

        self.assertEqual(plan.shape, (5, 2))
        self.assertAlmostEqual(
            normalized_action_plan_mismatch(actual, plan, [0.0, 0.0], [100.0, 1.0]),
            0.1,
        )


if __name__ == '__main__':
    unittest.main()