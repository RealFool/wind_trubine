from __future__ import annotations

import unittest

import numpy as np
import torch

from RL.models.load_predictor_tcn import CausalTCNLoadPredictor
from RL.replay.load_prediction_replay import LoadPredictionFrame, LoadPredictionReplay
from RL.utils.load_predictor_contract import (
    PLAN_KIND_REALIZED,
    PLAN_KIND_ZOH_CANDIDATE,
    LoadPredictorInterface,
)


def _interface() -> LoadPredictorInterface:
    return LoadPredictorInterface.from_path('RL/data/load_predictor_interface_v2.json')


def _frame(index: int, *, condition_instance_id: int = 1) -> LoadPredictionFrame:
    return LoadPredictionFrame(
        condition_id=3,
        condition_instance_id=condition_instance_id,
        timestamp_s=index * 0.02,
        sent_action_real=np.asarray([100.0 + index, 0.1], dtype=np.float64),
        actual_electrical_torque_nm=200.0 + index,
        actual_pitch_rad=0.2 + index * 0.001,
        blade_root_my_nm=np.asarray([100.0 + index, 90.0 + index, 80.0 + index]),
        pid_peak_abs_nm=np.asarray([120.0, 120.0, 120.0]),
    )


class TestLoadPredictionReplay(unittest.TestCase):
    def test_pending_sample_does_not_resolve_before_full_future_window(self) -> None:
        replay = LoadPredictionReplay(
            _interface(),
            action_low=[0.0, 0.0],
            action_high=[1_000.0, 1.0],
        )
        for index in range(26):
            replay.observe_frame(_frame(index))
            if index == 0:
                replay.add_pending(
                    history=np.zeros((50, 4), dtype=np.float32),
                    candidate_action_real=[100.0, 0.1],
                    plan_kind=PLAN_KIND_ZOH_CANDIDATE,
                    online=True,
                )
            if index < 25:
                self.assertEqual(replay.pending_count, 1)
        self.assertEqual(replay.pending_count, 0)

    def test_realized_and_zoh_semantics_are_separate_and_mismatch_gates_calibration(self) -> None:
        replay = LoadPredictionReplay(
            _interface(),
            action_low=[0.0, 0.0],
            action_high=[1_000.0, 1.0],
        )
        for index in range(26):
            records = replay.observe_frame(_frame(index))
            if index == 0:
                replay.add_pending(
                    history=np.ones((50, 4), dtype=np.float32),
                    candidate_action_real=[0.0, 0.0],
                    plan_kind=PLAN_KIND_ZOH_CANDIDATE,
                    online=True,
                )
        realized = [record for record in records if record.plan_kind == PLAN_KIND_REALIZED]
        candidate = [record for record in records if record.plan_kind == PLAN_KIND_ZOH_CANDIDATE]

        self.assertTrue(realized)
        self.assertFalse(candidate)
        self.assertTrue(all(not record.online_authorized for record in realized))

    def test_condition_boundary_drops_pending_without_cross_condition_labels(self) -> None:
        replay = LoadPredictionReplay(
            _interface(),
            action_low=[0.0, 0.0],
            action_high=[1_000.0, 1.0],
        )
        replay.observe_frame(_frame(0))
        replay.add_pending(
            history=np.zeros((50, 4), dtype=np.float32),
            candidate_action_real=[100.0, 0.1],
            plan_kind=PLAN_KIND_ZOH_CANDIDATE,
            online=True,
        )
        stats = replay.start_condition(3, 2)
        self.assertEqual(stats['dropped_pending_count'], 1)
        self.assertEqual(replay.pending_count, 0)

    def test_tcn_output_is_causal_candidate_conditioned_and_has_fixed_shape(self) -> None:
        interface = LoadPredictorInterface.from_path('RL/data/load_predictor_interface_v3.json')
        model = CausalTCNLoadPredictor(input_dim=31, interface=interface, hidden_dim=8)
        history = torch.zeros((1, interface.history_frames, 31))
        changed_future = history.clone()
        changed_future[:, 20:, :] = 100.0
        sequence_a = model.forward_sequence(history)
        sequence_b = model.forward_sequence(changed_future)

        plan = torch.zeros((1, max(interface.horizon_frames), 2))
        horizon_index = torch.tensor([0], dtype=torch.long)
        plan_kind = torch.tensor([1], dtype=torch.long)
        prediction = model.predict_candidate(history, plan, horizon_index, plan_kind)
        changed_plan = plan.clone()
        changed_plan[:, :interface.horizon_frames[0], 0] = 21_000_000.0
        changed_plan_prediction = model.predict_candidate(
            history,
            changed_plan,
            horizon_index,
            plan_kind,
        )

        self.assertEqual(tuple(prediction.shape), (1, 5))
        torch.testing.assert_close(sequence_a[:, :20], sequence_b[:, :20])
        self.assertFalse(torch.allclose(prediction, changed_plan_prediction))


if __name__ == '__main__':
    unittest.main()