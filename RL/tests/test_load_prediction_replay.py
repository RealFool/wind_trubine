from __future__ import annotations

import unittest

import numpy as np

from RL.replay.load_prediction_replay import LoadPredictionFrame, LoadPredictionReplay
from RL.utils.load_predictor_contract import LoadPredictorInterface, PLAN_KIND_ZOH_CANDIDATE


class TestLoadPredictionReplay(unittest.TestCase):
    @staticmethod
    def _interface() -> LoadPredictorInterface:
        return LoadPredictorInterface.from_path('RL/data/load_predictor_interface_v2.json')

    @staticmethod
    def _frame(index: int, torque: float | None = None) -> LoadPredictionFrame:
        return LoadPredictionFrame(
            condition_id=3,
            condition_instance_id=1,
            timestamp_s=index * 0.02,
            sent_action_real=np.asarray([100.0 + index if torque is None else torque, 0.1]),
            actual_electrical_torque_nm=200.0 + index,
            actual_pitch_rad=0.2 + index * 0.001,
            blade_root_my_nm=np.asarray([100.0 + index, 90.0 + index, 80.0 + index]),
            pid_peak_abs_nm=np.asarray([120.0, 120.0, 120.0]),
        )

    def test_pending_window_emits_realized_and_zoh_only_when_plan_matches(self) -> None:
        replay = LoadPredictionReplay(
            self._interface(),
            action_low=[0.0, 0.0],
            action_high=[1_000.0, 1.0],
            max_pending=20,
        )
        replay.start_condition(3, 1)
        replay.observe_frame(self._frame(0, torque=100.0))
        replay.add_pending(
            history=np.zeros((50, 4), dtype=np.float32),
            candidate_action_real=np.asarray([100.0, 0.1]),
            plan_kind=PLAN_KIND_ZOH_CANDIDATE,
            online=True,
        )
        completed = []
        for index in range(1, 26):
            completed.extend(replay.observe_frame(self._frame(index)))

        self.assertEqual(len(completed), 5)
        self.assertEqual(len(replay.samples(plan_kind='realized_plan')), 3)
        self.assertEqual(len(replay.samples(plan_kind=PLAN_KIND_ZOH_CANDIDATE)), 2)
        self.assertTrue(all(sample.calibration_eligible for sample in replay.samples(plan_kind=PLAN_KIND_ZOH_CANDIDATE)))

    def test_condition_boundary_and_incomplete_windows_are_dropped(self) -> None:
        replay = LoadPredictionReplay(
            self._interface(),
            action_low=[0.0, 0.0],
            action_high=[100.0, 1.0],
            max_pending=20,
        )
        replay.start_condition(3, 1)
        replay.observe_frame(self._frame(0, torque=10.0))
        replay.add_pending(
            history=np.zeros((50, 4), dtype=np.float32),
            candidate_action_real=np.asarray([10.0, 0.2]),
            plan_kind=PLAN_KIND_ZOH_CANDIDATE,
            online=True,
        )
        stats = replay.start_condition(4, 2)
        self.assertEqual(stats['dropped_pending_count'], 1)
        self.assertEqual(replay.pending_count, 0)

    def test_mismatched_realized_plan_does_not_authorize_zoh_sample(self) -> None:
        replay = LoadPredictionReplay(
            self._interface(),
            action_low=[0.0, 0.0],
            action_high=[100.0, 1.0],
            max_pending=20,
        )
        replay.start_condition(3, 1)
        replay.observe_frame(self._frame(0, torque=10.0))
        replay.add_pending(
            history=np.zeros((50, 4), dtype=np.float32),
            candidate_action_real=np.asarray([10.0, 0.2]),
            plan_kind=PLAN_KIND_ZOH_CANDIDATE,
            online=True,
        )
        for index in range(1, 26):
            replay.observe_frame(self._frame(index, torque=10.0 if index < 5 else 30.0))

        self.assertEqual(len(replay.samples(plan_kind='realized_plan')), 3)
        self.assertEqual(len(replay.samples(plan_kind=PLAN_KIND_ZOH_CANDIDATE)), 0)


if __name__ == '__main__':
    unittest.main()