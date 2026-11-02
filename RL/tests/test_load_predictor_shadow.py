from __future__ import annotations

import hashlib
import base64
import json
from pathlib import Path
import unittest
import zlib

import numpy as np
import torch

from RL.models.load_predictor_tcn import CausalTCNLoadPredictor
from RL.utils.load_predictor_contract import LoadPredictorInterface
from RL.utils.load_predictor_shadow import FEATURE_DIM, LoadPredictorShadow


class TestLoadPredictorShadow(unittest.TestCase):
    @staticmethod
    def _payload(index: int) -> dict[str, float]:
        return {
            'Time': index * 0.02,
            'I_RotorSpeed': 0.8,
            'I_GeneratorSpeed': 0.8,
            'I_RealPower': 15_000_000.0,
            'I_ElectricalTorque': 20_000_000.0,
            'I_PitchPosition1': 0.1,
            'I_NacelleForeAftAcceleration': 0.0,
            'I_NacelleSideSideAcceleration': 0.0,
            'I_AverageWindSpeed': 14.0,
            'I_RotorAzimuth': index * 0.1,
            'I_YawError': 0.0,
            'Blade root 1 My (Nm)': 50_000_000.0,
            'Blade root 2 My (Nm)': 49_000_000.0,
            'Blade root 3 My (Nm)': 48_000_000.0,
        }

    def test_shadow_waits_for_history_and_never_changes_candidate_action(self) -> None:
        interface_path = 'RL/data/load_predictor_interface_v3.json'
        interface = LoadPredictorInterface.from_path(interface_path)
        shadow = LoadPredictorShadow(
            interface,
            action_low=[0.0, 0.0],
            action_high=[21_586_451.33303, 0.43],
            interface_sha256=hashlib.sha256(Path(interface_path).read_bytes()).hexdigest(),
        )
        candidate = np.asarray([20_010_000.0, 0.10035], dtype=np.float64)
        last_result = None
        for index in range(50):
            last_result = shadow.observe(
                condition_id=3,
                condition_instance_id=1,
                payload=self._payload(index),
                sent_action_real=candidate,
                projected_candidate_action_real=candidate,
                state_encoding=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                pid_peak_abs_nm=[55_000_000.0, 55_000_000.0, 55_000_000.0],
            )
            self.assertTrue(np.array_equal(last_result.action_real, candidate))
            if index < 49:
                self.assertEqual(last_result.diagnostics['load_predictor_shadow_zoh_candidate_submitted'], 0.0)
        assert last_result is not None
        self.assertEqual(last_result.feature.shape, (FEATURE_DIM,))
        self.assertEqual(last_result.diagnostics['load_predictor_shadow_history_ready'], 1.0)
        self.assertEqual(last_result.diagnostics['load_predictor_shadow_zoh_candidate_submitted'], 1.0)
        self.assertEqual(last_result.diagnostics['load_predictor_shadow_action_changed'], 0.0)
        self.assertEqual(shadow.replay.pending_count, 1)

    def test_shadow_candidate_predictor_returns_each_horizon_but_keeps_action_unchanged(self) -> None:
        interface_path = 'RL/data/load_predictor_interface_v3.json'
        interface = LoadPredictorInterface.from_path(interface_path)
        model = CausalTCNLoadPredictor(input_dim=31, interface=interface, hidden_dim=8)
        shadow = LoadPredictorShadow(
            interface,
            action_low=[0.0, 0.0],
            action_high=[21_586_451.33303, 0.43],
            interface_sha256=hashlib.sha256(Path(interface_path).read_bytes()).hexdigest(),
            model=model,
            device=torch.device('cpu'),
        )
        result = None
        for index in range(interface.history_frames):
            result = shadow.observe(
                condition_id=3,
                condition_instance_id=101,
                payload=self._payload(index),
                sent_action_real=[20_000_000.0, 0.05],
                projected_candidate_action_real=[20_100_000.0, 0.051],
                state_encoding=[0.0] * 8,
                pid_peak_abs_nm=[50_000_000.0] * 3,
                optimization_state='above_rated',
            )

        assert result is not None
        self.assertEqual(result.predictor_output.shape, (len(interface.soft_guard_horizons_s), 5))
        self.assertTrue(np.array_equal(result.action_real, np.asarray([20_100_000.0, 0.051])))
        self.assertEqual(result.diagnostics['load_predictor_shadow_action_changed'], 0.0)
        self.assertEqual(result.diagnostics['load_predictor_shadow_prediction_available'], 1.0)
        self.assertIn('load_predictor_shadow_predicted_peak_ratio_blade_3_h200', result.diagnostics)
        completed = None
        for index in range(interface.history_frames, interface.history_frames + max(interface.horizon_frames)):
            completed = shadow.observe(
                condition_id=3,
                condition_instance_id=101,
                payload=self._payload(index),
                sent_action_real=[20_000_000.0, 0.05],
                projected_candidate_action_real=[20_100_000.0, 0.051],
                state_encoding=[0.0] * 8,
                pid_peak_abs_nm=[50_000_000.0] * 3,
                optimization_state='above_rated',
            )
        assert completed is not None
        audit_records = shadow.drain_audit_records()
        self.assertEqual(len(audit_records), len(interface.horizons_s))
        online_rows = [row for row in audit_records if row['online_authorized']]
        self.assertEqual(len(online_rows), 2)
        self.assertTrue(all(row['calibration_eligible'] for row in online_rows))
        self.assertTrue(all(row['optimization_state'] == 'above_rated' for row in online_rows))
        self.assertEqual(online_rows[0]['predictor_interface_sha256'], shadow.interface_sha256)
        self.assertEqual(online_rows[0]['actor_policy_version'], 0)
        self.assertEqual(online_rows[0]['actor_policy_sha256'], '')
        self.assertTrue(online_rows[0]['actor_policy_window_frozen'])
        self.assertEqual(online_rows[0]['actual_future_actor_policy_versions'], [0] * 5)
        long_horizon = next(row for row in audit_records if row['horizon_s'] == 0.5)
        self.assertEqual(long_horizon['plan_kind'], 'realized_plan')
        self.assertFalse(long_horizon['prediction_available'])
        self.assertFalse(long_horizon['online_authorized'])
        self.assertEqual(long_horizon['model_input_plan'], long_horizon['actual_future_action_plan'])
        restored_history = np.frombuffer(
            zlib.decompress(base64.b64decode(online_rows[0]['history_zlib_base64'])),
            dtype='<f4',
        ).reshape(online_rows[0]['history_shape'])
        self.assertEqual(restored_history.shape, (interface.history_frames, FEATURE_DIM))
        self.assertTrue(np.isfinite(restored_history).all())

    def test_shadow_condition_finish_drops_incomplete_pending_window(self) -> None:
        interface_path = 'RL/data/load_predictor_interface_v3.json'
        interface = LoadPredictorInterface.from_path(interface_path)
        shadow = LoadPredictorShadow(
            interface,
            action_low=[0.0, 0.0],
            action_high=[21_586_451.33303, 0.43],
            interface_sha256=hashlib.sha256(Path(interface_path).read_bytes()).hexdigest(),
        )
        candidate = np.asarray([20_000_000.0, 0.1], dtype=np.float64)
        for index in range(50):
            shadow.observe(
                condition_id=3,
                condition_instance_id=1,
                payload=self._payload(index),
                sent_action_real=candidate,
                projected_candidate_action_real=candidate,
                state_encoding=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                pid_peak_abs_nm=[55_000_000.0, 55_000_000.0, 55_000_000.0],
            )
        stats = shadow.finish_condition()

        self.assertEqual(stats['dropped_pending_count'], 1)
        self.assertEqual(shadow.replay.pending_count, 0)
        self.assertEqual(shadow.history_count, 0)

    def test_shadow_marks_future_window_crossing_actor_version_as_not_frozen(self) -> None:
        interface_path = 'RL/data/load_predictor_interface_v3.json'
        interface = LoadPredictorInterface.from_path(interface_path)
        model = CausalTCNLoadPredictor(input_dim=31, interface=interface, hidden_dim=8)
        shadow = LoadPredictorShadow(
            interface,
            action_low=[0.0, 0.0],
            action_high=[21_586_451.33303, 0.43],
            interface_sha256=hashlib.sha256(Path(interface_path).read_bytes()).hexdigest(),
            model=model,
            device=torch.device('cpu'),
        )
        for index in range(interface.history_frames):
            shadow.observe(
                condition_id=3,
                condition_instance_id=404,
                payload=self._payload(index),
                sent_action_real=[20_000_000.0, 0.05],
                projected_candidate_action_real=[20_100_000.0, 0.051],
                state_encoding=[0.0] * 8,
                pid_peak_abs_nm=[50_000_000.0] * 3,
                optimization_state='above_rated',
                actor_policy_version=0,
            )
        for index in range(interface.history_frames, interface.history_frames + max(interface.horizon_frames)):
            shadow.observe(
                condition_id=3,
                condition_instance_id=404,
                payload=self._payload(index),
                sent_action_real=[20_000_000.0, 0.05],
                projected_candidate_action_real=[20_100_000.0, 0.051],
                state_encoding=[0.0] * 8,
                pid_peak_abs_nm=[50_000_000.0] * 3,
                optimization_state='above_rated',
                actor_policy_version=1,
            )
        audit_records = shadow.drain_audit_records()
        online_rows = [row for row in audit_records if row['online_authorized']]
        self.assertEqual(len(online_rows), 2)
        self.assertTrue(all(not row['actor_policy_window_frozen'] for row in online_rows))


if __name__ == '__main__':
    unittest.main()
