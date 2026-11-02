from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import torch

from RL.utils.constants import OBS_DIM, OBSERVATION_VARIABLES
from RL.utils.operating_state import OperatingPoint, OperatingState, OperatingStateClassifier, OperatingStateConfig
from RL.utils.observation import build_observation_from_frame, build_observation_from_mapping
from RL.utils.safe_residual_control import SafeResidualController


class TestV2ResidualControl(unittest.TestCase):
    def _controller(self) -> SafeResidualController:
        controller = SafeResidualController(
            {'action_low': [13_000_000.0, 0.0], 'action_high': [21_000_000.0, 0.33]},
            {
                'baseline_action': [20_375_412.0, 0.170646548],
                'residual_limit': [2_500_000.0, 0.07],
                'safe_action_low': [13_000_000.0, 0.0],
                'safe_action_high': [21_000_000.0, 0.33],
                'ood_threshold': 3.0,
                'low_power_threshold': 7_000_000.0,
                'low_power_patience': 2,
                'negative_power_patience': 1,
            },
        )
        controller.set_observation_distribution(np.zeros(OBS_DIM, dtype=np.float32), np.ones(OBS_DIM, dtype=np.float32))
        return controller

    def test_observation_uses_raw_state_and_disturbance_variables(self) -> None:
        obs = build_observation_from_mapping({'I_RotorAzimuth': 1.25})
        self.assertEqual(obs.shape, (OBS_DIM,))
        self.assertEqual(float(obs[OBSERVATION_VARIABLES.index('I_RotorAzimuth')]), 1.25)

    def test_observation_wraps_unbounded_rotor_azimuth_to_radians_phase(self) -> None:
        obs = build_observation_from_mapping({'I_RotorAzimuth': 6.0 * np.pi + 0.25})

        self.assertAlmostEqual(float(obs[OBSERVATION_VARIABLES.index('I_RotorAzimuth')]), 0.25, places=5)

    def test_offline_observation_wraps_unbounded_rotor_azimuth_to_radians_phase(self) -> None:
        frame = pd.DataFrame(
            {
                name: [6.0 * np.pi + 0.25] if name == 'I_RotorAzimuth' else [0.0]
                for name in OBSERVATION_VARIABLES
            }
        )

        obs = build_observation_from_frame(frame)

        self.assertAlmostEqual(float(obs[0, OBSERVATION_VARIABLES.index('I_RotorAzimuth')]), 0.25, places=5)

    def test_defaults_use_current_si_safety_box(self) -> None:
        controller = SafeResidualController(
            {'action_low': [13_000_000.0, 0.0], 'action_high': [21_000_000.0, 0.33]},
            {},
        )
        np.testing.assert_allclose(controller.baseline, [20_375_412.0, 0.170646548])
        np.testing.assert_allclose(controller.residual_limit, [2_500_000.0, 0.07])
        np.testing.assert_allclose(controller.safe_low, [13_000_000.0, 0.0])
        np.testing.assert_allclose(controller.safe_high, [21_000_000.0, 0.33])
        self.assertEqual(controller.low_power_threshold, 7_000_000.0)

    def test_ood_falls_back_to_baseline_and_actions_stay_projected(self) -> None:
        controller = self._controller()
        safe_action = controller.action_from_residual(np.array([1.0, 1.0]), np.full(OBS_DIM, 10.0), allow_residual=True)
        self.assertTrue(safe_action.fallback_active)
        np.testing.assert_allclose(safe_action.action_real, np.array([20_375_412.0, 0.170646548], dtype=np.float32))

        in_distribution_action = controller.action_from_residual(
            np.array([100.0, -100.0]), np.zeros(OBS_DIM), allow_residual=True
        )
        self.assertTrue(np.all(in_distribution_action.action_real >= controller.safe_low))
        self.assertTrue(np.all(in_distribution_action.action_real <= controller.safe_high))

    def test_ood_diagnostic_identifies_the_largest_scaled_observation_feature(self) -> None:
        controller = self._controller()
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        yaw_error_index = OBSERVATION_VARIABLES.index('I_YawError')
        obs[yaw_error_index] = 4.0

        safe_action = controller.action_from_residual(np.zeros(2), obs, allow_residual=True)

        self.assertTrue(safe_action.fallback_active)
        self.assertEqual(safe_action.ood_feature_index, yaw_error_index)
        self.assertEqual(safe_action.ood_feature_name, 'I_YawError')
        self.assertAlmostEqual(safe_action.ood_feature_value, 4.0)
        self.assertAlmostEqual(safe_action.ood_feature_median, 0.0)
        self.assertAlmostEqual(safe_action.ood_feature_iqr, 1.0)
        self.assertAlmostEqual(safe_action.ood_score, 4.0)

    def test_low_rotor_speed_fallback_disables_residual_before_global_ood_threshold(self) -> None:
        controller = SafeResidualController(
            {
                'action_low': [0.0, 0.0],
                'action_high': [21_000_000.0, 0.33],
                'action_limiter_owner': 'controller',
                'action_delta_low': [-15_000.0, -0.0003],
                'action_delta_high': [15_000.0, 0.0003],
            },
            {
                'baseline_action': [20_375_412.0, 0.170646548],
                'residual_limit': [2_500_000.0, 0.07],
                'safe_action_low': [0.0, 0.0],
                'safe_action_high': [21_000_000.0, 0.33],
                'ood_threshold': 3.0,
                'low_rotor_speed_fallback_threshold': 0.4,
                'low_rotor_speed_recovery_threshold': 0.5,
                'low_rotor_speed_unload_torque': 0.0,
                'low_power_threshold': None,
            },
        )
        controller.set_observation_distribution(np.zeros(OBS_DIM, dtype=np.float32), np.ones(OBS_DIM, dtype=np.float32))
        rotor_speed_index = OBSERVATION_VARIABLES.index('I_RotorSpeed')
        low_speed_obs = np.zeros(OBS_DIM, dtype=np.float32)
        low_speed_obs[rotor_speed_index] = 0.39
        controller.reset_condition(previous_action_real=np.array([13_000_000.0, 0.05], dtype=np.float32))

        low_speed_action = controller.action_from_residual(np.array([1.0, 0.5]), low_speed_obs, allow_residual=True)

        self.assertTrue(low_speed_action.fallback_active)
        self.assertTrue(low_speed_action.low_rotor_speed_fallback_active)
        self.assertTrue(low_speed_action.low_rotor_speed_torque_unload_active)
        np.testing.assert_allclose(low_speed_action.residual_norm, [0.0, 0.0])
        np.testing.assert_allclose(low_speed_action.policy_action_norm, [1.0, 0.5])
        np.testing.assert_allclose(
            low_speed_action.executed_action_norm,
            controller.residual_target_numpy(
                low_speed_action.action_real,
                low_speed_action.baseline_action_real,
            ),
        )
        np.testing.assert_allclose(
            low_speed_action.pre_safety_action_real,
            [22_875_412.0, 0.20564655],
            rtol=0.0,
            atol=1.0,
        )
        np.testing.assert_allclose(
            low_speed_action.post_safety_action_real,
            [0.0, 0.17064655],
            rtol=0.0,
            atol=1.0e-5,
        )
        self.assertEqual(low_speed_action.low_speed_stage, 1)
        self.assertEqual(low_speed_action.low_speed_scale, 0.0)
        self.assertEqual(low_speed_action.low_speed_reason, 1)
        self.assertEqual(float(low_speed_action.action_real[0]), 0.0)
        self.assertTrue(low_speed_action.action_projection_applied)

        recovered_obs = low_speed_obs.copy()
        recovered_obs[rotor_speed_index] = 0.49
        recovered_action = controller.action_from_residual(np.array([0.1, 0.0]), recovered_obs, allow_residual=True)
        self.assertTrue(recovered_action.fallback_active)
        self.assertTrue(recovered_action.low_rotor_speed_torque_unload_active)
        self.assertEqual(float(recovered_action.action_real[0]), 0.0)

        recovered_obs[rotor_speed_index] = 0.5
        recovered_action = controller.action_from_residual(np.array([0.1, 0.0]), recovered_obs, allow_residual=True)
        self.assertFalse(recovered_action.fallback_active)
        self.assertFalse(recovered_action.low_rotor_speed_fallback_active)
        self.assertFalse(recovered_action.low_rotor_speed_torque_unload_active)

    def test_negative_power_abnormal_state_remains_non_mppt_eligible(self) -> None:
        classifier = OperatingStateClassifier(OperatingStateConfig(
            wind_filter_time_constant_s=0.0,
            region_confirmation_s=0.0,
            running_confirmation_s=0.0,
            stopped_confirmation_s=0.0,
        ))
        classification = classifier.classify(OperatingPoint(
            timestamp_s=0.0,
            wind_speed_mps=8.0,
            rotor_speed_rad_s=0.7,
            real_power_w=-2_000.0,
        ))

        self.assertEqual(classification.optimization_state, OperatingState.NEGATIVE_POWER_ABNORMAL)
        self.assertFalse(classification.normal_mppt_eligible)

    def test_above_rated_initial_pitch_floor_blocks_unfeather_and_allows_feather(self) -> None:
        controller = SafeResidualController(
            {
                'action_low': [0.0, 0.0],
                'action_high': [21_000_000.0, 0.43],
                'action_limiter_owner': 'controller',
                'action_delta_low': [-15_000.0, -0.0003],
                'action_delta_high': [15_000.0, 0.0003],
            },
            {
                'baseline_action': [20_000_000.0, 0.1],
                'residual_limit': [1_000_000.0, 0.1],
                'safe_action_low': [0.0, 0.0],
                'safe_action_high': [21_000_000.0, 0.43],
                'ood_threshold': 100.0,
                'above_rated_initial_pitch_floor_wind_speed': 12.8,
                'low_power_threshold': None,
            },
        )
        controller.set_observation_distribution(np.zeros(OBS_DIM, dtype=np.float32), np.ones(OBS_DIM, dtype=np.float32))
        wind_index = OBSERVATION_VARIABLES.index('I_AverageWindSpeed')
        pitch_index = OBSERVATION_VARIABLES.index('I_PitchPosition1')
        above_rated_obs = np.zeros(OBS_DIM, dtype=np.float32)
        above_rated_obs[wind_index] = 14.0
        above_rated_obs[pitch_index] = 0.181
        controller.reset_condition(
            above_rated_obs,
            previous_action_real=np.array([20_000_000.0, 0.181], dtype=np.float32),
        )

        unfeather_action = controller.action_from_residual(
            np.array([0.0, -1.0]),
            above_rated_obs,
            allow_residual=True,
        )
        self.assertTrue(unfeather_action.above_rated_initial_pitch_floor_active)
        self.assertTrue(unfeather_action.above_rated_initial_pitch_floor_applied)
        self.assertAlmostEqual(unfeather_action.pitch_protection_floor, 0.181, places=6)
        self.assertEqual(unfeather_action.pitch_protection_state, 2)
        np.testing.assert_allclose(
            unfeather_action.pre_safety_action_real,
            [20_000_000.0, 0.0],
            rtol=0.0,
            atol=1.0e-5,
        )
        np.testing.assert_allclose(
            unfeather_action.post_pitch_protection_action_real,
            [20_000_000.0, 0.181],
            rtol=0.0,
            atol=1.0e-5,
        )
        self.assertAlmostEqual(float(unfeather_action.action_real[1]), 0.181, places=6)

        feather_action = controller.action_from_residual(
            np.array([0.0, 1.0]),
            above_rated_obs,
            allow_residual=True,
        )
        self.assertTrue(feather_action.above_rated_initial_pitch_floor_active)
        self.assertFalse(feather_action.above_rated_initial_pitch_floor_applied)
        self.assertGreater(float(feather_action.action_real[1]), 0.181)

        below_rated_obs = above_rated_obs.copy()
        below_rated_obs[wind_index] = 12.7
        controller.reset_condition(
            below_rated_obs,
            previous_action_real=np.array([20_000_000.0, 0.181], dtype=np.float32),
        )
        below_rated_action = controller.action_from_residual(
            np.array([0.0, -1.0]),
            below_rated_obs,
            allow_residual=True,
        )
        self.assertFalse(below_rated_action.above_rated_initial_pitch_floor_active)
        self.assertLess(float(below_rated_action.action_real[1]), 0.181)

    def test_s1_disabled_pitch_floor_allows_rate_limited_unfeather(self) -> None:
        controller = SafeResidualController(
            {
                'action_low': [0.0, 0.0],
                'action_high': [21_000_000.0, 0.43],
                'action_limiter_owner': 'controller',
                'action_delta_low': [-15_000.0, -0.0003],
                'action_delta_high': [15_000.0, 0.0003],
            },
            {
                'baseline_action': [20_000_000.0, 0.1],
                'residual_limit': [1_000_000.0, 0.1],
                'safe_action_low': [0.0, 0.0],
                'safe_action_high': [21_000_000.0, 0.43],
                'ood_threshold': 100.0,
                'pitch_protection_mode': 'disabled',
                'above_rated_initial_pitch_floor_wind_speed': 12.8,
                'low_power_threshold': None,
            },
        )
        controller.set_observation_distribution(np.zeros(OBS_DIM), np.ones(OBS_DIM))
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        obs[OBSERVATION_VARIABLES.index('I_AverageWindSpeed')] = 14.0
        obs[OBSERVATION_VARIABLES.index('I_PitchPosition1')] = 0.181
        controller.reset_condition(
            obs,
            previous_action_real=np.array([20_000_000.0, 0.181], dtype=np.float32),
        )

        action = controller.action_from_residual(np.array([0.0, -1.0]), obs, allow_residual=True)

        self.assertFalse(action.above_rated_initial_pitch_floor_active)
        self.assertFalse(action.above_rated_initial_pitch_floor_applied)
        self.assertEqual(action.pitch_protection_state, 0)
        self.assertAlmostEqual(action.pitch_protection_floor, 0.0)
        self.assertAlmostEqual(float(action.action_real[1]), 0.1807, places=6)

    def test_s2_pid_envelope_pitch_floor_enters_and_releases_dynamically(self) -> None:
        controller = SafeResidualController(
            {
                'action_low': [0.0, 0.0],
                'action_high': [21_000_000.0, 0.43],
                'action_limiter_owner': 'controller',
                'action_delta_low': [-15_000.0, -0.0003],
                'action_delta_high': [15_000.0, 0.0003],
                'bladed_step_duration_ms': 20.0,
            },
            {
                'baseline_action': [20_000_000.0, 0.18],
                'residual_limit': [15_000.0, 0.0003],
                'safe_action_low': [0.0, 0.0],
                'safe_action_high': [21_000_000.0, 0.43],
                'ood_threshold': 100.0,
                'pitch_protection_mode': 'dynamic_pid_envelope',
                'low_speed_protection_mode': 'disabled',
                's2_u1_calibration_path': 'RL/data/s2_u1_pid_envelope_3_to_25.json',
                'low_power_threshold': None,
            },
        )
        controller.set_observation_distribution(np.zeros(OBS_DIM), np.ones(OBS_DIM))
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        obs[OBSERVATION_VARIABLES.index('I_AverageWindSpeed')] = 16.0
        obs[OBSERVATION_VARIABLES.index('I_PitchPosition1')] = 0.18
        obs[OBSERVATION_VARIABLES.index('I_RotorSpeed')] = (
            controller._envelope_value('s2_rotor_enter', 16.0) + 0.01
        )
        controller.reset_condition(
            obs,
            previous_action_real=np.array([20_000_000.0, 0.18]),
        )

        protected = controller.action_from_residual(np.array([0.0, -1.0]), obs, allow_residual=True)

        self.assertEqual(protected.pitch_protection_state, 2)
        self.assertEqual(protected.pitch_protection_reason & 1, 1)
        self.assertAlmostEqual(protected.pitch_protection_floor, 0.18, places=6)
        self.assertAlmostEqual(protected.pitch_protection_lift, 0.0003, places=6)
        self.assertAlmostEqual(float(protected.action_real[1]), 0.18, places=6)

        recovered = obs.copy()
        recovered[OBSERVATION_VARIABLES.index('I_RotorSpeed')] = (
            controller._envelope_value('s2_rotor_release', 16.0) - 0.01
        )
        released = controller.action_from_residual(np.array([0.0, -1.0]), recovered, allow_residual=True)

        self.assertEqual(released.pitch_protection_state, 0)
        self.assertEqual(released.pitch_protection_reason, 0)
        self.assertAlmostEqual(float(released.action_real[1]), 0.1797, places=6)

    def test_u1_pid_envelope_uses_acceleration_hysteresis_and_emergency_prior(self) -> None:
        controller = SafeResidualController(
            {
                'action_low': [0.0, 0.0],
                'action_high': [21_000_000.0, 0.43],
                'action_limiter_owner': 'controller',
                'action_delta_low': [-15_000.0, -0.0003],
                'action_delta_high': [15_000.0, 0.0003],
                'bladed_step_duration_ms': 20.0,
            },
            {
                'baseline_action': [13_000_000.0, 0.05],
                'residual_limit': [15_000.0, 0.0003],
                'safe_action_low': [0.0, 0.0],
                'safe_action_high': [21_000_000.0, 0.43],
                'ood_threshold': 100.0,
                'pitch_protection_mode': 'disabled',
                'low_speed_protection_mode': 'pid_envelope_staged',
                's2_u1_calibration_path': 'RL/data/s2_u1_pid_envelope_3_to_25.json',
                'low_speed_warning_torque_scale': 0.9,
                'low_speed_protection_torque_scale': 0.7,
                'low_rotor_speed_unload_torque': 0.0,
                'low_power_threshold': None,
            },
        )
        controller.set_observation_distribution(np.zeros(OBS_DIM), np.ones(OBS_DIM))
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        obs[OBSERVATION_VARIABLES.index('I_AverageWindSpeed')] = 8.0
        obs[OBSERVATION_VARIABLES.index('I_RealPower')] = 1_000_000.0
        obs[OBSERVATION_VARIABLES.index('I_RotorSpeed')] = (
            controller._envelope_value('u1_recovery_speed', 8.0) + 0.01
        )
        controller.reset_condition(previous_action_real=np.array([13_000_000.0, 0.05]))
        normal = controller.action_from_residual(np.zeros(2), obs, allow_residual=True)
        self.assertEqual(normal.low_speed_stage, 0)

        obs[OBSERVATION_VARIABLES.index('I_RotorSpeed')] = (
            controller._envelope_value('u1_protection_speed', 8.0) - 0.01
        )
        protection = controller.action_from_residual(np.zeros(2), obs, allow_residual=True)
        self.assertEqual(protection.low_speed_stage, 2)
        self.assertEqual(protection.low_speed_reason, 6)
        self.assertAlmostEqual(float(protection.action_real[0]), 12_985_000.0)

        obs[OBSERVATION_VARIABLES.index('I_RotorSpeed')] = (
            controller._envelope_value('u1_recovery_speed', 8.0) - 0.001
        )
        recovery = controller.action_from_residual(np.zeros(2), obs, allow_residual=True)
        self.assertEqual(recovery.low_speed_stage, 4)
        self.assertAlmostEqual(recovery.low_speed_scale, 1.0)
        self.assertEqual(float(recovery.action_real[0]), 13_000_000.0)
        obs[OBSERVATION_VARIABLES.index('I_RotorSpeed')] = (
            controller._envelope_value('u1_recovery_speed', 8.0) + 0.001
        )
        recovered = controller.action_from_residual(np.zeros(2), obs, allow_residual=True)
        self.assertEqual(recovered.low_speed_stage, 0)

        obs[OBSERVATION_VARIABLES.index('I_RotorSpeed')] = 0.39
        emergency = controller.action_from_residual(np.zeros(2), obs, allow_residual=True)
        self.assertEqual(emergency.low_speed_stage, 3)
        self.assertEqual(emergency.low_speed_reason, 7)
        self.assertEqual(float(emergency.action_real[0]), 0.0)

        obs[OBSERVATION_VARIABLES.index('I_RotorSpeed')] = 0.395
        recovering = controller.action_from_residual(np.zeros(2), obs, allow_residual=True)
        self.assertEqual(recovering.low_speed_stage, 4)
        self.assertGreater(float(recovering.action_real[0]), 0.0)

    def test_u1_staged_low_speed_unload_only_bypasses_rate_limit_in_emergency(self) -> None:
        controller = SafeResidualController(
            {
                'action_low': [0.0, 0.0],
                'action_high': [21_000_000.0, 0.43],
                'action_limiter_owner': 'controller',
                'action_delta_low': [-15_000.0, -0.0003],
                'action_delta_high': [15_000.0, 0.0003],
            },
            {
                'baseline_action': [13_000_000.0, 0.05],
                'residual_limit': [1_000_000.0, 0.05],
                'safe_action_low': [0.0, 0.0],
                'safe_action_high': [21_000_000.0, 0.43],
                'ood_threshold': 100.0,
                'low_speed_protection_mode': 'staged',
                'low_speed_warning_threshold': 0.55,
                'low_speed_protection_threshold': 0.45,
                'low_speed_emergency_threshold': 0.3,
                'low_speed_staged_recovery_threshold': 0.6,
                'low_speed_warning_torque_scale': 0.9,
                'low_speed_protection_torque_scale': 0.7,
                'low_rotor_speed_unload_torque': 0.0,
                'low_power_threshold': None,
            },
        )
        controller.set_observation_distribution(np.zeros(OBS_DIM), np.ones(OBS_DIM))
        rotor_speed_index = OBSERVATION_VARIABLES.index('I_RotorSpeed')
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        controller.reset_condition(previous_action_real=np.array([13_000_000.0, 0.05]))

        obs[rotor_speed_index] = 0.5
        warning = controller.action_from_residual(np.zeros(2), obs, allow_residual=True)
        self.assertEqual(warning.low_speed_stage, 1)
        self.assertAlmostEqual(warning.low_speed_scale, 0.9)
        self.assertTrue(warning.fallback_active)
        self.assertEqual(float(warning.action_real[0]), 12_985_000.0)

        obs[rotor_speed_index] = 0.4
        protection = controller.action_from_residual(np.zeros(2), obs, allow_residual=True)
        self.assertEqual(protection.low_speed_stage, 2)
        self.assertAlmostEqual(protection.low_speed_scale, 0.7)
        self.assertEqual(float(protection.action_real[0]), 12_970_000.0)

        obs[rotor_speed_index] = 0.25
        emergency = controller.action_from_residual(np.zeros(2), obs, allow_residual=True)
        self.assertEqual(emergency.low_speed_stage, 3)
        self.assertEqual(emergency.low_speed_scale, 0.0)
        self.assertEqual(float(emergency.action_real[0]), 0.0)

        obs[rotor_speed_index] = 0.58
        recovery = controller.action_from_residual(np.zeros(2), obs, allow_residual=True)
        self.assertEqual(recovery.low_speed_stage, 4)
        self.assertTrue(recovery.fallback_active)
        self.assertAlmostEqual(recovery.low_speed_scale, 1.0)
        self.assertEqual(float(recovery.action_real[0]), 15_000.0)

        obs[rotor_speed_index] = 0.6
        recovered = controller.action_from_residual(np.zeros(2), obs, allow_residual=True)
        self.assertEqual(recovered.low_speed_stage, 0)
        self.assertFalse(recovered.fallback_active)

    def test_staged_low_speed_protection_rejects_missing_or_unordered_thresholds(self) -> None:
        env_config = {'action_low': [0.0, 0.0], 'action_high': [21_000_000.0, 0.43]}
        control_config = {
            'safe_action_low': [0.0, 0.0],
            'safe_action_high': [21_000_000.0, 0.43],
            'low_speed_protection_mode': 'staged',
        }
        with self.assertRaisesRegex(ValueError, 'requires warning'):
            SafeResidualController(env_config, control_config)

        control_config.update(
            {
                'low_speed_warning_threshold': 0.4,
                'low_speed_protection_threshold': 0.45,
                'low_speed_emergency_threshold': 0.3,
                'low_speed_staged_recovery_threshold': 0.6,
            }
        )
        with self.assertRaisesRegex(ValueError, 'emergency < protection < warning < recovery'):
            SafeResidualController(env_config, control_config)

    def test_residual_target_is_relative_to_physical_baseline(self) -> None:
        controller = self._controller()
        actions = torch.tensor([[20_375_412.0, 0.170646548], [22_875_412.0, 0.240646548]])
        target = controller.residual_target_tensor(actions)
        self.assertTrue(torch.allclose(target, torch.tensor([[0.0, 0.0], [1.0, 1.0]]), atol=1e-5))

    def test_formal_actor_residual_scale_is_applied_to_targets_and_actions(self) -> None:
        controller = SafeResidualController(
            {
                'action_low': [0.0, 0.0],
                'action_high': [21_000_000.0, 0.43],
                'action_limiter_owner': 'controller',
                'action_delta_low': [-15_000.0, -0.0003],
                'action_delta_high': [15_000.0, 0.0003],
            },
            {
                'baseline_action': [20_000_000.0, 0.1],
                'action_representation': 'residual',
                'actor_residual_target_scale': 0.25,
                'residual_limit': [1_000_000.0, 0.01],
                'safe_action_low': [0.0, 0.0],
                'safe_action_high': [21_000_000.0, 0.43],
                'low_speed_protection_mode': 'disabled',
                'pitch_protection_mode': 'disabled',
            },
        )
        actions = torch.tensor([[21_000_000.0, 0.11]])
        target = controller.residual_target_tensor(actions)
        self.assertTrue(torch.allclose(target, torch.tensor([[0.25, 0.25]]), atol=1e-5))
        real_action = controller.action_real_tensor(torch.tensor([[1.0, 1.0]]))
        self.assertTrue(torch.allclose(real_action, torch.tensor([[20_250_000.0, 0.1025]]), atol=1e-5))

    def test_residual_replay_action_reflects_safety_modified_action(self) -> None:
        controller = SafeResidualController(
            {
                'action_low': [0.0, 0.0],
                'action_high': [21_000_000.0, 0.43],
                'action_delta_low': [-15_000.0, -0.0003],
                'action_delta_high': [15_000.0, 0.0003],
                'action_limiter_owner': 'controller',
            },
            {
                'baseline_action': [20_000_000.0, 0.1],
                'action_representation': 'residual',
                'residual_limit': [1_000_000.0, 0.01],
                'safe_action_low': [0.0, 0.0],
                'safe_action_high': [21_000_000.0, 0.43],
                'low_speed_protection_mode': 'disabled',
                'pitch_protection_mode': 'disabled',
            },
        )
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        obs[OBSERVATION_VARIABLES.index('I_RotorSpeed')] = 0.8
        controller.reset_condition(obs, previous_action_real=np.array([20_000_000.0, 0.1]))

        action = controller.action_from_residual(
            np.array([1.0, 1.0], dtype=np.float32),
            obs,
            allow_residual=True,
        )

        np.testing.assert_allclose(action.action_real, [20_015_000.0, 0.1003], rtol=0.0, atol=1e-6)
        np.testing.assert_allclose(action.executed_action_norm, [0.015, 0.03], rtol=0.0, atol=1e-6)
        np.testing.assert_allclose(action.residual_norm, [1.0, 1.0], rtol=0.0, atol=0.0)
        np.testing.assert_allclose(action.policy_action_norm, [1.0, 1.0], rtol=0.0, atol=0.0)

    def test_negative_power_requires_configured_consecutive_frames(self) -> None:
        controller = SafeResidualController(
            {'action_low': [13_000_000.0, 0.0], 'action_high': [21_000_000.0, 0.33]},
            {
                'baseline_action': [20_375_412.0, 0.170646548],
                'residual_limit': [2_500_000.0, 0.07],
                'safe_action_low': [13_000_000.0, 0.0],
                'safe_action_high': [21_000_000.0, 0.33],
                'low_power_threshold': None,
                'negative_power_patience': 3,
            },
        )

        self.assertFalse(controller.observe_step(-1.0))
        self.assertFalse(controller.observe_step(-1.0))
        self.assertTrue(controller.observe_step(-1.0))
        self.assertEqual(controller.negative_power_steps, 3)

    def test_control_fingerprint_changes_with_reward_contract(self) -> None:
        control_config = {
            'baseline_action': [20_375_412.0, 0.170646548],
            'residual_limit': [2_500_000.0, 0.07],
            'safe_action_low': [13_000_000.0, 0.0],
            'safe_action_high': [21_000_000.0, 0.33],
        }
        base_env = {
            'action_low': [13_000_000.0, 0.0],
            'action_high': [21_000_000.0, 0.33],
            'rated_power': 15_000_000.0,
            'bladed_step_duration_ms': 20.0,
            'episode_length': 512,
            'reward': {'power_reward_mode': 'available_fraction'},
        }
        energy_env = {
            **base_env,
            'reward': {'power_reward_mode': 'energy_increment', 'energy_reference_j': 153_600_000.0},
        }

        base_controller = SafeResidualController(base_env, control_config)
        energy_controller = SafeResidualController(energy_env, control_config)

        self.assertNotEqual(base_controller.control_fingerprint, energy_controller.control_fingerprint)

    def test_actor_contract_is_stable_when_only_online_reward_contract_changes(self) -> None:
        control_config = {
            'baseline_action': [20_375_412.0, 0.170646548],
            'residual_limit': [2_500_000.0, 0.07],
            'safe_action_low': [13_000_000.0, 0.0],
            'safe_action_high': [21_000_000.0, 0.33],
        }
        base_env = {
            'action_low': [13_000_000.0, 0.0],
            'action_high': [21_000_000.0, 0.33],
            'rated_power': 15_000_000.0,
            'bladed_step_duration_ms': 20.0,
            'episode_length': 512,
            'reward': {'power_reward_mode': 'available_fraction'},
        }
        queue_env = {
            **base_env,
            'udp_protocol_version': 'bladed_udp_19_si_my_queue_v1',
            'require_condition_metadata': True,
            'queue_lifecycle_enabled': True,
            'rollout_segment_steps': 512,
            'reward': {'power_reward_mode': 'energy_increment', 'energy_reference_j': 153_600_000.0},
        }

        base_controller = SafeResidualController(base_env, control_config)
        queue_controller = SafeResidualController(queue_env, control_config)

        self.assertEqual(base_controller.actor_control_fingerprint, queue_controller.actor_control_fingerprint)
        self.assertNotEqual(base_controller.online_training_fingerprint, queue_controller.online_training_fingerprint)

    def test_negative_or_sustained_low_power_stops_episode(self) -> None:
        controller = self._controller()
        self.assertFalse(controller.observe_step(6_000_000.0))
        self.assertTrue(controller.observe_step(6_000_000.0))
        controller.reset_episode()
        self.assertTrue(controller.observe_step(-1.0))

    def test_controller_owned_rate_projection_matches_configured_action_limits(self) -> None:
        controller = SafeResidualController(
            {
                'action_low': [13_000_000.0, 0.0],
                'action_high': [21_000_000.0, 0.33],
                'action_limiter_owner': 'controller',
                'action_delta_low': [-15_000.0, -0.0003],
                'action_delta_high': [15_000.0, 0.0003],
            },
            {
                'baseline_action': [20_000_000.0, 0.05],
                'residual_limit': [1_000_000.0, 0.05],
                'safe_action_low': [13_000_000.0, 0.0],
                'safe_action_high': [21_000_000.0, 0.33],
                'ood_threshold': 100.0,
            },
        )
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        controller.reset_condition(obs)

        safe_action = controller.action_from_residual(np.array([1.0, 1.0]), obs, allow_residual=True)

        np.testing.assert_allclose(safe_action.action_real, [20_015_000.0, 0.0503], rtol=0.0, atol=1e-5)
        np.testing.assert_allclose(safe_action.baseline_action_real, [20_000_000.0, 0.05], rtol=0.0, atol=1e-5)
        np.testing.assert_allclose(safe_action.policy_delta_real, [1_000_000.0, 0.05], rtol=0.0, atol=1e-5)
        np.testing.assert_allclose(safe_action.pre_safety_action_real, [21_000_000.0, 0.1], rtol=0.0, atol=1e-5)
        np.testing.assert_allclose(safe_action.post_safety_action_real, [21_000_000.0, 0.1], rtol=0.0, atol=1e-5)
        np.testing.assert_allclose(safe_action.projected_action_real, [21_000_000.0, 0.1], rtol=0.0, atol=1e-5)
        np.testing.assert_allclose(safe_action.rate_limited_action_real, safe_action.action_real, rtol=0.0, atol=0.0)
        self.assertEqual(safe_action.pitch_protection_state, 0)
        self.assertEqual(safe_action.pitch_protection_floor, 0.0)
        self.assertEqual(safe_action.low_speed_stage, 0)
        self.assertEqual(safe_action.low_speed_scale, 1.0)
        self.assertEqual(safe_action.low_speed_reason, 0)
        self.assertTrue(safe_action.action_projection_applied)
        self.assertTrue(safe_action.action_rate_limited)

    def test_controller_owned_rate_projection_uses_external_command_at_condition_start(self) -> None:
        controller = SafeResidualController(
            {
                'action_low': [0.0, 0.0],
                'action_high': [21_000_000.0, 0.43],
                'action_limiter_owner': 'controller',
                'action_delta_low': [-15_000.0, -0.0003],
                'action_delta_high': [15_000.0, 0.0003],
            },
            {
                'baseline_action': [16_950_580.0, 0.0],
                'residual_limit': [1_000_000.0, 0.05],
                'safe_action_low': [0.0, 0.0],
                'safe_action_high': [21_000_000.0, 0.43],
                'ood_threshold': 100.0,
            },
        )
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        controller.reset_condition(
            obs,
            previous_action_real=np.array([20_400_000.0, 0.049461], dtype=np.float32),
        )

        safe_action = controller.action_from_residual(np.zeros(2), obs, allow_residual=True)

        np.testing.assert_allclose(safe_action.action_real, [20_385_000.0, 0.049161], rtol=0.0, atol=1e-5)
        self.assertTrue(safe_action.action_projection_applied)
        self.assertTrue(safe_action.action_rate_limited)

    def test_fallback_action_override_uses_the_supplied_absolute_action(self) -> None:
        controller = SafeResidualController(
            {
                'action_low': [0.0, 0.0],
                'action_high': [21_000_000.0, 0.43],
                'action_limiter_owner': 'controller',
                'action_delta_low': [-15_000.0, -0.0003],
                'action_delta_high': [15_000.0, 0.0003],
            },
            {
                'baseline_action': [16_950_580.0, 0.0],
                'residual_limit': [1_000_000.0, 0.05],
                'safe_action_low': [0.0, 0.0],
                'safe_action_high': [21_000_000.0, 0.43],
                'ood_threshold': 100.0,
            },
        )
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        controller.reset_condition(
            obs,
            previous_action_real=np.array([20_400_000.0, 0.049461], dtype=np.float32),
        )

        safe_action = controller.action_from_residual(
            np.array([1.0, 1.0]),
            obs,
            allow_residual=False,
            fallback_action_real=np.array([21_000_000.0, 0.0], dtype=np.float32),
        )

        self.assertTrue(safe_action.fallback_active)
        np.testing.assert_allclose(safe_action.residual_norm, [0.0, 0.0], rtol=0.0, atol=1e-6)
        np.testing.assert_allclose(safe_action.action_real, [20_415_000.0, 0.049161], rtol=0.0, atol=1e-5)
        self.assertTrue(safe_action.action_rate_limited)


if __name__ == '__main__':
    unittest.main()