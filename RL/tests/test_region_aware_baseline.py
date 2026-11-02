from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from RL.utils.constants import ACTION_VARIABLES, DATA_SCHEMA_VERSION, OBS_DIM, OBSERVATION_VARIABLES
from RL.evaluate_region_aware_baseline import evaluate_region_aware_baseline
from RL.utils.operating_state import ControlRegion, OPERATING_STATE_CONTRACT_VERSION, OperatingStateConfig
from RL.utils.region_aware_baseline import (
    REGION_AWARE_BASELINE_SCHEMA_VERSION,
    RegionAwareJointBaseline,
)
from RL.utils.safe_residual_control import SafeResidualController


class TestRegionAwareJointBaseline(unittest.TestCase):
    @staticmethod
    def _document() -> dict[str, object]:
        feature_names = [
            'filtered_wind_speed_mps',
            'rotor_speed_rad_s',
            'generator_speed_rad_s',
            'real_power_w',
            'previous_torque_nm',
            'previous_pitch_rad',
        ]
        feature_mean = [12.0, 0.8, 0.8, 15_000_000.0, 19_000_000.0, 0.1]
        feature_scale = [2.0, 0.1, 0.1, 1_000_000.0, 1_000_000.0, 0.1]

        def model(intercept: list[float]) -> dict[str, object]:
            return {
                'relation': 'joint_next_action_delta',
                'sample_count': 100,
                'feature_mean': feature_mean,
                'feature_scale': feature_scale,
                'intercept': intercept,
                'coefficient_matrix': [
                    [500.0, 0.00001],
                    [1_000.0, -0.00002],
                    [2_000.0, 0.00001],
                    [3_000.0, 0.00002],
                    [4_000.0, -0.00001],
                    [5_000.0, 0.00004],
                ],
                'action_delta_low': [-15_000.0, -0.0003],
                'action_delta_high': [15_000.0, 0.0003],
                'joint_residual_covariance': [[1.0, 0.25], [0.25, 1.0]],
            }

        return {
            'schema_version': REGION_AWARE_BASELINE_SCHEMA_VERSION,
            'metadata': {
                'data_schema_version': DATA_SCHEMA_VERSION,
                'source_role': 'raw_pid_calibration',
                'sampling_strategy': 'all_valid_causal_transitions',
                'input_manifest_sha256': 'a' * 64,
                'action_source': list(ACTION_VARIABLES),
                'action_semantics': 'recorded_pid_demand',
                'operating_state_contract': OperatingStateConfig().to_metadata(),
            },
            'control': {
                'region_aware_joint_baseline': {
                    'feature_names': feature_names,
                    'target': 'next_realized_action_delta',
                    'action_units': ['N m', 'rad'],
                    'action_low': [0.0, 0.0],
                    'action_high': [21_000_000.0, 0.43],
                    'causal': True,
                    'future_feature_lookahead_steps': 0,
                    'models': {
                        'below_rated': model([1_000.0, 0.00001]),
                        'transition': model([2_000.0, 0.00002]),
                        'above_rated': model([3_000.0, 0.00003]),
                    },
                    'transition_blending': {
                        'method': 'smoothstep_with_hysteretic_region',
                        'transition_exit_wind_mps': 11.2,
                        'transition_enter_wind_mps': 11.8,
                        'above_rated_exit_wind_mps': 13.2,
                        'above_rated_enter_wind_mps': 13.8,
                    },
                }
            },
        }

    @staticmethod
    def _observation() -> np.ndarray:
        observation = np.zeros(OBS_DIM, dtype=np.float32)
        values = {
            'I_AverageWindSpeed': 8.0,
            'I_RotorSpeed': 0.75,
            'I_GeneratorSpeed': 0.77,
            'I_RealPower': 12_000_000.0,
            'I_ElectricalTorque': 18_000_000.0,
            'I_PitchPosition1': 0.05,
        }
        for name, value in values.items():
            observation[OBSERVATION_VARIABLES.index(name)] = value
        return observation

    def test_schema_and_operating_state_contract_are_strict(self) -> None:
        baseline = RegionAwareJointBaseline.from_document(self._document(), step_duration_s=0.02)

        self.assertEqual(baseline.schema_version, REGION_AWARE_BASELINE_SCHEMA_VERSION)
        self.assertEqual(
            baseline.operating_state_contract_version,
            OPERATING_STATE_CONTRACT_VERSION,
        )
        self.assertEqual(baseline.feature_names, (
            'filtered_wind_speed_mps',
            'rotor_speed_rad_s',
            'generator_speed_rad_s',
            'real_power_w',
            'previous_torque_nm',
            'previous_pitch_rad',
        ))

    def test_prediction_jointly_uses_full_state_and_previous_action(self) -> None:
        baseline = RegionAwareJointBaseline.from_document(self._document(), step_duration_s=0.02)
        observation = self._observation()
        previous_action = np.array([18_000_000.0, 0.05], dtype=np.float32)
        baseline.reset()

        original = baseline.predict(observation, previous_action_real=previous_action)

        changed_observation = observation.copy()
        changed_observation[OBSERVATION_VARIABLES.index('I_GeneratorSpeed')] += 0.1
        changed_observation[OBSERVATION_VARIABLES.index('I_RealPower')] += 1_000_000.0
        baseline.reset()
        state_changed = baseline.predict(
            changed_observation,
            previous_action_real=previous_action,
        )

        changed_history = previous_action.copy()
        changed_history[1] += 0.1
        baseline.reset()
        history_changed = baseline.predict(
            observation,
            previous_action_real=changed_history,
        )

        self.assertEqual(original.shape, (2,))
        self.assertTrue(np.isfinite(original).all())
        self.assertNotEqual(float(original[0]), float(state_changed[0]))
        self.assertNotEqual(float(original[1]), float(state_changed[1]))
        self.assertNotEqual(float(original[0]), float(history_changed[0]))
        self.assertNotEqual(float(original[1]), float(history_changed[1]))
        np.testing.assert_array_less(
            np.abs(original - previous_action),
            np.array([15_000.01, 0.00030001], dtype=np.float32),
        )

    def test_smoothstep_weights_are_continuous_across_hysteretic_region_switches(self) -> None:
        baseline = RegionAwareJointBaseline.from_document(self._document(), step_duration_s=0.02)

        below_side = baseline._region_weights(ControlRegion.BELOW_RATED, 11.5)
        transition_low_side = baseline._region_weights(ControlRegion.TRANSITION, 11.5)
        transition_high_side = baseline._region_weights(ControlRegion.TRANSITION, 13.5)
        above_side = baseline._region_weights(ControlRegion.ABOVE_RATED, 13.5)

        self.assertEqual(below_side, transition_low_side)
        self.assertEqual(transition_high_side, above_side)
        self.assertAlmostEqual(sum(below_side.values()), 1.0)
        self.assertAlmostEqual(sum(above_side.values()), 1.0)

    def test_safe_controller_uses_one_joint_baseline_for_fallback(self) -> None:
        with tempfile.TemporaryDirectory(prefix='region_baseline_controller_') as tmpdir:
            artifact_path = Path(tmpdir) / 'region_baseline.json'
            artifact_path.write_text(json.dumps(self._document()), encoding='utf-8')
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
                    'baseline_action': [18_000_000.0, 0.05],
                    'action_representation': 'residual',
                    'residual_limit': [15_000.0, 0.0003],
                    'safe_action_low': [0.0, 0.0],
                    'safe_action_high': [21_000_000.0, 0.43],
                    'region_aware_baseline_path': str(artifact_path),
                    'low_power_threshold': None,
                },
            )
            observation = self._observation()
            previous_action = np.array([18_000_000.0, 0.05], dtype=np.float32)
            controller.reset_condition(observation, previous_action_real=previous_action)

            baseline_action = controller.baseline_action_for_obs(observation)
            safe_action = controller.action_from_residual(
                np.ones(2, dtype=np.float32),
                observation,
                allow_residual=False,
                baseline_action_real=baseline_action,
            )

            self.assertIsNotNone(controller.region_aware_baseline)
            self.assertTrue(controller.checkpoint_contract_required())
            self.assertFalse(controller.checkpoint_contract_required(baseline_only=True))
            np.testing.assert_allclose(safe_action.baseline_action_real, baseline_action)
            np.testing.assert_allclose(safe_action.action_real, baseline_action)
            np.testing.assert_array_less(
                np.abs(baseline_action - previous_action),
                np.array([15_000.01, 0.00030001], dtype=np.float32),
            )

    def test_emergency_zero_torque_does_not_poison_recovery_baseline_history(self) -> None:
        controller = SafeResidualController(
            {
                'action_low': [0.0, 0.0],
                'action_high': [21_586_451.33303, 0.43],
                'action_limiter_owner': 'controller',
                'action_delta_low': [-90_000.0, -0.000698],
                'action_delta_high': [90_000.0, 0.000698],
                'bladed_step_duration_ms': 20.0,
            },
            {
                'baseline_action': [18_000_000.0, 0.05],
                'action_representation': 'residual',
                'residual_limit': [15_000.0, 0.0003],
                'safe_action_low': [0.0, 0.0],
                'safe_action_high': [21_586_451.33303, 0.43],
                'region_aware_baseline_path': 'RL/data/region_aware_joint_baseline_state20_3_to_25_v3.json',
                'low_speed_protection_mode': 'pid_envelope_staged',
                's2_u1_calibration_path': 'RL/data/s2_u1_pid_envelope_3_to_25.json',
                'low_rotor_speed_fallback_threshold': 0.4,
                'low_rotor_speed_recovery_threshold': 0.5,
                'low_power_threshold': None,
            },
        )
        controller.set_observation_distribution(np.zeros(OBS_DIM), np.ones(OBS_DIM))
        observation = self._observation()
        rotor_index = OBSERVATION_VARIABLES.index('I_RotorSpeed')
        power_index = OBSERVATION_VARIABLES.index('I_RealPower')
        controller.reset_condition(observation, previous_action_real=np.array([18_000_000.0, 0.05]))

        observation[rotor_index] = 0.39
        observation[power_index] = 1_000_000.0
        emergency = controller.action_from_residual(np.zeros(2), observation, allow_residual=True)
        self.assertEqual(emergency.low_speed_stage, 3)
        self.assertEqual(float(emergency.action_real[0]), 0.0)
        self.assertLessEqual(
            float(emergency.baseline_action_real[1]),
            0.0311,
        )

        observation[rotor_index] = 0.45
        recovery = controller.action_from_residual(np.zeros(2), observation, allow_residual=True)
        self.assertEqual(recovery.low_speed_stage, 4)
        self.assertGreater(float(recovery.baseline_action_real[0]), 0.0)
        self.assertGreater(float(recovery.action_real[0]), 0.0)

        observation[rotor_index] = 0.41
        stage_two = controller.action_from_residual(np.zeros(2), observation, allow_residual=True)
        self.assertEqual(stage_two.low_speed_stage, 2)
        self.assertGreater(float(stage_two.baseline_action_real[0]), 0.0)

        observation[rotor_index] = 0.50
        recovery_again = controller.action_from_residual(np.zeros(2), observation, allow_residual=True)
        self.assertEqual(recovery_again.low_speed_stage, 4)
        self.assertGreater(float(recovery_again.baseline_action_real[0]), 0.0)

        anchored_baseline = float(recovery_again.baseline_action_real[0])
        for _ in range(100):
            repeated_recovery = controller.action_from_residual(
                np.zeros(2),
                observation,
                allow_residual=True,
            )
            self.assertEqual(repeated_recovery.low_speed_stage, 4)
        self.assertGreater(float(repeated_recovery.baseline_action_real[0]), anchored_baseline * 0.9)

        observation[rotor_index] = 0.497
        stage_one = controller.action_from_residual(np.zeros(2), observation, allow_residual=True)
        self.assertEqual(stage_one.low_speed_stage, 1)
        self.assertGreater(float(stage_one.baseline_action_real[0]), 0.0)

        observation[rotor_index] = 0.50
        recovery_after_warning = controller.action_from_residual(np.zeros(2), observation, allow_residual=True)
        self.assertEqual(recovery_after_warning.low_speed_stage, 4)
        self.assertGreater(float(recovery_after_warning.baseline_action_real[0]), 0.0)

    def test_generating_fallback_does_not_ramp_torque_to_zero(self) -> None:
        controller = SafeResidualController(
            {
                'action_low': [0.0, 0.0],
                'action_high': [21_586_451.33303, 0.43],
                'action_limiter_owner': 'controller',
                'action_delta_low': [-90_000.0, -0.000698],
                'action_delta_high': [90_000.0, 0.000698],
                'bladed_step_duration_ms': 20.0,
            },
            {
                'baseline_action': [18_000_000.0, 0.05],
                'action_representation': 'residual',
                'residual_limit': [15_000.0, 0.0003],
                'safe_action_low': [0.0, 0.0],
                'safe_action_high': [21_586_451.33303, 0.43],
                'region_aware_baseline_path': 'RL/data/region_aware_joint_baseline_state20_3_to_25_v3.json',
                'low_speed_protection_mode': 'pid_envelope_staged',
                's2_u1_calibration_path': 'RL/data/s2_u1_pid_envelope_3_to_25.json',
                'low_rotor_speed_fallback_threshold': 0.4,
                'low_rotor_speed_recovery_threshold': 0.5,
                'low_power_threshold': None,
            },
        )
        controller.set_observation_distribution(np.zeros(OBS_DIM), np.ones(OBS_DIM))
        observation = self._observation()
        for name, value in {
            'I_AverageWindSpeed': 12.7,
            'I_RotorSpeed': 0.77,
            'I_GeneratorSpeed': 0.77,
            'I_RealPower': 50.0,
            'I_ElectricalTorque': 7_000.0,
            'I_PitchPosition1': 0.24,
        }.items():
            observation[OBSERVATION_VARIABLES.index(name)] = value
        controller.reset_condition(observation, previous_action_real=np.array([7_000.0, 0.24]))

        action = controller.action_from_residual(np.zeros(2), observation, allow_residual=False)

        self.assertTrue(action.fallback_active)
        self.assertTrue(action.fallback_generating_torque_floor_applied)
        self.assertGreater(float(action.baseline_action_real[0]), 1_000_000.0)
        self.assertGreater(float(action.action_real[0]), 7_000.0)

    def test_region_baseline_requires_residual_representation(self) -> None:
        with tempfile.TemporaryDirectory(prefix='region_baseline_controller_') as tmpdir:
            artifact_path = Path(tmpdir) / 'region_baseline.json'
            artifact_path.write_text(json.dumps(self._document()), encoding='utf-8')
            env_config = {
                'action_low': [0.0, 0.0],
                'action_high': [21_586_451.33303, 0.43],
                'action_delta_low': [-90_000.0, -0.000698],
                'action_delta_high': [90_000.0, 0.000698],
                'bladed_step_duration_ms': 20.0,
            }
            base_control = {
                'baseline_action': [18_000_000.0, 0.05],
                'residual_limit': [15_000.0, 0.0003],
                'safe_action_low': [0.0, 0.0],
                'safe_action_high': [21_586_451.33303, 0.43],
                'region_aware_baseline_path': str(artifact_path),
            }

            with self.assertRaisesRegex(ValueError, "action_representation must be 'residual'"):
                SafeResidualController(
                    env_config,
                    {
                        **base_control,
                        'action_representation': 'unsupported',
                    },
                )
            residual_controller = SafeResidualController(
                env_config,
                {**base_control, 'action_representation': 'residual'},
            )
            self.assertIsNotNone(residual_controller.region_aware_baseline)

    def test_holdout_evaluation_is_explicitly_diagnostic_only(self) -> None:
        with tempfile.TemporaryDirectory(prefix='region_baseline_holdout_') as tmpdir:
            root = Path(tmpdir)
            artifact_path = root / 'region_baseline.json'
            artifact_path.write_text(json.dumps(self._document()), encoding='utf-8')
            holdout_dir = root / 'raw_infer'
            holdout_dir.mkdir()
            path = holdout_dir / 'data_windspeed8.csv'
            fieldnames = ['Time', *OBSERVATION_VARIABLES, *ACTION_VARIABLES]
            with path.open('w', encoding='utf-8', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for index, timestamp in enumerate((0.0, 0.02, 0.04)):
                    row = {name: 0.0 for name in fieldnames}
                    row.update({
                        'Time': timestamp,
                        'I_RotorSpeed': 0.75,
                        'I_ElectricalTorque': 18_000_000.0 + index * 1_000.0,
                        'I_GeneratorSpeed': 0.77,
                        'I_RealPower': 12_000_000.0,
                        'I_PitchPosition1': 0.05 + index * 0.00001,
                        'I_AverageWindSpeed': 8.0,
                        ACTION_VARIABLES[0]: 18_000_000.0 + index * 1_000.0,
                        ACTION_VARIABLES[1]: 0.05 + index * 0.00001,
                    })
                    writer.writerow(row)

            report = evaluate_region_aware_baseline(
                artifact_path,
                holdout_dir,
                step_duration_s=0.02,
            )

            self.assertEqual(report['validation_status'], 'diagnostic_only')
            self.assertEqual(
                report['physical_gate_status'],
                'not_evaluated_requires_closed_loop_bladed',
            )
            self.assertEqual(report['condition_ids'], [8])
            self.assertEqual(report['aggregate']['sample_count'], 2)
            self.assertEqual(len(report['aggregate']['action_mae']), 2)


if __name__ == '__main__':
    unittest.main()