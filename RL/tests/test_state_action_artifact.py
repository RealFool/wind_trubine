from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import torch

from RL.data.offline_dataset import OfflineBCDataset
from RL.envs.wind_turbine_env import WindTurbineEnv
from RL.tests.test_fake_bladed_integration import FakeBladedUdpServer, _pick_free_udp_port
from RL.utils.safe_residual_control import SafeResidualController
from RL.utils.constants import ACTION_VARIABLES, RAW_OBSERVATION_VARIABLES
from RL.utils.state_action_artifact import (
    STATE_AWARE_CONTROL_OBS_DIM,
    StateActionEnvelopeArtifact,
    append_state_regime_observation,
    build_state_action_artifact,
)
from RL.utils.operating_state import OperatingPoint


class TestStateActionArtifact(unittest.TestCase):
    @staticmethod
    def _write_raw_files(raw_dir: Path) -> None:
        raw_dir.mkdir()
        fields = ['Time', *RAW_OBSERVATION_VARIABLES, *ACTION_VARIABLES]
        for condition_id in range(3, 26):
            wind = float(condition_id)
            with (raw_dir / f'data_windspeed{condition_id}.csv').open('w', encoding='utf-8', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows([
                    {
                        'Time': 0.0,
                        'I_AverageWindSpeed': wind,
                        'I_RotorSpeed': 0.2,
                        'I_RotorAzimuth': 0.0,
                        'I_YawError': 0.0,
                        'I_ElectricalTorque': 0.0,
                        'I_GeneratorSpeed': 0.2,
                        'I_RealPower': 0.0,
                        'I_PitchPosition1': 0.02,
                        'I_NacelleForeAftAcceleration': 0.0,
                        'I_NacelleSideSideAcceleration': 0.0,
                        'O_PcsTorqueDemand': 21_586_452.0,
                        'O_PitchPositionDemand1': 0.02,
                    },
                    {
                        'Time': 1.0,
                        'I_AverageWindSpeed': wind,
                        'I_RotorSpeed': 0.7,
                        'I_RotorAzimuth': 0.2,
                        'I_YawError': 0.0,
                        'I_ElectricalTorque': 20_000_000.0,
                        'I_GeneratorSpeed': 0.7,
                        'I_RealPower': 2_000_000.0,
                        'I_PitchPosition1': 0.03,
                        'I_NacelleForeAftAcceleration': 0.0,
                        'I_NacelleSideSideAcceleration': 0.0,
                        'O_PcsTorqueDemand': 20_000_000.0,
                        'O_PitchPositionDemand1': 0.03,
                    },
                    {
                        'Time': 600.0,
                        'I_AverageWindSpeed': wind,
                        'I_RotorSpeed': 0.7,
                        'I_RotorAzimuth': 0.4,
                        'I_YawError': 0.0,
                        'I_ElectricalTorque': 20_015_000.0,
                        'I_GeneratorSpeed': 0.7,
                        'I_RealPower': 2_100_000.0,
                        'I_PitchPosition1': 0.04,
                        'I_NacelleForeAftAcceleration': 0.0,
                        'I_NacelleSideSideAcceleration': 0.0,
                        'O_PcsTorqueDemand': 20_015_000.0,
                        'O_PitchPositionDemand1': 0.04,
                    },
                ])

    def test_artifact_has_frozen_calibration_causal_schema_and_effective_cap(self) -> None:
        with tempfile.TemporaryDirectory(prefix='state_action_artifact_') as tmpdir:
            root = Path(tmpdir)
            raw_dir = root / 'raw'
            self._write_raw_files(raw_dir)
            document = build_state_action_artifact(raw_dir)
            path = root / 'state_action_artifact.json'
            path.write_text(json.dumps(document), encoding='utf-8')
            artifact = StateActionEnvelopeArtifact.from_path(path)

        self.assertEqual(document['observation_schema']['width'], STATE_AWARE_CONTROL_OBS_DIM)
        self.assertTrue(document['observation_schema']['causal'])
        self.assertEqual(document['metadata']['operating_state_contract']['thresholds']['wind_filter_time_constant_s'], 2.0)
        self.assertEqual(document['metadata']['operating_state_contract']['thresholds']['transition_enter_wind_mps'], 11.8)
        self.assertEqual(document['metadata']['effective_pid_demand_high_nm'], 21_586_451.33303)
        self.assertEqual(
            sum(item['raw_recorded_demand_above_engineering_cap_count'] for item in document['metadata']['raw_source_files']),
            23,
        )
        classifier = artifact.classifier()
        classification = classifier.classify(OperatingPoint(0.0, 12.0, 0.7, 2_000_000.0, 0.03, 20_000_000.0, 0.03))
        observation = append_state_regime_observation(np.zeros(12, dtype=np.float32), classification)
        self.assertEqual(observation.shape, (STATE_AWARE_CONTROL_OBS_DIM,))
        self.assertEqual(float(np.sum(observation[-8:-3])), 1.0)
        self.assertEqual(float(np.sum(observation[-3:])), 1.0)
        self.assertTrue(np.all(artifact.residual_limit_for(classification) <= [20_000.0, 0.00035]))

    def test_artifact_rejects_envelope_above_engineering_rate_limit(self) -> None:
        with tempfile.TemporaryDirectory(prefix='state_action_artifact_invalid_') as tmpdir:
            root = Path(tmpdir)
            raw_dir = root / 'raw'
            self._write_raw_files(raw_dir)
            document = build_state_action_artifact(raw_dir)
            document['soft_residual_envelopes']['states']['above_rated']['residual_limit_abs'][0] = 90_001.0
            with self.assertRaisesRegex(ValueError, 'exceeds engineering rate limit'):
                StateActionEnvelopeArtifact.from_document(document)

    def test_offline_dataset_binds_state_aware_schema_to_artifact(self) -> None:
        with tempfile.TemporaryDirectory(prefix='state_action_dataset_') as tmpdir:
            root = Path(tmpdir)
            raw_dir = root / 'raw'
            self._write_raw_files(raw_dir)
            artifact_path = root / 'state_action_artifact.json'
            artifact_path.write_text(json.dumps(build_state_action_artifact(raw_dir)), encoding='utf-8')
            data_root = root / 'data'
            train_root = data_root / 'train'
            train_root.mkdir(parents=True)
            shutil.copy(raw_dir / 'data_windspeed12.csv', train_root / 'data_windspeed12.csv')
            dataset = OfflineBCDataset(
                str(data_root),
                include_previous_action_observation=True,
                action_low=[0.0, 0.0],
                action_high=[21_586_451.33303, 0.43],
                include_operating_state_observation=True,
                state_action_artifact_path=str(artifact_path),
                cache_enabled=True,
            )
            cache_paths = dataset._cache_paths(
                data_root,
                'train',
                None,
                None,
                None,
                include_previous_action_observation=True,
                action_low=np.asarray([0.0, 0.0]),
                action_high=np.asarray([21_586_451.33303, 0.43]),
                include_operating_state_observation=True,
                state_action_artifact_sha256=dataset.state_action_artifact_sha256,
            )
            metadata = json.loads(cache_paths['metadata'].read_text(encoding='utf-8'))

        self.assertEqual(dataset.obs.shape, (3, STATE_AWARE_CONTROL_OBS_DIM))
        self.assertEqual(dataset.obs_median.shape, (STATE_AWARE_CONTROL_OBS_DIM,))
        self.assertEqual(float(np.sum(dataset.obs[0, -8:-3])), 1.0)
        self.assertEqual(float(np.sum(dataset.obs[0, -3:])), 1.0)
        self.assertTrue(metadata['include_operating_state_observation'])
        self.assertEqual(metadata['state_action_artifact_sha256'], dataset.state_action_artifact_sha256)

    def test_live_environment_uses_the_same_state_aware_observation_schema(self) -> None:
        with tempfile.TemporaryDirectory(prefix='state_action_live_env_') as tmpdir:
            root = Path(tmpdir)
            raw_dir = root / 'raw'
            self._write_raw_files(raw_dir)
            artifact_path = root / 'state_action_artifact.json'
            artifact_path.write_text(json.dumps(build_state_action_artifact(raw_dir)), encoding='utf-8')
            port = _pick_free_udp_port()
            environment = WindTurbineEnv(
                {
                    'backend': 'bladed_udp',
                    'udp_host': '127.0.0.1',
                    'udp_port': port,
                    'udp_timeout_s': 0.1,
                    'udp_reset_wait_s': 2.0,
                    'udp_inter_frame_wait_s': 2.0,
                    'udp_require_frame_id': True,
                    'action_low': [0.0, 0.0],
                    'action_high': [21_586_451.33303, 0.43],
                    'safe_action_low': [0.0, 0.0],
                    'safe_action_high': [21_586_451.33303, 0.43],
                    'include_previous_action_observation': True,
                    'include_operating_state_observation': True,
                    'state_action_artifact_path': str(artifact_path),
                    'rated_power': 15_000_000.0,
                    'reward': {'power_weight': 0.0, 'blade_root_weight': 0.0},
                },
                {},
                torch.device('cpu'),
            )
            server = FakeBladedUdpServer('127.0.0.1', port)
            server.start()
            try:
                observation, reset_info = environment.reset()
                result = environment.step(np.zeros(2, dtype=np.float32))
            finally:
                server.stop()
                environment.close()

        self.assertEqual(observation.shape, (STATE_AWARE_CONTROL_OBS_DIM,))
        self.assertEqual(result.obs.shape, (STATE_AWARE_CONTROL_OBS_DIM,))
        self.assertEqual(reset_info['control_state_observation_enabled'], 1.0)
        self.assertEqual(result.info['control_state_observation_enabled'], 1.0)
        self.assertEqual(float(np.sum(result.obs[-8:-3])), 1.0)
        self.assertEqual(float(np.sum(result.obs[-3:])), 1.0)

    def test_controller_applies_state_envelopes_to_tensor_and_runtime_actions(self) -> None:
        with tempfile.TemporaryDirectory(prefix='state_action_controller_') as tmpdir:
            root = Path(tmpdir)
            raw_dir = root / 'raw'
            self._write_raw_files(raw_dir)
            artifact_path = root / 'state_action_artifact.json'
            artifact_path.write_text(json.dumps(build_state_action_artifact(raw_dir)), encoding='utf-8')
            env_config = {
                'action_low': [0.0, 0.0],
                'action_high': [21_586_451.33303, 0.43],
                'safe_action_low': [0.0, 0.0],
                'safe_action_high': [21_586_451.33303, 0.43],
                'action_limiter_owner': 'controller',
                'action_delta_low': [-90_000.0, -0.000698],
                'action_delta_high': [90_000.0, 0.000698],
                'bladed_step_duration_ms': 20.0,
                'include_previous_action_observation': True,
                'include_operating_state_observation': True,
                'state_action_artifact_path': str(artifact_path),
            }
            controller = SafeResidualController(
                env_config,
                {
                    'baseline_action': [20_000_000.0, 0.1],
                    'action_representation': 'residual',
                    'residual_limit': [15_000.0, 0.0003],
                    'safe_action_low': [0.0, 0.0],
                    'safe_action_high': [21_586_451.33303, 0.43],
                    'state_action_artifact_path': str(artifact_path),
                    'ood_threshold': 1.0e12,
                    'low_speed_protection_mode': 'disabled',
                    'pitch_protection_mode': 'disabled',
                    'low_power_threshold': None,
                },
            )
            encoded_observations = torch.zeros((2, STATE_AWARE_CONTROL_OBS_DIM), dtype=torch.float32)
            encoded_observations[0, 12] = 1.0
            encoded_observations[0, 17] = 1.0
            encoded_observations[1, 15] = 1.0
            encoded_observations[1, 19] = 1.0
            residuals = torch.ones((2, 2), dtype=torch.float32)
            real_actions = controller.action_real_tensor(residuals, encoded_observations)
            targets = controller.residual_target_tensor(
                torch.tensor([[20_020_000.0, 0.10035], [20_020_000.0, 0.10035]]),
                encoded_observations,
            )

            observation = np.zeros(STATE_AWARE_CONTROL_OBS_DIM, dtype=np.float32)
            observation[9] = 14.0
            observation[1] = 0.7
            observation[5] = 2_000_000.0
            observation[6] = 0.1
            controller.reset_condition(
                observation,
                previous_action_real=np.asarray([20_000_000.0, 0.1]),
            )
            runtime_action = None
            for _ in range(51):
                runtime_action = controller.action_from_residual(
                    np.ones(2, dtype=np.float32),
                    observation,
                    allow_residual=True,
                )

        assert runtime_action is not None
        np.testing.assert_allclose(real_actions[0].numpy(), [20_000_000.0, 0.1], rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(real_actions[1].numpy(), [20_020_000.0, 0.10035], rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(targets[0].numpy(), [0.0, 0.0], rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(targets[1].numpy(), [1.0, 1.0], rtol=0.0, atol=1.0e-5)
        self.assertEqual(runtime_action.optimization_state_code, 3)
        np.testing.assert_allclose(runtime_action.state_residual_limit_real, [20_000.0, 0.00035])
        np.testing.assert_allclose(runtime_action.policy_delta_real, [20_000.0, 0.00035])


if __name__ == '__main__':
    unittest.main()