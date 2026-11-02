from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest
import warnings

import numpy as np
import torch

from RL.agents import TD3Agent
from RL.replay import ConditionStratifiedReplayBuffer
from RL.train_rl import run_queue_training
from RL.utils.constants import OBSERVATION_VARIABLES
from RL.utils.logger import ExperimentLogger
from RL.utils.paired_pid_reference import build_paired_pid_reference
from RL.utils.queue_case_identity import (
    QueueCaseIdentityMismatch,
    QueueStrictCaseIdentityValidator,
    QueueWindTraceIdentityValidator,
    build_queue_case_identity_validator,
)
from RL.utils.safe_residual_control import SafeResidualController


class _IdentityMismatchQueueEnvironment:
    queue_lifecycle_enabled = True
    rollout_segment_steps = 2
    obs_dim = len(OBSERVATION_VARIABLES)
    action_dim = 2

    def __init__(self, wind_speed: float) -> None:
        self._obs = np.zeros(self.obs_dim, dtype=np.float32)
        self._obs[OBSERVATION_VARIABLES.index('I_AverageWindSpeed')] = wind_speed
        self._last_sent_action = np.asarray([20_400_000.0, 0.049461], dtype=np.float32)
        self.step_calls = 0

    @property
    def last_sent_action(self) -> np.ndarray:
        return self._last_sent_action.copy()

    def reset(self) -> tuple[np.ndarray, dict[str, float]]:
        return self._obs.copy(), {
            'udp_condition_id': 3.0,
            'udp_condition_instance_id': 101.0,
            'condition_train_active': 1.0,
            'udp_phase_elapsed_s': 0.0,
        }

    def step(self, action: np.ndarray, on_action_sent=None):
        del action, on_action_sent
        self.step_calls += 1
        raise AssertionError('identity mismatch must stop before env.step')


class TestQueueCaseIdentity(unittest.TestCase):
    @staticmethod
    def _write_trace(path: Path, winds: list[float]) -> None:
        with path.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['I_AverageWindSpeed'])
            writer.writeheader()
            writer.writerows({'I_AverageWindSpeed': wind} for wind in winds)

    @staticmethod
    def _observation(wind_speed: float) -> np.ndarray:
        observation = np.zeros(len(OBSERVATION_VARIABLES), dtype=np.float32)
        observation[OBSERVATION_VARIABLES.index('I_AverageWindSpeed')] = wind_speed
        return observation

    @staticmethod
    def _write_paired_reference_raw_files(raw_dir: Path, *, first_time_s: float = 0.0) -> None:
        raw_dir.mkdir(parents=True)
        fields = [
            'Time',
            'I_AverageWindSpeed',
            'Electrical power',
            'Blade root 1 My (Nm)',
            'Blade root 2 My (Nm)',
            'Blade root 3 My (Nm)',
        ]
        for condition_id in range(3, 26):
            with (raw_dir / f'data_windspeed{condition_id}.csv').open('w', encoding='utf-8', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for step in range(3):
                    time_s = first_time_s + 0.02 * step
                    writer.writerow({
                        'Time': time_s,
                        'I_AverageWindSpeed': condition_id + 0.1 * step,
                        'Electrical power': condition_id * 1_000_000.0,
                        'Blade root 1 My (Nm)': condition_id * 1_000_000.0,
                        'Blade root 2 My (Nm)': condition_id * 1_000_000.0,
                        'Blade root 3 My (Nm)': condition_id * 1_000_000.0,
                    })

    def _strict_validator(
        self,
        root: Path,
        *,
        first_time_s: float = 0.0,
        fixed_offset_s: float = 0.0,
    ) -> QueueStrictCaseIdentityValidator:
        raw_dir = root / 'raw'
        self._write_paired_reference_raw_files(raw_dir, first_time_s=first_time_s)
        document = build_paired_pid_reference(
            raw_dir,
            time_window_start_s=first_time_s,
            time_window_end_s=first_time_s + 0.04,
            project_root=root,
        )
        reference_path = root / 'paired_pid_reference.json'
        reference_path.write_text(__import__('json').dumps(document), encoding='utf-8')
        return QueueStrictCaseIdentityValidator(
            (3, 4),
            paired_pid_reference_path=str(reference_path),
            fixed_simulation_time_offset_s=fixed_offset_s,
            wind_absolute_tolerance=0.001,
            time_absolute_tolerance_s=1.0e-6,
        )

    @staticmethod
    def _controller() -> SafeResidualController:
        return SafeResidualController(
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

    def test_validator_identifies_a_raw_wind_trace_mismatch(self) -> None:
        with tempfile.TemporaryDirectory(prefix='queue_case_identity_') as temporary_directory:
            root = Path(temporary_directory)
            self._write_trace(root / 'data_windspeed3.csv', [3.0, 3.1, 3.2])
            self._write_trace(root / 'data_windspeed4.csv', [4.0, 4.1, 4.2])
            validator = QueueWindTraceIdentityValidator(
                (3, 4),
                source_glob=str(root / 'data_windspeed*.csv'),
                wind_sample_count=3,
                wind_absolute_tolerance=0.01,
            )

            for wind_speed in (3.0, 3.1, 3.2):
                metrics = validator.observe(3, 101, self._observation(wind_speed))

            self.assertEqual(metrics['queue_case_identity_validated'], 1.0)
            validator.reset_condition(4, 102)
            with self.assertRaises(QueueCaseIdentityMismatch) as raised:
                validator.observe(4, 102, self._observation(3.0))

        self.assertEqual(raised.exception.metrics['queue_case_identity_condition_id'], 4.0)
        self.assertEqual(raised.exception.metrics['queue_case_identity_closest_condition_id'], 3.0)
        self.assertGreater(raised.exception.metrics['queue_case_identity_expected_max_abs_error'], 0.01)

    def test_validator_classifies_a_unique_complete_physical_prefix_and_primes_it(self) -> None:
        with tempfile.TemporaryDirectory(prefix='queue_case_identity_probe_') as temporary_directory:
            root = Path(temporary_directory)
            self._write_trace(root / 'data_windspeed3.csv', [3.0, 3.1, 3.2])
            self._write_trace(root / 'data_windspeed4.csv', [4.0, 4.1, 4.2])
            validator = QueueWindTraceIdentityValidator(
                (3, 4),
                source_glob=str(root / 'data_windspeed*.csv'),
                wind_sample_count=3,
                wind_absolute_tolerance=0.01,
            )
            observations = [self._observation(wind_speed) for wind_speed in (4.0, 4.1, 4.2)]

            matched_condition_id, metrics = validator.classify_observation_prefix(observations)
            primed_metrics = validator.prime_condition(4, 202, observations)

        self.assertEqual(matched_condition_id, 4)
        self.assertEqual(metrics['queue_case_identity_probe_match_count'], 1.0)
        self.assertEqual(metrics['queue_case_identity_probe_closest_condition_id'], 4.0)
        self.assertEqual(primed_metrics['queue_case_identity_validated'], 1.0)

    def test_validator_accepts_a_prefix_starting_after_socket_drain(self) -> None:
        with tempfile.TemporaryDirectory(prefix='queue_case_identity_offset_') as temporary_directory:
            root = Path(temporary_directory)
            self._write_trace(root / 'data_windspeed3.csv', [3.0, 3.1, 3.2, 3.3, 3.4])
            self._write_trace(root / 'data_windspeed4.csv', [4.0, 4.1, 4.2, 4.3, 4.4])
            validator = QueueWindTraceIdentityValidator(
                (3, 4),
                source_glob=str(root / 'data_windspeed*.csv'),
                wind_sample_count=3,
                wind_absolute_tolerance=0.01,
            )

            for wind_speed in (4.2, 4.3, 4.4):
                metrics = validator.observe(4, 202, self._observation(wind_speed))

        self.assertEqual(metrics['queue_case_identity_validated'], 1.0)
        self.assertEqual(metrics['queue_case_identity_reference_offset'], 2.0)
        self.assertAlmostEqual(metrics['queue_case_identity_expected_wind'], 4.4, places=6)

    def test_validator_rejects_a_shifted_prefix_from_the_wrong_condition(self) -> None:
        with tempfile.TemporaryDirectory(prefix='queue_case_identity_offset_mismatch_') as temporary_directory:
            root = Path(temporary_directory)
            self._write_trace(root / 'data_windspeed3.csv', [3.0, 3.1, 3.2, 3.3, 3.4])
            self._write_trace(root / 'data_windspeed4.csv', [4.0, 4.1, 4.2, 4.3, 4.4])
            validator = QueueWindTraceIdentityValidator(
                (3, 4),
                source_glob=str(root / 'data_windspeed*.csv'),
                wind_sample_count=3,
                wind_absolute_tolerance=0.01,
            )

            with self.assertRaises(QueueCaseIdentityMismatch):
                for wind_speed in (3.2, 3.3, 3.4):
                    validator.observe(4, 202, self._observation(wind_speed))

    def test_validator_rejects_an_ambiguous_physical_prefix(self) -> None:
        with tempfile.TemporaryDirectory(prefix='queue_case_identity_probe_ambiguous_') as temporary_directory:
            root = Path(temporary_directory)
            self._write_trace(root / 'data_windspeed3.csv', [3.0, 3.1, 3.2])
            self._write_trace(root / 'data_windspeed4.csv', [3.0, 3.1, 3.2])
            validator = QueueWindTraceIdentityValidator(
                (3, 4),
                source_glob=str(root / 'data_windspeed*.csv'),
                wind_sample_count=3,
                wind_absolute_tolerance=0.01,
            )

            matched_condition_id, metrics = validator.classify_observation_prefix(
                [self._observation(wind_speed) for wind_speed in (3.0, 3.1, 3.2)]
            )

        self.assertIsNone(matched_condition_id)
        self.assertEqual(metrics['queue_case_identity_probe_match_count'], 2.0)
        self.assertEqual(metrics['queue_case_identity_probe_matched_condition_id'], -1.0)

    def test_strict_validator_matches_complete_fixed_offset_trace(self) -> None:
        with tempfile.TemporaryDirectory(prefix='queue_case_identity_strict_') as temporary_directory:
            validator = self._strict_validator(
                Path(temporary_directory),
                first_time_s=10.0,
                fixed_offset_s=10.0,
            )
            for sim_time_s, wind_speed in ((0.0, 3.0), (0.02, 3.1), (0.04, 3.2)):
                metrics = validator.observe(
                    3,
                    101,
                    self._observation(wind_speed),
                    sim_time_s=sim_time_s,
                    wind_speed_mps=wind_speed,
                )
            final_metrics, summary = validator.finish_condition(3, 101)

        self.assertEqual(metrics['queue_case_identity_validation_level_strict'], 1.0)
        self.assertEqual(final_metrics['queue_case_identity_full_trace_complete'], 1.0)
        self.assertEqual(final_metrics['queue_case_identity_trace_signature_match'], 1.0)
        self.assertEqual(final_metrics['queue_case_identity_strict_case_matched'], 1.0)
        self.assertEqual(summary['identity_status'], 'strict_case_matched')
        self.assertEqual(summary['observed_trace_sample_count'], 3)

    def test_strict_validator_marks_incomplete_trace_reference_only(self) -> None:
        with tempfile.TemporaryDirectory(prefix='queue_case_identity_strict_incomplete_') as temporary_directory:
            validator = self._strict_validator(Path(temporary_directory))
            validator.observe(3, 101, self._observation(3.0), sim_time_s=0.0)
            validator.observe(3, 101, self._observation(3.1), sim_time_s=0.02)
            metrics, summary = validator.finish_condition(3, 101)

        self.assertEqual(metrics['queue_case_identity_full_trace_complete'], 0.0)
        self.assertEqual(metrics['queue_case_identity_strict_case_matched'], 0.0)
        self.assertEqual(metrics['queue_case_identity_reference_matched_only'], 1.0)
        self.assertEqual(summary['identity_status'], 'reference_matched_only')

    def test_strict_validator_rejects_wrong_wind_or_fixed_time_offset(self) -> None:
        with tempfile.TemporaryDirectory(prefix='queue_case_identity_strict_mismatch_') as temporary_directory:
            validator = self._strict_validator(Path(temporary_directory), fixed_offset_s=0.0)
            with self.assertRaises(QueueCaseIdentityMismatch):
                validator.observe(3, 101, self._observation(4.0), sim_time_s=0.0)
            validator = self._strict_validator(
                Path(temporary_directory) / 'offset',
                first_time_s=10.0,
                fixed_offset_s=0.0,
            )
            with self.assertRaises(QueueCaseIdentityMismatch):
                validator.observe(3, 101, self._observation(3.0), sim_time_s=0.0)

    def test_factory_requires_explicit_strict_identity_fields(self) -> None:
        with tempfile.TemporaryDirectory(prefix='queue_case_identity_strict_factory_') as temporary_directory:
            root = Path(temporary_directory)
            validator = self._strict_validator(root)
            reference_path = root / 'paired_pid_reference.json'
            configured = build_queue_case_identity_validator(
                {
                    'case_identity_validation': {
                        'enabled': True,
                        'validation_level': 'strict',
                        'paired_pid_reference_path': str(reference_path),
                        'fixed_simulation_time_offset_s': 0.0,
                        'time_absolute_tolerance_s': 1.0e-6,
                        'wind_absolute_tolerance': 0.001,
                    }
                },
                (3, 4),
            )
            self.assertIsInstance(configured, QueueStrictCaseIdentityValidator)
            self.assertIsInstance(validator, QueueStrictCaseIdentityValidator)
            with self.assertRaisesRegex(ValueError, 'fixed_simulation_time_offset_s'):
                build_queue_case_identity_validator(
                    {
                        'case_identity_validation': {
                            'enabled': True,
                            'validation_level': 'strict',
                            'paired_pid_reference_path': str(reference_path),
                            'time_absolute_tolerance_s': 1.0e-6,
                        }
                    },
                    (3, 4),
                )

    def test_operator_wind_warning_continues_but_never_validates_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            validator = self._strict_validator(Path(tmp))
            validator.wind_absolute_tolerance = 0.01
            validator.development_wind_warning = True
            validator.expected_project_case_fingerprint = 'a' * 64
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                for index, wind in enumerate((3.0104407, 30.0, 3.2)):
                    metrics = validator.observe(3, 101, self._observation(wind),
                        sim_time_s=index * 0.02, wind_speed_mps=wind,
                        runtime_case_fingerprint='a' * 64)
                final, summary = validator.finish_condition(3, 101)
            self.assertEqual(len(caught), 2)  # first mismatch + END, not every frame
            self.assertEqual(metrics['queue_case_identity_wind_mismatch_count'], 2)
            self.assertAlmostEqual(final['queue_case_identity_wind_max_abs_error_mps'], 26.9)
            self.assertEqual(summary['observed_trace_sample_count'], 3)
            self.assertNotEqual(summary['observed_trace_signature'], summary['expected_trace_signature'])
            self.assertEqual(summary['identity_status'], 'development_unverified')
            self.assertEqual(summary['project_case_fingerprint_match'], 1)
            for key in ('validated', 'full_trace_complete', 'strict_case_matched', 'reference_matched_only'):
                self.assertEqual(final['queue_case_identity_' + key], 0)
            validator.reset_condition(3, 102)
            self.assertEqual(validator._wind_mismatch_count, 0)
            with self.assertRaises(QueueCaseIdentityMismatch):
                validator.observe(3, 102, self._observation(3.0), sim_time_s=0.02)

    def test_default_rejects_exact_reported_wind_difference_and_development_keeps_fingerprint_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            validator = self._strict_validator(Path(tmp))
            validator.wind_absolute_tolerance = 0.01
            with self.assertRaises(QueueCaseIdentityMismatch):
                validator.observe(3, 101, self._observation(3.0104407),
                                  sim_time_s=0.0, wind_speed_mps=3.0104407)
            validator.development_wind_warning = True
            validator.expected_project_case_fingerprint = 'a' * 64
            validator.observe(3, 101, self._observation(3.0),
                              sim_time_s=0.0, runtime_case_fingerprint='b' * 64)
            with self.assertRaises(QueueCaseIdentityMismatch):
                validator.finish_condition(3, 101)

    def test_warning_policy_requires_explicit_authorized_strict_factory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self._strict_validator(Path(tmp))
            settings = dict(enabled=True, validation_level='strict',
                paired_pid_reference_path=str(Path(tmp) / 'paired_pid_reference.json'),
                fixed_simulation_time_offset_s=0.0, time_absolute_tolerance_s=1e-6,
                wind_mismatch_policy='warn_development_only')
            with self.assertRaisesRegex(ValueError, 'authorized operator'):
                build_queue_case_identity_validator({'case_identity_validation': settings}, (3, 4))
            validator = build_queue_case_identity_validator({'case_identity_validation': settings},
                (3, 4), operator_development_authorized=True)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                validator.observe(3, 101, self._observation(4.0), sim_time_s=0.0)
            with self.assertRaises(ValueError):
                validator.observe(3, 101, self._observation(3.1), sim_time_s=0.0)
            with self.assertRaises(ValueError):
                validator.observe(3, 101, self._observation(float('nan')), sim_time_s=0.02)
            settings['validation_level'] = 'prefix'
            with self.assertRaises(ValueError):
                build_queue_case_identity_validator({'case_identity_validation': settings},
                    (3, 4), operator_development_authorized=True)

    def test_queue_training_stops_before_an_action_on_identity_mismatch(self) -> None:
        with tempfile.TemporaryDirectory(prefix='queue_case_identity_training_') as temporary_directory:
            root = Path(temporary_directory)
            self._write_trace(root / 'data_windspeed3.csv', [3.0, 3.1, 3.2])
            self._write_trace(root / 'data_windspeed4.csv', [4.0, 4.1, 4.2])
            config = {
                'data_schema_version': 'bladed_udp_19_si_my_v1',
                'algorithm': 'td3',
                'agent': {'hidden_dims': [8], 'td3_bc_enabled': False, 'td3_bc_weight': 1.0},
                'train': {
                    'online_rl_enabled': False,
                    'online_eval_enabled': False,
                    'eval_every': 0,
                    'batch_size': 2,
                    'updates_per_step': 1,
                    'update_every_steps': 1,
                    'step_log_every': 10,
                    'condition_replay': {
                        'sampling': 'condition_stratified',
                        'capacity_per_condition': 4,
                        'min_valid_transitions_per_condition': 1,
                        'min_per_condition_in_batch': 1,
                    },
                },
                'queue': {
                    'condition_ids': [3, 4],
                    'total_condition_rounds': 1,
                    'critic_start_after_rounds': 1,
                    'actor_start_after_rounds': 2,
                    'case_identity_validation': {
                        'enabled': True,
                        'source_glob': str(root / 'data_windspeed*.csv'),
                        'wind_sample_count': 3,
                        'wind_absolute_tolerance': 0.01,
                    },
                },
                'offline_eval': {'enabled': False},
            }
            controller = self._controller()
            controller.set_observation_distribution(
                np.zeros(len(OBSERVATION_VARIABLES), dtype=np.float32),
                np.ones(len(OBSERVATION_VARIABLES), dtype=np.float32),
            )
            agent = TD3Agent(
                obs_dim=len(OBSERVATION_VARIABLES),
                action_dim=2,
                config=config['agent'],
                device=torch.device('cpu'),
            )
            agent.set_actor_obs_normalizer(torch.zeros(len(OBSERVATION_VARIABLES)), torch.ones(len(OBSERVATION_VARIABLES)))
            agent.set_critic_obs_normalizer(torch.zeros(len(OBSERVATION_VARIABLES)), torch.ones(len(OBSERVATION_VARIABLES)))
            replay = ConditionStratifiedReplayBuffer(
                len(OBSERVATION_VARIABLES),
                2,
                [3, 4],
                capacity_per_condition=4,
                min_per_condition_in_batch=1,
            )
            environment = _IdentityMismatchQueueEnvironment(4.0)
            logger = ExperimentLogger(root / 'output')

            with self.assertRaisesRegex(RuntimeError, 'Queue physical case identity mismatch'):
                run_queue_training(agent, environment, controller, replay, config, logger, torch.device('cpu'))
            logger.flush()
            mismatch_path = root / 'output' / 'queue_case_identity_mismatch_condition_03_instance_101.json'
            with mismatch_path.open(encoding='utf-8') as handle:
                mismatch = __import__('json').load(handle)

        self.assertEqual(environment.step_calls, 0)
        self.assertEqual(mismatch['queue_case_identity_condition_id'], 3.0)
        self.assertEqual(mismatch['queue_case_identity_closest_condition_id'], 4.0)
        self.assertEqual(mismatch['queue_case_identity_mismatch'], 1.0)


if __name__ == '__main__':
    unittest.main()