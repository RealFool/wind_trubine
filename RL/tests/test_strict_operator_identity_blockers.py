from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import warnings

import numpy as np
import torch

from RL.agents import TD3Agent
from RL.envs.wind_turbine_env import WindTurbineEnv, UdpPacket, StepResult, QUEUE_UDP_PROTOCOL_VERSION, BLADED_CSV_HEADER
from RL.replay import ConditionStratifiedReplayBuffer
from RL.train_rl import (_strict_queue_identity_sample, _validate_strict_manifest_identity,
                         _finish_physical_operator_identity, run_queue_training)
from RL.utils.constants import OBSERVATION_VARIABLES
from RL.utils.logger import ExperimentLogger
from RL.utils.paired_pid_reference import CanonicalTraceHasher
from RL.utils.queue_case_identity import QueueStrictCaseIdentityValidator, QueueCaseIdentityMismatch
from RL.utils.queue_manifest import expand_queue_manifest
from RL.utils.safe_residual_control import SafeResidualController
from RL.utils.physical_load_shadow_launch import OPERATOR_ROUTE


class StrictOperatorIdentityBlockerTests(unittest.TestCase):
    def setUp(self):
        self.slot = SimpleNamespace(condition_id=8, condition_instance_id=1, queue_epoch=1,
                                    train_sim_s=600.0)
        self.obs = np.zeros(len(OBSERVATION_VARIABLES), dtype=np.float32)
        self.env = WindTurbineEnv.__new__(WindTurbineEnv)
        self.env.udp_protocol_version = QUEUE_UDP_PROTOCOL_VERSION
        self.env.strict_field_count = True
        self.env.require_condition_metadata = True

    def packet(self, timestamp, wind=9.0, *, phase='END', condition=8, instance=1, epoch=1,
               raw_time=None):
        values = {key: '0' for key in BLADED_CSV_HEADER}
        values['Time'] = str(timestamp if raw_time is None else raw_time)
        values['I_AverageWindSpeed'] = str(wind)
        text = ','.join([QUEUE_UDP_PROTOCOL_VERSION, str(epoch), str(instance), str(condition),
                         phase, str(timestamp), str(round(timestamp / .02)), str(timestamp),
                         *(values[key] for key in BLADED_CSV_HEADER)])
        parsed = self.env._parse_payload(text, include_queue_metadata=True, include_raw_validity=True)
        payload, frame, sim, epoch, instance, condition, phase, elapsed, protocol, invalid = parsed
        self.env._pending_packet = UdpPacket(payload, frame, sim, 0.0, 0, epoch, instance,
                                             condition, phase, elapsed, protocol, invalid)
        return self.env

    def validator(self, *, samples=30001, require_end=True):
        times = np.arange(samples, dtype=np.float64) * .02
        hasher = CanonicalTraceHasher(time_quantization_s=1e-6, wind_quantization_mps=1e-6)
        for timestamp in times:
            hasher.update(timestamp, 8.0)
        entry = SimpleNamespace(condition_id=8, timestamps_s=times,
            wind_speeds_mps=np.full(samples, 8.0), time_signature_quantization_s=1e-6,
            wind_signature_quantization_mps=1e-6, trace_signature=hasher.hexdigest(),
            scenario_id='fixture', pid_raw_csv_path=Path('fixture'), pid_raw_sha256='b'*64,
            time_window_start_s=0.0, time_window_end_s=float(times[-1]))
        reference = SimpleNamespace(entries={8: entry}, entry=lambda _: entry)
        with patch('RL.utils.queue_case_identity.PairedPidReference.from_path', return_value=reference):
            return QueueStrictCaseIdentityValidator((8,), paired_pid_reference_path='unused',
                fixed_simulation_time_offset_s=0.0, wind_absolute_tolerance=.01,
                time_absolute_tolerance_s=1e-6, development_wind_warning=True,
                require_complete_trace_and_end=require_end)

    def observe_prefix(self, validator, count, wind=9.0):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            for index in range(count):
                validator.observe(8, 1, self.obs, sim_time_s=index*.02, wind_speed_mps=wind)

    def test_actual_parse_zero_fill_raw_invalid_aborts_before_warning(self):
        for field, value in [('wind', 'nan'), ('wind', 'inf'), ('time', 'nan')]:
            with self.subTest(field=field, value=value):
                validator = self.validator(samples=3)
                env = self.packet(0.0, value if field == 'wind' else 9.0,
                                  phase='TRAIN', raw_time=value if field == 'time' else None)
                name = 'I_AverageWindSpeed' if field == 'wind' else 'Time'
                self.assertEqual(env.current_payload[name], 0.0)
                self.assertIn(name, env.current_payload_invalid_fields)
                with warnings.catch_warnings(record=True) as caught:
                    with self.assertRaisesRegex(RuntimeError, 'raw invalid'):
                        timestamp, wind, _ = _strict_queue_identity_sample(env)
                        validator.observe(8, 1, self.obs, sim_time_s=timestamp, wind_speed_mps=wind)
                self.assertEqual(caught, [])
                self.assertEqual(validator._sample_count, 0)

    def test_full_600_trace_mismatched_wind_endpoint_end_completes_development_only(self):
        validator = self.validator()
        self.observe_prefix(validator, 30000)
        with warnings.catch_warnings(record=True):
            metrics, summary = _finish_physical_operator_identity(
                validator, self.packet(600.0), self.slot, self.obs)
        self.assertEqual(summary['observed_trace_sample_count'], 30001)
        self.assertEqual(summary['identity_status'], 'development_unverified')
        self.assertEqual(metrics['queue_case_identity_observed_frame_count_complete'], 1)
        self.assertEqual(metrics['queue_case_identity_end_received'], 1)
        self.assertEqual(metrics['queue_case_identity_wind_mismatch_count'], 30001)
        for suffix in ('validated', 'full_trace_complete', 'strict_case_matched', 'trace_signature_match'):
            self.assertEqual(metrics['queue_case_identity_' + suffix], 0)

    def test_next_tick_end_marker_after_exact_600s_train_trace_completes_once(self):
        validator = self.validator()
        self.observe_prefix(validator, 30001)
        # The DLL labels the delimiter one output tick after its final t=600.00
        # TRAIN frame; that delimiter is queue control metadata, not a PID sample.
        with warnings.catch_warnings():
            metrics, summary = _finish_physical_operator_identity(
                validator, self.packet(600.01), self.slot, self.obs)
        self.assertEqual(summary['observed_trace_sample_count'], 30001)
        self.assertEqual(summary['identity_status'], 'development_unverified')
        self.assertEqual(metrics['queue_case_identity_observed_frame_count_complete'], 1)
        self.assertEqual(metrics['queue_case_identity_end_received'], 1)
        self.assertEqual(metrics['queue_case_identity_end_marker_time_s'], 600.01)
        self.assertAlmostEqual(metrics['queue_case_identity_end_marker_offset_from_endpoint_s'], .01)
        self.assertEqual(metrics['queue_case_identity_wind_mismatch_count'], 30001)

    def test_next_tick_end_does_not_replace_missing_600s_train_frame(self):
        validator = self.validator()
        self.observe_prefix(validator, 30000)  # last TRAIN/reference sample is 599.98 s
        with self.assertRaisesRegex(RuntimeError, 'cannot fill the missing trace endpoint'):
            _finish_physical_operator_identity(
                validator, self.packet(600.01), self.slot, self.obs)

    def test_end_repeats_observed_endpoint_without_counting_twice(self):
        validator = self.validator(samples=3)
        self.observe_prefix(validator, 3)
        with warnings.catch_warnings(record=True):
            metrics, summary = _finish_physical_operator_identity(
                validator, self.packet(.04), self.slot, self.obs)
        self.assertEqual(summary['observed_trace_sample_count'], 3)
        self.assertEqual(metrics['queue_case_identity_observed_frame_count_complete'], 1)
        with self.assertRaises(QueueCaseIdentityMismatch):
            validator.observe(8, 1, self.obs, sim_time_s=.04, wind_speed_mps=9.0)

    def test_missing_end_is_not_a_completed_trace_even_with_all_samples(self):
        validator = self.validator(samples=3)
        self.observe_prefix(validator, 3)
        with self.assertRaises(QueueCaseIdentityMismatch):
            validator.finish_condition(8, 1)
        for phase in ('TRAIN', 'SETTLE'):
            with self.subTest(phase=phase), self.assertRaisesRegex(RuntimeError, 'genuine same-slot END'):
                _finish_physical_operator_identity(validator, self.packet(.04, phase=phase),
                                                   self.slot, self.obs)

    def test_599_98_end_and_missing_endpoint_abort(self):
        validator = self.validator()
        self.observe_prefix(validator, 30000)
        with self.assertRaisesRegex(RuntimeError, 'cannot fill the missing trace endpoint'):
            _finish_physical_operator_identity(validator, self.packet(599.98), self.slot, self.obs)
        with self.assertRaises(QueueCaseIdentityMismatch):
            validator.finish_condition(8, 1, end_received=True)

    def test_end_cannot_fill_missing_pre_endpoint_frame_or_internal_gap(self):
        validator = self.validator(samples=3)
        self.observe_prefix(validator, 1)
        with self.assertRaises(QueueCaseIdentityMismatch):
            _finish_physical_operator_identity(validator, self.packet(.04), self.slot, self.obs)
        self.assertEqual(validator._sample_count, 1)

    def test_end_wrong_slot_and_raw_invalid_abort(self):
        validator = self.validator(samples=3)
        self.observe_prefix(validator, 2)
        for overrides in (dict(condition=11), dict(instance=2), dict(epoch=2), dict(wind='nan'),
                          dict(raw_time='nan')):
            with self.subTest(overrides=overrides), self.assertRaises(RuntimeError):
                _finish_physical_operator_identity(validator, self.packet(.04, **overrides),
                                                   self.slot, self.obs)
        self.assertEqual(validator._sample_count, 2)

    def test_full_slot_identity_expanded_rotation_and_nondefault_starts(self):
        manifest = dict(protocol_version=QUEUE_UDP_PROTOCOL_VERSION,
            execution_mode='independent_case_reload', total_condition_rounds=2,
            base_condition_order=[8, 11], rotation_stride=1, queue_epoch_start=42,
            condition_instance_start=101, train_sim_s=600.0)
        for slot in expand_queue_manifest(manifest):
            metadata = dict(udp_condition_id=slot.condition_id,
                            udp_condition_instance_id=slot.condition_instance_id,
                            udp_queue_epoch=slot.queue_epoch)
            _validate_strict_manifest_identity(slot, metadata, context='fixture')
            for key in metadata:
                wrong = dict(metadata, **{key: metadata[key] + 1})
                with self.subTest(slot=slot, key=key), self.assertRaisesRegex(RuntimeError, 'slot mismatch'):
                    _validate_strict_manifest_identity(slot, wrong, context='fixture')

    def test_default_nonphysical_queue_v2_worker_aborts_wrong_initial_slot_before_control(self):
        for overrides in (dict(instance=2), dict(epoch=2), dict(condition=11)):
            with self.subTest(overrides=overrides), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                manifest_path = root / 'manifest.json'
                manifest_path.write_text(json.dumps(dict(protocol_version=QUEUE_UDP_PROTOCOL_VERSION,
                    execution_mode='independent_case_reload', total_condition_rounds=1,
                    base_condition_order=[8], train_sim_s=.04)))
                config = dict(data_schema_version='bladed_udp_19_si_my_v1', algorithm='td3',
                    agent=dict(hidden_dims=[8], td3_bc_enabled=False),
                    train=dict(online_rl_enabled=False, batch_size=2, eval_every=0),
                    queue=dict(condition_ids=[8], total_condition_rounds=1, critic_start_after_rounds=1,
                        actor_start_after_rounds=2, train_sim_s=.04, manifest_path=str(manifest_path)),
                    offline_eval=dict(enabled=False))
                controller = SafeResidualController(
                    dict(action_low=[13000000., 0.], action_high=[21000000., .33],
                         action_delta_low=[-15000., -.0003], action_delta_high=[15000., .0003]),
                    dict(baseline_action=[20000000., .05], residual_limit=[1000000., .05],
                         safe_action_low=[13000000., 0.], safe_action_high=[21000000., .33]))
                env = self.packet(0.0, phase='TRAIN', **overrides)
                metadata = env.current_queue_metadata | dict(condition_train_active=1.0)
                fake = SimpleNamespace(queue_lifecycle_enabled=True, rollout_segment_steps=2,
                    udp_protocol_version=QUEUE_UDP_PROTOCOL_VERSION, last_sent_action=np.array([20000000., .05]),
                    current_payload=env.current_payload, current_payload_invalid_fields=(),
                    current_queue_metadata=env.current_queue_metadata, reset=Mock(return_value=(self.obs, metadata)),
                    step=Mock(side_effect=AssertionError('must not send')), obs_dim=len(self.obs), action_dim=2)
                agent = TD3Agent(len(self.obs), 2, config['agent'], torch.device('cpu'))
                replay = ConditionStratifiedReplayBuffer(len(self.obs), 2, [8], capacity_per_condition=4,
                                                        min_per_condition_in_batch=1)
                logger = ExperimentLogger(root / 'logs')
                try:
                    with patch('RL.train_rl.build_queue_case_identity_validator', return_value=self.validator(samples=3)), \
                            self.assertRaisesRegex(RuntimeError, 'slot mismatch|Queue condition mismatch'):
                        run_queue_training(agent, fake, controller, replay, config, logger, torch.device('cpu'))
                finally:
                    logger.flush()
                fake.step.assert_not_called()

    def test_physical_operator_worker_finish_integration_and_failure_no_receipt(self):
        test = self

        class Env:
            queue_lifecycle_enabled = True
            rollout_segment_steps = 2
            udp_protocol_version = QUEUE_UDP_PROTOCOL_VERSION
            obs_dim = len(OBSERVATION_VARIABLES)
            action_dim = 2
            last_sent_action = np.array([20000000., .05])

            def __init__(self, mode):
                self.mode, self.index = mode, 0

            @property
            def current_payload(self):
                return test.env.current_payload

            @property
            def current_payload_invalid_fields(self):
                return test.env.current_payload_invalid_fields

            @property
            def current_queue_metadata(self):
                return test.env.current_queue_metadata

            def reset(self):
                test.packet(0.0, phase='TRAIN')
                return test.obs.copy(), self.current_queue_metadata | {'condition_train_active': 1.0}

            def step(self, action, on_action_sent=None):
                if on_action_sent:
                    on_action_sent(0.0)
                previous = self.current_queue_metadata
                self.index += 1
                boundary = self.index >= (3 if self.mode == 'duplicate' else 2)
                timestamp = min(self.index, 2) * .02
                if self.mode == 'missingframe':
                    timestamp, boundary = .04, True
                overrides = {}
                if boundary and self.mode == 'wrong_slot':
                    overrides['epoch'] = 2
                if boundary and self.mode == 'raw_invalid':
                    overrides['wind'] = 'nan'
                test.packet(timestamp, phase='TRAIN' if not boundary or self.mode == 'no_end' else 'END',
                            **overrides)
                info = previous | {key.replace('udp_', 'udp_next_', 1): value
                                   for key, value in self.current_queue_metadata.items()}
                info.update(condition_boundary=float(boundary), condition_end=float(boundary),
                    condition_train_active=float(not boundary), transition_store_eligible=float(not boundary),
                    power=10000000., udp_phase_elapsed_s=timestamp)
                return StepResult(test.obs.copy(), 0.0, boundary, info)

        for mode in ('good', 'duplicate', 'no_end', 'wrong_slot', 'raw_invalid', 'missingframe'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                manifest_path = root / 'manifest.json'
                manifest_path.write_text(json.dumps(dict(protocol_version=QUEUE_UDP_PROTOCOL_VERSION,
                    execution_mode='independent_case_reload', total_condition_rounds=1,
                    base_condition_order=[8], train_sim_s=.04)))
                config = dict(data_schema_version='bladed_udp_19_si_my_v1', algorithm='td3',
                    agent=dict(hidden_dims=[8], td3_bc_enabled=False),
                    train=dict(online_rl_enabled=False, batch_size=2, eval_every=0),
                    queue=dict(condition_ids=[8], total_condition_rounds=1, critic_start_after_rounds=1,
                        actor_start_after_rounds=2, train_sim_s=.04, manifest_path=str(manifest_path)),
                    load_predictor=dict(backend='physical_fir_v2', mode='shadow'),
                    physical_shadow_launch=dict(route=OPERATOR_ROUTE), offline_eval=dict(enabled=False))
                controller = SafeResidualController(
                    dict(action_low=[13000000., 0.], action_high=[21000000., .33],
                         action_delta_low=[-15000., -.0003], action_delta_high=[15000., .0003]),
                    dict(baseline_action=[20000000., .05], residual_limit=[1000000., .05],
                         safe_action_low=[13000000., 0.], safe_action_high=[21000000., .33]))
                controller.set_observation_distribution(np.zeros(len(self.obs)), np.ones(len(self.obs)))
                agent = TD3Agent(len(self.obs), 2, config['agent'], torch.device('cpu'))
                agent.set_actor_obs_normalizer(torch.zeros(len(self.obs)), torch.ones(len(self.obs)))
                replay = ConditionStratifiedReplayBuffer(len(self.obs), 2, [8], capacity_per_condition=4,
                                                        min_per_condition_in_batch=1)
                validator, shadow = self.validator(samples=3), Mock(diagnostics={})
                logger = ExperimentLogger(root / 'logs')
                try:
                    # Synthetic tests only: substitute launch gates and the shadow sink,
                    # but exercise the real training loop, parser, identity and finish integration.
                    with patch('RL.train_rl.build_queue_case_identity_validator', return_value=validator), \
                            patch('RL.train_rl.validate_physical_launch'), \
                            patch('RL.train_rl.validate_physical_training_binding'), \
                            patch('RL.train_rl.submit_physical_sent_frame'), \
                            patch('RL.train_rl.complete_physical_step'), \
                            patch('RL.train_rl.physical_condition_metadata', side_effect=lambda _, **kw: kw), \
                            warnings.catch_warnings():
                        warnings.simplefilter('ignore', RuntimeWarning)
                        if mode in ('good', 'duplicate'):
                            summary = run_queue_training(agent, Env(mode), controller, replay, config,
                                logger, torch.device('cpu'), physical_load_shadow=shadow)
                            self.assertEqual(summary['queue_completed_rounds'], 1)
                            shadow.finish_condition.assert_called_once()
                            receipt = shadow.finish_condition.call_args.args[0]
                            self.assertEqual(receipt['identity_status'], 'development_unverified')
                            self.assertEqual(receipt['identity_summary']['observed_trace_sample_count'], 3)
                        else:
                            with self.assertRaises(RuntimeError):
                                run_queue_training(agent, Env(mode), controller, replay, config,
                                    logger, torch.device('cpu'), physical_load_shadow=shadow)
                            shadow.finish_condition.assert_not_called()
                finally:
                    logger.flush()


if __name__ == '__main__':
    unittest.main()