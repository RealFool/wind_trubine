from __future__ import annotations

import copy
from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import queue
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from RL.tests import test_physical_load_margin_dynamic as dynamic_fixture
from RL.tests import test_queue_training as queue_fixture
from RL.tests import test_bladed_inference as inference_fixture
import RL.train_rl as training_entry
from RL.train_rl import _build_queue_load_predictor_shadow, run_queue_training
from RL.agents import TD3Agent
from RL.envs.wind_turbine_env import StepResult, WindTurbineEnv, UdpPacket
from RL.replay import ConditionStratifiedReplayBuffer
from RL.utils.constants import DATA_SCHEMA_VERSION, OBS_DIM
from RL.utils.logger import ExperimentLogger
from RL.utils.safe_residual_control import SafeResidualController
from RL.infer.run_bladed_inference import _build_load_predictor_shadow, run_inference
from RL.utils.physical_load_margin import INPUT_FIELDS, LOAD_FIELDS, REGIONS, document_sha256
from RL.utils.physical_load_shadow import (
    PhysicalLoadShadow, PhysicalShadowCore, SentPhysicalFrame, complete_physical_step,
    physical_backend_selected, physical_outcome_tag, preflight_physical_config,
    physical_condition_metadata, submit_physical_sent_frame,
)
from RL.utils.physical_load_shadow_artifact import (
    ACTION_SEMANTICS, ARTIFACT_FILENAMES, BACKEND, BUNDLE_SCHEMA, LABEL_SEMANTICS,
    RUNTIME_AUTHORIZATION, RUNTIME_SCHEMA, file_sha256, load_physical_shadow_artifacts,
)
from RL.utils.physical_load_shadow_launch import (
    LAUNCH_SCHEMA, REQUIRED_GATES, physical_runtime_identity, validate_physical_launch,
    load_verified_physical_checkpoint,
    validate_physical_agent_parameters,
    validate_physical_training_binding,
)
from RL.evaluate_physical_load_shadow import _PointSummaryAccumulator, evaluate_shadow_records
from RL.utils.paired_pid_reference import canonical_trace_signature


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REVIEWED = PROJECT_ROOT / 'RL/configs/physical_shadow_state20_v1_reviewed'


class PhysicalLoadShadowTests(unittest.TestCase):
    def setUp(self) -> None:
        helper = dynamic_fixture.PhysicalLoadMarginDynamicTests()
        helper.setUp()
        self.addCleanup(helper.doCleanups)
        self.root = helper.root
        model, self.case, _, scale = helper.fitted()
        base_path = self.root / ARTIFACT_FILENAMES['base_protocol']
        base_path.write_text(json.dumps(model.physics.protocol), encoding='utf-8')
        model.spec = copy.deepcopy(model.spec)
        model.spec['base_protocol_path'] = str(base_path)
        model.spec['base_protocol_sha256'] = document_sha256(model.physics.protocol)
        _, calibration = helper.synthetic_calibration(model, scale)
        documents = {
            'base_protocol': model.physics.protocol, 'development_protocol': model.spec,
            'physics_model': model.physics.to_document(), 'dynamic_model': model.to_document(),
            'error_scales': {'dynamic_fir': scale.to_document()},
            'joint_calibration': {'dynamic_fir': calibration},
        }
        entries = {}
        for role, doc in documents.items():
            path = self.root / ARTIFACT_FILENAMES[role]
            path.write_text(json.dumps(doc, allow_nan=False), encoding='utf-8')
            entries[role] = {'path': str(path), 'file_sha256': file_sha256(path),
                             'document_sha256': document_sha256(doc)}
        project = PROJECT_ROOT / 'bladed_env/15MW_rl.prj'
        self.bundle = {
            'schema_version': BUNDLE_SCHEMA, 'runtime_schema_version': RUNTIME_SCHEMA,
            'authorization': dict(RUNTIME_AUTHORIZATION), 'external_fields': list(INPUT_FIELDS),
            'action_semantics': ACTION_SEMANTICS, 'label_semantics': LABEL_SEMANTICS,
            'primary_horizons_s': [0.1, 0.2], 'artifacts': entries,
            'engineering_project': {'path': str(project), 'file_sha256': file_sha256(project)},
        }
        self.bundle_path = self.root / 'bundle.json'
        self._save_bundle()
        self.artifacts = self._load()
        # Synthetic reference/identity only, never a real gate receipt.
        trace = canonical_trace_signature(self.case.time, self.case.values['I_AverageWindSpeed'])
        self.reference_entry = SimpleNamespace(scenario_id='wind3', pid_raw_csv_path=self.root / 'fixture.csv',
            pid_raw_sha256='c' * 64, timestamps_s=self.case.time, time_window_start_s=0.0,
            time_window_end_s=float(self.case.time[-1]), trace_signature=trace,
            blade_root_peak_abs_nm=self.case.pid_peak_nm)
        self.reference = SimpleNamespace(entry=lambda cid: self.reference_entry,
            blade_root_peak=lambda cid: self.case.pid_peak_nm)
        self.reference_sha256 = 'b' * 64

    def _save_bundle(self) -> None:
        self.bundle_path.write_text(json.dumps(self.bundle, allow_nan=False), encoding='utf-8')

    def _load(self):
        return load_physical_shadow_artifacts(self.bundle_path,
            expected_sha256=file_sha256(self.bundle_path), project_root=PROJECT_ROOT)

    def _mutate(self, role, mutate) -> None:
        entry = self.bundle['artifacts'][role]
        path = Path(entry['path'])
        doc = json.loads(path.read_text())
        mutate(doc)
        path.write_text(json.dumps(doc, allow_nan=False), encoding='utf-8')
        entry['file_sha256'] = file_sha256(path)
        entry['document_sha256'] = document_sha256(doc)
        self._save_bundle()

    def frame(self, index: int, *, instance: int = 101, actor_version: int = 0,
              action=None, condition_id: int = 3) -> SentPhysicalFrame:
        values = tuple(float(self.case.values[name][index]) for name in INPUT_FIELDS)
        command = self.case.actions[index] if action is None else action
        return SentPhysicalFrame(condition_id, instance, index, values, tuple(command),
            tuple(self.case.pid_peak_nm), actor_version, 'a' * 64, time.monotonic())

    def payload(self, index: int) -> dict[str, float]:
        return {name: float(self.case.values[name][index]) for name in INPUT_FIELDS}

    def core(self, stride=5) -> PhysicalShadowCore:
        return PhysicalShadowCore(self.artifacts, origin_stride_frames=stride)

    def test_strict_loader_roundtrip_nullable_unfitted_and_authorization(self) -> None:
        self.assertEqual(document_sha256(self.artifacts.model.to_document()),
                         self.bundle['artifacts']['dynamic_model']['document_sha256'])
        self.assertTrue(np.isnan(self.artifacts.model.coefficients[2]).all())
        self.assertEqual(self.artifacts.bundle['authorization'], RUNTIME_AUTHORIZATION)
        self.assertFalse(self.artifacts.calibration['groups'][REGIONS[2]]['supported'])

    def test_loader_rejects_file_sha_and_added_variable_or_control_authorization(self) -> None:
        with self.assertRaisesRegex(ValueError, 'SHA'):
            load_physical_shadow_artifacts(self.bundle_path, expected_sha256='0' * 64,
                                          project_root=PROJECT_ROOT)
        for mutation in ('variable', 'mode'):
            changed = copy.deepcopy(self.bundle)
            if mutation == 'variable':
                changed['external_fields'].append('new_sensor')
            else:
                changed['authorization']['soft_authorized'] = True
            self.bundle_path.write_text(json.dumps(changed), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'schema/whitelist/authorization'):
                self._load()
        self._save_bundle()

    def test_loader_rejects_wrong_feature_shape_even_after_rehash(self) -> None:
        self._mutate('dynamic_model', lambda d: d['coefficients'][REGIONS[0]][0]['weights'].pop())
        with self.assertRaisesRegex(ValueError, 'shape'):
            self._load()

    def test_loader_rejects_old_audit_calibration_role_after_rehash(self) -> None:
        def mutate(doc):
            doc['dynamic_fir']['groups'][REGIONS[0]]['case_origin_counts'] = {'11': 200}
        self._mutate('joint_calibration', mutate)
        with self.assertRaisesRegex(ValueError, 'only frozen calibration'):
            self._load()

    def test_loader_rejects_calibration_count_mismatch_after_rehash(self) -> None:
        def mutate(doc):
            doc['dynamic_fir']['groups'][REGIONS[0]]['case_origin_counts']['8'] = 199
        self._mutate('joint_calibration', mutate)
        with self.assertRaisesRegex(ValueError, 'counts do not close'):
            self._load()

    def test_loader_rejects_nonfinite_json_and_wrong_identity(self) -> None:
        path = Path(self.bundle['artifacts']['dynamic_model']['path'])
        text = path.read_text().replace('0.0001', 'NaN', 1)
        if 'NaN' not in text:
            text = '{"bad":NaN}'
        path.write_text(text)
        self.bundle['artifacts']['dynamic_model']['file_sha256'] = file_sha256(path)
        self._save_bundle()
        with self.assertRaisesRegex(ValueError, 'Non-finite'):
            self._load()

    def test_streaming_matches_offline_predictions_derivatives_and_classifier(self) -> None:
        core = self.core()
        for index in range(401):
            result = core.observe(self.frame(index))
            if index in (100, 200, 300, 400):
                offline = self.artifacts.model.predict(self.case, np.asarray([index]))
                np.testing.assert_allclose(result['peaks'], offline.peaks_nm['dynamic_fir'][0, :2], rtol=1e-12)
                np.testing.assert_array_equal(result['supported'], offline.supported[0, :2])
                np.testing.assert_array_equal(core._derivative, self.case.derivatives[index])
                self.assertEqual(core._history[-1][2], self.case.region[index])
        self.assertLessEqual(core.history_count, 41)
        self.assertLessEqual(core.pending_count, 2)

    def test_future_payload_and_alias_actions_never_enter_prediction(self) -> None:
        a, b = self.core(), self.core()
        for index in range(101):
            frame = self.frame(index)
            r1 = a.observe(frame)
            values = list(frame.values)
            values[INPUT_FIELDS.index('O_PcsTorqueDemand')] = -12345.0
            values[INPUT_FIELDS.index('O_PitchPositionDemand1')] = 9.0
            r2 = b.observe(SentPhysicalFrame(**{**frame.__dict__, 'values': tuple(values)}))
        np.testing.assert_array_equal(r1['peaks'], r2['peaks'])
        self.assertEqual(a.records[-1]['risk_status'], 'unknown_rl_margin_not_validated')

    def test_short_horizons_mature_separately_not_waiting_for_long_window(self) -> None:
        core = self.core(stride=50)
        for index in range(56):
            core.observe(self.frame(index))
        self.assertEqual([row['horizon_s'] for row in core.records], [0.1])
        stats = core.finish_condition()
        self.assertEqual(stats['dropped_pending_count'], 1)
        self.assertEqual(len(core.drain_records()), 1)

    def test_origin_action_interval_excludes_endpoint_action_and_actor_version(self) -> None:
        core = self.core(stride=50)
        origin_action = self.case.actions[50]
        for index in range(56):
            command = [21e6, 0.4] if index == 55 else origin_action
            core.observe(self.frame(index, action=command, actor_version=1 if index == 55 else 0))
        row = core.records[-1]
        self.assertTrue(row['approximate_zoh'])
        self.assertTrue(row['actor_policy_window_frozen'])
        self.assertEqual(row['executed_interval_actions'], [origin_action.tolist()] * 5)
        self.assertEqual(row['label_semantics'], LABEL_SEMANTICS)

    def test_actor_update_inside_interval_is_development_ineligible(self) -> None:
        core = self.core(stride=50)
        for index in range(56):
            core.observe(self.frame(index, actor_version=int(index >= 54)))
        row = core.records[-1]
        self.assertFalse(row['actor_policy_window_frozen'])
        self.assertFalse(row['development_point_eligible'])
        self.assertFalse(row['independent_audit_eligible'])

    def test_terminal_observation_matures_short_window_without_fake_endpoint_action(self) -> None:
        core = self.core(stride=50)
        for index in range(55):
            core.observe(self.frame(index))
        core.observe_terminal(payload=self.payload(55), condition_id=3, condition_instance_id=101, frame_id=55)
        self.assertEqual(len(core.records), 1)
        self.assertEqual(core.records[0]['label_endpoint_time_s'], 1.1)
        self.assertEqual(len(core.records[0]['executed_interval_actions']), 5)
        self.assertEqual(core.finish_condition()['dropped_pending_count'], 1)

    def test_gap_resets_all_causal_state_preserving_observed_exceedance(self) -> None:
        core = self.core()
        for index in range(101):
            core.observe(self.frame(index))
        core._cumulative[0] = 2.0 * self.case.pid_peak_nm[0]
        core.observe(self.frame(103))
        self.assertEqual(core.history_count, 1)
        self.assertEqual(core.pending_count, 0)
        np.testing.assert_array_equal(core._derivative, np.zeros(5))
        self.assertGreater(core._cumulative[0], self.case.pid_peak_nm[0])
        core.reset_condition(3, 102)
        np.testing.assert_array_equal(core._cumulative, np.zeros(3))

    def test_timeout_invalid_and_unconfirmed_outcomes_never_cross_label_window(self) -> None:
        for reason in ('timeout', 'invalid', 'unsafe_stop', 'frame_gap'):
            with self.subTest(reason=reason):
                core = self.core()
                for index in range(51):
                    core.observe(self.frame(index), require_outcome=True)
                    core.confirm_outcome(index * 0.02, 'complete')
                core.confirm_outcome(1.0, reason)
                self.assertEqual(core.history_count, 0)
                self.assertEqual(core.pending_count, 0)
        core = self.core()
        core.observe(self.frame(0), require_outcome=True)
        core.observe(self.frame(1), require_outcome=True)
        self.assertEqual(core.history_count, 1)

    def test_record_buffer_is_bounded_and_never_grants_rl_coverage(self) -> None:
        core = PhysicalShadowCore(self.artifacts, record_capacity=1)
        for index in range(100):
            core.observe(self.frame(index))
        self.assertEqual(len(core.records), 1)
        self.assertGreater(core.stats['record_capacity_dropped'], 0)
        row = core.records[0]
        self.assertFalse(row['rl_margin_coverage_authorized'])
        self.assertFalse(row['calibration_eligible'])
        self.assertFalse(row['authorization']['soft_authorized'])

    def test_evaluator_point_summary_uses_bounded_deterministic_reservoirs(self) -> None:
        accumulator = _PointSummaryAccumulator('fixture|below_rated|H0.10', capacity=64)
        exact_absolute_error = np.zeros(3, dtype=np.float64)
        expected_errors = []
        for index in range(5000):
            actual = np.asarray([50.0 + index % 17, 60.0 + index % 13, 70.0 + index % 11])
            reference = np.asarray([100.0, 110.0, 120.0])
            predicted = actual - np.asarray([index % 5, index % 7, index % 9], dtype=np.float64)
            row = {
                'actual_future_peak_nm_by_blade': actual.tolist(),
                'pid_peak_abs_nm': reference.tolist(),
                'predicted_peak_nm_by_blade': predicted.tolist(),
                'predicted_v1_peak_nm_by_blade': predicted.tolist(),
                'predicted_persistence_peak_nm_by_blade': predicted.tolist(),
            }
            accumulator.add(row)
            exact_absolute_error += np.abs(actual - predicted)
            expected_errors.extend(np.abs(actual - predicted).tolist())

        summary = accumulator.summary()
        self.assertEqual(summary['origin_count'], 5000)
        self.assertAlmostEqual(summary['methods']['dynamic_fir']['mae_nm'],
                               float(exact_absolute_error.sum() / (5000 * 3)))
        self.assertEqual(summary['quantile_reservoir_capacity_per_blade'], 64)
        self.assertEqual(len(accumulator.under_all['dynamic_fir'].values), 192)
        self.assertEqual(len(accumulator.under_by_blade['dynamic_fir'][0].values), 64)
        self.assertAlmostEqual(summary['methods']['dynamic_fir']['positive_underestimate_p95_nm'],
                               float(np.quantile(expected_errors, 0.95)), delta=1.5)

    def test_worker_full_cycle_exact_demand_not_udp_alias_and_strict_json(self) -> None:
        shadow = PhysicalLoadShadow(self.artifacts, output_dir=self.root, queue_capacity=1024)
        try:
            for index in range(71):
                payload = self.payload(index)
                payload['O_PcsTorqueDemand'] = float('nan')
                payload['O_PitchPositionDemand1'] = float('nan')
                shadow.submit_sent(condition_id=3, condition_instance_id=101, payload=payload,
                    sent_action_real=self.case.actions[index], pid_peak_abs_nm=self.case.pid_peak_nm,
                    actor_policy_version=0, actor_policy_sha256='a' * 64,
                    sent_monotonic_s=time.monotonic(), frame_id=index)
                shadow.confirm_outcome(index * 0.02, 'complete')
            shadow.finish_condition({'condition_id': 3, 'condition_instance_id': 101,
                                     'identity_status': 'strict_case_matched'})
        finally:
            summary = shadow.close()
        self.assertTrue(summary['finalized'])
        self.assertFalse(summary['faulted'])
        rows = [json.loads(line) for line in shadow.path.read_text().splitlines()]
        self.assertGreater(len(rows), 0)
        self.assertTrue(all(row['identity_status'] == 'unconfirmed_missing_physical_reference' for row in rows))
        self.assertTrue(all(not row['development_point_eligible'] for row in rows))
        self.assertEqual(summary['counters']['strict_complete_conditions'], 0)
        self.assertTrue(all(not row['independent_audit_eligible'] for row in rows))
        self.assertGreaterEqual(rows[0]['sent_to_result_available_ms'], 0.0)
        self.assertFalse(summary['authorization']['rl_margin_coverage_authorized'])

    def test_slot_metadata_can_never_overwrite_old_record_identity(self) -> None:
        shadow = PhysicalLoadShadow(self.artifacts, output_dir=self.root)
        try:
            for index in range(61):
                shadow.core.observe(self.frame(index))
            shadow._write_records({'condition_id': 11, 'condition_instance_id': 102,
                                   'identity_status': 'strict_case_matched'})
        finally:
            shadow.close()
        rows = [json.loads(line) for line in shadow.path.read_text().splitlines()]
        self.assertTrue(all(row['condition_id'] == 3 and row['condition_instance_id'] == 101 for row in rows))
        self.assertTrue(all(not row['development_point_eligible'] for row in rows))
        self.assertTrue(all(row['identity_status'] == 'unconfirmed_slot_metadata' for row in rows))

    def enqueue_fixture_slot(self, shadow: PhysicalLoadShadow, *, epoch: int = 1,
                             finish_epoch: int = 1, duplicate_finish: bool = False,
                             development_unverified: bool = False) -> None:
        shadow.evaluation_reference = self.reference
        shadow.evaluation_reference_file_sha256 = self.reference_sha256
        for index in range(len(self.case.time) - 1):
            frame = self.frame(index)
            frame = SentPhysicalFrame(**{**frame.__dict__, 'queue_epoch': epoch})
            self.assertTrue(shadow._put(('frame', frame, False)))
            self.assertTrue(shadow._put(('outcome', frame.simulation_time_s, 'complete')))
        shadow.submit_terminal(payload=self.payload(len(self.case.time) - 1), condition_id=3,
            condition_instance_id=101, frame_id=len(self.case.time) - 1, queue_epoch=epoch)
        metadata = physical_condition_metadata(shadow, condition_id=3, condition_instance_id=101,
            queue_epoch=finish_epoch, identity_status='strict_case_matched', identity_summary={
                'observed_trace_signature': self.reference_entry.trace_signature,
                'observed_trace_sample_count': len(self.case.time),
                'fixed_simulation_time_offset_s': 0.0, 'trace_signature_match': 1.0})
        if development_unverified:
            metadata.update(identity_status='development_unverified', strict_physical_trace_complete=False,
                observed_trace_signature='d' * 64, trace_signature_match=0.0,
                wind_mismatch_policy='warn_development_only', queue_case_identity_wind_mismatch_count=2.0,
                queue_case_identity_wind_max_abs_error_mps=0.0104407,
                queue_case_identity_development_unverified=1.0)
        shadow.finish_condition(metadata)
        if duplicate_finish:
            shadow.finish_condition(metadata)

    def test_worker_complete_capture_closes_real_fixture_slot_receipt_and_records(self) -> None:
        shadow = PhysicalLoadShadow(self.artifacts, output_dir=self.root, queue_capacity=2048)
        try:
            self.enqueue_fixture_slot(shadow)
        finally:
            summary = shadow.close(maximum_wait_s=10.0)
        self.assertTrue(summary['finalized'])
        self.assertFalse(summary['faulted'])
        self.assertEqual(summary['counters']['strict_complete_conditions'], 1)
        report = self.evaluate_records(shadow.path, worker_summary=summary, expected_slot_count=1,
                                       expected_slot_identities=[(1, 101, 3)])
        self.assertTrue(report['capture_completeness']['complete_capture'])
        self.assertTrue(report['capture_completeness']['record_slot_order_matches_manifest'])

    def test_operator_unverified_worker_persists_metrics_without_strict_qualification(self) -> None:
        shadow = PhysicalLoadShadow(self.artifacts, output_dir=self.root, queue_capacity=2048)
        try:
            self.enqueue_fixture_slot(shadow, development_unverified=True)
        finally:
            summary = shadow.close(maximum_wait_s=10.0)
        receipt = summary['condition_receipts'][0]
        self.assertEqual(receipt['identity_status'], 'development_unverified')
        self.assertEqual(receipt['queue_case_identity_wind_mismatch_count'], 2)
        self.assertAlmostEqual(receipt['queue_case_identity_wind_max_abs_error_mps'], 0.0104407)
        self.assertFalse(receipt['strict_physical_trace_complete'])
        self.assertEqual(summary['counters']['strict_complete_conditions'], 0)
        rows = [json.loads(line) for line in shadow.path.read_text().splitlines()]
        self.assertTrue(all(row['identity_status'] == 'development_unverified' for row in rows))
        self.assertTrue(all(not row['development_point_eligible'] for row in rows))
        report = self.evaluate_records(shadow.path, worker_summary=summary, expected_slot_count=1,
                                       expected_slot_identities=[(1, 101, 3)])
        self.assertFalse(report['capture_completeness']['complete_capture'])
        self.assertEqual(report['point_metrics'], {})
        self.assertEqual(report['pid_illustrative_joint_metrics'], {})
        self.assertGreater(report['counters']['operator_unverified_point_records'], 0)
        self.assertTrue(report['operator_unverified_point_diagnostics']['point_metrics'])
        self.assertFalse(report['operator_unverified_point_diagnostics']['strict_eligible'])

    def test_worker_wrong_epoch_or_duplicate_finish_never_creates_complete_receipt(self) -> None:
        for mutation in ('wrong_epoch', 'duplicate'):
            with self.subTest(mutation=mutation):
                output = self.root / mutation
                output.mkdir()
                shadow = PhysicalLoadShadow(self.artifacts, output_dir=output, queue_capacity=2048)
                try:
                    self.enqueue_fixture_slot(shadow, epoch=9 if mutation == 'wrong_epoch' else 1,
                                              duplicate_finish=mutation == 'duplicate')
                finally:
                    summary = shadow.close(maximum_wait_s=10.0)
                self.assertFalse(summary['faulted'])
                self.assertGreater(summary['counters']['finish_slot_identity_mismatch'], 0)
                if mutation == 'wrong_epoch':
                    self.assertEqual(summary['condition_receipts'][0]['queue_epoch'], 9)
                    self.assertFalse(summary['condition_receipts'][0]['strict_physical_trace_complete'])
                    self.assertEqual(summary['counters']['strict_complete_conditions'], 0)
                    with self.assertRaisesRegex(ValueError, 'record slot'):
                        self.evaluate_records(shadow.path, worker_summary=summary, expected_slot_count=1,
                                              expected_slot_identities=[(1, 101, 3)])
                else:
                    self.assertEqual(summary['counters']['strict_complete_conditions'], 1)
                    self.assertEqual(summary['counters']['duplicate_slot_finish'], 1)
                    self.assertFalse(summary['condition_receipts'][1]['strict_physical_trace_complete'])
                    report = self.evaluate_records(shadow.path, worker_summary=summary, expected_slot_count=1,
                                                   expected_slot_identities=[(1, 101, 3)])
                    self.assertFalse(report['capture_completeness']['complete_capture'])

    def test_worker_fault_is_contained_and_bypasses_future_shadow_only(self) -> None:
        shadow = PhysicalLoadShadow(self.artifacts, output_dir=self.root, queue_capacity=1024)
        with patch.object(shadow.core.model, 'predict', side_effect=RuntimeError('fixture_failure')):
            for index in range(61):
                shadow._put(('frame', self.frame(index), False))
                shadow._put(('outcome', index * 0.02, 'complete'))
            summary = shadow.close()
        self.assertTrue(summary['faulted'])
        self.assertEqual(summary['last_error_type'], 'RuntimeError')
        self.assertEqual(shadow.diagnostics['physical_load_shadow_action_changed'], 0.0)

    def test_submit_queue_full_is_nonblocking_and_sets_discontinuity(self) -> None:
        shadow = PhysicalLoadShadow(self.artifacts, output_dir=self.root)
        try:
            with patch.object(shadow._queue, 'put_nowait', side_effect=queue.Full):
                self.assertFalse(shadow._put(('reset', 3, 101)))
            self.assertTrue(shadow._submission_invalidated)
            self.assertEqual(shadow.diagnostics['physical_load_shadow_queue_dropped'], 1.0)
        finally:
            shadow.close()

    def test_send_callback_uses_successful_wire_snapshot_and_is_fault_isolated(self) -> None:
        shadow = PhysicalLoadShadow(self.artifacts, output_dir=self.root)
        wire = np.asarray([12345678.123456, 0.012345])
        env = SimpleNamespace(last_successful_action_send={'condition_id': 3,
            'condition_instance_id': 101, 'frame_id': 9, 'action_real': wire},
            reward_fn=SimpleNamespace(pid_blade_root_peak_reference=lambda cid: self.case.pid_peak_nm))
        try:
            with patch.object(shadow, 'submit_sent') as submit:
                submit_physical_sent_frame(shadow, env, payload=self.payload(9), condition_id=3,
                    condition_instance_id=101, sent_action_real=[999.0, 0.0], actor_policy_version=0,
                    actor_policy_sha256='', sent_monotonic_s=time.monotonic())
                np.testing.assert_array_equal(submit.call_args.kwargs['sent_action_real'], wire)
                self.assertEqual(submit.call_args.kwargs['frame_id'], 9)
            env.last_successful_action_send['condition_instance_id'] = 999
            submit_physical_sent_frame(shadow, env, payload={}, condition_id=3,
                condition_instance_id=101, sent_action_real=wire, actor_policy_version=0,
                actor_policy_sha256='', sent_monotonic_s=time.monotonic())
            self.assertEqual(shadow.diagnostics['physical_load_shadow_invalid_submissions'], 1.0)
        finally:
            shadow.close()

    def test_terminal_step_completion_submits_only_same_stream_real_terminal(self) -> None:
        shadow = PhysicalLoadShadow(self.artifacts, output_dir=self.root)
        info = {'condition_boundary': 1.0, 'udp_condition_id': 3.0, 'udp_next_condition_id': 3.0,
                'udp_condition_instance_id': 101.0, 'udp_next_condition_instance_id': 101.0,
            'udp_next_frame_id': 55.0, 'udp_next_phase_code': 3.0,
            'udp_queue_epoch': 1.0, 'udp_next_queue_epoch': 1.0}
        try:
            with patch.object(shadow, 'confirm_outcome') as confirm, patch.object(shadow, 'submit_terminal') as terminal:
                complete_physical_step(shadow, SimpleNamespace(current_payload=self.payload(55)),
                    payload=self.payload(54), info=info, unsafe_stop=False)
                self.assertEqual(confirm.call_args.args[1], 'complete')
                terminal.assert_called_once()
                info['udp_next_condition_instance_id'] = 102.0
                complete_physical_step(shadow, SimpleNamespace(current_payload=self.payload(55)),
                    payload=self.payload(54), info=info, unsafe_stop=False)
                self.assertEqual(confirm.call_args.args[1], 'condition_boundary')
                self.assertEqual(terminal.call_count, 1)
        finally:
            shadow.close()

    def test_nn_factories_do_not_load_model_for_physical_backend(self) -> None:
        config = {'load_predictor': {'backend': BACKEND, 'mode': 'shadow'}}
        with patch('RL.train_rl.load_candidate_conditioned_predictor') as train_load, \
                patch('RL.infer.run_bladed_inference.load_candidate_conditioned_predictor') as infer_load:
            self.assertIsNone(_build_queue_load_predictor_shadow(config, None, torch.device('cpu'))[0])
            self.assertIsNone(_build_load_predictor_shadow(config, None, torch.device('cpu')))
            train_load.assert_not_called()
            infer_load.assert_not_called()
        with self.assertRaisesRegex(ValueError, 'Unsupported'):
            physical_backend_selected({'load_predictor': {'backend': 'silently_use_nn'}})

    def test_physical_launch_blocks_before_inference_environment_or_output(self) -> None:
        output = self.root / 'must_not_exist'
        config = {'load_predictor': {'backend': BACKEND}, 'physical_shadow_launch': {'authorized': False}}
        with patch('RL.infer.run_bladed_inference.WindTurbineEnv') as env:
            with self.assertRaisesRegex(ValueError, 'BLOCKED'):
                run_inference(config, {}, self.root / 'not.pt', output, 1, 1, torch.device('cpu'), 'fixture')
            env.assert_not_called()
        self.assertFalse(output.exists())

    def test_preflight_rejects_export_or_nonshadow_before_artifact_access(self) -> None:
        config = {'load_predictor': {'backend': BACKEND, 'mode': 'shadow', 'nn_active': False,
                                    'authorization': dict(RUNTIME_AUTHORIZATION)},
                  'env': {'queue_lifecycle_enabled': True, 'action_limiter_owner': 'controller'},
                  'bc': {'export_checkpoint_path': 'must_not_overwrite.pt'}}
        with self.assertRaisesRegex(ValueError, 'not export'):
            preflight_physical_config(config, project_root=PROJECT_ROOT)
        config['bc']['export_checkpoint_path'] = ''
        config['load_predictor']['mode'] = 'soft'
        with self.assertRaisesRegex(ValueError, 'shadow only'):
            preflight_physical_config(config, project_root=PROJECT_ROOT)

    def test_outcome_priority_never_treats_invalid_or_unsafe_as_complete(self) -> None:
        self.assertEqual(physical_outcome_tag({'condition_boundary': 1.0, 'env_timeout': 1.0}), 'timeout')
        self.assertEqual(physical_outcome_tag({}, unsafe_stop=True), 'unsafe_stop')
        self.assertEqual(physical_outcome_tag({'env_invalid_reward': 1.0}), 'invalid')
        self.assertEqual(physical_outcome_tag({'udp_frame_lag': 1.0}), 'frame_gap')

    def test_training_entry_shadow_on_off_keeps_all_sent_actions_identical(self) -> None:
        test_case = self

        class Env(queue_fixture._PredictorShadowQueueEnvironment):
            supports_physical_action = True

            def __init__(self):
                super().__init__(condition_steps=80)
                self.actions = []

            @property
            def current_payload(self):
                return test_case.payload(self._step_index)

            def step(self, action, on_action_sent=None, *, action_real=None):
                wire = np.asarray([float(f'{value:.6f}') for value in action_real])
                self.actions.append(wire.copy())
                self.last_successful_action_send = {'action_real': wire, 'frame_id': self._step_index,
                    'condition_id': 3, 'condition_instance_id': 101}
                if on_action_sent is not None:
                    on_action_sent(time.monotonic())
                self._step_index += 1
                self._last_sent_action = wire.copy()
                boundary = self._step_index >= self.condition_steps
                info = self._info(boundary=boundary, phase_elapsed_s=self._step_index * 0.02)
                info.update(udp_next_condition_instance_id=101.0, udp_next_frame_id=float(self._step_index),
                            udp_next_phase_code=3.0 if boundary else 2.0)
                return StepResult(self._obs.copy(), 1.0, boundary, info)

        config = {
            'data_schema_version': DATA_SCHEMA_VERSION, 'algorithm': 'td3',
            'agent': {'hidden_dims': [8], 'td3_bc_enabled': False}, 'env': {'reward': {}},
            'train': {'online_rl_enabled': False, 'online_eval_enabled': False, 'eval_every': 0,
                'batch_size': 2, 'step_log_every': 20,
                'condition_replay': {'capacity_per_condition': 128, 'min_valid_transitions_per_condition': 1}},
            'queue': {'condition_ids': [3], 'total_condition_rounds': 1, 'train_sim_s': 1.6,
                      'train_sim_tolerance_s': 1e-6}, 'offline_eval': {'enabled': False},
        }
        actions = []
        for enabled in (False, True):
            output = self.root / ('with_shadow' if enabled else 'without_shadow')
            output.mkdir()
            torch.manual_seed(713)
            np.random.seed(713)
            cfg = copy.deepcopy(config)
            if enabled:
                cfg['load_predictor'] = {'backend': BACKEND, 'mode': 'shadow'}
            env = Env()
            controller = SafeResidualController(
                {'action_low': [0.0, 0.0], 'action_high': [21586451.33303, 0.43],
                 'action_limiter_owner': 'controller', 'action_delta_low': [-90000.0, -0.000698],
                 'action_delta_high': [90000.0, 0.000698]},
                {'baseline_action': [10e6, 0.02], 'residual_limit': [1e6, 0.01],
                 'safe_action_low': [0.0, 0.0], 'safe_action_high': [21586451.33303, 0.43],
                 'ood_threshold': 100.0})
            controller.set_observation_distribution(np.zeros(3), np.ones(3))
            agent = TD3Agent(3, 2, cfg['agent'], torch.device('cpu'))
            agent.set_actor_obs_normalizer(torch.zeros(3), torch.ones(3))
            agent.set_critic_obs_normalizer(torch.zeros(3), torch.ones(3))
            replay = ConditionStratifiedReplayBuffer(3, 2, [3], capacity_per_condition=128,
                                                     min_per_condition_in_batch=1)
            logger = ExperimentLogger(str(output))
            shadow = PhysicalLoadShadow(self.artifacts, output_dir=output, queue_capacity=1024) if enabled else None
            try:
                # Only this synthetic in-memory test substitutes the real gate.
                with patch('RL.train_rl.validate_physical_launch'), \
                    patch('RL.train_rl.validate_physical_training_binding'), redirect_stdout(io.StringIO()):
                    summary = run_queue_training(agent, env, controller, replay, cfg, logger,
                        torch.device('cpu'), physical_load_shadow=shadow)
            finally:
                logger.flush()
                if shadow is not None:
                    result = shadow.close()
            actions.append(np.asarray(env.actions))
            self.assertEqual(summary['queue_load_predictor_shadow_enabled'], 0.0)
            if enabled:
                self.assertFalse(result['faulted'])
                self.assertEqual(result['counters']['submitted_frames'], 80)
                self.assertGreater(result['counters']['records_written'], 0)
        np.testing.assert_array_equal(actions[0], actions[1])

    def test_fake_udp_inference_physical_entry_does_not_load_nn_or_change_wire(self) -> None:
        accepted = []
        for enabled in (False, True):
            port = inference_fixture._pick_free_udp_port()
            config = inference_fixture._inference_config(port)
            config['env'].update(queue_lifecycle_enabled=True, require_condition_metadata=True,
                udp_protocol_version='bladed_udp_19_si_my_queue_v2')
            if enabled:
                config['load_predictor'] = {'backend': BACKEND, 'mode': 'shadow'}
                config['logging'] = {
                    'mode': 'compact_streaming',
                    'quantile_sample_capacity': 8,
                    'event_ring_capacity': 8,
                    'episode_ring_capacity': 4,
                    'queue_condition_ring_capacity': 4,
                    'queue_condition_max_record_bytes': 4096,
                    'queue_event_pre_frames': 1,
                    'queue_event_post_frames': 1,
                    'queue_event_window_capacity': 16,
                    'queue_event_window_max_record_bytes': 64 * 1024,
                    'queue_event_windows_max_total_bytes': 2 * 1024 * 1024,
                    'queue_event_full_state_enabled': True,
                    'debug_long_form_enabled': False,
                    'debug_max_rows': 0,
                    'debug_max_bytes': 0,
                }
            torch.manual_seed(711)
            agent = TD3Agent(OBS_DIM, 2, config['agent'], torch.device('cpu'))
            agent.set_actor_obs_normalizer(torch.zeros(OBS_DIM), torch.ones(OBS_DIM))
            agent.set_critic_obs_normalizer(torch.zeros(OBS_DIM), torch.ones(OBS_DIM))
            checkpoint = {'agent': agent.state_dict(), 'config': config, 'data_schema_version': DATA_SCHEMA_VERSION}
            checkpoint_path = self.root / f'udp_fixture_{enabled}.pt'
            torch.save(checkpoint, checkpoint_path)
            output = self.root / ('udp_with' if enabled else 'udp_without')
            server = inference_fixture.FakeBladedUdpServer('127.0.0.1', port, use_queue_envelope=True,
                queue_protocol_version='bladed_udp_19_si_my_queue_v2', switch_after_accepted_actions=2,
                queue_emit_end_before_switch=True)
            def make_shadow(cfg, env, *, output_dir, project_root):
                if not enabled:
                    return None
                shadow = PhysicalLoadShadow(self.artifacts, output_dir=output_dir)
                shadow.evaluation_reference = self.reference
                shadow.evaluation_reference_file_sha256 = self.reference_sha256
                return shadow
            server.start()
            try:
                # These explicit patches are confined to the local FakeUDP
                # fixture; they do not produce or save production PASS evidence.
                with patch('RL.infer.run_bladed_inference.validate_physical_launch',
                           return_value={'actor_checkpoint_sha256': file_sha256(checkpoint_path)}), \
                        patch('RL.infer.run_bladed_inference.preflight_physical_config'), \
                        patch('RL.infer.run_bladed_inference.build_physical_shadow', side_effect=make_shadow), \
                        patch('RL.infer.run_bladed_inference.load_candidate_conditioned_predictor') as nn, \
                        redirect_stdout(io.StringIO()):
                    summary = run_inference(config, checkpoint, checkpoint_path,
                        output, 1, 3, torch.device('cpu'), 'synthetic_fixture_only')
                    nn.assert_not_called()
            finally:
                server.stop()
            accepted.append(server.accepted_actions)
            if enabled:
                self.assertIsNone(summary['load_predictor_shadow_audit_jsonl'])
                self.assertTrue(summary['physical_load_shadow']['finalized'])
                self.assertFalse(summary['physical_load_shadow']['faulted'])
                self.assertEqual(summary['physical_load_shadow']['counters']['submitted_frames'], 2)
                self.assertTrue(summary['queue_event_capture']['event_windows_jsonl'])
                windows_path = Path(summary['queue_event_capture']['event_windows_jsonl'])
                self.assertTrue(windows_path.is_file())
                windows = [json.loads(line) for line in windows_path.read_text(encoding='utf-8').splitlines()]
                self.assertGreater(len(windows), 0)
                self.assertTrue(all(row['schema_version'] == 'queue_condition_event_window_v2' for row in windows))
                trigger = next(frame for row in windows for frame in row['frames']
                               if 'origin_payload' in frame)
                self.assertIn('I_AverageWindSpeed', trigger['origin_payload'])
                self.assertIn('I_RotorAzimuth', trigger['observed_payload'])
                self.assertEqual(len(trigger['wire_action_real']), 2)
                self.assertEqual(trigger['actor_policy_sha256'], file_sha256(checkpoint_path))
        self.assertEqual(accepted[0], accepted[1])

    def test_wire_snapshot_matches_bytes_and_is_immutable_to_reader(self) -> None:
        env = object.__new__(WindTurbineEnv)
        env._remote_addr = ('127.0.0.1', 12345)
        env._socket = Mock()
        packet = UdpPacket(payload={}, frame_id=7, sim_time=0.0, recv_monotonic_s=0.0,
            dropped_stale_frames=0, queue_epoch=1, condition_instance_id=101, condition_id=3,
            phase='TRAIN', phase_elapsed_s=0.0, queue_protocol_version='bladed_udp_19_si_my_queue_v2')
        env._send_action(np.asarray([12345678.1234567, 0.01234567]), packet)
        wire = env._socket.sendto.call_args.args[0].decode()
        np.testing.assert_array_equal(env.last_successful_action_send['action_real'],
                                      [float(v) for v in wire.split(',')[-2:]])
        snapshot = env.last_successful_action_send
        snapshot['action_real'][0] = -1.0
        self.assertGreater(env.last_successful_action_send['action_real'][0], 0.0)

    def development_records(self) -> Path:
        core = self.core()
        for index in range(101):
            core.observe(self.frame(index))
        rows = core.drain_records()
        for row in rows:
            row.update(identity_status='strict_case_matched', strict_physical_trace_complete=True,
                evaluation_reference_file_sha256=self.reference_sha256, pid_raw_sha256='c' * 64,
                scenario_id='wind3', time_window_start_s=0.0, time_window_end_s=self.reference_entry.time_window_end_s,
                expected_trace_signature=self.reference_entry.trace_signature,
                observed_trace_signature=self.reference_entry.trace_signature, trace_signature_match=1.0,
                expected_trace_sample_count=len(self.case.time), observed_trace_sample_count=len(self.case.time),
                fixed_simulation_time_offset_s=0.0)
        path = self.root / 'development_records.jsonl'
        path.write_text(''.join(json.dumps(row, allow_nan=False) + '\n' for row in rows), encoding='utf-8')
        return path

    def evaluate_records(self, path: Path, **kwargs):
        return evaluate_shadow_records(path, expected_bundle_sha256=self.artifacts.bundle_file_sha256,
            evaluation_reference=self.reference, expected_reference_sha256=self.reference_sha256, **kwargs)

    def test_evaluator_reports_only_matched_development_points_and_six_joint_margin(self) -> None:
        path = self.development_records()
        report = self.evaluate_records(path)
        self.assertGreater(report['counters']['development_eligible_records'], 0)
        self.assertIn('all|H0.10', report['point_metrics'])
        self.assertIn('all', report['pid_illustrative_joint_metrics'])
        self.assertFalse(report['events']['measured'])
        self.assertFalse(report['fresh_independent_audit'])
        self.assertEqual(report['decision'], 'DEVELOPMENT_DIAGNOSTIC_ONLY')
        json.dumps(report, allow_nan=False)

    def test_evaluator_rejects_duplicate_horizon_and_forged_plan_actor_flags(self) -> None:
        path = self.development_records()
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for mutation in ('duplicate', 'plan', 'actor'):
            changed = copy.deepcopy(rows)
            if mutation == 'duplicate':
                changed.append(changed[0])
            elif mutation == 'plan':
                changed[0]['executed_interval_actions'][0][0] = 21e6
            else:
                changed[0]['executed_interval_actor_policy_versions'][0] += 1
            path.write_text(''.join(json.dumps(row) + '\n' for row in changed), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'duplicate|ZOH|Actor'):
                self.evaluate_records(path)

    def test_evaluator_rejects_margin_upper_mismatch_and_formal_authorization(self) -> None:
        path = self.development_records()
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for mutation in ('margin', 'authorization'):
            changed = copy.deepcopy(rows)
            if mutation == 'margin':
                changed[0]['pid_empirical_upper_nm_by_blade_illustrative_only'][0] += 1.0
            else:
                changed[0]['authorization']['rl_margin_coverage_authorized'] = True
            path.write_text(''.join(json.dumps(row) + '\n' for row in changed), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'upper|DEVELOPMENT'):
                self.evaluate_records(path)

    def test_raw_invalidity_is_retained_without_changing_zero_filled_control_payload(self) -> None:
        env = object.__new__(WindTurbineEnv)
        env.strict_field_count = True
        from RL.envs.wind_turbine_env import BLADED_CSV_HEADER
        cells = [str(self.payload(0).get(name, 0.0)) for name in BLADED_CSV_HEADER]
        cells[BLADED_CSV_HEADER.index(LOAD_FIELDS[0])] = 'NaN'
        text = ','.join(cells)
        old = env._parse_payload(text)
        parsed = env._parse_payload(text, include_raw_validity=True)
        self.assertEqual(old, parsed[:3])
        self.assertEqual(parsed[0][LOAD_FIELDS[0]], 0.0)
        self.assertIn(LOAD_FIELDS[0], parsed[-1])
        shadow = Mock()
        env._last_successful_action_send = {'condition_id': 3, 'condition_instance_id': 101,
            'action_real': self.case.actions[0], 'frame_id': 0, 'invalid_payload_fields': parsed[-1]}
        submit_physical_sent_frame(shadow, env, payload=parsed[0], condition_id=3, condition_instance_id=101,
            sent_action_real=self.case.actions[0], actor_policy_version=0, actor_policy_sha256='',
            sent_monotonic_s=time.monotonic())
        shadow.submit_sent.assert_not_called()
        shadow.submission_failed.assert_called_once()
        env._pending_packet = UdpPacket(parsed[0], 1, 0.02, time.monotonic(), 0,
                                      invalid_payload_fields=parsed[-1])
        complete_physical_step(shadow, env, payload=self.payload(0), info={}, unsafe_stop=False)
        self.assertEqual(shadow.confirm_outcome.call_args.args[1], 'raw_input_invalid')

    def test_epoch_change_resets_history_and_terminal_never_crosses_epoch_or_phase(self) -> None:
        core = self.core()
        for index in range(56):
            core.observe(self.frame(index))
        frame = self.frame(56)
        core.observe(SentPhysicalFrame(**{**frame.__dict__, 'queue_epoch': 2}))
        self.assertEqual(core.history_count, 1)
        self.assertEqual(core.pending_count, 0)
        shadow = Mock()
        info = {'condition_boundary': 1.0, 'udp_condition_id': 3.0, 'udp_next_condition_id': 3.0,
            'udp_condition_instance_id': 101.0, 'udp_next_condition_instance_id': 101.0,
            'udp_queue_epoch': 1.0, 'udp_next_queue_epoch': 2.0, 'udp_next_phase_code': 3.0}
        complete_physical_step(shadow, SimpleNamespace(current_payload=self.payload(56)),
            payload=self.payload(55), info=info, unsafe_stop=False)
        shadow.submit_terminal.assert_not_called()
        info.update(udp_next_queue_epoch=1.0, udp_next_phase_code=2.0)
        complete_physical_step(shadow, SimpleNamespace(current_payload=self.payload(56)),
            payload=self.payload(55), info=info, unsafe_stop=False)
        shadow.submit_terminal.assert_not_called()

    def test_off_grid_end_delimiter_after_reference_endpoint_is_not_a_shadow_sample(self) -> None:
        endpoint = 600.0
        reference = SimpleNamespace(entry=lambda _condition: SimpleNamespace(
            timestamps_s=np.asarray([0.0, endpoint], dtype=np.float64)))
        info = {'condition_boundary': 1.0, 'udp_next_phase_code': 3.0,
            'udp_queue_epoch': 1.0, 'udp_next_queue_epoch': 1.0,
            'udp_condition_id': 3.0, 'udp_next_condition_id': 3.0,
            'udp_condition_instance_id': 101.0, 'udp_next_condition_instance_id': 101.0,
            'udp_next_frame_id': 30002.0}

        shadow = Mock()
        shadow.evaluation_reference = reference
        shadow.core = SimpleNamespace(dt=0.02)
        env = SimpleNamespace(current_payload=self.payload(0) | {'Time': 600.01},
                              current_payload_invalid_fields=())
        sent_payload = self.payload(0) | {'Time': endpoint}
        complete_physical_step(shadow, env, payload=sent_payload, info=info, unsafe_stop=False)
        shadow.confirm_outcome.assert_called_once_with(endpoint, 'complete')
        shadow.note_off_grid_terminal_marker.assert_called_once_with()
        shadow.submit_terminal.assert_not_called()

        # An END frame that is the missing in-window endpoint remains a real sample.
        shadow = Mock()
        shadow.evaluation_reference = reference
        shadow.core = SimpleNamespace(dt=0.02)
        env = SimpleNamespace(current_payload=self.payload(0) | {'Time': endpoint},
                              current_payload_invalid_fields=())
        sent_payload = self.payload(0) | {'Time': endpoint - 0.02}
        complete_physical_step(shadow, env, payload=sent_payload, info=info, unsafe_stop=False)
        shadow.submit_terminal.assert_called_once()
        shadow.note_off_grid_terminal_marker.assert_not_called()

    def test_close_timeout_is_nonfinal_and_a_later_join_can_finalize(self) -> None:
        shadow = PhysicalLoadShadow(self.artifacts, output_dir=self.root, queue_capacity=1024)
        entered, release = threading.Event(), threading.Event()
        original = shadow.core.model.predict
        def slow_prediction(*args, **kwargs):
            entered.set()
            release.wait(3.0)
            return original(*args, **kwargs)
        try:
            with patch.object(shadow.core.model, 'predict', side_effect=slow_prediction):
                for index in range(56):
                    shadow._put(('frame', self.frame(index), False))
                    shadow._put(('outcome', index * 0.02, 'complete'))
                self.assertTrue(entered.wait(2.0))
                summary = shadow.close(maximum_wait_s=0.0)
                self.assertTrue(summary['worker_alive'])
                self.assertFalse(summary['finalized'])
                self.assertIsNone(summary['records_file_sha256'])
                release.set()
                self.assertTrue(shadow.close()['finalized'])
        finally:
            release.set()
            shadow.close()

    def test_evaluator_rejects_cross_window_prefix_and_reference_mismatches(self) -> None:
        path = self.development_records()
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for mutation in ('prefix', 'run', 'reference'):
            changed = copy.deepcopy(rows)
            long = next(row for row in changed if row['horizon_s'] == 0.2)
            if mutation == 'prefix':
                long['actual_future_signed_my_nm_by_blade'][0][0] += 1.0
                long['actual_future_peak_nm_by_blade'] = np.abs(long['actual_future_signed_my_nm_by_blade']).max(axis=0).tolist()
            elif mutation == 'run':
                long['run_id'] = '0' * 32
            else:
                long['time_window_end_s'] += 0.02
            path.write_text(''.join(json.dumps(row) + '\n' for row in changed))
            with self.assertRaisesRegex(ValueError, 'prefix|different runs|reference'):
                self.evaluate_records(path)

    def test_evaluator_worker_receipt_exposes_budget_drops_and_incomplete_slots(self) -> None:
        path = self.development_records()
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        summary = {'run_id': rows[0]['run_id'], 'bundle_sha256': self.artifacts.bundle_file_sha256,
            'dynamic_model_file_sha256': self.artifacts.dynamic_file_sha256,
            'evaluation_reference_file_sha256': self.reference_sha256, 'records_file_sha256': file_sha256(path),
            'finalized': True, 'faulted': False, 'counters': {'records_written': len(rows),
                'bytes_written': path.stat().st_size, 'strict_complete_conditions': 1, 'write_budget_dropped': 3},
            'core_counters': {}}
        report = self.evaluate_records(path, worker_summary=summary, expected_slot_count=4)
        self.assertTrue(report['capture_completeness']['verified'])
        self.assertFalse(report['capture_completeness']['complete_capture'])
        summary['counters'].update(write_budget_dropped=0, strict_complete_conditions=4)
        summary['core_counters']['invalidation_raw_input_invalid'] = 1
        report = self.evaluate_records(path, worker_summary=summary, expected_slot_count=4)
        self.assertFalse(report['capture_completeness']['complete_capture'])
        summary['records_file_sha256'] = 'd' * 64
        with self.assertRaisesRegex(ValueError, 'worker summary'):
            self.evaluate_records(path, worker_summary=summary, expected_slot_count=4)

    def test_evaluator_cross_checks_complete_receipt_trace_and_logged_manifest_slot(self) -> None:
        path = self.development_records()
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        receipt = {name: rows[0][name] for name in ('queue_epoch', 'condition_instance_id', 'condition_id',
            'identity_status', 'strict_physical_trace_complete', 'evaluation_reference_file_sha256',
            'scenario_id', 'pid_raw_sha256', 'time_window_start_s', 'time_window_end_s',
            'expected_trace_signature', 'observed_trace_signature', 'expected_trace_sample_count',
            'observed_trace_sample_count', 'fixed_simulation_time_offset_s', 'trace_signature_match')}
        receipt['finish_slot_identity_match'] = True
        summary = {'run_id': rows[0]['run_id'], 'bundle_sha256': self.artifacts.bundle_file_sha256,
            'dynamic_model_file_sha256': self.artifacts.dynamic_file_sha256,
            'evaluation_reference_file_sha256': self.reference_sha256, 'records_file_sha256': file_sha256(path),
            'finalized': True, 'faulted': False, 'counters': {'records_written': len(rows),
                'bytes_written': path.stat().st_size, 'finished_conditions': 1, 'strict_complete_conditions': 1},
            'core_counters': {}, 'condition_receipts': [receipt]}
        kwargs = {'expected_slot_count': 1, 'expected_slot_identities': [(-1, 101, 3)]}
        self.assertTrue(self.evaluate_records(path, worker_summary=summary, **kwargs)['capture_completeness']['complete_capture'])
        for name, altered in (('strict_physical_trace_complete', False), ('finish_slot_identity_match', False),
                              ('observed_trace_signature', 'd' * 64), ('observed_trace_sample_count', 1)):
            with self.subTest(receipt=name):
                changed = copy.deepcopy(summary)
                changed['condition_receipts'][0][name] = altered
                self.assertFalse(self.evaluate_records(path, worker_summary=changed, **kwargs)['capture_completeness']['complete_capture'])
        changed_rows = copy.deepcopy(rows)
        for row in changed_rows:
            row['queue_epoch'] = 9
        path.write_text(''.join(json.dumps(row) + '\n' for row in changed_rows))
        summary['records_file_sha256'] = file_sha256(path)
        summary['counters']['bytes_written'] = path.stat().st_size
        with self.assertRaisesRegex(ValueError, 'record slot'):
            self.evaluate_records(path, worker_summary=summary, **kwargs)
        for row in rows:
            row.update(identity_status='unconfirmed', strict_physical_trace_complete=False,
                       development_point_eligible=False)
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        summary['records_file_sha256'] = file_sha256(path)
        summary['counters']['bytes_written'] = path.stat().st_size
        report = self.evaluate_records(path, worker_summary=summary, **kwargs)
        self.assertFalse(report['capture_completeness']['complete_capture'])
        self.assertEqual(report['counters']['unconfirmed_slot_records'], len(rows))

    def test_verified_checkpoint_load_binds_file_bytes(self) -> None:
        path = self.root / 'authorized_fixture.pt'
        torch.save({'agent': {'actor': {'weight': torch.ones(2)}}}, path)
        digest = file_sha256(path)
        checkpoint = load_verified_physical_checkpoint(path, digest)
        self.assertEqual(checkpoint['agent']['actor']['weight'].numel(), 2)
        with self.assertRaisesRegex(ValueError, 'SHA mismatch'):
            load_verified_physical_checkpoint(path, '0' * 64)

    def test_public_training_rejects_worker_with_unselected_backend_before_reset(self) -> None:
        env = SimpleNamespace(queue_lifecycle_enabled=True, reset=Mock())
        replay = ConditionStratifiedReplayBuffer(3, 2, [3], capacity_per_condition=8, min_per_condition_in_batch=1)
        with self.assertRaisesRegex(ValueError, 'backend mismatch'):
            run_queue_training(None, env, None, replay, {}, None, torch.device('cpu'),
                               physical_load_shadow=SimpleNamespace())
        env.reset.assert_not_called()

    def test_identical_checkpoint_state_cannot_hide_changed_td3_runtime_scalars(self) -> None:
        config = {'agent': {'hidden_dims': [8], 'actor_residual_target_scale': 0.25}}
        agent = TD3Agent(3, 2, config['agent'], torch.device('cpu'))
        validate_physical_agent_parameters(agent, config, torch.device('cpu'))
        for name, altered in (('actor_residual_target_scale', 1.0), ('gamma', 0.1), ('policy_delay', 99)):
            original = getattr(agent, name)
            setattr(agent, name, altered)
            with self.assertRaisesRegex(ValueError, 'runtime parameter mismatch'):
                validate_physical_agent_parameters(agent, config, torch.device('cpu'))
            setattr(agent, name, original)
        with self.assertRaisesRegex(ValueError, 'device mismatch'):
            validate_physical_agent_parameters(agent, config, torch.device('cuda:0'))


class PhysicalShadowProfileTests(unittest.TestCase):
    def test_actual_frozen_state20_binding_passes_without_udp_and_rejects_agent_or_reference_change(self) -> None:
        config = json.loads((REVIEWED / 'state20_physical_4wind_1round_600s_frozen_diagnostic.json').read_text())
        config['device'] = 'cpu'
        artifacts, reference = preflight_physical_config(config, project_root=PROJECT_ROOT)
        device = torch.device('cpu')
        agent = TD3Agent(20, 2, config['agent'], device)
        checkpoint = load_verified_physical_checkpoint(PROJECT_ROOT / config['bc']['resume_checkpoint'],
            config['state20_queue_contract']['bc_warmstart_checkpoint_sha256'])
        agent.load_state_dict(checkpoint['agent'])
        controller = SafeResidualController(config['env'], config['control'])
        controller.set_observation_distribution(agent.actor.obs_median.cpu().numpy(), agent.actor.obs_iqr.cpu().numpy())
        env = SimpleNamespace(initialization_config_sha256=document_sha256(config['env']))
        settings = config['load_predictor']
        with tempfile.TemporaryDirectory() as tmp:
            shadow = PhysicalLoadShadow(artifacts, output_dir=Path(tmp),
                max_total_bytes=settings['max_total_bytes'])
            shadow.evaluation_reference = reference
            shadow.evaluation_reference_file_sha256 = settings['evaluation_reference']['file_sha256']
            try:
                validate_physical_training_binding(config, env, controller, agent, shadow, device,
                                                   project_root=PROJECT_ROOT)
                agent.actor_residual_target_scale = 1.0
                with self.assertRaisesRegex(ValueError, 'runtime parameter mismatch'):
                    validate_physical_training_binding(config, env, controller, agent, shadow, device,
                                                       project_root=PROJECT_ROOT)
                agent.actor_residual_target_scale = config['agent']['actor_residual_target_scale']
                shadow.evaluation_reference = SimpleNamespace(manifest_sha256='0' * 64)
                with self.assertRaisesRegex(ValueError, 'worker/bundle/reference'):
                    validate_physical_training_binding(config, env, controller, agent, shadow, device,
                                                       project_root=PROJECT_ROOT)
            finally:
                shadow.close()

    def test_training_main_blocks_before_environment_or_bc_write(self) -> None:
        path = REVIEWED / 'state20_physical_4wind_4round_600s_training.json'
        with patch('sys.argv', ['train', '--config', str(path)]), \
                patch('RL.train_rl.WindTurbineEnv') as env, \
                patch('RL.train_rl._export_bc_checkpoint') as export, redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, 'BLOCKED'):
                training_entry.main()
        env.assert_not_called()
        export.assert_not_called()

    def test_initialized_training_failure_and_shadow_summary_failure_always_close_udp(self) -> None:
        env = SimpleNamespace(requires_single_instance=True, obs_dim=20, action_dim=2, close=Mock())
        with ExitStack() as resources, patch('RL.train_rl.build_agent', side_effect=RuntimeError('fixture_init_failure')), \
                redirect_stdout(io.StringIO()):
            resources.callback(env.close)
            with self.assertRaisesRegex(RuntimeError, 'fixture_init_failure'):
                training_entry._run_initialized_training({}, torch.device('cpu'), env, DATA_SCHEMA_VERSION,
                                                         None, resources)
        env.close.assert_called_once()
        env.close.reset_mock()
        shadow, logger = Mock(), Mock()
        shadow.close.side_effect = RuntimeError('fixture_close_failure')
        logger.save_summary.side_effect = OSError('fixture_disk_failure')
        with ExitStack() as resources, redirect_stdout(io.StringIO()), patch('sys.stderr', io.StringIO()):
            resources.callback(env.close)
            resources.callback(training_entry._close_physical_shadow_on_exit, shadow, logger)
        env.close.assert_called_once()

    def test_early_identity_abort_saves_partial_summary_preserves_primary_exception(self) -> None:
        helper = PhysicalLoadShadowTests()
        helper.setUp()
        self.addCleanup(helper.doCleanups)
        shadow = PhysicalLoadShadow(helper.artifacts, output_dir=helper.root, queue_capacity=1024)
        logger = ExperimentLogger(helper.root)
        shadow.evaluation_reference = helper.reference
        shadow.evaluation_reference_file_sha256 = helper.reference_sha256
        env_close = Mock()
        primary = training_entry.QueueCaseIdentityMismatch({
            'queue_case_identity_condition_id': 3.0,
            'queue_case_identity_instance_id': 101.0,
        })
        try:
            with self.assertRaises(training_entry.QueueCaseIdentityMismatch) as caught:
                with ExitStack() as resources:
                    resources.callback(env_close)
                    resources.callback(training_entry._close_physical_shadow_on_exit, shadow, logger)
                    # Stop before a complete slot; no terminal or finish receipt.
                    for index in range(64):
                        shadow.submit_sent(condition_id=3, condition_instance_id=101,
                            payload=helper.payload(index), sent_action_real=helper.case.actions[index],
                            pid_peak_abs_nm=helper.case.pid_peak_nm, actor_policy_version=0,
                            actor_policy_sha256='', sent_monotonic_s=time.monotonic(), frame_id=index)
                        shadow.confirm_outcome(index * 0.02, 'complete')
                    raise primary
            self.assertIs(caught.exception, primary)
            env_close.assert_called_once()
            summary = json.loads((helper.root / 'physical_load_shadow_summary.json').read_text(),
                                 parse_constant=lambda value: self.fail(value))
            self.assertEqual(summary['backend'], BACKEND)
            self.assertIs(summary['finalized'], True)
            self.assertIs(summary['faulted'], False)
            self.assertEqual(summary['counters']['submitted_frames'], 64)
            self.assertEqual(summary['condition_receipts'], [])
            self.assertEqual(summary['counters'].get('strict_complete_conditions', 0), 0)
            self.assertEqual(summary['records_file_sha256'], file_sha256(shadow.path))
            rows = [json.loads(line) for line in shadow.path.read_text().splitlines()]
            self.assertGreater(len(rows), 0)
            self.assertTrue(all(not row['strict_physical_trace_complete'] for row in rows))
            self.assertTrue(all(not row['development_point_eligible'] for row in rows))
            self.assertTrue(all(not row['independent_audit_eligible'] for row in rows))
        finally:
            shadow.close(maximum_wait_s=10.0)

    def test_generated_profiles_use_exact_reference_keep_bc_contract_and_block_launch(self) -> None:
        source = json.loads((PROJECT_ROOT / 'RL/configs/physical_shadow_state20_v1_final_checked/state20_physical_4wind_4round_600s_training.json').read_text())
        for name in ('4wind_4round_180s_integration', '4wind_1round_600s_frozen_diagnostic',
                     '4wind_4round_600s_training'):
            config = json.loads((REVIEWED / f'state20_physical_{name}.json').read_text())
            self.assertEqual(config['control_contract_fingerprint'], source['control_contract_fingerprint'])
            self.assertEqual(config['env']['reward'], source['env']['reward'])
            self.assertEqual(config['bc']['export_checkpoint_path'], '')
            self.assertEqual(config['load_predictor']['backend'], BACKEND)
            self.assertNotIn('predictor_artifact_path', config['load_predictor'])
            _, reference = preflight_physical_config(config, project_root=PROJECT_ROOT)
            self.assertEqual(reference.entry(11).time_window_end_s, config['queue']['train_sim_s'])
            with self.assertRaisesRegex(ValueError, 'BLOCKED'):
                validate_physical_launch(config, project_root=PROJECT_ROOT)

    def test_launch_identity_binds_effective_configuration_checkpoint_and_code(self) -> None:
        config = json.loads((REVIEWED / 'state20_physical_4wind_4round_600s_training.json').read_text())
        identity = physical_runtime_identity(config, PROJECT_ROOT)
        for key, value in (('device', 'cpu'), ('physical_shadow_run_role', {'stage': 'changed'})):
            altered = copy.deepcopy(config)
            altered[key] = value
            self.assertNotEqual(physical_runtime_identity(altered, PROJECT_ROOT), identity)
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / 'actor.pt'
            checkpoint.write_bytes(b'fixture_only_A')
            a = physical_runtime_identity(config, PROJECT_ROOT, checkpoint_path=checkpoint)
            checkpoint.write_bytes(b'fixture_only_B')
            b = physical_runtime_identity(config, PROJECT_ROOT, checkpoint_path=checkpoint)
            self.assertNotEqual(a, b)
        self.assertIn('RL/utils/operating_state.py', identity['runtime_code_sha256'])

    def test_checked_evidence_requires_target_not_synthetic_receipts(self) -> None:
        config = json.loads((REVIEWED / 'state20_physical_4wind_4round_600s_training.json').read_text())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            identity = physical_runtime_identity(config, PROJECT_ROOT)
            entries = {}
            for name in REQUIRED_GATES:
                path = root / f'{name}.json'
                path.write_text(json.dumps({'gate_id': name, 'status': 'PASS', 'runtime_identity': identity,
                    'evidence_kind': 'synthetic_linux_fixture_only', 'verified_at_utc': 'fixture',
                    'verification_reference': 'synthetic_not_real'}))
                entries[name] = {'status': 'PASS', 'runtime_identity': identity,
                                 'evidence_path': str(path), 'evidence_sha256': file_sha256(path)}
            evidence = root / 'evidence.json'
            evidence.write_text(json.dumps({'schema_version': LAUNCH_SCHEMA,
                'purpose': 'real_bladed_development_shadow_only', 'runtime_identity': identity,
                'policy_acceptance': False, 'gates': entries}))
            config['physical_shadow_launch'].update(authorized=True, evidence_path=str(evidence),
                                                     evidence_sha256=file_sha256(evidence))
            with self.assertRaisesRegex(ValueError, 'Synthetic/Linux'):
                validate_physical_launch(config, project_root=PROJECT_ROOT)


if __name__ == '__main__':
    unittest.main()