from __future__ import annotations

import copy
from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from RL.demo_physical_load_margin import (
    episode_metrics,
    feasibility_gate,
    load_episodes,
    prediction_origins,
    run_demo,
    sensitivity_records,
)
from RL.utils.constants import ACTION_VARIABLES, DISTURBANCE_VARIABLES, REWARD_VARIABLES, STATE_VARIABLES
from RL.utils.physical_load_margin import (
    INPUT_FIELDS,
    REGIONS,
    GreyboxFitter,
    GreyboxModel,
    PidCase,
    PredictionBatch,
    document_sha256,
    future_labels,
    load_protocol,
    margin_status,
    read_pid_case,
    surface_features,
    verify_static_project,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class PhysicalLoadMarginTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.protocol = load_protocol(PROJECT_ROOT / 'RL/configs/physical_load_margin_demo_v1.json')
        self.protocol['reference_window_s'] = [0.0, 4.0]

    def write_case(self, name: str = 'pid.csv', *, change_future: bool = False,
                   gap: bool = False, invalid_row: int | None = None) -> Path:
        path = self.root / name
        with path.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=[*INPUT_FIELDS, 'I_PitchPosition2', 'forbidden_new_sensor'])
            writer.writeheader()
            for index in range(201):
                if gap and index == 110:
                    continue
                timestamp = index * 0.02
                row = {field: 0.0 for field in INPUT_FIELDS}
                row.update({
                    'Time': timestamp, 'I_RotorAzimuth': 0.7 * timestamp,
                    'I_RotorSpeed': 0.7 + 0.001 * np.sin(timestamp),
                    'I_GeneratorSpeed': 0.7, 'I_RealPower': 10_000_000.0,
                    'Electrical power': 10_000_000.0, 'I_AverageWindSpeed': 8.0 + 0.05 * np.sin(timestamp),
                    'I_ElectricalTorque': 10_000_000.0, 'O_PcsTorqueDemand': 10_000_000.0,
                    'I_PitchPosition1': 0.02, 'O_PitchPositionDemand1': 0.02,
                    'Blade root 1 My (Nm)': 30_000_000.0 + 100_000.0 * np.sin(timestamp),
                    'Blade root 2 My (Nm)': 29_000_000.0 + 100_000.0 * np.sin(timestamp + 2.0),
                    'Blade root 3 My (Nm)': 28_000_000.0 + 100_000.0 * np.sin(timestamp + 4.0),
                    'I_PitchPosition2': 'not_an_allowed_numeric_input', 'forbidden_new_sensor': 'ignored',
                })
                if change_future and index > 100:
                    row.update({'I_AverageWindSpeed': 20.0, 'I_PitchPosition1': 0.3,
                                'O_PcsTorqueDemand': 21_000_000.0, 'O_PitchPositionDemand1': 0.3,
                                'Blade root 1 My (Nm)': 90_000_000.0})
                if invalid_row == index:
                    row['O_PcsTorqueDemand'] = float('nan')
                writer.writerow(row)
        return path

    def fitted(self) -> tuple[GreyboxModel, PidCase]:
        case = read_pid_case(self.write_case(), 3, self.protocol)
        fitter = GreyboxFitter(self.protocol)
        fitter.add_case(case)
        return fitter.finish(), case

    def test_whitelist_ignores_extra_columns(self) -> None:
        allowed = {'Time', *STATE_VARIABLES, *DISTURBANCE_VARIABLES, *REWARD_VARIABLES, *ACTION_VARIABLES}
        self.assertEqual(set(INPUT_FIELDS), allowed)
        self.assertNotIn('I_PitchPosition2', INPUT_FIELDS)
        case = read_pid_case(self.write_case(), 3, self.protocol)
        self.assertEqual(case.audit['ignored_columns'], ['I_PitchPosition2', 'forbidden_new_sensor'])
        self.assertEqual(case.audit['rows'], 201)

    def test_protocol_rejects_split_overlap_and_authorization(self) -> None:
        invalid = copy.deepcopy(self.protocol)
        invalid['split']['audit'][0] = 3
        path = self.root / 'invalid.json'
        path.write_text(json.dumps(invalid), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'disjoint'):
            load_protocol(path)
        invalid = copy.deepcopy(self.protocol)
        invalid['authorization']['soft_authorized'] = True
        path.write_text(json.dumps(invalid), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'never authorizes'):
            load_protocol(path)

    def test_exact_window_and_gap_rejected_for_reference(self) -> None:
        with self.assertRaisesRegex(ValueError, 'incomplete 20 ms'):
            read_pid_case(self.write_case(gap=True), 3, self.protocol)
        protocol = copy.deepcopy(self.protocol)
        protocol['reference_window_s'] = [0.0, 3.0]
        with self.assertRaisesRegex(ValueError, 'exact PID reference'):
            read_pid_case(self.write_case('other.csv'), 3, protocol)

    def test_fitter_enforces_train_only(self) -> None:
        case = read_pid_case(self.write_case(), 11, self.protocol)
        with self.assertRaisesRegex(ValueError, 'Only frozen train'):
            GreyboxFitter(self.protocol).add_case(case)

    def test_prediction_causal_under_changed_future_signals_and_labels(self) -> None:
        model, case = self.fitted()
        changed = read_pid_case(self.write_case('changed.csv', change_future=True), 3, self.protocol)
        origins = np.asarray([80, 90, 100])
        original_prediction = model.predict(case, origins, 0.3)
        changed_prediction = model.predict(changed, origins, 0.3)
        np.testing.assert_array_equal(original_prediction.supported, changed_prediction.supported)
        for method in original_prediction.peaks_nm:
            np.testing.assert_array_equal(original_prediction.peaks_nm[method], changed_prediction.peaks_nm[method])
        np.testing.assert_array_equal(original_prediction.actual_response_prediction, changed_prediction.actual_response_prediction)
        original_labels = future_labels(case, origins, self.protocol)
        changed_labels = future_labels(changed, origins, self.protocol)
        self.assertGreater(changed_labels['peak_nm'][-1, -1, 0], original_labels['peak_nm'][-1, -1, 0])
        self.assertGreater(changed_labels['plan_mismatch'][-1, -1], 0.05)

    def test_invalid_recorded_action_yields_unknown_and_rewarm(self) -> None:
        model, _ = self.fitted()
        case = read_pid_case(self.write_case('invalid.csv', invalid_row=110), 3, self.protocol)
        result = model.predict(case, np.asarray([100, 110, 111, 120, 165]), 0.3)
        self.assertEqual(result.reasons[1, 0], 'invalid_input')
        self.assertEqual(result.reasons[2, 0], 'warmup_or_discontinuity')
        self.assertFalse(result.supported[1].any())
        self.assertFalse(result.supported[2].any())
        self.assertGreater(case.segment[111], case.segment[100])
        self.assertTrue(case.warm[165])
        labels = future_labels(case, np.asarray([100]), self.protocol)
        self.assertFalse(labels['valid'][0, -1])

    def test_actuator_response_and_torque_saturation(self) -> None:
        model, case = self.fitted()
        params = self.protocol['static_parameters']
        candidate = np.asarray([[params['request_torque_max_nm'], 0.02]])
        result = model.predict(case, np.asarray([100]), 0.3, candidate)
        horizon = np.asarray(self.protocol['horizons_s'])
        expected = params['generator_actual_torque_max_nm'] - (
            params['generator_actual_torque_max_nm'] - case.values['I_ElectricalTorque'][100]
        ) * np.exp(-horizon / params['generator_time_constant_s'])
        np.testing.assert_allclose(result.actual_response_prediction[0, :, 0], expected)
        np.testing.assert_allclose(result.actual_response_prediction[0, :, 1], 0.02)
        self.assertTrue(np.all(result.actual_response_prediction[0, :, 0] < params['generator_actual_torque_max_nm']))

    def test_pitch_step_is_second_order_not_request_as_actual(self) -> None:
        model, case = self.fitted()
        params = self.protocol['static_parameters']
        result = model.predict(case, np.asarray([100]), 0.3, np.asarray([[10_000_000.0, 0.03]]))
        horizon = np.asarray(self.protocol['horizons_s'])
        wn = params['pitch_natural_frequency_rad_s']
        zeta = params['pitch_damping_ratio']
        wd = wn * np.sqrt(1.0 - zeta ** 2)
        expected = 0.03 - 0.01 * np.exp(-zeta * wn * horizon) * (
            np.cos(wd * horizon) + zeta * wn / wd * np.sin(wd * horizon)
        )
        np.testing.assert_allclose(result.actual_response_prediction[0, :, 1], expected)
        self.assertNotAlmostEqual(float(result.actual_response_prediction[0, 0, 1]), 0.03)

    def test_candidate_batch_is_stateless_and_order_invariant(self) -> None:
        model, case = self.fitted()
        candidates = np.asarray([[10_000_000.0, 0.02], [10_020_000.0, 0.02], [10_000_000.0, 0.02035]])
        saved_actions = case.actions.copy()
        saved_derivatives = case.derivatives.copy()
        batched = model.predict(case, np.full(3, 100), 0.3, candidates)
        reversed_batch = model.predict(case, np.full(3, 100), 0.3, candidates[::-1])
        np.testing.assert_allclose(batched.peaks_nm['physical_greybox'], reversed_batch.peaks_nm['physical_greybox'][::-1])
        for index in range(3):
            single = model.predict(case, np.asarray([100]), 0.3, candidates[index:index + 1])
            np.testing.assert_allclose(batched.peaks_nm['physical_greybox'][index], single.peaks_nm['physical_greybox'][0])
        np.testing.assert_array_equal(case.actions, saved_actions)
        np.testing.assert_array_equal(case.derivatives, saved_derivatives)

    def test_future_derivative_domain_is_checked(self) -> None:
        model, case = self.fitted()
        model.domain_minimum[:] = np.asarray([0.0, 0.01, -0.1, -4.0, -10.0, -10.0, -0.001, -1.0])
        model.domain_maximum[:] = np.asarray([40.0, 2.0, 0.6, 4.0, 10.0, 10.0, 0.001, 1.0])
        result = model.predict(case, np.asarray([100]), 0.3, np.asarray([[10_000_000.0, 0.08]]))
        self.assertFalse(result.supported.any())
        self.assertEqual(result.reasons[0, 0], 'future_rollout_outside_domain')

    def test_each_horizon_label_matures_separately(self) -> None:
        _, case = self.fitted()
        labels = future_labels(case, np.asarray([193]), self.protocol)
        np.testing.assert_array_equal(labels['valid'], [[True, False, False]])

    def test_margin_uses_cumulative_peak_and_never_claims_safe(self) -> None:
        reference = np.asarray([100.0, 100.0, 100.0])
        state, margin = margin_status(reference, np.asarray([101.0, 50.0, 50.0]), np.zeros(3), None, False)
        self.assertEqual(state, 'observed_exceeded_latched')
        self.assertEqual(margin[0], -1.0)
        state, _ = margin_status(reference, np.zeros(3), np.zeros(3), None, True)
        self.assertEqual(state, 'unknown_model_or_margin')
        state, margin = margin_status(reference, np.full(3, 95.0), np.full(3, 80.0), np.full(3, 10.0), True)
        self.assertEqual(state, 'model_within_reference_no_guarantee')
        np.testing.assert_array_equal(margin, np.full(3, 5.0))
        state, _ = margin_status(reference, np.zeros(3), np.full(3, 99.0), np.full(3, 2.0), True)
        self.assertEqual(state, 'model_upper_crosses_reference')

    def test_surface_dynamic_pressure_scaling(self) -> None:
        params = self.protocol['static_parameters']
        inputs = [np.asarray([8.0]), np.asarray([0.7]), np.asarray([0.02]), np.asarray([0.0]),
                  np.asarray([0.5]), np.asarray([0.0]), np.asarray([0.0])]
        original = surface_features(*inputs, 0, params, 1e7)
        inputs[0] *= 2.0
        inputs[1] *= 2.0
        scaled = surface_features(*inputs, 0, params, 1e7)
        np.testing.assert_allclose(scaled[:, :14], original[:, :14] * 4.0)

    def test_sensitivity_does_not_create_counterfactual_truth(self) -> None:
        model, case = self.fitted()
        protocol = copy.deepcopy(self.protocol)
        protocol['evaluation']['candidate_sensitivity']['audit_times_s'] = [2.0]
        calibration = {'groups': {
            f'{region}|{horizon:.2f}': {'methods': {'physical_greybox': {'supported': True, 'error_nm': [1.0, 1.0, 1.0]}}}
            for region in REGIONS for horizon in protocol['horizons_s']
        }}
        records = sensitivity_records(model, case, calibration, 0.3, protocol)
        self.assertEqual(len(records), 15)
        self.assertTrue(all(not item['has_physical_counterfactual_label'] for item in records))
        self.assertTrue(all(item['candidate_safety_status'] == 'unknown_unexecuted_action'
                            for item in records if item['candidate'] != 'pid_zoh'))
        self.assertTrue(all('actual_future_peak_nm' not in item for item in records))

    def test_sensitivity_invalid_and_unfitted_origins_are_json_null(self) -> None:
        model, _ = self.fitted()
        case = read_pid_case(self.write_case('invalid.csv', invalid_row=100), 3, self.protocol)
        protocol = copy.deepcopy(self.protocol)
        protocol['evaluation']['candidate_sensitivity']['audit_times_s'] = [2.0]
        records = sensitivity_records(model, case, {'groups': {}}, 0.3, protocol)
        self.assertEqual(records[0]['candidate_safety_status'], 'unknown_invalid_input')
        self.assertIsNone(records[0]['action'])
        json.dumps(records, allow_nan=False)
        case = read_pid_case(self.write_case('unfitted.csv'), 3, self.protocol)
        case.region[:] = 2
        calibration = {'groups': {
            f'{region}|{horizon:.2f}': {'methods': {'physical_greybox': {'supported': True, 'error_nm': [1.0, 1.0, 1.0]}}}
            for region in REGIONS for horizon in protocol['horizons_s']
        }}
        records = sensitivity_records(model, case, calibration, 0.3, protocol)
        self.assertEqual(len(records), 15)
        self.assertTrue(all(item['predicted_peak_nm'] is None for item in records))
        json.dumps(records, allow_nan=False)

    def test_alert_after_event_end_is_not_attributed_as_late(self) -> None:
        _, case = self.fitted()
        case.my_nm[:] = 1.0
        case.my_nm[100:103] = 100.0
        case.pid_peak_nm[:] = 100.0
        origins = np.asarray([95, 100, 105])
        peaks = np.zeros((3, 3, 3))
        peaks[2] = 100.0
        batch = PredictionBatch(origins, {'physical_greybox': peaks}, np.ones((3, 3), dtype=bool),
                                np.full((3, 3), 'ok'), np.zeros((3, 3, 2)), np.zeros((3, 3)))
        calibration = {'groups': {
            f'{region}|{horizon:.2f}': {'methods': {'physical_greybox': {'supported': True, 'error_nm': [0.0, 0.0, 0.0]}}}
            for region in REGIONS for horizon in self.protocol['horizons_s']
        }}
        events = episode_metrics(case, batch, calibration, 'physical_greybox', 0, 0.95, self.protocol)
        self.assertEqual(events['total_events'], 3)
        self.assertEqual(events['timely_events'], 0)
        self.assertEqual(events['late_events'], 0)
        peaks[1] = 100.0
        events = episode_metrics(case, batch, calibration, 'physical_greybox', 0, 0.95, self.protocol)
        self.assertEqual(events['late_events'], 3)

    def test_episode_merge_and_origin_stride(self) -> None:
        times = np.arange(0.0, 2.0, 0.02)
        load = np.zeros_like(times)
        load[10:13] = 1.0
        load[20:23] = 1.0
        load[80:83] = 1.0
        episodes = load_episodes(times, load, 0.95, 0.5)
        self.assertEqual(len(episodes), 2)
        self.assertAlmostEqual(episodes[0][0], 0.2)
        self.assertAlmostEqual(episodes[0][1], 0.44)
        _, case = self.fitted()
        np.testing.assert_array_equal(prediction_origins(case, self.protocol)[:3], [0, 5, 10])

    def test_static_project_binding_and_identity_hash(self) -> None:
        protocol = load_protocol(PROJECT_ROOT / 'RL/configs/physical_load_margin_demo_v1.json')
        result = verify_static_project(PROJECT_ROOT / protocol['static_parameters']['source_project'], protocol)
        self.assertFalse(result['ct_cp_table_loaded'])
        self.assertEqual(len(result['sha256']), 64)
        changed = copy.deepcopy(protocol)
        changed['static_parameters']['generator_time_constant_s'] = 0.3
        self.assertNotEqual(document_sha256(protocol), document_sha256(changed))
        with self.assertRaisesRegex(ValueError, 'parameter mismatch'):
            verify_static_project(PROJECT_ROOT / protocol['static_parameters']['source_project'], changed)

    def test_complete_fixture_demo_preserves_gate_and_isolated_output(self) -> None:
        raw = self.root / 'raw'
        raw.mkdir()
        template = self.write_case().read_bytes()
        for cid in range(3, 26):
            (raw / f'data_windspeed{cid}.csv').write_bytes(template)
        protocol_path = self.root / 'protocol.json'
        protocol_path.write_text(json.dumps(self.protocol), encoding='utf-8')
        output = self.root / 'result'
        with redirect_stdout(io.StringIO()):
            report = run_demo(raw, protocol_path, output)
        self.assertEqual(report['gate']['decision'], 'REVIEW_DO_NOT_INTEGRATE')
        self.assertTrue(any(check['status'] == 'INCONCLUSIVE' for check in report['gate']['checks']))
        self.assertFalse(report['authorization']['soft_authorized'])
        self.assertFalse(report['authorization']['counterfactual_validation'])
        self.assertEqual(report['audit']['strict_pid_exceedance']['positive_events'], 0)
        self.assertIsNone(report['audit']['strict_pid_exceedance']['recall'])
        self.assertTrue((output / 'audit_samples.csv').is_file())
        self.assertTrue((output / 'diagnostic.png').is_file())
        self.assertEqual(feasibility_gate(report['audit'], report['calibration'], self.protocol), report['gate'])
        self.assertEqual(json.loads((output / 'report.json').read_text())['protocol_sha256'], document_sha256(self.protocol))
        self.assertIn('above_rated / ', (output / 'README.md').read_text())
        with self.assertRaisesRegex(FileExistsError, 'already exists'):
            run_demo(raw, protocol_path, output)


if __name__ == '__main__':
    unittest.main()