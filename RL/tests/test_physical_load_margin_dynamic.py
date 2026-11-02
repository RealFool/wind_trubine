from __future__ import annotations

import copy
from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from RL.demo_physical_load_margin_dynamic import (
    DevelopmentReader, calibration_records, development_gate, evaluate_validation,
    joint_metrics, run_experiment, select_dynamic_model, validation_events,
)
from RL.utils.physical_load_margin import (
    INPUT_FIELDS, REGIONS, GreyboxFitter, document_sha256, read_pid_case,
)
from RL.utils.physical_load_margin_dynamic import (
    AUTHORIZATION, MARGIN_METHODS, DynamicFitter, ErrorScaleFitter,
    dynamic_feature_names, dynamic_features, joint_margin_arrays, lag_indices,
    load_development_protocol, make_joint_calibration, physical_signed_trajectory,
    prepare_case, read_development_case, scale_features, solve_ridge,
)
from RL.utils.raw_dataset_contract import sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class PhysicalLoadMarginDynamicTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.spec, self.base = load_development_protocol(
            PROJECT_ROOT / 'RL/configs/physical_load_margin_dynamic_demo_v2.json', PROJECT_ROOT)
        self.base['reference_window_s'] = [0.0, 16.0]

    def write_case(self, name: str = 'data_windspeed3.csv', *, changed_future: bool = False,
                   invalid_row: int | None = None) -> Path:
        path = self.root / name
        with path.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=[*INPUT_FIELDS, 'I_PitchPosition2', 'forbidden_sensor'])
            writer.writeheader()
            for index in range(801):
                t = index * 0.02
                beta = 0.02 + 0.0005 * np.sin(0.44 * t)
                row = {field: 0.0 for field in INPUT_FIELDS}
                row.update({
                    'Time': t, 'I_RotorAzimuth': 0.7 * t, 'I_RotorSpeed': 0.7 + 0.002 * np.sin(0.61 * t),
                    'I_GeneratorSpeed': 0.7, 'I_AverageWindSpeed': 8.0 + 0.2 * np.sin(0.37 * t),
                    'I_RealPower': 10e6, 'Electrical power': 10e6, 'I_ElectricalTorque': 10e6,
                    'O_PcsTorqueDemand': 10e6 + 2000.0 * np.sin(0.2 * t),
                    'I_PitchPosition1': beta, 'O_PitchPositionDemand1': beta,
                    'I_NacelleForeAftAcceleration': 0.02 * np.sin(2.0 * t),
                    'I_NacelleSideSideAcceleration': 0.01 * np.cos(3.0 * t),
                    'I_PitchPosition2': 'ignored_not_numeric', 'forbidden_sensor': 'ignored',
                })
                for blade in range(3):
                    row[f'Blade root {blade + 1} My (Nm)'] = (30e6 - blade * 1e6 +
                        8e5 * np.sin(0.9 * t + blade * 2.0) + 2e5 * np.sin(4.5 * t + blade))
                if changed_future and index > 400:
                    row.update({'I_AverageWindSpeed': 22.0, 'I_RotorSpeed': 1.1, 'I_PitchPosition1': 0.3,
                                'O_PcsTorqueDemand': 21e6, 'O_PitchPositionDemand1': 0.3,
                                'Blade root 1 My (Nm)': 90e6})
                if index == invalid_row:
                    row['O_PcsTorqueDemand'] = float('nan')
                writer.writerow(row)
        return path

    def fitted(self):
        case = read_pid_case(self.write_case(), 3, self.base)
        fitter = GreyboxFitter(self.base)
        fitter.add_case(case)
        physics = fitter.finish()
        prepared = prepare_case(physics, case, 5)
        dynamic_fitter = DynamicFitter(physics, self.spec, (1, 2, 5, 10, 20, 40))
        dynamic_fitter.add_case(prepared)
        model = dynamic_fitter.finish(0.001)
        predictions = model.predict(case, prepared.origins, prepared=prepared)
        scale_fitter = ErrorScaleFitter(model, 'dynamic_fir')
        scale_fitter.add_case(prepared, predictions)
        return model, case, prepared, scale_fitter.finish()

    def fixture_protocol(self) -> Path:
        base_path = self.root / 'base_protocol.json'
        base_path.write_text(json.dumps(self.base), encoding='utf-8')
        spec = copy.deepcopy(self.spec)
        spec['base_protocol_path'] = str(base_path)
        spec['base_protocol_sha256'] = document_sha256(self.base)
        path = self.root / 'development_protocol.json'
        path.write_text(json.dumps(spec), encoding='utf-8')
        return path

    def fixture_raw(self) -> Path:
        raw = self.root / 'raw'
        raw.mkdir()
        template = self.write_case().read_bytes()
        for cid in range(3, 26):
            (raw / f'data_windspeed{cid}.csv').write_bytes(
                b'BLOCKED_AUDIT_MUST_NOT_BE_PARSED' if cid in self.base['split']['audit'] else template)
        return raw

    def synthetic_calibration(self, model, scale, count: int = 200):
        actual = np.zeros((count, 3, 3))
        for index in range(min(20, count)):
            dimension = index % 6
            actual[index, dimension // 3, dimension % 3] = 10.0
        actual[:, 2] = 1000.0  # Diagnostic 0.5s must NOT enter the joint score.
        record = {'condition_id': 8, 'region': np.zeros(count, dtype=int),
                  'eligible': np.ones(count, dtype=bool), 'actual': actual,
                  'predicted': np.zeros_like(actual), 'scale': np.full_like(actual, 2.0)}
        return record, make_joint_calibration([record], model, scale)

    def test_protocol_grid_identity_and_non_authorization(self) -> None:
        path = self.fixture_protocol()
        spec, base = load_development_protocol(path, PROJECT_ROOT)
        self.assertEqual(spec['authorization'], AUTHORIZATION)
        self.assertEqual(len(spec['dynamic_model']['history_lag_candidates_frames']) *
                         len(spec['dynamic_model']['ridge_fraction_candidates']), 6)
        self.assertEqual(document_sha256(base), spec['base_protocol_sha256'])
        for mutate in (
            lambda s: s['authorization'].update(soft_authorized=True),
            lambda s: s['data_policy'].update(read_old_audit=True),
            lambda s: s['development_gate'].update(integration_decision_always='PASS'),
            lambda s: s['conditional_scale'].update(history_lags_frames=[1, 5]),
            lambda s: s['dynamic_model'].update(history_lag_candidates_frames=[]),
            lambda s: s.update(base_protocol_sha256='0' * 64),
        ):
            altered = copy.deepcopy(spec)
            mutate(altered)
            path.write_text(json.dumps(altered), encoding='utf-8')
            with self.assertRaises(ValueError):
                load_development_protocol(path, PROJECT_ROOT)

    def test_old_audit_and_role_mismatch_block_before_file_access(self) -> None:
        with patch('RL.utils.physical_load_margin_dynamic.read_pid_case') as read:
            for cid, role in ((11, 'audit'), (11, 'train'), (5, 'train'), (3, 'calibration')):
                with self.assertRaises(ValueError):
                    read_development_case(self.root / f'data_windspeed{cid}.csv', cid, role, self.spec, self.base)
            with self.assertRaisesRegex(ValueError, 'filename/condition'):
                read_development_case(self.root / 'data_windspeed11.csv', 3, 'train', self.spec, self.base)
            read.assert_not_called()

    def test_signed_physics_is_exact_v1_physical_only(self) -> None:
        model, case, _, _ = self.fitted()
        origins = np.asarray([100, 300, 400])
        signed = physical_signed_trajectory(model.physics, case, origins)
        steps = np.rint(np.asarray(self.base['horizons_s']) / 0.02).astype(int)
        peaks = np.maximum.accumulate(np.abs(signed), axis=1)[:, steps - 1]
        v1 = model.physics.predict(case, origins, 0.1)
        np.testing.assert_array_equal(peaks, v1.peaks_nm['physical_only'])
        self.assertTrue(np.isfinite(signed).all())

    def test_future_signals_and_pid_reference_do_not_change_features_or_predictions(self) -> None:
        model, case, _, scale = self.fitted()
        changed = read_pid_case(self.write_case('changed.csv', changed_future=True), 3, self.base)
        origins = np.asarray([300, 390, 400])
        self.assertGreater(changed.pid_peak_nm[0], case.pid_peak_nm[0])
        for blade in range(3):
            old_x, old_valid = dynamic_features(model.physics, case, origins, model.lags, blade)
            new_x, new_valid = dynamic_features(model.physics, changed, origins, model.lags, blade)
            np.testing.assert_array_equal(old_x, new_x)
            np.testing.assert_array_equal(old_valid, new_valid)
            np.testing.assert_array_equal(scale_features(case, origins, blade)[0], scale_features(changed, origins, blade)[0])
        before = model.predict(case, origins)
        after = model.predict(changed, origins)
        np.testing.assert_array_equal(before.supported, after.supported)
        for method in before.peaks_nm:
            np.testing.assert_array_equal(before.peaks_nm[method], after.peaks_nm[method])
        np.testing.assert_array_equal(scale.predict(case, origins)[0], scale.predict(changed, origins)[0])
        np.testing.assert_array_equal(scale.predict(case, origins)[1], scale.predict(changed, origins)[1])

    def test_prediction_never_consumes_cached_future_labels(self) -> None:
        model, case, prepared, _ = self.fitted()
        before = model.predict(case, prepared.origins, prepared=prepared)

        class ForbiddenLabels(dict):
            def __getitem__(self, key):
                raise AssertionError('Prediction must never consume evaluation labels')

        prepared.labels = ForbiddenLabels()
        after = model.predict(case, prepared.origins, prepared=prepared)
        np.testing.assert_array_equal(before.peaks_nm['dynamic_fir'], after.peaks_nm['dynamic_fir'])
        np.testing.assert_array_equal(before.supported, after.supported)
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            model.predict(case, np.asarray([300]), prepared=prepared)
        with self.assertRaises(TypeError):
            model.predict(case, np.asarray([300]), candidate_actions=np.asarray([[10e6, 0.02]]))

    def test_dynamic_and_error_scale_fit_only_train(self) -> None:
        model, _, prepared, _ = self.fitted()
        for cid in (5, 8, 11):
            prepared.case.condition_id = cid
            with self.assertRaisesRegex(ValueError, 'Only frozen train'):
                DynamicFitter(model.physics, self.spec, model.lags).add_case(prepared)
            predictions = model.predict(prepared.case, prepared.origins, prepared=prepared)
            with self.assertRaisesRegex(ValueError, 'train residuals only'):
                ErrorScaleFitter(model, 'dynamic_fir').add_case(prepared, predictions)

    def test_history_reset_invalid_rows_and_feature_count(self) -> None:
        model, _, _, _ = self.fitted()
        case = read_pid_case(self.write_case('invalid.csv', invalid_row=400), 3, self.base)
        origins = np.asarray([20, 390, 400, 410, 450, 460])
        _, eligible = lag_indices(case, origins, model.lags)
        np.testing.assert_array_equal(eligible, [False, True, False, False, True, True])
        batch = model.predict(case, origins)
        self.assertFalse(batch.supported[2:5].any())
        self.assertEqual(batch.reasons[2, 0], 'invalid_input')
        self.assertEqual(batch.reasons[3, 0], 'warmup_or_discontinuity')
        self.assertEqual(len(dynamic_feature_names((1, 2, 5, 10))), 40)
        self.assertEqual(len(dynamic_feature_names(model.lags)), 52)
        self.assertEqual(dynamic_features(model.physics, case, origins, model.lags, 0)[0].shape, (6, 52))

    def test_history_surface_uses_origin_region_not_past_classifier_switch(self) -> None:
        model, case, _, _ = self.fitted()
        origin = np.asarray([300])
        before = dynamic_features(model.physics, case, origin, model.lags, 0)[0]
        case.region[260:300] = 2
        after = dynamic_features(model.physics, case, origin, model.lags, 0)[0]
        np.testing.assert_array_equal(before, after)

    def test_unfitted_and_outside_domain_are_unknown_not_ok(self) -> None:
        model, case, _, _ = self.fitted()
        origin = np.asarray([300])
        model.minimum[0] = 1e6
        model.maximum[0] = 2e6
        outside = model.predict(case, origin)
        self.assertFalse(outside.supported.any())
        self.assertEqual(outside.reasons[0, 0], 'dynamic_feature_outside_train_domain')
        model.coefficients[0] = np.nan
        unfitted = model.predict(case, origin)
        self.assertFalse(unfitted.supported.any())
        self.assertEqual(unfitted.reasons[0, 0], 'dynamic_group_unfitted')
        self.assertIsNone(model.to_document()['coefficients'][REGIONS[0]][0]['weights'])
        json.dumps(model.to_document(), allow_nan=False)

    def test_nonfinite_rollout_fails_closed_per_horizon_and_peak_is_monotonic(self) -> None:
        model, case, prepared, _ = self.fitted()
        before = model.predict(case, prepared.origins, prepared=prepared)
        known = before.supported.all(axis=1)
        self.assertTrue(known.any())
        self.assertTrue((np.diff(before.peaks_nm['dynamic_fir'][known], axis=1) >= 0.0).all())
        oi = int(np.flatnonzero(prepared.origins == 300)[0])
        self.assertTrue(before.supported[oi].all())
        prepared.physical_signed_nm[oi, 12, 0] = np.nan
        after = model.predict(case, prepared.origins, prepared=prepared)
        np.testing.assert_array_equal(after.supported[oi], [True, True, False])
        self.assertEqual(after.reasons[oi, 2], 'dynamic_prediction_nonfinite')
        prepared.physical_signed_nm[:] = np.nan
        fitter = DynamicFitter(model.physics, self.spec, model.lags)
        fitter.add_case(prepared)
        self.assertEqual(int(fitter.counts.sum()), 0)

    def test_error_scale_floor_cap_unknown_and_no_future(self) -> None:
        model, case, _, scale = self.fitted()
        scale.minimum[0] = -1e6
        scale.maximum[0] = 1e6
        scale.coefficients[0] = 0.0
        scale.coefficients[0, :, 0, :] = np.log(100.0)
        values, known = scale.predict(case, np.asarray([300]))
        self.assertTrue(known.all())
        np.testing.assert_allclose(values, self.spec['conditional_scale']['error_floor_nm'])
        scale.coefficients[0, :, 0, 1] = np.log(2.0 * self.spec['conditional_scale']['maximum_scale_nm'])
        values, known = scale.predict(case, np.asarray([300]))
        np.testing.assert_array_equal(known, [[True, False, True]])
        self.assertTrue(np.isnan(values[:, 1]).all())
        self.assertEqual(model.spec['conditional_scale']['fit_role'], 'train')

    def test_joint_score_is_max_all_six_not_individual95_and_excludes_long_window(self) -> None:
        model, _, _, scale = self.fitted()
        record, calibration = self.synthetic_calibration(model, scale)
        group = calibration['groups'][REGIONS[0]]
        self.assertEqual(group['origin_count'], 200)
        np.testing.assert_array_equal(group['legacy_per_blade95_nm'], np.zeros((2, 3)))
        self.assertEqual(group['joint_constant95_nm'], 10.0)
        self.assertEqual(group['joint_conditional95'], 5.0)
        legacy = np.broadcast_to(group['legacy_per_blade95_nm'], (200, 2, 3))
        entries = [{'actual': record['actual'][:, :2], 'predicted': record['predicted'][:, :2],
                    'margin': legacy, 'reference': np.full((200, 3), 100.0)}]
        metrics = joint_metrics(entries, self.base)
        self.assertAlmostEqual(metrics['joint_coverage'], 0.9)
        entries[0]['margin'] = np.full((200, 2, 3), group['joint_constant95_nm'])
        self.assertEqual(joint_metrics(entries, self.base)['joint_coverage'], 1.0)
        json.dumps(calibration, allow_nan=False)

    def test_calibration_role_nonfinite_and_missing_group(self) -> None:
        model, case, _, scale = self.fitted()
        record, calibration = self.synthetic_calibration(model, scale)
        record['condition_id'] = 5
        with self.assertRaisesRegex(ValueError, 'calibration cases only'):
            make_joint_calibration([record], model, scale)
        record['condition_id'] = 8
        record['actual'][0, 0, 0] = np.nan
        filtered = make_joint_calibration([record], model, scale)
        self.assertEqual(filtered['groups'][REGIONS[0]]['origin_count'], 199)
        json.dumps(filtered, allow_nan=False)
        _, small = self.synthetic_calibration(model, scale, 50)
        self.assertFalse(small['groups'][REGIONS[0]]['supported'])
        arrays, known = joint_margin_arrays(small, scale, case, np.asarray([300]))
        self.assertFalse(known.any())
        self.assertTrue(all(np.isnan(value).all() for value in arrays.values()))
        altered = copy.deepcopy(calibration)
        altered['groups'][REGIONS[0]]['joint_constant95_nm'] = -1.0
        self.assertFalse(joint_margin_arrays(altered, scale, case, np.asarray([300]))[1].any())

    def test_model_identity_binds_actual_physics_and_calibration(self) -> None:
        model, case, _, scale = self.fitted()
        _, calibration = self.synthetic_calibration(model, scale)
        before = document_sha256(model.to_document())
        self.assertEqual(model.to_document()['physics_model_sha256'], document_sha256(model.physics.to_document()))
        self.assertEqual(set(model.to_document()['external_fields']), set(INPUT_FIELDS))
        model.physics.coefficients[0, 0, 0] += 1.0
        self.assertNotEqual(before, document_sha256(model.to_document()))
        with self.assertRaisesRegex(ValueError, 'identity'):
            joint_margin_arrays(calibration, scale, case, np.asarray([300]))

    def test_ridge_solves_multi_future_targets_in_correct_scale(self) -> None:
        rng = np.random.default_rng(41)
        x = np.column_stack((np.ones(200), rng.normal(size=(200, 4))))
        beta = rng.normal(size=(5, 25))
        y = x @ beta
        solved, conditioning = solve_ridge(x.T @ x, x.T @ y, len(x), 1e-10)
        np.testing.assert_allclose(solved, beta, rtol=1e-7, atol=1e-7)
        self.assertTrue(np.isfinite(conditioning))

    def test_selection_no_candidate_preserves_all_diagnostics(self) -> None:
        model, case, _, _ = self.fitted()
        models = []
        for lags in self.spec['dynamic_model']['history_lag_candidates_frames']:
            fitter = DynamicFitter(model.physics, self.spec, tuple(lags))
            fitter.add_case(prepare_case(model.physics, case, 5))
            models.extend(fitter.finish(ridge) for ridge in self.spec['dynamic_model']['ridge_fraction_candidates'])
        models[-1].coefficients[:] = np.nan

        class Reader:
            def read(self, cid, role, stage):
                result = copy.deepcopy(case)
                result.condition_id = cid
                return result

        selected, diagnostics = select_dynamic_model(models, Reader(), self.spec, self.base)
        self.assertIsNone(selected)
        self.assertEqual(len(diagnostics['candidates']), 6)
        self.assertEqual(diagnostics['minimum_common_comparison_fraction'], 0.0)
        self.assertEqual(diagnostics['status'], 'INCONCLUSIVE_NO_ELIGIBLE_CANDIDATE')
        json.dumps(diagnostics, allow_nan=False)

    def test_cross_model_calibration_and_validation_use_common_scale_support(self) -> None:
        model, case, _, scale = self.fitted()
        raw = self.fixture_raw()
        reader = DevelopmentReader(raw, self.spec, self.base)
        old = copy.deepcopy(scale)
        old.method = 'v1_greybox'
        scale.minimum[0, :, 1] = 0.426
        scale.maximum[0, :, 1] = 0.427  # Subset even after the .02 domain floor.
        records = calibration_records(model, reader, {'v1_greybox': old, 'dynamic_fir': scale})
        for a, b in zip(records['v1_greybox'], records['dynamic_fir']):
            np.testing.assert_array_equal(a['eligible'], b['eligible'])
            self.assertGreater(int(a['eligible'].sum()), 0)
            self.assertLess(int(a['eligible'].sum()), len(a['eligible']) * 0.7)
        calibration = {method: make_joint_calibration(records[method], model, s)
                       for method, s in (('v1_greybox', old), ('dynamic_fir', scale))}
        output = self.root / 'joint_comparison'
        output.mkdir()
        with redirect_stdout(io.StringIO()):
            result = evaluate_validation(model, reader, self.spec, self.base,
                                         {'v1_greybox': old, 'dynamic_fir': scale}, calibration, output)
        for method in MARGIN_METHODS:
            a = result['joint_margins'][f'v1_greybox|{method}|all']
            b = result['joint_margins'][f'dynamic_fir|{method}|all']
            self.assertEqual(a['origin_count'], b['origin_count'])
        json.dumps(result, allow_nan=False)

    def test_events_do_not_cross_invalid_segment_or_count_censored_onset(self) -> None:
        _, case, _, _ = self.fitted()
        case.my_nm[:] = 1.0
        case.pid_peak_nm[:] = 100.0
        case.valid[100] = False
        case.segment[100] = -1
        case.segment[101:] = 1
        case.my_nm[101:104] = 100.0  # Already above threshold at reset: censored.
        case.my_nm[110:113] = 100.0
        case.my_nm[200:204] = 100.0
        case.valid[210] = False
        case.segment[210] = -1
        case.segment[211:] = 2
        case.my_nm[214:217] = 100.0
        origins = np.asarray([99, 105, 110, 205, 209, 214])
        peaks = np.zeros((6, 3, 3))
        peaks[0] = 100.0  # Cannot warn across reset.
        peaks[3] = 100.0  # After event end, not late.
        peaks[4] = 100.0  # Within 0.1s of next onset, but in the old segment.
        result = validation_events(case, origins, peaks, np.zeros((6, 2, 3)),
                                   np.ones(6, dtype=bool), self.base, [0, 1])
        for entry in result.values():
            self.assertEqual(entry['censored_onset_events'], 3)
            self.assertEqual(entry['total_events'], 6)
            self.assertEqual(entry['timely_events'], 0)
            self.assertEqual(entry['late_events'], 0)

    def test_complete_fixture_pipeline_audit_block_output_isolation_and_hashes(self) -> None:
        raw = self.fixture_raw()
        protocol = self.fixture_protocol()
        output = self.root / 'complete'
        original_open = Path.open
        blocked_names = {f'data_windspeed{cid}.csv' for cid in self.base['split']['audit']}

        def guard_open(path, *args, **kwargs):
            if path.name in blocked_names:
                raise AssertionError('Blocked audit content was opened')
            return original_open(path, *args, **kwargs)

        with patch.object(Path, 'open', guard_open), redirect_stdout(io.StringIO()):
            report = run_experiment(raw, protocol, output)
        self.assertIsNotNone(report['validation'])
        self.assertEqual(len(report['sources']), 19)
        self.assertEqual(report['unique_input_rows'], 19 * 801)
        self.assertEqual(report['blocked_audit_access_count'], 0)
        self.assertEqual(report['authorization'], AUTHORIZATION)
        self.assertEqual(report['gate']['integration_decision'], 'BLOCKED_NO_FRESH_INDEPENDENT_AUDIT')
        self.assertTrue(any(row['status'] == 'INCONCLUSIVE' for row in report['gate']['checks']))
        self.assertEqual(development_gate(report['validation'], report['protocol']), report['gate'])
        self.assertFalse(report['validation']['fresh_independent_audit'])
        self.assertEqual(report['validation']['strict_pid_exceedance']['positive_events'], 0)
        self.assertIsNone(report['validation']['strict_pid_exceedance']['recall'])
        self.assertTrue((output / 'validation_samples.csv').is_file())
        self.assertTrue((output / 'diagnostic.png').is_file())
        model_doc = json.loads((output / 'frozen_dynamic_model.json').read_text())
        self.assertEqual(document_sha256(model_doc), report['model_sha256'])
        for name, digest in report['artifact_file_sha256'].items():
            self.assertEqual(sha256_file(output / name), digest)
        self.assertIn('不是新的独立 audit', (output / 'README.md').read_text())
        with self.assertRaisesRegex(FileExistsError, 'already exists'):
            run_experiment(raw, protocol, output)

    def test_inconclusive_fixture_writes_failure_report_without_calibration(self) -> None:
        raw = self.fixture_raw()
        protocol = self.fixture_protocol()
        original_predict = DynamicFitter.finish

        def unfit(fitter, ridge):
            model = original_predict(fitter, ridge)
            model.coefficients[:] = np.nan
            return model

        with patch.object(DynamicFitter, 'finish', unfit), redirect_stdout(io.StringIO()):
            report = run_experiment(raw, protocol, self.root / 'inconclusive')
        self.assertEqual(len(report['sources']), 16)
        self.assertIsNone(report['model_sha256'])
        self.assertIsNone(report['validation'])
        self.assertEqual(report['calibration'], {})
        self.assertEqual(report['gate']['point_decision'], 'INCONCLUSIVE_NO_ELIGIBLE_CANDIDATE')
        self.assertTrue((self.root / 'inconclusive/report.json').is_file())
        self.assertFalse((self.root / 'inconclusive/frozen_joint_calibration.json').exists())
        json.dumps(report, allow_nan=False)


if __name__ == '__main__':
    unittest.main()