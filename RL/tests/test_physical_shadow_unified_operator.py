from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4

from RL.build_physical_load_shadow_profiles import build_profiles
from RL.utils.physical_load_shadow_artifact import file_sha256, strict_json
from RL.utils.physical_load_shadow_launch import (
    OPERATOR_23WIND_PROFILE_BUILD_REPORT,
    OPERATOR_23WIND_RELEASE_ID,
    OPERATOR_23WIND_ROUTE,
    OPERATOR_23WIND_TEMPLATE,
    _operator_23wind_schedule_contract,
    _operator_23wind_confirmation,
    operator_resource_inventory,
    physical_runtime_identity,
    validate_operator_23wind_scope,
)
from RL.utils.physical_queue_schedule import (
    DEFAULT_FULL_23WIND_ROUNDS,
    FULL_23WIND_PROFILE,
    MAX_FULL_23WIND_ROUNDS,
    build_full_23wind_manifest,
    configure_full_23wind_rounds,
)
from RL.utils.queue_manifest import expand_queue_manifest


ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_PATH = ROOT / OPERATOR_23WIND_TEMPLATE
REPORT_PATH = ROOT / OPERATOR_23WIND_PROFILE_BUILD_REPORT


class UnifiedPhysicalOperatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not TEMPLATE_PATH.is_file() or not REPORT_PATH.is_file():
            raise unittest.SkipTest('Unified R2 profile has not been built yet')
        cls.template = strict_json(TEMPLATE_PATH)
        cls.report = strict_json(REPORT_PATH)

    def _validate(self, config: dict) -> dict:
        candidate = copy.deepcopy(config)
        candidate['queue']['case_identity_validation']['wind_mismatch_policy'] = 'warn_development_only'
        return validate_operator_23wind_scope(candidate, ROOT)

    def test_default_release_is_one_blocked_four_round_profile(self) -> None:
        entry = self.report['profiles'][FULL_23WIND_PROFILE]
        contract = self._validate(self.template)
        manifest = strict_json(ROOT / self.template['queue']['manifest_path'])
        slots = expand_queue_manifest(manifest)

        self.assertEqual(self.template['release_id'], OPERATOR_23WIND_RELEASE_ID)
        self.assertEqual(self.report['release_id'], OPERATOR_23WIND_RELEASE_ID)
        self.assertEqual(self.template['queue']['total_condition_rounds'], DEFAULT_FULL_23WIND_ROUNDS)
        self.assertEqual(len(slots), 23 * DEFAULT_FULL_23WIND_ROUNDS)
        self.assertEqual(contract['rounds'], DEFAULT_FULL_23WIND_ROUNDS)
        self.assertEqual(contract['slot_count'], 92)
        self.assertEqual(contract['expected_scheduled_updates'], 69_000)
        reward = self.template['env']['reward']
        self.assertNotIn('negative_power_weight', reward)
        self.assertNotIn('lambda_speed', reward)
        self.assertNotIn('shaft_mode', reward)
        self.assertNotIn('acc_metric_mode', reward)
        self.assertNotIn('rotor_speed_reference_schedule_path', reward)
        self.assertNotIn('region_rotor_speed_weight', reward)
        self.assertNotIn('rotor_speed_variation_weight', reward)
        self.assertNotIn('sustained_low_rotor_speed_weight', reward)
        self.assertIn('negative_power_abnormal', reward['state_conditioning']['weights'])
        self.assertEqual(self.template['control']['low_rotor_speed_fallback_threshold'], 0.4)
        self.assertEqual(entry['slots'], 92)
        self.assertEqual(entry['config_sha256'], file_sha256(TEMPLATE_PATH))
        self.assertEqual(entry['manifest_sha256'], file_sha256(ROOT / self.template['queue']['manifest_path']))
        self.assertIs(self.template['physical_shadow_launch']['authorized'], False)
        self.assertIs(self.template['physical_shadow_run_role']['policy_acceptance'], False)
        self.assertIs(self.template['physical_shadow_run_role']['soft_enabled'], False)
        self.assertIs(self.template['physical_shadow_run_role']['hard_enabled'], False)

    def test_one_profile_accepts_multiple_round_values_with_closed_budgets(self) -> None:
        for rounds, slots_expected, updates_expected in (
            (4, 92, 69_000),
            (6, 138, 115_000),
            (MAX_FULL_23WIND_ROUNDS, 184, 161_000),
        ):
            with self.subTest(rounds=rounds), tempfile.TemporaryDirectory(
                prefix='unified_rounds_', dir=ROOT / 'RL/configs'
            ) as tmp:
                run_dir = Path(tmp)
                manifest_path = run_dir / 'queue_manifest.json'
                source_manifest = strict_json(ROOT / self.template['queue']['manifest_path'])
                manifest = build_full_23wind_manifest(source_manifest, rounds)
                manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding='utf-8')
                config = copy.deepcopy(self.template)
                config['experiment_name'] = f'rounds_fixture_{rounds}'
                config['output_dir'] = 'RL/experiments'
                config['append_timestamp'] = True
                config['physical_shadow_launch'] = {
                    'authorized': False,
                    'route': OPERATOR_23WIND_ROUTE,
                    'evidence_path': '',
                    'evidence_sha256': '',
                    'decision': 'BLOCKED_PENDING_OPERATOR_CONFIRMATION_FINALIZE',
                }
                configure_full_23wind_rounds(config, manifest_path, rounds, project_root=ROOT)
                config['queue']['manifest_path'] = str(manifest_path)
                config['release_id'] = OPERATOR_23WIND_RELEASE_ID
                config['queue']['case_identity_validation']['wind_mismatch_policy'] = 'warn_development_only'
                contract = _operator_23wind_schedule_contract(config, ROOT)
                self.assertEqual(contract['rounds'], rounds)
                self.assertEqual(contract['slot_count'], slots_expected)
                self.assertEqual(contract['expected_scheduled_updates'], updates_expected)
                self.assertGreaterEqual(
                    config['load_predictor']['max_records'],
                    contract['shadow_expected_record_upper_bound'],
                )
                self.assertLessEqual(config['load_predictor']['max_total_bytes'], 18 * 1024**3)

    def test_rounds_above_shadow_budget_are_rejected(self) -> None:
        source_manifest = strict_json(ROOT / self.template['queue']['manifest_path'])
        with self.assertRaisesRegex(ValueError, 'rounds must be in'):
            build_full_23wind_manifest(source_manifest, MAX_FULL_23WIND_ROUNDS + 1)

    def test_operator_confirmation_attests_the_actual_round_count_and_schedule(self) -> None:
        contract = self._validate(self.template)
        identity = physical_runtime_identity(self.template, ROOT)
        confirmation = _operator_23wind_confirmation(contract, identity)
        self.assertEqual(confirmation['manual_batch_rounds_confirmed'], contract['rounds'])
        self.assertEqual(confirmation['manual_batch_slot_count_confirmed'], contract['slot_count'])
        self.assertEqual(confirmation['manual_batch_schedule_sha256_confirmed'], contract['schedule_sha256'])
        self.assertFalse(confirmation.get('policy_acceptance', False))

    def test_resource_inventory_does_not_require_removed_rotor_speed_schedule(self) -> None:
        with tempfile.TemporaryDirectory(prefix='operator_resource_inventory_') as tmp:
            root = Path(tmp)
            data = root / 'data'
            (data / 'train').mkdir(parents=True)
            (data / 'val').mkdir()
            (data / 'train' / 'train.csv').write_text('row\n1\n', encoding='utf-8')
            (data / 'val' / 'val.csv').write_text('row\n1\n', encoding='utf-8')
            (data / 'manifest.json').write_text('{}\n', encoding='utf-8')
            (root / 'static.json').write_text('{}\n', encoding='utf-8')
            (root / 'resolved.json').write_text('{}\n', encoding='utf-8')
            config = {
                'control_contract_static_path': 'static.json',
                'resolved_control_contract_path': 'resolved.json',
                'data': {'root_dir': 'data'},
                'b4_residual_contract': {
                    'data_manifest_sha256': file_sha256(data / 'manifest.json'),
                },
                'env': {'reward': {}},
            }
            with patch(
                'RL.utils.control_contract.resolve_configured_control_contract',
                return_value=({'artifact_paths': {}}, 'fixture-fingerprint'),
            ):
                inventory = operator_resource_inventory(config, root)

        self.assertEqual(len(inventory), 5)
        self.assertTrue(any(path.endswith('/manifest.json') for path in inventory))

    def test_receipt_validator_rejects_round_count_reuse(self) -> None:
        from RL.utils.physical_load_shadow_launch import validate_operator_23wind_development

        config = copy.deepcopy(self.template)
        config['queue']['case_identity_validation']['wind_mismatch_policy'] = 'warn_development_only'
        config['physical_shadow_launch'] = {
            'authorized': True,
            'route': OPERATOR_23WIND_ROUTE,
            'decision': 'OPERATOR_CONFIRMED_23WIND_DEVELOPMENT_ONLY_NOT_SIX_GATE_VERIFIED',
            'evidence_path': str(
                ROOT / 'RL/configs' / config['experiment_name'] / 'operator_physical_receipt.json'
            ),
            'evidence_sha256': '0' * 64,
        }
        identity = physical_runtime_identity(config, ROOT)
        receipt = {
            'schema_version': 'physical_shadow_operator_23wind_round600_training_receipt_v2',
            'route': OPERATOR_23WIND_ROUTE,
            'release_id': OPERATOR_23WIND_RELEASE_ID,
            'rounds': 6,
            'slots': 138,
        }
        with self.assertRaisesRegex(ValueError, 'receipt must be colocated|scope/identity/attestation mismatch'):
            validate_operator_23wind_development(
                config, receipt, identity, ROOT, checkpoint_path=None, baseline_only=False,
            )

    def test_old_r1_remains_archived_unchanged_and_blocked(self) -> None:
        old_release_path = ROOT / 'RL/configs/WT15_STATE20_RL_20261022_R1/release_manifest.json'
        old_release = strict_json(old_release_path)
        self.assertEqual(old_release['release_id'], 'WT15_STATE20_RL_20261022_R1')
        self.assertEqual(old_release['profiles']['primary_23wind6_training_development']['schedule']['rounds'], 6)
        self.assertIs(old_release['release_ready'], False)
        for value in old_release['authorization'].values():
            self.assertIs(value, False)

    def test_run_id_preparation_defaults_to_four_rounds_without_starting(self) -> None:
        from RL.prepare_physical_operator_run import prepare_operator_run

        run_id = f'unified_operator_fixture_{uuid4().hex[:12]}'
        package = prepare_operator_run(run_id)
        config_dir = ROOT / 'RL/configs' / run_id
        self.addCleanup(shutil.rmtree, config_dir, ignore_errors=True)
        report = strict_json(config_dir / 'run_preparation_report.json')
        config = strict_json(config_dir / 'train.json')
        self.assertEqual(package['rounds'], DEFAULT_FULL_23WIND_ROUNDS)
        self.assertEqual(package['operator_route_contract']['slot_count'], 92)
        self.assertIs(report['physical_shadow_launch_authorized'], False)
        self.assertIs(report['operator_receipt_written'], False)
        self.assertFalse((config_dir / 'operator_physical_receipt.json').exists())
        self.assertFalse((ROOT / 'RL/experiments' / run_id).exists())
        self.assertTrue(config['append_timestamp'])
        self.assertEqual(config['physical_shadow_launch']['route'], OPERATOR_23WIND_ROUTE)

    def test_six_round_run_uses_same_route_and_receipt_binds_138_slots(self) -> None:
        from RL.finalize_physical_operator_run import finalize_operator_run
        from RL.prepare_physical_operator_run import prepare_operator_run

        run_id = f'unified_operator_six_round_fixture_{uuid4().hex[:12]}'
        package = prepare_operator_run(run_id, rounds=6)
        config_dir = ROOT / 'RL/configs' / run_id
        experiment_root = ROOT / 'RL/experiments'
        self.addCleanup(shutil.rmtree, config_dir, ignore_errors=True)
        try:
            finalized = finalize_operator_run(run_id)
            receipt = strict_json(config_dir / 'operator_physical_receipt.json')
            self.assertEqual(package['operator_route_contract']['slot_count'], 138)
            self.assertEqual(receipt['rounds'], 6)
            self.assertEqual(receipt['slots'], 138)
            self.assertEqual(receipt['route'], OPERATOR_23WIND_ROUTE)
            self.assertTrue(receipt['operator_attestation']['manual_batch_schedule_sha256_confirmed'])
            self.assertIs(receipt['policy_acceptance'], False)
            self.assertIs(receipt['independently_verified'], False)
            self.assertTrue(finalized['physical_shadow_launch_authorized'])
            self.assertFalse(any(experiment_root.glob(f'{run_id}_*')))
        finally:
            (config_dir / 'operator_physical_receipt.json').unlink(missing_ok=True)
            (config_dir / 'train.json').unlink(missing_ok=True)
            (config_dir / 'run_preparation_report.json').unlink(missing_ok=True)
            (config_dir / 'queue_manifest.json').unlink(missing_ok=True)
            (config_dir / 'physical_load_shadow_bundle.json').unlink(missing_ok=True)
            config_dir.rmdir()


if __name__ == '__main__':
    unittest.main()
