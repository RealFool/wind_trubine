"""Execute documented independent offline commands; never start training/UDP."""
import copy
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import textwrap
import unittest
from unittest.mock import patch
from uuid import uuid4

from RL.prepare_physical_shadow_development import check_resources
from RL.utils.physical_load_shadow_artifact import file_sha256, strict_json
from RL.utils.physical_load_shadow_launch import validate_physical_launch, OPERATOR_TEMPLATE
from RL.utils.physical_load_shadow_launch import FROZEN_R4_ROUTE
from RL.utils.physical_load_shadow_launch import validate_round4_frozen_actor_development
from RL.utils.logger import ExperimentLogger

ROOT = Path(__file__).resolve().parents[2]
NOTES = ROOT / 'doc/训练操作笔记.md'


def execute_documented_preparation(run_id):
    """Run steps 2–8 verbatim with the paths/environment established by step 1."""
    config_dir = ROOT / 'RL/configs' / run_id
    experiment_dir = ROOT / 'RL/experiments' / run_id
    if experiment_dir.exists():
        raise ValueError('Experiment directory must stay fresh')
    config_dir.mkdir(exist_ok=False)
    env = dict(os.environ, RUN_ID=run_id, CONFIG_DIR=str(config_dir),
               RUN_CONFIG=str(config_dir / 'train.json'), EXP_DIR=str(experiment_dir),
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    for name in ('RL_EXPERIMENT_SUFFIX', 'RL_DATA_ROOT', 'RL_DYNAMICS_CHECKPOINT'):
        env.pop(name, None)
    text = NOTES.read_text(encoding='utf-8')
    for number in range(2, 9):
        section = re.search(rf'^{number}\. .*?(?=^\d+\. |\Z)', text, re.M | re.S).group()
        script = textwrap.dedent(re.search(r'```bash\n(.*?)\n\s*```', section, re.S).group(1))
        result = subprocess.run(['bash', '-e'], input=script, text=True, cwd=ROOT,
                                env=env, capture_output=True)
        if result.returncode:
            raise AssertionError(f'Documented step {number} failed:\n{result.stdout}\n{result.stderr}')
    if experiment_dir.exists():
        raise AssertionError('Offline preparation precreated the logger directory')
    return config_dir


class OperatorDevelopmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.run_id = 'operator_fixture_' + uuid4().hex
        cls.config_dir = ROOT / 'RL/configs' / cls.run_id
        cls.addClassCleanup(shutil.rmtree, cls.config_dir, ignore_errors=True)
        cls.config_dir = execute_documented_preparation(cls.run_id)
        cls.config = strict_json(cls.config_dir / 'train.json')

    def test_documented_workflow_and_saved_effective_config_pass(self):
        config = copy.deepcopy(self.config)
        config['resolved_experiment_name'] = self.run_id
        identity = validate_physical_launch(config, project_root=ROOT)
        self.assertEqual(identity['execution_entry'], 'training')
        self.assertEqual(config['output_dir'], 'RL/experiments')
        self.assertEqual(config['experiment_name'], self.run_id)
        self.assertIs(config['append_timestamp'], False)
        receipt = strict_json(self.config_dir / 'operator_development_receipt.json')
        self.assertFalse(receipt['independently_verified'])
        self.assertNotIn('gates', receipt)
        self.assertNotIn('run_root', receipt)
        self.assertEqual(receipt['config_dir'], str(self.config_dir))
        self.assertEqual(receipt['experiment_dir'], str(ROOT / 'RL/experiments' / self.run_id))
        self.assertEqual(strict_json(self.config_dir / 'bc_holdout.json')['samples'], 90003)
        self.assertFalse((ROOT / 'RL/experiments' / self.run_id).exists())
        self.assertNotIn('-m RL.prepare_physical_shadow_development', NOTES.read_text())

    def test_round4_route_dispatches_to_separate_frozen_inference_validator(self):
        import tempfile
        from RL.utils import physical_load_shadow_launch as launch

        with tempfile.TemporaryDirectory(prefix='rl_round4_route_dispatch_') as tmp:
            evidence_path = Path(tmp) / 'receipt.json'
            evidence_path.write_text('{}', encoding='utf-8')
            checkpoint_path = Path(tmp) / 'queue_round_4.pt'
            checkpoint_path.write_bytes(b'fixture-checkpoint')
            config = copy.deepcopy(self.config)
            config['physical_shadow_launch'] = {
                'authorized': True,
                'route': FROZEN_R4_ROUTE,
                'evidence_path': str(evidence_path),
                'evidence_sha256': file_sha256(evidence_path),
            }
            identity = {'actor_checkpoint_sha256': 'a' * 64, 'execution_entry': 'inference'}
            with patch.object(launch, 'physical_runtime_identity', return_value=identity), \
                    patch.object(launch, 'validate_round4_frozen_actor_development', return_value=identity) as route:
                result = validate_physical_launch(
                    config, project_root=ROOT, checkpoint_path=checkpoint_path, baseline_only=False,
                )
        self.assertEqual(result, identity)
        route.assert_called_once()

    def test_round4_frozen_route_rejects_baseline_before_checkpoint_load(self):
        with self.assertRaisesRegex(ValueError, 'persisted policy checkpoint'):
            validate_round4_frozen_actor_development(
                {}, {}, {}, ROOT, checkpoint_path=None, baseline_only=True,
            )

    def test_round4_frozen_route_rejects_online_optimizer_before_checkpoint_load(self):
        config = {
            'physical_shadow_run_role': {
                'stage': 'round4_frozen_actor_event_window_validation',
                'source_run_id': 'source', 'source_round': 4, 'actor_frozen': True,
                'optimizer_updates': False, 'independent_audit': False, 'policy_acceptance': False,
                'soft_enabled': False, 'hard_enabled': False,
            },
            'policy_acceptance': False,
            'train': {'online_rl_enabled': True},
            'queue': {'case_identity_validation': {'wind_mismatch_policy': 'warn_development_only'}},
            'physical_shadow_execution_scope': {
                'entry': 'inference', 'episodes': 4, 'max_steps': None, 'baseline_only': False,
            },
        }
        with self.assertRaisesRegex(ValueError, 'online training disabled'):
            validate_round4_frozen_actor_development(
                config, {'source_run': {'run_id': 'source'}}, {}, ROOT,
                checkpoint_path=Path('/tmp/queue_round_4.pt'), baseline_only=False,
            )

    def test_actual_logger_keeps_old_layout(self):
        # Logger itself is not changed: no extra output/runs directory.
        import tempfile
        from RL.train_rl import build_experiment_name
        with tempfile.TemporaryDirectory() as tmp:
            config = copy.deepcopy(self.config)
            config['output_dir'] = tmp
            with patch.dict(os.environ, {'RL_EXPERIMENT_SUFFIX': ''}):
                output = Path(config['output_dir']) / build_experiment_name(config)
            logger = ExperimentLogger(output, config.get('logging'))
            logger.save_config(config)
            self.assertEqual(logger.output_dir, Path(tmp) / self.run_id)
            self.assertTrue((logger.output_dir / 'config.json').is_file())
            self.assertFalse((logger.output_dir / 'output').exists())
            logger.flush()

    def test_relative_and_absolute_output_paths_are_equivalent(self):
        config = copy.deepcopy(self.config)
        config['output_dir'] = str(ROOT / 'RL/experiments')
        validate_physical_launch(config, project_root=ROOT)

    def test_runtime_mutations_rejected(self):
        for section, key, value in (
                ('load_predictor', 'mode', 'hard'), ('load_predictor', 'mode', 'soft'),
                ('load_predictor', 'backend', 'nn_candidate_v7'),
                ('agent', 'gamma', 0.1), ('train', 'online_rl_enabled', False),
                ('queue', 'total_condition_rounds', 1), ('bc', 'export_checkpoint_path', 'bad.pt'),
                ('physical_shadow_run_role', 'policy_acceptance', True),
                ('physical_shadow_run_role', 'independent_audit', True)):
            config = copy.deepcopy(self.config)
            config[section][key] = value
            with self.subTest(section=section, key=key), self.assertRaises(ValueError):
                validate_physical_launch(config, project_root=ROOT)

    def test_only_wind_policy_exception_is_allowed_not_tolerance_or_time(self):
        from RL.utils.physical_load_shadow_launch import validate_operator_scope
        self.assertEqual(self.config['queue']['case_identity_validation']['wind_mismatch_policy'],
                         'warn_development_only')
        receipt = strict_json(self.config_dir / 'operator_development_receipt.json')
        self.assertIs(receipt['operator_attestation']['project_wind_seed_personally_confirmed_matches_pid'], True)
        for key, value in (('wind_mismatch_policy', 'ignore'), ('wind_absolute_tolerance', 10),
                           ('fixed_simulation_time_offset_s', 0.02), ('enabled', False)):
            config = copy.deepcopy(self.config)
            config['queue']['case_identity_validation'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_operator_scope(config, ROOT)

    def test_python_startup_rejects_existing_run_before_udp(self):
        import tempfile
        from RL.train_rl import require_fresh_operator_experiment
        with tempfile.TemporaryDirectory() as tmp:
            config = dict(output_dir=tmp, experiment_name='failed')
            directory = require_fresh_operator_experiment(config)
            self.assertFalse(directory.exists())
            directory.mkdir()
            marker = directory / 'preserve.txt'
            marker.write_text('unchanged')
            with self.assertRaisesRegex(FileExistsError, 'already exists.*Old experiment preserved'):
                require_fresh_operator_experiment(config)
            self.assertEqual(marker.read_text(), 'unchanged')

    def test_documented_startup_failures_are_descriptive_and_never_run_python(self):
        text = NOTES.read_text()
        section = re.search(r'^10\. .*?(?=^11\.)', text, re.M | re.S).group()
        script = re.search(r'```bash\n(.*?)\n```', section, re.S).group(1)
        base_env = dict(os.environ)
        base_env.pop('RUN_CONFIG', None)
        base_env.pop('EXP_DIR', None)
        cases = (({}, '未设置'),
                 ({'RUN_CONFIG': '/nonexistent/config', 'EXP_DIR': '/nonexistent/experiment'}, '配置不存在'),
                 ({'RUN_CONFIG': str(self.config_dir / 'train.json'), 'EXP_DIR': str(self.config_dir)}, '实验目录已存在'))
        for values, expected in cases:
            result = subprocess.run(['bash'], input=script, text=True, cwd=ROOT,
                env={**base_env, **values}, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(expected, result.stderr)
            self.assertNotIn('启动训练', result.stdout)

    def test_retry_id_generated_without_overwriting_old_config_or_experiment(self):
        import tempfile
        text = NOTES.read_text()
        section = re.search(r'^1\. .*?(?=^2\.)', text, re.M | re.S).group()
        script = re.search(r'export PREVIOUS_CONFIG_DIR=.*?echo "\$RUN_ID"', section, re.S).group()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root / 'RL/configs/failed'
            old.mkdir(parents=True)
            (root / 'RL/experiments/failed').mkdir(parents=True)
            marker = old / 'preserve.txt'
            marker.write_text('unchanged')
            result = subprocess.run(['bash'], input=script, text=True, cwd=root,
                env={**os.environ, 'RUN_BASE_ID': 'failed', 'CONFIG_DIR': str(old)}, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            run_id = result.stdout.strip()
            self.assertTrue(run_id.startswith('failed_retry_'))
            self.assertTrue((root / 'RL/configs' / run_id).is_dir())
            self.assertFalse((root / 'RL/experiments' / run_id).exists())
            self.assertEqual(marker.read_text(), 'unchanged')

    def test_documented_retry_reuses_hash_verified_dependencies_without_loading_datasets(self):
        import tempfile
        text = NOTES.read_text()
        section = re.search(r'^2\. .*?(?=^3\.)', text, re.M | re.S).group()
        script = re.search(r'```bash\n(.*?)\n```', section, re.S).group(1)
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(['bash', '-e'], input=script, text=True, cwd=ROOT,
                env={**os.environ, 'CONFIG_DIR': tmp, 'PREVIOUS_CONFIG_DIR': str(self.config_dir)},
                capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('Unchanged dependencies/cache verified by actual SHA', result.stdout)
            self.assertNotIn('[BC-Data] loading', result.stdout)
            self.assertEqual(strict_json(Path(tmp) / 'dependencies.json'),
                             strict_json(self.config_dir / 'dependencies.json'))

    def test_output_layout_and_run_id_mutations_rejected(self):
        for key, value in (
                ('output_dir', str(ROOT / 'RL/experiments/runs')),
                ('output_dir', str(ROOT / 'RL/experiments' / self.run_id)),
                ('output_dir', 'RL/configs'), ('experiment_name', 'output'),
                ('experiment_name', '../escape'), ('experiment_name', 'runs/escape'),
                ('experiment_name', ''), ('experiment_name', 'a' * 81),
                ('append_timestamp', True)):
            config = copy.deepcopy(self.config)
            config[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_physical_launch(config, project_root=ROOT)

    def test_mutated_receipt_identity_layout_attestation_and_old_schema_rejected(self):
        path = self.config_dir / 'operator_development_receipt.json'
        original_bytes = path.read_bytes()
        original = strict_json(path)
        for field, value in (
                ('runtime_identity', {}), ('independently_verified', True),
                ('online_latency_status', 'PASS'), ('policy_acceptance', True),
                ('resource_inventory', {}), ('operator_attestation', {}),
                ('operator_attestation', {**original['operator_attestation'], 'dll_personally_confirmed': False}),
                ('operator_attestation', {**original['operator_attestation'], 'protocol_personally_confirmed': False}),
                ('operator_attestation', {**original['operator_attestation'], 'project_wind_seed_personally_confirmed_matches_pid': False}),
                ('schema_version', 'physical_shadow_operator_development_receipt_v1'),
                ('run_id', 'wrong'), ('config_dir', str(ROOT / 'RL/experiments' / self.run_id)),
                ('run_config', str(ROOT / 'RL/experiments' / self.run_id / 'train.json')),
                ('experiment_dir', str(ROOT / 'RL/experiments' / self.run_id / 'output')),
                ('run_root', str(self.config_dir))):
            receipt = copy.deepcopy(original)
            receipt[field] = value
            try:
                path.write_text(json.dumps(receipt))
                config = copy.deepcopy(self.config)
                config['physical_shadow_launch']['evidence_sha256'] = file_sha256(path)
                with self.subTest(field=field), self.assertRaises(ValueError):
                    validate_physical_launch(config, project_root=ROOT)
            finally:
                path.write_bytes(original_bytes)

    def test_receipt_and_bundle_queue_cannot_live_outside_config_batch(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            for section, field, filename in (
                    ('physical_shadow_launch', 'evidence_path', 'operator_development_receipt.json'),
                    ('queue', 'manifest_path', 'queue_manifest.json'),
                    ('load_predictor', 'bundle_path', 'physical_load_shadow_bundle.json')):
                path = Path(tmp) / filename
                shutil.copyfile(self.config_dir / filename, path)
                config = copy.deepcopy(self.config)
                config[section][field] = str(path)
                if section != 'physical_shadow_launch':
                    # Rebind runtime identity to ensure placement fails even after rehash.
                    from RL.utils.physical_load_shadow_launch import physical_runtime_identity
                    receipt_path = self.config_dir / 'operator_development_receipt.json'
                    original = receipt_path.read_bytes()
                    receipt = strict_json(receipt_path)
                    receipt['runtime_identity'] = physical_runtime_identity(config, ROOT)
                    receipt_path.write_text(json.dumps(receipt))
                    config['physical_shadow_launch']['evidence_sha256'] = file_sha256(receipt_path)
                try:
                    with self.subTest(section=section), self.assertRaises(ValueError):
                        validate_physical_launch(config, project_root=ROOT)
                finally:
                    if section != 'physical_shadow_launch':
                        receipt_path.write_bytes(original)

    def test_inference_and_baseline_cannot_use_training_receipt(self):
        for kwargs in ({'checkpoint_path': ROOT / self.config['bc']['resume_checkpoint']},
                       {'baseline_only': True}):
            with self.assertRaisesRegex(ValueError, 'training-only'):
                validate_physical_launch(self.config, project_root=ROOT, **kwargs)

    def test_disk_memory_and_device_availability_fail_closed(self):
        with patch('RL.prepare_physical_shadow_development.shutil.disk_usage',
                   return_value=type('Disk', (), {'free': 0})()):
            with self.assertRaisesRegex(ValueError, 'Insufficient'):
                check_resources(self.config, ROOT / 'RL/experiments' / self.run_id)
        changed = copy.deepcopy(self.config)
        changed['device'] = 'cuda:0'
        with patch('torch.cuda.is_available', return_value=False):
            with self.assertRaisesRegex(ValueError, 'unavailable'):
                check_resources(changed, ROOT / 'RL/experiments' / self.run_id)

    def test_frozen_default_still_blocked_and_no_overwrite(self):
        with self.assertRaisesRegex(ValueError, 'BLOCKED'):
            validate_physical_launch(strict_json(ROOT / OPERATOR_TEMPLATE), project_root=ROOT)
        with self.assertRaises(FileExistsError):
            execute_documented_preparation(self.run_id)

    def test_environment_cannot_change_actual_logger_identity(self):
        for name in ('RL_EXPERIMENT_SUFFIX', 'RL_DATA_ROOT', 'RL_DYNAMICS_CHECKPOINT'):
            with self.subTest(name=name), patch.dict(os.environ, {name: 'unauthorized'}):
                with self.assertRaises(ValueError):
                    validate_physical_launch(self.config, project_root=ROOT)


if __name__ == '__main__':
    unittest.main()