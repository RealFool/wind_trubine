from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from RL.evaluate_multi_scenario_acceptance import evaluate_job


class TestEvaluateMultiScenarioAcceptance(unittest.TestCase):
    def _write_scenario(
        self,
        root: Path,
        *,
        identity_status: str = 'strict_case_matched',
        include_terminal_observation: bool = False,
    ) -> Path:
        inference = root / 'inference.csv'
        inference_fields = [
            'record_type',
            'inference_episode',
            'inference_global_control_step',
            'info_control_fallback_active',
            'info_control_ood_score',
            'sent_action_torque',
            'sent_action_pitch',
            'Time',
            'I_AverageWindSpeed',
            'Electrical power',
            'Blade root 1 My (Nm)',
            'Blade root 2 My (Nm)',
            'Blade root 3 My (Nm)',
        ]
        with inference.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=inference_fields)
            writer.writeheader()
            for index, timestamp in enumerate((1.0, 1.1, 1.2), start=1):
                writer.writerow({
                    'record_type': 'control_step',
                    'inference_episode': 1,
                    'inference_global_control_step': index,
                    'info_control_fallback_active': 0,
                    'info_control_ood_score': 0.0,
                    'sent_action_torque': 20_000_000.0,
                    'sent_action_pitch': 0.05,
                    'Time': timestamp,
                    'I_AverageWindSpeed': 12.0,
                    'Electrical power': 11_000_000.0,
                    'Blade root 1 My (Nm)': 90.0,
                    'Blade root 2 My (Nm)': 90.0,
                    'Blade root 3 My (Nm)': 90.0,
                })
            if include_terminal_observation:
                writer.writerow({
                    'record_type': 'terminal_observation',
                    'inference_episode': 1,
                    'inference_global_control_step': 3,
                    'info_control_fallback_active': '',
                    'info_control_ood_score': '',
                    'sent_action_torque': '',
                    'sent_action_pitch': '',
                    'Time': 1.3,
                    'I_AverageWindSpeed': 12.0,
                    'Electrical power': 11_000_000.0,
                    'Blade root 1 My (Nm)': 90.0,
                    'Blade root 2 My (Nm)': 90.0,
                    'Blade root 3 My (Nm)': 90.0,
                })
        pid = root / 'pid.csv'
        pid_fields = [
            'Time',
            'I_AverageWindSpeed',
            'Electrical power',
            'O_PcsTorqueDemand',
            'O_PitchPositionDemand1',
            'Blade root 1 My (Nm)',
            'Blade root 2 My (Nm)',
            'Blade root 3 My (Nm)',
        ]
        with pid.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=pid_fields)
            writer.writeheader()
            pid_timestamps = (0.0, 0.1, 0.2, 0.3) if include_terminal_observation else (0.0, 0.1, 0.2)
            for index, timestamp in enumerate(pid_timestamps):
                writer.writerow({
                    'Time': timestamp,
                    'I_AverageWindSpeed': 12.0,
                    'Electrical power': 10_000_000.0,
                    'O_PcsTorqueDemand': 20_000_000.0,
                    'O_PitchPositionDemand1': 0.05,
                    'Blade root 1 My (Nm)': 100.0,
                    'Blade root 2 My (Nm)': 100.0,
                    'Blade root 3 My (Nm)': 100.0,
                })
        pid_sha = hashlib.sha256(pid.read_bytes()).hexdigest()
        identity = root / 'identity.json'
        identity.write_text(json.dumps({
            'identity_status': identity_status,
            'condition_id': 3,
            'condition_instance_id': 101,
            'pid_raw_sha256': pid_sha,
            'expected_trace_signature': 'a' * 64,
            'observed_trace_signature': 'a' * 64,
            'expected_trace_sample_count': len(pid_timestamps),
            'observed_trace_sample_count': len(pid_timestamps),
            'time_window_start_s': 0.0,
            'time_window_end_s': pid_timestamps[-1],
            'queue_case_identity_full_trace_complete': 1.0,
            'queue_case_identity_trace_signature_match': 1.0,
            'project_case_fingerprint_match': 0.0,
        }), encoding='utf-8')
        run_summary = root / 'run_summary.json'
        run_summary.write_text(json.dumps({
            'status': 'completed',
            'edge_performance': {'T1_pass': True},
            'case_identity_summaries': [{
                'identity_status': identity_status,
                'condition_id': 3,
                'condition_instance_id': 101,
                'summary_path': str(identity.resolve()),
                'pid_raw_sha256': pid_sha,
                'expected_trace_signature': 'a' * 64,
                'observed_trace_signature': 'a' * 64,
                'observed_trace_sample_count': len(pid_timestamps),
            }],
        }), encoding='utf-8')
        job = root / 'job.json'
        job.write_text(json.dumps({
            'schema_version': 'multi_scenario_evaluation_job_v1',
            'expected_scenario_ids': ['3'],
            'scenarios': [{
                'scenario_id': '3',
                'condition_id': 3,
                'inference_csv': str(inference),
                'pid_csv': str(pid),
                'identity_summary_json': str(identity),
                'run_summary_json': str(run_summary),
            }],
        }), encoding='utf-8')
        return job

    def test_evaluator_accepts_only_strict_identity_and_checks_paired_pid_hash(self) -> None:
        with tempfile.TemporaryDirectory(prefix='multi_scenario_eval_') as temporary_directory:
            root = Path(temporary_directory)
            job = self._write_scenario(root)
            output_path = root / 'acceptance.json'
            report = evaluate_job(job, output_path=output_path)
            self.assertTrue(output_path.is_file())
            self.assertTrue(report['all_expected_scenarios_valid'])
            self.assertTrue(report['official_acceptance_pass'])
            self.assertTrue(report['strict_per_blade_pass'])
            self.assertTrue(report['strict_case_identity_pass'])
            self.assertTrue(report['edge_T1_pass_all_scenarios'])
            self.assertTrue(report['deployment_release_pass'])
            self.assertEqual(report['scenario_evidence'][0]['condition_instance_id'], 101)

    def test_evaluator_fails_closed_when_case_identity_is_not_strict(self) -> None:
        with tempfile.TemporaryDirectory(prefix='multi_scenario_eval_identity_') as temporary_directory:
            root = Path(temporary_directory)
            job = self._write_scenario(root, identity_status='reference_matched_only')
            report = evaluate_job(job)

        self.assertTrue(report['all_expected_scenarios_valid'])
        self.assertTrue(report['official_acceptance_pass'])
        self.assertTrue(report['strict_per_blade_pass'])
        self.assertFalse(report['strict_case_identity_pass'])
        self.assertFalse(report['deployment_release_pass'])

    def test_evaluator_accepts_verified_project_case_fingerprint_proof(self) -> None:
        with tempfile.TemporaryDirectory(prefix='multi_scenario_eval_fingerprint_') as temporary_directory:
            root = Path(temporary_directory)
            job = self._write_scenario(root)
            job_document = json.loads(job.read_text(encoding='utf-8'))
            identity_path = Path(job_document['scenarios'][0]['identity_summary_json'])
            identity = json.loads(identity_path.read_text(encoding='utf-8'))
            identity.update({
                'observed_trace_signature': 'b' * 64,
                'queue_case_identity_full_trace_complete': 0.0,
                'queue_case_identity_trace_signature_match': 0.0,
                'queue_case_identity_project_case_fingerprint_available': 1.0,
                'queue_case_identity_project_case_fingerprint_match': 1.0,
                'project_case_fingerprint_match': 1.0,
            })
            identity_path.write_text(json.dumps(identity), encoding='utf-8')
            summary_path = Path(job_document['scenarios'][0]['run_summary_json'])
            run_summary = json.loads(summary_path.read_text(encoding='utf-8'))
            run_summary['case_identity_summaries'][0].update({
                'observed_trace_signature': 'b' * 64,
                'project_case_fingerprint_match': 1.0,
            })
            summary_path.write_text(json.dumps(run_summary), encoding='utf-8')
            report = evaluate_job(job)

        self.assertTrue(report['all_expected_scenarios_valid'])
        self.assertTrue(report['strict_case_identity_pass'])
        self.assertTrue(report['deployment_release_pass'])

    def test_evaluator_rejects_scenario_condition_aliasing_and_incomplete_trace(self) -> None:
        with tempfile.TemporaryDirectory(prefix='multi_scenario_eval_alias_') as temporary_directory:
            root = Path(temporary_directory)
            job = self._write_scenario(root)
            job_document = json.loads(job.read_text(encoding='utf-8'))
            job_document['scenarios'][0]['scenario_id'] = '4'
            job.write_text(json.dumps(job_document), encoding='utf-8')
            aliased_report = evaluate_job(job)
            self.assertFalse(aliased_report['all_expected_scenarios_valid'])
            self.assertFalse(aliased_report['strict_case_identity_pass'])

        with tempfile.TemporaryDirectory(prefix='multi_scenario_eval_truncated_') as temporary_directory:
            root = Path(temporary_directory)
            job = self._write_scenario(root)
            job_document = json.loads(job.read_text(encoding='utf-8'))
            inference_path = Path(job_document['scenarios'][0]['inference_csv'])
            with inference_path.open('r', encoding='utf-8', newline='') as handle:
                rows = list(csv.DictReader(handle))
            fields = list(rows[0])
            with inference_path.open('w', encoding='utf-8', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows[:-1])
            truncated_report = evaluate_job(job)
            self.assertFalse(truncated_report['all_expected_scenarios_valid'])
            self.assertFalse(truncated_report['strict_case_identity_pass'])

    def test_evaluator_rejects_pid_rows_from_the_wrong_time_window(self) -> None:
        with tempfile.TemporaryDirectory(prefix='multi_scenario_eval_window_') as temporary_directory:
            root = Path(temporary_directory)
            job = self._write_scenario(root)
            pid_path = root / 'pid.csv'
            with pid_path.open('r', encoding='utf-8', newline='') as handle:
                reader = csv.DictReader(handle)
                fields = reader.fieldnames
                rows = list(reader)
            for row in rows:
                row['Time'] = str(float(row['Time']) + 10.0)
            with pid_path.open('w', encoding='utf-8', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            report = evaluate_job(job)

        self.assertFalse(report['all_expected_scenarios_valid'])
        self.assertFalse(report['strict_case_identity_pass'])
        self.assertIn('strict identity paired-reference time window', report['scenarios'][0]['reason'])

    def test_evaluator_includes_terminal_observation_for_full_pid_window(self) -> None:
        with tempfile.TemporaryDirectory(prefix='multi_scenario_eval_terminal_') as temporary_directory:
            root = Path(temporary_directory)
            job = self._write_scenario(root, include_terminal_observation=True)
            report = evaluate_job(job)

        self.assertTrue(report['all_expected_scenarios_valid'])
        self.assertTrue(report['strict_case_identity_pass'])
        self.assertEqual(report['scenario_evidence'][0]['formal_acceptance']['sample_count'], 4)


if __name__ == '__main__':
    unittest.main()
