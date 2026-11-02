from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from RL.infer.plot_inference_vs_pid import (
    _formal_acceptance_invalid,
    build_multi_scenario_acceptance_report,
    evaluate_streaming_scenario,
)


JOB_SCHEMA_VERSION = 'multi_scenario_evaluation_job_v1'


def _resolve_input(path_value: Any, *, job_dir: Path) -> Path:
    if not isinstance(path_value, str) or not path_value.strip():
        raise ValueError('Evaluation input paths must be non-empty strings')
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve()
    local = (job_dir / path).resolve()
    if local.exists():
        return local
    return (PROJECT_ROOT / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_job(job_path: Path, *, output_path: Path | None = None) -> dict[str, Any]:
    job_path = job_path.expanduser().resolve()
    try:
        job = json.loads(job_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f'Unable to read multi-scenario evaluation job: {job_path}') from exc
    if not isinstance(job, dict) or job.get('schema_version') != JOB_SCHEMA_VERSION:
        raise ValueError(f'Evaluation job schema must be {JOB_SCHEMA_VERSION!r}')
    expected_ids = job.get('expected_scenario_ids')
    scenario_inputs = job.get('scenarios')
    if not isinstance(expected_ids, list) or not expected_ids:
        raise ValueError('Evaluation job requires non-empty expected_scenario_ids')
    if not isinstance(scenario_inputs, list):
        raise ValueError('Evaluation job scenarios must be a list')
    comparison_tolerance_nm = float(job.get('comparison_tolerance_nm', 0.0))
    if comparison_tolerance_nm < 0.0:
        raise ValueError('comparison_tolerance_nm must be non-negative')

    scenarios_for_aggregate: list[dict[str, Any]] = []
    scenario_evidence: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for item in scenario_inputs:
        if not isinstance(item, dict):
            raise ValueError('Each scenario input must be an object')
        scenario_id = str(item.get('scenario_id', '')).strip()
        if not scenario_id or scenario_id in seen_ids:
            raise ValueError(f'Duplicate or empty scenario_id: {scenario_id!r}')
        seen_ids.add(scenario_id)
        try:
            if item.get('condition_id') is None:
                raise ValueError('Each expected state20 scenario must declare its numeric condition_id')
            expected_condition_id = int(item['condition_id'])
            if scenario_id != str(expected_condition_id):
                raise ValueError('scenario_id must equal the canonical condition_id string')
            inference_csv = _resolve_input(item.get('inference_csv'), job_dir=job_path.parent)
            pid_csv = _resolve_input(item.get('pid_csv'), job_dir=job_path.parent)
            identity_path = _resolve_input(item.get('identity_summary_json'), job_dir=job_path.parent)
            run_summary_path = _resolve_input(item.get('run_summary_json'), job_dir=job_path.parent)
            identity = json.loads(identity_path.read_text(encoding='utf-8'))
            run_summary = json.loads(run_summary_path.read_text(encoding='utf-8'))
            if not isinstance(identity, dict) or not isinstance(run_summary, dict):
                raise ValueError('identity and run summary artifacts must contain JSON objects')
            pid_csv_sha256 = _sha256(pid_csv)
            matching_run_identity = any(
                isinstance(record, dict)
                and record.get('identity_status') == 'strict_case_matched'
                and str(record.get('summary_path', '')) == str(identity_path)
                and int(record.get('condition_id', -1)) == int(identity.get('condition_id', -2))
                and int(record.get('condition_instance_id', -1)) == int(identity.get('condition_instance_id', -2))
                and str(record.get('pid_raw_sha256', '')).lower()
                == str(identity.get('pid_raw_sha256', '')).lower()
                and str(record.get('expected_trace_signature', '')).lower()
                == str(identity.get('expected_trace_signature', '')).lower()
                and str(record.get('observed_trace_signature', '')).lower()
                == str(identity.get('observed_trace_signature', '')).lower()
                and int(record.get('observed_trace_sample_count', -1))
                == int(identity.get('observed_trace_sample_count', -2))
                and float(record.get('project_case_fingerprint_match', 0.0))
                == float(identity.get('project_case_fingerprint_match', 0.0))
                for record in run_summary.get('case_identity_summaries', ())
            )
            expected_trace_sample_count = int(identity.get('expected_trace_sample_count', -1))
            observed_trace_sample_count = int(identity.get('observed_trace_sample_count', -1))
            window_start_s = float(identity.get('time_window_start_s', math.nan))
            window_end_s = float(identity.get('time_window_end_s', math.nan))
            if (
                not math.isfinite(window_start_s)
                or not math.isfinite(window_end_s)
                or window_end_s <= window_start_s
            ):
                raise ValueError('Strict identity summary is missing its finite paired PID time window')
            pid_start_row = int(item.get('pid_start_row', 0))
            full_trace_proof = (
                float(identity.get('queue_case_identity_full_trace_complete', 0.0)) > 0.5
                and float(identity.get('queue_case_identity_trace_signature_match', 0.0)) > 0.5
            )
            project_case_proof = (
                float(identity.get('queue_case_identity_project_case_fingerprint_available', 0.0)) > 0.5
                and float(identity.get('queue_case_identity_project_case_fingerprint_match', 0.0)) > 0.5
            )
            strict_matched = (
                identity.get('identity_status') == 'strict_case_matched'
                and (full_trace_proof or project_case_proof)
                and str(identity.get('pid_raw_sha256', '')).lower() == pid_csv_sha256
                and matching_run_identity
                and int(identity.get('condition_id', -1)) == expected_condition_id
                and expected_trace_sample_count >= 2
                and observed_trace_sample_count == expected_trace_sample_count
            )
            if int(identity.get('condition_instance_id', -1)) < 0:
                strict_matched = False
            formal_report = evaluate_streaming_scenario(
                inference_csv,
                pid_csv,
                pid_start_row=pid_start_row,
                expected_pid_window_s=(window_start_s, window_end_s),
                expected_trace_sample_count=expected_trace_sample_count,
            )
            if formal_report.get('valid', False):
                expected_duration_s = window_end_s - window_start_s
                if (
                    int(formal_report.get('sample_count', -1)) != expected_trace_sample_count
                    or abs(float(formal_report.get('model_duration_s', math.nan)) - expected_duration_s) > 1.0e-6
                    or abs(float(formal_report.get('pid_duration_s', math.nan)) - expected_duration_s) > 1.0e-6
                ):
                    formal_report = _formal_acceptance_invalid(
                        'Inference/PID evaluator rows do not cover the complete frozen strict identity trace window.'
                    )
                    strict_matched = False
            run_completed = run_summary.get('status') == 'completed'
            edge = run_summary.get('edge_performance', {})
            edge_t1_pass = bool(isinstance(edge, dict) and edge.get('T1_pass', False))
            if not run_completed:
                formal_report = _formal_acceptance_invalid(
                    f'Inference run did not complete successfully: {run_summary.get("status", "unknown")}'
                )
            scenarios_for_aggregate.append({
                'scenario_id': scenario_id,
                'report': formal_report,
                'strict_case_matched': strict_matched,
            })
            scenario_evidence.append({
                'scenario_id': scenario_id,
                'condition_id': item.get('condition_id'),
                'condition_instance_id': identity.get('condition_instance_id'),
                'identity_status': identity.get('identity_status'),
                'strict_case_matched': strict_matched,
                'run_status': run_summary.get('status'),
                'edge_T1_pass': edge_t1_pass,
                'inference_csv': str(inference_csv),
                'inference_csv_sha256': _sha256(inference_csv),
                'pid_csv': str(pid_csv),
                'pid_csv_sha256': pid_csv_sha256,
                'identity_summary_json': str(identity_path),
                'identity_summary_sha256': _sha256(identity_path),
                'run_summary_json': str(run_summary_path),
                'run_summary_sha256': _sha256(run_summary_path),
                'formal_acceptance': formal_report,
            })
        except (OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
            invalid_report = _formal_acceptance_invalid(f'{type(exc).__name__}: {exc}')
            scenarios_for_aggregate.append({
                'scenario_id': scenario_id,
                'report': invalid_report,
                'strict_case_matched': False,
            })
            scenario_evidence.append({
                'scenario_id': scenario_id,
                'condition_id': item.get('condition_id'),
                'valid': False,
                'strict_case_matched': False,
                'edge_T1_pass': False,
                'reason': f'{type(exc).__name__}: {exc}',
            })

    aggregate = build_multi_scenario_acceptance_report(
        scenarios_for_aggregate,
        expected_scenario_ids=[str(value) for value in expected_ids],
        comparison_tolerance_nm=comparison_tolerance_nm,
    )
    expected_set = {str(value) for value in expected_ids}
    evidence_by_id = {str(item['scenario_id']): item for item in scenario_evidence}
    expected_evidence = [evidence_by_id.get(scenario_id) for scenario_id in map(str, expected_ids)]
    all_t1_pass = bool(expected_evidence) and all(
        item is not None and bool(item.get('edge_T1_pass', False)) for item in expected_evidence
    )
    aggregate.update({
        'schema_version': 'multi_scenario_deployment_acceptance_v1',
        'edge_T1_pass_all_scenarios': all_t1_pass,
        'deployment_release_pass': bool(aggregate['release_candidate_pass'] and all_t1_pass),
        'scenario_evidence': scenario_evidence,
        'unexpected_evidence_scenario_ids': sorted(set(evidence_by_id) - expected_set),
    })
    if output_path is not None:
        target = output_path.expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Evaluate frozen Bladed inference slots against paired PID traces with strict identity and dual gates.'
    )
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()
    report = evaluate_job(args.job, output_path=args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
