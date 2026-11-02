from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from RL.utils.release_manifest import build_release_manifest, write_release_manifest


def _load_job(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f'Unable to read release manifest job: {path}') from exc
    if not isinstance(value, dict):
        raise ValueError('Release manifest job must be a JSON object')
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Create a hash-bound state20 candidate/release manifest; all unmet gates remain blocked.'
    )
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    job_path = args.job.expanduser().resolve()
    job = _load_job(job_path)
    required = (
        'candidate_id',
        'checkpoint_path',
        'config_path',
        'paired_pid_reference_path',
        'predictor_artifact_path',
        'dll_evidence_path',
        'resource_evidence_path',
    )
    missing = [name for name in required if not job.get(name)]
    if missing:
        raise ValueError(f'Release manifest job is missing required fields: {missing}')
    manifest = build_release_manifest(
        candidate_id=str(job['candidate_id']),
        checkpoint_path=job['checkpoint_path'],
        config_path=job['config_path'],
        paired_pid_reference_path=job['paired_pid_reference_path'],
        predictor_artifact_path=job['predictor_artifact_path'],
        dll_evidence_path=job['dll_evidence_path'],
        resource_evidence_path=job['resource_evidence_path'],
        project_root=job.get('project_root', PROJECT_ROOT),
        acceptance_report_path=job.get('acceptance_report_path'),
        calibration_artifact_path=job.get('calibration_artifact_path'),
    )
    output_path = write_release_manifest(manifest, args.output)
    print(json.dumps({
        'manifest_path': str(output_path),
        'candidate_id': manifest['candidate_id'],
        'release_status': manifest['release_status'],
        'ready_for_v1': manifest['gates']['offline_ready_for_v1'],
        'release_ready': manifest['gates']['release_ready'],
        'failed_gates': [name for name, passed in manifest['gates'].items() if not passed],
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
