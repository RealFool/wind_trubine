from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_CONDITION_IDS = tuple(range(3, 26))
EXPECTED_START_TIME_S = 0.0
EXPECTED_END_TIME_S = 600.0
TIME_TOLERANCE_S = 1e-6


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def validate_raw_source_root(data_root: Path) -> tuple[Path, list[tuple[int, Path]]]:
    resolved = data_root.resolve()
    if resolved.name != 'raw':
        raise ValueError("--data-root must be named 'raw' and point directly to bladed-data-new/raw")
    if not resolved.is_dir():
        raise FileNotFoundError(f'Raw PID data root does not exist: {resolved}')
    expected = [(condition_id, resolved / f'data_windspeed{condition_id}.csv') for condition_id in EXPECTED_CONDITION_IDS]
    missing = [path.name for _, path in expected if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required raw PID files: {', '.join(missing)}")
    expected_names = {path.name for _, path in expected}
    unexpected = sorted(
        path.name
        for path in resolved.glob('data_windspeed*.csv')
        if path.is_file() and path.name not in expected_names
    )
    if unexpected:
        raise ValueError(f"Unexpected raw PID files outside wind3-wind25: {', '.join(unexpected)}")
    return resolved, expected


def manifest_sha256(file_audits: list[dict[str, Any]]) -> str:
    manifest = ''.join(
        f"{item['condition_id']}:{item['filename']}:{item['sha256']}\n"
        for item in sorted(file_audits, key=lambda value: int(value['condition_id']))
    )
    return hashlib.sha256(manifest.encode('utf-8')).hexdigest()
