"""Resource availability utility only; the bundled preparation CLI is retired.

Execute the independent commands in doc/训练操作笔记.md. This module does not
generate configs/receipts, evaluate BC, open UDP, or train any model.
"""
from __future__ import annotations

from pathlib import Path
import shutil

import torch

GIB = 1024 ** 3


def check_resources(config: dict, experiment_dir: Path) -> dict:
    """Availability floors only; no latency/RSS qualification is claimed."""
    parent = experiment_dir.resolve()
    while not parent.exists():
        parent = parent.parent
    memory = {line.split(':')[0]: int(line.split()[1]) * 1024
              for line in Path('/proc/meminfo').read_text().splitlines() if ':' in line}
    available = memory['MemAvailable']
    cgroup = Path('/sys/fs/cgroup')
    if (cgroup / 'memory.max').is_file() and (cgroup / 'memory.current').is_file():
        limit = (cgroup / 'memory.max').read_text().strip()
        if limit != 'max':
            available = min(available, max(0, int(limit) - int((cgroup / 'memory.current').read_text())))
    disk = shutil.disk_usage(parent).free
    logging = config.get('logging', {})
    predictor = config['load_predictor']
    disk_floor = (
        int(predictor['max_total_bytes'])
        + int(logging.get('queue_event_windows_max_total_bytes', 0))
        + int(logging.get('queue_condition_summaries_max_total_bytes', 0))
        + int(predictor.get('shadow_audit_max_total_bytes', 0))
        + 3 * GIB
    )
    device = torch.device(config['device'])
    gpu = None
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError('Prepared CUDA device is unavailable; prepare a new run on CPU')
        free, total = torch.cuda.mem_get_info(device)
        gpu = {'available_bytes': free, 'total_bytes': total, 'minimum_available_bytes': 2 * GIB}
        if free < 2 * GIB:
            raise ValueError('Insufficient available GPU memory (development floor 2 GiB)')
    elif device.type != 'cpu':
        raise ValueError('Operator development supports CPU or CUDA only')
    if available < 4 * GIB or disk < disk_floor:
        raise ValueError('Insufficient available memory/disk for development preparation')
    return {'memory_available_bytes': available, 'memory_minimum_bytes': 4 * GIB,
            'disk_available_bytes': disk, 'disk_minimum_bytes': disk_floor,
            'disk_budget_components': {
                'physical_shadow_max_total_bytes': int(predictor['max_total_bytes']),
                'event_windows_max_total_bytes': int(logging.get('queue_event_windows_max_total_bytes', 0)),
                'queue_condition_summaries_max_total_bytes': int(
                    logging.get('queue_condition_summaries_max_total_bytes', 0)),
                'load_predictor_shadow_audit_max_total_bytes': int(
                    predictor.get('shadow_audit_max_total_bytes', 0)),
                'safety_margin_bytes': 3 * GIB,
            },
            'gpu': gpu,
            'qualification': 'availability_floor_only_not_peak_RSS_or_online_latency_validation'}


if __name__ == '__main__':
    raise SystemExit('Bundled preparation is retired. Execute the numbered independent commands in doc/训练操作笔记.md.')