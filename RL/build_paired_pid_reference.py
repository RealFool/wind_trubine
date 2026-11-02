from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from RL.utils.paired_pid_reference import build_paired_pid_reference
from RL.utils.raw_dataset_contract import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Build a fixed-window paired PID power/load reference artifact from raw traces.',
    )
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--time-window-start-s', type=float, default=0.0)
    parser.add_argument('--time-window-end-s', type=float, default=600.0)
    parser.add_argument('--time-signature-quantization-s', type=float, default=1.0e-6)
    parser.add_argument('--wind-signature-quantization-mps', type=float, default=1.0e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    document = build_paired_pid_reference(
        args.data_root,
        time_window_start_s=args.time_window_start_s,
        time_window_end_s=args.time_window_end_s,
        project_root=PROJECT_ROOT,
        time_signature_quantization_s=args.time_signature_quantization_s,
        wind_signature_quantization_mps=args.wind_signature_quantization_mps,
    )
    write_json(args.output, document)
    print(json.dumps({
        'output': str(args.output),
        'schema_version': document['schema_version'],
        'condition_count': document['metadata']['condition_count'],
        'time_window_s': [args.time_window_start_s, args.time_window_end_s],
        'input_manifest_sha256': document['metadata']['input_manifest_sha256'],
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()