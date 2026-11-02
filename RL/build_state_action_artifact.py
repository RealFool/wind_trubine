from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from RL.utils.raw_dataset_contract import write_json
from RL.utils.state_action_artifact import build_state_action_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Build the causal state/action-envelope artifact from raw PID traces.',
    )
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--raw-infer-root', type=Path, default=None)
    parser.add_argument('--output', type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    document = build_state_action_artifact(
        args.data_root,
        raw_infer_root=args.raw_infer_root,
    )
    write_json(args.output, document)
    print(json.dumps({
        'output': str(args.output),
        'schema_version': document['schema_version'],
        'observation_width': document['observation_schema']['width'],
        'input_manifest_sha256': document['metadata']['input_manifest_sha256'],
        'state_coverage': document['state_coverage']['optimization_state_counts'],
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()