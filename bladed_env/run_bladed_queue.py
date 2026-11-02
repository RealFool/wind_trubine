from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from RL.utils.queue_manifest import QueueSlot, expand_queue_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run independent Bladed case reloads from a v7 queue manifest.')
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument(
        '--command-template',
        default='',
        help='Command with {case_path}, {condition_id}, {round_id}, and {condition_instance_id} placeholders.',
    )
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--condition-ids', nargs='*', type=int, default=None, help='Optional subset of condition IDs.')
    parser.add_argument('--max-slots', type=int, default=0, help='Optional maximum selected slots; 0 keeps all.')
    parser.add_argument('--log-path', type=Path, default=Path('bladed_queue_runs.jsonl'))
    return parser.parse_args()


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        with path.open('r', encoding='utf-8') as file:
            document = json.load(file)
    except FileNotFoundError as exc:
        raise SystemExit(f'Manifest does not exist: {path}') from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f'Unable to read manifest {path}: {exc}') from exc
    if not isinstance(document, dict):
        raise SystemExit('Manifest must contain a JSON object')
    return document


def _command(command_template: str, slot: QueueSlot) -> str:
    if not slot.case_path:
        raise ValueError('Manifest case_path_template is empty; provide a real Bladed case-path template before running')
    return command_template.format(
        case_path=slot.case_path,
        condition_id=slot.condition_id,
        round_id=slot.round_id,
        condition_instance_id=slot.condition_instance_id,
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')


def _slot_environment_overrides(slot: QueueSlot) -> dict[str, str]:
    return {
        'BLADED_RL_UDP_PROTOCOL': 'bladed_udp_19_si_my_queue_v1',
        'BLADED_RL_QUEUE_EPOCH': str(slot.queue_epoch),
        'BLADED_RL_CONDITION_INSTANCE_ID': str(slot.condition_instance_id),
        'BLADED_RL_CONDITION_ID': str(slot.condition_id),
        'BLADED_RL_QUEUE_PHASE': 'AUTO',
        'BLADED_RL_POST_LOAD_DISCARD_SIM_S': f'{slot.post_load_discard_sim_s:.6f}',
        'BLADED_RL_TRAIN_SIM_S': f'{slot.train_sim_s:.6f}',
    }


def _slot_environment(slot: QueueSlot) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(_slot_environment_overrides(slot))
    return environment


def _select_slots(
    slots: list[QueueSlot],
    condition_ids: list[int] | None,
    max_slots: int,
) -> list[QueueSlot]:
    if max_slots < 0:
        raise ValueError('--max-slots must be non-negative')
    selected = slots
    if condition_ids is not None:
        requested = set(condition_ids)
        if any(condition_id <= 0 for condition_id in requested):
            raise ValueError('--condition-ids must be positive integers')
        selected = [slot for slot in selected if slot.condition_id in requested]
    if max_slots > 0:
        selected = selected[:max_slots]
    if not selected:
        raise ValueError('No queue slots matched the requested selection')
    return selected


def _append_log(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as file:
        file.write(json.dumps(row, ensure_ascii=False) + '\n')


def main() -> None:
    args = parse_args()
    manifest = _load_manifest(args.manifest)
    command_template = args.command_template.strip() or str(manifest.get('bladed_command_template', '')).strip()
    if not command_template:
        raise SystemExit('Provide --command-template or manifest.bladed_command_template')
    try:
        slots = expand_queue_manifest(manifest)
        slots = _select_slots(slots, args.condition_ids, args.max_slots)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f'Invalid queue manifest: {exc}') from exc

    for slot in slots:
        try:
            command = _command(command_template, slot)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        log_row = {
            'round_id': slot.round_id,
            'queue_epoch': slot.queue_epoch,
            'condition_instance_id': slot.condition_instance_id,
            'condition_id': slot.condition_id,
            'case_path': slot.case_path,
            'command': command,
            'dry_run': bool(args.dry_run),
            'controller_environment': _slot_environment_overrides(slot),
            'process_started_utc': _utc_now(),
        }
        print(
            f'[Queue] round={slot.round_id} epoch={slot.queue_epoch} '
            f'instance={slot.condition_instance_id} condition={slot.condition_id} case={slot.case_path}',
            flush=True,
        )
        if args.dry_run:
            log_row.update({'status': 'dry_run', 'process_finished_utc': _utc_now(), 'process_wall_duration_s': 0.0})
            _append_log(args.log_path, log_row)
            continue
        process_started = time.perf_counter()
        completed = subprocess.run(command, shell=True, env=_slot_environment(slot), check=False)
        log_row.update(
            {
                'status': 'completed' if completed.returncode == 0 else 'failed',
                'returncode': int(completed.returncode),
                'process_finished_utc': _utc_now(),
                'process_wall_duration_s': time.perf_counter() - process_started,
            }
        )
        _append_log(args.log_path, log_row)
        if completed.returncode != 0:
            raise SystemExit(
                f'Bladed command failed for condition={slot.condition_id} '
                f'instance={slot.condition_instance_id} returncode={completed.returncode}'
            )


if __name__ == '__main__':
    main()