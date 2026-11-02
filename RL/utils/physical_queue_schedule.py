from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

from RL.utils.queue_manifest import expand_queue_manifest


FULL_23WIND_PROFILE = '23wind_round600_training'
FULL_23WIND_CONDITIONS = tuple(range(3, 26))
FULL_23WIND_BASE_ORDER = (
    12, 11, 13, 10, 14, 9, 15, 8, 16, 7, 17, 6,
    18, 5, 19, 4, 20, 3, 21, 22, 23, 24, 25,
)
FULL_23WIND_ROTATION_STRIDE = 7
FULL_23WIND_DURATION_S = 600.0
DEFAULT_FULL_23WIND_ROUNDS = 4
MAX_FULL_23WIND_ROUNDS = 8
PHYSICAL_SHADOW_MAX_TOTAL_BYTES = 18 * 1024**3
PHYSICAL_SHADOW_RECORD_MAX_BYTES = 8 * 1024
PHYSICAL_SHADOW_ORIGIN_STRIDE_FRAMES = 5


def validate_full_23wind_rounds(rounds: int) -> int:
    if isinstance(rounds, bool) or not isinstance(rounds, int):
        raise ValueError('23-wind rounds must be an integer')
    if not 1 <= rounds <= MAX_FULL_23WIND_ROUNDS:
        raise ValueError(
            f'23-wind rounds must be in [1, {MAX_FULL_23WIND_ROUNDS}] '
            'to remain within the frozen shadow storage budget'
        )
    return rounds


def build_full_23wind_manifest(
    source_manifest: dict[str, Any],
    rounds: int,
) -> dict[str, Any]:
    rounds = validate_full_23wind_rounds(rounds)
    manifest = copy.deepcopy(source_manifest)
    manifest.update(
        total_condition_rounds=rounds,
        base_condition_order=list(FULL_23WIND_BASE_ORDER),
        rotation_stride=FULL_23WIND_ROTATION_STRIDE,
        train_sim_s=FULL_23WIND_DURATION_S,
        expanded_slot_count=len(FULL_23WIND_CONDITIONS) * rounds,
        manual_batch_profile='full',
        manual_batch_rounds=rounds,
        state20_profile=FULL_23WIND_PROFILE,
        identity_validation_level='strict',
        notes='State20 TD3+BC with read-only physical FIR shadow; operator development only.',
    )
    slots = expand_queue_manifest(manifest)
    expected_order: list[int] = []
    base = list(FULL_23WIND_BASE_ORDER)
    for round_index in range(rounds):
        offset = (round_index * FULL_23WIND_ROTATION_STRIDE) % len(base)
        expected_order.extend(base[offset:] + base[:offset])
    if (
        len(slots) != len(FULL_23WIND_CONDITIONS) * rounds
        or [slot.condition_id for slot in slots] != expected_order
        or any(slot.train_sim_s != FULL_23WIND_DURATION_S for slot in slots)
    ):
        raise ValueError('Generated 23-wind manifest does not match the configured rounds/order')
    return manifest


def configure_full_23wind_rounds(
    config: dict[str, Any],
    manifest_path: Path,
    rounds: int,
    *,
    project_root: Path,
) -> dict[str, Any]:
    """Apply one rounds value to queue, training contract, and physical budgets."""
    rounds = validate_full_23wind_rounds(rounds)
    slots = len(FULL_23WIND_CONDITIONS) * rounds
    queue = config['queue']
    predictor = config['load_predictor']
    logging = config['logging']
    train = config['train']
    step_duration_s = float(config['env']['bladed_step_duration_ms']) / 1000.0
    if not math.isfinite(step_duration_s) or step_duration_s <= 0.0:
        raise ValueError('Bladed step duration must be finite and positive')
    steps_per_slot_float = FULL_23WIND_DURATION_S / step_duration_s
    if not math.isclose(steps_per_slot_float, round(steps_per_slot_float), rel_tol=0.0, abs_tol=1e-9):
        raise ValueError('600-second slot must close on an integer number of simulator steps')
    steps_per_slot = int(round(steps_per_slot_float))
    stride = int(predictor.get('origin_stride_frames', PHYSICAL_SHADOW_ORIGIN_STRIDE_FRAMES))
    if stride <= 0:
        raise ValueError('Physical shadow origin stride must be positive')
    expected_shadow_records = slots * 2 * (steps_per_slot // stride + 1)
    minimum_shadow_bytes = expected_shadow_records * PHYSICAL_SHADOW_RECORD_MAX_BYTES
    if minimum_shadow_bytes > PHYSICAL_SHADOW_MAX_TOTAL_BYTES:
        raise ValueError('Configured rounds exceed the frozen physical-shadow total-byte budget')

    update_every_steps = int(train.get('update_every_steps', 0))
    if update_every_steps <= 0:
        raise ValueError('train.update_every_steps must be positive')
    expected_updates = (rounds - 1) * len(FULL_23WIND_CONDITIONS) * steps_per_slot / update_every_steps
    if not math.isclose(expected_updates, round(expected_updates), rel_tol=0.0, abs_tol=1e-9):
        raise ValueError('Configured schedule must close on an integer number of Critic updates')

    relative_manifest = manifest_path.resolve().relative_to(project_root.resolve()).as_posix()
    queue.update(
        manifest_path=relative_manifest,
        execution_mode='independent_case_reload',
        condition_ids=list(FULL_23WIND_CONDITIONS),
        total_condition_rounds=rounds,
        train_sim_s=FULL_23WIND_DURATION_S,
    )
    config['state20_queue_contract'].update(
        profile=FULL_23WIND_PROFILE,
        condition_ids=list(FULL_23WIND_CONDITIONS),
        rounds=rounds,
        train_sim_s=FULL_23WIND_DURATION_S,
        expected_scheduled_updates=int(round(expected_updates)),
    )
    predictor.update(
        max_records=expected_shadow_records,
        max_record_bytes=PHYSICAL_SHADOW_RECORD_MAX_BYTES,
        max_total_bytes=PHYSICAL_SHADOW_MAX_TOTAL_BYTES,
        origin_stride_frames=stride,
    )
    required_event_bytes = (
        slots
        * int(logging.get('queue_event_window_capacity', 0))
        * int(logging.get('queue_event_window_max_record_bytes', 0))
    )
    required_summary_bytes = slots * int(logging.get('queue_condition_max_record_bytes', 0))
    if (
        int(logging.get('queue_condition_ring_capacity', 0)) < slots
        or int(logging.get('queue_event_windows_max_total_bytes', 0)) < required_event_bytes
        or int(logging.get('queue_condition_summaries_max_total_bytes', 0)) < required_summary_bytes
    ):
        raise ValueError('Configured rounds exceed the frozen compact-logger budget')
    return config