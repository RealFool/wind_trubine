from __future__ import annotations

from dataclasses import dataclass
from typing import Any


DEFAULT_DIAGNOSTIC_ORDER = (
    12, 11, 13, 10, 14, 9, 15, 8, 16, 7, 17, 6,
    18, 5, 19, 4, 20, 3, 21, 22, 23, 24, 25,
)
QUEUE_PROTOCOL_VERSIONS = {
    'bladed_udp_19_si_my_queue_v1',
    'bladed_udp_19_si_my_queue_v2',
}


@dataclass(frozen=True)
class QueueSlot:
    round_id: int
    queue_epoch: int
    condition_instance_id: int
    condition_id: int
    case_path: str
    post_load_discard_sim_s: float
    train_sim_s: float


def _positive_int(document: dict[str, Any], key: str) -> int:
    value = int(document[key])
    if value <= 0:
        raise ValueError(f'{key} must be positive')
    return value


def _condition_order(document: dict[str, Any]) -> tuple[int, ...]:
    raw_order = document.get('base_condition_order', list(DEFAULT_DIAGNOSTIC_ORDER))
    if not isinstance(raw_order, list):
        raise ValueError('base_condition_order must be a list')
    order = tuple(int(condition_id) for condition_id in raw_order)
    if len(order) != len(set(order)) or not order:
        raise ValueError('base_condition_order must contain unique condition IDs')
    return order


def expand_queue_manifest(document: dict[str, Any]) -> list[QueueSlot]:
    if document.get('protocol_version') not in QUEUE_PROTOCOL_VERSIONS:
        raise ValueError('manifest protocol_version must be a supported Bladed queue protocol')
    if document.get('execution_mode') != 'independent_case_reload':
        raise ValueError('manifest execution_mode must be independent_case_reload')
    rounds = _positive_int(document, 'total_condition_rounds')
    order = _condition_order(document)
    rotation_stride = int(document.get('rotation_stride', 7)) % len(order)
    queue_epoch_start = int(document.get('queue_epoch_start', 1))
    instance_start = int(document.get('condition_instance_start', 1))
    if queue_epoch_start < 0 or instance_start < 0:
        raise ValueError('queue epoch and condition instance starts must be non-negative')
    case_path_template = str(document.get('case_path_template', '')).strip()
    post_load_discard_sim_s = float(document.get('post_load_discard_sim_s', 0.0))
    train_sim_s = float(document.get('train_sim_s', 600.0))
    if post_load_discard_sim_s < 0.0 or train_sim_s <= 0.0:
        raise ValueError('post_load_discard_sim_s must be non-negative and train_sim_s must be positive')

    slots: list[QueueSlot] = []
    for round_index in range(rounds):
        offset = (round_index * rotation_stride) % len(order)
        rotated_order = order[offset:] + order[:offset]
        for index_in_round, condition_id in enumerate(rotated_order):
            sequence_index = len(slots)
            case_path = case_path_template.format(
                condition_id=condition_id,
                round_id=round_index + 1,
                index_in_round=index_in_round + 1,
            ) if case_path_template else ''
            slots.append(
                QueueSlot(
                    round_id=round_index + 1,
                    queue_epoch=queue_epoch_start + sequence_index,
                    condition_instance_id=instance_start + sequence_index,
                    condition_id=condition_id,
                    case_path=case_path,
                    post_load_discard_sim_s=post_load_discard_sim_s,
                    train_sim_s=train_sim_s,
                )
            )
    return slots