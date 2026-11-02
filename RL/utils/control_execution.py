from __future__ import annotations

from enum import Enum
from typing import Any, Mapping


class PreSendExecutionMode(str, Enum):
    ACTOR = 'actor'
    EXPLICIT_BASELINE_STAGE = 'explicit_baseline_stage'
    ACTOR_EXECUTION_ROLLBACK_BASELINE = 'actor_execution_rollback_baseline'
    OOD_BASELINE = 'ood_baseline'
    LOW_SPEED_WARNING_HOLD = 'low_speed_warning_hold'
    LOW_SPEED_PROTECTION_HOLD = 'low_speed_protection_hold'
    LOW_SPEED_EMERGENCY_HOLD = 'low_speed_emergency_hold'
    LOW_SPEED_RECOVERY_HOLD = 'low_speed_recovery_hold'
    STARTUP_STOPPED_BASELINE = 'startup_stopped_baseline'
    NEGATIVE_POWER_ABNORMAL_BASELINE = 'negative_power_abnormal_baseline'


PRE_SEND_EXECUTION_MODE_CODES = {
    mode: index
    for index, mode in enumerate(PreSendExecutionMode, start=1)
}


TD3_CRITIC_REPLAY_MODES = frozenset({
    PreSendExecutionMode.ACTOR,
    PreSendExecutionMode.EXPLICIT_BASELINE_STAGE,
    PreSendExecutionMode.ACTOR_EXECUTION_ROLLBACK_BASELINE,
    PreSendExecutionMode.LOW_SPEED_RECOVERY_HOLD,
})
ACTOR_REPLAY_SOURCE = 'actor'
EXPLICIT_BASELINE_REPLAY_SOURCE = 'explicit_baseline'
ACTOR_EXECUTION_ROLLBACK_REPLAY_SOURCE = 'actor_execution_rollback_baseline'
LOW_SPEED_RECOVERY_REPLAY_SOURCE = 'low_speed_recovery_hold'
EXECUTION_MODE_REPLAY_SOURCE = {
    PreSendExecutionMode.ACTOR: ACTOR_REPLAY_SOURCE,
    PreSendExecutionMode.EXPLICIT_BASELINE_STAGE: EXPLICIT_BASELINE_REPLAY_SOURCE,
    PreSendExecutionMode.ACTOR_EXECUTION_ROLLBACK_BASELINE: ACTOR_EXECUTION_ROLLBACK_REPLAY_SOURCE,
    PreSendExecutionMode.LOW_SPEED_RECOVERY_HOLD: LOW_SPEED_RECOVERY_REPLAY_SOURCE,
}
EXECUTION_REPLAY_CONTRACT_SCHEMA_VERSION = 'execution_mode_replay_contract_v1'


def pre_send_execution_mode(value: PreSendExecutionMode | str) -> PreSendExecutionMode:
    if isinstance(value, PreSendExecutionMode):
        return value
    try:
        return PreSendExecutionMode(str(value))
    except ValueError as exc:
        raise ValueError(f'Unknown pre-send execution mode: {value!r}') from exc


def pre_send_execution_mode_code(value: PreSendExecutionMode | str) -> int:
    return PRE_SEND_EXECUTION_MODE_CODES[pre_send_execution_mode(value)]


def pre_send_execution_mode_from_code(value: float | int) -> PreSendExecutionMode:
    try:
        code = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'Invalid pre-send execution mode code: {value!r}') from exc
    if float(value) != float(code):
        raise ValueError(f'Invalid pre-send execution mode code: {value!r}')
    for mode, mode_code in PRE_SEND_EXECUTION_MODE_CODES.items():
        if mode_code == code:
            return mode
    raise ValueError(f'Unknown pre-send execution mode code: {value!r}')


def td3_critic_replay_allowed(value: PreSendExecutionMode | str) -> bool:
    return pre_send_execution_mode(value) in TD3_CRITIC_REPLAY_MODES


def replay_source_for_execution_mode(value: PreSendExecutionMode | str) -> str | None:
    return EXECUTION_MODE_REPLAY_SOURCE.get(pre_send_execution_mode(value))


def post_step_outcome_metrics(
    step_info: dict[str, float],
    *,
    unsafe_stop: bool,
) -> dict[str, float]:
    timeout = bool(
        step_info.get('env_timeout', 0.0) > 0.0
        or step_info.get('udp_inter_frame_timeout', 0.0) > 0.0
        or step_info.get('udp_condition_load_timeout', 0.0) > 0.0
    )
    invalid = bool(
        step_info.get('env_invalid_prediction', 0.0) > 0.0
        or step_info.get('env_invalid_reward', 0.0) > 0.0
    )
    return {
        'control_post_step_timeout': float(timeout),
        'control_post_step_invalid': float(invalid),
        'control_post_step_unsafe_stop': float(unsafe_stop),
        'control_post_step_condition_boundary': float(
            step_info.get('condition_boundary', 0.0) > 0.0
        ),
    }


def execution_mode_replay_contract(
    train_config: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    document = train_config.get('execution_replay_contract')
    if document is None:
        return None
    if not isinstance(document, Mapping):
        raise ValueError('train.execution_replay_contract must be an object')
    if document.get('schema_version') != EXECUTION_REPLAY_CONTRACT_SCHEMA_VERSION:
        raise ValueError('train.execution_replay_contract schema version is incompatible')
    if document.get('use_execution_mode_replay') is not True:
        raise ValueError('train.execution_replay_contract must enable execution-mode replay')
    expected_modes = sorted(mode.value for mode in TD3_CRITIC_REPLAY_MODES)
    configured_modes = document.get('td3_critic_replay_modes')
    if sorted(configured_modes or ()) != expected_modes:
        raise ValueError('train.execution_replay_contract td3 critic replay modes are incompatible')
    if document.get('reject_post_step_outcomes') is not True:
        raise ValueError('train.execution_replay_contract must reject post-step outcomes')
    return document