from __future__ import annotations

import argparse
import json
from pathlib import Path
import tracemalloc
from typing import Any

from RL.utils.logger import ExperimentLogger


def benchmark_compact_logger(
    output_dir: Path,
    *,
    rounds: int = 4,
    conditions_per_round: int = 23,
    steps_per_condition: int = 30_000,
    event_pre_frames: int = 25,
    event_post_frames: int = 25,
    event_capacity_per_condition: int = 16,
) -> dict[str, Any]:
    if min(rounds, conditions_per_round, steps_per_condition) <= 0:
        raise ValueError('rounds, conditions, and steps must be positive')
    slot_count = rounds * conditions_per_round
    logger_config = {
        'mode': 'compact_streaming',
        'quantile_sample_capacity': 2048,
        'event_ring_capacity': 512,
        'episode_ring_capacity': 128,
        'queue_condition_ring_capacity': slot_count + 1,
        'queue_condition_max_record_bytes': 512 * 1024,
        'queue_condition_summaries_max_total_bytes': 256 * 1024 * 1024,
        'queue_event_pre_frames': event_pre_frames,
        'queue_event_post_frames': event_post_frames,
        'queue_event_window_capacity': event_capacity_per_condition,
        'queue_event_window_max_record_bytes': 256 * 1024,
        'queue_event_windows_max_total_bytes': 512 * 1024 * 1024,
        'debug_long_form_enabled': False,
        'debug_max_rows': 0,
        'debug_max_bytes': 0,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    tracemalloc.start()
    logger = ExperimentLogger(output_dir, logger_config)
    condition_rows = 0
    total_step_rows = 0
    for round_index in range(1, rounds + 1):
        for condition_id in range(3, 3 + conditions_per_round):
            instance_id = (round_index - 1) * conditions_per_round + condition_id - 2
            for frame_index in range(steps_per_condition):
                step = total_step_rows + 1
                elapsed_s = frame_index * 0.02
                mode_code = 1.0 if frame_index % 257 < 220 else 8.0
                stage = 0.0 if mode_code == 1.0 else 4.0
                my_value = 45_000_000.0 + (frame_index % 10_000) * 100.0
                logger.log_metrics('queue_train_step', step, {
                    'queue_condition_round_index': float(round_index),
                    'queue_condition_id': float(condition_id),
                    'queue_condition_instance_id': float(instance_id),
                    'udp_sim_time': max(0.0, elapsed_s - 0.02),
                    'udp_next_sim_time': elapsed_s,
                    'udp_phase_elapsed_s': elapsed_s,
                    'wind_t': 12.0,
                    'power': 15_000_000.0,
                    'power_t_metric_raw': 15_000_000.0,
                    'power_metric_raw': 15_000_000.0,
                    'power_reward_metric': 15_000_000.0,
                    'power_state_real': 15_000_000.0,
                    'action_torque': 20_000_000.0,
                    'action_pitch': 0.05,
                    'action_torque_delta': 0.0,
                    'action_pitch_delta': 0.0,
                    'reward_acc_raw': 0.1,
                    'reward_shaft_raw': 20_000_000.0,
                    'reward_relative_power_raw': 0.0,
                    'reward_relative_energy_increment_j': 0.0,
                    'reward_persistent_power_deficit_ewma': 0.0,
                    'reward_pid_power_reference_raw': 15_000_000.0,
                    'control_pre_send_execution_mode': mode_code,
                    'control_optimization_state_code': 2.0,
                    'control_region_code': 1.0,
                    'control_low_speed_stage': stage,
                    'control_ood_detected': 0.0,
                    'control_ood_score': 0.1,
                    'control_baseline_action_torque': 20_000_000.0,
                    'control_baseline_action_pitch': 0.05,
                    'control_policy_delta_torque': 0.0,
                    'control_policy_delta_pitch': 0.0,
                    'control_residual_torque': 0.0,
                    'control_residual_pitch': 0.0,
                    'control_pre_safety_action_torque': 20_000_000.0,
                    'control_pre_safety_action_pitch': 0.05,
                    'control_projected_action_torque': 20_000_000.0,
                    'control_projected_action_pitch': 0.05,
                    'control_sent_action_torque': 20_000_000.0,
                    'control_sent_action_pitch': 0.05,
                    'control_action_projection_applied': float(frame_index % 101 == 0),
                    'control_action_rate_limited': float(frame_index % 73 == 0),
                    'actor_update_gate_open_before_critic': 1.0,
                    'actor_update_gate_open_after_critic': 1.0,
                    'actor_optimizer_health_gate_open': 1.0,
                    'actor_q_health_gate_open_after_update': 1.0,
                    'control_post_step_timeout': 0.0,
                    'control_post_step_invalid': 0.0,
                    'control_post_step_unsafe_stop': 0.0,
                    'control_post_step_condition_boundary': 0.0,
                    'reward_blade_root_1_raw': my_value,
                    'reward_blade_root_2_raw': my_value - 100_000.0,
                    'reward_blade_root_3_raw': my_value - 200_000.0,
                    'reward_blade_root_1_peak_reference_nm': 55_000_000.0,
                    'reward_blade_root_2_peak_reference_nm': 55_000_000.0,
                    'reward_blade_root_3_peak_reference_nm': 55_000_000.0,
                })
                total_step_rows += 1
            condition_metrics = {
                'condition_id': float(condition_id),
                'condition_instance_id': float(instance_id),
                'queue_condition_round_index': float(round_index),
                'queue_completed_round': float(condition_id == 3 + conditions_per_round - 1),
                'queue_completed_rounds': float(round_index),
                'queue_condition_raw_frame_count': float(steps_per_condition),
                'queue_condition_train_frame_count': float(steps_per_condition),
                'queue_condition_eligible_transition_count': float(steps_per_condition - 1),
                'queue_condition_stored_transition_count': float(steps_per_condition - 1),
                'queue_condition_rejected_transition_count': 1.0,
                'queue_condition_batch_count': float(steps_per_condition // 30),
                'queue_condition_batch_sample_count': float((steps_per_condition // 30) * 128),
                'queue_condition_batch_quota_relaxed_count': 0.0,
                'queue_condition_replay_count': 40_000.0,
                'queue_condition_execution_mode_actor_count': float(steps_per_condition * 220 / 257),
                'queue_condition_execution_mode_low_speed_recovery_hold_count': float(steps_per_condition * 37 / 257),
                'queue_condition_execution_mode_transition_count': float(steps_per_condition // 257),
                'queue_condition_action_projection_count': float(steps_per_condition // 101),
                'queue_condition_action_rate_limited_count': float(steps_per_condition // 73),
                'queue_condition_ood_score_count': float(steps_per_condition),
                'queue_condition_ood_score_sum': float(steps_per_condition * 0.1),
                'queue_condition_ood_score_min': 0.1,
                'queue_condition_ood_score_max': 0.1,
                'queue_condition_rl_pdt_diagnostic_energy_j': float(steps_per_condition * 300_000.0),
                'queue_condition_pid_pdt_diagnostic_energy_j': float(steps_per_condition * 300_000.0),
                'queue_condition_rl_trapezoid_energy_j': float(steps_per_condition * 300_000.0),
                'queue_condition_pid_trapezoid_energy_j': float(steps_per_condition * 300_000.0),
                'queue_condition_trapezoid_inputs_complete': 1.0,
                'dual_constraint_condition_complete': 1.0,
            }
            logger.log_metrics('queue_condition', total_step_rows, condition_metrics)
            condition_rows += 1
    logger.flush()
    _current_bytes, peak_python_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    artifacts = [
        output_dir / 'compact_metrics_summary.json',
        output_dir / 'compact_execution_events.jsonl',
        output_dir / 'compact_queue_condition_summaries.jsonl',
        output_dir / 'compact_queue_condition_event_windows.jsonl',
    ]
    for path in artifacts:
        if not path.is_file():
            raise RuntimeError(f'Compact logger benchmark did not write expected artifact: {path}')
    total_bytes = sum(path.stat().st_size for path in artifacts)
    return {
        'schema_version': 'compact_logger_resource_benchmark_v1',
        'mode': 'synthetic_no_udp',
        'rounds': rounds,
        'conditions_per_round': conditions_per_round,
        'slot_count': condition_rows,
        'steps_per_condition': steps_per_condition,
        'step_rows_emitted': total_step_rows,
        'event_pre_frames': event_pre_frames,
        'event_post_frames': event_post_frames,
        'event_capacity_per_condition': event_capacity_per_condition,
        'total_artifact_bytes': total_bytes,
        'total_artifact_bytes_budget': 1024 * 1024 * 1024,
        'python_heap_peak_bytes': peak_python_bytes,
        'python_heap_peak_budget_bytes': 256 * 1024 * 1024,
        'compact_metrics_summary': str(artifacts[0]),
        'condition_summary_jsonl': str(artifacts[2]),
        'event_windows_jsonl': str(artifacts[3]),
        'condition_rows_seen': condition_rows,
        'condition_rows_retained': logger._queue_condition_rows_seen,
        'condition_rows_dropped_in_summary_ring': logger._queue_condition_rows_dropped,
        'event_windows_written': logger._queue_event_windows_written,
        'event_windows_dropped': logger._queue_event_windows_dropped,
        'event_bytes_written': logger._queue_event_bytes_written,
        'pass': total_bytes <= 1024 * 1024 * 1024 and peak_python_bytes <= 256 * 1024 * 1024,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Synthetic full-size compact logger/event budget benchmark.')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=4)
    parser.add_argument('--conditions-per-round', type=int, default=23)
    parser.add_argument('--steps-per-condition', type=int, default=30_000)
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()
    report = benchmark_compact_logger(
        args.output_dir,
        rounds=args.rounds,
        conditions_per_round=args.conditions_per_round,
        steps_per_condition=args.steps_per_condition,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
