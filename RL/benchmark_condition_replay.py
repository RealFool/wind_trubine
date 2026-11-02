from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import time
from typing import Any

import numpy as np

from RL.replay import ConditionStratifiedReplayBuffer, REPLAY_SOURCE_NAMES
from RL.utils.control_execution import ACTOR_REPLAY_SOURCE, LOW_SPEED_RECOVERY_REPLAY_SOURCE


def _rss_bytes() -> int:
    try:
        with Path('/proc/self/status').open('r', encoding='ascii') as handle:
            for line in handle:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    maximum = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return maximum * (1024 if maximum < 10**9 else 1)


def _buffer_array_bytes(buffer: ConditionStratifiedReplayBuffer) -> int:
    return sum(
        array.nbytes
        for bucket in buffer._buckets.values()
        for array in (
            bucket.obs,
            bucket.next_obs,
            bucket.actions,
            bucket.rewards,
            bucket.dones,
            bucket.condition_instance_ids,
            bucket.phase_elapsed_s,
            bucket.sources,
            bucket.source_indices,
            bucket.source_counts,
            bucket.source_positions,
        )
    )


def benchmark_condition_replay(
    *,
    condition_ids: tuple[int, ...] = tuple(range(3, 26)),
    capacity_per_condition: int = 40_000,
    transitions_per_condition: int = 40_000,
    batch_size: int = 128,
    sample_batches: int = 1_000,
    seed: int = 42,
) -> dict[str, Any]:
    if not condition_ids or capacity_per_condition <= 0 or transitions_per_condition <= 0:
        raise ValueError('conditions and replay capacities must be positive')
    if transitions_per_condition > capacity_per_condition:
        raise ValueError('transitions_per_condition cannot exceed the configured ring capacity')
    if batch_size <= 0 or sample_batches <= 0:
        raise ValueError('batch_size and sample_batches must be positive')
    np.random.seed(seed)
    rss_before = _rss_bytes()
    buffer = ConditionStratifiedReplayBuffer(
        obs_dim=20,
        action_dim=2,
        condition_ids=condition_ids,
        capacity_per_condition=capacity_per_condition,
        min_per_condition_in_batch=min(5, max(1, batch_size // len(condition_ids))),
        source_max_fraction={LOW_SPEED_RECOVERY_REPLAY_SOURCE: 0.25},
    )
    rss_allocated = _rss_bytes()
    zero_action = np.zeros(2, dtype=np.float32)
    zero_next_obs = np.zeros(20, dtype=np.float32)
    started = time.perf_counter()
    actor_per_condition = transitions_per_condition - transitions_per_condition // 4
    recovery_per_condition = transitions_per_condition - actor_per_condition
    # Synthetic bounded replay workload; never reads raw experiment files or starts UDP.
    for condition_id in condition_ids:
        for index in range(transitions_per_condition):
            obs = np.full(20, condition_id + index * 1.0e-6, dtype=np.float32)
            source = ACTOR_REPLAY_SOURCE if index < actor_per_condition else LOW_SPEED_RECOVERY_REPLAY_SOURCE
            buffer.add(
                obs,
                zero_action,
                float(index % 7) / 7.0,
                zero_next_obs,
                False,
                condition_id=condition_id,
                condition_instance_id=condition_id * 100,
                phase_elapsed_s=index * 0.02,
                source=source,
            )
    fill_seconds = time.perf_counter() - started
    rss_filled = _rss_bytes()
    latencies_ms: list[float] = []
    total_samples = 0
    total_quota_relaxed = 0
    source_totals = np.zeros(len(REPLAY_SOURCE_NAMES), dtype=np.int64)
    recovery_index = REPLAY_SOURCE_NAMES.index(LOW_SPEED_RECOVERY_REPLAY_SOURCE)
    actor_index = REPLAY_SOURCE_NAMES.index(ACTOR_REPLAY_SOURCE)
    sample_started = time.perf_counter()
    for _ in range(sample_batches):
        started_batch = time.perf_counter()
        batch = buffer.sample(batch_size)
        latencies_ms.append((time.perf_counter() - started_batch) * 1000.0)
        counts = batch['source_sample_counts'].numpy()
        source_totals += counts
        total_samples += int(counts.sum())
        total_quota_relaxed += int(batch['source_quota_relaxed_count'].item())
        if int(batch['condition_source_sample_counts'].sum()) != batch_size:
            raise RuntimeError('Replay sample condition/source counts do not close')
        if int(batch['condition_quota_relaxed_count'].sum()) != int(batch['source_quota_relaxed_count'].item()):
            raise RuntimeError('Replay quota-relaxed counts do not close by condition')
    sampling_seconds = time.perf_counter() - sample_started
    latency_array = np.asarray(latencies_ms, dtype=np.float64)
    total_expected = len(condition_ids) * transitions_per_condition
    recovery_fraction = float(source_totals[recovery_index] / max(total_samples, 1))
    actor_fraction = float(source_totals[actor_index] / max(total_samples, 1))
    relaxed_rate = total_quota_relaxed / max(total_samples, 1)
    if len(REPLAY_SOURCE_NAMES) and actor_fraction > 0.0 and recovery_fraction > 0.25 + 1.0e-12:
        raise RuntimeError('Configured recovery source quota was violated while Actor samples were available')
    if total_samples != sample_batches * batch_size:
        raise RuntimeError('Replay sampled transition count does not match requested batches')
    return {
        'schema_version': 'condition_replay_resource_benchmark_v1',
        'mode': 'synthetic_no_udp',
        'condition_count': len(condition_ids),
        'condition_ids': list(condition_ids),
        'capacity_per_condition': capacity_per_condition,
        'transitions_per_condition': transitions_per_condition,
        'stored_transition_count': len(buffer),
        'expected_stored_transition_count': total_expected,
        'batch_size': batch_size,
        'sample_batches': sample_batches,
        'sampled_transition_count': total_samples,
        'fill_duration_s': fill_seconds,
        'sample_duration_s': sampling_seconds,
        'sample_batches_per_second': sample_batches / max(sampling_seconds, 1.0e-12),
        'batch_latency_ms': {
            'p50': float(np.percentile(latency_array, 50)),
            'p95': float(np.percentile(latency_array, 95)),
            'p99': float(np.percentile(latency_array, 99)),
            'max': float(np.max(latency_array)),
        },
        'source_sample_count': {
            name: int(source_totals[index])
            for index, name in enumerate(REPLAY_SOURCE_NAMES)
        },
        'recovery_sample_fraction': recovery_fraction,
        'actor_sample_fraction': actor_fraction,
        'quota_relaxed_count': total_quota_relaxed,
        'quota_relaxed_rate': relaxed_rate,
        'replay_numpy_array_bytes': _buffer_array_bytes(buffer),
        'process_rss_before_bytes': rss_before,
        'process_rss_after_allocation_bytes': rss_allocated,
        'process_rss_after_fill_bytes': rss_filled,
        'process_rss_after_sampling_bytes': _rss_bytes(),
        'quota_pass': recovery_fraction <= 0.25 + 1.0e-12,
        'batch_count_closes': total_samples == sample_batches * batch_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Synthetic no-UDP condition/source replay resource benchmark.')
    parser.add_argument('--conditions', type=int, default=23)
    parser.add_argument('--capacity-per-condition', type=int, default=40_000)
    parser.add_argument('--transitions-per-condition', type=int, default=40_000)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--sample-batches', type=int, default=1_000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()
    report = benchmark_condition_replay(
        condition_ids=tuple(range(3, 3 + args.conditions)),
        capacity_per_condition=args.capacity_per_condition,
        transitions_per_condition=args.transitions_per_condition,
        batch_size=args.batch_size,
        sample_batches=args.sample_batches,
        seed=args.seed,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
