from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import torch

from RL.utils.control_execution import (
    ACTOR_EXECUTION_ROLLBACK_REPLAY_SOURCE,
    ACTOR_REPLAY_SOURCE,
    EXPLICIT_BASELINE_REPLAY_SOURCE,
    LOW_SPEED_RECOVERY_REPLAY_SOURCE,
)


ONLINE_BLADED_SOURCE = 'online_bladed'
_SOURCE_CODES = {
    ONLINE_BLADED_SOURCE: 0,
    ACTOR_REPLAY_SOURCE: 1,
    EXPLICIT_BASELINE_REPLAY_SOURCE: 2,
    ACTOR_EXECUTION_ROLLBACK_REPLAY_SOURCE: 3,
    LOW_SPEED_RECOVERY_REPLAY_SOURCE: 4,
}
REPLAY_SOURCE_NAMES = tuple(
    source for source, _source_code in sorted(_SOURCE_CODES.items(), key=lambda item: item[1])
)


class ReplayBuffer:
    def __init__(self, obs_dim: int, action_dim: int, capacity: int) -> None:
        self.capacity = int(capacity)
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.position = 0
        self.size = 0

    def add(self, obs: np.ndarray, action: np.ndarray, reward: float, next_obs: np.ndarray, done: bool) -> None:
        idx = self.position
        self.obs[idx] = obs
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_obs[idx] = next_obs
        self.dones[idx] = float(done)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def __len__(self) -> int:
        return self.size

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        indices = np.random.randint(0, self.size, size=batch_size)
        return {
            'obs': torch.from_numpy(self.obs[indices]),
            'actions': torch.from_numpy(self.actions[indices]),
            'rewards': torch.from_numpy(self.rewards[indices]),
            'next_obs': torch.from_numpy(self.next_obs[indices]),
            'dones': torch.from_numpy(self.dones[indices]),
        }


@dataclass
class _ConditionBucket:
    obs: np.ndarray
    next_obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    condition_instance_ids: np.ndarray
    phase_elapsed_s: np.ndarray
    sources: np.ndarray
    source_indices: np.ndarray
    source_counts: np.ndarray
    source_positions: np.ndarray
    position: int = 0
    size: int = 0


class ConditionStratifiedReplayBuffer:
    """A replay buffer that preserves equal sampling opportunity for configured wind conditions."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        condition_ids: Iterable[int],
        capacity_per_condition: int,
        min_per_condition_in_batch: int = 5,
        source_max_fraction: dict[str, float] | None = None,
    ) -> None:
        self.condition_ids = tuple(sorted({int(condition_id) for condition_id in condition_ids}))
        if not self.condition_ids:
            raise ValueError('condition_ids must contain at least one condition')
        if capacity_per_condition <= 0:
            raise ValueError('capacity_per_condition must be positive')
        if min_per_condition_in_batch <= 0:
            raise ValueError('min_per_condition_in_batch must be positive')
        self.capacity_per_condition = int(capacity_per_condition)
        self.min_per_condition_in_batch = int(min_per_condition_in_batch)
        self.source_max_fraction = self._validate_source_max_fraction(source_max_fraction)
        self._source_names = REPLAY_SOURCE_NAMES
        self._round_robin_offset = 0
        self._buckets = {
            condition_id: self._new_bucket(obs_dim, action_dim)
            for condition_id in self.condition_ids
        }

    def _new_bucket(self, obs_dim: int, action_dim: int) -> _ConditionBucket:
        capacity = self.capacity_per_condition
        return _ConditionBucket(
            obs=np.zeros((capacity, obs_dim), dtype=np.float32),
            next_obs=np.zeros((capacity, obs_dim), dtype=np.float32),
            actions=np.zeros((capacity, action_dim), dtype=np.float32),
            rewards=np.zeros((capacity, 1), dtype=np.float32),
            dones=np.zeros((capacity, 1), dtype=np.float32),
            condition_instance_ids=np.full((capacity, 1), -1, dtype=np.int64),
            phase_elapsed_s=np.full((capacity, 1), np.nan, dtype=np.float32),
            sources=np.full((capacity, 1), _SOURCE_CODES[ONLINE_BLADED_SOURCE], dtype=np.int8),
            source_indices=np.full((len(_SOURCE_CODES), capacity), -1, dtype=np.int32),
            source_counts=np.zeros(len(_SOURCE_CODES), dtype=np.int32),
            source_positions=np.full(capacity, -1, dtype=np.int32),
        )

    @property
    def source_names(self) -> tuple[str, ...]:
        return self._source_names

    @staticmethod
    def _remove_source_index(bucket: _ConditionBucket, source_code: int, slot: int) -> None:
        source_count = int(bucket.source_counts[source_code])
        list_position = int(bucket.source_positions[slot])
        if list_position < 0 or source_count <= 0:
            raise RuntimeError('Replay source index is missing for an occupied ring slot')
        last_position = source_count - 1
        last_slot = int(bucket.source_indices[source_code, last_position])
        if list_position != last_position:
            bucket.source_indices[source_code, list_position] = last_slot
            bucket.source_positions[last_slot] = list_position
        bucket.source_indices[source_code, last_position] = -1
        bucket.source_positions[slot] = -1
        bucket.source_counts[source_code] = last_position

    @staticmethod
    def _add_source_index(bucket: _ConditionBucket, source_code: int, slot: int) -> None:
        source_count = int(bucket.source_counts[source_code])
        bucket.source_indices[source_code, source_count] = slot
        bucket.source_positions[slot] = source_count
        bucket.source_counts[source_code] = source_count + 1

    @staticmethod
    def _validate_source_max_fraction(
        source_max_fraction: dict[str, float] | None,
    ) -> dict[str, float]:
        if source_max_fraction is None:
            return {}
        if not isinstance(source_max_fraction, dict):
            raise ValueError('source_max_fraction must be an object when configured')
        result: dict[str, float] = {}
        for source, fraction in source_max_fraction.items():
            if source not in _SOURCE_CODES:
                raise ValueError(f'Unknown replay source quota={source!r}')
            parsed = float(fraction)
            if not np.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
                raise ValueError(f'Replay source quota for {source!r} must be in [0, 1]')
            result[source] = parsed
        return result

    @property
    def total_capacity(self) -> int:
        return len(self.condition_ids) * self.capacity_per_condition

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        *,
        condition_id: int,
        condition_instance_id: int | None = None,
        phase_elapsed_s: float | None = None,
        source: str = ONLINE_BLADED_SOURCE,
    ) -> None:
        condition_id = int(condition_id)
        if condition_id not in self._buckets:
            raise ValueError(f'Unknown condition_id={condition_id}; expected one of {self.condition_ids}')
        source_code = _SOURCE_CODES.get(source)
        if source_code is None:
            raise ValueError(f'Unknown replay source={source!r}; expected one of {list(_SOURCE_CODES)}')
        bucket = self._buckets[condition_id]
        index = bucket.position
        if bucket.size == self.capacity_per_condition:
            previous_source_code = int(bucket.sources[index, 0])
            self._remove_source_index(bucket, previous_source_code, index)
        bucket.obs[index] = obs
        bucket.actions[index] = action
        bucket.rewards[index] = reward
        bucket.next_obs[index] = next_obs
        bucket.dones[index] = float(done)
        bucket.condition_instance_ids[index] = -1 if condition_instance_id is None else int(condition_instance_id)
        bucket.phase_elapsed_s[index] = np.nan if phase_elapsed_s is None else float(phase_elapsed_s)
        bucket.sources[index] = source_code
        self._add_source_index(bucket, source_code, index)
        bucket.position = (index + 1) % self.capacity_per_condition
        bucket.size = min(bucket.size + 1, self.capacity_per_condition)

    def __len__(self) -> int:
        return sum(bucket.size for bucket in self._buckets.values())

    def condition_counts(self) -> dict[int, int]:
        return {condition_id: bucket.size for condition_id, bucket in self._buckets.items()}

    def condition_source_counts(self) -> dict[int, dict[str, int]]:
        return {
            condition_id: {
                source: int(np.count_nonzero(bucket.sources[:bucket.size] == source_code))
                for source, source_code in _SOURCE_CODES.items()
            }
            for condition_id, bucket in self._buckets.items()
        }

    def missing_conditions(self, min_valid_transitions: int) -> list[int]:
        if min_valid_transitions < 0:
            raise ValueError('min_valid_transitions must be non-negative')
        return [
            condition_id
            for condition_id, count in self.condition_counts().items()
            if count < min_valid_transitions
        ]

    def all_conditions_covered(self, min_valid_transitions: int) -> bool:
        return not self.missing_conditions(min_valid_transitions)

    def _sample_allocation(self, batch_size: int) -> dict[int, int]:
        if batch_size <= 0:
            raise ValueError('batch_size must be positive')
        available = [condition_id for condition_id in self.condition_ids if self._buckets[condition_id].size > 0]
        if not available:
            raise ValueError('Cannot sample an empty condition-stratified replay buffer')

        allocation = {condition_id: 0 for condition_id in available}
        guaranteed_count = min(
            self.min_per_condition_in_batch,
            batch_size // len(available),
        )
        if guaranteed_count > 0:
            for condition_id in available:
                allocation[condition_id] = guaranteed_count

        remaining = batch_size - guaranteed_count * len(available)
        for index in range(remaining):
            condition_id = available[(self._round_robin_offset + index) % len(available)]
            allocation[condition_id] += 1
        self._round_robin_offset = (self._round_robin_offset + remaining) % len(available)
        return allocation

    def _source_budgets(self, batch_size: int) -> dict[int, int]:
        return {
            source_code: (
                batch_size
                if source not in self.source_max_fraction
                else int(np.floor(batch_size * self.source_max_fraction[source]))
            )
            for source, source_code in _SOURCE_CODES.items()
        }

    @staticmethod
    def _bucket_source_codes(bucket: _ConditionBucket) -> list[int]:
        return [
            source_code
            for source_code, source_count in enumerate(bucket.source_counts)
            if source_count > 0
        ]

    def _sample_bucket_indices(
        self,
        bucket: _ConditionBucket,
        count: int,
        source_budgets: dict[int, int],
    ) -> tuple[np.ndarray, int, dict[int, int]]:
        selected: list[int] = []
        quota_relaxed = 0
        quota_relaxed_by_source = {source_code: 0 for source_code in _SOURCE_CODES.values()}
        source_codes = self._bucket_source_codes(bucket)
        for _ in range(count):
            eligible_codes = [code for code in source_codes if source_budgets[code] > 0]
            if not eligible_codes:
                eligible_codes = source_codes
                quota_relaxed += 1
            selected_code = int(np.random.choice(eligible_codes))
            source_count = int(bucket.source_counts[selected_code])
            selected_position = int(np.random.randint(source_count))
            selected.append(int(bucket.source_indices[selected_code, selected_position]))
            if source_budgets[selected_code] > 0:
                source_budgets[selected_code] -= 1
            else:
                quota_relaxed_by_source[selected_code] += 1
        return np.asarray(selected, dtype=np.int64), quota_relaxed, quota_relaxed_by_source

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        allocation = self._sample_allocation(batch_size)
        obs_chunks: list[np.ndarray] = []
        action_chunks: list[np.ndarray] = []
        reward_chunks: list[np.ndarray] = []
        next_obs_chunks: list[np.ndarray] = []
        done_chunks: list[np.ndarray] = []
        condition_chunks: list[np.ndarray] = []
        instance_chunks: list[np.ndarray] = []
        phase_elapsed_chunks: list[np.ndarray] = []
        source_chunks: list[np.ndarray] = []
        sampled_condition_chunks: list[np.ndarray] = []
        source_budgets = self._source_budgets(batch_size)
        quota_relaxed_count = 0
        quota_relaxed_by_source = {source_code: 0 for source_code in _SOURCE_CODES.values()}
        condition_row = {condition_id: row for row, condition_id in enumerate(self.condition_ids)}
        condition_quota_relaxed_count = np.zeros(len(self.condition_ids), dtype=np.int64)
        condition_source_quota_relaxed_by_source = np.zeros(
            (len(self.condition_ids), len(_SOURCE_CODES)),
            dtype=np.int64,
        )

        for condition_id, count in allocation.items():
            if count == 0:
                continue
            bucket = self._buckets[condition_id]
            indices, relaxed, relaxed_by_source = self._sample_bucket_indices(bucket, count, source_budgets)
            quota_relaxed_count += relaxed
            condition_quota_relaxed_count[condition_row[condition_id]] += relaxed
            for source_code, source_count in relaxed_by_source.items():
                quota_relaxed_by_source[source_code] += source_count
                condition_source_quota_relaxed_by_source[condition_row[condition_id], source_code] += source_count
            obs_chunks.append(bucket.obs[indices])
            action_chunks.append(bucket.actions[indices])
            reward_chunks.append(bucket.rewards[indices])
            next_obs_chunks.append(bucket.next_obs[indices])
            done_chunks.append(bucket.dones[indices])
            condition_chunks.append(np.full((count, 1), condition_id, dtype=np.int64))
            instance_chunks.append(bucket.condition_instance_ids[indices])
            phase_elapsed_chunks.append(bucket.phase_elapsed_s[indices])
            source_chunks.append(bucket.sources[indices])
            sampled_condition_chunks.append(np.full((count, 1), condition_id, dtype=np.int64))

        permutation = np.random.permutation(batch_size)
        sampled_sources = np.concatenate(source_chunks, axis=0)[permutation, 0]
        sampled_conditions = np.concatenate(sampled_condition_chunks, axis=0)[permutation, 0]
        source_sample_counts = np.bincount(
            sampled_sources.astype(np.int64, copy=False),
            minlength=len(_SOURCE_CODES),
        )
        condition_source_sample_counts = np.zeros(
            (len(self.condition_ids), len(_SOURCE_CODES)),
            dtype=np.int64,
        )
        np.add.at(
            condition_source_sample_counts,
            (np.fromiter((condition_row[int(value)] for value in sampled_conditions), dtype=np.int64), sampled_sources.astype(np.int64, copy=False)),
            1,
        )
        if int(condition_quota_relaxed_count.sum()) != quota_relaxed_count:
            raise RuntimeError('Replay quota-relaxed count does not close across conditions')
        if int(condition_source_quota_relaxed_by_source.sum()) != quota_relaxed_count:
            raise RuntimeError('Replay quota-relaxed source count does not close across conditions')
        if sum(quota_relaxed_by_source.values()) != quota_relaxed_count:
            raise RuntimeError('Replay quota-relaxed source counts do not match the total')
        return {
            'obs': torch.from_numpy(np.concatenate(obs_chunks, axis=0)[permutation]),
            'actions': torch.from_numpy(np.concatenate(action_chunks, axis=0)[permutation]),
            'rewards': torch.from_numpy(np.concatenate(reward_chunks, axis=0)[permutation]),
            'next_obs': torch.from_numpy(np.concatenate(next_obs_chunks, axis=0)[permutation]),
            'dones': torch.from_numpy(np.concatenate(done_chunks, axis=0)[permutation]),
            'condition_ids': torch.from_numpy(np.concatenate(condition_chunks, axis=0)[permutation]),
            'condition_instance_ids': torch.from_numpy(np.concatenate(instance_chunks, axis=0)[permutation]),
            'phase_elapsed_s': torch.from_numpy(np.concatenate(phase_elapsed_chunks, axis=0)[permutation]),
            'sources': torch.from_numpy(np.concatenate(source_chunks, axis=0)[permutation]),
            'condition_ids': torch.from_numpy(np.concatenate(condition_chunks, axis=0)[permutation]),
            'source_sample_counts': torch.from_numpy(source_sample_counts),
            'condition_source_sample_counts': torch.from_numpy(condition_source_sample_counts),
            'condition_quota_relaxed_count': torch.from_numpy(condition_quota_relaxed_count),
            'condition_source_quota_relaxed_by_source': torch.from_numpy(condition_source_quota_relaxed_by_source),
            'source_quota_relaxed_by_source': torch.as_tensor(
                [quota_relaxed_by_source[source_code] for source_code in range(len(_SOURCE_CODES))],
                dtype=torch.int64,
            ),
            'source_quota_relaxed_count': torch.tensor(float(quota_relaxed_count)),
        }
