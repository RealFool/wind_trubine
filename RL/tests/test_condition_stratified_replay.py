from __future__ import annotations

import unittest

import numpy as np

from RL.replay import ConditionStratifiedReplayBuffer
from RL.replay.replay_buffer import REPLAY_SOURCE_NAMES
from RL.utils.control_execution import (
    ACTOR_REPLAY_SOURCE,
    EXPLICIT_BASELINE_REPLAY_SOURCE,
    LOW_SPEED_RECOVERY_REPLAY_SOURCE,
)


class TestConditionStratifiedReplayBuffer(unittest.TestCase):
    def _buffer(self) -> ConditionStratifiedReplayBuffer:
        return ConditionStratifiedReplayBuffer(
            obs_dim=2,
            action_dim=2,
            condition_ids=[3, 4, 5],
            capacity_per_condition=8,
            min_per_condition_in_batch=2,
        )

    @staticmethod
    def _add(buffer: ConditionStratifiedReplayBuffer, condition_id: int, count: int) -> None:
        for index in range(count):
            value = float(condition_id * 100 + index)
            buffer.add(
                obs=np.asarray([value, value + 1.0], dtype=np.float32),
                action=np.asarray([value, -value], dtype=np.float32),
                reward=value,
                next_obs=np.asarray([value + 2.0, value + 3.0], dtype=np.float32),
                done=False,
                condition_id=condition_id,
                condition_instance_id=condition_id * 10,
                phase_elapsed_s=float(index),
            )

    def test_balanced_batch_includes_each_available_condition(self) -> None:
        buffer = self._buffer()
        for condition_id in [3, 4, 5]:
            self._add(buffer, condition_id, count=3)

        batch = buffer.sample(batch_size=10)
        condition_ids = batch['condition_ids'].numpy().reshape(-1)
        counts = {condition_id: int(np.count_nonzero(condition_ids == condition_id)) for condition_id in [3, 4, 5]}

        self.assertEqual(sum(counts.values()), 10)
        self.assertGreaterEqual(counts[3], 2)
        self.assertGreaterEqual(counts[4], 2)
        self.assertGreaterEqual(counts[5], 2)
        self.assertEqual(batch['condition_instance_ids'].shape, (10, 1))
        self.assertEqual(batch['phase_elapsed_s'].shape, (10, 1))

        source_counts = batch['source_sample_counts'].numpy()
        condition_source_counts = batch['condition_source_sample_counts'].numpy()
        condition_quota_relaxed = batch['condition_quota_relaxed_count'].numpy()
        condition_source_quota_relaxed = batch['condition_source_quota_relaxed_by_source'].numpy()
        self.assertEqual(source_counts.shape, (len(REPLAY_SOURCE_NAMES),))
        self.assertEqual(condition_source_counts.shape, (3, len(REPLAY_SOURCE_NAMES)))
        self.assertEqual(condition_quota_relaxed.shape, (3,))
        self.assertEqual(condition_source_quota_relaxed.shape, (3, len(REPLAY_SOURCE_NAMES)))
        self.assertEqual(int(source_counts.sum()), 10)
        self.assertEqual(int(condition_source_counts.sum()), 10)
        self.assertEqual(int(batch['condition_ids'].shape[0]), 10)

    def test_coverage_gate_reports_missing_conditions(self) -> None:
        buffer = self._buffer()
        self._add(buffer, 3, count=3)
        self._add(buffer, 4, count=2)
        self._add(buffer, 5, count=3)

        self.assertEqual(buffer.condition_counts(), {3: 3, 4: 2, 5: 3})
        self.assertEqual(buffer.missing_conditions(min_valid_transitions=3), [4])
        self.assertFalse(buffer.all_conditions_covered(min_valid_transitions=3))

        self._add(buffer, 4, count=1)
        self.assertTrue(buffer.all_conditions_covered(min_valid_transitions=3))

    def test_unknown_condition_is_rejected(self) -> None:
        buffer = self._buffer()
        with self.assertRaisesRegex(ValueError, 'Unknown condition_id=6'):
            buffer.add(
                obs=np.zeros(2, dtype=np.float32),
                action=np.zeros(2, dtype=np.float32),
                reward=0.0,
                next_obs=np.zeros(2, dtype=np.float32),
                done=False,
                condition_id=6,
            )

    def test_source_quota_limits_recovery_sampling_when_actor_data_is_available(self) -> None:
        buffer = ConditionStratifiedReplayBuffer(
            obs_dim=2,
            action_dim=2,
            condition_ids=[3],
            capacity_per_condition=16,
            min_per_condition_in_batch=1,
            source_max_fraction={LOW_SPEED_RECOVERY_REPLAY_SOURCE: 0.25},
        )
        for index in range(8):
            source = ACTOR_REPLAY_SOURCE if index < 4 else LOW_SPEED_RECOVERY_REPLAY_SOURCE
            buffer.add(
                obs=np.asarray([float(index), 0.0]),
                action=np.zeros(2),
                reward=0.0,
                next_obs=np.zeros(2),
                done=False,
                condition_id=3,
                source=source,
            )

        batch = buffer.sample(8)
        recovery_code = 4
        self.assertLessEqual(int(np.count_nonzero(batch['sources'].numpy() == recovery_code)), 2)
        self.assertEqual(float(batch['source_quota_relaxed_count']), 0.0)
        self.assertEqual(int(batch['source_sample_counts'].sum()), 8)
        self.assertEqual(int(batch['source_quota_relaxed_by_source'].sum()), 0)
        self.assertLessEqual(
            int(batch['source_sample_counts'][recovery_code]),
            int(np.floor(8 * 0.25)),
        )

    def test_source_index_cache_tracks_ring_overwrites_and_quota_relaxation(self) -> None:
        buffer = ConditionStratifiedReplayBuffer(
            obs_dim=2,
            action_dim=2,
            condition_ids=[3],
            capacity_per_condition=3,
            min_per_condition_in_batch=1,
            source_max_fraction={LOW_SPEED_RECOVERY_REPLAY_SOURCE: 0.25},
        )
        for index, source in enumerate((ACTOR_REPLAY_SOURCE, ACTOR_REPLAY_SOURCE, ACTOR_REPLAY_SOURCE)):
            buffer.add(
                np.asarray([index, 0.0], dtype=np.float32),
                np.zeros(2, dtype=np.float32),
                0.0,
                np.zeros(2, dtype=np.float32),
                False,
                condition_id=3,
                source=source,
            )

        # Replace each ring slot with two new sources and one recovery sample.
        for index, source in enumerate((LOW_SPEED_RECOVERY_REPLAY_SOURCE, EXPLICIT_BASELINE_REPLAY_SOURCE, EXPLICIT_BASELINE_REPLAY_SOURCE)):
            buffer.add(
                np.asarray([10 + index, 0.0], dtype=np.float32),
                np.zeros(2, dtype=np.float32),
                0.0,
                np.zeros(2, dtype=np.float32),
                False,
                condition_id=3,
                source=source,
            )

        bucket = buffer._buckets[3]
        for source_code, source_count in enumerate(bucket.source_counts):
            slots = bucket.source_indices[source_code, :int(source_count)]
            self.assertEqual(len(slots), len(np.unique(slots)))
            self.assertTrue(all(int(bucket.sources[slot, 0]) == source_code for slot in slots))
            self.assertTrue(all(bucket.source_positions[slot] == offset for offset, slot in enumerate(slots)))
        self.assertEqual(int(bucket.source_counts[1]), 0)
        self.assertEqual(int(bucket.source_counts[2]), 2)
        self.assertEqual(int(bucket.source_counts[4]), 1)

        # With only recovery data available, the buffer must explicitly report
        # that it relaxed the 25% source cap rather than silently hiding it.
        recovery_only = ConditionStratifiedReplayBuffer(
            obs_dim=2,
            action_dim=2,
            condition_ids=[3],
            capacity_per_condition=8,
            min_per_condition_in_batch=1,
            source_max_fraction={LOW_SPEED_RECOVERY_REPLAY_SOURCE: 0.25},
        )
        for index in range(4):
            recovery_only.add(
                np.asarray([float(index), 0.0], dtype=np.float32),
                np.zeros(2, dtype=np.float32),
                0.0,
                np.zeros(2, dtype=np.float32),
                False,
                condition_id=3,
                source=LOW_SPEED_RECOVERY_REPLAY_SOURCE,
            )
        batch = recovery_only.sample(4)
        recovery_code = REPLAY_SOURCE_NAMES.index(LOW_SPEED_RECOVERY_REPLAY_SOURCE)
        self.assertEqual(int(batch['source_sample_counts'][recovery_code]), 4)
        self.assertEqual(float(batch['source_quota_relaxed_count']), 3.0)
        self.assertEqual(int(batch['source_quota_relaxed_by_source'][recovery_code]), 3)
        self.assertEqual(int(batch['condition_source_sample_counts'].sum()), 4)
        self.assertEqual(int(batch['condition_quota_relaxed_count'][0]), 3)
        self.assertEqual(
            int(batch['condition_source_quota_relaxed_by_source'][0, recovery_code]),
            3,
        )


if __name__ == '__main__':
    unittest.main()