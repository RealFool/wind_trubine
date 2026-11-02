from __future__ import annotations

import unittest

from RL.utils.condition_queue import ConditionQueueLifecycle


class TestConditionQueueLifecycle(unittest.TestCase):
    def test_round_requires_each_condition_once(self) -> None:
        lifecycle = ConditionQueueLifecycle(
            condition_ids=[3, 4, 5],
            total_rounds=2,
            critic_start_after_rounds=1,
            actor_start_after_rounds=2,
        )

        self.assertEqual(lifecycle.training_stage(all_conditions_covered=True), 'collect')
        lifecycle.complete_condition(3, 101)
        lifecycle.complete_condition(4, 102)
        duplicate = lifecycle.complete_condition(4, 102)
        self.assertFalse(duplicate.completed_round)
        self.assertEqual(duplicate.conditions_completed_in_round, 2)

        first_round = lifecycle.complete_condition(5, 103)
        self.assertTrue(first_round.completed_round)
        self.assertEqual(first_round.completed_rounds, 1)
        self.assertEqual(lifecycle.conditions_completed_in_round, 0)
        self.assertEqual(lifecycle.training_stage(all_conditions_covered=True), 'critic')

        lifecycle.complete_condition(3, 201)
        lifecycle.complete_condition(4, 202)
        second_round = lifecycle.complete_condition(5, 203)
        self.assertTrue(second_round.completed_round)
        self.assertTrue(lifecycle.finished)
        self.assertEqual(lifecycle.training_stage(all_conditions_covered=True), 'actor')

    def test_coverage_gate_holds_collect_stage(self) -> None:
        lifecycle = ConditionQueueLifecycle(
            condition_ids=[3],
            total_rounds=1,
            critic_start_after_rounds=0,
            actor_start_after_rounds=0,
        )

        self.assertEqual(lifecycle.training_stage(all_conditions_covered=False), 'collect')
        self.assertEqual(lifecycle.training_stage(all_conditions_covered=True), 'actor')

    def test_rejects_unknown_condition(self) -> None:
        lifecycle = ConditionQueueLifecycle(condition_ids=[3], total_rounds=1)
        with self.assertRaisesRegex(ValueError, 'Unknown condition_id=4'):
            lifecycle.complete_condition(4, 1)


if __name__ == '__main__':
    unittest.main()