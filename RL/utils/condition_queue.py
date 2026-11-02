from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ConditionCompletion:
    condition_id: int
    condition_instance_id: int
    completed_round: bool
    completed_rounds: int
    conditions_completed_in_round: int


class ConditionQueueLifecycle:
    """Tracks full condition rounds independently from control-frame or log segments."""

    def __init__(
        self,
        condition_ids: Iterable[int],
        total_rounds: int,
        critic_start_after_rounds: int = 1,
        actor_start_after_rounds: int = 2,
    ) -> None:
        self.condition_ids = tuple(sorted({int(condition_id) for condition_id in condition_ids}))
        if not self.condition_ids:
            raise ValueError('condition_ids must contain at least one condition')
        if total_rounds <= 0:
            raise ValueError('total_rounds must be positive')
        if critic_start_after_rounds < 0 or actor_start_after_rounds < 0:
            raise ValueError('round gates must be non-negative')
        if actor_start_after_rounds < critic_start_after_rounds:
            raise ValueError('actor_start_after_rounds must not precede critic_start_after_rounds')
        self.total_rounds = int(total_rounds)
        self.critic_start_after_rounds = int(critic_start_after_rounds)
        self.actor_start_after_rounds = int(actor_start_after_rounds)
        self.completed_rounds = 0
        self._completed_condition_ids: set[int] = set()
        self._completed_instance_ids: set[int] = set()

    @property
    def conditions_completed_in_round(self) -> int:
        return len(self._completed_condition_ids)

    @property
    def finished(self) -> bool:
        return self.completed_rounds >= self.total_rounds

    def training_stage(self, all_conditions_covered: bool) -> str:
        if not all_conditions_covered or self.completed_rounds < self.critic_start_after_rounds:
            return 'collect'
        if self.completed_rounds < self.actor_start_after_rounds:
            return 'critic'
        return 'actor'

    def complete_condition(self, condition_id: int, condition_instance_id: int) -> ConditionCompletion:
        condition_id = int(condition_id)
        condition_instance_id = int(condition_instance_id)
        if condition_id not in self.condition_ids:
            raise ValueError(f'Unknown condition_id={condition_id}; expected one of {self.condition_ids}')
        if condition_instance_id in self._completed_instance_ids:
            return ConditionCompletion(
                condition_id=condition_id,
                condition_instance_id=condition_instance_id,
                completed_round=False,
                completed_rounds=self.completed_rounds,
                conditions_completed_in_round=self.conditions_completed_in_round,
            )

        self._completed_instance_ids.add(condition_instance_id)
        self._completed_condition_ids.add(condition_id)
        completed_round = len(self._completed_condition_ids) == len(self.condition_ids)
        if completed_round:
            self.completed_rounds += 1
            self._completed_condition_ids.clear()
            self._completed_instance_ids.clear()
        return ConditionCompletion(
            condition_id=condition_id,
            condition_instance_id=condition_instance_id,
            completed_round=completed_round,
            completed_rounds=self.completed_rounds,
            conditions_completed_in_round=self.conditions_completed_in_round,
        )