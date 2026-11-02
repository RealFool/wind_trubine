from __future__ import annotations

import unittest

from RL.utils.critic_q_health import resolve_actor_gate_mode


class TestActorGateMode(unittest.TestCase):
    def test_auto_prefers_optimizer(self) -> None:
        result = resolve_actor_gate_mode({
            'actor_q_health_gate': {'enabled': True},
            'actor_optimizer_health_gate': {'enabled': True},
        })
        self.assertEqual(result['mode'], 'optimizer')
        self.assertTrue(result['trend_configured'])
        self.assertTrue(result['optimizer_configured'])
        self.assertIn('not instantiated', result['note'])

    def test_auto_falls_back_to_trend(self) -> None:
        result = resolve_actor_gate_mode({
            'actor_q_health_gate': {'enabled': True},
        })
        self.assertEqual(result['mode'], 'trend')

    def test_explicit_none_disables_configured_gates(self) -> None:
        result = resolve_actor_gate_mode({
            'actor_gate_mode': 'none',
            'actor_q_health_gate': {'enabled': True},
            'actor_optimizer_health_gate': {'enabled': True},
        })
        self.assertEqual(result['mode'], 'none')
        self.assertIn('disabled', result['note'])

    def test_explicit_trend_requires_enabled_trend(self) -> None:
        with self.assertRaises(ValueError):
            resolve_actor_gate_mode({'actor_gate_mode': 'trend'})

    def test_invalid_mode_rejected(self) -> None:
        with self.assertRaises(ValueError):
            resolve_actor_gate_mode({'actor_gate_mode': 'both'})


if __name__ == '__main__':
    unittest.main()
