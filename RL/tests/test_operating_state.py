from __future__ import annotations

import unittest

from RL.utils.operating_state import (
    ControlRegion,
    OperatingPoint,
    OperatingState,
    OperatingStateClassifier,
    OperatingStateConfig,
)


class TestOperatingStateClassifier(unittest.TestCase):
    @staticmethod
    def _point(
        timestamp_s: float,
        wind_speed_mps: float,
        *,
        rotor_speed_rad_s: float = 0.7,
        real_power_w: float = 5_000_000.0,
    ) -> OperatingPoint:
        return OperatingPoint(
            timestamp_s=timestamp_s,
            wind_speed_mps=wind_speed_mps,
            rotor_speed_rad_s=rotor_speed_rad_s,
            real_power_w=real_power_w,
        )

    def test_single_wind_spike_does_not_switch_region(self) -> None:
        classifier = OperatingStateClassifier(OperatingStateConfig(wind_filter_time_constant_s=0.0))

        initial = classifier.classify(self._point(0.0, 8.0))
        spike = classifier.classify(self._point(0.02, 16.0))
        recovered = classifier.classify(self._point(0.04, 8.0))

        self.assertEqual(initial.control_region, ControlRegion.BELOW_RATED)
        self.assertEqual(spike.control_region, ControlRegion.BELOW_RATED)
        self.assertEqual(recovered.control_region, ControlRegion.BELOW_RATED)

    def test_sustained_crossing_uses_ordered_regions_and_hysteresis(self) -> None:
        classifier = OperatingStateClassifier(OperatingStateConfig(
            wind_filter_time_constant_s=0.0,
            region_confirmation_s=0.3,
            running_confirmation_s=0.0,
        ))
        classifier.classify(self._point(0.0, 8.0))

        transition = None
        for step in range(1, 5):
            transition = classifier.classify(self._point(step * 0.1, 12.2))
        assert transition is not None
        self.assertEqual(transition.control_region, ControlRegion.TRANSITION)
        self.assertEqual(transition.operating_state, OperatingState.TRANSITION)

        above = transition
        for step in range(5, 9):
            above = classifier.classify(self._point(step * 0.1, 14.2))
        self.assertEqual(above.control_region, ControlRegion.ABOVE_RATED)
        self.assertEqual(above.operating_state, OperatingState.ABOVE_RATED_REGULATION)

        for step, wind in enumerate((13.1, 13.3, 13.1, 13.3), start=9):
            held = classifier.classify(self._point(step * 0.1, wind))
            self.assertEqual(held.control_region, ControlRegion.ABOVE_RATED)

        returned = held
        for step in range(13, 17):
            returned = classifier.classify(self._point(step * 0.1, 12.8))
        self.assertEqual(returned.control_region, ControlRegion.TRANSITION)

    def test_operating_states_are_causal_and_non_generating_rows_are_not_mppt(self) -> None:
        classifier = OperatingStateClassifier(OperatingStateConfig(
            wind_filter_time_constant_s=0.0,
            region_confirmation_s=0.2,
            running_confirmation_s=0.2,
            stopped_confirmation_s=0.2,
        ))

        startup = classifier.classify(self._point(
            0.0,
            8.0,
            rotor_speed_rad_s=0.2,
            real_power_w=0.0,
        ))
        self.assertEqual(startup.operating_state, OperatingState.STARTUP_CUT_IN)
        self.assertFalse(startup.normal_mppt_eligible)

        running = startup
        for step in range(1, 4):
            running = classifier.classify(self._point(step * 0.1, 8.0))
        self.assertEqual(running.operating_state, OperatingState.NORMAL_GENERATION)
        self.assertTrue(running.normal_mppt_eligible)

        abnormal = classifier.classify(self._point(0.4, 8.0, real_power_w=-2_000.0))
        self.assertEqual(abnormal.operating_state, OperatingState.NEGATIVE_POWER_ABNORMAL)
        self.assertFalse(abnormal.normal_mppt_eligible)

        low_power_pending = classifier.classify(self._point(0.5, 8.0, real_power_w=0.0))
        self.assertFalse(low_power_pending.normal_mppt_eligible)

        stopped = low_power_pending
        for step in range(6, 9):
            stopped = classifier.classify(self._point(
                step * 0.1,
                8.0,
                rotor_speed_rad_s=0.55,
                real_power_w=0.0,
            ))
        self.assertEqual(stopped.operating_state, OperatingState.SHUTDOWN_STOPPED)
        self.assertFalse(stopped.normal_mppt_eligible)

        restarting = classifier.classify(self._point(0.9, 8.0))
        self.assertEqual(restarting.operating_state, OperatingState.STARTUP_CUT_IN)
        self.assertFalse(restarting.normal_mppt_eligible)

    def test_rejects_non_monotonic_or_nonfinite_inputs(self) -> None:
        classifier = OperatingStateClassifier()
        classifier.classify(self._point(0.0, 8.0))

        with self.assertRaisesRegex(ValueError, 'strictly increasing'):
            classifier.classify(self._point(0.0, 8.0))
        with self.assertRaisesRegex(ValueError, 'finite'):
            OperatingStateClassifier().classify(self._point(0.0, float('nan')))


if __name__ == '__main__':
    unittest.main()