from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

from RL.envs.wind_turbine_env import BLADED_CSV_HEADER


def _load_connectivity_module():
    path = Path(__file__).resolve().parents[2] / 'bladed_env' / 'udp_python-DLC1.3-new.py'
    spec = importlib.util.spec_from_file_location('udp_connectivity_queue_protocol', path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Unable to load {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestUdpConnectivityQueueProtocol(unittest.TestCase):
    def test_v7_queue_frame_round_trips_matching_hold_action(self) -> None:
        module = _load_connectivity_module()
        row = ['0'] * len(BLADED_CSV_HEADER)
        row[BLADED_CSV_HEADER.index('O_PcsTorqueDemand')] = '20000000'
        row[BLADED_CSV_HEADER.index('O_PitchPositionDemand1')] = '0.05'
        packet = 'bladed_udp_19_si_my_queue_v2,7,19,12,TRAIN,3.500000,41,13.500000,' + ','.join(row)

        payload, frame_id, sim_time, queue_envelope = module.parse_frame(
            packet,
            require_frame_id=True,
            require_condition_metadata=True,
        )
        reply = module.build_hold_reply(payload, frame_id, queue_envelope).decode('ascii')

        self.assertEqual(frame_id, 41)
        self.assertEqual(sim_time, 13.5)
        self.assertEqual(queue_envelope.protocol_version, 'bladed_udp_19_si_my_queue_v2')
        self.assertEqual(queue_envelope.queue_epoch, 7)
        self.assertEqual(queue_envelope.condition_instance_id, 19)
        self.assertEqual(queue_envelope.condition_id, 12)
        self.assertEqual(queue_envelope.phase, 'TRAIN')
        self.assertEqual(reply, '7,19,41,20000000.000000,0.050000')

    def test_packet_prefix_diagnostic_uses_the_wire_field_count(self) -> None:
        module = _load_connectivity_module()
        legacy_packet = '41,13.500000,' + ','.join(['0'] * len(BLADED_CSV_HEADER))

        fields = [value.strip() for value in legacy_packet.split(',')]

        self.assertEqual(len(fields), len(BLADED_CSV_HEADER) + 2)
        self.assertEqual(fields[:2], ['41', '13.500000'])


if __name__ == '__main__':
    unittest.main()