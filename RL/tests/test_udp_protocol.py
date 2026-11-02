from __future__ import annotations

import socket
import unittest

import numpy as np

from RL.envs.wind_turbine_env import (
    BLADED_CSV_HEADER,
    QUEUE_UDP_PROTOCOL_VERSION,
    UdpProtocolVersionMismatch,
    WindTurbineEnv,
)


class TestUdpProtocol(unittest.TestCase):
    def setUp(self) -> None:
        self.env = object.__new__(WindTurbineEnv)
        self.env.strict_field_count = True

    def test_parses_frame_id_prefixed_packet(self) -> None:
        fields = ['0'] * len(BLADED_CSV_HEADER)
        fields[BLADED_CSV_HEADER.index('Time')] = '1.25'
        payload, frame_id, sim_time = self.env._parse_payload('42,11.25,' + ','.join(fields))

        self.assertEqual(frame_id, 42)
        self.assertEqual(sim_time, 11.25)
        self.assertEqual(payload['Time'], 1.25)

    def test_parses_legacy_packet_without_frame_id(self) -> None:
        fields = ['0'] * len(BLADED_CSV_HEADER)
        fields[BLADED_CSV_HEADER.index('Time')] = '1.25'
        payload, frame_id, sim_time = self.env._parse_payload(','.join(fields))

        self.assertIsNone(frame_id)
        self.assertEqual(sim_time, 1.25)
        self.assertEqual(payload['Time'], 1.25)

    def test_parses_the_v2_queue_envelope(self) -> None:
        fields = ['0'] * len(BLADED_CSV_HEADER)
        fields[BLADED_CSV_HEADER.index('Time')] = '1.25'
        self.env.udp_protocol_version = QUEUE_UDP_PROTOCOL_VERSION
        (
            payload,
            frame_id,
            sim_time,
            queue_epoch,
            condition_instance_id,
            condition_id,
            phase,
            phase_elapsed_s,
            protocol_version,
        ) = self.env._parse_payload(
            f'{QUEUE_UDP_PROTOCOL_VERSION},7,19,12,TRAIN,3.500000,41,13.500000,' + ','.join(fields),
            include_queue_metadata=True,
        )

        self.assertEqual(payload['Time'], 1.25)
        self.assertEqual(frame_id, 41)
        self.assertEqual(sim_time, 13.5)
        self.assertEqual(queue_epoch, 7)
        self.assertEqual(condition_instance_id, 19)
        self.assertEqual(condition_id, 12)
        self.assertEqual(phase, 'TRAIN')
        self.assertEqual(phase_elapsed_s, 3.5)
        self.assertEqual(protocol_version, QUEUE_UDP_PROTOCOL_VERSION)

    def test_v2_configuration_rejects_a_v1_queue_envelope(self) -> None:
        fields = ['0'] * len(BLADED_CSV_HEADER)
        self.env.udp_protocol_version = QUEUE_UDP_PROTOCOL_VERSION

        with self.assertRaises(UdpProtocolVersionMismatch):
            self.env._parse_payload('7,19,12,TRAIN,3.500000,41,13.500000,' + ','.join(fields))

    def test_uses_new_compact_payload_schema(self) -> None:
        self.assertEqual(len(BLADED_CSV_HEADER), 19)
        self.assertEqual(
            BLADED_CSV_HEADER,
            [
                'Time',
                'I_RotorAzimuth',
                'I_RotorSpeed',
                'I_YawError',
                'I_ElectricalTorque',
                'O_PcsTorqueDemand',
                'I_GeneratorSpeed',
                'I_PitchPosition1',
                'O_PitchPositionDemand1',
                'I_PitchPosition2',
                'I_PitchPosition3',
                'I_NacelleForeAftAcceleration',
                'I_NacelleSideSideAcceleration',
                'I_AverageWindSpeed',
                'Electrical power',
                'I_RealPower',
                'Blade root 1 My (Nm)',
                'Blade root 2 My (Nm)',
                'Blade root 3 My (Nm)',
            ],
        )

    def test_fills_missing_real_power_from_electrical_power(self) -> None:
        fields = ['0'] * len(BLADED_CSV_HEADER)
        fields[BLADED_CSV_HEADER.index('Electrical power')] = '15013806'
        fields[BLADED_CSV_HEADER.index('I_RealPower')] = ''

        payload, _, _ = self.env._parse_payload(','.join(fields))

        self.assertEqual(payload['Electrical power'], 15013806.0)
        self.assertEqual(payload['I_RealPower'], 15013806.0)

    def test_fills_missing_electrical_power_from_real_power(self) -> None:
        fields = ['0'] * len(BLADED_CSV_HEADER)
        fields[BLADED_CSV_HEADER.index('Electrical power')] = 'nan'
        fields[BLADED_CSV_HEADER.index('I_RealPower')] = '14987560'

        payload, _, _ = self.env._parse_payload(','.join(fields))

        self.assertEqual(payload['Electrical power'], 14987560.0)
        self.assertEqual(payload['I_RealPower'], 14987560.0)

    def test_formal_handshake_ignores_stale_frame(self) -> None:
        receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            receiver.bind(('127.0.0.1', 0))
            receiver.settimeout(0.02)
            env = object.__new__(WindTurbineEnv)
            env.strict_field_count = True
            env.require_frame_id = True
            env.recv_buffer_size = 8192
            env._socket = receiver
            env._remote_addr = None

            fields = ['0'] * len(BLADED_CSV_HEADER)
            fields[BLADED_CSV_HEADER.index('Time')] = '1.25'
            wire_fields = ','.join(fields)
            target = receiver.getsockname()
            sender.sendto(f'7,11.25,{wire_fields}'.encode('ascii'), target)
            sender.sendto(f'8,11.27,{wire_fields}'.encode('ascii'), target)

            packet = env._recv_payload(stage='test', max_wait_s=0.5, after_frame_id=7)

            self.assertEqual(packet.frame_id, 8)
            self.assertEqual(packet.sim_time, 11.27)
            self.assertGreaterEqual(packet.dropped_stale_frames, 1)
        finally:
            sender.close()
            receiver.close()

    def test_formal_handshake_echoes_action_frame_id(self) -> None:
        receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            receiver.bind(('127.0.0.1', 0))
            receiver.settimeout(0.5)
            env = object.__new__(WindTurbineEnv)
            env._socket = sender
            env._remote_addr = receiver.getsockname()
            env._last_sent_frame_id = None

            env._send_action(np.asarray([20_000_000.0, 0.05], dtype=np.float32), frame_id=8)
            reply, _ = receiver.recvfrom(256)

            self.assertEqual(reply.decode('ascii'), '8,20000000.000000,0.050000')
            self.assertEqual(env._last_sent_frame_id, 8)
        finally:
            sender.close()
            receiver.close()


if __name__ == '__main__':
    unittest.main()