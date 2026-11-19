"""Regression coverage for braces in the Piper UDP envelope, without motion."""

import json
import struct

import numpy as np
import pytest

from piper_handeye.robot import PiperUdpRobot, extract_json


def frame(value, *, sequence=1, topic=b"system_state", prefix=b"1601"):
    body = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
    return prefix + struct.pack("<III", sequence, len(topic), len(body)) + topic + body


@pytest.mark.parametrize("prefix", [b"1601", b"1602", b"1603", b"\x16\x01", b"\x16\x02", b"\x16\x03"])
def test_entire_failed_sequence_interval_and_boundaries(prefix):
    value = {"hardware_ok": 1, "message": "机械臂状态正常"}
    for sequence in range(0x457AFF, 0x457C01):
        assert extract_json(frame(value, sequence=sequence, prefix=prefix)) == value


@pytest.mark.parametrize("byte_index", range(4))
@pytest.mark.parametrize("mode", ["ascii", "binary"])
def test_every_sequence_byte_value_round_trips_production_encoder(byte_index, mode):
    robot = PiperUdpRobot({"controller_ip": "127.0.0.1", "local_ip": "127.0.0.1", "header_mode": mode})
    value = {"tree_name": "move_joint", "task_id": "123", "result": 1}
    for byte in range(256):
        robot._seq = (byte << (8 * byte_index)) - 1
        assert extract_json(robot._packet(value)) == value


@pytest.mark.parametrize("prefix", [b"1601", b"\x16\x01"])
def test_topic_length_byte_can_be_opening_brace(prefix):
    value = {"result": 1}
    assert extract_json(frame(value, topic=b"x" * 123, prefix=prefix)) == value


@pytest.mark.parametrize("prefix", [b"1601", b"\x16\x01"])
def test_data_length_byte_can_be_opening_brace(prefix):
    value = {"padding": "x" * 109}
    assert len(json.dumps(value, separators=(",", ":")).encode()) == 123
    assert extract_json(frame(value, prefix=prefix)) == value


@pytest.mark.parametrize("prefix", [b"1601", b"\x16\x01"])
def test_braces_in_topic_do_not_select_json_start(prefix):
    assert extract_json(frame({"result": 1}, topic=b"{not_json}", prefix=prefix)) == {"result": 1}


@pytest.mark.parametrize("prefix", [b"1601", b"\x16\x01"])
def test_bad_lengths_and_truncation_never_fall_back_to_scanning(prefix):
    value = {"hardware_ok": 1}
    packet = frame(value, prefix=prefix)
    offset = len(prefix)
    _, topic_size, data_size = struct.unpack_from("<III", packet, offset)
    for topic_len, data_len in [(0, data_size), (topic_size, 0), (topic_size + 1, data_size),
                                (topic_size, data_size - 1), (0xFFFFFFFF, data_size),
                                (topic_size, 0xFFFFFFFF)]:
        altered = bytearray(packet)
        struct.pack_into("<II", altered, offset + 4, topic_len, data_len)
        assert extract_json(bytes(altered)) is None
    for size in range(len(packet)):
        assert extract_json(packet[:size]) is None
    assert extract_json(packet + b"trailing") is None


@pytest.mark.parametrize("prefix", [b"1600", b"1604", b"\x16\x00", b"\x16\x04"])
def test_unknown_message_type_is_rejected(prefix):
    assert extract_json(frame({"result": 1}, prefix=prefix)) is None


@pytest.mark.parametrize("prefix", [b"", b"system_state", b"bt_result", b"bt_log"])
def test_existing_unframed_json_is_supported(prefix):
    value = {"message": "原位正常，文本中有 { 和 }"}
    assert extract_json(prefix + json.dumps(value, ensure_ascii=False).encode()) == value


@pytest.mark.parametrize("value", [[], 1, True, None, "text"])
def test_non_object_payload_is_rejected(value):
    assert extract_json(frame(value)) is None


def test_invalid_utf8_and_json_are_rejected():
    for body in [b"{\xff}", b"{bad}", b'{"result":1} garbage']:
        packet = b"1601" + struct.pack("<III", 0x457B00, 12, len(body)) + b"system_state" + body
        assert extract_json(packet) is None


class PacketSocket:
    def __init__(self, *packets):
        self.packets = iter(packets)

    def recvfrom(self, _size):
        try:
            return next(self.packets), ("127.0.0.1", 5004)
        except StopIteration:
            raise OSError("offline packet stream complete")


def make_offline_robot():
    robot = PiperUdpRobot({"controller_ip": "127.0.0.1", "local_ip": "127.0.0.1", "state_timeout_sec": 0.01})
    robot._running = True
    return robot


def test_state_receiver_accepts_failed_sequence_and_recovers_from_bad_packet():
    joints = [0.0, 0.1, -0.1, 0.0, 0.06, 0.09]
    value = {
        "hardware_ok": 1,
        "joints_angle": [{"key": f"piper_joint{i}", "value": v} for i, v in enumerate(joints, 1)],
        "pose": {"position": {"x": 0.2, "y": 0.0, "z": 0.18},
                 "orientation": {"x": 0, "y": 0, "z": 0, "w": 1}},
    }
    robot = make_offline_robot()
    robot._state_socket = PacketSocket(frame(value)[:-1], frame(value, sequence=0x457B80))
    robot._state_loop()
    sample = robot.state(max_age=0.35)
    np.testing.assert_array_equal(sample.joints, joints)
    evidence = robot._pose_timing.snapshot()
    assert evidence['count'] == 2
    assert evidence['first'][0]['accepted'] is False
    received = evidence['first'][1]
    assert received['accepted'] is True
    assert received['sample_monotonic'] == sample.timestamp
    assert received['received_monotonic'] <= sample.timestamp <= received['ready_monotonic']
    assert sample.raw == value


def test_task_result_receiver_accepts_brace_in_high_sequence_byte():
    robot = make_offline_robot()
    value = {"tree_name": "move_joint", "task_id": "123", "result": 1}
    robot._result_socket = PacketSocket(frame(value, sequence=0x7B000000, topic=b"bt_result"))
    robot._result_loop()
    assert robot._results[("move_joint", "123")] == 1
