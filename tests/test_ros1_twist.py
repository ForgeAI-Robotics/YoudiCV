import socket
import struct
import time

import pytest

from piper_handeye.ros1_twist import Ros1TwistPublisher, _encode_header, _read_exact


def _decode_header(sock):
    size = struct.unpack("<I", _read_exact(sock, 4))[0]
    payload = _read_exact(sock, size)
    fields = {}
    offset = 0
    while offset < len(payload):
        length = struct.unpack_from("<I", payload, offset)[0]
        offset += 4
        key, value = payload[offset:offset + length].decode().split("=", 1)
        fields[key] = value
        offset += length
    return fields


def test_tcpros_twist_serialization_and_handshake():
    publisher = Ros1TwistPublisher(
        master_uri="http://127.0.0.1:9",
        bind_ip="127.0.0.1",
        register_with_master=False,
    )
    client = socket.create_connection(("127.0.0.1", publisher.tcp_port), timeout=1)
    client.sendall(_encode_header({
        "callerid": "/test_subscriber",
        "topic": "/cmd_vel",
        "type": "geometry_msgs/Twist",
        "md5sum": "9f195f881246fdfa2798d1d3eebca84a",
        "tcp_nodelay": "1",
    }))
    response = _decode_header(client)
    assert response["type"] == "geometry_msgs/Twist"
    assert response["topic"] == "/cmd_vel"
    deadline = time.monotonic() + 1.0
    while publisher.subscriber_count != 1 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert publisher.subscriber_names == ("/test_subscriber",)
    publisher.publish(0.125, -0.4)
    size = struct.unpack("<I", _read_exact(client, 4))[0]
    values = struct.unpack("<6d", _read_exact(client, size))
    assert values == pytest.approx((0.125, 0, 0, 0, 0, -0.4))
    client.close()
    publisher.close(unregister=False)


def test_request_topic_rejects_wrong_topic_and_transport():
    publisher = Ros1TwistPublisher(
        master_uri="http://127.0.0.1:9",
        bind_ip="127.0.0.1",
        register_with_master=False,
    )
    assert publisher.requestTopic("/x", "/bad", [["TCPROS"]])[0] == -1
    assert publisher.requestTopic("/x", "/cmd_vel", [["UDPROS"]])[0] == -1
    publisher.close(unregister=False)
