import json
import queue
import socket
import time

from piper_handeye.android_agx_protocol import (
    AcceptedPacketCache,
    AndroidUdpListener,
    NEUTRAL_ACK,
    decode_packet,
    encode_packet,
)


def test_android_packet_round_trip():
    payload = encode_packet(17, "bt_control", {"command": 10, "value": "8"})
    packet = decode_packet(payload, ("192.168.168.110", 5005))
    assert packet.sequence == 17
    assert packet.topic == "bt_control"
    assert packet.data == {"command": 10, "value": "8"}


def test_listener_never_replies_until_reply_is_explicit(tmp_path):
    events = queue.Queue()
    listener = AndroidUdpListener(
        events,
        bind_ip="127.0.0.1",
        port=0,
        allowed_source_ip="127.0.0.1",
        journal=tmp_path / "packets.jsonl",
    )

    # Bind synchronously so the test can discover the kernel-selected port,
    # while retaining the production listener's explicit-reply behavior.
    listener.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    listener.sock.bind(("127.0.0.1", 0))
    listener.sock.settimeout(0.15)
    target = listener.sock.getsockname()
    client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    client.bind(("127.0.0.1", 0))
    client.settimeout(0.15)
    payload = encode_packet(1, "bt_control", {"command": 10, "value": "1"})
    client.sendto(payload, target)
    raw, source = listener.sock.recvfrom(4096)
    packet = decode_packet(raw, source)
    try:
        client.recvfrom(128)
        assert False, "listener emitted an implicit ACK"
    except socket.timeout:
        pass

    listener.reply(packet.source)
    reply, _ = client.recvfrom(128)
    assert reply == b"accepted"
    client.close()
    listener.sock.close()
    listener.sock = None


def test_unknown_topic_is_still_parseable_without_action_semantics():
    payload = encode_packet(928001, "codex_probe_no_action", {})
    packet = decode_packet(payload, ("192.168.168.110", 5002))
    assert packet.topic == "codex_probe_no_action"
    assert packet.data == {}


def test_neutral_ack_is_nonempty_but_has_no_protocol_command():
    assert NEUTRAL_ACK == b"null"
    assert NEUTRAL_ACK
    assert b"command" not in NEUTRAL_ACK


def test_accepted_packet_cache_is_idempotent_and_bounded():
    cache = AcceptedPacketCache(max_entries=2)
    first = decode_packet(
        encode_packet(17, "bt_control", {"command": 10, "value": "8"}),
        ("192.168.168.110", 5005),
    )
    same_retry_port_changed = decode_packet(
        encode_packet(17, "bt_control", {"command": 10, "value": "8"}),
        ("192.168.168.110", 5100),
    )
    second = decode_packet(
        encode_packet(18, "bt_control", {"command": 10, "value": "8"}),
        ("192.168.168.110", 5005),
    )
    third = decode_packet(
        encode_packet(19, "bt_control", {"command": 10, "value": "9"}),
        ("192.168.168.110", 5005),
    )
    assert not cache.contains(first)
    cache.mark_accepted(first)
    assert cache.contains(same_retry_port_changed)
    conflicting = decode_packet(
        encode_packet(17, "bt_control", {"command": 10, "value": "16"}),
        ("192.168.168.110", 5005),
    )
    assert cache.classify(conflicting) == "conflict"
    cache.mark_accepted(second)
    cache.mark_accepted(third)
    assert not cache.contains(first)
    assert cache.contains(second)
    assert cache.contains(third)
