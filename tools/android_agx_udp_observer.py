#!/usr/bin/env python3
"""Passive observer for the Android RobotArm AGX UDP protocol.

This process intentionally has no transmit path.  It only binds the AGX command
port, parses packets, and appends them to a JSONL journal.  In particular it
does not acknowledge commands, publish ROS messages, or invoke robot programs.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import signal
import socket
import struct
from pathlib import Path


HEADER = struct.Struct("<HHIII")
EXPECTED_MAGIC = 16
EXPECTED_MESSAGE_TYPE = 1


def parse_packet(payload: bytes) -> dict[str, object]:
    if len(payload) < HEADER.size:
        raise ValueError(f"packet shorter than {HEADER.size}-byte header")

    magic, message_type, sequence, topic_length, data_length = HEADER.unpack_from(payload)
    expected_length = HEADER.size + topic_length + data_length
    if expected_length != len(payload):
        raise ValueError(
            f"length mismatch: header says {expected_length}, received {len(payload)}"
        )
    if magic != EXPECTED_MAGIC:
        raise ValueError(f"unexpected magic {magic}")
    if message_type != EXPECTED_MESSAGE_TYPE:
        raise ValueError(f"unexpected message type {message_type}")

    topic_end = HEADER.size + topic_length
    topic = payload[HEADER.size:topic_end].decode("utf-8")
    data_text = payload[topic_end:expected_length].decode("utf-8")
    try:
        data: object = json.loads(data_text)
    except json.JSONDecodeError:
        data = data_text

    return {
        "magic": magic,
        "message_type": message_type,
        "sequence": sequence,
        "topic": topic,
        "data": data,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Passively journal Android AGX UDP packets")
    parser.add_argument("--bind-ip", default="192.168.168.162")
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument(
        "--journal",
        type=Path,
        default=Path("/home/qianjue/contral_test/android_agx_udp_observer.jsonl"),
    )
    args = parser.parse_args()

    running = True

    def stop(_signum: int, _frame: object) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.bind_ip, args.port))
    sock.settimeout(0.5)
    args.journal.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"PASSIVE_ONLY listening on udp://{args.bind_ip}:{args.port}; "
        "no ACK, ROS, chassis, or arm output",
        flush=True,
    )
    with args.journal.open("a", encoding="utf-8", buffering=1) as journal:
        while running:
            try:
                payload, source = sock.recvfrom(65535)
            except socket.timeout:
                continue

            event: dict[str, object] = {
                "timestamp": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(),
                "source_ip": source[0],
                "source_port": source[1],
                "received_bytes": len(payload),
                "mode": "passive_no_reply",
            }
            try:
                event.update(parse_packet(payload))
                event["valid"] = True
            except (UnicodeDecodeError, ValueError) as exc:
                event.update(valid=False, error=str(exc), payload_hex=payload.hex())

            line = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
            journal.write(line + "\n")
            print(line, flush=True)

    sock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
