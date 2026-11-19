#!/usr/bin/env python3
"""Read one sample from selected ROS 1 chassis topics without rospy."""

from __future__ import annotations

import argparse
import json
import socket
import struct
import time
import xmlrpc.client


MASTER = "http://192.168.168.100:11311"


def read_exact(sock: socket.socket, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise ConnectionError("peer closed")
        data.extend(chunk)
    return bytes(data)


def header(fields: dict[str, str]) -> bytes:
    payload = b"".join(
        struct.pack("<I", len(item)) + item
        for item in (f"{key}={value}".encode() for key, value in fields.items())
    )
    return struct.pack("<I", len(payload)) + payload


def publisher_for(master: xmlrpc.client.ServerProxy, topic: str) -> str:
    code, _message, state = master.getSystemState("/codex_chassis_probe")
    for candidate, nodes in state[0]:
        if candidate == topic and nodes:
            return str(nodes[0])
    raise RuntimeError(f"no publisher for {topic}")


def one(topic: str, message_type: str, timeout: float) -> bytes:
    master = xmlrpc.client.ServerProxy(MASTER)
    node_name = publisher_for(master, topic)
    code, message, node_uri = master.lookupNode("/codex_chassis_probe", node_name)
    if int(code) != 1:
        raise RuntimeError(message)
    node = xmlrpc.client.ServerProxy(node_uri)
    code, message, protocol = node.requestTopic(
        "/codex_chassis_probe", topic, [["TCPROS"]]
    )
    if int(code) != 1 or protocol[0] != "TCPROS":
        raise RuntimeError(message)
    sock = socket.create_connection((protocol[1], int(protocol[2])), timeout)
    sock.settimeout(timeout)
    try:
        sock.sendall(header({
            "callerid": "/codex_chassis_probe",
            "topic": topic,
            "type": message_type,
            "md5sum": "*",
            "tcp_nodelay": "1",
        }))
        response_size = struct.unpack("<I", read_exact(sock, 4))[0]
        read_exact(sock, response_size)
        size = struct.unpack("<I", read_exact(sock, 4))[0]
        return read_exact(sock, size)
    finally:
        sock.close()


def decode(message_type: str, payload: bytes) -> object:
    if message_type == "std_msgs/Bool":
        return bool(payload[0])
    if message_type == "std_msgs/Int8":
        return struct.unpack_from("<b", payload)[0]
    if message_type == "std_msgs/Int64":
        return struct.unpack_from("<q", payload)[0]
    if message_type == "std_msgs/Float64":
        return struct.unpack_from("<d", payload)[0]
    if message_type == "std_msgs/String":
        size = struct.unpack_from("<I", payload)[0]
        return payload[4:4 + size].decode(errors="replace")
    if message_type == "geometry_msgs/TwistStamped":
        seq, sec, nsec, size = struct.unpack_from("<4I", payload)
        offset = 16 + size
        values = struct.unpack_from("<6d", payload, offset)
        return {
            "seq": seq,
            "stamp": sec + nsec / 1e9,
            "frame_id": payload[16:16 + size].decode(errors="replace"),
            "linear_x": values[0],
            "angular_z": values[5],
        }
    return {"size": len(payload), "hex": payload[:64].hex()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=float, default=2.0)
    args = parser.parse_args()
    topics = {
        "/chassis/e_stop": "std_msgs/Bool",
        "/emergency": "std_msgs/Bool",
        "/chassis_control_state": "std_msgs/Int8",
        "/chassis_work_mode": "std_msgs/Int64",
        "/chassis_reply": "std_msgs/String",
        "/battery": "std_msgs/Float64",
        "/plt_cmd_vel": "geometry_msgs/TwistStamped",
    }
    result = {"timestamp": time.time(), "topics": {}}
    for topic, message_type in topics.items():
        try:
            result["topics"][topic] = decode(
                message_type, one(topic, message_type, args.timeout)
            )
        except Exception as exc:
            result["topics"][topic] = {
                "error": f"{type(exc).__name__}: {exc}"
            }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
