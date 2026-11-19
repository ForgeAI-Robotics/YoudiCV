"""Protocol helpers and a passive-by-default Android AGX UDP listener."""

from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict
import datetime as dt
import json
from pathlib import Path
import queue
import socket
import struct
import threading


HEADER = struct.Struct("<HHIII")
MAGIC = 16
MESSAGE_TYPE = 1
NEUTRAL_ACK = b"null"


@dataclass(frozen=True)
class AndroidPacket:
    source: tuple[str, int]
    sequence: int
    topic: str
    data: object
    raw_data: str


class AcceptedPacketCache:
    """Bounded replay cache for commands that have already started a task.

    Android retries are transport behavior.  A repeated, already accepted
    packet must receive the same ACK without starting the physical action a
    second time.  Packets that were only observed or staged are deliberately
    not entered in this cache.
    """

    def __init__(self, max_entries: int = 256):
        if max_entries < 1:
            raise ValueError("accepted packet cache size must be positive")
        self.max_entries = int(max_entries)
        self._entries: OrderedDict[tuple[str, int, str], str] = OrderedDict()

    @staticmethod
    def key(packet: AndroidPacket) -> tuple[str, int, str]:
        return packet.source[0], int(packet.sequence), str(packet.topic)

    def classify(self, packet: AndroidPacket) -> str:
        """Return ``new``, ``replay``, or ``conflict`` for one packet."""
        key = self.key(packet)
        if key not in self._entries:
            return "new"
        self._entries.move_to_end(key)
        return "replay" if self._entries[key] == packet.raw_data else "conflict"

    def contains(self, packet: AndroidPacket) -> bool:
        return self.classify(packet) == "replay"

    def mark_accepted(self, packet: AndroidPacket) -> None:
        key = self.key(packet)
        self._entries[key] = packet.raw_data
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)


def encode_packet(sequence: int, topic: str, data: object) -> bytes:
    topic_bytes = topic.encode("utf-8")
    if isinstance(data, str):
        data_text = data
    else:
        data_text = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    data_bytes = data_text.encode("utf-8")
    return (
        HEADER.pack(MAGIC, MESSAGE_TYPE, int(sequence), len(topic_bytes), len(data_bytes))
        + topic_bytes
        + data_bytes
    )


def decode_packet(payload: bytes, source: tuple[str, int]) -> AndroidPacket:
    if len(payload) < HEADER.size:
        raise ValueError(f"数据包短于 {HEADER.size} 字节协议头")
    magic, message_type, sequence, topic_length, data_length = HEADER.unpack_from(payload)
    expected = HEADER.size + topic_length + data_length
    if magic != MAGIC or message_type != MESSAGE_TYPE:
        raise ValueError(f"协议头无效: magic={magic}, type={message_type}")
    if expected != len(payload):
        raise ValueError(f"长度无效: 声明 {expected} 字节，实际 {len(payload)} 字节")
    topic_end = HEADER.size + topic_length
    topic = payload[HEADER.size:topic_end].decode("utf-8")
    raw_data = payload[topic_end:expected].decode("utf-8")
    try:
        data: object = json.loads(raw_data)
    except json.JSONDecodeError:
        data = raw_data
    return AndroidPacket(source, sequence, topic, data, raw_data)


class AndroidUdpListener:
    """Receive Android commands; transmission only occurs via explicit reply()."""

    def __init__(
        self,
        events: queue.Queue,
        *,
        bind_ip: str = "192.168.168.162",
        port: int = 5005,
        allowed_source_ip: str = "192.168.168.110",
        journal: Path | None = None,
    ):
        self.events = events
        self.bind_ip = bind_ip
        self.port = int(port)
        self.allowed_source_ip = allowed_source_ip
        self.journal = journal
        self.sock: socket.socket | None = None
        self.stop_requested = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_requested.set()
        sock = self.sock
        if sock is not None:
            sock.close()

    def reply(self, source: tuple[str, int], payload: bytes = b"accepted") -> None:
        """Return only a transport ACK. This never publishes a robot command."""
        sock = self.sock
        if sock is None:
            raise RuntimeError("Android UDP 监听器尚未就绪")
        if source[0] != self.allowed_source_ip:
            raise ValueError("拒绝向非 Android 地址回包")
        if not payload:
            raise ValueError("ACK 不得为空")
        sock.sendto(payload, source)

    def stop_repeating(self, source: tuple[str, int]) -> None:
        """Stop the APK retry timer without reporting task acceptance."""
        self.reply(source, NEUTRAL_ACK)

    def _record(self, event: dict) -> None:
        if self.journal is not None:
            self.journal.parent.mkdir(parents=True, exist_ok=True)
            journal_event = {
                key: value for key, value in event.items() if key != "packet"
            }
            with self.journal.open("a", encoding="utf-8") as output:
                output.write(json.dumps(
                    journal_event, ensure_ascii=False, separators=(",", ":")
                ) + "\n")
        self.events.put(event)

    def _run(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock = sock
        try:
            sock.bind((self.bind_ip, self.port))
            sock.settimeout(0.5)
            self._record({
                "event": "listener_ready",
                "bind": f"{self.bind_ip}:{self.port}",
                "allowed_source_ip": self.allowed_source_ip,
            })
            while not self.stop_requested.is_set():
                try:
                    payload, source = sock.recvfrom(65535)
                except socket.timeout:
                    continue
                except OSError:
                    if self.stop_requested.is_set():
                        break
                    raise
                timestamp = dt.datetime.now(dt.timezone.utc).astimezone().isoformat()
                if source[0] != self.allowed_source_ip:
                    self._record({
                        "event": "packet_rejected",
                        "timestamp": timestamp,
                        "source": list(source),
                        "reason": "unexpected_source_ip",
                    })
                    continue
                try:
                    packet = decode_packet(payload, source)
                    self._record({
                        "event": "android_packet",
                        "timestamp": timestamp,
                        "source": list(source),
                        "sequence": packet.sequence,
                        "topic": packet.topic,
                        "data": packet.data,
                        "packet": packet,
                    })
                except Exception as exc:
                    self._record({
                        "event": "packet_rejected",
                        "timestamp": timestamp,
                        "source": list(source),
                        "reason": f"{type(exc).__name__}: {exc}",
                    })
        except Exception as exc:
            self._record({
                "event": "listener_error",
                "error": f"{type(exc).__name__}: {exc}",
            })
        finally:
            try:
                sock.close()
            except OSError:
                pass
            self.sock = None
