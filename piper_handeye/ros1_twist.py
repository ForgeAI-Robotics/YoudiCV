"""Small dependency-free ROS 1 TCPROS publisher for geometry_msgs/Twist."""

from __future__ import annotations

import os
import socket
import struct
import threading
import xmlrpc.client
from xmlrpc.server import SimpleXMLRPCServer


TWIST_TYPE = "geometry_msgs/Twist"
TWIST_MD5 = "9f195f881246fdfa2798d1d3eebca84a"
TWIST_DEFINITION = """geometry_msgs/Vector3 linear
geometry_msgs/Vector3 angular

================================================================================
MSG: geometry_msgs/Vector3
float64 x
float64 y
float64 z
"""


class _TimeoutTransport(xmlrpc.client.Transport):
    def __init__(self, timeout_sec: float = 2.0) -> None:
        super().__init__()
        self.timeout_sec = float(timeout_sec)

    def make_connection(self, host: object) -> object:
        connection = super().make_connection(host)
        connection.timeout = self.timeout_sec
        return connection


def _server_proxy(uri: str) -> xmlrpc.client.ServerProxy:
    return xmlrpc.client.ServerProxy(
        uri, allow_none=True, transport=_TimeoutTransport()
    )


def _read_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("TCPROS peer closed during handshake")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _encode_header(fields: dict[str, str]) -> bytes:
    payload = b"".join(
        struct.pack("<I", len(item)) + item
        for item in (f"{key}={value}".encode() for key, value in fields.items())
    )
    return struct.pack("<I", len(payload)) + payload


def _decode_header(payload: bytes) -> dict[str, str]:
    fields: dict[str, str] = {}
    offset = 0
    while offset < len(payload):
        if offset + 4 > len(payload):
            raise ValueError("truncated TCPROS header field")
        size = struct.unpack_from("<I", payload, offset)[0]
        offset += 4
        item = payload[offset:offset + size]
        if len(item) != size:
            raise ValueError("truncated TCPROS header value")
        offset += size
        key, separator, value = item.decode(errors="replace").partition("=")
        if not separator:
            raise ValueError("invalid TCPROS header field")
        fields[key] = value
    return fields


class Ros1TwistPublisher:
    """Publish Twist without rospy while implementing the ROS 1 slave API."""

    def __init__(
        self,
        *,
        master_uri: str,
        bind_ip: str,
        topic: str = "/cmd_vel",
        caller_id: str | None = None,
        register_with_master: bool = True,
    ) -> None:
        self.master_uri = master_uri.rstrip("/") + "/"
        self.bind_ip = bind_ip
        self.topic = topic
        self.caller_id = caller_id or f"/dualsense_chassis_gui_{os.getpid()}"
        self._closed = False
        self._clients: list[socket.socket] = []
        self._client_callers: dict[socket.socket, str] = {}
        self._clients_lock = threading.Lock()

        self._tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._tcp.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._tcp.bind((bind_ip, 0))
        self._tcp.listen(4)
        self._tcp.settimeout(0.2)
        self.tcp_port = int(self._tcp.getsockname()[1])

        self._rpc = SimpleXMLRPCServer(
            (bind_ip, 0), allow_none=True, logRequests=False
        )
        self.rpc_port = int(self._rpc.server_address[1])
        self.rpc_uri = f"http://{bind_ip}:{self.rpc_port}/"
        self._rpc.register_function(self.getPid, "getPid")
        self._rpc.register_function(self.getPublications, "getPublications")
        self._rpc.register_function(self.getSubscriptions, "getSubscriptions")
        self._rpc.register_function(self.requestTopic, "requestTopic")
        self._rpc.register_function(self.publisherUpdate, "publisherUpdate")
        self._rpc.register_function(self.getBusInfo, "getBusInfo")

        self._rpc_thread = threading.Thread(
            target=self._rpc.serve_forever, name="dualsense-ros1-rpc", daemon=True
        )
        self._accept_thread = threading.Thread(
            target=self._accept_loop, name="dualsense-tcpros", daemon=True
        )
        self._rpc_thread.start()
        self._accept_thread.start()
        if register_with_master:
            self._register()

    def _register(self) -> None:
        master = _server_proxy(self.master_uri)
        code, message, _subscribers = master.registerPublisher(
            self.caller_id, self.topic, TWIST_TYPE, self.rpc_uri
        )
        if int(code) != 1:
            self.close(unregister=False)
            raise ConnectionError(f"ROS master rejected publisher: {message}")

    def getPid(self, _caller_id: str) -> list[object]:
        return [1, "", os.getpid()]

    def getPublications(self, _caller_id: str) -> list[object]:
        return [1, "", [[self.topic, TWIST_TYPE]]]

    def getSubscriptions(self, _caller_id: str) -> list[object]:
        return [1, "", []]

    def publisherUpdate(
        self, _caller_id: str, _topic: str, _publishers: list[str]
    ) -> list[object]:
        return [1, "", 0]

    def getBusInfo(self, _caller_id: str) -> list[object]:
        with self._clients_lock:
            callers = [self._client_callers.get(item, "unknown") for item in self._clients]
        info = [
            [index, caller, "o", "TCPROS", self.topic, True, ""]
            for index, caller in enumerate(callers)
        ]
        return [1, "", info]

    def requestTopic(
        self, _caller_id: str, topic: str, protocols: list[list[object]]
    ) -> list[object]:
        if topic != self.topic:
            return [-1, f"unknown topic {topic}", []]
        if not any(protocol and protocol[0] == "TCPROS" for protocol in protocols):
            return [-1, "TCPROS required", []]
        return [1, "", ["TCPROS", self.bind_ip, self.tcp_port]]

    def _accept_loop(self) -> None:
        while not self._closed:
            try:
                client, _address = self._tcp.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            threading.Thread(
                target=self._handshake_client,
                args=(client,),
                name="dualsense-tcpros-client",
                daemon=True,
            ).start()

    def _handshake_client(self, client: socket.socket) -> None:
        try:
            client.settimeout(2.0)
            header_size = struct.unpack("<I", _read_exact(client, 4))[0]
            if header_size > 64 * 1024:
                raise ValueError("unreasonable TCPROS header")
            request = _decode_header(_read_exact(client, header_size))
            client.sendall(_encode_header({
                "callerid": self.caller_id,
                "type": TWIST_TYPE,
                "md5sum": TWIST_MD5,
                "message_definition": TWIST_DEFINITION,
                "latching": "0",
                "topic": self.topic,
            }))
            client.settimeout(None)
            with self._clients_lock:
                if self._closed:
                    client.close()
                else:
                    self._clients.append(client)
                    self._client_callers[client] = request.get("callerid", "unknown")
        except (OSError, ConnectionError, ValueError, struct.error):
            client.close()

    @property
    def subscriber_count(self) -> int:
        with self._clients_lock:
            return len(self._clients)

    @property
    def subscriber_names(self) -> tuple[str, ...]:
        with self._clients_lock:
            return tuple(
                self._client_callers.get(item, "unknown") for item in self._clients
            )

    def refresh_subscriber_node(self, subscriber_node: str) -> tuple[str, ...]:
        """Force a legacy subscriber to notice publishers started at runtime."""
        master = _server_proxy(self.master_uri)
        code, message, state = master.getSystemState(self.caller_id)
        if int(code) != 1:
            raise ConnectionError(message)
        publisher_nodes = next(
            (nodes for topic, nodes in state[0] if topic == self.topic), []
        )
        publisher_uris: list[str] = []
        for node_name in publisher_nodes:
            result, _detail, uri = master.lookupNode(self.caller_id, node_name)
            if int(result) == 1 and uri not in publisher_uris:
                publisher_uris.append(str(uri))
        if self.rpc_uri not in publisher_uris:
            publisher_uris.append(self.rpc_uri)
        result, detail, subscriber_uri = master.lookupNode(
            self.caller_id, subscriber_node
        )
        if int(result) != 1:
            raise ConnectionError(detail)
        subscriber = _server_proxy(str(subscriber_uri))
        result, detail, _ = subscriber.publisherUpdate(
            self.caller_id, self.topic, publisher_uris
        )
        if int(result) != 1:
            raise ConnectionError(detail)
        return tuple(publisher_uris)

    def publish(self, linear_x: float, angular_z: float) -> None:
        if self._closed:
            raise RuntimeError("publisher is closed")
        payload = struct.pack(
            "<6d", float(linear_x), 0.0, 0.0, 0.0, 0.0, float(angular_z)
        )
        frame = struct.pack("<I", len(payload)) + payload
        failed: list[socket.socket] = []
        with self._clients_lock:
            for client in self._clients:
                try:
                    client.sendall(frame)
                except OSError:
                    failed.append(client)
            for client in failed:
                self._clients.remove(client)
                self._client_callers.pop(client, None)
                client.close()

    def close(self, *, unregister: bool = True) -> None:
        if self._closed:
            return
        if unregister:
            try:
                master = _server_proxy(self.master_uri)
                master.unregisterPublisher(self.caller_id, self.topic, self.rpc_uri)
            except (OSError, xmlrpc.client.Error):
                pass
        self._closed = True
        self._rpc.shutdown()
        self._rpc.server_close()
        self._tcp.close()
        with self._clients_lock:
            clients, self._clients = self._clients, []
            self._client_callers.clear()
        for client in clients:
            try:
                client.close()
            except OSError:
                pass

    def __enter__(self) -> "Ros1TwistPublisher":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
