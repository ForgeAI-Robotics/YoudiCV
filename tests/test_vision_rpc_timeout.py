import json
from unittest.mock import MagicMock, patch

import pytest

from piper_handeye.vision_rpc import VisionRpcClient


@pytest.mark.parametrize("command,expected", [
    ("safety_capture_matrix_reference", 8.0), ("safety", .2),
    ("safety_camera_pose", .2), ("visible_buttons", .2),
])
def test_only_reference_build_has_longer_read_deadline(command, expected):
    sock = MagicMock()
    sock.makefile.return_value.__enter__.return_value = sock.makefile.return_value
    sock.makefile.return_value.readline.return_value = b'{"ok":true}\n'
    with patch("socket.create_connection") as connect:
        connect.return_value.__enter__.return_value = sock
        assert VisionRpcClient("localhost", 5020, "test", timeout=.2).request(command)["ok"]
        assert connect.call_args.kwargs["timeout"] == .2
    sock.settimeout.assert_called_once_with(expected)
    payload = json.loads(sock.makefile.return_value.write.call_args.args[0])
    assert payload["command"] == command
