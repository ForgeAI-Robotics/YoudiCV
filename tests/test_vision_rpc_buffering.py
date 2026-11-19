import io
import json
from unittest.mock import patch
import pytest
from piper_handeye.vision_rpc import VisionRpcClient

class FragmentedWire(io.RawIOBase):
    def __init__(self, response, fragment=307):
        super().__init__()
        self.response = bytearray(response)
        self.written = bytearray()
        self.read_calls = 0
        self.fragment = fragment
    def readable(self): return True
    def writable(self): return True
    def readinto(self, target):
        assert self.written.endswith(b'\n'), 'request must be flushed before read'
        self.read_calls += 1
        n = min(len(target), len(self.response), self.fragment)
        target[:n] = self.response[:n]
        del self.response[:n]
        return n
    def write(self, data):
        self.written.extend(data)
        return len(data)

class FakeSocket:
    def __init__(self, wire): self.wire = wire
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def settimeout(self, value): self.timeout = value
    def makefile(self, mode, buffering=-1):
        return self.wire if buffering == 0 else io.BufferedRWPair(self.wire, self.wire)

def test_fragmented_response_and_request_flush():
    payload = {'ok': True, 'samples': [{'sequence': n, 'x': '汉字'*30} for n in range(6)]}
    encoded = json.dumps(payload).encode()+b'\n'
    wire = FragmentedWire(encoded)
    with patch('socket.create_connection', return_value=FakeSocket(wire)):
        assert VisionRpcClient('test', 1, 'test').request('latest', after_sequence=10) == payload
    assert wire.closed
    assert json.loads(wire.written)['after_sequence'] == 10
    assert wire.read_calls < len(encoded)//100
    baseline = FragmentedWire(encoded)
    baseline.written.extend(b'{}\n')
    assert baseline.readline() == encoded
    assert baseline.read_calls == len(encoded)

@pytest.mark.parametrize('response, message', [(b'', 'closed without'), (b'{"ok":false,"error":"test failure"}\n', 'test failure')])
def test_errors_still_propagate(response, message):
    wire = FragmentedWire(response)
    with patch('socket.create_connection', return_value=FakeSocket(wire)):
        with pytest.raises(RuntimeError, match=message):
            VisionRpcClient('test', 1, 'test').request('latest')
    assert wire.closed
