import time

from piper_handeye.smoother_policy import VelocitySmootherRuntimeLease


class FakeTransport:
    def __init__(self):
        self.calls = []

    def __call__(self, enabled):
        self.calls.append(enabled)
        return {
            "active": enabled,
            "use_limit_speedup": False if enabled else True,
            "use_limit_back": False if enabled else True,
            "topology": "vendor_pause_lease",
            "vendor_smoother_paused": enabled,
        }


def test_runtime_lease_acquires_heartbeats_and_restores():
    transport = FakeTransport()
    beats = []
    lease = VelocitySmootherRuntimeLease(
        transport=transport, heartbeat_transport=lambda: beats.append(True),
        heartbeat_interval_sec=None,
    )
    assert lease.acquire()["active"] is True
    assert lease.active
    assert beats == [True]
    lease.heartbeat()
    assert beats == [True, True]
    assert lease.restore()["active"] is False
    assert not lease.active
    assert transport.calls == [True, False]


def test_policy_lease_restore_is_idempotent():
    transport = FakeTransport()
    lease = VelocitySmootherRuntimeLease(
        transport=transport, heartbeat_transport=lambda: None,
        heartbeat_interval_sec=None,
    )
    lease.acquire()
    lease.restore()
    calls = list(transport.calls)
    assert lease.restore() is None
    assert transport.calls == calls


def test_runtime_lease_heartbeats_independently_of_gui_loop():
    transport = FakeTransport()
    beats = []
    lease = VelocitySmootherRuntimeLease(
        transport=transport,
        heartbeat_transport=lambda: beats.append(time.monotonic()),
        heartbeat_interval_sec=0.01,
    )
    lease.acquire()
    time.sleep(0.045)
    lease.restore()
    count_after_restore = len(beats)
    assert count_after_restore >= 4
    time.sleep(0.025)
    assert len(beats) == count_after_restore
