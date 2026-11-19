import struct

import pytest

from piper_handeye.ros1_topic_monitor import decode_lr_speed


def test_decode_lr_speed():
    assert decode_lr_speed(struct.pack("<2d", -0.125, 0.25)) == {
        "left_speed": -0.125,
        "right_speed": 0.25,
    }


def test_decode_lr_speed_rejects_short_payload():
    with pytest.raises(ValueError, match="short LRSpeed payload"):
        decode_lr_speed(b"\0" * 15)
