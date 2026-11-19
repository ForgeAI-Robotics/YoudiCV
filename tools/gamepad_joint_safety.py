"""Pure joint-limit policy used by the OrangePi gamepad bridge."""


def allowed_by_soft_limit(current, candidate, low, high):
    """Allow motion into a soft-limit interval, including from a margin band."""
    if current > high:
        return candidate < current
    if current < low:
        return candidate > current
    return low <= candidate <= high
