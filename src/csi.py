"""Reader for MTK CSI frames.

The vendor parser (tools/CSIPacket.py) returns nothing under NumPy 2 because
it uses `np.complex_` inside a bare `except`. This reader needs only NumPy.
"""
import struct
import numpy as np

_TAGS = {8: "i", 9: "q", 15: "tx", 16: "rx"}


def parse_record(buf, pos):
    """Parse the frame at `pos` in a bytes-like buffer.

    Returns (tx, rx, amplitude float32, next_pos), or None for a bad frame.
    """
    if pos + 3 > len(buf) or buf[pos] != 0xAC:
        return None
    (size,) = struct.unpack_from("<H", buf, pos + 1)
    end = pos + 3 + size
    if end > len(buf):
        return None
    got, p = {}, pos + 3
    while p + 3 <= end:
        tag, n = struct.unpack_from("<BH", buf, p)
        p += 3
        if p + n > end:
            return None
        name = _TAGS.get(tag)
        if name in ("i", "q"):
            got[name] = np.frombuffer(buf, "<i2", n // 2, p)
        elif name:
            got[name] = int.from_bytes(buf[p:p + n], "little")
        p += n
    if len(got) < 4 or len(got["i"]) != len(got["q"]):
        return None
    return got["tx"], got["rx"], np.hypot(got["i"], got["q"]).astype(np.float32), end
