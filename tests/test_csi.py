import io, struct, sys, glob
import numpy as np
import pandas as pd
import pytest
from config import Config
from src.csi import parse_record


def tlv(tag, payload):
    return struct.pack("<BH", tag, len(payload)) + payload


def make_record(i, q, tx, rx):
    body = (tlv(2, struct.pack("<I", 12345)) + tlv(3, struct.pack("<b", -60))
            + tlv(8, np.asarray(i, "<i2").tobytes()) + tlv(9, np.asarray(q, "<i2").tobytes())
            + tlv(15, bytes([tx])) + tlv(16, bytes([rx])))
    return b"\xac" + struct.pack("<H", len(body)) + body


def test_synthetic_record_amplitude_and_links():
    i, q = [3, -4, 0], [4, 3, 0]
    blob = b"junk" + make_record(i, q, 1, 0) + make_record(i, q, 0, 1)
    tx, rx, amp, end = parse_record(blob, 4)
    assert (tx, rx) == (1, 0)
    np.testing.assert_allclose(amp, [5.0, 5.0, 0.0])
    tx2, rx2, _, _ = parse_record(blob, end)
    assert (tx2, rx2) == (0, 1)


def test_bad_magic_returns_none():
    assert parse_record(b"\x00\x01\x02\x03", 0) is None


def test_missing_iq_returns_none():
    body = tlv(15, b"\x00") + tlv(16, b"\x00")
    assert parse_record(b"\xac" + struct.pack("<H", len(body)) + body, 0) is None


def _sessions():
    cfg = Config().paths
    a = sorted(cfg.collection_a.glob("2026-08-28/*/timestamps.csv"))
    b = sorted(cfg.collection_b.glob("2026-09-17/*/timestamps.csv"))
    return {"A": a[0].parent if a else None, "B": b[0].parent if b else None}


@pytest.mark.parametrize("collection,width", [("A", 512), ("B", 256)])
def test_reader_equals_vendor_parser_on_real_packets(collection, width, monkeypatch):
    session = _sessions()[collection]
    if session is None:
        pytest.skip("dataset not available")
    monkeypatch.setattr(np, "complex_", np.complex128, raising=False)  # vendor code needs it under NumPy 2
    monkeypatch.syspath_prepend(str(Config().paths.dataset / "CSI_handoff_20260903_123743" / "tools"))
    from CSIPacket import csi_packets
    ts = pd.read_csv(session / "timestamps.csv", nrows=400000)
    rows = ts.iloc[np.linspace(0, len(ts) - 1, 200).astype(int)]
    checked = 0
    for r in rows.itertuples():
        with open(session / r.file, "rb") as f:
            f.seek(int(r.offset))
            head = f.read(3)
            f.seek(int(r.offset))
            blob = f.read(3 + int.from_bytes(head[1:3], "little"))
        vendor = next(csi_packets(io.BytesIO(blob)))
        tx, rx, amp, _ = parse_record(blob, 0)
        assert (tx, rx) == (vendor.tx_path_idx, vendor.rx_path_idx) == (int(r.tx), int(r.rx))
        np.testing.assert_allclose(amp, np.abs(vendor.csi), rtol=1e-5)
        assert len(amp) == width
        checked += 1
    assert checked == 200
