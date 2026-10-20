import numpy as np
import pytest
from config import Config
from src.index import discover_sessions, load_segments, build_windows

CFG = Config()
pytestmark = pytest.mark.skipif(not CFG.paths.dataset.exists(), reason="dataset not available")


def test_inventory_matches_exploration_notebook():
    sessions = discover_sessions(CFG.paths)
    assert len(sessions) == 28
    assert sum(s["annotation"] is not None for s in sessions) == 22
    assert {s["collection"] for s in sessions} == {"A", "B"}
    assert len({s["key"] for s in sessions}) == 28


def test_raw_class_seconds_match_notebook():
    seg = load_segments(discover_sessions(CFG.paths))
    sec = seg.groupby("label").duration.sum()
    assert sec["empty"] == pytest.approx(7020, abs=1)
    assert sec["multiple"] == pytest.approx(3868, abs=1)
    assert sec["single-adult"] == pytest.approx(1088, abs=1)
    assert sec["single-child"] == pytest.approx(3329.4, abs=1)
    assert sec["transition"] == pytest.approx(1187.5, abs=1)


def test_unknown_child_type_is_removed_everywhere():
    seg = load_segments(discover_sessions(CFG.paths))
    unknown = seg[seg.child_type == "unknown"]
    assert unknown.groupby("label").duration.sum()["single-child"] == pytest.approx(657, abs=1)
    assert unknown.groupby("label").duration.sum()["multiple"] == pytest.approx(561, abs=1)
    win = build_windows(CFG, seg)
    assert not win.segment_id.isin(unknown.segment_id).any()
    assert set(win.label) == {"empty", "single-adult", "single-child", "multiple"}


def test_window_counts_match_data_contract():
    win = build_windows(CFG, load_segments(discover_sessions(CFG.paths)))
    ev = win[win.eval_nonoverlap]
    assert ev.groupby("role").size().to_dict() == {"pool": 1648, "primary": 305, "shift": 851}
    assert win.groupby("role").size().to_dict() == {"pool": 8031, "primary": 1493, "shift": 4182}
    child = win[win.label == "single-child"].groupby("role").segment_id.nunique().to_dict()
    assert child == {"pool": 11, "primary": 2, "shift": 6}


def test_windows_stay_inside_one_segment():
    seg = load_segments(discover_sessions(CFG.paths)).set_index("segment_id")
    win = build_windows(CFG, seg.reset_index())
    s = seg.loc[win.segment_id]
    assert (win.start.values >= s.start.values - 1e-6).all()
    assert (win.end.values <= s.end.values + 1e-6).all()
    assert ((win.end - win.start - 5.0).abs() < 1e-6).all()
    assert win.window_id.is_unique
    assert win.groupby("segment_id").role.nunique().max() == 1


def test_edge_distance_is_distance_to_the_nearest_segment_edge():
    seg = load_segments(discover_sessions(CFG.paths)).set_index("segment_id")
    win = build_windows(CFG, seg.reset_index())
    s = seg.loc[win.segment_id]
    expect = np.minimum(win.start.values - s.start.values, s.end.values - win.end.values)
    np.testing.assert_allclose(win.edge_distance_s.values, expect, atol=1e-6)
    assert (win.edge_distance_s >= -1e-6).all()
