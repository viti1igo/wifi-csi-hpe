import time
import numpy as np
import pytest
from config import Config
from src.features import pool_bins, resample_links, despike, bandpass, acf_windows

DSP = Config().dsp


def test_pool_bins_averages_contiguous_groups():
    x = np.arange(2 * 512, dtype="f4").reshape(2, 512)
    out = pool_bins(x, 64)
    assert out.shape == (2, 64)
    np.testing.assert_allclose(out[0, 0], np.mean(np.arange(8)))
    with pytest.raises(ValueError):
        pool_bins(np.zeros((1, 100)), 64)


def test_resample_flags_gaps_and_uncovered_edges():
    t = np.r_[np.arange(0, 4, 0.02), np.arange(5.5, 10, 0.02)]      # 1.5 s hole between 4 and 5.5
    v = np.tile(t[:, None], (1, 64)).astype("f4")
    grid = np.arange(-1, 11, 0.02)
    vals, ok = resample_links([t, t, t, t], [v, v, v, v], grid, max_gap_s=0.5)
    assert vals.shape == (len(grid), 4, 64)
    assert ok[np.searchsorted(grid, 2.0)] and ok[np.searchsorted(grid, 8.0)]
    assert not ok[np.searchsorted(grid, 4.5)]      # inside the hole
    assert not ok[np.searchsorted(grid, -0.5)]     # before the first record
    np.testing.assert_allclose(vals[np.searchsorted(grid, 2.0), 0, 0], 2.0, atol=1e-3)


def test_bandpass_keeps_breathing_and_removes_fast_motion():
    fs = DSP.grid_hz
    t = np.arange(0, 60, 1 / fs)
    ok = np.ones(len(t), bool)
    for f, low in ((0.3, False), (3.0, True), (0.02, True)):
        x = np.sin(2 * np.pi * f * t)[:, None, None] * np.ones((1, 4, 64))
        y = bandpass(x, ok, DSP)[500:-500]
        gain = np.abs(y).max()
        assert (gain < 0.1) if low else (gain > 0.9)


def test_bandpass_skips_short_runs_and_does_not_cross_gaps():
    n = 3000
    x = np.random.default_rng(0).normal(size=(n, 4, 64)).astype("f4")
    ok = np.zeros(n, bool)
    ok[:1000] = True          # long run: filtered
    ok[1100:1110] = True      # 10 samples: too short, dropped
    y = bandpass(x, ok, DSP)
    assert np.isfinite(y[:1000]).all()
    assert np.isnan(y[1000:]).all()


def test_despike_removes_single_outlier():
    x = 1 + 0.01 * np.random.default_rng(2).normal(size=(200, 4, 64)).astype("f4")
    x[100] = 50
    y = despike(x, DSP.hampel_size, DSP.hampel_sigma)
    assert y[100].max() < 2
    noise = np.random.default_rng(1).normal(size=(1000, 4, 64)).astype("f4")
    changed = despike(noise, DSP.hampel_size, DSP.hampel_sigma) != noise
    assert changed[100:-100].mean() < 0.02      # about 0.3% expected for Gaussian noise at 3 sigma


def test_acf_lag_zero_is_one_and_sine_peaks_at_period():
    fs, n = 10, 200
    t = np.arange(n) / fs
    x = np.sin(2 * np.pi * 0.5 * t)[:, None, None] * np.ones((1, 4, 64))    # period 2 s = 20 samples
    out = acf_windows(x, np.array([0, 40, 100]), DSP.lags)
    assert out.shape == (3, 50, 4, 64) and out.dtype == np.float32
    np.testing.assert_allclose(out[:, 0], 1.0, atol=1e-4)
    assert out[0, 20, 0, 0] > 0.8 and out[0, 10, 0, 0] < -0.5


def test_acf_of_constant_is_zero_not_nan():
    out = acf_windows(np.ones((100, 4, 64)), np.array([0]), DSP.lags)
    assert np.isfinite(out).all() and np.abs(out).max() == 0


def test_real_segment_gives_valid_acf_windows():
    from src.index import discover_sessions, load_segments, build_windows
    from src.features import segment_features
    cfg = Config()
    if not cfg.paths.dataset.exists():
        pytest.skip("dataset not available")
    sessions = {s["key"]: s for s in discover_sessions(cfg.paths)}
    win = build_windows(cfg, load_segments(list(sessions.values())))
    seg_id = win[(win.label == "single-child") & (win.role == "primary")].segment_id.iloc[0]
    w = win[win.segment_id == seg_id]
    session = sessions[w.session_key.iloc[0]]
    t0 = time.time()
    feats, valid = segment_features(cfg, session, w.start.min(), w.end.max(), w.start.to_numpy())
    print(f"segment {seg_id}: {len(w)} windows, {valid.sum()} valid, {time.time() - t0:.1f}s")
    assert feats.shape == (valid.sum(), 50, 4, 64) and feats.dtype == np.float32
    assert valid.mean() > 0.9
    assert np.isfinite(feats).all() and np.abs(feats).max() <= DSP.lags + 1e-3   # |acf(l)| <= N/(N-l)
    np.testing.assert_allclose(feats[:, 0], 1.0, atol=1e-3)
