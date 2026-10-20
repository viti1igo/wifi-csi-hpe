"""Signal chain: CSI records -> 4 links at 50 Hz -> band-pass -> 10 Hz -> ACF windows."""
import mmap
from functools import lru_cache
import numpy as np
import pandas as pd
from scipy import ndimage, signal as sp

from src.csi import parse_record


def pool_bins(amp, bins):
    """Average contiguous subcarriers into `bins` groups: (n, W) -> (n, bins)."""
    n, width = amp.shape
    if width % bins:
        raise ValueError(f"{width} subcarriers do not divide into {bins} bins")
    return amp.reshape(n, bins, width // bins).mean(axis=2)


def resample_links(times, values, grid, max_gap_s):
    """Interpolate each link onto `grid`. Returns (len(grid), links, bins) and a validity mask.

    A grid point is invalid when a link has no record around it or its neighbours are > max_gap_s apart.
    """
    out = np.empty((len(grid), len(times), values[0].shape[1]), np.float32)
    ok = np.ones(len(grid), bool)
    for j, (t, v) in enumerate(zip(times, values)):
        out[:, j] = _interp(t, v, grid)
        i = np.clip(np.searchsorted(t, grid), 1, len(t) - 1)
        ok &= (grid >= t[0]) & (grid <= t[-1]) & (t[i] - t[i - 1] <= max_gap_s)
    return out, ok


def _interp(t, v, grid):
    idx = np.clip(np.searchsorted(t, grid), 1, len(t) - 1)
    w = np.clip((grid - t[idx - 1]) / np.maximum(t[idx] - t[idx - 1], 1e-9), 0, 1)[:, None]
    return (v[idx - 1] * (1 - w) + v[idx] * w).astype(np.float32)


def despike(x, size, sigma):
    """Hampel filter along time: replace points more than `sigma` robust deviations from the running median."""
    med = ndimage.median_filter(x, size=(size, 1, 1), mode="reflect")
    dev = np.abs(x - med)
    mad = 1.4826 * ndimage.median_filter(dev, size=(size, 1, 1), mode="reflect")
    return np.where((mad > 0) & (dev > sigma * mad), med, x)


def bandpass(x, ok, dsp):
    """Zero-phase Butterworth band-pass inside each run of valid samples; NaN elsewhere.

    Runs shorter than one window are dropped, so the filter never crosses a gap.
    """
    sos = sp.butter(dsp.filter_order, dsp.band_hz, btype="bandpass", fs=dsp.grid_hz, output="sos")
    out = np.full(x.shape, np.nan, np.float32)
    edges = np.flatnonzero(np.diff(np.r_[False, ok, False]))
    for a, b in edges.reshape(-1, 2):
        if b - a >= 5 * dsp.grid_hz:
            out[a:b] = sp.sosfiltfilt(sos, x[a:b], axis=0)
    return out


def acf_windows(x, starts, lags):
    """Autocorrelation per (link, bin) of `lags` consecutive samples from each start index.

    Lag-dependent divisor as in Demo4. Constant windows give zeros.
    """
    w = x[np.asarray(starts)[:, None] + np.arange(lags)]                  # (n, lags, links, bins)
    w = w - w.mean(axis=1, keepdims=True)
    f = np.fft.rfft(w, n=2 * lags, axis=1)
    corr = np.fft.irfft(f * np.conj(f), n=2 * lags, axis=1)[:, :lags]
    denom = np.arange(lags, 0, -1)[None, :, None, None] * w.var(axis=1, keepdims=True)
    return np.divide(corr, denom, out=np.zeros_like(corr), where=denom > 1e-12).astype(np.float32)


@lru_cache(maxsize=4)
def _session_table(path):
    ts = pd.read_csv(f"{path}/timestamps.csv", usecols=["file", "offset", "epoch", "tx", "rx"])
    return ts.epoch.to_numpy(), ts.file.to_numpy(), ts.offset.to_numpy(), (ts.tx * 2 + ts.rx).to_numpy()


def read_links(session, t0, t1, width, bins):
    """Pooled amplitude of the four links between epochs t0 and t1.

    Returns per-link (times, values). Records of a non-dominant width are skipped.
    """
    epoch, files, offsets, link = _session_table(str(session["path"]))
    lo, hi = np.searchsorted(epoch, [t0, t1])
    times, vals, maps = [[] for _ in range(4)], [[] for _ in range(4)], {}
    for k in range(lo, hi):
        name = files[k]
        if name not in maps:
            fh = open(f"{session['path']}/{name}", "rb")
            maps[name] = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        rec = parse_record(maps[name], int(offsets[k]))
        if rec is None or len(rec[2]) != width:
            continue
        times[link[k]].append(epoch[k])
        vals[link[k]].append(rec[2].reshape(bins, -1).mean(axis=1))
    for m in maps.values():
        m.close()
    return [np.asarray(t) for t in times], [np.asarray(v, np.float32) for v in vals]


def segment_features(cfg, session, seg_start, seg_end, win_starts):
    """ACF windows for one segment. Returns (features for valid windows, boolean valid mask over win_starts)."""
    d = cfg.dsp
    width = dict(d.width)[session["collection"]]
    g0 = seg_start - d.margin_s
    grid = g0 + np.arange(int((seg_end - seg_start + 2 * d.margin_s) * d.grid_hz)) / d.grid_hz
    times, vals = read_links(session, grid[0] - 1, grid[-1] + 1, width, d.bins)
    if any(len(t) < 2 for t in times):
        return np.empty((0, d.lags, 4, d.bins), np.float32), np.zeros(len(win_starts), bool)
    x, ok = resample_links(times, vals, grid, d.max_gap_s)
    x = bandpass(despike(x, d.hampel_size, d.hampel_sigma), ok, d)
    step = int(d.grid_hz / d.out_hz)
    x10 = x[::step]
    first = np.rint((np.asarray(win_starts) - g0) * d.out_hz).astype(int)
    fin = np.isfinite(x10).all(axis=(1, 2))
    valid = np.array([f + d.lags <= len(x10) and fin[f:f + d.lags].all() for f in first])
    feats = acf_windows(np.nan_to_num(x10), first[valid], d.lags) if valid.any() else np.empty((0, d.lags, 4, d.bins), np.float32)
    return feats, valid
