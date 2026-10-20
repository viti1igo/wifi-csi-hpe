"""Preprocessed window cache: ACF arrays, an index, and a manifest with hashes."""
import dataclasses
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from src.features import segment_features
from src.index import build_windows, discover_sessions, load_segments

CODE = ("csi.py", "features.py", "index.py", "cache.py")


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def source_hashes(sessions):
    """Per-session fingerprint: hashes of the small files plus name and size of every .bin file."""
    out = {}
    for s in sessions:
        h = hashlib.sha256()
        for name in ("timestamps.csv", "meta.json", "annotations_v2.json"):
            p = s["path"] / name
            h.update(_sha(p).encode() if p.is_file() else b"-")
        for b in sorted(s["path"].glob("csi_*.bin")):
            h.update(f"{b.name}:{b.stat().st_size}:{b.stat().st_mtime_ns}".encode())
        out[s["key"]] = h.hexdigest()
    return out


def cache_id(cfg, sources):
    here = Path(__file__).parent
    payload = dict(dsp=dataclasses.asdict(cfg.dsp), windows=dataclasses.asdict(cfg.windows),
                   roles=dataclasses.asdict(cfg.roles), sources=sources,
                   code={n: _sha(here / n) for n in CODE})
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:12]


def write_cache(directory, feats, index, manifest):
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(directory)
    if len(feats) != len(index):
        raise ValueError(f"{len(feats)} arrays but {len(index)} index rows")
    directory.mkdir(parents=True)
    np.save(directory / "windows.npy", feats)
    index.to_csv(directory / "index.csv", index=False)
    manifest = dict(manifest, shape=list(feats.shape), windows_sha256=_sha(directory / "windows.npy"),
                    index_sha256=_sha(directory / "index.csv"))
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))


def load_cache(directory):
    """Return (features, index, manifest). Fails loudly if any file changed or shapes disagree."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, key in (("windows.npy", "windows_sha256"), ("index.csv", "index_sha256")):
        if _sha(directory / name) != manifest[key]:
            raise ValueError(f"hash mismatch for {name}: the cache changed")
    feats = np.load(directory / "windows.npy")
    index = pd.read_csv(directory / "index.csv")
    if list(feats.shape) != manifest["shape"] or len(index) != len(feats) or not np.isfinite(feats).all():
        raise ValueError("cache arrays are inconsistent or not finite")
    return feats, index, manifest


def check_cache_current(cfg, directory, deep=False):
    """Fail if the cache no longer matches the current code and configuration.

    deep=True also re-fingerprints the raw session files (reads about 1 GB of small files).
    """
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if directory.name != manifest["cache_id"]:
        raise ValueError(f"directory {directory.name} does not match its manifest id {manifest['cache_id']}")
    sources = manifest["sources"]
    if deep:
        by_key = {s["key"]: s for s in discover_sessions(cfg.paths)}
        sources = source_hashes([by_key[k] for k in sorted(sources)])
    if cache_id(cfg, sources) != manifest["cache_id"]:
        raise ValueError("cache is stale: the code, configuration or raw sources changed since it was built; rebuild it")


def _work(args):
    cfg, session, seg_id, start, end, starts = args
    feats, valid = segment_features(cfg, session, start, end, starts)
    return seg_id, feats, valid


def build_cache(cfg, workers=8):
    """Preprocess every eligible window. Refuses to overwrite an existing cache."""
    t0 = time.time()
    sessions = discover_sessions(cfg.paths)
    by_key = {s["key"]: s for s in sessions}
    seg = load_segments(sessions)
    win = build_windows(cfg, seg)
    sources = source_hashes([by_key[k] for k in sorted(set(win.session_key))])
    cid = cache_id(cfg, sources)
    directory = cfg.paths.cache / cid
    if directory.exists():
        raise FileExistsError(directory)
    segs = seg.set_index("segment_id")
    jobs = []
    for seg_id, w in win.groupby("segment_id", sort=False):
        r = segs.loc[seg_id]
        jobs.append((cfg, by_key[r.session_key], seg_id, w.start.min(), w.end.max(), w.start.to_numpy()))
    parts, valid_by_seg = {}, {}
    with ProcessPoolExecutor(workers) as pool:
        for n, (seg_id, feats, valid) in enumerate(pool.map(_work, jobs), 1):
            parts[seg_id], valid_by_seg[seg_id] = feats, valid
            if n % 20 == 0 or n == len(jobs):
                print(f"PREPROCESS {n}/{len(jobs)} segments, {time.time() - t0:.0f}s", flush=True)
    keep, feats = [], []
    for seg_id, w in win.groupby("segment_id", sort=False):
        keep.append(w[valid_by_seg[seg_id]])
        feats.append(parts[seg_id])
    index = pd.concat(keep, ignore_index=True)
    rejected = len(win) - len(index)
    manifest = dict(cache_id=cid, sources=sources, dsp=dataclasses.asdict(cfg.dsp), windows_total=len(win),
                    windows_rejected=int(rejected), seconds=round(time.time() - t0, 1))
    write_cache(directory, np.concatenate(feats), index, manifest)
    print(f"CACHE {directory} windows={len(index)} rejected={rejected} seconds={time.time() - t0:.0f}", flush=True)
    return directory
