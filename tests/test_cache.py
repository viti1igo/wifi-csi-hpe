import dataclasses
import numpy as np
import pandas as pd
import pytest
from config import Config
from src.cache import cache_id, write_cache, load_cache


def toy():
    feats = np.random.default_rng(0).normal(size=(6, 50, 4, 64)).astype("f4")
    index = pd.DataFrame(dict(window_id=list("abcdef"), label="empty", class4=0, session_key="A:x/y", segment_id="s#0",
                              role="pool", date="2026-08-25", collection="A", start=0.0, end=5.0, eval_nonoverlap=True))
    return feats, index


def test_roundtrip_and_refuse_overwrite(tmp_path):
    feats, index = toy()
    d = tmp_path / "abc"
    write_cache(d, feats, index, {"cache_id": "abc"})
    f2, i2, m = load_cache(d)
    np.testing.assert_array_equal(f2, feats)
    assert list(i2.window_id) == list(index.window_id) and m["shape"] == [6, 50, 4, 64]
    with pytest.raises(FileExistsError):
        write_cache(d, feats, index, {"cache_id": "abc"})


def test_tampered_array_is_refused(tmp_path):
    feats, index = toy()
    write_cache(tmp_path / "c", feats, index, {"cache_id": "c"})
    with open(tmp_path / "c" / "windows.npy", "r+b") as f:
        f.seek(-4, 2)
        f.write(b"\x00\x00\x80\x3f")
    with pytest.raises(ValueError, match="hash"):
        load_cache(tmp_path / "c")


def test_index_and_array_must_align(tmp_path):
    feats, index = toy()
    with pytest.raises(ValueError):
        write_cache(tmp_path / "bad", feats, index.iloc[:5], {"cache_id": "bad"})


def test_cache_id_reacts_to_config_and_sources():
    cfg = Config()
    src = {"A:x/y": "h1"}
    base = cache_id(cfg, src)
    assert base == cache_id(cfg, src)
    assert base != cache_id(cfg, {"A:x/y": "h2"})
    changed = dataclasses.replace(cfg, dsp=dataclasses.replace(cfg.dsp, band_hz=(0.1, 1.0)))
    assert base != cache_id(changed, src)
