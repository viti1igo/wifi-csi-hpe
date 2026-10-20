import dataclasses
import json
import numpy as np
import pandas as pd
import pytest
import torch

from config import Config, Paths
from src.cache import cache_id, check_cache_current, write_cache
from src.model import C1, parameter_count
from src.train import augment, class_weights, fit, predict, run, subset_rows

CFG = Config()
LABELS = ("empty", "single-adult", "single-child", "multiple")


def batch(n=6, seed=0):
    return torch.from_numpy(np.random.default_rng(seed).normal(size=(n, 50, 4, 64)).astype("f4"))


def test_forward_shape_parameter_count_and_bad_input():
    model = C1()
    out = model(batch())
    assert out.shape == (6, 4) and torch.isfinite(out).all()
    n = parameter_count(model)
    print("C1 parameters", n)
    assert 300_000 < n < 450_000
    with pytest.raises(ValueError):
        model(torch.zeros(2, 50, 2, 64))


def test_gradients_reach_encoder_and_head():
    model = C1()
    model(batch()).sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_checkpoint_roundtrip_gives_identical_outputs(tmp_path):
    a = C1().eval()
    torch.save({"model": a.state_dict()}, tmp_path / "m.pt")
    b = C1().eval()
    b.load_state_dict(torch.load(tmp_path / "m.pt", weights_only=True)["model"])
    x = batch()
    assert torch.equal(a(x), b(x))


def test_class_weights_come_from_the_labels_given_and_reject_missing_class():
    w = class_weights(np.array([0] * 6 + [1] * 2 + [2] * 2 + [3] * 2))
    assert w[0] < w[1] and w.mean() == pytest.approx(w.mean()) and float(w[0]) == pytest.approx(12 / (4 * 6))
    with pytest.raises(ValueError):
        class_weights(np.array([0, 1, 2]))


def test_augment_permutes_links_per_window_and_is_seeded():
    x = batch(8)
    t = dataclasses.replace(CFG.train, bin_dropout=0.0, jitter=0.0)
    g = torch.Generator().manual_seed(0)
    y = augment(x, g, t)
    assert y.shape == x.shape
    for i in range(8):                                 # same set of link matrices, possibly reordered
        assert sorted(y[i].sum(dim=(0, 2)).tolist()) == pytest.approx(sorted(x[i].sum(dim=(0, 2)).tolist()), rel=1e-4)
    g2 = torch.Generator().manual_seed(0)
    assert torch.equal(y, augment(x, g2, t))
    assert not torch.equal(x, y)


def toy_data(n=80, seed=0):
    rng = np.random.default_rng(seed)
    y = np.repeat(np.arange(4), n // 4)
    X = rng.normal(size=(n, 50, 4, 64)).astype("f4")
    X[y == 2] += 1.0
    return X, y


def tiny(epochs=2, patience=8):
    return dataclasses.replace(CFG.train, epochs=epochs, patience=patience, batch_size=32)


def test_fit_is_deterministic_on_cpu_for_a_seed(tmp_path):
    X, y = toy_data()
    states = []
    for k in range(2):
        torch.manual_seed(0)
        m = C1()
        fit(m, X, y, lambda mm: 0.5, tiny(), torch.device("cpu"), 3, tmp_path / f"{k}.pt")
        states.append({n: p.clone() for n, p in m.state_dict().items()})
    assert all(torch.equal(states[0][n], states[1][n]) for n in states[0])


def test_fit_keeps_the_best_epoch_first_on_ties_and_stops_early(tmp_path):
    X, y = toy_data()
    scripted = iter([0.2, 0.7, 0.7, 0.1, 0.1, 0.1])
    out = fit(C1(), X, y, lambda m: next(scripted), tiny(epochs=6, patience=3), torch.device("cpu"), 0, tmp_path / "b.pt")
    assert out["best_epoch"] == 2 and out["best_criterion"] == 0.7
    assert out["epochs_run"] == 5            # two more non-improving epochs after the tie, patience 3 -> stops at epoch 5
    assert torch.load(tmp_path / "b.pt", weights_only=True)["epoch"] == 2


def test_fit_learns_a_separable_toy_problem(tmp_path):
    X, y = toy_data(160)
    m = C1()
    child_vs_rest = lambda mm: float(((predict(mm, X, torch.device("cpu")).argmax(1) == 2) == (y == 2)).mean())
    out = fit(m, X, y, child_vs_rest, tiny(epochs=4), torch.device("cpu"), 0, tmp_path / "l.pt")
    assert out["best_criterion"] > 0.85          # base rate is 0.75


def test_subset_rows_is_class_balanced_and_repeatable():
    y = np.repeat(np.arange(4), 100)
    a, b = subset_rows(y, 0.1, 0), subset_rows(y, 0.1, 0)
    assert (a == b).all() and np.bincount(y[a]).tolist() == [10, 10, 10, 10]


def fake_cache(root, cfg=None, sources=None):
    rows, feats = [], []
    rng = np.random.default_rng(0)
    plan = [("pool", d) for d in CFG.roles.pool] + [("primary", "2026-08-28"), ("shift", "2026-09-11")]
    for role, date in plan:
        for c, label in enumerate(LABELS):
            for seg in range(2):
                for k in range(5):
                    rows.append(dict(window_id=f"{date}-{label}-{seg}-{k}", segment_id=f"{date}-{label}-{seg}", session_key=f"{date}-s{seg}",
                                     collection="A", date=date, label=label, class4=c, start=5.0 * k, end=5.0 * k + 5, edge_distance_s=5.0 * k, role=role,
                                     eval_nonoverlap=True))
                    feats.append(rng.normal(size=(50, 4, 64)) + (c == 2))
    index = pd.DataFrame(rows)
    cfg = cfg or dataclasses.replace(CFG, paths=Paths(root=root))
    sources = sources or {}
    cid = cache_id(cfg, sources)
    write_cache(root / "cache" / cid, np.asarray(feats, "f4"), index, {"cache_id": cid, "sources": sources})


def test_run_writes_all_artifacts_and_refuses_to_overwrite(tmp_path):
    fake_cache(tmp_path)
    cfg = dataclasses.replace(CFG, paths=Paths(root=tmp_path))
    date = CFG.roles.pool[0]
    for cand in ("b0", "b1", "c1"):
        d = run(cfg, cand, date, 0, "t", "cpu", epochs=1)
        names = {p.name for p in d.iterdir()}
        assert {"status.json", "run.json", "metrics.json", "predictions.csv"} <= names
        assert json.loads((d / "status.json").read_text())["state"] == "complete"
        m = json.loads((d / "metrics.json").read_text())
        assert set(m) >= {"M", "threshold", "val", "primary", "shift"}
    assert {"best.pt", "final.pt", "epoch_metrics.jsonl"} <= {p.name for p in (tmp_path / "runs/t/c1" / f"val_{date}/seed_0").iterdir()}
    b0 = json.loads((tmp_path / "runs/t/b0" / f"val_{date}/seed_0/metrics.json").read_text())
    assert b0["primary"]["recall"] == 0 and b0["primary"]["fa_rate"] == 0
    with pytest.raises(FileExistsError):
        run(cfg, "b0", date, 0, "t", "cpu")


def test_failed_run_is_kept_and_marked_failed(tmp_path, monkeypatch):
    fake_cache(tmp_path)
    cfg = dataclasses.replace(CFG, paths=Paths(root=tmp_path))
    import src.train as train
    monkeypatch.setattr(train, "finalize", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError):
        run(cfg, "b0", CFG.roles.pool[0], 0, "f", "cpu")
    status = json.loads((tmp_path / "runs/f/b0" / f"val_{CFG.roles.pool[0]}/seed_0/status.json").read_text())
    assert status["state"] == "failed" and "boom" in status["error"]


def test_bad_names_and_devices_are_rejected(tmp_path):
    cfg = dataclasses.replace(CFG, paths=Paths(root=tmp_path))
    with pytest.raises(ValueError):
        run(cfg, "b0", CFG.roles.pool[0], 0, "../evil", "cpu")
    with pytest.raises(ValueError):
        run(cfg, "b0", CFG.roles.pool[0], 0, "ok", "gpu0")
    with pytest.raises(ValueError):
        run(cfg, "nope", CFG.roles.pool[0], 0, "ok", "cpu")
    assert not (tmp_path / "runs").exists()


def test_c2_shape_size_gradients_and_decomposition():
    from src.model import C2, Decomposition
    model = C2()
    out = model(batch())
    assert out.shape == (6, 4) and torch.isfinite(out).all()
    n = parameter_count(model)
    print("C2 parameters", n)
    assert 50_000 < n < 250_000
    out.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    with pytest.raises(ValueError):
        model(torch.zeros(2, 50, 2, 64))
    const = torch.full((2, 50, 8), 3.0)
    assert Decomposition(5)(const).abs().max() < 1e-5          # a constant series is all trend
    ramp = torch.arange(50.0)[None, :, None].expand(1, 50, 4)
    assert Decomposition(5)(ramp)[:, 5:-5].abs().max() < 1e-4  # a linear ramp is all trend away from the edges


def test_stale_cache_is_refused(tmp_path):
    fake_cache(tmp_path)
    cfg = dataclasses.replace(CFG, paths=Paths(root=tmp_path))
    d = next((tmp_path / "cache").iterdir())
    check_cache_current(cfg, d)                                          # fresh: fine
    changed = dataclasses.replace(cfg, dsp=dataclasses.replace(cfg.dsp, band_hz=(0.1, 1.0)))
    with pytest.raises(ValueError, match="stale"):
        check_cache_current(changed, d)
    with pytest.raises(ValueError, match="stale"):
        run(changed, "b0", CFG.roles.pool[0], 0, "stale", "cpu")
    (tmp_path / "cache" / "renamed").mkdir()
    d.rename(tmp_path / "cache" / "renamed2")
    with pytest.raises(ValueError, match="does not match"):
        check_cache_current(cfg, tmp_path / "cache" / "renamed2")


def make_dataset(root, names=("s1",)):
    for name in names:
        d = root / "MTK_dataset/CSI_handoff_20260903_123743/data/2026-08-25" / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "meta.json").write_text("{}")
        (d / "timestamps.csv").write_text("file,offset,epoch,tx,rx\n")
        (d / "csi_0000.bin").write_bytes(b"x" * 10)
        (d / "annotations_v2.json").write_text(json.dumps({"annotations": [
            {"label": "empty", "start": 1000.0, "end": 1020.0, "child_type": None}]}))


def dataset_cfg(root):
    from src.cache import source_hashes
    from src.index import discover_sessions
    cfg = dataclasses.replace(CFG, paths=Paths(root=root, dataset=root / "MTK_dataset"))
    fake_cache(root, cfg, source_hashes(discover_sessions(cfg.paths)))
    return cfg


def test_raw_source_change_is_caught_when_sources_are_verified(tmp_path):
    make_dataset(tmp_path)
    cfg = dataset_cfg(tmp_path)
    run(cfg, "b0", CFG.roles.pool[0], 0, "ok", "cpu", verify_sources=True)
    (tmp_path / "MTK_dataset/CSI_handoff_20260903_123743/data/2026-08-25/s1/timestamps.csv").write_text("file,offset,epoch,tx,rx\n0,0,1.0,0,0\n")
    with pytest.raises(ValueError, match="stale"):
        run(cfg, "b0", CFG.roles.pool[0], 0, "bad", "cpu", verify_sources=True)
    run(cfg, "b0", CFG.roles.pool[1], 0, "shallow", "cpu")                                   # the cheap check alone does not notice


def test_added_or_removed_eligible_sessions_are_caught(tmp_path):
    make_dataset(tmp_path)
    cfg = dataset_cfg(tmp_path)
    make_dataset(tmp_path, ("s2",))                                                        # a new eligible session appears
    with pytest.raises(ValueError, match="added"):
        run(cfg, "b0", CFG.roles.pool[0], 0, "added", "cpu", verify_sources=True)
    import shutil
    shutil.rmtree(tmp_path / "MTK_dataset/CSI_handoff_20260903_123743/data/2026-08-25/s1")   # and the original one disappears
    shutil.rmtree(tmp_path / "MTK_dataset/CSI_handoff_20260903_123743/data/2026-08-25/s2")
    make_dataset(tmp_path, ("s3",))
    with pytest.raises(ValueError, match="removed"):
        run(cfg, "b0", CFG.roles.pool[0], 0, "removed", "cpu", verify_sources=True)


def test_training_overrides_are_applied_recorded_and_restricted(tmp_path):
    fake_cache(tmp_path)
    cfg = dataclasses.replace(CFG, paths=Paths(root=tmp_path))
    d = run(cfg, "c2", CFG.roles.pool[0], 0, "ov", "cpu", epochs=1, overrides={"encoder_lr": 0.01, "head_lr": 0.02, "patience": 3})
    recorded = json.loads((d / "run.json").read_text())["config"]
    assert (recorded["encoder_lr"], recorded["head_lr"], recorded["patience"], recorded["epochs"]) == (0.01, 0.02, 3, 1)
    with pytest.raises(ValueError, match="can be overridden"):
        run(cfg, "c2", CFG.roles.pool[0], 0, "ov2", "cpu", overrides={"weight_decay": 0.5})
    default = run(cfg, "c2", CFG.roles.pool[0], 1, "ov", "cpu", epochs=1)
    assert json.loads((default / "run.json").read_text())["config"]["encoder_lr"] == CFG.train.encoder_lr     # no override: unchanged
