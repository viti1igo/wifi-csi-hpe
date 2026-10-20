import dataclasses
import json
import numpy as np
import pandas as pd
import pytest
from config import Config, Paths
from src.report import build_report, check_runs, collect, interior_summary, select_candidate, summarize

IDENT = dict(cache_id="cache1", split_hash="split1", source_sha256={"a.py": "x"}, config={"epochs": 40})


def fake_run(root, run_id, cand, date, seed, state="complete", val_pauc=0.9, primary_recall=0.5, ident=None, M=1):
    d = root / "runs" / run_id / cand / f"val_{date}" / f"seed_{seed}"
    d.mkdir(parents=True)
    (d / "status.json").write_text(json.dumps({"state": state, "best_epoch": 3, "epochs_run": 5}))
    if state != "complete":
        return
    (d / "run.json").write_text(json.dumps(ident or IDENT))
    split = lambda rec, pauc=0.8: dict(n=10, n_pos=4, n_neg=6, recall=rec, fa_rate=0.0, alarm_fraction=0.2, auroc=0.9, ap=0.8, pauc05=pauc,
                                        fa_by_negative={"empty": 0.0, "single-adult": None, "multiple": 0.0},
                                        recall_at={"0.0": rec, "0.001": rec, "0.01": rec, "0.05": rec},
                                        per_session=[dict(session_key="A:x/y", n=10, n_pos=4, n_neg=6, recall=rec, fa_rate=0.0)],
                                        confusion_matrix=np.eye(4, dtype=int).tolist())
    (d / "metrics.json").write_text(json.dumps({"M": M, "threshold": 0.5, "val": split(0.9, val_pauc),
                                                "primary": split(primary_recall), "shift": split(0.1)}))
    rows = [dict(split=s, label=l, eval_nonoverlap=True, edge_distance_s=e, segment_id=f"g{i}", start=5.0 * i, **{"p_single-child": p})
            for s in ("val", "primary", "shift") for i, (l, p, e) in enumerate((("single-child", 0.9, 30.0), ("empty", 0.1, 21.0)))]
    pd.DataFrame(rows).to_csv(d / "predictions.csv", index=False)


def cfg_at(tmp_path):
    return dataclasses.replace(Config(), paths=Paths(root=tmp_path))


def keys(runs):
    return {(r["candidate"], r["val_date"], r["seed"]) for r in runs}


def test_rerun_replaces_a_failed_run_and_failures_are_listed(tmp_path):
    fake_run(tmp_path, "s1", "c1", "2026-08-25", 0)
    fake_run(tmp_path, "s1", "c1", "2026-08-26", 0, state="failed")
    fake_run(tmp_path, "s1r", "c1", "2026-08-26", 0, primary_recall=0.7)
    runs, failed = collect(cfg_at(tmp_path), ["s1", "s1r"])
    assert len(runs) == 2 and [f["run_id"] for f in failed] == ["s1"]
    assert {r["run_id"]: r["primary_recall"] for r in runs}["s1r"] == 0.7


def test_summary_is_mean_and_std_over_runs(tmp_path):
    for seed, rec in enumerate((0.4, 0.6)):
        fake_run(tmp_path, "s", "c1", "2026-08-25", seed, primary_recall=rec)
    runs, _ = collect(cfg_at(tmp_path), ["s"])
    s = summarize(runs).set_index(["candidate", "split"])
    assert s.loc[("c1", "primary"), "recall_mean"] == pytest.approx(0.5)
    assert s.loc[("c1", "primary"), "recall_std"] == pytest.approx(np.std([0.4, 0.6], ddof=1))


def test_runs_with_different_identity_or_missing_jobs_are_refused(tmp_path):
    fake_run(tmp_path, "s", "c1", "2026-08-25", 0)
    fake_run(tmp_path, "t", "c1", "2026-08-26", 0, ident=dict(IDENT, source_sha256={"a.py": "CHANGED"}))
    runs, _ = collect(cfg_at(tmp_path), ["s", "t"])
    with pytest.raises(ValueError, match="source_sha256"):
        check_runs(runs, keys(runs))
    same, _ = collect(cfg_at(tmp_path), ["s"])
    with pytest.raises(ValueError, match="expected runs"):
        check_runs(same, keys(same) | {("c2", "2026-08-25", 0)})
    check_runs(same, keys(same))


def test_selection_uses_validation_and_seed_spread_not_fold_spread(tmp_path):
    # Dates differ a lot (0.5 to 0.9) but seeds agree; c2 is 0.05 better everywhere.
    for d, base in zip(("2026-08-25", "2026-08-26", "2026-09-01"), (0.5, 0.9, 0.7)):
        for seed, jitter in enumerate((-0.002, 0.0, 0.002)):
            fake_run(tmp_path, "s", "c1", d, seed, val_pauc=base + jitter, primary_recall=0.1)
            fake_run(tmp_path, "s", "c2", d, seed, val_pauc=base + 0.05 + jitter, primary_recall=0.1)
    runs, _ = collect(cfg_at(tmp_path), ["s"])
    assert select_candidate(runs)["chosen"] == "c2"
    # Seeds disagree strongly, c2 only slightly better: keep c1 (and test scores never matter).
    for d in ("2026-08-25", "2026-08-26"):
        for seed, jitter in enumerate((-0.1, 0.0, 0.1)):
            fake_run(tmp_path, "t", "c1", d, seed, val_pauc=0.8 + jitter, primary_recall=0.1)
            fake_run(tmp_path, "t", "c2", d, seed, val_pauc=0.82 + jitter, primary_recall=0.99)
    runs, _ = collect(cfg_at(tmp_path), ["t"])
    assert select_candidate(runs)["chosen"] == "c1"


def test_names_that_escape_their_folder_are_rejected(tmp_path):
    with pytest.raises(ValueError):
        collect(cfg_at(tmp_path), ["../evil"])
    fake_run(tmp_path, "s", "c1", "2026-08-25", 0)
    with pytest.raises(ValueError):
        build_report(cfg_at(tmp_path), ["s"], "../../outside")
    assert not (tmp_path.parent / "outside").exists()


def test_build_report_writes_everything(tmp_path):
    for cand in ("b0", "c1", "c2"):
        for seed in (0, 1):
            fake_run(tmp_path, "s", cand, "2026-08-25", seed)
    cfg = cfg_at(tmp_path)
    runs, _ = collect(cfg, ["s"])
    out = build_report(cfg, ["s"], "s", expected=keys(runs))
    names = {p.name for p in out.iterdir()}
    assert {"runs.csv", "summary.csv", "per_session.csv", "selection.json", "manifest.json", "roc.png", "REPORT.md"} <= names
    text = (out / "REPORT.md").read_text()
    assert "dummy" in text and "(oracle)" in text and "Headline" in text and "Interior windows only" in text
    assert text.index("Selection") < text.index("Primary test") and "true \\ predicted" in text
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["identity"]["cache_id"] == "cache1" and len(manifest["accepted"]) == 6


def test_extra_unexpected_runs_are_refused(tmp_path):
    fake_run(tmp_path, "s", "b1", "2026-08-25", 0)
    fake_run(tmp_path, "s", "b1", "2026-08-25", 7)                       # a seed outside the study matrix
    runs, _ = collect(cfg_at(tmp_path), ["s"])
    with pytest.raises(ValueError, match="unexpected"):
        check_runs(runs, {("b1", "2026-08-25", 0)})


def test_interior_table_has_all_three_splits(tmp_path):
    fake_run(tmp_path, "s", "c1", "2026-08-25", 0, M=1)
    fake_run(tmp_path, "s", "c1", "2026-08-25", 1, M=3)
    runs, _ = collect(cfg_at(tmp_path), ["s"])
    df = interior_summary(runs, 20.0).set_index(["candidate", "split"])
    assert set(df.index.get_level_values("split")) == {"val", "primary", "shift"}
    assert df.loc[("c1", "primary"), "windows"] == pytest.approx(2.0)      # single-row segments: each row is its own average
