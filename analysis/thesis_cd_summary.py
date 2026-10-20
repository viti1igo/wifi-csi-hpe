"""Summarise thesis analyses C (DECISIONS D17) and D (D18) next to the matching study4 numbers.

C: does training on one Collection B date help on the other B date? Reference: study4 runs scored on the same date.
D: leave one pool session out; per session and pooled over sessions.
Every number uses each run's own validation-chosen M and threshold (src.metrics.evaluate_split).

    OMP_NUM_THREADS=2 python analysis/thesis_cd_summary.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.metrics import evaluate_split, smooth  # noqa: E402

RUNS = ROOT / "runs"
OUT = ROOT / "results" / "extra"
CLASS4 = {"empty": 0, "single-adult": 1, "single-child": 2, "multiple": 3}
PROBS = ["p_empty", "p_single-adult", "p_single-child", "p_multiple"]
KEYS = ("recall", "fa_rate", "auroc", "pauc05")


def runs_of(run_id, candidate):
    for run in sorted((RUNS / run_id / candidate).glob("*/seed_*")):
        if json.loads((run / "status.json").read_text())["state"] != "complete":
            raise RuntimeError(f"{run} is not complete")
        yield run, json.loads((run / "metrics.json").read_text()), pd.read_csv(run / "predictions.csv")


def score(p, split, met, date=None):
    d = p[(p.split == split) & p.eval_nonoverlap.astype(bool)]
    if date is not None:
        d = d[d.date == date]
    d = d.reset_index(drop=True).assign(class4=lambda x: x.label.map(CLASS4))
    m = evaluate_split(d, d[PROBS].to_numpy(), met["M"], met["threshold"])
    return {k: m[k] for k in KEYS} | dict(n_pos=m["n_pos"], n_neg=m["n_neg"])


def mean_sd(rows):
    df = pd.DataFrame(rows)
    out = {"runs": len(df)}
    for k in KEYS:
        v = df[k].dropna().astype(float)
        out[k] = f"{v.mean():.3f} ± {v.std(ddof=1):.3f}" if len(v) > 1 else (f"{v.mean():.3f}" if len(v) else "n/a")
    return out


def analysis_c():
    rows = []
    for test_date, run_id in (("2026-09-17", "thesisC_0911"), ("2026-09-11", "thesisC_0917")):
        for cand in ("b1", "c1"):
            ref = [score(p, "shift", met, test_date) for _, met, p in runs_of("study4", cand)]
            new = [score(p, "shift", met) for _, met, p in runs_of(run_id, cand)]
            prim_ref = [score(p, "primary", met) for _, met, p in runs_of("study4", cand)]
            prim_new = [score(p, "primary", met) for _, met, p in runs_of(run_id, cand)]
            for name, r, split in (("study4 (A only)", ref, f"shift {test_date}"), (f"{run_id} (A + other B date)", new, f"shift {test_date}"),
                                   ("study4 (A only)", prim_ref, "primary 2026-08-28"), (f"{run_id} (A + other B date)", prim_new, "primary 2026-08-28")):
                rows.append(dict(test=split, candidate=cand, training=name) | mean_sd(r))
    return pd.DataFrame(rows)


def analysis_d():
    per_session, pooled = [], []
    for cand in ("b1", "c1"):
        by_seed = {}
        for run, met, p in runs_of("thesisD_loso", cand):
            s = score(p, "test", met)
            per_session.append(dict(candidate=cand, session=run.parent.name.removeprefix("test_"), seed=run.name) | s)
            d = p[(p.split == "test") & p.eval_nonoverlap.astype(bool)].copy()
            d["alarm"] = smooth(d.reset_index(drop=True), d["p_single-child"].to_numpy(), met["M"]) > met["threshold"]
            by_seed.setdefault(run.name, []).append(d)
        for seed, frames in by_seed.items():
            d = pd.concat(frames)
            y = (d.label == "single-child").to_numpy()
            pooled.append(dict(candidate=cand, seed=seed, sessions=len(frames), recall=d.alarm[y].mean(),
                               fa_rate=d.alarm[~y].mean(), n_pos=int(y.sum()), n_neg=int((~y).sum())))
    ps = pd.DataFrame(per_session)
    table = ps.groupby(["candidate", "session"]).agg(seeds=("seed", "size"), n_pos=("n_pos", "first"), n_neg=("n_neg", "first"),
                                                     recall=("recall", "mean"), fa_rate=("fa_rate", "mean"),
                                                     auroc=("auroc", "mean")).round(3).reset_index()
    pool = pd.DataFrame(pooled).groupby("candidate").agg(seeds=("seed", "size"), sessions=("sessions", "first"),
                                                         recall_mean=("recall", "mean"), recall_sd=("recall", "std"),
                                                         fa_mean=("fa_rate", "mean"), fa_sd=("fa_rate", "std")).round(3).reset_index()
    return table, pool


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    c = analysis_c()
    c.to_csv(OUT / "analysis_c_both_collections.csv", index=False)
    d_sessions, d_pooled = analysis_d()
    d_sessions.to_csv(OUT / "analysis_d_loso_sessions.csv", index=False)
    d_pooled.to_csv(OUT / "analysis_d_loso_pooled.csv", index=False)
    print(c.to_string(index=False), "\n")
    print(d_sessions.to_string(index=False), "\n")
    print(d_pooled.to_string(index=False))


if __name__ == "__main__":
    main()
