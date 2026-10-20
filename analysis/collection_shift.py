"""Analysis B (docs/DECISIONS.md D16): how different are Collections A and B, and where do the shift-test false alarms fall?

Descriptive and post hoc. Reads cache 6ddd3107cff2 and the study4 predictions, changes nothing,
and trains nothing that is used for model selection.

    OMP_NUM_THREADS=2 python analysis/collection_shift.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.baselines import _features  # noqa: E402  (the B1 features: bin mean and spread per lag and link)
from src.metrics import smooth  # noqa: E402

CACHE = ROOT / "cache" / "6ddd3107cff2"
RUNS = ROOT / "runs" / "study4"
OUT = ROOT / "results" / "extra" / "collection_shift"
CLASSES = ("empty", "single-adult", "single-child", "multiple")
LAGS = (1, 5, 10, 25, 49)  # lag index k is k x 0.1 s (10 Hz after decimation)


def separability(index, X):
    """Can a logistic regression on the B1 features tell Collection A from B? Session-grouped CV, AUROC 0.5 = no."""
    rows = []
    for label in ("all",) + CLASSES:
        m = index.eval_nonoverlap.to_numpy() & ((index.label == label).to_numpy() if label != "all" else True)
        d = index[m]
        y = (d.collection == "B").to_numpy()
        n_a, n_b = d[~y].session_key.nunique(), d[y].session_key.nunique()
        cv = StratifiedGroupKFold(n_splits=min(5, n_a, n_b), shuffle=True, random_state=0)
        model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
        p = cross_val_predict(model, _features(X[np.flatnonzero(m)]), y, groups=d.session_key.to_numpy(), cv=cv,
                              method="predict_proba")[:, 1]
        rows.append(dict(label=label, windows=int(m.sum()), sessions_a=n_a, sessions_b=n_b, folds=cv.n_splits,
                         auroc_a_vs_b=round(float(roc_auc_score(y, p)), 3)))
    return pd.DataFrame(rows)


def acf_summary(index, X):
    """Mean ACF (over links and bins) at a few lags, per date and class, non-overlapping windows."""
    m = np.flatnonzero(index.eval_nonoverlap.to_numpy())
    curve = X[m].mean(axis=(2, 3))  # (windows, lags)
    d = index.iloc[m].reset_index(drop=True)
    cols = {f"acf_lag{k}": curve[:, k] for k in LAGS}
    table = pd.concat([d[["collection", "date", "label"]], pd.DataFrame(cols)], axis=1)
    out = table.groupby(["collection", "date", "label"]).agg(windows=("acf_lag1", "size"),
                                                             **{c: (c, "mean") for c in cols}).round(3)
    return out.reset_index(), d, curve


def plot_curves(d, curve, path):
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), sharey=True)
    lag_s = np.arange(curve.shape[1]) / 10
    for ax, label in zip(axes, CLASSES):
        for coll, colour in (("A", "#1f77b4"), ("B", "#d62728")):
            m = ((d.label == label) & (d.collection == coll)).to_numpy()
            if m.any():
                mu, sd = curve[m].mean(0), curve[m].std(0)
                ax.plot(lag_s, mu, color=colour, label=f"Collection {coll} (n={m.sum()})")
                ax.fill_between(lag_s, mu - sd, mu + sd, color=colour, alpha=0.15)
        ax.set_title(label)
        ax.set_xlabel("lag (s)")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("mean ACF over links and bins")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def shift_alarms(candidate):
    """Alarm rate per shift-test date, session and label, at each run's own validation-chosen M and threshold."""
    frames = []
    for run in sorted((RUNS / candidate).glob("val_*/seed_*")):
        met = json.loads((run / "metrics.json").read_text())
        p = pd.read_csv(run / "predictions.csv")
        d = p[(p.split == "shift") & p.eval_nonoverlap.astype(bool)].reset_index(drop=True)
        d["alarm"] = smooth(d, d["p_single-child"].to_numpy(), met["M"]) > met["threshold"]
        g = d.groupby(["date", "session_key", "label"]).alarm.agg(["size", "mean"]).reset_index()
        g["run"] = f"{run.parent.name}/{run.name}"
        frames.append(g)
    g = pd.concat(frames)
    out = g.groupby(["date", "session_key", "label"]).agg(windows=("size", "first"), runs=("run", "nunique"),
                                                          alarm_rate_mean=("mean", "mean"),
                                                          alarm_rate_sd=("mean", "std")).round(3).reset_index()
    out.insert(0, "candidate", candidate)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    index = pd.read_csv(CACHE / "index.csv")
    X = np.load(CACHE / "windows.npy", mmap_mode="r")
    assert len(index) == len(X), "cache index and windows disagree"

    sep = separability(index, X)
    sep.to_csv(OUT / "separability.csv", index=False)
    acf, d, curve = acf_summary(index, X)
    acf.to_csv(OUT / "acf_by_date_class.csv", index=False)
    plot_curves(d, curve, OUT / "acf_curves.png")
    alarms = pd.concat([shift_alarms(c) for c in ("b1", "c1")])
    alarms.to_csv(OUT / "shift_alarms_by_session.csv", index=False)
    (OUT / "provenance.json").write_text(json.dumps(dict(cache="6ddd3107cff2", runs="study4", decision="D16",
                                                         script="analysis/collection_shift.py"), indent=1))
    print(sep.to_string(index=False), "\n")
    print(alarms.to_string(index=False))


if __name__ == "__main__":
    main()
