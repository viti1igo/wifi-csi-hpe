"""Read finished run directories and write tables, an ROC figure and REPORT.md. Never edits a run."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import safe_name
from src.metrics import smooth
from src.study import matrix

SPLITS = ("val", "primary", "shift")
SCALARS = ("recall", "fa_rate", "auroc", "ap", "pauc05", "alarm_fraction")
FA_LEVELS = ("0.0", "0.001", "0.01", "0.05")
NEGATIVES = ("empty", "single-adult", "multiple")
CLASSES = ("empty", "single-adult", "single-child", "multiple")
ORDER = ("b0", "b1", "c1", "c2")
IDENTITY = ("cache_id", "split_hash", "source_sha256", "config")


def _flat(status, metrics, identity, run_id, cand, date, seed, directory):
    row = dict(run_id=run_id, candidate=cand, val_date=date, seed=seed, best_epoch=status.get("best_epoch"),
               epochs_run=status.get("epochs_run"), M=metrics["M"], threshold=metrics["threshold"], dir=directory,
               sessions={}, confusion={}, **{k: identity[k] for k in IDENTITY})
    for s in SPLITS:
        m = metrics[s]
        row.update({f"{s}_{k}": m[k] for k in SCALARS})
        row.update({f"{s}_recall_at_{k}": m["recall_at"][k] for k in FA_LEVELS})
        row.update({f"{s}_fa_{k}": m["fa_by_negative"][k] for k in NEGATIVES})
        row["sessions"][s], row["confusion"][s] = m["per_session"], m["confusion_matrix"]
    return row


def collect(cfg, run_ids):
    """Complete runs (a later run id replaces an earlier one for the same job) and every non-complete run."""
    best, failed = {}, []
    for run_id in map(safe_name, run_ids):
        for status_file in sorted((cfg.paths.runs / run_id).glob("*/val_*/seed_*/status.json")):
            d = status_file.parent
            cand, date, seed = d.parent.parent.name, d.parent.name[4:], int(d.name[5:])
            status = json.loads(status_file.read_text())
            if status.get("state") != "complete":
                failed.append(dict(run_id=run_id, candidate=cand, val_date=date, seed=seed, state=status.get("state")))
                continue
            identity = json.loads((d / "run.json").read_text())
            best[(cand, date, seed)] = _flat(status, json.loads((d / "metrics.json").read_text()), identity, run_id, cand, date, seed, d)
    return list(best.values()), failed


def check_runs(runs, expected):
    """The study must be complete and every run must come from the same cache, split, code and configuration."""
    actual, expected = {(r["candidate"], r["val_date"], r["seed"]) for r in runs}, set(expected)
    missing, extra = sorted(expected - actual), sorted(actual - expected)
    if missing or extra:
        raise ValueError(f"missing {len(missing)} expected runs {missing[:3]}; {len(extra)} unexpected runs {extra[:3]}")
    for key in IDENTITY:
        values = {json.dumps(r[key], sort_keys=True) for r in runs}
        if len(values) > 1:
            raise ValueError(f"runs differ in {key} ({len(values)} distinct values); they cannot be combined into one report")


def _num(col):
    return pd.to_numeric(pd.Series(col), errors="coerce")


def summarize(runs):
    df = pd.DataFrame(runs)
    out = []
    for cand, g in df.groupby("candidate"):
        for s in SPLITS:
            row = dict(candidate=cand, split=s, runs=len(g))
            for k in SCALARS + tuple(f"recall_at_{x}" for x in FA_LEVELS) + tuple(f"fa_{n}" for n in NEGATIVES):
                v = _num(g[f"{s}_{k}"])
                row[f"{k}_mean"], row[f"{k}_std"] = v.mean(), v.std(ddof=1)
            out.append(row)
    return pd.DataFrame(out)


def select_candidate(runs):
    """Validation only. C2 replaces C1 when its mean per-date validation pAUC gain exceeds the seed-to-seed spread.

    Seed spread = mean over dates and both candidates of the standard deviation across seeds within a date.
    """
    df = pd.DataFrame(runs)
    df = df[df.candidate.isin(("c1", "c2"))].assign(v=lambda x: _num(x.val_pauc05))
    if df.candidate.nunique() < 2:
        return dict(chosen=df.candidate.iloc[0] if len(df) else None, rule="only one learned candidate has runs")
    g = df.groupby(["candidate", "val_date"]).v
    mean, std = g.mean().unstack(0), g.std(ddof=1).unstack(0)
    gain, spread = float((mean.c2 - mean.c1).mean()), float(np.nanmean(std.to_numpy()))
    return dict(chosen="c2" if gain > spread else "c1", c1_mean=float(mean.c1.mean()), c2_mean=float(mean.c2.mean()),
                gain=gain, seed_spread=spread,
                rule="c2 only if its mean per-date validation pAUC gain over c1 exceeds the mean within-date standard deviation across seeds")


def _fmt(mean, std):
    return "n/a" if pd.isna(mean) else (f"{mean:.3f}" if pd.isna(std) else f"{mean:.3f} ± {std:.3f}")


def _table(summary, split, cols):
    s = summary[summary.split == split].set_index("candidate").reindex([c for c in ORDER if c in set(summary.candidate)])
    head = "| candidate | runs | " + " | ".join(name for name, _ in cols) + " |\n|---|---:|" + "---:|" * len(cols)
    rows = [f"| {c} | {int(r.runs)} | " + " | ".join(_fmt(r[f"{k}_mean"], r[f"{k}_std"]) for _, k in cols) + " |" for c, r in s.iterrows()]
    return head + "\n" + "\n".join(rows)


def _confusion(runs, cand, split):
    total = np.sum([r["confusion"][split] for r in runs if r["candidate"] == cand], axis=0)
    head = "| true \\ predicted | " + " | ".join(CLASSES) + " | support |\n|---|" + "---:|" * (len(CLASSES) + 1)
    return head + "\n" + "\n".join(f"| {c} | " + " | ".join(str(int(v)) for v in row) + f" | {int(row.sum())} |" for c, row in zip(CLASSES, total))


def interior_summary(runs, interior_s):
    """Recall and false alarms on windows at least `interior_s` from a segment edge, at each run's validation operating point.

    The smoothed score of a window averages the previous M-1 rows too, so a row is kept only if the minimum edge distance
    over the rows that actually enter its average is at least `interior_s`.
    """
    rows = []
    for r in runs:
        p = pd.read_csv(Path(r["dir"]) / "predictions.csv")
        for split in SPLITS:
            d = p[(p.split == split) & p.eval_nonoverlap.astype(bool)].reset_index(drop=True)
            score = smooth(d, d["p_single-child"].to_numpy(), r["M"])
            alarm, y, keep = score > r["threshold"], (d.label == "single-child").to_numpy(), (smooth(d, d.edge_distance_s.to_numpy(), r["M"], "min") >= interior_s)
            rows.append(dict(candidate=r["candidate"], split=split, windows=int(keep.sum()),
                             recall=alarm[y & keep].mean() if (y & keep).any() else np.nan,
                             fa_rate=alarm[~y & keep].mean() if (~y & keep).any() else np.nan))
    df = pd.DataFrame(rows)
    out = df.groupby(["candidate", "split"]).agg(windows=("windows", "mean"), recall_mean=("recall", "mean"), recall_std=("recall", "std"),
                                                 fa_mean=("fa_rate", "mean"), fa_std=("fa_rate", "std")).reset_index()
    return out


def pooled_val_auroc(runs):
    """Validation scores of the three folds pooled, per candidate and seed, then averaged over seeds."""
    from sklearn.metrics import roc_auc_score
    out = {}
    for cand in sorted({r["candidate"] for r in runs}):
        per_seed = {}
        for r in (x for x in runs if x["candidate"] == cand):
            p = pd.read_csv(Path(r["dir"]) / "predictions.csv")
            per_seed.setdefault(r["seed"], []).append(p[(p.split == "val") & p.eval_nonoverlap.astype(bool)])
        vals = []
        for frames in per_seed.values():
            p = pd.concat(frames)
            y = (p.label == "single-child").to_numpy()
            if y.any() and not y.all():
                vals.append(roc_auc_score(y, p["p_single-child"]))
        out[cand] = float(np.mean(vals)) if vals else None
    return out


def _roc_figure(runs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    colors = {"b0": "grey", "b1": "tab:orange", "c1": "tab:green", "c2": "tab:purple"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, split, title in zip(axes, ("primary", "shift"), ("Primary test (2026-08-28)", "Shift test (Collection B)")):
        seen = set()
        for r in runs:
            p = pd.read_csv(Path(r["dir"]) / "predictions.csv")
            p = p[(p.split == split) & p.eval_nonoverlap.astype(bool)]
            y = (p.label == "single-child").to_numpy()
            if not (y.any() and not y.all()):
                continue
            fpr, tpr, _ = roc_curve(y, p["p_single-child"])
            ax.plot(fpr, tpr, color=colors[r["candidate"]], alpha=0.5, lw=1, label=None if r["candidate"] in seen else r["candidate"])
            seen.add(r["candidate"])
        ax.set(xlim=(0, 0.2), ylim=(0, 1.02), xlabel="false-alarm rate on negatives", ylabel="recall", title=title)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _per_session(runs):
    rows = [dict(candidate=r["candidate"], split=split, **s) for r in runs for split in ("primary", "shift") for s in r["sessions"][split]]
    df = pd.DataFrame(rows)
    df["recall"], df["fa_rate"] = _num(df.recall), _num(df.fa_rate)
    return df.groupby(["candidate", "split", "session_key"]).agg(
        runs=("recall", "size"), n_pos=("n_pos", "first"), n_neg=("n_neg", "first"),
        recall_mean=("recall", "mean"), fa_rate_mean=("fa_rate", "mean")).reset_index()


def build_report(cfg, run_ids, name, expected=None):
    safe_name(name)
    out = cfg.paths.results / name
    out.resolve().relative_to(cfg.paths.results.resolve())          # must stay under results/
    runs, failed = collect(cfg, run_ids)
    if not runs:
        raise ValueError("no complete runs found")
    check_runs(runs, set(matrix(cfg)) if expected is None else expected)
    out.mkdir(parents=True, exist_ok=True)
    summary, selection = summarize(runs), select_candidate(runs)
    per_session = _per_session(runs)
    pd.DataFrame(runs).drop(columns=["sessions", "confusion", "dir", "source_sha256", "config"]).to_csv(out / "runs.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    per_session.to_csv(out / "per_session.csv", index=False)
    (out / "selection.json").write_text(json.dumps(selection, indent=2, default=float))
    ident = {k: runs[0][k] for k in IDENTITY}
    (out / "manifest.json").write_text(json.dumps(dict(
        run_ids=list(run_ids), identity=ident, failed_or_incomplete=failed,
        accepted=[dict(run_id=r["run_id"], candidate=r["candidate"], val_date=r["val_date"], seed=r["seed"]) for r in runs]), indent=2, default=str))
    _roc_figure(runs, out / "roc.png")
    pooled, interior = pooled_val_auroc(runs), interior_summary(runs, cfg.windows.interior_s)
    chosen = selection["chosen"]
    main = [("recall", "recall"), ("false alarms", "fa_rate"), ("AUROC", "auroc"), ("pAUC ≤5%", "pauc05"),
            ("recall at 0% FA (oracle)", "recall_at_0.0"), ("at 0.1% (oracle)", "recall_at_0.001"),
            ("at 1% (oracle)", "recall_at_0.01"), ("at 5% (oracle)", "recall_at_0.05")]
    neg = [(n, f"fa_{n}") for n in NEGATIVES]
    row = lambda split: summary[(summary.candidate == chosen) & (summary.split == split)].iloc[0]
    p, s = row("primary"), row("shift")
    interior_rows = "\n".join(
        f"| {r.candidate} | {r.split} | {r.windows:.0f} | {_fmt(r.recall_mean, r.recall_std)} | {_fmt(r.fa_mean, r.fa_std)} |"
        for r in interior.itertuples())
    n_learned = sum(r["candidate"] in ("c1", "c2") for r in runs)
    text = f"""# Demo5 report: {name}

Runs read: {', '.join(run_ids)}. Complete runs used: {len(runs)} ({n_learned} neural, {len(runs) - n_learned} baseline); all {len(set(matrix(cfg)))} expected jobs are present.
Non-complete runs (kept on disk, listed here): {json.dumps(failed) if failed else 'none'}.
Identity shared by every run: cache `{ident['cache_id']}`, split hash `{ident['split_hash']}`, one source revision (see `manifest.json`).

Every number is a mean ± standard deviation over runs (three validation dates times three seeds for neural candidates, three dates for baselines).
Alarm rule: a window alarms when its smoothed child score is above the highest negative validation score (zero validation false alarms); the smoothing length was also picked on validation. Columns marked "(oracle)" set the threshold on the evaluated split's own scores, so they show the trade-off and are not deployable estimates.

## Selection (validation only)

Chosen learned candidate: **{chosen}**. Rule: {selection['rule']}.
{'' if 'gain' not in selection else f"Mean validation pAUC: c1 {selection['c1_mean']:.3f}, c2 {selection['c2_mean']:.3f}; gain of c2 {selection['gain']:+.3f} against a seed spread of {selection['seed_spread']:.3f}."}
The test tables below were not used for this choice.

## Headline: {chosen} at its validation-chosen operating point

| split | recall | false-alarm rate |
|---|---:|---:|
| primary test (2026-08-28) | {_fmt(p.recall_mean, p.recall_std)} | {_fmt(p.fa_rate_mean, p.fa_rate_std)} |
| shift test (Collection B) | {_fmt(s.recall_mean, s.recall_std)} | {_fmt(s.fa_rate_mean, s.fa_rate_std)} |
| change from primary to shift | {s.recall_mean - p.recall_mean:+.3f} | {s.fa_rate_mean - p.fa_rate_mean:+.3f} |

The change combines a different bandwidth (512 versus 256 subcarriers), sample rate, days, class mix and the subcarrier pooling. It has no single cause, and no adaptation was attempted.

## Validation (the three held-out dates)

{_table(summary, 'val', main)}

## Primary test (2026-08-28, two sessions)

{_table(summary, 'primary', main)}

False alarms by negative type on the primary test (the test has no lone-adult windows):

{_table(summary, 'primary', neg)}

Four-class confusion matrix for {chosen}, summed over its runs (rows: true class):

{_confusion(runs, chosen, 'primary')}

## Shift test (Collection B, 09-11 and 09-17, eight sessions)

{_table(summary, 'shift', main)}

False alarms by negative type on the shift test:

{_table(summary, 'shift', neg)}

Four-class confusion matrix for {chosen}, summed over its runs:

{_confusion(runs, chosen, 'shift')}

Per-session recall and false-alarm rates for both tests are in `per_session.csv`. ROC curves for every run are in `roc.png`.

## Interior windows only

Only windows at least {cfg.windows.interior_s:.0f} s from the edge of their annotation segment, and, when the run smooths over M windows, only if every window that enters the average is at least that far from an edge. The operating point is the run's own validation choice; the validation rows show how the threshold behaves without edge windows. The 0.1 Hz high-pass filter has a memory of about 20 s, so windows nearer an edge partly reflect the neighbouring segment (see `DATA_CONTRACT.md`). The headline above is unchanged and includes edge windows.

| candidate | split | mean windows kept | recall | false-alarm rate |
|---|---|---:|---:|---:|
{interior_rows}

## Pooled out-of-fold validation AUROC

{', '.join(f'{k}: ' + ('n/a' if v is None else f'{v:.3f}') for k, v in pooled.items())}. Scores come from different models, so this differs from the per-fold means above.

## How to read this

- The "child" is a dummy infant with simulated weak breathing. Nothing here describes real children.
- There are 11 child segments in the training pool, 2 in the primary test and 6 in the shift test. Windows within a segment are strongly correlated, so I report no p-values and no window-level confidence intervals.
- The primary test has two sessions from one day and no lone-adult windows; the shift test has many changes at once.
- Differences between candidates smaller than the standard deviation across runs are noise.
"""
    (out / "REPORT.md").write_text(text)
    return out
