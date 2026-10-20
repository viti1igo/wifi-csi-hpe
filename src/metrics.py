"""Metrics and the alarm operating point for the child-alone detector.

Alarm rule: a window raises an alarm when its (smoothed) child score is strictly above the threshold.
The threshold and the smoothing length M come from validation windows only.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

NEGATIVES = ("empty", "single-adult", "multiple")
FA_LEVELS = (0.0, 0.001, 0.01, 0.05)
SMOOTHING = (1, 3, 5)
PRED_COLUMNS = ["window_id", "session_key", "segment_id", "date", "role", "label", "start", "edge_distance_s", "eval_nonoverlap"]


def _num(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)


def threshold_for(neg_scores, k):
    """Score such that at most k negatives lie strictly above it. k = 0 gives the highest negative score."""
    neg = np.sort(np.asarray(neg_scores, float))[::-1]
    return -np.inf if k >= len(neg) else float(neg[k])


def recall_at(y, score, fa):
    y, s = np.asarray(y, bool), np.asarray(score, float)
    if not y.any():
        return float("nan")
    thr = threshold_for(s[~y], int(np.floor(fa * (~y).sum() + 1e-9)))
    return float((s[y] > thr).mean())


def pauc(y, score, max_fpr=0.05):
    """Standardised partial AUC over false-alarm rates up to max_fpr (0.5 = chance, 1 = perfect)."""
    y = np.asarray(y)
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, score, max_fpr=max_fpr))


def smooth(df, score, M, how="mean"):
    """Trailing mean (or minimum) of the last M values inside each segment, in time order."""
    score = np.asarray(score, float)
    if M == 1:
        return score
    seg, start = df.segment_id.to_numpy(), df.start.to_numpy()
    order = np.lexsort((start, seg))
    tmp = pd.DataFrame({"seg": seg[order], "s": score[order]})
    out = np.empty_like(score)
    out[order] = tmp.groupby("seg", sort=False).s.transform(lambda x: getattr(x.rolling(M, min_periods=1), how)()).to_numpy()
    return out


def choose_operating_point(df, probs, Ms=SMOOTHING):
    """M with the highest validation recall at zero validation false alarms. Ties go to smaller M."""
    y = (df.label == "single-child").to_numpy()
    best = None
    for M in Ms:
        s = smooth(df, probs[:, 2], M)
        thr = threshold_for(s[~y], 0)
        recall = float((s[y] > thr).mean()) if y.any() else 0.0
        if best is None or recall > best[0]:
            best = (recall, M, thr)
    return best[1], best[2]


def evaluate_split(df, probs, M, thr):
    y = (df.label == "single-child").to_numpy()
    score = smooth(df, probs[:, 2], M)
    alarm = score > thr

    sessions = []
    for key, g in df.reset_index(drop=True).groupby("session_key"):
        m = g.index.to_numpy()
        sessions.append(dict(session_key=key, n=len(m), n_pos=int(y[m].sum()), n_neg=int((~y[m]).sum()),
                             recall=rate_of(alarm[m], y[m]), fa_rate=rate_of(alarm[m], ~y[m])))
    both = y.any() and (~y).any()
    return dict(
        n=len(df), n_pos=int(y.sum()), n_neg=int((~y).sum()), M=M, threshold=float(thr),
        recall=rate_of(alarm, y), fa_rate=rate_of(alarm, ~y),
        fa_by_negative={k: rate_of(alarm, (df.label == k).to_numpy()) for k in NEGATIVES},
        alarm_fraction=float(alarm.mean()),
        auroc=_num(roc_auc_score(y, score)) if both else None,
        ap=_num(average_precision_score(y, score)) if both else None,
        pauc05=_num(pauc(y, score)),
        recall_at={str(f): _num(recall_at(y, score, f)) for f in FA_LEVELS},
        per_session=sessions,
        confusion_matrix=confusion_matrix(df.class4.to_numpy(), probs.argmax(1), labels=range(4)).tolist())


def rate_of(alarm, mask):
    return _num(alarm[mask].mean()) if mask.any() else None


def finalize(index, splits, Ms=SMOOTHING):
    """splits: name -> (row positions in `index`, probabilities (n, 4)). Must contain 'val'.

    M and the threshold come from the val windows; every split is then scored with them.
    Metrics use the non-overlapping windows. Returns (metrics dict, predictions frame for every window).
    """
    rows, probs = splits["val"]
    v = index.iloc[rows]
    keep = v.eval_nonoverlap.to_numpy()
    M, thr = choose_operating_point(v[keep], probs[keep], Ms)
    metrics, frames = {"M": M, "threshold": thr}, []
    for name, (rows, p) in splits.items():
        d = index.iloc[rows]
        keep = d.eval_nonoverlap.to_numpy()
        metrics[name] = evaluate_split(d[keep], p[keep], M, thr)
        f = d[PRED_COLUMNS].reset_index(drop=True)
        f.insert(0, "split", name)
        for c, label in enumerate(("empty", "single-adult", "single-child", "multiple")):
            f[f"p_{label}"] = p[:, c]
        frames.append(f)
    return metrics, pd.concat(frames, ignore_index=True)
