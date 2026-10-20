import numpy as np
import pandas as pd
import pytest
from src.metrics import (threshold_for, recall_at, pauc, smooth, choose_operating_point, evaluate_split, finalize)


def test_zero_false_alarm_threshold_sits_at_highest_negative():
    neg = np.array([0.1, 0.2, 0.9])
    assert threshold_for(neg, 0) == 0.9
    pos = np.array([0.5, 0.95, 0.99])
    assert (pos > threshold_for(neg, 0)).mean() == pytest.approx(2 / 3)
    assert (neg > threshold_for(neg, 0)).sum() == 0


def test_recall_at_fa_allows_k_alarms():
    neg = np.linspace(0, 1, 100)          # 100 negatives
    y = np.r_[np.zeros(100), np.ones(10)]
    s = np.r_[neg, np.full(10, 2.0)]
    assert recall_at(y, s, 0.0) == 1.0
    assert recall_at(y, s, 0.05) == 1.0
    ys = np.r_[np.zeros(100), np.ones(2)]
    ss = np.r_[neg, [0.98, 2.0]]           # one positive hides inside the negatives
    assert recall_at(ys, ss, 0.0) == 0.5
    assert recall_at(ys, ss, 0.05) == 1.0


def test_pauc_perfect_and_uninformative():
    y = np.r_[np.zeros(200), np.ones(200)]
    assert pauc(y, np.r_[np.zeros(200), np.ones(200)]) == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    assert pauc(y, rng.random(400)) == pytest.approx(0.5, abs=0.1)
    assert np.isnan(pauc(np.zeros(5), np.arange(5.0)))


def test_smoothing_stays_inside_segments_and_is_trailing():
    df = pd.DataFrame(dict(segment_id=["a"] * 4 + ["b"] * 2, start=[0, 5, 10, 15, 0, 5]))
    out = smooth(df, np.array([0.0, 3.0, 6.0, 9.0, 100.0, 200.0]), 3)
    np.testing.assert_allclose(out, [0.0, 1.5, 3.0, 6.0, 100.0, 150.0])
    np.testing.assert_allclose(smooth(df, np.arange(6.0), 1), np.arange(6.0))


def _eval_df(n=60, seed=0):
    rng = np.random.default_rng(seed)
    labels = np.array(["empty", "single-adult", "single-child", "multiple"])
    lab = np.repeat(labels, n // 4)
    df = pd.DataFrame(dict(window_id=[f"w{i}" for i in range(n)], label=lab,
                           class4=[list(labels).index(x) for x in lab],
                           session_key=np.where(np.arange(n) % 2, "s1", "s2"),
                           segment_id=[f"{l}#{i // 5}" for i, l in enumerate(lab)],
                           start=(np.arange(n) % 5) * 5.0, edge_distance_s=(np.arange(n) % 5) * 5.0, date="2026-08-25", role="pool", eval_nonoverlap=True))
    probs = rng.dirichlet(np.ones(4), n)
    probs[lab == "single-child", 2] += 2
    return df, probs / probs.sum(1, keepdims=True)


def test_choose_operating_point_prefers_higher_recall_then_smaller_m():
    df, probs = _eval_df()
    m, thr = choose_operating_point(df, probs, (1, 3, 5))
    y = (df.label == "single-child").to_numpy()
    s = smooth(df, probs[:, 2], m)
    assert (s[~y] > thr).sum() == 0
    assert choose_operating_point(df, np.zeros((len(df), 4)), (1, 3, 5))[0] == 1


def test_never_alarm_baseline_has_zero_recall_zero_false_alarms_auroc_half():
    df, _ = _eval_df()
    zeros = np.zeros((len(df), 4))
    m, thr = choose_operating_point(df, zeros, (1, 3, 5))
    r = evaluate_split(df, zeros, m, thr)
    assert r["recall"] == 0 and r["fa_rate"] == 0 and r["auroc"] == pytest.approx(0.5)


def test_evaluate_split_counts_false_alarms_per_negative_type():
    df, probs = _eval_df()
    r = evaluate_split(df, probs, 1, thr=-1.0)          # everything alarms
    assert r["recall"] == 1 and r["fa_rate"] == 1
    assert set(r["fa_by_negative"]) == {"empty", "single-adult", "multiple"}
    assert sum(x["n"] for x in r["per_session"]) == len(df)
    assert np.array(r["confusion_matrix"]).sum() == len(df)


def test_finalize_uses_validation_only_for_the_threshold():
    df, probs = _eval_df()
    idx = df.copy()
    idx["role"] = ["pool"] * 30 + ["primary"] * 15 + ["shift"] * 15
    parts = {"val": (np.arange(0, 30), probs[:30]), "primary": (np.arange(30, 45), probs[30:45]),
             "shift": (np.arange(45, 60), probs[45:60])}
    a, pred = finalize(idx, parts)
    b, _ = finalize(idx, dict(parts, primary=(np.arange(30, 45), 1 - probs[30:45])))   # change test scores only
    assert a["threshold"] == b["threshold"] and a["M"] == b["M"]
    assert a["val"] == b["val"] and a["primary"] != b["primary"]
    assert set(pred.split) == {"val", "primary", "shift"} and len(pred) == 60


def test_rolling_minimum_uses_the_rows_that_actually_enter_the_average():
    df = pd.DataFrame(dict(segment_id=["a"] * 3, start=[0, 5, 30]))          # windows 10 to 25 s were rejected
    out = smooth(df, np.array([0.0, 5.0, 30.0]), 3, "min")
    np.testing.assert_allclose(out, [0.0, 0.0, 0.0])                        # the 30 s row still averages the two edge rows
    contiguous = pd.DataFrame(dict(segment_id=["a"] * 4, start=[0, 5, 10, 15]))
    np.testing.assert_allclose(smooth(contiguous, np.array([0.0, 5.0, 10.0, 15.0]), 3, "min"), [0.0, 0.0, 0.0, 5.0])
