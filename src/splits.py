"""Leave-one-date-out folds over the pool, plus the fixed primary and shift tests."""
import hashlib
import json


def make_folds(cfg, win):
    """Return {validation_date: {train, val, primary, shift}} as row-index arrays of `win`."""
    idx = win.index
    primary, shift = idx[win.role == "primary"], idx[win.role == "shift"]
    folds = {}
    for date in cfg.roles.pool:
        pool = win.role == "pool"
        folds[date] = dict(train=idx[pool & (win.date != date)], val=idx[pool & (win.date == date)],
                           primary=primary, shift=shift)
    return folds


DESIGNS = ("lodo", "both_0911", "both_0917", "loso")


def design_folds(cfg, win, design="lodo"):
    """Folds for one split design. `lodo` is the study design; the others are thesis analyses (DECISIONS D17, D18).

    both_0911 / both_0917: as lodo, but one Collection B date joins every training set and the shift test keeps the other.
    loso: each pool session is the test once; validation is the next pool date after the session's date whose removal
    still leaves all four classes in training; training is the rest of the pool, so other sessions of the same day are seen.
    """
    if design not in DESIGNS:
        raise ValueError(f"design must be one of {DESIGNS}, got {design!r}")
    folds = make_folds(cfg, win)
    if design == "lodo":
        return folds
    idx = win.index
    if design.startswith("both_"):
        moved = f"2026-{design[5:7]}-{design[7:]}"
        extra, kept = idx[win.date == moved], idx[(win.role == "shift") & (win.date != moved)]
        return {d: dict(f, train=f["train"].append(extra), shift=kept) for d, f in folds.items()}
    pool_dates, out = list(cfg.roles.pool), {}
    for session in sorted(set(win.loc[win.role == "pool", "session_key"])):
        rows = win.session_key == session
        date = win.loc[rows, "date"].iloc[0]
        pool = win.role == "pool"
        start = pool_dates.index(date)
        val_date = next(d for d in (pool_dates[(start + k) % len(pool_dates)] for k in range(1, len(pool_dates)))
                        if win.loc[pool & ~rows & (win.date != d), "label"].nunique() == 4)
        key = session.split(":", 1)[1].replace("/", "_")
        out[key] = dict(train=idx[pool & ~rows & (win.date != val_date)], val=idx[pool & (win.date == val_date)],
                        test=idx[rows], primary=folds[val_date]["primary"], shift=folds[val_date]["shift"])
    return out


def check_no_leakage(win, fold):
    """No session and no segment may appear in two of train, val, primary, shift."""
    for col in ("session_key", "segment_id"):
        seen = {}
        for name, rows in fold.items():
            for value in set(win.loc[rows, col]):
                assert value not in seen, f"{col} {value} is in both {seen[value]} and {name}"
                seen[value] = name


def splits_hash(win, folds):
    payload = {d: {k: sorted(win.loc[v, "window_id"]) for k, v in f.items()} for d, f in sorted(folds.items())}
    return hashlib.sha256(json.dumps(payload).encode()).hexdigest()[:16]
