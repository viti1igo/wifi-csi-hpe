"""Session discovery, annotation segments and the window index."""
import json
import numpy as np
import pandas as pd

CLASSES = ("empty", "single-adult", "single-child", "multiple")


def discover_sessions(paths):
    """One dict per session folder (any folder holding meta.json and timestamps.csv)."""
    out = []
    for collection, root in (("A", paths.collection_a), ("B", paths.collection_b)):
        for meta in sorted(root.rglob("meta.json")):
            d = meta.parent
            if not (d / "timestamps.csv").is_file():
                continue
            first = d.relative_to(root).parts[0]
            date = "2026-09-01" if first == "handoff_0901" else first
            ann = d / "annotations_v2.json"
            out.append(dict(key=f"{collection}:{date}/{d.name}", collection=collection, date=date,
                            session=d.name, path=d, annotation=ann if ann.is_file() else None))
    return out


def load_segments(sessions):
    """All annotation segments, with epoch bounds and the child type."""
    rows = []
    for s in sessions:
        if s["annotation"] is None:
            continue
        for i, a in enumerate(json.loads(s["annotation"].read_text())["annotations"]):
            rows.append(dict(segment_id=f"{s['key']}#{i}", session_key=s["key"], collection=s["collection"],
                             date=s["date"], session=s["session"], label=a["label"], start=float(a["start"]),
                             end=float(a["end"]), child_type=a.get("child_type")))
    seg = pd.DataFrame(rows)
    seg["duration"] = (seg.end - seg.start).clip(lower=0)
    return seg


def build_windows(cfg, seg):
    """5 s windows fully inside one stable segment. Drops transition and unknown-child segments (D4)."""
    w, roles = cfg.windows, cfg.roles
    role_of = {d: "pool" for d in roles.pool} | {d: "primary" for d in roles.primary_test} | {d: "shift" for d in roles.shift_test}
    keep = seg[seg.label.isin(CLASSES) & (seg.child_type != "unknown") & seg.date.isin(role_of)]
    parts = []
    for r in keep.itertuples():
        n = int(np.floor((r.duration - w.length_s) / w.stride_s + 1e-9)) + 1
        if n < 1:
            continue
        k = np.arange(n)
        parts.append(pd.DataFrame(dict(
            window_id=[f"{r.segment_id}@{j}" for j in k], segment_id=r.segment_id, session_key=r.session_key,
            collection=r.collection, date=r.date, label=r.label, class4=CLASSES.index(r.label),
            start=r.start + k * w.stride_s, role=role_of[r.date],
            edge_distance_s=np.minimum(k * w.stride_s, r.duration - k * w.stride_s - w.length_s),
            eval_nonoverlap=(k * w.stride_s) % w.eval_stride_s == 0)))
    win = pd.concat(parts, ignore_index=True)
    win["end"] = win.start + w.length_s
    return win
