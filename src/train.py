"""Training loop and the run driver (one candidate, one validation date, one seed)."""
import dataclasses
import hashlib
import json
import os
import random
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from config import safe_name
from src.baselines import fit_b1, never_alarm
from src.cache import check_cache_current, load_cache
from src.index import build_windows, discover_sessions, load_segments
from src.metrics import finalize, pauc
from src.model import MODELS, parameter_count
from src.splits import check_no_leakage, design_folds, splits_hash
from src.study import run_dir as study_run_dir

BASELINES = ("b0", "b1")
OVERRIDABLE = {"encoder_lr", "head_lr", "patience"}          # training settings an experiment may change


def seed_everything(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.enable_flash_sdp(False)              # the fast attention kernels have non-deterministic backward;
    torch.backends.cuda.enable_mem_efficient_sdp(False)      # 50 tokens make the math kernel cheap


def class_weights(y):
    """Inverse class frequency from the training labels, averaging 1."""
    counts = np.bincount(y, minlength=4)
    if (counts == 0).any():
        raise ValueError("training data lacks a class")
    return torch.tensor(len(y) / (4 * counts), dtype=torch.float32)


def augment(x, gen, t):
    """Random link order per window, random bin dropout, small Gaussian jitter."""
    b, lags, links, bins = x.shape
    if t.link_permutation:
        order = torch.rand(b, links, device=x.device, generator=gen).argsort(1)
        x = torch.gather(x, 2, order[:, None, :, None].expand(-1, lags, -1, bins))
    keep = (torch.rand(b, 1, 1, bins, device=x.device, generator=gen) > t.bin_dropout).to(x.dtype)
    return x * keep + t.jitter * torch.randn(x.shape, device=x.device, generator=gen)


@torch.no_grad()
def predict(model, X, device, batch=512):
    model.eval()
    out = [F.softmax(model(torch.from_numpy(X[i:i + batch]).to(device)), 1).cpu() for i in range(0, len(X), batch)]
    return torch.cat(out).numpy().astype(np.float64)


def fit(model, X, y, val_fn, t, device, seed, ckpt, log=lambda record: None):
    """Train with the predeclared rule: keep the epoch with the highest validation criterion (first wins ties)."""
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    Xt, yt = torch.from_numpy(X).to(device), torch.from_numpy(y).long().to(device)
    w = class_weights(y).to(device)
    head = list(model.head.parameters())                 # everything else is the encoder
    encoder = [p for p in model.parameters() if id(p) not in {id(q) for q in head}]
    opt = torch.optim.AdamW([{"params": encoder, "lr": t.encoder_lr}, {"params": head, "lr": t.head_lr}],
                            weight_decay=t.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t.epochs, eta_min=t.lr_min)
    best, best_epoch, stale, history = -np.inf, -1, 0, []
    for epoch in range(1, t.epochs + 1):
        model.train()
        perm = torch.randperm(len(Xt), generator=gen, device=device)
        total = 0.0
        for i in range(0, len(perm), t.batch_size):
            idx = perm[i:i + t.batch_size]
            if len(idx) < 2:                        # batch norm needs more than one sample
                continue
            loss = F.cross_entropy(model(augment(Xt[idx], gen, t)), yt[idx], weight=w)
            if not torch.isfinite(loss):
                raise ValueError("non-finite training loss")
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), t.grad_clip)
            opt.step()
            total += loss.item() * len(idx)
        sched.step()
        criterion = val_fn(model)
        record = dict(epoch=epoch, train_loss=total / len(Xt), criterion=None if np.isnan(criterion) else float(criterion),
                      lr=[g["lr"] for g in opt.param_groups])
        history.append(record)
        log(record)
        score = -np.inf if np.isnan(criterion) else criterion
        if best_epoch < 0 or score > best:
            best, best_epoch, stale = score, epoch, 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "criterion": record["criterion"]}, ckpt)
        else:
            stale += 1
        if stale >= t.patience:
            break
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True)["model"])
    return dict(history=history, best_epoch=best_epoch, best_criterion=None if best == -np.inf else float(best),
                epochs_run=len(history))


def subset_rows(y, frac, seed):
    """Deterministic class-balanced subset for smoke runs."""
    rng = np.random.default_rng(seed)
    keep = [rng.choice(np.flatnonzero(y == c), max(2, int(round(frac * (y == c).sum()))), replace=False) for c in range(4)]
    return np.sort(np.concatenate(keep))


def _json(value):
    if isinstance(value, dict):
        return {k: _json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(_json(value), indent=2, allow_nan=False, default=str))


def find_cache(cfg, cache_id=None):
    if cache_id:
        return cfg.paths.cache / cache_id
    found = [d for d in cfg.paths.cache.iterdir() if (d / "manifest.json").is_file()] if cfg.paths.cache.is_dir() else []
    if len(found) != 1:
        raise ValueError(f"expected exactly one cache in {cfg.paths.cache}, found {len(found)}; pass --cache")
    return found[0]


def check_device(name):
    """Fail early and print what will be used. `cpu` or `cuda:N` only."""
    if name == "cpu":
        return torch.device("cpu"), None
    if not re.fullmatch(r"cuda:\d+", name):
        raise ValueError(f"device must be 'cpu' or 'cuda:N', got {name!r}")
    index = int(name.split(":")[1])
    if not torch.cuda.is_available() or index >= torch.cuda.device_count():
        raise ValueError(f"{name} is not available (visible devices: {torch.cuda.device_count()})")
    dev = torch.device(name)
    torch.zeros(1, device=dev)                      # a small allocation before loading data
    free, total = torch.cuda.mem_get_info(dev)
    gpu = torch.cuda.get_device_name(dev)
    print(f"DEVICE {name} {gpu} free {free / 2**30:.1f} of {total / 2**30:.1f} GiB "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')!r}", flush=True)
    return dev, gpu


def check_sources_current(cfg, directory):
    """Deep check: the set of eligible sessions equals the cache's, and the raw session files are unchanged."""
    directory = Path(directory)
    old = set(json.loads((directory / "manifest.json").read_text())["sources"])
    now = set(build_windows(cfg, load_segments(discover_sessions(cfg.paths))).session_key)
    if now != old:
        raise ValueError(f"eligible sessions changed since the cache was built (added {sorted(now - old)}, removed {sorted(old - now)}); rebuild it")
    check_cache_current(cfg, directory, deep=True)


def source_hashes():
    root = Path(__file__).resolve().parent.parent          # the code, not the data root
    files = [root / "config.py", root / "main.py", *sorted((root / "src").glob("*.py"))]
    return {str(f.relative_to(root)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}


def run(cfg, candidate, val_date, seed, run_id, device="cpu", epochs=None, subset=None, cache_id=None, verify_sources=False, overrides=None,
        design="lodo"):
    """Train (or fit) one model, pick the operating point on validation, score every split, write artifacts."""
    if candidate not in BASELINES + tuple(MODELS):
        raise ValueError(f"unknown candidate {candidate!r}")
    safe_name(run_id)
    if design != "loso" and val_date not in cfg.roles.pool:          # a loso fold is a session, checked once the index is loaded
        raise ValueError(f"validation date must be one of {cfg.roles.pool}")
    safe_name(val_date)
    dev, gpu = check_device(device)
    run_dir = study_run_dir(cfg, run_id, candidate, val_date, seed, design)
    run_dir.mkdir(parents=True)                      # raises FileExistsError: failed runs are never overwritten
    status = lambda state, **extra: write_json(run_dir / "status.json", dict(state=state, **extra))
    status("running")
    started = time.time()
    try:
        directory = find_cache(cfg, cache_id)
        X, index, manifest = load_cache(directory)
        check_sources_current(cfg, directory) if verify_sources else check_cache_current(cfg, directory)
        folds = design_folds(cfg, index, design)
        if val_date not in folds:
            raise ValueError(f"fold {val_date!r} is not in design {design!r}: {sorted(folds)}")
        fold = folds[val_date]
        check_no_leakage(index, fold)
        t = dataclasses.replace(cfg.train, epochs=epochs) if epochs else cfg.train
        if overrides:
            if not set(overrides) <= OVERRIDABLE:
                raise ValueError(f"only {sorted(OVERRIDABLE)} can be overridden, got {sorted(overrides)}")
            t = dataclasses.replace(t, **overrides)
        y = index.class4.to_numpy()
        train_rows = fold["train"].to_numpy()
        if subset:
            train_rows = train_rows[subset_rows(y[train_rows], subset, seed)]
        write_json(run_dir / "run.json", dict(
            candidate=candidate, design=design, val_date=val_date, seed=seed, device=device, gpu=gpu, subset=subset,
            command=sys.argv, sources_verified=verify_sources, python=sys.version.split()[0], torch=torch.__version__, numpy=np.__version__,
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), cache_id=manifest["cache_id"],
            split_hash=splits_hash(index, folds), train_windows=len(train_rows), config=dataclasses.asdict(cfg.train) | dataclasses.asdict(t),
            source_sha256=source_hashes(), deterministic="cudnn deterministic, use_deterministic_algorithms(warn_only)"))
        info = {}
        if candidate == "b0":
            predict_fn = never_alarm
        elif candidate == "b1":
            predict_fn = fit_b1(X[train_rows], y[train_rows])
        else:
            seed_everything(seed)
            model = MODELS[candidate](dropout=t.dropout).to(dev)
            val_rows = fold["val"].to_numpy()[index.eval_nonoverlap.to_numpy()[fold["val"]]]
            val_y = (index.label.to_numpy()[val_rows] == "single-child").astype(int)
            val_X = X[val_rows]
            val_fn = lambda m: pauc(val_y, predict(m, val_X, dev)[:, 2])
            log = lambda r: print(f"EPOCH {r['epoch']} loss {r['train_loss']:.4f} val_pauc {r['criterion']}", flush=True)
            info = fit(model, X[train_rows], y[train_rows], val_fn, t, dev, seed, run_dir / "best.pt", log)
            info["parameters"] = parameter_count(model)
            torch.save({"model": model.state_dict()}, run_dir / "final.pt")
            (run_dir / "epoch_metrics.jsonl").write_text("\n".join(json.dumps(_json(r)) for r in info.pop("history")))
            if dev.type == "cuda":
                info["peak_gpu_memory_mib"] = torch.cuda.max_memory_allocated(dev) / 2**20
                info["peak_gpu_reserved_mib"] = torch.cuda.max_memory_reserved(dev) / 2**20
            predict_fn = lambda Z: predict(model, Z, dev)
        splits = {name: (fold[key].to_numpy(), predict_fn(X[fold[key].to_numpy()]))
                  for name, key in (("val", "val"), ("primary", "primary"), ("shift", "shift"), ("test", "test")) if key in fold}
        metrics, predictions = finalize(index, splits)
        write_json(run_dir / "metrics.json", metrics)
        predictions.to_csv(run_dir / "predictions.csv", index=False)
        status("complete", seconds=round(time.time() - started, 1), **_json(info))
    except BaseException:
        status("failed", error=traceback.format_exc(), seconds=round(time.time() - started, 1))
        raise
    return run_dir
