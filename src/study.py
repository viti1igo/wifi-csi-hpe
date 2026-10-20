"""Launch the full study: many `main.py train` jobs, several at once per device."""
import hashlib
import json
import os
import subprocess
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor
from itertools import cycle

from config import safe_name

GPU_CANDIDATES = ("c1", "c2")
CPU_CANDIDATES = ("b0", "b1")          # deterministic: one seed is enough
# Worst peak seen for one C1 job while nine shared a card (study1): 5.5 GB. Alone it needs 2.5 to 2.7 GB.
PEAK_MIB = 5632


def matrix(cfg, candidates=None, keys=None):
    """Every (candidate, fold, seed) the study must contain. Folds are the pool dates unless `keys` names others."""
    keys = cfg.roles.pool if keys is None else keys
    jobs = ([(c, d, s) for c in GPU_CANDIDATES for d in keys for s in cfg.train.seeds]
            + [(c, d, 0) for c in CPU_CANDIDATES for d in keys])
    return [j for j in jobs if candidates is None or j[0] in candidates]


def run_dir(cfg, run_id, candidate, fold, seed, design="lodo"):
    """Folder of one job. A loso fold is named by its test session, every other fold by its validation date."""
    return cfg.paths.runs / run_id / candidate / f"{'test' if design == 'loso' else 'val'}_{fold}" / f"seed_{seed}"


def job_list(cfg, run_id, candidates=None, keys=None, design="lodo"):
    return [dict(candidate=c, val_date=d, seed=s, dir=run_dir(cfg, run_id, c, d, s, design))
            for c, d, s in matrix(cfg, candidates, keys)]


def plan(cfg, run_id, candidates=None, keys=None, design="lodo"):
    """Jobs still to run. A complete run is skipped; a running or failed one stops the study."""
    todo = []
    for j in job_list(cfg, run_id, candidates, keys, design):
        status = j["dir"] / "status.json"
        if not j["dir"].exists():
            todo.append(j)
            continue
        state = json.loads(status.read_text()).get("state") if status.is_file() else "unknown"
        if state != "complete":
            raise RuntimeError(f"{j['dir']} is {state}; pick a new run id (failed runs are kept, never overwritten)")
    return todo


def gpu_room_error(free_mib, per_gpu, peak_mib=PEAK_MIB):
    """None if `per_gpu` jobs fit in every card's free memory, else a message with the largest safe number."""
    for device, free in free_mib.items():
        if per_gpu * peak_mib > free:
            return f"{device} has {free:.0f} MiB free; {per_gpu} jobs x {peak_mib} MiB do not fit. Use --per-gpu {int(free // peak_mib)} or fewer."
    return None


def preflight(gpus, per_gpu):
    import torch
    free = {}
    for device in gpus:
        index = int(device.split(":")[1])
        if not torch.cuda.is_available() or index >= torch.cuda.device_count():
            raise RuntimeError(f"{device} is not available")
        free[device] = torch.cuda.mem_get_info(index)[0] / 2**20
    error = gpu_room_error(free, per_gpu)
    if error:
        raise RuntimeError(error)
    print("PREFLIGHT " + ", ".join(f"{d}: {f:.0f} MiB free" for d, f in free.items()), flush=True)


def _source_files(cfg):
    root = cfg.paths.root
    return {str(f.relative_to(root)): f for f in [root / "config.py", root / "main.py", *sorted((root / "src").glob("*.py"))]}


def archive_source(cfg, run_id):
    """Keep the exact code of the study next to its results. An existing archive must match the current source."""
    archive = cfg.paths.runs / run_id / "source.tar.gz"
    files = _source_files(cfg)
    current = {name: hashlib.sha256(f.read_bytes()).hexdigest() for name, f in files.items()}
    if archive.exists():
        try:
            with tarfile.open(archive) as tar:
                stored = {m.name: hashlib.sha256(tar.extractfile(m).read()).hexdigest() for m in tar.getmembers()}
        except (tarfile.TarError, EOFError, OSError) as error:
            raise RuntimeError(f"{archive} is unreadable ({error}); pick a new run id") from error
        if stored != current:
            raise RuntimeError(f"{archive} does not match the current source; pick a new run id")
        return
    archive.parent.mkdir(parents=True, exist_ok=True)
    partial = archive.with_name("source.tar.gz.partial")
    with tarfile.open(partial, "w:gz") as tar:
        for name, f in files.items():
            tar.add(f, arcname=name)
    os.replace(partial, archive)                    # atomic: a killed run never leaves a half archive under the final name


def launch(cfg, run_id, gpus, per_gpu, cache_id=None, cpu_parallel=2, candidates=None, extra=(), design="lodo"):
    from src.train import check_sources_current, find_cache
    safe_name(run_id)
    directory = find_cache(cfg, cache_id)
    keys = None
    if design != "lodo":
        import pandas as pd
        from src.splits import design_folds
        keys = sorted(design_folds(cfg, pd.read_csv(directory / "index.csv"), design))
        extra = (*extra, "--design", design)
    todo = plan(cfg, run_id, candidates, keys, design)
    check_sources_current(cfg, directory)
    preflight(gpus, per_gpu)
    archive_source(cfg, run_id)
    logs = cfg.paths.root / "logs" / run_id
    logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="2")
    print(f"STUDY {run_id} ({design}): {len(todo)} jobs, cache {directory.name}, {per_gpu} at once on each of {list(gpus)}", flush=True)

    def go(job, device):
        name = f"{job['candidate']}_{job['val_date']}_seed{job['seed']}"
        cmd = [sys.executable, "main.py", "train", "--candidate", job["candidate"], "--val-date", job["val_date"],
               "--seed", str(job["seed"]), "--run", run_id, "--device", device, "--cache", directory.name, *extra]
        with open(logs / f"{name}.log", "w") as log:
            code = subprocess.run(cmd, cwd=cfg.paths.root, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        print(f"{'DONE' if code == 0 else 'FAILED'} {name} on {device}", flush=True)
        return name, code

    pools = {g: ThreadPoolExecutor(per_gpu) for g in gpus} | {"cpu": ThreadPoolExecutor(cpu_parallel)}
    devices = cycle(gpus)
    futures = []
    for job in todo:
        device = "cpu" if job["candidate"] in CPU_CANDIDATES else next(devices)
        futures.append(pools[device].submit(go, job, device))
    results = [f.result() for f in futures]
    for p in pools.values():
        p.shutdown()
    failed = [n for n, c in results if c]
    print(f"STUDY {run_id} finished: {len(results) - len(failed)} ok, {len(failed)} failed {failed}", flush=True)
    return failed
