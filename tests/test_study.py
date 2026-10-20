import dataclasses
import json
import pytest
from config import Config, Paths
from src.study import gpu_room_error, job_list, matrix, plan


def cfg_at(tmp_path):
    return dataclasses.replace(Config(), paths=Paths(root=tmp_path))


def test_job_list_covers_every_candidate_date_and_seed(tmp_path):
    jobs = job_list(cfg_at(tmp_path), "s")
    assert len(jobs) == 2 * 3 * 3 + 2 * 3
    assert {(j["candidate"], j["seed"]) for j in jobs if j["candidate"] in ("b0", "b1")} == {("b0", 0), ("b1", 0)}
    assert len({str(j["dir"]) for j in jobs}) == len(jobs)


def test_plan_skips_complete_and_refuses_running_or_failed(tmp_path):
    cfg = cfg_at(tmp_path)
    jobs = job_list(cfg, "s")
    jobs[0]["dir"].mkdir(parents=True)
    (jobs[0]["dir"] / "status.json").write_text(json.dumps({"state": "complete"}))
    assert len(plan(cfg, "s")) == len(jobs) - 1
    jobs[1]["dir"].mkdir(parents=True)
    (jobs[1]["dir"] / "status.json").write_text(json.dumps({"state": "running"}))
    with pytest.raises(RuntimeError, match="running"):
        plan(cfg, "s")


def test_matrix_and_job_list_agree(tmp_path):
    cfg = cfg_at(tmp_path)
    assert {(j["candidate"], j["val_date"], j["seed"]) for j in job_list(cfg, "s")} == set(matrix(cfg))


def test_gpu_preflight_refuses_unsafe_concurrency():
    assert gpu_room_error({"cuda:0": 21_000.0, "cuda:1": 21_500.0}, 3) is None
    msg = gpu_room_error({"cuda:0": 21_000.0, "cuda:1": 8_000.0}, 3)
    assert msg and "cuda:1" in msg and "--per-gpu 1" in msg
    assert gpu_room_error({"cuda:0": 22_000.0}, 9) is not None      # the nine-job launch that ran out of memory


def test_launch_rejects_unsafe_run_names(tmp_path):
    from src.study import launch
    with pytest.raises(ValueError):
        launch(cfg_at(tmp_path), "../x", ["cuda:0"], 1)


def _source_root(tmp_path):
    (tmp_path / "src").mkdir()
    for name in ("config.py", "main.py", "src/a.py", "src/b.py"):
        (tmp_path / name).write_text(f"# {name}\n")
    return cfg_at(tmp_path)


def test_archive_is_written_once_and_verified_on_restart(tmp_path):
    import tarfile
    from src.study import archive_source
    cfg = _source_root(tmp_path)
    archive_source(cfg, "s")
    with tarfile.open(tmp_path / "runs/s/source.tar.gz") as tar:
        assert sorted(tar.getnames()) == ["config.py", "main.py", "src/a.py", "src/b.py"]
    assert not list((tmp_path / "runs/s").glob("*.partial"))
    archive_source(cfg, "s")                                             # unchanged source: fine
    (tmp_path / "src/a.py").write_text("# changed\n")
    with pytest.raises(RuntimeError, match="does not match"):
        archive_source(cfg, "s")                                         # stale archive for a resumed run id


def test_partial_archive_is_refused(tmp_path):
    from src.study import archive_source
    cfg = _source_root(tmp_path)
    (tmp_path / "runs/s").mkdir(parents=True)
    (tmp_path / "runs/s/source.tar.gz").write_bytes(b"not a tar file")
    with pytest.raises(RuntimeError, match="unreadable"):
        archive_source(cfg, "s")


def test_candidate_filter_limits_the_matrix(tmp_path):
    cfg = cfg_at(tmp_path)
    only = matrix(cfg, ["c2"])
    assert len(only) == 9 and {c for c, _, _ in only} == {"c2"}
    assert {(j["candidate"]) for j in job_list(cfg, "s", ["c2"])} == {"c2"}
    assert len(matrix(cfg)) == 24


def test_job_list_follows_the_fold_keys_of_a_design(tmp_path):
    cfg = cfg_at(tmp_path)
    jobs = job_list(cfg, "s", candidates=("c1", "b1"), keys=("k1", "k2"), design="loso")
    assert len(jobs) == 2 * 3 + 2
    assert {j["dir"].parent.name for j in jobs} == {"test_k1", "test_k2"}
    assert {j["dir"].parent.name for j in job_list(cfg, "s", design="both_0911")} == {f"val_{d}" for d in cfg.roles.pool}
