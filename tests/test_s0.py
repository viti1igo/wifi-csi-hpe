import subprocess, sys, importlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_import_has_no_side_effects(tmp_path):
    before = sorted(p.name for p in ROOT.iterdir())
    for name in ("config", "main"):
        importlib.import_module(name)
    assert sorted(p.name for p in ROOT.iterdir()) == before


def test_help_lists_check():
    out = subprocess.run([sys.executable, "main.py", "--help"], cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0 and "check" in out.stdout


def test_check_reports_environment():
    out = subprocess.run([sys.executable, "main.py", "check"], cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0
    for key in ("python", "torch", "numpy", "cuda_available", "CUDA_VISIBLE_DEVICES"):
        assert key in out.stdout
