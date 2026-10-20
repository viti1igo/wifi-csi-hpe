"""Demo5 settings. One place for every value; nothing runs on import."""
import re
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def safe_name(value):
    """Run ids, report names: letters, digits, '_' and '-' only, so a path can never leave its folder."""
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError(f"name may contain letters, digits, '_' and '-' only, got {value!r}")
    return value


@dataclass(frozen=True)
class Paths:
    root: Path = ROOT
    dataset: Path = ROOT.parent / "MTK_dataset"

    @property
    def collection_a(self) -> Path:
        return self.dataset / "CSI_handoff_20260903_123743" / "data"

    @property
    def collection_b(self) -> Path:
        return self.dataset / "OneDrive_2026-09-29" / "5G 80 MHz"

    @property
    def cache(self) -> Path:
        return self.root / "cache"

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    @property
    def results(self) -> Path:
        return self.root / "results"


@dataclass(frozen=True)
class Roles:
    """Dates per role (decision D3)."""
    pool: tuple = ("2026-08-25", "2026-08-26", "2026-09-01")
    primary_test: tuple = ("2026-08-28",)
    shift_test: tuple = ("2026-09-11", "2026-09-17")


@dataclass(frozen=True)
class Windows:
    length_s: float = 5.0
    stride_s: float = 1.0
    eval_stride_s: float = 5.0  # reported metrics use non-overlapping windows
    interior_s: float = 20.0    # robustness tables keep windows at least this far from a segment edge


@dataclass(frozen=True)
class DSP:
    grid_hz: float = 50.0
    out_hz: float = 10.0
    bins: int = 64
    band_hz: tuple = (0.1, 2.0)
    filter_order: int = 4
    hampel_size: int = 51      # samples at grid_hz, about 1 s
    hampel_sigma: float = 3.0
    max_gap_s: float = 0.5     # a window is rejected if any link has a longer gap
    margin_s: float = 5.0      # extra signal before and after a segment for the filter
    lags: int = 50
    width: tuple = (("A", 512), ("B", 256))  # dominant subcarrier count per collection


@dataclass(frozen=True)
class Train:
    """Demo4's optimiser settings as the starting point; not tuned on any test data."""
    batch_size: int = 128
    epochs: int = 40
    patience: int = 8
    encoder_lr: float = 3e-4
    head_lr: float = 1e-3
    weight_decay: float = 1e-2
    lr_min: float = 1e-6
    grad_clip: float = 1.0
    dropout: float = 0.2
    bin_dropout: float = 0.15
    jitter: float = 0.01
    link_permutation: bool = True
    seeds: tuple = (0, 1, 2)


@dataclass(frozen=True)
class Config:
    paths: Paths = field(default_factory=Paths)
    roles: Roles = field(default_factory=Roles)
    windows: Windows = field(default_factory=Windows)
    dsp: DSP = field(default_factory=DSP)
    train: Train = field(default_factory=Train)
