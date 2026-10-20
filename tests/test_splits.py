import pytest
from config import Config
from src.index import discover_sessions, load_segments, build_windows
from src.splits import make_folds, check_no_leakage, splits_hash

CFG = Config()
pytestmark = pytest.mark.skipif(not CFG.paths.dataset.exists(), reason="dataset not available")


@pytest.fixture(scope="module")
def win():
    return build_windows(CFG, load_segments(discover_sessions(CFG.paths)))


def test_three_folds_leave_one_date_out(win):
    folds = make_folds(CFG, win)
    assert sorted(folds) == list(CFG.roles.pool)
    for date, f in folds.items():
        assert set(win.loc[f["val"], "date"]) == {date}
        assert date not in set(win.loc[f["train"], "date"])
        assert set(win.loc[f["primary"], "date"]) == {"2026-08-28"}
        assert set(win.loc[f["shift"], "collection"]) == {"B"}
        check_no_leakage(win, f)


def test_leakage_check_catches_a_shared_session(win):
    f = make_folds(CFG, win)[CFG.roles.pool[0]]
    bad = dict(f, train=f["train"].append(f["val"][:1]))
    with pytest.raises(AssertionError):
        check_no_leakage(win, bad)


def test_hash_is_stable_and_sensitive(win):
    a = make_folds(CFG, win)
    assert splits_hash(win, a) == splits_hash(win, make_folds(CFG, win))
    b = {d: dict(f, train=f["train"][1:]) for d, f in a.items()}
    assert splits_hash(win, a) != splits_hash(win, b)


# Thesis analyses C and D (docs/DECISIONS.md D17, D18): other split designs over the same cache.
from src.splits import design_folds  # noqa: E402


def test_lodo_design_is_the_original_folds(win):
    assert splits_hash(win, design_folds(CFG, win, "lodo")) == splits_hash(win, make_folds(CFG, win))


@pytest.mark.parametrize("design,train_date,test_date", [("both_0911", "2026-09-11", "2026-09-17"),
                                                         ("both_0917", "2026-09-17", "2026-09-11")])
def test_both_collections_moves_one_shift_date_into_training(win, design, train_date, test_date):
    folds = design_folds(CFG, win, design)
    assert sorted(folds) == list(CFG.roles.pool)
    for date, f in folds.items():
        assert set(win.loc[f["val"], "date"]) == {date}
        train_dates = set(win.loc[f["train"], "date"])
        assert train_date in train_dates and date not in train_dates
        assert test_date not in train_dates and "2026-08-28" not in train_dates
        assert set(win[win.date == train_date].index) <= set(f["train"])
        assert set(win.loc[f["shift"], "date"]) == {test_date}
        assert set(win.loc[f["primary"], "date"]) == {"2026-08-28"}
        check_no_leakage(win, f)


def test_loso_holds_out_each_pool_session_once(win):
    folds = design_folds(CFG, win, "loso")
    pool_sessions = set(win.loc[win.role == "pool", "session_key"])
    assert len(folds) == len(pool_sessions) == 11
    tested = set()
    for key, f in folds.items():
        session = set(win.loc[f["test"], "session_key"])
        assert len(session) == 1
        tested |= session
        val_dates = set(win.loc[f["val"], "date"])
        assert len(val_dates) == 1 and val_dates.isdisjoint(set(win.loc[f["test"], "date"]))
        assert val_dates.isdisjoint(set(win.loc[f["train"], "date"]))
        assert set(win.loc[f["train"], "role"]) == {"pool"}
        assert session.isdisjoint(set(win.loc[f["train"], "session_key"]))
        assert set(win.loc[f["train"], "label"]) == {"empty", "single-adult", "single-child", "multiple"}
        check_no_leakage(win, f)
    assert tested == pool_sessions


def test_unknown_design_is_refused(win):
    with pytest.raises(ValueError):
        design_folds(CFG, win, "nope")


def test_cli_designs_match_the_split_code():
    import main
    from src.splits import DESIGNS
    assert main.DESIGNS == DESIGNS
