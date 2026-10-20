import numpy as np
from src.baselines import fit_b1, never_alarm


def toy(n=200, seed=0):
    rng = np.random.default_rng(seed)
    y = np.repeat(np.arange(4), n // 4)
    X = rng.normal(size=(n, 50, 4, 64)).astype("f4")
    X[y == 2] += 0.5 * np.linspace(0, 1, 50)[None, :, None, None]     # class 2 carries a lag trend
    return X, y


def test_never_alarm_is_constant_zero():
    X, _ = toy(8)
    p = never_alarm(X)
    assert p.shape == (8, 4) and (p == 0).all()


def test_b1_learns_a_simple_signal_and_returns_four_probabilities():
    X, y = toy()
    p = fit_b1(X, y)(X)
    assert p.shape == (200, 4) and np.allclose(p.sum(1), 1)
    assert (p.argmax(1) == y).mean() > 0.8


def test_b1_scaler_uses_training_data_only():
    X, y = toy()
    Xt, _ = toy(seed=1)
    fit = fit_b1(X, y)
    np.testing.assert_allclose(fit(Xt), fit_b1(X, y)(Xt))       # deterministic, fitted on X only
    assert not np.allclose(fit(Xt * 100), fit(Xt))               # test rescaling does not refit anything
