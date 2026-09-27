import pytest
from src.core.pricing import black_scholes, Greeks


def test_atm_call_price():
    """ATM call with known inputs should match Black-Scholes textbook value."""
    price, _ = black_scholes(S=100, K=100, T=0.25, r=0.06, sigma=0.20, kind="call")
    assert price == pytest.approx(4.76, abs=0.02)


def test_atm_put_price():
    """ATM put with same inputs should be lower than call (positive rate)."""
    price, _ = black_scholes(S=100, K=100, T=0.25, r=0.06, sigma=0.20, kind="put")
    assert price == pytest.approx(3.27, abs=0.02)


def test_atm_call_delta():
    """ATM call delta should be slightly above 0.5 when r > 0."""
    _, greeks = black_scholes(S=100, K=100, T=0.25, r=0.06, sigma=0.20, kind="call")
    assert greeks.delta == pytest.approx(0.5793, abs=0.001)


def test_put_delta_is_negative():
    """Put delta should be negative."""
    _, greeks = black_scholes(S=100, K=100, T=0.25, r=0.06, sigma=0.20, kind="put")
    assert greeks.delta < 0
    assert greeks.delta == pytest.approx(-0.4207, abs=0.001)


def test_gamma_positive():
    """Gamma is always positive for both calls and puts."""
    _, call_g = black_scholes(S=100, K=100, T=0.25, r=0.06, sigma=0.20, kind="call")
    _, put_g = black_scholes(S=100, K=100, T=0.25, r=0.06, sigma=0.20, kind="put")
    assert call_g.gamma > 0
    assert put_g.gamma > 0
    assert call_g.gamma == pytest.approx(put_g.gamma, abs=1e-9)


def test_expired_option_is_intrinsic():
    """At expiry (T=0), option value equals intrinsic value."""
    itm_call, greeks = black_scholes(S=110, K=100, T=0, r=0.06, sigma=0.20, kind="call")
    assert itm_call == 10.0
    assert greeks.delta == 0.0

    otm_call, _ = black_scholes(S=90, K=100, T=0, r=0.06, sigma=0.20, kind="call")
    assert otm_call == 0.0

    itm_put, _ = black_scholes(S=90, K=100, T=0, r=0.06, sigma=0.20, kind="put")
    assert itm_put == 10.0


def test_put_call_parity():
    """
    Put-Call Parity: C - P = S - K * exp(-rT)
    This is a fundamental identity that must hold for any valid pricing model.
    """
    S, K, T, r, sigma = 100, 100, 0.25, 0.06, 0.20
    call_price, _ = black_scholes(S, K, T, r, sigma, kind="call")
    put_price, _ = black_scholes(S, K, T, r, sigma, kind="put")

    lhs = call_price - put_price
    rhs = S - K * __import__("numpy").exp(-r * T)

    assert lhs == pytest.approx(rhs, abs=0.01)


def test_higher_vol_raises_call_price():
    """More volatility = more expensive call. Basic monotonicity."""
    low_vol, _ = black_scholes(S=100, K=100, T=0.25, r=0.06, sigma=0.10, kind="call")
    high_vol, _ = black_scholes(S=100, K=100, T=0.25, r=0.06, sigma=0.40, kind="call")
    assert high_vol > low_vol