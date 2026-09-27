from dataclasses import dataclass
import numpy as np
from scipy.stats import norm


@dataclass(frozen=True)
class Greeks:
    delta: float
    gamma: float
    theta: float
    vega: float


def black_scholes(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    kind: str = "call",
) -> tuple[float, Greeks]:
    """
    Price a European option using Black-Scholes.

    S      : current spot price
    K      : strike price
    T      : time to expiry in YEARS
    r      : risk-free rate (e.g. 0.06 for 6%)
    sigma  : annualized volatility (e.g. 0.15 for 15%)
    kind   : "call" or "put"
    """
    if T <= 0:
        intrinsic = max(S - K, 0.0) if kind == "call" else max(K - S, 0.0)
        return intrinsic, Greeks(0.0, 0.0, 0.0, 0.0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if kind == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1.0

    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100.0

    theta_common = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    if kind == "call":
        theta = (theta_common - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365.0
    else:
        theta = (theta_common + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365.0

    return price, Greeks(delta, gamma, theta, vega)