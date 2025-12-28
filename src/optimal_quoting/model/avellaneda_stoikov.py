from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ASParams:
    """
    Avellaneda–Stoikov quoting parameters (reduced-form).

    We assume:
      - mid dynamics: dS ~ sigma dW (discretized elsewhere)
      - execution intensity: lambda(delta) = A exp(-k delta)
      - utility: exponential with risk aversion gamma

    The canonical closed-form (reduced) deltas we use:
      delta_ask = 1/k + (gamma * sigma^2 * tau / 2) * q
      delta_bid = 1/k - (gamma * sigma^2 * tau / 2) * q

    where tau = (T - t) is time-to-horizon (>= 0).
    """
    gamma: float
    sigma: float     # per-step sigma used in your toy simulation
    k: float         # intensity decay
    T: float         # horizon in seconds


def as_deltas(q: float, t: float, p: ASParams) -> tuple[float, float]:
    """
    Returns (delta_bid, delta_ask) for inventory q at time t.

    Notes:
    - deltas are clipped to be >= 0
    - t in [0, T]
    """
    if p.gamma <= 0:
        raise ValueError("gamma must be > 0")
    if p.sigma < 0:
        raise ValueError("sigma must be >= 0")
    if p.k <= 0:
        raise ValueError("k must be > 0")
    if p.T <= 0:
        raise ValueError("T must be > 0")
    if t < 0 or t > p.T:
        raise ValueError("t must be in [0, T]")

    tau = p.T - t  # time-to-horizon
    base = 1.0 / p.k
    skew = 0.5 * p.gamma * (p.sigma ** 2) * tau * q

    delta_bid = max(0.0, base - skew)
    delta_ask = max(0.0, base + skew)

    return delta_bid, delta_ask


def as_quotes(mid: float, q: float, t: float, p: ASParams) -> tuple[float, float, float, float]:
    """
    Returns (bid, ask, delta_bid, delta_ask).
    """
    d_bid, d_ask = as_deltas(q=q, t=t, p=p)
    bid = mid - d_bid
    ask = mid + d_ask
    return bid, ask, d_bid, d_ask
