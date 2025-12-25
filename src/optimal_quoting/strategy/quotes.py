from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Quotes:
    bid: float
    ask: float
    delta_bid: float
    delta_ask: float


def compute_quotes(mid: float, q: float, base_spread: float, phi: float) -> Quotes:
    """
    Baseline quoting with inventory skew:
        δa = s/2 + φ q
        δb = s/2 - φ q
    where s = base_spread.

    q > 0 means long inventory -> widen ask (sell faster), tighten bid (buy slower).
    """
    if base_spread < 0:
        raise ValueError("base_spread must be >= 0")

    half = 0.5 * base_spread
    delta_ask = max(0.0, half + phi * q)
    delta_bid = max(0.0, half - phi * q)
    ask = mid + delta_ask
    bid = mid - delta_bid
    return Quotes(bid=bid, ask=ask, delta_bid=delta_bid, delta_ask=delta_ask)
