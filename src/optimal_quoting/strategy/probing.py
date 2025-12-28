from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from optimal_quoting.strategy.quotes import Quotes, compute_quotes


@dataclass(frozen=True)
class ProbingConfig:
    p_explore: float          # probability of exploratory quote at each step
    jitter: float             # additive jitter on deltas (>=0)
    widen_only: bool = True   # if True: only widen (increase deltas), else allow +/- jitter


def compute_probing_quotes(
    mid: float,
    q: float,
    base_spread: float,
    phi: float,
    cfg: ProbingConfig,
    rng: np.random.Generator,
) -> Quotes:
    """
    Baseline quotes + randomized probing to diversify deltas for calibration.

    - With prob p_explore, we perturb deltas by jitter.
    - widen_only=True keeps perturbations non-negative and only increases deltas.
    """
    if not (0.0 <= cfg.p_explore <= 1.0):
        raise ValueError("p_explore must be in [0,1]")
    if cfg.jitter < 0:
        raise ValueError("jitter must be >= 0")

    base = compute_quotes(mid, q, base_spread, phi)

    if rng.random() >= cfg.p_explore or cfg.jitter == 0.0:
        return base

    if cfg.widen_only:
        # Only widen deltas: add U(0, jitter)
        eps_b = rng.random() * cfg.jitter
        eps_a = rng.random() * cfg.jitter
    else:
        # Allow +/- jitter (clipped to keep deltas >= 0)
        eps_b = (2.0 * rng.random() - 1.0) * cfg.jitter
        eps_a = (2.0 * rng.random() - 1.0) * cfg.jitter

    delta_bid = max(0.0, base.delta_bid + eps_b)
    delta_ask = max(0.0, base.delta_ask + eps_a)

    bid = mid - delta_bid
    ask = mid + delta_ask
    return Quotes(bid=bid, ask=ask, delta_bid=delta_bid, delta_ask=delta_ask)
