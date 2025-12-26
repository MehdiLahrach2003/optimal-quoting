from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class IntensityMLE:
    A: float
    k: float
    nll: float  # negative log-likelihood (Poisson approx)


def _A_hat_given_k(delta: np.ndarray, n: np.ndarray, dt: float, k: float) -> float:
    """
    Closed-form MLE for A given k:
        A_hat(k) = sum n_t / sum (dt * exp(-k*delta_t))
    """
    if dt <= 0:
        raise ValueError("dt must be > 0")
    if k < 0:
        raise ValueError("k must be >= 0")

    w = np.exp(-k * delta)
    denom = float(dt * np.sum(w))
    num = float(np.sum(n))
    if denom <= 0:
        raise ValueError("Degenerate denominator in A_hat(k)")
    # If there are no events, A_hat would be 0; we keep a tiny floor to avoid log(0).
    return max(num / denom, 1e-12)


def _poisson_nll(delta: np.ndarray, n: np.ndarray, dt: float, A: float, k: float) -> float:
    """
    Poisson negative log-likelihood up to an additive constant:
        nll = sum( lambda_t dt - n_t log(lambda_t) ), with lambda_t = A exp(-k delta_t).
    """
    if A <= 0:
        raise ValueError("A must be > 0")
    if k < 0:
        raise ValueError("k must be >= 0")
    if dt <= 0:
        raise ValueError("dt must be > 0")

    lam = A * np.exp(-k * delta)
    # avoid log(0)
    lam = np.maximum(lam, 1e-18)
    return float(np.sum(lam * dt - n * np.log(lam)))


def fit_intensity_exp_mle(
    delta: np.ndarray,
    n: np.ndarray,
    dt: float,
    k_bounds: tuple[float, float] = (0.0, 20.0),
    grid_size: int = 200,
) -> IntensityMLE:
    """
    MLE for lambda(delta) = A exp(-k delta), using:
      - closed-form A_hat(k)
      - 1D search on k (coarse grid + local refinement)

    Parameters
    ----------
    delta : array, shape (T,)
        distances >= 0
    n : array, shape (T,)
        counts in {0,1} (or nonnegative ints)
    dt : float
        time step
    k_bounds : (k_min, k_max)
        search interval for k
    grid_size : int
        number of grid points in coarse search
    """
    delta = np.asarray(delta, dtype=float)
    n = np.asarray(n, dtype=float)

    if delta.ndim != 1 or n.ndim != 1 or len(delta) != len(n):
        raise ValueError("delta and n must be 1D arrays with same length")
    if (delta < 0).any():
        raise ValueError("delta must be >= 0")
    if (n < 0).any():
        raise ValueError("n must be >= 0")
    if dt <= 0:
        raise ValueError("dt must be > 0")

    k_min, k_max = k_bounds
    if not (0 <= k_min < k_max):
        raise ValueError("Invalid k_bounds")

    # --- Coarse grid search on k
    ks = np.linspace(k_min, k_max, grid_size)
    best = (math.inf, None, None)  # (nll, A, k)

    for k in ks:
        A = _A_hat_given_k(delta, n, dt, float(k))
        nll = _poisson_nll(delta, n, dt, A, float(k))
        if nll < best[0]:
            best = (nll, A, float(k))

    _, _, k0 = best

    # --- Local refinement around best grid point via golden-section search
    # Define a small bracket around k0 (one grid step each side)
    step = (k_max - k_min) / max(grid_size - 1, 1)
    a = max(k_min, k0 - step)
    b = min(k_max, k0 + step)

    def f(k: float) -> float:
        A = _A_hat_given_k(delta, n, dt, k)
        return _poisson_nll(delta, n, dt, A, k)

    # golden section
    phi = (1 + math.sqrt(5)) / 2
    invphi = 1 / phi

    c = b - invphi * (b - a)
    d = a + invphi * (b - a)
    fc = f(c)
    fd = f(d)

    for _ in range(60):
        if abs(b - a) < 1e-8:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - invphi * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + invphi * (b - a)
            fd = f(d)

    k_hat = float(0.5 * (a + b))
    A_hat = float(_A_hat_given_k(delta, n, dt, k_hat))
    nll_hat = float(_poisson_nll(delta, n, dt, A_hat, k_hat))

    return IntensityMLE(A=A_hat, k=k_hat, nll=nll_hat)
