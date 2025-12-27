from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class EmpiricalIntensity:
    bin_centers: np.ndarray
    lambda_hat: np.ndarray
    counts: np.ndarray
    exposure: np.ndarray


def empirical_intensity_binned(
    delta: np.ndarray,
    n: np.ndarray,
    dt: float,
    nbins: int = 40,
    dmax_quantile: float = 0.995,
) -> EmpiricalIntensity:
    """
    Empirical intensity estimator using exposure-time binning:
        λ̂_b = (#events in bin b) / (exposure time in bin b)

    exposure time in bin b = (#samples in bin b) * dt
    """
    delta = np.asarray(delta, dtype=float)
    n = np.asarray(n, dtype=float)
    if dt <= 0:
        raise ValueError("dt must be > 0")
    if delta.ndim != 1 or n.ndim != 1 or len(delta) != len(n):
        raise ValueError("delta and n must be 1D arrays with same length")
    if (delta < 0).any():
        raise ValueError("delta must be >= 0")
    if (n < 0).any():
        raise ValueError("n must be >= 0")

    dmax = float(np.quantile(delta, dmax_quantile))
    dmax = max(dmax, 1e-12)

    edges = np.linspace(0.0, dmax, nbins + 1)
    idx = np.digitize(delta, edges) - 1
    idx = np.clip(idx, 0, nbins - 1)

    counts = np.zeros(nbins, dtype=float)
    samples = np.zeros(nbins, dtype=float)

    for b in range(nbins):
        mask = idx == b
        samples[b] = float(np.sum(mask))
        counts[b] = float(np.sum(n[mask]))

    exposure = samples * dt
    lambda_hat = np.where(exposure > 0, counts / exposure, np.nan)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])

    return EmpiricalIntensity(
        bin_centers=bin_centers,
        lambda_hat=lambda_hat,
        counts=counts,
        exposure=exposure,
    )
