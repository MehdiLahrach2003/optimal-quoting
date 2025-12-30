from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from optimal_quoting.backtest.engine import MMParams, run_mm_toy
from optimal_quoting.metrics.performance import performance_summary
from optimal_quoting.calibration.dataset import build_intensity_dataset_from_mm
from optimal_quoting.calibration.mle import fit_intensity_exp_mle


@dataclass(frozen=True)
class FrontierConfig:
    p_grid: list[float]
    jitter_grid: list[float]
    seeds: list[int]
    k_bounds: tuple[float, float] = (0.0, 5.0)
    grid_size: int = 300


def run_probing_frontier(base: MMParams, cfg: FrontierConfig) -> pd.DataFrame:
    """
    Runs a 2D grid over (probing_p, probing_jitter), aggregates over seeds.
    Returns a tidy dataframe with one row per (p, jitter, seed) and metrics.
    """
    rows: list[dict] = []

    for p_explore in cfg.p_grid:
        for jitter in cfg.jitter_grid:
            for seed in cfg.seeds:
                p = MMParams(**{**base.__dict__})
                p.seed = int(seed)
                p.policy = "probing" if (p_explore > 0 and jitter > 0) else "baseline"
                p.probing_p = float(p_explore)
                p.probing_jitter = float(jitter)
                p.probing_widen_only = True

                df = run_mm_toy(p)
                perf = performance_summary(df)

                delta, n = build_intensity_dataset_from_mm(df, dt=p.dt)
                est = fit_intensity_exp_mle(
                    delta, n, dt=p.dt,
                    k_bounds=cfg.k_bounds,
                    grid_size=cfg.grid_size
                )

                row = {
                    "p_explore": float(p_explore),
                    "jitter": float(jitter),
                    "seed": int(seed),
                    "A_hat": float(est.A),
                    "k_hat": float(est.k),
                    "k_abs_error": float(abs(est.k - p.k)),
                    **perf,
                }
                rows.append(row)

    return pd.DataFrame(rows)
