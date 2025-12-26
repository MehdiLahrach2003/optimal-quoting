import numpy as np

from optimal_quoting.backtest.engine import MMParams, run_mm_toy
from optimal_quoting.calibration.dataset import build_intensity_dataset_from_mm
from optimal_quoting.calibration.mle import fit_intensity_exp_mle


def test_mle_recovers_params_reasonably():
    # Long enough horizon for stability
    p = MMParams(
        dt=1.0,
        T=20000.0,
        mid0=100.0,
        sigma=0.02,
        A=1.2,
        k=1.0,
        base_spread=0.2,
        phi=0.0,
        order_size=0.01,
        fee_bps=0.0,
        seed=123,
    )
    df = run_mm_toy(p)
    delta, n = build_intensity_dataset_from_mm(df, dt=p.dt)

    est = fit_intensity_exp_mle(delta, n, dt=p.dt, k_bounds=(0.0, 5.0), grid_size=200)

    # Loose tolerances: stochastic + discretization + finite sample
    assert np.isfinite(est.A) and est.A > 0
    assert np.isfinite(est.k) and est.k >= 0
    assert abs(est.k - p.k) < 0.35
