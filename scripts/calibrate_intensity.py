from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from optimal_quoting.backtest.engine import MMParams, run_mm_toy
from optimal_quoting.calibration.dataset import build_intensity_dataset_from_mm
from optimal_quoting.calibration.diagnostics import empirical_intensity_binned
from optimal_quoting.calibration.mle import fit_intensity_exp_mle, profile_nll_over_k


def main() -> None:
    cfg = yaml.safe_load(Path("configs/mm_toy.yaml").read_text(encoding="utf-8"))

    p = MMParams(
        dt=float(cfg["dt"]),
        T=float(cfg["T"]),
        mid0=float(cfg["mid0"]),
        sigma=float(cfg["sigma"]),
        A=float(cfg["intensity"]["A"]),
        k=float(cfg["intensity"]["k"]),
        base_spread=float(cfg["strategy"]["base_spread"]),
        phi=float(cfg["strategy"]["phi"]),
        order_size=float(cfg["strategy"]["order_size"]),
        fee_bps=float(cfg["costs"]["fee_bps"]),
        seed=int(cfg["seed"]),
    )

    df = run_mm_toy(p)
    delta, n = build_intensity_dataset_from_mm(df, dt=p.dt)

    est = fit_intensity_exp_mle(delta, n, dt=p.dt, k_bounds=(0.0, 10.0), grid_size=300)

    print("=== True params ===")
    print(f"A_true={p.A:.6f}, k_true={p.k:.6f}")
    print("=== MLE estimate ===")
    print(f"A_hat={est.A:.6f}, k_hat={est.k:.6f}, nll={est.nll:.3f}")

    Path("reports/figures").mkdir(parents=True, exist_ok=True)

    # --- 1) true vs fitted intensity curve
    xs = np.linspace(0.0, float(np.quantile(delta, 0.995)), 200)
    lam_true = p.A * np.exp(-p.k * xs)
    lam_hat = est.A * np.exp(-est.k * xs)

    plt.figure()
    plt.plot(xs, lam_true, label="true")
    plt.plot(xs, lam_hat, label="fitted")
    plt.title("Intensity fit: λ(δ)=A exp(-kδ)")
    plt.xlabel("delta")
    plt.ylabel("lambda")
    plt.legend()
    plt.tight_layout()
    plt.savefig("reports/figures/intensity_fit.png")
    plt.close()

    # --- 2) empirical binned intensity vs fitted
    emp = empirical_intensity_binned(delta, n, dt=p.dt, nbins=40, dmax_quantile=0.995)
    lam_fit_centers = est.A * np.exp(-est.k * emp.bin_centers)

    plt.figure()
    plt.plot(emp.bin_centers, emp.lambda_hat, label="empirical (binned)")
    plt.plot(emp.bin_centers, lam_fit_centers, label="fitted")
    plt.title("Empirical intensity vs fitted")
    plt.xlabel("delta (bin centers)")
    plt.ylabel("lambda")
    plt.legend()
    plt.tight_layout()
    plt.savefig("reports/figures/intensity_empirical_fit.png")
    plt.close()

    # --- 3) profile likelihood for k
    k_grid = np.linspace(0.0, 5.0, 250)
    _, nlls = profile_nll_over_k(delta, n, dt=p.dt, k_grid=k_grid)
    nlls = nlls - np.nanmin(nlls)

    plt.figure()
    plt.plot(k_grid, nlls)
    plt.title("Profile NLL(k) (shifted)")
    plt.xlabel("k")
    plt.ylabel("NLL(k) - min")
    plt.tight_layout()
    plt.savefig("reports/figures/intensity_profile_nll.png")
    plt.close()

    print("Saved plots:")
    print(" - reports/figures/intensity_fit.png")
    print(" - reports/figures/intensity_empirical_fit.png")
    print(" - reports/figures/intensity_profile_nll.png")


if __name__ == "__main__":
    main()
