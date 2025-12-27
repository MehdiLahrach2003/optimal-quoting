import numpy as np

from optimal_quoting.calibration.diagnostics import empirical_intensity_binned


def test_empirical_intensity_shapes():
    rng = np.random.default_rng(0)
    delta = rng.random(1000)
    n = rng.integers(0, 2, size=1000)
    emp = empirical_intensity_binned(delta, n, dt=0.1, nbins=20)

    assert emp.bin_centers.shape == (20,)
    assert emp.lambda_hat.shape == (20,)
    assert emp.counts.shape == (20,)
    assert emp.exposure.shape == (20,)
