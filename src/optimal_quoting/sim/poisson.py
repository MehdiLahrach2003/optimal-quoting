from __future__ import annotations

import math
import numpy as np


def event_happens(lmbda: float, dt: float, rng: np.random.Generator) -> bool:
    """
    Poisson arrival within dt:
        P(event) = 1 - exp(-λ dt)
    """
    if lmbda < 0:
        raise ValueError("lambda must be >= 0")
    if dt <= 0:
        raise ValueError("dt must be > 0")
    p = 1.0 - math.exp(-lmbda * dt)
    return bool(rng.random() < p)
