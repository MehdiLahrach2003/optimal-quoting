from __future__ import annotations

import numpy as np
import pandas as pd


def add_mid_spread(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["mid"] = 0.5 * (out["bid"] + out["ask"])
    out["spread"] = out["ask"] - out["bid"]
    return out


def add_log_returns(df: pd.DataFrame, price_col: str = "mid") -> pd.DataFrame:
    out = df.copy()
    p = out[price_col].astype(float)
    out["logret"] = np.log(p).diff()
    return out


def realized_vol(df: pd.DataFrame, ret_col: str = "logret") -> float:
    r = df[ret_col].dropna().to_numpy(dtype=float)
    if len(r) == 0:
        return float("nan")
    return float(np.sqrt(np.mean(r * r)))
