from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class Quote:
    bid: Optional[float]
    ask: Optional[float]


@dataclass(frozen=True)
class StrategyState:
    t: float
    inventory: float
    cash: float


StrategyOutput = Dict[str, float]
