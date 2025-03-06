"""Trading signal module."""
from dataclasses import dataclass
from typing import Dict, Any
import pandas as pd


@dataclass
class Signal:
    """Trading signal information."""
    symbol: str
    direction: int  # 1 for long, -1 for short, 0 for flat
    strength: float  # Signal strength between 0 and 1
    timestamp: pd.Timestamp
    metadata: Dict[str, Any] = None
