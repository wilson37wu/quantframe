"""Position model for tracking trading positions.

This module provides a Position class for managing and tracking individual trading positions,
including entry/exit points, position size, and profit/loss calculations.
"""

from dataclasses import dataclass
from typing import Optional
import pandas as pd


@dataclass
class Position:
    """Class representing a trading position.
    
    A Position tracks the lifecycle of a single trade, including entry and exit points,
    position size, and calculates the resulting profit/loss.
    
    Attributes:
        symbol (str): Trading symbol/ticker
        entry_time (pd.Timestamp): Position entry timestamp
        entry_price (float): Entry price of the position
        size (float): Position size (positive for long, negative for short)
        exit_time (Optional[pd.Timestamp]): Position exit timestamp, None if position is open
        exit_price (Optional[float]): Exit price of the position, None if position is open
        pnl (Optional[float]): Realized profit/loss, None if position is open
    
    Example:
        >>> pos = Position("AAPL", pd.Timestamp("2023-01-01"), 150.0, 100)
        >>> pos.close(pd.Timestamp("2023-01-02"), 155.0)
        >>> print(pos.pnl)  # Shows profit of 500.0
    """
    
    symbol: str
    entry_time: pd.Timestamp
    entry_price: float
    size: float
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    
    def close(self, timestamp: pd.Timestamp, price: float) -> None:
        """Close the position and calculate PnL.
        
        Args:
            timestamp (pd.Timestamp): Time of position closure
            price (float): Exit price
            
        Note:
            PnL is calculated as (exit_price - entry_price) * size.
            For short positions (negative size), the formula automatically
            handles the sign correctly.
        """
        self.exit_time = timestamp
        self.exit_price = price
        self.pnl = (self.exit_price - self.entry_price) * self.size
        
    @property
    def is_open(self) -> bool:
        """Check if position is still open.
        
        Returns:
            bool: True if position hasn't been closed, False otherwise
        """
        return self.exit_time is None
        
    @property
    def duration(self) -> Optional[pd.Timedelta]:
        """Get position duration.
        
        Returns:
            Optional[pd.Timedelta]: Time difference between exit and entry time.
                Returns None if position is still open.
        """
        if not self.exit_time:
            return None
        return self.exit_time - self.entry_time
