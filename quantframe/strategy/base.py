"""Base classes for trading strategies.

This module provides the foundational classes for implementing trading strategies
in the quantframe framework. It includes:
- Position tracking with PnL calculations
- Signal generation and management
- Abstract base class for strategy implementation

The BaseStrategy class defines the core interface that all trading strategies
must implement, ensuring consistent behavior across different implementations.

Example:
    >>> class MyStrategy(BaseStrategy):
    ...     def update(self, timestamp, data):
    ...         # Process new market data
    ...         pass
    ...     def calculate_position_size(self, symbol, price, signal):
    ...         return 100  # Fixed position size
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import pandas as pd

@dataclass
class Position:
    """Trading position information.
    
    Tracks both the entry and exit points of a trade, along with real-time
    profit/loss calculations for both realized and unrealized gains.
    
    Attributes:
        symbol (str): Trading symbol/ticker
        entry_time (pd.Timestamp): Position entry timestamp
        entry_price (float): Entry price of the position
        size (float): Position size (positive for long, negative for short)
        exit_time (Optional[pd.Timestamp]): Position exit timestamp, None if position is open
        exit_price (Optional[float]): Exit price, None if position is open
        unrealized_pnl (float): Current unrealized profit/loss
        realized_pnl (float): Realized profit/loss after position closure
        
    Example:
        >>> pos = Position("BTC/USD", pd.Timestamp("2023-01-01"), 50000.0, 0.1)
        >>> pos.unrealized_pnl = (51000.0 - pos.entry_price) * pos.size  # $100 profit
        >>> pos.close(pd.Timestamp("2023-01-02"), 51000.0)  # Realize the profit
    """
    
    symbol: str
    entry_time: pd.Timestamp
    entry_price: float
    size: float
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    
    def close(self, timestamp: pd.Timestamp, price: float) -> None:
        """Close the position and calculate realized PnL.
        
        Args:
            timestamp (pd.Timestamp): Time of position closure
            price (float): Exit price
            
        Note:
            This method updates both realized and unrealized PnL.
            Realized PnL is calculated as (exit_price - entry_price) * size,
            and unrealized PnL is set to 0 as the position is closed.
        """
        self.exit_time = timestamp
        self.exit_price = price
        self.realized_pnl = (self.exit_price - self.entry_price) * self.size
        self.unrealized_pnl = 0.0

@dataclass
class Signal:
    """Trading signal data class.
    
    Represents a trading signal with direction, strength, and additional metadata.
    This class standardizes signal generation across different strategies.
    
    Attributes:
        symbol (str): Trading symbol/ticker
        direction (int): Signal direction (1: long, -1: short, 0: no position)
        strength (float): Signal strength, normalized between 0 and 1
        timestamp (pd.Timestamp): Time when signal was generated
        metadata (Dict[str, Any]): Additional signal information (e.g., indicators)
        
    Example:
        >>> signal = Signal("ETH/USD", 1, 0.8, pd.Timestamp.now(),
        ...                {"rsi": 30, "trend": "bullish"})
    """
    symbol: str
    direction: int
    strength: float
    timestamp: pd.Timestamp
    metadata: Dict[str, Any]

class BaseStrategy(ABC):
    """Abstract base class for trading strategies.
    
    This class defines the interface that all trading strategies must implement.
    It provides a framework for:
    - Processing market data updates
    - Position sizing calculations
    - Portfolio state management
    - Risk management (stop-loss and take-profit)
    
    To implement a new strategy:
    1. Inherit from this class
    2. Implement all abstract methods
    3. Add strategy-specific logic and parameters
    
    Example:
        >>> class SimpleMAStrategy(BaseStrategy):
        ...     def __init__(self, ma_period: int = 20):
        ...         self.ma_period = ma_period
        ...         self.positions = {}
        ...     
        ...     def update(self, timestamp, data):
        ...         # Calculate moving average and generate signals
        ...         pass
    """
    
    @abstractmethod
    def update(self, timestamp: pd.Timestamp, data: pd.DataFrame) -> None:
        """Update strategy state with new market data.
        
        This method is called whenever new market data is available. It should:
        1. Update internal state with new data
        2. Generate new signals if applicable
        3. Update position states and PnL calculations
        
        Args:
            timestamp (pd.Timestamp): Current market timestamp
            data (pd.DataFrame): New market data with OHLCV columns
        """
        pass
    
    @abstractmethod
    def calculate_position_size(self, symbol: str, price: float, signal: int) -> float:
        """Calculate position size for a trade.
        
        Determines the appropriate position size based on:
        - Available capital
        - Risk parameters
        - Signal strength
        - Current market conditions
        
        Args:
            symbol (str): Trading symbol
            price (float): Current market price
            signal (int): Trading signal direction (-1, 0, 1)
            
        Returns:
            float: Position size in base currency units
            
        Note:
            Position size should account for maximum drawdown limits
            and overall portfolio risk management rules.
        """
        pass
    
    @abstractmethod
    def get_portfolio_state(self) -> Dict[str, Any]:
        """Get current portfolio state.
        
        Provides a snapshot of the current portfolio including:
        - Open positions
        - Available capital
        - Current PnL
        - Risk metrics
        
        Returns:
            Dict[str, Any]: Portfolio state information including:
                - positions: Dict of open positions
                - capital: Available trading capital
                - total_value: Total portfolio value
                - risk_metrics: Dict of risk measurements
        """
        pass
    
    @abstractmethod
    def get_stop_loss(self, symbol: str) -> Optional[float]:
        """Get stop loss price for a symbol.
        
        Calculates the appropriate stop loss level based on:
        - Position entry price
        - Market volatility
        - Maximum allowed loss
        
        Args:
            symbol (str): Trading symbol
            
        Returns:
            Optional[float]: Stop loss price or None if no position
            
        Note:
            The stop loss calculation should consider both fixed
            and dynamic (trailing) stop loss strategies.
        """
        pass
    
    @abstractmethod
    def get_take_profit(self, symbol: str) -> Optional[float]:
        """Get take profit price for a symbol.
        
        Calculates the appropriate take profit level based on:
        - Position entry price
        - Expected return
        - Market conditions
        
        Args:
            symbol (str): Trading symbol
            
        Returns:
            Optional[float]: Take profit price or None if no position
            
        Note:
            Take profit levels can be static or dynamic based on
            market conditions and strategy parameters.
        """
        pass
