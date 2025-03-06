"""Grid trading strategy implementation."""
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from .base import BaseStrategy, Position

class GridStrategy(BaseStrategy):
    """
    Grid trading strategy that places buy and sell orders at regular price intervals.
    """
    
    def __init__(self,
                 grid_size: float = 0.01,
                 take_profit: float = 0.05,
                 num_grids: int = 10):
        """
        Initialize grid strategy.
        
        Args:
            grid_size: Grid size as percentage of price
            take_profit: Take profit percentage
            num_grids: Number of grids above and below current price
        """
        super().__init__()
        self.grid_size = grid_size
        self.take_profit = take_profit
        self.num_grids = num_grids
        
        self.positions: Dict[str, Position] = {}
        self.grids: Dict[str, List[float]] = {}
        self.portfolio_value = 0.0
        self.cash = 0.0
        self.equity = 0.0
    
    def update(self, timestamp: pd.Timestamp, data: pd.DataFrame):
        """
        Update strategy state with new data.
        
        Args:
            timestamp: Current timestamp
            data: New market data
        """
        for symbol in data.index.get_level_values('symbol').unique():
            try:
                symbol_data = data.loc[symbol]
                current_price = symbol_data.loc[timestamp, 'close']
                
                # Initialize grids if not already done
                if symbol not in self.grids:
                    self._initialize_grids(symbol, current_price)
                
                # Check for grid level crossings
                self._check_grid_levels(symbol, timestamp, current_price)
                
                # Check take profit for existing positions
                if symbol in self.positions:
                    position = self.positions[symbol]
                    returns = (current_price - position.entry_price) / position.entry_price
                    if returns >= self.take_profit:
                        self.close_position(symbol, timestamp, current_price)
                
            except KeyError:
                continue
    
    def _initialize_grids(self, symbol: str, price: float):
        """Initialize grid levels for a symbol."""
        grid_levels = []
        for i in range(-self.num_grids, self.num_grids + 1):
            grid_price = price * (1 + i * self.grid_size)
            grid_levels.append(grid_price)
        self.grids[symbol] = sorted(grid_levels)
    
    def _check_grid_levels(self, symbol: str, timestamp: pd.Timestamp, price: float):
        """Check if price has crossed any grid levels."""
        if symbol not in self.grids:
            return
            
        for i, grid_price in enumerate(self.grids[symbol][:-1]):
            next_grid_price = self.grids[symbol][i + 1]
            
            # Price crossed grid level from below
            if grid_price <= price < next_grid_price:
                if symbol not in self.positions:
                    # Open long position
                    size = self.calculate_position_size(symbol, price, 1)
                    self.open_position(symbol, timestamp, price, size)
                break
                
            # Price crossed grid level from above
            elif grid_price > price >= next_grid_price:
                if symbol not in self.positions:
                    # Open short position
                    size = self.calculate_position_size(symbol, price, -1)
                    self.open_position(symbol, timestamp, price, size)
                break
    
    def calculate_position_size(self, symbol: str, price: float, signal: int) -> float:
        """
        Calculate position size for a trade.
        
        Args:
            symbol: Trading symbol
            price: Current price
            signal: Trading signal (-1, 0, 1)
            
        Returns:
            Position size in units
        """
        if signal == 0:
            return 0.0
        
        # Use fixed position size per grid
        position_value = self.portfolio_value / (2 * self.num_grids)
        units = position_value / price
        
        return units * signal
    
    def get_portfolio_state(self) -> Dict[str, Any]:
        """
        Get current portfolio state.
        
        Returns:
            Dictionary with portfolio state information
        """
        return {
            'cash': self.cash,
            'equity': self.equity,
            'portfolio_value': self.portfolio_value,
            'positions': {
                symbol: {
                    'size': pos.size,
                    'entry_price': pos.entry_price,
                    'entry_time': pos.entry_time,
                    'pnl': pos.unrealized_pnl
                }
                for symbol, pos in self.positions.items()
            },
            'grids': self.grids
        }
    
    def get_stop_loss(self, symbol: str) -> Optional[float]:
        """
        Get stop loss price for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Stop loss price or None if no position
        """
        if symbol not in self.positions:
            return None
            
        position = self.positions[symbol]
        grid_size = self.grid_size
        
        if position.size > 0:
            return position.entry_price * (1 - grid_size)
        else:
            return position.entry_price * (1 + grid_size)
    
    def get_take_profit(self, symbol: str) -> Optional[float]:
        """
        Get take profit price for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Take profit price or None if no position
        """
        if symbol not in self.positions:
            return None
            
        position = self.positions[symbol]
        if position.size > 0:
            return position.entry_price * (1 + self.take_profit)
        else:
            return position.entry_price * (1 - self.take_profit)
    
    def open_position(self, symbol: str, timestamp: pd.Timestamp, price: float, size: float):
        """Open a new position."""
        self.positions[symbol] = Position(
            symbol=symbol,
            entry_time=timestamp,
            entry_price=price,
            size=size
        )
    
    def close_position(self, symbol: str, timestamp: pd.Timestamp, price: float):
        """Close an existing position."""
        if symbol in self.positions:
            position = self.positions[symbol]
            position.close(timestamp, price)
            del self.positions[symbol]
