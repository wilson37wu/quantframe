"""Mean reversion trading strategy implementation."""
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from .base import BaseStrategy, Position

class MeanReversionStrategy(BaseStrategy):
    """
    Mean reversion strategy using moving averages and Bollinger Bands.
    """
    
    def __init__(self, config_manager):
        """
        Initialize mean reversion strategy.
        """
        self.config_manager = config_manager
        self.config = None
        
        # Strategy parameters (these should come from config_manager)
        config = self.config_manager.load_strategy_config('mean_reversion')
        self.ma_period = config.get('ma_period', 20)
        self.std_dev = config.get('std_dev', 2.0)
        self.stop_loss = config.get('stop_loss', 0.02)
        self.take_profit = config.get('take_profit', 0.05)
        
        # State variables
        self.positions: Dict[str, Position] = {}
        self.portfolio_value = 0.0
        self.cash = 0.0
        self.equity = 0.0
    
    def initialize_state(self, config: Dict):
        """Initialize strategy state with configuration.
        
        Args:
            config: Strategy configuration dictionary
        """
        self.config = config
        self.ma_period = config.get('ma_period', self.ma_period)
        self.std_dev = config.get('std_dev', self.std_dev)
        self.stop_loss = config.get('stop_loss', self.stop_loss)
        self.take_profit = config.get('take_profit', self.take_profit)
    
    def update(self, timestamp: pd.Timestamp, data: pd.DataFrame):
        """
        Update strategy state with new data.
        
        Args:
            timestamp: Current timestamp
            data: New market data
        """
        # Generate signals
        signals = self.generate_signals(data)
        
        # Update positions based on signals
        for symbol in data.index.get_level_values('symbol').unique():
            try:
                symbol_data = data.loc[symbol]
                signal = signals.loc[symbol].loc[timestamp, 'signal']
                current_price = symbol_data.loc[timestamp, 'close']
                
                # Calculate position size
                size = self.calculate_position_size(symbol, current_price, signal)
                
                if signal != 0:
                    # Close existing position if opposite direction
                    if symbol in self.positions:
                        current_position = self.positions[symbol]
                        if np.sign(current_position.size) != np.sign(signal):
                            self.close_position(symbol, timestamp, current_price)
                    
                    # Open new position
                    if size != 0:
                        self.open_position(
                            symbol=symbol,
                            timestamp=timestamp,
                            price=current_price,
                            size=size
                        )
                
                # Check stop loss and take profit
                if symbol in self.positions:
                    position = self.positions[symbol]
                    returns = (current_price - position.entry_price) / position.entry_price
                    if position.size < 0:
                        returns = -returns
                    
                    if returns <= -self.stop_loss:
                        self.close_position(symbol, timestamp, current_price)
                    elif returns >= self.take_profit:
                        self.close_position(symbol, timestamp, current_price)
                        
            except KeyError:
                continue
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on mean reversion indicators.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with signals (-1 for sell, 0 for hold, 1 for buy)
        """
        signals = pd.DataFrame(
            0,
            index=data.index,
            columns=['signal']
        )
        
        for symbol in data.index.get_level_values('symbol').unique():
            try:
                symbol_data = data.loc[symbol]
                prices = symbol_data['close']
                
                # Calculate moving average and standard deviation
                ma = prices.rolling(window=self.ma_period).mean()
                std = prices.rolling(window=self.ma_period).std()
                
                # Calculate Bollinger Bands
                upper_band = ma + self.std_dev * std
                lower_band = ma - self.std_dev * std
                
                # Generate signals
                # Buy when price is below lower band
                buy_signal = prices < lower_band
                
                # Sell when price is above upper band
                sell_signal = prices > upper_band
                
                # Create signal series
                symbol_signals = pd.Series(0, index=prices.index)
                symbol_signals[buy_signal] = 1
                symbol_signals[sell_signal] = -1
                
                # Update signals DataFrame
                signals.loc[symbol, 'signal'] = symbol_signals
                
            except Exception as e:
                print(f"Error generating signals for {symbol}: {str(e)}")
                continue
        
        return signals
    
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
        
        # Calculate position value based on portfolio value
        position_value = self.portfolio_value * 0.1  # Use 10% of portfolio per position
        
        # Calculate units based on price
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
            }
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
        if position.size > 0:
            return position.entry_price * (1 - self.stop_loss)
        else:
            return position.entry_price * (1 + self.stop_loss)
    
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
