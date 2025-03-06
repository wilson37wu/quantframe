"""Momentum trading strategy implementation."""
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from .base import BaseStrategy, Position

class MomentumStrategy(BaseStrategy):
    """
    Momentum trading strategy using multiple technical indicators:
    - RSI for overbought/oversold conditions
    - MACD for trend confirmation
    - Moving averages for trend direction
    """
    
    def __init__(self, 
                 rsi_period: int = 14,
                 rsi_overbought: float = 70,
                 rsi_oversold: float = 30,
                 sma_fast: int = 20,
                 sma_slow: int = 50,
                 macd_fast: int = 12,
                 macd_slow: int = 26,
                 macd_signal: int = 9,
                 position_size: float = 0.1,
                 stop_loss: float = 0.02,
                 take_profit: float = 0.05):
        """
        Initialize momentum strategy.
        
        Args:
            rsi_period: Period for RSI calculation
            rsi_overbought: RSI level considered overbought
            rsi_oversold: RSI level considered oversold
            sma_fast: Fast SMA period
            sma_slow: Slow SMA period
            macd_fast: Fast period for MACD
            macd_slow: Slow period for MACD
            macd_signal: Signal period for MACD
            position_size: Position size as fraction of portfolio
            stop_loss: Stop loss percentage
            take_profit: Take profit percentage
        """
        super().__init__()
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.sma_fast = sma_fast
        self.sma_slow = sma_slow
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.position_size = position_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        
        # Initialize portfolio variables
        self.positions: Dict[str, Position] = {}
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
        # Generate signals
        signals = self.generate_signals(data)
        
        # Update positions based on signals
        for symbol in data.index.get_level_values('symbol').unique():
            try:
                symbol_data = data.loc[symbol]
                signal = signals.loc[symbol].loc[timestamp, 'signal']
                
                # Calculate position size
                size = self.calculate_position_size(symbol, symbol_data.loc[timestamp, 'Close'], signal)
                
                if signal != 0:
                    # Close existing position if opposite direction
                    if symbol in self.positions:
                        current_position = self.positions[symbol]
                        if np.sign(current_position.size) != np.sign(signal):
                            self.close_position(symbol, timestamp, symbol_data.loc[timestamp, 'Close'])
                    
                    # Open new position
                    if size != 0:
                        self.open_position(
                            symbol=symbol,
                            timestamp=timestamp,
                            price=symbol_data.loc[timestamp, 'Close'],
                            size=size
                        )
                
                # Check stop loss and take profit
                if symbol in self.positions:
                    position = self.positions[symbol]
                    current_price = symbol_data.loc[timestamp, 'Close']
                    
                    # Calculate returns
                    returns = (current_price - position.entry_price) / position.entry_price
                    if position.size < 0:
                        returns = -returns
                    
                    # Check stop loss
                    if returns <= -self.stop_loss:
                        self.close_position(symbol, timestamp, current_price)
                    
                    # Check take profit
                    elif returns >= self.take_profit:
                        self.close_position(symbol, timestamp, current_price)
                        
            except KeyError:
                continue  # Skip if data is missing
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on momentum indicators.
        
        Args:
            data: DataFrame with OHLCV data and technical indicators
            
        Returns:
            DataFrame with signals (-1 for sell, 0 for hold, 1 for buy)
        """
        # Initialize signals DataFrame with multi-index
        signals = pd.DataFrame(
            0,
            index=data.index,
            columns=['signal']
        )
        
        for symbol in data.index.get_level_values('symbol').unique():
            try:
                symbol_data = data.loc[symbol]
                
                # Ensure all required indicators are present
                required_columns = ['RSI', 'MACD', 'Signal', 'SMA_20', 'SMA_50']
                missing_columns = [col for col in required_columns if col not in symbol_data.columns]
                if missing_columns:
                    print(f"Missing indicators for {symbol}: {missing_columns}")
                    continue
                
                # Buy conditions:
                # 1. RSI crosses above oversold
                # 2. MACD line crosses above signal line
                # 3. Fast MA above slow MA
                buy_condition = (
                    (symbol_data['RSI'] > self.rsi_oversold) &
                    (symbol_data['MACD'] > symbol_data['Signal']) &
                    (symbol_data['SMA_20'] > symbol_data['SMA_50'])
                )
                
                # Sell conditions:
                # 1. RSI crosses below overbought
                # 2. MACD line crosses below signal line
                # 3. Fast MA below slow MA
                sell_condition = (
                    (symbol_data['RSI'] < self.rsi_overbought) &
                    (symbol_data['MACD'] < symbol_data['Signal']) &
                    (symbol_data['SMA_20'] < symbol_data['SMA_50'])
                )
                
                # Create signal series
                symbol_signals = pd.Series(0, index=symbol_data.index)
                symbol_signals[buy_condition] = 1
                symbol_signals[sell_condition] = -1
                
                # Fill NaN values with 0
                symbol_signals = symbol_signals.fillna(0).astype(int)
                
                # Update signals DataFrame
                signals.loc[symbol] = symbol_signals
                
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
        position_value = self.portfolio_value * self.position_size
        
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
