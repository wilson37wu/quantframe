"""ICT (Inner Circle Trader) strategy implementation.

This module implements a basic ICT trading strategy based on market structure,
order blocks, and fair value gaps. Key features include:
- Fair Value Gap (FVG) identification
- Order Block detection
- Breaker Block patterns
- Institutional order flow analysis
- Market structure based entry/exit
"""

from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from .base import BaseStrategy, Position, Signal
from .config.ict_config import ICTConfig

class ICTStrategy(BaseStrategy):
    """ICT trading strategy using institutional concepts.
    
    This strategy implements core ICT concepts including:
    - Fair Value Gaps (FVG) for potential reversal zones
    - Order Blocks for institutional trading zones
    - Breaker Blocks for market structure shifts
    - Volume analysis for institutional activity
    """
    
    def __init__(self, config: ICTConfig):
        """Initialize ICT strategy.
        
        Args:
            config: ICT strategy configuration
        """
        super().__init__()
        self.config = config
        
        # Strategy parameters
        self.fvg_threshold = config.fvg_threshold
        self.ob_lookback = config.ob_lookback
        self.volume_threshold = config.volume_threshold
        self.stop_loss = config.stop_loss
        self.take_profit = config.take_profit
        self.risk_per_trade = config.risk_per_trade
        self.min_volume = config.min_volume
        self.max_positions = config.max_positions
        
        # State variables
        self.positions: Dict[str, Position] = {}
        self.portfolio_value = 100000.0  # Initial portfolio value
        self.cash = self.portfolio_value
        self.equity = self.portfolio_value
        
        # Market structure tracking
        self.swing_highs = []
        self.swing_lows = []
        self.order_blocks = []
        self.fvgs = []
        
    def calculate_position_size(self, symbol: str, price: float, signal: int) -> float:
        """Calculate position size based on risk parameters.
        
        Args:
            symbol: Trading symbol
            price: Current price
            signal: Trade direction (1 for long, -1 for short)
            
        Returns:
            Position size in base currency
        """
        if len(self.positions) >= self.max_positions:
            return 0.0
            
        risk_amount = self.portfolio_value * self.risk_per_trade
        stop_distance = price * self.stop_loss
        
        if stop_distance == 0:
            return 0.0
            
        size = risk_amount / stop_distance
        return size * signal
        
    def get_portfolio_state(self) -> Dict[str, Any]:
        """Get current portfolio state.
        
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
            'risk_metrics': {
                'max_drawdown': self.stop_loss,
                'position_count': len(self.positions),
                'risk_per_trade': self.risk_per_trade
            }
        }
        
    def get_stop_loss(self, symbol: str) -> Optional[float]:
        """Get stop loss price for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Stop loss price or None if no position
        """
        if symbol not in self.positions:
            return None
            
        position = self.positions[symbol]
        if position.size > 0:  # Long position
            return position.entry_price * (1 - self.stop_loss)
        else:  # Short position
            return position.entry_price * (1 + self.stop_loss)
        
    def generate_signals(self, data: pd.DataFrame) -> Dict[str, Signal]:
        """Generate trading signals based on ICT concepts.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            Dictionary of trading signals by symbol
        """
        signals = {}
        
        # Update market structure
        self.fvgs = self.find_fair_value_gaps(data)
        self.order_blocks = self.find_order_blocks(data)
        
        current_price = data.iloc[-1]['close']
        current_volume = data.iloc[-1]['volume']
        
        # Skip if volume is too low
        if current_volume < self.min_volume:
            return signals
            
        # Check for FVG signals
        for fvg in self.fvgs:
            # Price approaching FVG zone
            if fvg['direction'] == 1:  # Bullish FVG
                if current_price <= fvg['bottom']:
                    signals['BTCUSDT'] = Signal(
                        symbol='BTCUSDT',
                        direction=1,
                        strength=fvg['size'],
                        timestamp=data.index[-1],
                        metadata={'type': 'fvg', 'size': fvg['size']}
                    )
            else:  # Bearish FVG
                if current_price >= fvg['top']:
                    signals['BTCUSDT'] = Signal(
                        symbol='BTCUSDT',
                        direction=-1,
                        strength=fvg['size'],
                        timestamp=data.index[-1],
                        metadata={'type': 'fvg', 'size': fvg['size']}
                    )
                    
        # Check for Order Block signals
        for ob in self.order_blocks:
            # Price returning to Order Block
            if ob['direction'] == 1:  # Bullish OB
                if current_price <= ob['top'] and current_price >= ob['bottom']:
                    signals['BTCUSDT'] = Signal(
                        symbol='BTCUSDT',
                        direction=1,
                        strength=ob['strength'],
                        timestamp=data.index[-1],
                        metadata={'type': 'ob', 'strength': ob['strength']}
                    )
            else:  # Bearish OB
                if current_price <= ob['top'] and current_price >= ob['bottom']:
                    signals['BTCUSDT'] = Signal(
                        symbol='BTCUSDT',
                        direction=-1,
                        strength=ob['strength'],
                        timestamp=data.index[-1],
                        metadata={'type': 'ob', 'strength': ob['strength']}
                    )
                    
        return signals
        
    def update(self, timestamp: pd.Timestamp, data: pd.DataFrame) -> None:
        """Update strategy state with new data.
        
        Args:
            timestamp: Current timestamp
            data: New market data
        """
        # Generate signals
        signals = self.generate_signals(data)
        
        current_price = data.iloc[-1]['close']
        
        # Process signals
        for symbol, signal in signals.items():
            # Calculate position size
            size = self.calculate_position_size(symbol, current_price, signal.direction)
            
            if size != 0:
                # Close existing position if opposite direction
                if symbol in self.positions:
                    current_position = self.positions[symbol]
                    if np.sign(current_position.size) != np.sign(signal.direction):
                        self.close_position(symbol, timestamp, current_price)
                
                # Open new position
                self.positions[symbol] = Position(
                    symbol=symbol,
                    entry_time=timestamp,
                    entry_price=current_price,
                    size=size
                )
                
                # Update cash and equity
                self.cash -= abs(size * current_price)
                self.equity = self.cash + sum(
                    pos.size * current_price for pos in self.positions.values()
                )
        
        # Update existing positions
        positions_to_close = []
        for symbol, position in self.positions.items():
            # Update unrealized PnL
            position.unrealized_pnl = (current_price - position.entry_price) * position.size
            
            # Check stop loss and take profit
            returns = position.unrealized_pnl / (position.entry_price * abs(position.size))
            
            if returns <= -self.stop_loss or returns >= self.take_profit:
                positions_to_close.append(symbol)
                
        # Close positions
        for symbol in positions_to_close:
            self.close_position(symbol, timestamp, current_price)
            
    def close_position(self, symbol: str, timestamp: pd.Timestamp, price: float) -> None:
        """Close an existing position.
        
        Args:
            symbol: Trading symbol
            timestamp: Exit timestamp
            price: Exit price
        """
        if symbol in self.positions:
            position = self.positions[symbol]
            position.close(timestamp, price)
            
            # Update portfolio value and cash
            self.cash += abs(position.size * price)
            self.portfolio_value += position.realized_pnl
            self.equity = self.portfolio_value
            
            del self.positions[symbol]

    def find_fair_value_gaps(self, data: pd.DataFrame) -> List[Dict]:
        """Identify Fair Value Gaps in price action.
        
        A Fair Value Gap occurs when price moves rapidly, leaving an unfilled gap
        that often acts as a magnet for price to return to.
        
        Args:
            data: OHLCV DataFrame with at least 3 bars
            
        Returns:
            List of FVGs with their properties (price levels, direction)
        """
        fvgs = []
        
        if len(data) < 3:
            return fvgs
            
        for i in range(1, len(data)-1):
            # Bullish FVG
            if data.iloc[i-1]['low'] > data.iloc[i+1]['high']:
                gap_size = data.iloc[i-1]['low'] - data.iloc[i+1]['high']
                if gap_size >= self.fvg_threshold * data.iloc[i]['close']:
                    fvgs.append({
                        'timestamp': data.index[i],
                        'direction': 1,  # Bullish
                        'top': data.iloc[i-1]['low'],
                        'bottom': data.iloc[i+1]['high'],
                        'size': gap_size
                    })
                    
            # Bearish FVG
            elif data.iloc[i-1]['high'] < data.iloc[i+1]['low']:
                gap_size = data.iloc[i+1]['low'] - data.iloc[i-1]['high']
                if gap_size >= self.fvg_threshold * data.iloc[i]['close']:
                    fvgs.append({
                        'timestamp': data.index[i],
                        'direction': -1,  # Bearish
                        'top': data.iloc[i+1]['low'],
                        'bottom': data.iloc[i-1]['high'],
                        'size': gap_size
                    })
                    
        return fvgs

    def find_order_blocks(self, data: pd.DataFrame) -> List[Dict]:
        """Identify potential Order Blocks.
        
        Order Blocks are areas where significant institutional orders were filled
        before a strong move, often serving as support/resistance.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            List of Order Blocks with their properties
        """
        order_blocks = []
        
        if len(data) < self.ob_lookback:
            return order_blocks
            
        for i in range(self.ob_lookback, len(data)):
            window = data.iloc[i-self.ob_lookback:i]
            
            # Look for high volume candles before strong moves
            avg_volume = window['volume'].mean()
            for j in range(len(window)-1):
                if window.iloc[j]['volume'] > self.volume_threshold * avg_volume:
                    # Check for subsequent move
                    future_move = (window.iloc[-1]['close'] - window.iloc[j]['close']) / window.iloc[j]['close']
                    
                    if abs(future_move) > self.fvg_threshold:
                        order_blocks.append({
                            'timestamp': window.index[j],
                            'direction': np.sign(future_move),
                            'top': window.iloc[j]['high'],
                            'bottom': window.iloc[j]['low'],
                            'volume': window.iloc[j]['volume'],
                            'strength': abs(future_move)
                        })
                        
        return order_blocks
