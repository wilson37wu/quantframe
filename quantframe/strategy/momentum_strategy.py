from ..strategy.base import Signal, BaseStrategy
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from ..models.position import Position

class MomentumStrategy(BaseStrategy):
    """A momentum trading strategy that uses price trends and technical indicators."""

    def __init__(self, config_manager):
        """Initialize the Momentum Strategy.
        
        Args:
            config_manager: Strategy configuration manager
        """
        self.config_manager = config_manager
        self.config = None
        self.signals = []
        self.positions = {}
        
        # Initialize state with default configuration
        self.initialize_state(self.config_manager.get_default_config('momentum'))

    def initialize_state(self, config: Dict):
        """Initialize strategy state with configuration.
        
        Args:
            config: Strategy configuration dictionary
        """
        self.config = config
        
        # Initialize strategy parameters with default values
        self.config['momentum_period'] = config.get('momentum_period', 14)
        self.config['rsi_period'] = config.get('rsi_period', 14)
        self.config['macd_fast'] = config.get('macd_fast', 12)
        self.config['macd_slow'] = config.get('macd_slow', 26)
        self.config['macd_signal'] = config.get('macd_signal', 9)
        self.config['roc_threshold'] = config.get('roc_threshold', 0.02)
        self.config['roc_weight'] = config.get('roc_weight', 0.4)
        self.config['rsi_weight'] = config.get('rsi_weight', 0.3)
        self.config['macd_weight'] = config.get('macd_weight', 0.3)
        self.config['signal_threshold'] = config.get('signal_threshold', 0.1)
        self.config['stop_loss_pct'] = config.get('stop_loss_pct', 0.02)
        self.config['take_profit_pct'] = config.get('take_profit_pct', 0.05)
        self.config['max_position_size'] = config.get('max_position_size', 100.0)
        self.config['base_order_size'] = config.get('base_order_size', 1.0)
        self.config['volatility_scaling'] = config.get('volatility_scaling', True)
        self.config['volatility_lookback'] = config.get('volatility_lookback', 20)
        self.config['volatility_multiplier'] = config.get('volatility_multiplier', 1.0)
        self.config['rsi_overbought'] = config.get('rsi_overbought', 70)
        self.config['rsi_oversold'] = config.get('rsi_oversold', 30)
        self.config['macd_threshold'] = config.get('macd_threshold', 0.01)
        
        # Reset state variables
        self.signals = []
        self.positions = {}

    def _debug_data(self, data: pd.DataFrame, location: str):
        """Debug helper to print data information.
        
        Args:
            data: DataFrame to debug
            location: String indicating where the debug is called from
        """
        print(f"\n=== Debug Info from {location} ===")
        print(f"DataFrame shape: {data.shape}")
        print(f"DataFrame index type: {type(data.index)}")
        if isinstance(data.index, pd.MultiIndex):
            print(f"MultiIndex levels: {data.index.names}")
        print(f"DataFrame columns: {data.columns.tolist()}")
        print(f"DataFrame types:\n{data.dtypes}")
        print(f"First row:\n{data.iloc[0]}")
        print(f"Last row:\n{data.iloc[-1]}")
        if 'Close' in data.columns:
            print(f"\nClose column type: {type(data['Close'])}")
            print(f"Close values head:\n{data['Close'].head()}")
            print(f"Close values tail:\n{data['Close'].tail()}")
        print("=" * 50)

    def calculate_momentum_signal(self, data: pd.DataFrame) -> float:
        """Calculate momentum signal based on multiple indicators.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            Signal strength between -1 and 1
        """
        try:
            # Debug data before calculations
            self._debug_data(data, "calculate_momentum_signal - start")
            
            # Type checking and data validation
            if not isinstance(data, pd.DataFrame):
                raise TypeError(f"Expected DataFrame, got {type(data)}")
            
            required_columns = ['Close']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Ensure we're working with numeric data
            if not pd.api.types.is_numeric_dtype(data['Close']):
                raise TypeError(f"Close column must be numeric, got {data['Close'].dtype}")
            
            # Calculate indicators with explicit type conversion
            close_series = pd.to_numeric(data['Close'], errors='coerce')
            
            # Calculate momentum indicators
            roc = close_series.pct_change(self.config['momentum_period'])
            rsi = self._calculate_rsi(close_series, self.config['rsi_period'])
            macd = self._calculate_macd(close_series)
            
            # Debug intermediate calculations
            print(f"\nROC last value type: {type(roc.iloc[-1])}")
            print(f"RSI last value type: {type(rsi.iloc[-1])}")
            print(f"MACD last value type: {type(macd['macd'].iloc[-1])}")
            
            # Get the latest values with explicit conversion to float
            roc_last = float(roc.iloc[-1])
            rsi_last = float(rsi.iloc[-1])
            macd_last = float(macd['macd'].iloc[-1])
            
            # Combine signals
            momentum_signal = 0.0
            
            # ROC signal
            roc_signal = float(np.clip(roc_last / self.config['roc_threshold'], -1, 1))
            momentum_signal += roc_signal * self.config['roc_weight']
            
            # RSI signal
            rsi_signal = 0.0
            if rsi_last > self.config['rsi_overbought']:
                rsi_signal = -1.0
            elif rsi_last < self.config['rsi_oversold']:
                rsi_signal = 1.0
            momentum_signal += rsi_signal * self.config['rsi_weight']
            
            # MACD signal
            macd_signal = float(np.sign(macd_last) * min(abs(macd_last) / self.config['macd_threshold'], 1))
            momentum_signal += macd_signal * self.config['macd_weight']
            
            # Debug final calculation
            print(f"\nFinal momentum signal: {momentum_signal}")
            
            return float(np.clip(momentum_signal, -1, 1))
            
        except Exception as e:
            print(f"\nError in calculate_momentum_signal: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            print(f"Traceback:\n{''.join(traceback.format_tb(e.__traceback__))}")
            raise

    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Relative Strength Index.
        
        Args:
            prices: Price series
            period: RSI period
            
        Returns:
            RSI values
        """
        # Calculate price changes
        delta = prices.diff()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0.0)
        losses = -delta.where(delta < 0, 0.0)
        
        # Calculate average gains and losses
        avg_gains = gains.rolling(window=period).mean()
        avg_losses = losses.rolling(window=period).mean()
        
        # Calculate RS and RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        return rsi

    def _calculate_macd(self, prices: pd.Series) -> pd.DataFrame:
        """Calculate MACD indicator.
        
        Args:
            prices: Price series
            
        Returns:
            DataFrame with MACD and signal line
        """
        # Calculate EMAs
        fast_ema = prices.ewm(span=self.config['macd_fast'], adjust=False).mean()
        slow_ema = prices.ewm(span=self.config['macd_slow'], adjust=False).mean()
        
        # Calculate MACD and signal line
        macd = fast_ema - slow_ema
        signal = macd.ewm(span=self.config['macd_signal'], adjust=False).mean()
        
        return pd.DataFrame({'macd': macd, 'signal': signal})

    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """Generate trading signals based on momentum indicators.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            List of Signal objects
        """
        try:
            # Debug data
            self._debug_data(data, "generate_signals - start")
            
            if len(data) < max(self.config['momentum_period'], self.config['rsi_period'], 26):
                return []
            
            # Calculate momentum signal
            signal_strength = self.calculate_momentum_signal(data)
            
            # Generate signal only if strength exceeds threshold
            if abs(signal_strength) >= self.config['signal_threshold']:
                # Get symbol from index if it's a MultiIndex
                if isinstance(data.index, pd.MultiIndex):
                    symbol = data.index.get_level_values('symbol')[0]
                else:
                    symbol = 'Unknown'
                    
                # Create Signal object with full signal strength
                signal = Signal(
                    symbol=symbol,
                    direction=signal_strength,  # Use full signal strength instead of just sign
                    strength=float(abs(signal_strength)),
                    timestamp=data.index[-1],
                    metadata={
                        'price': float(data['Close'].iloc[-1]),
                        'volume': float(data['Volume'].iloc[-1]),
                        'momentum_score': float(signal_strength)
                    }
                )
                
                self.signals.append(signal)
                return [signal]
            
            return []
            
        except Exception as e:
            print(f"\nError in generate_signals: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            print(f"Traceback:\n{''.join(traceback.format_tb(e.__traceback__))}")
            raise

    def calculate_position_size(self, symbol: str, price: float, signal: int) -> float:
        """Calculate position size for a trade."""
        base_size = self.config['base_order_size']
        max_size = self.config['max_position_size']
        
        # Scale position size by signal strength
        position_size = base_size * abs(signal)
        
        # Apply volatility adjustment if enabled
        if self.config.get('volatility_scaling', True):
            volatility = self.calculate_volatility(symbol)  # You'll need to implement this
            vol_multiplier = 1 / (volatility * self.config['volatility_multiplier'])
            position_size *= vol_multiplier
        
        return min(max(position_size, base_size), max_size)

    def calculate_volatility(self, symbol: str) -> float:
        """Calculate historical volatility for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Historical volatility value
        """
        lookback = self.config.get('volatility_lookback', 20)
        
        # Get the last lookback days of data
        if symbol not in self.positions:
            return 1.0  # Default volatility if no position data
            
        # Calculate daily returns
        returns = pd.Series(self.positions[symbol]).pct_change().dropna()
        
        if len(returns) < 2:
            return 1.0
            
        # Calculate annualized volatility
        daily_vol = returns.std()
        annualized_vol = daily_vol * np.sqrt(252)  # Annualize using trading days
        
        return max(annualized_vol, 0.01)  # Minimum volatility of 1%

    def apply_risk_management(self, data: pd.DataFrame, signals: List[Dict]) -> List[Dict]:
        """Apply risk management rules to signals.
        
        Args:
            data: Market data DataFrame
            signals: List of signal dictionaries
            
        Returns:
            Filtered and modified signals
        """
        if not signals:
            return []
        
        managed_signals = []
        current_price = data['Close'].iloc[-1]
        
        for signal in signals:
            # Calculate stop loss and take profit levels
            if signal['direction'] == 1:  # Long position
                stop_loss = current_price * (1 - self.config['stop_loss_pct'])
                take_profit = current_price * (1 + self.config['take_profit_pct'])
            else:  # Short position
                stop_loss = current_price * (1 + self.config['stop_loss_pct'])
                take_profit = current_price * (1 - self.config['take_profit_pct'])
            
            # Add risk management metadata
            signal['metadata'].update({
                'stop_loss': stop_loss,
                'take_profit': take_profit
            })
            
            managed_signals.append(signal)
        
        return managed_signals

    def get_portfolio_state(self) -> Dict[str, Any]:
        """Get current portfolio state."""
        return {
            'positions': self.positions,
            'signals': self.signals,
            'total_exposure': sum(abs(pos.size) for pos in self.positions.values())
        }

    def get_stop_loss(self, symbol: str) -> Optional[float]:
        """Get stop loss price for a symbol."""
        if symbol not in self.positions:
            return None
        position = self.positions[symbol]
        return position.entry_price * (1 - self.config['base_stop_loss_pct'])

    def get_take_profit(self, symbol: str) -> Optional[float]:
        """Get take profit price for a symbol."""
        if symbol not in self.positions:
            return None
        position = self.positions[symbol]
        return position.entry_price * (1 + self.config['base_take_profit_pct'])

    def update(self, timestamp: pd.Timestamp, data: pd.DataFrame):
        """Update strategy state with new data."""
        try:
            # Get scalar values
            current_price = float(data['Close'].iloc[-1])
            
            # Generate signals
            new_signals = self.generate_signals(data)
            
            # Process signals and update positions
            for signal in new_signals:
                if signal.direction != 0:  # If there's an active signal
                    position_size = self.calculate_position_size(
                        signal.symbol,
                        current_price,
                        signal.direction
                    )
                    
                    # Update positions based on signal
                    if signal.symbol not in self.positions:
                        # Open new position
                        self.positions[signal.symbol] = Position(
                            symbol=signal.symbol,
                            entry_time=timestamp,
                            entry_price=current_price,
                            size=position_size
                        )
                    else:
                        # Update existing position
                        position = self.positions[signal.symbol]
                        if signal.direction != np.sign(position.size):
                            # Close position if signal is in opposite direction
                            position.close(timestamp, current_price)
                            del self.positions[signal.symbol]
                            
                            # Open new position
                            self.positions[signal.symbol] = Position(
                                symbol=signal.symbol,
                                entry_time=timestamp,
                                entry_price=current_price,
                                size=position_size
                            )
        except Exception as e:
            print(f"Error in update method: {str(e)}")
            raise
