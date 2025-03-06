from typing import Dict, List, Union
import numpy as np
import pandas as pd
from quantframe.strategy.base import BaseStrategy, Position
from quantframe.strategy.config_manager import StrategyConfigManager
from quantframe.strategy.signal import Signal  # Assuming Signal class is defined in .signal module
from typing import Optional


class GridTradingStrategy(BaseStrategy):
    """Grid Trading Strategy implementation."""

    def __init__(self, config_manager):
        """Initialize the Grid Trading Strategy.
        
        Args:
            config_manager: Strategy configuration manager
        """
        self.config_manager = config_manager
        
        # Load configuration using the correct method
        self.config = None
        self.initialize_state(self.config_manager.get_default_config('grid'))
        
        # Initialize state variables
        self.grid_levels = []
        self.signals = []
        self.positions = {}

    def initialize_state(self, config: Dict):
        """Initialize strategy state with configuration.
        
        Args:
            config: Strategy configuration dictionary
        """
        self.config = config
        
        # Initialize strategy parameters from config with defaults
        if isinstance(self.config, pd.DataFrame):
            # If config is a DataFrame, add missing columns with default values
            if 'base_order_size' not in self.config.columns:
                self.config['base_order_size'] = 1.0
            if 'size_scaling_factor' not in self.config.columns:
                self.config['size_scaling_factor'] = 1.0
            if 'dynamic_spacing' not in self.config.columns:
                self.config['dynamic_spacing'] = False
            if 'volatility_scaling' not in self.config.columns:
                self.config['volatility_scaling'] = False
            if 'vol_lookback' not in self.config.columns:
                self.config['vol_lookback'] = 20
            if 'vol_multiplier' not in self.config.columns:
                self.config['vol_multiplier'] = 1.0
            
            # Ensure all required grid parameters exist
            if 'grid_levels' not in self.config.columns:
                self.config['grid_levels'] = 10
            if 'grid_type' not in self.config.columns:
                self.config['grid_type'] = 'arithmetic'
            if 'rebalance_threshold' not in self.config.columns:
                self.config['rebalance_threshold'] = 0.02
            if 'max_drawdown' not in self.config.columns:
                self.config['max_drawdown'] = 0.1
            if 'base_stop_loss_pct' not in self.config.columns:
                self.config['base_stop_loss_pct'] = 0.05
            if 'base_take_profit_pct' not in self.config.columns:
                self.config['base_take_profit_pct'] = 0.05
            if 'max_position_size' not in self.config.columns:
                self.config['max_position_size'] = 100.0
        else:
            # If config is a dictionary, set default values
            self.grid_spacing = config.get('grid_spacing_percentage', 0.01)
            self.base_order_size = config.get('base_order_size', 1.0)
            self.size_scaling_factor = config.get('size_scaling_factor', 1.0)
            self.dynamic_spacing = config.get('dynamic_spacing', False)
            self.volatility_scaling = config.get('volatility_scaling', False)
            self.vol_lookback = config.get('vol_lookback', 20)
            self.vol_multiplier = config.get('vol_multiplier', 1.0)
            self.profit_taking = config.get('profit_taking', {'enabled': False})
            self.loss_handling = config.get('loss_handling', {'enabled': False})
            self.volatility_adjustments = config.get('volatility_adjustments', {'enabled': False})
            
            # Grid-specific parameters
            self.config['grid_levels'] = config.get('grid_levels', 10)
            self.config['grid_type'] = config.get('grid_type', 'arithmetic')
            self.config['rebalance_threshold'] = config.get('rebalance_threshold', 0.02)
            self.config['max_drawdown'] = config.get('max_drawdown', 0.1)
            self.config['base_stop_loss_pct'] = config.get('base_stop_loss_pct', 0.05)
            self.config['base_take_profit_pct'] = config.get('base_take_profit_pct', 0.05)
            self.config['max_position_size'] = config.get('max_position_size', 100.0)
        
        # Initialize state variables
        self.grid_levels = []
        self.signals = []
        self.positions = {}

    def create_grid(self, current_price: float) -> List[float]:
        """Create grid levels around current price.
        
        Args:
            current_price: Current market price
            
        Returns:
            List of grid price levels
        """
        try:
            # Debug data before grid creation
            print(f"\nCreating grid for price: {current_price}")
            print(f"Config type: {type(self.config)}")
            
            # Extract scalar values from config if it's a DataFrame
            if isinstance(self.config, pd.DataFrame):
                grid_levels = int(self.config['grid_levels'].iloc[0])
                grid_type = str(self.config['grid_type'].iloc[0])
                rebalance_threshold = float(self.config['rebalance_threshold'].iloc[0])
            else:
                grid_levels = int(self.config.get('grid_levels', 10))
                grid_type = str(self.config.get('grid_type', 'arithmetic'))
                rebalance_threshold = float(self.config.get('rebalance_threshold', 0.02))
            
            print(f"Grid type (scalar): {grid_type}")
            print(f"Grid levels (scalar): {grid_levels}")
            print(f"Rebalance threshold (scalar): {rebalance_threshold}")
            
            spacing = self.calculate_grid_spacing(pd.DataFrame({'Close': [current_price]}))
            half_levels = (grid_levels - 1) // 2
            
            if grid_type == 'arithmetic':
                # Create arithmetic grid with descending order
                upper_levels = [current_price + i * spacing for i in range(half_levels, 0, -1)]
                lower_levels = [current_price - i * spacing for i in range(1, half_levels + 1)]
                if grid_levels % 2 == 0:
                    # Add one more level to match grid_levels when even
                    lower_levels.append(current_price - (half_levels + 1) * spacing)
            else:
                # Create geometric grid with descending order
                upper_levels = [current_price * (1 + i * spacing) for i in range(half_levels, 0, -1)]
                lower_levels = [current_price * (1 - i * spacing) for i in range(1, half_levels + 1)]
                if grid_levels % 2 == 0:
                    # Add one more level to match grid_levels when even
                    lower_levels.append(current_price * (1 - (half_levels + 1) * spacing))
            
            self.grid_levels = sorted(upper_levels + [current_price] + lower_levels, reverse=True)
            return self.grid_levels
            
        except Exception as e:
            print(f"\nError in create_grid: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            print(f"Traceback:\n{''.join(traceback.format_tb(e.__traceback__))}")
            raise

    def rebalance_grid(self, current_price: float, old_grid: List[float]) -> List[float]:
        """Rebalance grid levels based on price movement.
        
        Args:
            current_price: Current market price
            old_grid: Previous grid levels
            
        Returns:
            Updated grid levels
        """
        # Calculate price deviation from grid center
        old_center = sum(old_grid) / len(old_grid)
        price_deviation = abs(current_price - old_center) / old_center
        
        # Rebalance if deviation exceeds threshold
        if price_deviation > self.config.rebalance_threshold:
            return self.create_grid(current_price)
        return old_grid

    def check_profit_taking(self, trade: Dict) -> bool:
        """Check if profit taking conditions are met.
        
        Args:
            trade: Dictionary containing trade information
            
        Returns:
            True if profit taking conditions are met
        """
        if not self.profit_taking['enabled']:
            return False
            
        profit = (trade['current_price'] - trade['entry_price']) / trade['entry_price']
        return abs(profit) >= self.profit_taking['threshold']

    def handle_losses(self, trade: Dict) -> Dict:
        """Handle losing positions according to strategy rules.
        
        Args:
            trade: Dictionary containing trade information
            
        Returns:
            Dictionary of actions to take
        """
        actions = {
            'stop_and_reset': self.loss_handling['stop_and_reset'],
            'hedge': self.loss_handling['hedge_at_extremes'],
            'close_position': False,
            'partial_close': self.loss_handling['partial_close']
        }
        
        loss = (trade['entry_price'] - trade['current_price']) / trade['entry_price']
        
        if loss > self.config.max_drawdown:
            actions['close_position'] = True
                
        return actions

    def calculate_grid_spacing(self, data: pd.DataFrame) -> float:
        """Calculate grid spacing based on market conditions.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            Grid spacing value
        """
        if self.dynamic_spacing:
            # Calculate ATR-based spacing
            volatility = data['Close'].pct_change().std()
            return max(self.grid_spacing * volatility, self.grid_spacing)
        return self.grid_spacing

    def calculate_volatility_adjustments(self, data: pd.DataFrame) -> Dict:
        """Calculate volatility-based adjustments for grid parameters.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            Dictionary of volatility adjustments
        """
        # Calculate volatility using the last vol_lookback periods
        volatility = data['Close'].pct_change().rolling(self.vol_lookback).std().iloc[-1]
        
        adjustments = {
            'spacing': self.calculate_grid_spacing(data),
            'position_size': self.config.get('base_order_size', 1.0),
            'spacing_multiplier': 1.0
        }
        
        if self.volatility_scaling and not pd.isna(volatility):
            vol_threshold_high = self.volatility_adjustments.get('vol_threshold_high', 0.02)
            vol_threshold_low = self.volatility_adjustments.get('vol_threshold_low', 0.01)
            
            if volatility > vol_threshold_high:
                adjustments['spacing_multiplier'] = 2.0  # Double multiplier for high volatility
                adjustments['position_size'] *= 0.5
            elif volatility < vol_threshold_low:
                adjustments['spacing_multiplier'] = 0.5  # Half multiplier for low volatility
                adjustments['position_size'] *= 1.5
        
        return adjustments

    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """Generate trading signals based on grid levels.

        Args:
            data: Market data DataFrame with OHLCV columns

        Returns:
            List of trading signals
            
        Raises:
            ValueError: If price data is invalid
        """
        try:
            # Debug data
            self._debug_data(data, "generate_signals - start")
            
            if data.empty or data['Close'].isnull().any() or (data['Close'] <= 0).any():
                raise ValueError("Invalid price data: prices must be positive and not null")
            
            signals = []
            current_price = float(data['Close'].iloc[-1])
            current_time = data.index[-1]
            
            # Get scalar config values
            if isinstance(self.config, pd.DataFrame):
                rebalance_threshold = float(self.config['rebalance_threshold'].iloc[0])
            else:
                rebalance_threshold = float(self.config.get('rebalance_threshold', 0.02))

            # Initialize grid if not already done
            if not self.grid_levels:
                self.create_grid(current_price)

            # Check each grid level and generate at least one signal
            signal_generated = False
            for level in self.grid_levels:
                if current_price <= level * (1 - rebalance_threshold):
                    # Buy signal if price drops below grid level
                    signals.append(Signal(
                        symbol=data.index.get_level_values('symbol')[0] if isinstance(data.index, pd.MultiIndex) else 'Unknown',
                        direction=1,  # Long
                        strength=abs(current_price - level) / level,
                        timestamp=current_time,
                        metadata={
                            'grid_level': level,
                            'grid_price': level * (1 - rebalance_threshold)
                        }
                    ))
                    signal_generated = True
                elif current_price >= level * (1 + rebalance_threshold):
                    # Sell signal if price rises above grid level
                    signals.append(Signal(
                        symbol=data.index.get_level_values('symbol')[0] if isinstance(data.index, pd.MultiIndex) else 'Unknown',
                        direction=-1,  # Short
                        strength=abs(current_price - level) / level,
                        timestamp=current_time,
                        metadata={
                            'grid_level': level,
                            'grid_price': level * (1 + rebalance_threshold)
                        }
                    ))
                    signal_generated = True

            # If no signals were generated, create a signal based on closest grid level
            if not signal_generated:
                closest_level = min(self.grid_levels, key=lambda x: abs(x - current_price))
                signals.append(Signal(
                    symbol=data.index.get_level_values('symbol')[0] if isinstance(data.index, pd.MultiIndex) else 'Unknown',
                    direction=1 if current_price < closest_level else -1,
                    strength=abs(current_price - closest_level) / closest_level,
                    timestamp=current_time,
                    metadata={
                        'grid_level': closest_level,
                        'grid_price': closest_level * (1 + (0.5 * rebalance_threshold) * (-1 if current_price < closest_level else 1))
                    }
                ))

            return signals
            
        except Exception as e:
            print(f"\nError in generate_signals: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            print(f"Traceback:\n{''.join(traceback.format_tb(e.__traceback__))}")
            raise

    def calculate_position_size(self, signal: Signal, data: pd.DataFrame) -> float:
        """Calculate position size for a trade.
        
        Args:
            signal: Trading signal object
            data: Current market data
            
        Returns:
            Position size in units
        """
        try:
            # Get current price
            current_price = float(data['Close'].iloc[-1])
            
            # Start with base order size
            if isinstance(self.config, pd.DataFrame):
                base_size = float(self.config['base_order_size'].iloc[0])
                max_size = float(self.config['max_position_size'].iloc[0])
                volatility_scaling = bool(self.config['volatility_scaling'].iloc[0])
                vol_multiplier = float(self.config['vol_multiplier'].iloc[0])
            else:
                base_size = float(self.config.get('base_order_size', 1.0))
                max_size = float(self.config.get('max_position_size', 100.0))
                volatility_scaling = bool(self.config.get('volatility_scaling', False))
                vol_multiplier = float(self.config.get('vol_multiplier', 1.0))
            
            size = base_size
            
            # Apply volatility scaling if enabled
            if volatility_scaling:
                volatility = self._calculate_volatility(data)
                size *= (1 / (volatility * vol_multiplier))
                
            # Scale by signal strength
            size *= abs(signal.strength)
            
            # Ensure size is within bounds
            return min(max(size, base_size), max_size)
            
        except Exception as e:
            print(f"\nError in calculate_position_size: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            print(f"Traceback:\n{''.join(traceback.format_tb(e.__traceback__))}")
            raise

    def apply_risk_management(self, data: pd.DataFrame, signals: List[Signal]) -> List[Signal]:
        """Apply risk management rules to the grid trading strategy.
        
        Args:
            data: Market data DataFrame
            signals: List of trading signals
            
        Returns:
            Risk-adjusted signals
        """
        risk_adjusted_signals = signals.copy()
        
        # Calculate volatility for the last period
        volatility = data['Close'].pct_change().rolling(self.vol_lookback).std().iloc[-1]
        high_volatility = volatility > self.volatility_adjustments.get('vol_threshold_high', 0.02)
        
        for signal in risk_adjusted_signals:
            # Cancel signals during high volatility if configured
            if high_volatility and self.volatility_adjustments.get('enabled', False):
                signal.direction = 0
            
            # Apply profit taking rules
            if self.profit_taking.get('enabled', False):
                grid_level = signal.metadata['grid_level']
                current_price = data['Close'].iloc[-1]
                profit = (current_price - grid_level) / grid_level
                
                if abs(profit) >= self.profit_taking.get('threshold', 0.05):
                    if not self.profit_taking.get('reinvest', False):
                        signal.direction = 0
            
            # Apply loss handling rules
            if self.loss_handling.get('stop_and_reset', False):
                if signal.direction != 0:  # Only for active signals
                    grid_level = signal.metadata['grid_level']
                    current_price = data['Close'].iloc[-1]
                    loss = (grid_level - current_price) / grid_level
                    
                    if loss > self.config.get('max_drawdown', 0.1):
                        signal.direction = 0
                        # Reset grid if configured
                        self.create_grid(current_price)
        
        return risk_adjusted_signals

    def get_portfolio_state(self):
        """Get current portfolio state"""
        return {
            'positions': self.positions,
            'grid_levels': self.grid_levels,
            'total_exposure': sum(abs(pos.size * pos.direction) for pos in self.positions.values())
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
        return position.entry_price * (1 - self.config['base_stop_loss_pct'])

    def get_take_profit(self, symbol: str) -> Optional[float]:
        """Get take profit price for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Take profit price or None if no position
        """
        if symbol not in self.positions:
            return None
        position = self.positions[symbol]
        return position.entry_price * (1 + self.config['base_take_profit_pct'])

    def update(self, timestamp: pd.Timestamp, data: pd.DataFrame):
        """Update strategy state and generate orders.
        
        Args:
            timestamp: Current timestamp
            data: Market data DataFrame
        """
        try:
            # Ensure we're working with scalar values
            current_price = float(data['Close'].iloc[-1])
            
            # Generate signals
            new_signals = self.generate_signals(data)
            self.signals.extend(new_signals)

            # Process signals and update positions
            for signal in new_signals:
                if signal.direction != 0:  # If there's an active signal
                    position_size = self.calculate_position_size(
                        signal,
                        data
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
            logger.error(f"Error in update method: {str(e)}")
            raise

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
