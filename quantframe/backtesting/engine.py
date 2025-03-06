"""Event-driven backtesting engine for trading strategies.

This module provides a comprehensive backtesting framework for evaluating trading
strategies. Key features include:
- Event-driven architecture for realistic simulation
- Transaction cost modeling (commission and slippage)
- Position and trade tracking
- Performance analytics
- Visualization tools

The framework consists of three main components:
- Backtester: Core simulation engine
- BacktestResult: Results container with visualization
- Trade: Data structure for completed trades

Example:
    >>> from quantframe.strategy import MyStrategy
    >>> strategy = MyStrategy(params={'ma_period': 20})
    >>> backtester = Backtester(strategy, initial_capital=100000)
    >>> result = backtester.run(data)
    >>> result.plot_results()
    >>> print(f"Sharpe Ratio: {result.metrics['sharpe_ratio']:.2f}")
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
from ..strategy.base import BaseStrategy, Position
from ..analytics.performance import calculate_performance_metrics

@dataclass
class Trade:
    """Represents a completed trade with its full lifecycle information.
    
    This class captures all relevant information about a completed trade,
    including entry/exit points, size, profitability, and metadata.
    
    Attributes:
        symbol (str): Trading symbol/ticker
        entry_time (pd.Timestamp): Trade entry timestamp
        exit_time (pd.Timestamp): Trade exit timestamp
        entry_price (float): Entry price
        exit_price (float): Exit price
        direction (int): Trade direction (1: long, -1: short)
        size (float): Position size in base units
        pnl (float): Realized profit/loss
        return_pct (float): Percentage return on the trade
        reason (str): Reason for trade exit (e.g., 'signal', 'stop_loss')
        metadata (Dict[str, Any]): Additional trade information
        
    Example:
        >>> trade = Trade(
        ...     symbol="AAPL",
        ...     entry_time=pd.Timestamp("2023-01-01"),
        ...     exit_time=pd.Timestamp("2023-01-02"),
        ...     entry_price=150.0,
        ...     exit_price=155.0,
        ...     direction=1,
        ...     size=100,
        ...     pnl=500.0,
        ...     return_pct=0.0333,
        ...     reason="take_profit",
        ...     metadata={"strategy": "momentum"}
        ... )
    """
    symbol: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    direction: int
    size: float
    pnl: float
    return_pct: float
    reason: str
    metadata: Dict[str, Any]

class BacktestResult:
    """Container for backtest results with analysis and visualization capabilities.
    
    This class provides a structured way to store and analyze backtest results,
    including trade history, equity curve, position history, and performance metrics.
    It also includes visualization methods for result analysis.
    
    Attributes:
        trades (List[Trade]): List of completed trades
        equity_curve (pd.Series): Portfolio equity over time
        positions (pd.DataFrame): Position sizes over time
        metrics (Dict[str, float]): Performance metrics (e.g., Sharpe ratio)
        
    Example:
        >>> result = backtester.run(data)
        >>> print(f"Total Return: {result.metrics['total_return']:.2%}")
        >>> result.plot_results()
    """
    
    def __init__(self, 
                 trades: List[Trade],
                 equity_curve: pd.Series,
                 positions: pd.DataFrame,
                 metrics: Dict[str, float]):
        """Initialize backtest results container.
        
        Args:
            trades (List[Trade]): List of completed trades
            equity_curve (pd.Series): Portfolio equity over time
            positions (pd.DataFrame): Position sizes over time
            metrics (Dict[str, float]): Performance metrics
        """
        self.trades = trades
        self.equity_curve = equity_curve
        self.positions = positions
        self.metrics = metrics
    
    def plot_results(self, figsize: Tuple[int, int] = (15, 10)):
        """Plot backtest results in a multi-panel figure.
        
        Creates a figure with three subplots:
        1. Equity curve showing portfolio value over time
        2. Drawdown chart showing portfolio drawdowns
        3. Position size chart showing exposure over time
        
        Args:
            figsize (Tuple[int, int], optional): Figure size in inches.
                Defaults to (15, 10).
                
        Returns:
            matplotlib.figure.Figure: The created figure object
            
        Example:
            >>> result = backtester.run(data)
            >>> fig = result.plot_results(figsize=(12, 8))
            >>> fig.savefig('backtest_results.png')
        """
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize)
        
        # Plot equity curve
        ax1.plot(self.equity_curve.index, self.equity_curve.values)
        ax1.set_title('Equity Curve')
        ax1.grid(True)
        
        # Plot drawdown
        drawdown = (self.equity_curve - self.equity_curve.expanding().max()) / self.equity_curve.expanding().max()
        ax2.fill_between(drawdown.index, drawdown.values, 0, color='red', alpha=0.3)
        ax2.set_title('Drawdown')
        ax2.grid(True)
        
        # Plot position sizes over time
        ax3.plot(self.positions.index, self.positions.values)
        ax3.set_title('Position Size')
        ax3.grid(True)
        
        plt.tight_layout()
        return fig

class Backtester:
    """Event-driven backtesting engine for trading strategies.
    
    This class implements a realistic backtesting engine that simulates
    trading strategy execution with:
    - Transaction cost modeling (commission and slippage)
    - Position tracking and P&L calculation
    - Event-driven architecture for realistic order execution
    - Support for multiple symbols
    - Stop-loss and take-profit handling
    
    Attributes:
        strategy (BaseStrategy): Trading strategy instance
        initial_capital (float): Initial portfolio capital
        commission (float): Commission rate per trade
        slippage (float): Slippage rate per trade
        equity (float): Current portfolio equity
        positions (Dict[str, Position]): Currently open positions
        trades (List[Trade]): Completed trades
        equity_curve (List): Portfolio equity history
        position_history (List): Position size history
        
    Example:
        >>> strategy = MeanReversionStrategy(lookback=20)
        >>> backtester = Backtester(
        ...     strategy=strategy,
        ...     initial_capital=100000,
        ...     commission=0.001,  # 0.1%
        ...     slippage=0.001     # 0.1%
        ... )
        >>> result = backtester.run(data)
    """
    
    def __init__(self, 
                 strategy: BaseStrategy,
                 initial_capital: float = 1_000_000,
                 commission: float = 0.001,
                 slippage: float = 0.001):
        """Initialize the backtesting engine.
        
        Args:
            strategy (BaseStrategy): Trading strategy to test
            initial_capital (float, optional): Starting capital.
                Defaults to 1,000,000.
            commission (float, optional): Commission rate as decimal.
                Defaults to 0.001 (0.1%).
            slippage (float, optional): Slippage rate as decimal.
                Defaults to 0.001 (0.1%).
        """
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        
        self.reset()
    
    def reset(self) -> None:
        """Reset the backtester to initial state.
        
        Clears all tracking variables and resets equity to initial capital.
        Should be called before starting a new backtest.
        """
        self.equity = self.initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve = []
        self.position_history = []
    
    def run(self, data: pd.DataFrame, start_date: Optional[pd.Timestamp] = None, 
            end_date: Optional[pd.Timestamp] = None) -> BacktestResult:
        """Run backtest simulation over the provided data.
        
        Simulates strategy execution on historical data, processing each
        timestamp sequentially to maintain realistic causality. Handles:
        1. Position P&L updates
        2. Strategy signal generation
        3. Order execution
        4. Performance tracking
        
        Args:
            data (pd.DataFrame): Historical price data with OHLCV columns
            start_date (Optional[pd.Timestamp], optional): Backtest start date.
                If None, uses earliest available date.
            end_date (Optional[pd.Timestamp], optional): Backtest end date.
                If None, uses latest available date.
            
        Returns:
            BacktestResult: Container with backtest results and analytics
            
        Example:
            >>> data = pd.DataFrame({
            ...     'open': [100, 101, 102],
            ...     'high': [102, 103, 104],
            ...     'low': [99, 100, 101],
            ...     'close': [101, 102, 103],
            ...     'volume': [1000, 1100, 1200]
            ... }, index=pd.date_range('2023-01-01', periods=3))
            >>> result = backtester.run(
            ...     data,
            ...     start_date=pd.Timestamp('2023-01-01'),
            ...     end_date=pd.Timestamp('2023-01-03')
            ... )
        """
        self.reset()
        
        # Filter data by date range if specified
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]
        
        # Run simulation
        for timestamp in data.index:
            current_data = data.loc[:timestamp]
            current_price = current_data['close'].iloc[-1].item()
            
            # Update positions P&L
            self._update_positions(current_price, timestamp)
            
            # Get strategy orders
            orders = self.strategy.update(current_data)
            
            # Process orders
            for order in orders:
                self._process_order(order, current_price, timestamp)
            
            # Record equity and positions
            self.equity_curve.append((timestamp, self.equity))
            self.position_history.append((timestamp, 
                                       sum(pos.size * pos.direction 
                                           for pos in self.positions.values())))
        
        # Create results
        equity_curve = pd.Series(dict(self.equity_curve))
        positions = pd.Series(dict(self.position_history))
        
        # Calculate performance metrics
        metrics = calculate_performance_metrics(equity_curve)
        
        return BacktestResult(
            trades=self.trades,
            equity_curve=equity_curve,
            positions=positions,
            metrics=metrics
        )
    
    def _process_order(self, order: Dict[str, Any], current_price: float, 
                      timestamp: pd.Timestamp) -> None:
        """Process a single trading order.
        
        Handles order execution including:
        - Transaction cost calculation
        - Position entry/exit
        - P&L calculation
        - Trade recording
        
        Args:
            order (Dict[str, Any]): Order specification containing:
                - symbol: Trading symbol
                - direction: Trade direction (1, -1, 0)
                - size: Position size
                - reason: Order reason (optional)
                - metadata: Additional order data (optional)
            current_price (float): Current market price
            timestamp (pd.Timestamp): Current timestamp
            
        Note:
            This is an internal method used by the run() method.
            Transaction costs include both commission and slippage.
        """
        symbol = order['symbol']
        direction = order['direction']
        size = order['size']
        reason = order.get('reason', 'signal')
        
        # Calculate transaction costs
        transaction_cost = current_price * size * (self.commission + self.slippage)
        
        if direction == 0:  # Close position
            if symbol in self.positions:
                position = self.positions[symbol]
                entry_price = position.entry_price
                position_direction = position.direction
                position_size = position.size
                
                # Calculate P&L
                price_diff = current_price - entry_price
                pnl = price_diff * position_size * position_direction - transaction_cost
                return_pct = (price_diff / entry_price) * position_direction
                
                # Record trade
                self.trades.append(Trade(
                    symbol=symbol,
                    entry_time=position.entry_time,
                    exit_time=timestamp,
                    entry_price=entry_price,
                    exit_price=current_price,
                    direction=position_direction,
                    size=position_size,
                    pnl=pnl,
                    return_pct=return_pct,
                    reason=reason,
                    metadata=position.metadata if position.metadata else {}
                ))
                
                # Update equity
                self.equity += pnl
                
                # Remove position
                del self.positions[symbol]
        
        else:  # Open new position
            # Record position
            self.positions[symbol] = Position(
                symbol=symbol,
                direction=direction,
                size=size,
                entry_price=current_price,
                entry_time=timestamp,
                metadata=order.get('metadata', {})
            )
            
            # Deduct transaction costs
            self.equity -= transaction_cost
    
    def _update_positions(self, current_price: float, timestamp: pd.Timestamp) -> None:
        """Update unrealized P&L and check risk management rules.
        
        Updates the unrealized P&L for all open positions and checks
        stop-loss and take-profit conditions.
        
        Args:
            current_price (float): Current market price
            timestamp (pd.Timestamp): Current timestamp
            
        Note:
            This is an internal method used by the run() method.
            Stop-loss and take-profit levels are checked from position metadata.
        """
        for symbol, position in list(self.positions.items()):
            # Check stop loss and take profit
            entry_price = position.entry_price
            direction = position.direction
            size = position.size
            
            if 'stop_loss' in position.metadata and direction == 1 and current_price <= position.metadata['stop_loss']:
                self._process_order({'symbol': symbol, 'direction': 0, 'size': size, 
                                   'reason': 'stop_loss'}, current_price, timestamp)
            
            elif 'stop_loss' in position.metadata and direction == -1 and current_price >= position.metadata['stop_loss']:
                self._process_order({'symbol': symbol, 'direction': 0, 'size': size,
                                   'reason': 'stop_loss'}, current_price, timestamp)
            
            elif 'take_profit' in position.metadata and direction == 1 and current_price >= position.metadata['take_profit']:
                self._process_order({'symbol': symbol, 'direction': 0, 'size': size,
                                   'reason': 'take_profit'}, current_price, timestamp)
            
            elif 'take_profit' in position.metadata and direction == -1 and current_price <= position.metadata['take_profit']:
                self._process_order({'symbol': symbol, 'direction': 0, 'size': size,
                                   'reason': 'take_profit'}, current_price, timestamp)

class BacktestEngine:
    """Engine for running trading strategy backtests."""

    def __init__(self):
        """Initialize the backtest engine."""
        self.reset()
        self.portfolio = Portfolio()

    def reset(self):
        """Reset the backtest engine state."""
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        self.current_capital = 0.0
        if hasattr(self, 'portfolio'):
            self.portfolio.reset()
        else:
            self.portfolio = Portfolio()

    def run(self, strategy, data: pd.DataFrame, initial_capital: float = 100000.0) -> pd.DataFrame:
        """Run backtest for the strategy.
        
        Args:
            strategy: Trading strategy instance
            data: Market data DataFrame
            initial_capital: Initial capital for the backtest
            
        Returns:
            DataFrame with backtest results
        """
        try:
            # Initialize portfolio
            self.portfolio.reset(initial_capital)
            
            # Get all unique timestamps
            timestamps = data.index.get_level_values('date').unique()
            
            # Process each timestamp
            for timestamp in timestamps:
                # Get data up to current timestamp using proper index slicing
                historical_data = data.loc[data.index.get_level_values('date') <= timestamp]
                current_data = data.xs(timestamp, level='date')
                
                # Generate signals
                signals = strategy.generate_signals(historical_data)
                
                # Process signals
                for signal in signals:
                    if signal.direction != 0:  # If there's an active signal
                        position_size = strategy.calculate_position_size(
                            signal.symbol,
                            float(current_data.loc[signal.symbol, 'Close']),
                            signal.direction
                        )
                        
                        try:
                            # Get price for the specific symbol
                            symbol_price = float(current_data.loc[signal.symbol, 'Close'])
                            
                            # Execute trade
                            self.portfolio.execute_trade(
                                symbol=signal.symbol,
                                direction=signal.direction,
                                size=position_size,
                                price=symbol_price,
                                timestamp=timestamp
                            )
                        except KeyError as e:
                            print(f"Warning: Could not execute trade for symbol {signal.symbol}: {str(e)}")
                            continue
            
            return self.portfolio.get_history()
            
        except Exception as e:
            print(f"\nError in backtest run: {str(e)}")
            print(f"Error type: {type(e)}")
            print(f"Data structure:")
            print(f"Index levels: {data.index.names}")
            print(f"Columns: {data.columns.tolist()}")
            print(f"Sample data:\n{data.head()}")
            print(f"MultiIndex structure:\n{data.index.to_frame().head()}")
            import traceback
            print(f"Traceback:\n{''.join(traceback.format_tb(e.__traceback__))}")
            raise

class Portfolio:
    """Manages portfolio state during backtest."""
    
    def __init__(self):
        """Initialize portfolio."""
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        self.current_capital = 0.0
        
    def reset(self, initial_capital: float = 100000.0):
        """Reset portfolio state."""
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        self.current_capital = initial_capital
    
    def execute_trade(self, symbol: str, direction: int, size: float, 
                     price: float, timestamp: pd.Timestamp):
        """Execute a trade and update portfolio state."""
        cost = price * size
        
        if direction == 1:  # Buy
            if symbol not in self.positions:
                self.positions[symbol] = 0
            self.positions[symbol] += size
            self.current_capital -= cost
        else:  # Sell
            if symbol in self.positions:
                self.positions[symbol] = max(0, self.positions[symbol] - size)
            self.current_capital += cost
        
        # Record trade
        self.trades.append({
            "timestamp": timestamp,
            "symbol": symbol,
            "direction": direction,
            "price": price,
            "size": size,
            "cost": cost
        })
        
        # Update equity curve
        self.equity_curve.append({
            "timestamp": timestamp,
            "equity": self.calculate_equity(price)
        })
    
    def calculate_equity(self, current_price: float) -> float:
        """Calculate current portfolio value."""
        position_value = sum(pos * current_price for pos in self.positions.values())
        return self.current_capital + position_value
    
    def get_history(self) -> pd.DataFrame:
        """Get portfolio history as DataFrame."""
        return pd.DataFrame(self.equity_curve).set_index('timestamp')
