"""QuantFrame controller implementation."""
import os
from typing import List, Dict, Any, Optional
import pandas as pd
from datetime import datetime
from rich.console import Console
from rich.table import Table
import logging
from .strategy.base import BaseStrategy
from .strategy.momentum import MomentumStrategy
from .strategy.grid import GridStrategy
from .strategy.mean_reversion import MeanReversionStrategy
from .data.base import DataManager
from .brokers.ib.connection import IBConnection

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
PAPER_TRADING_PORT = 7497  # IB Paper Trading port
LIVE_TRADING_PORT = 7496   # IB Live Trading port

class QuantFrameController:
    """Controller class for QuantFrame trading system."""
    
    def __init__(self):
        """Initialize controller."""
        self.console = Console()
        self.data_manager = DataManager()
        self.strategy: Optional[BaseStrategy] = None
        self.broker: Optional[IBConnection] = None
        self.is_paper_trading = True  # Force paper trading mode
        self.strategies = {
            'grid': {
                'name': 'Grid',
                'description': 'Grid Trading Strategy implementation.',
                'class': GridStrategy
            },
            'momentum': {
                'name': 'Momentum',
                'description': 'Momentum Trading Strategy using multiple technical indicators.',
                'class': MomentumStrategy
            },
            'mean_reversion': {
                'name': 'Mean_reversion',
                'description': 'Mean Reversion Strategy using moving averages and Bollinger Bands.',
                'class': MeanReversionStrategy
            }
        }
    
    def print_welcome(self):
        """Print welcome message."""
        self.console.print("\nWelcome to QuantFrame Trading System")
        self.console.print("=" * 50)
        self.console.print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.console.print(f"Available strategies: {', '.join(self.strategies.keys())}")
        self.console.print("=" * 50 + "\n")
    
    def select_strategy(self) -> str:
        """
        Let user select a trading strategy.
        
        Returns:
            Selected strategy name
        """
        logger.info("Asking user to select a strategy")
        
        # Create strategy table
        strategy_table = Table(title="Available Strategies")
        strategy_table.add_column("ID", style="cyan")
        strategy_table.add_column("Name", style="magenta")
        strategy_table.add_column("Description", style="green")
        
        for strategy_id, strategy_info in self.strategies.items():
            strategy_table.add_row(
                strategy_id,
                strategy_info['name'],
                strategy_info['description']
            )
        
        self.console.print(strategy_table)
        
        while True:
            try:
                strategy_name = input("Select strategy [grid/momentum/mean_reversion] (grid): ").lower() or 'grid'
                if strategy_name in self.strategies:
                    logger.info(f"User selected strategy: {strategy_name}")
                    return strategy_name
                else:
                    self.console.print(f"Invalid strategy: {strategy_name}. Please try again.")
            except EOFError:
                logger.warning("EOFError encountered, using default strategy: grid")
                return 'grid'
    
    def configure_strategy(self, strategy_name: str) -> BaseStrategy:
        """
        Configure trading strategy parameters.
        
        Args:
            strategy_name: Name of the strategy to configure
            
        Returns:
            Configured strategy instance
        """
        logger.info(f"Asking user if they want to modify parameters for {strategy_name}")
        
        try:
            modify = input("Would you like to modify the default parameters? [y/n]: ").lower() == 'y'
        except EOFError:
            logger.warning("EOFError encountered, using default parameters")
            modify = False
        
        if strategy_name == 'momentum':
            if modify:
                try:
                    rsi_period = int(input("RSI period (14): ") or "14")
                    rsi_overbought = float(input("RSI overbought level (70): ") or "70")
                    rsi_oversold = float(input("RSI oversold level (30): ") or "30")
                    sma_fast = int(input("Fast SMA period (20): ") or "20")
                    sma_slow = int(input("Slow SMA period (50): ") or "50")
                    position_size = float(input("Position size as fraction (0.1): ") or "0.1")
                    stop_loss = float(input("Stop loss percentage (0.02): ") or "0.02")
                    take_profit = float(input("Take profit percentage (0.05): ") or "0.05")
                    
                    logger.info(f"Creating MomentumStrategy with parameters: rsi_period={rsi_period}, rsi_overbought={rsi_overbought}, rsi_oversold={rsi_oversold}, sma_fast={sma_fast}, sma_slow={sma_slow}, position_size={position_size}, stop_loss={stop_loss}, take_profit={take_profit}")
                    
                    logger.info(f"Instantiating MomentumStrategy with parameters: {rsi_period=}, {rsi_overbought=}, {rsi_oversold=}, {sma_fast=}, {sma_slow=}, {position_size=}, {stop_loss=}, {take_profit=}")
                    
                    return MomentumStrategy(
                        rsi_period=rsi_period,
                        rsi_overbought=rsi_overbought,
                        rsi_oversold=rsi_oversold,
                        sma_fast=sma_fast,
                        sma_slow=sma_slow,
                        position_size=position_size,
                        stop_loss=stop_loss,
                        take_profit=take_profit
                    )
                except (ValueError, EOFError) as e:
                    logger.warning(f"Error configuring parameters: {str(e)}. Using defaults.")
            else:
                # Set default parameters directly
                rsi_period = 14
                rsi_overbought = 70
                rsi_oversold = 30
                sma_fast = 20
                sma_slow = 50
                position_size = 0.1
                stop_loss = 0.02
                take_profit = 0.05
                
                logger.info(f"Creating MomentumStrategy with default parameters: rsi_period={rsi_period}, rsi_overbought={rsi_overbought}, rsi_oversold={rsi_oversold}, sma_fast={sma_fast}, sma_slow={sma_slow}, position_size={position_size}, stop_loss={stop_loss}, take_profit={take_profit}")
                
                logger.info(f"Instantiating MomentumStrategy with parameters: {rsi_period=}, {rsi_overbought=}, {rsi_oversold=}, {sma_fast=}, {sma_slow=}, {position_size=}, {stop_loss=}, {take_profit=}")
                
                return MomentumStrategy(
                    rsi_period=rsi_period,
                    rsi_overbought=rsi_overbought,
                    rsi_oversold=rsi_oversold,
                    sma_fast=sma_fast,
                    sma_slow=sma_slow,
                    position_size=position_size,
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )
            
        elif strategy_name == 'grid':
            if modify:
                try:
                    grid_size = float(input("Grid size percentage (0.01): ") or "0.01")
                    take_profit = float(input("Take profit percentage (0.05): ") or "0.05")
                    num_grids = int(input("Number of grids (10): ") or "10")
                    
                    return GridStrategy(
                        grid_size=grid_size,
                        take_profit=take_profit,
                        num_grids=num_grids
                    )
                except (ValueError, EOFError) as e:
                    logger.warning(f"Error configuring parameters: {str(e)}. Using defaults.")
            else:
                # Set default parameters directly
                grid_size = 0.01
                take_profit = 0.05
                num_grids = 10
                
                logger.info(f"Creating GridStrategy with default parameters: grid_size={grid_size}, take_profit={take_profit}, num_grids={num_grids}")
                
                return GridStrategy(
                    grid_size=grid_size,
                    take_profit=take_profit,
                    num_grids=num_grids
                )
            
        elif strategy_name == 'mean_reversion':
            if modify:
                try:
                    ma_period = int(input("Moving average period (20): ") or "20")
                    std_dev = float(input("Standard deviation multiplier (2.0): ") or "2.0")
                    stop_loss = float(input("Stop loss percentage (0.02): ") or "0.02")
                    take_profit = float(input("Take profit percentage (0.05): ") or "0.05")
                    
                    return MeanReversionStrategy(
                        ma_period=ma_period,
                        std_dev=std_dev,
                        stop_loss=stop_loss,
                        take_profit=take_profit
                    )
                except (ValueError, EOFError) as e:
                    logger.warning(f"Error configuring parameters: {str(e)}. Using defaults.")
            else:
                # Set default parameters directly
                ma_period = 20
                std_dev = 2.0
                stop_loss = 0.02
                take_profit = 0.05
                
                logger.info(f"Creating MeanReversionStrategy with default parameters: ma_period={ma_period}, std_dev={std_dev}, stop_loss={stop_loss}, take_profit={take_profit}")
                
                return MeanReversionStrategy(
                    ma_period=ma_period,
                    std_dev=std_dev,
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )
        
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")
    
    def get_symbols(self) -> List[str]:
        """
        Get trading symbols from user input.
        
        Returns:
            List of trading symbols
        """
        symbols = []
        while True:
            try:
                symbol = input("Enter symbol (or press Enter to finish): ").upper()
                if not symbol:
                    break
                symbols.append(symbol)
            except EOFError:
                break
        
        return symbols if symbols else ['AAPL']  # Default to AAPL if no symbols provided
    
    def get_timeframe(self) -> str:
        """
        Get timeframe from user input.
        
        Returns:
            Selected timeframe
        """
        try:
            timeframe = input("Enter timeframe [1m/5m/15m/30m/1h/4h/1d/1w/1mo] (1d): ") or "1d"
            return timeframe
        except EOFError:
            return "1d"
    
    def get_start_date(self) -> str:
        """
        Get start date from user input.
        
        Returns:
            Start date in YYYY-MM-DD format
        """
        try:
            start_date = input("Enter start date [YYYY-MM-DD]: ")
            return start_date
        except EOFError:
            # Default to 60 days ago
            return (datetime.now() - pd.Timedelta(days=60)).strftime('%Y-%m-%d')
    
    def get_end_date(self) -> str:
        """
        Get end date from user input.
        
        Returns:
            End date in YYYY-MM-DD format
        """
        try:
            end_date = input("Enter end date [YYYY-MM-DD]: ")
            return end_date
        except EOFError:
            return datetime.now().strftime('%Y-%m-%d')
    
    def configure_broker(self) -> Optional[IBConnection]:
        """
        Configure Interactive Brokers connection for paper trading.
        
        Returns:
            Configured IBConnection instance or None if connection fails
        """
        logger.info("Configuring paper trading connection to Interactive Brokers")
        
        try:
            # Get IB connection parameters with enforced paper trading
            host = input("Enter TWS/Gateway host (127.0.0.1): ") or "127.0.0.1"
            port = PAPER_TRADING_PORT  # Force paper trading port
            client_id = int(input("Enter client ID (1): ") or "1")
            account = input("Enter IB paper trading account ID: ")
            
            if not account:
                logger.warning("Paper trading account ID is required")
                return None
                
            # Verify paper trading account format
            if not account.endswith('DU'):  # Paper trading accounts typically end with 'DU'
                logger.warning("This doesn't appear to be a paper trading account. Please use a paper trading account.")
                return None
                
            # Create and test connection
            broker = IBConnection(
                host=host,
                port=port,
                client_id=client_id,
                account=account
            )
            
            # Add safety check for paper trading
            if broker.is_connected():
                # Verify account type through account info
                account_values = broker.get_account_values()
                if account_values:
                    account_type = account_values.get('AccountType', ('', ''))[0]
                    if 'PAPER' not in str(account_type).upper():
                        logger.error("Connected account is not a paper trading account")
                        broker.disconnect()
                        return None
                    
                logger.info("Successfully connected to IB Paper Trading")
                return broker
            else:
                logger.error("Failed to connect to IB Paper Trading")
                return None
                
        except Exception as e:
            logger.error(f"Error configuring paper trading: {str(e)}")
            return None

    def execute_trade(self, symbol: str, action: str, quantity: float):
        """
        Execute a paper trade using the configured broker.
        
        Args:
            symbol: Trading symbol
            action: Trade action ('BUY' or 'SELL')
            quantity: Trade quantity
        """
        if not self.broker or not self.is_paper_trading:
            logger.warning("Paper trading broker not configured, skipping trade execution")
            return
            
        try:
            # Add safety check before order execution
            if not symbol or not action or quantity <= 0:
                logger.warning("Invalid order parameters")
                return
                
            order_id = self.broker.send_order(
                symbol=symbol,
                action=action,
                quantity=quantity
            )
            logger.info(f"Paper trade order {order_id} sent: {action} {quantity} {symbol}")
            
        except Exception as e:
            logger.error(f"Error executing paper trade: {str(e)}")

    def run(self):
        """Run the trading system."""
        self.print_welcome()
        
        # Display paper trading notice
        self.console.print("\n[yellow]PAPER TRADING MODE ENABLED[/yellow]")
        self.console.print("All trades will be simulated without real money\n")
        
        # Configure paper trading broker
        self.broker = self.configure_broker()
        
        # Select and configure strategy
        strategy_name = self.select_strategy()
        self.strategy = self.configure_strategy(strategy_name)
        
        # Get trading parameters
        symbols = self.get_symbols()
        timeframe = self.get_timeframe()
        start_date = self.get_start_date()
        end_date = self.get_end_date()
        
        # Load and preprocess data
        data = self.data_manager.load_data(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe
        )
        
        data = self.data_manager.preprocess_data(data)
        
        # Run strategy
        for timestamp in data.index.get_level_values('date').unique():
            current_data = data.xs(timestamp, level='date', drop_level=False)
            signals = self.strategy.update(timestamp, current_data)
            
            # Execute paper trades
            if self.broker and signals and self.is_paper_trading:
                for signal in signals:
                    self.execute_trade(
                        symbol=signal['symbol'],
                        action=signal['action'],
                        quantity=signal['quantity']
                    )
            
            # Get and display portfolio state
            portfolio_state = self.strategy.get_portfolio_state()
            self.console.print(f"\nPortfolio state at {timestamp}:")
            self.console.print(portfolio_state)
            
            if self.broker and self.is_paper_trading:
                # Get and display paper trading account information
                positions = self.broker.get_positions()
                account_values = self.broker.get_account_values()
                pnl = self.broker.get_pnl()
                
                self.console.print("\nPaper Trading Account Information:")
                self.console.print(f"Paper Positions: {positions}")
                self.console.print(f"Paper Account Values: {account_values}")
                self.console.print(f"Paper PnL: {pnl}")
