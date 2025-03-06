#!/usr/bin/env python
import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import logging

import pandas as pd
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.table import Table

from quantframe.strategy.config_manager import StrategyConfigManager
from quantframe.backtesting.engine import BacktestEngine
from quantframe.data.base import DataManager

console = Console()
logger = logging.getLogger(__name__)

class QuantFrameController:
    """Main controller for the QuantFrame trading system."""
    
    def __init__(self):
        # Define directory paths using Path from pathlib
        self.config_dir = Path("config/strategies")  # Directory for strategy configurations
        self.data_dir = Path("data")                # Directory for market data
        self.results_dir = Path("results")          # Directory for backtest results
        
        # Create directories if they don't exist
        self.config_dir.mkdir(parents=True, exist_ok=True)   # Creates config/strategies
        self.data_dir.mkdir(parents=True, exist_ok=True)     # Creates data/
        self.results_dir.mkdir(parents=True, exist_ok=True)  # Creates results/
        
        # Initialize core components
        self.config_manager = StrategyConfigManager(str(self.config_dir))  # Manages strategy configurations
        self.data_manager = DataManager()                                  # Handles market data
        self.backtest_engine = BacktestEngine()                           # Runs backtests
        
        # Load available trading strategies
        self.strategies = self.get_available_strategies()  # Gets dict of strategy names and descriptions

    def get_available_strategies(self):
        """Get list of available trading strategies."""
        return {
            'grid': 'Grid Trading Strategy implementation.',
            'momentum': 'Momentum Trading Strategy using multiple technical indicators.',
            'mean_reversion': 'Mean Reversion Strategy using moving averages and Bollinger Bands.'
        }

    def display_welcome(self):
        """Display welcome message and system status."""
        console.print("\n[bold blue]Welcome to QuantFrame Trading System[/bold blue]")
        console.print("=" * 50)
        console.print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        console.print(f"Available strategies: {', '.join(self.strategies.keys())}")
        console.print("=" * 50 + "\n")

    def select_strategy(self) -> str:
        """Let user select a trading strategy."""
        logger.info("Asking user to select a strategy")
        strategy_table = Table(title="Available Strategies")
        strategy_table.add_column("ID", style="cyan")
        strategy_table.add_column("Name", style="magenta")
        strategy_table.add_column("Description", style="green")
        
        for key in self.strategies.keys():
            strategy_table.add_row(
                key,
                key.capitalize(),
                self.strategies[key]
            )
        
        console.print(strategy_table)
        strategy_name = Prompt.ask(
            "Please select a strategy by entering its ID",
            choices=list(self.strategies.keys())
        )
        logger.info(f"User selected strategy: {strategy_name}")
        return strategy_name

    def configure_strategy(self, strategy_name: str) -> Dict:
        """Configure strategy parameters."""
        console.print(f"\n[yellow]Configuring {strategy_name} strategy[/yellow]")
        
        # Load default config
        config = self.config_manager.get_default_config(strategy_name)
        descriptions = self.config_manager.get_config_descriptions(strategy_name)
        
        # Allow user to modify parameters
        logger.info(f"Asking user if they want to modify parameters for {strategy_name}")
        if Confirm.ask("Would you like to modify the default parameters?"):
            logger.info(f"User chose to modify parameters for {strategy_name}")
            console.print("\n[cyan]Current configuration:[/cyan]")
            config_table = Table(title="Strategy Configuration")
            config_table.add_column("Parameter", style="magenta")
            config_table.add_column("Value", style="cyan")
            config_table.add_column("Description", style="green")
            
            for key, value in config.items():
                config_table.add_row(
                    key,
                    str(value),
                    descriptions.get(key, "No description available")
                )
            
            console.print(config_table)
            
            for key, value in config.items():
                logger.info(f"Asking user for value for {key}")
                new_value = Prompt.ask(
                    f"Enter value for {key} ({descriptions.get(key, 'No description')})",
                    default=str(value)
                )
                logger.info(f"User entered value for {key}: {new_value}")
                # Convert to appropriate type
                if isinstance(value, bool):
                    config[key] = new_value.lower() in ('true', 'yes', '1', 'y')
                elif isinstance(value, int):
                    config[key] = int(new_value)
                elif isinstance(value, float):
                    config[key] = float(new_value)
                else:
                    config[key] = new_value
        else:
            logger.info(f"User chose not to modify parameters for {strategy_name}")
        
        return config

    def select_assets(self) -> List[str]:
        """Select assets to trade."""
        logger.info("Asking user to select assets")
        console.print("\n[yellow]Asset Selection[/yellow]")
        
        # Ask if user wants to use default asset (AAPL)
        use_default = Confirm.ask(
            "Would you like to use the default asset (AAPL)?",
            default=True
        )
        
        if use_default:
            logger.info("User selected default asset: AAPL")
            return ["AAPL"]
        
        assets = Prompt.ask(
            "Enter asset symbols (comma-separated)",
            default="BTCUSDT"
        )
        logger.info(f"User selected assets: {assets}")
        return [asset.strip() for asset in assets.split(",")]

    def configure_backtest(self) -> Dict:
        """Configure backtest parameters."""
        logger.info("Asking user to configure backtest")
        console.print("\n[yellow]Backtest Configuration[/yellow]")
        
        # Ask if user wants to use default settings
        use_defaults = Confirm.ask(
            "Would you like to use default backtest parameters? (AAPL, 1d timeframe, past 365 days)",
            default=True
        )
        
        if use_defaults:
            # Default backtest configuration
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            return {
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
                "initial_capital": 10000.0,
                "timeframe": "1d",
                "symbols": ["AAPL"]
            }
        
        # Custom configuration if user doesn't want defaults
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        config = {
            "start_date": Prompt.ask(
                "Start date (YYYY-MM-DD)",
                default=start_date.strftime("%Y-%m-%d")
            ),
            "end_date": Prompt.ask(
                "End date (YYYY-MM-DD)",
                default=end_date.strftime("%Y-%m-%d")
            ),
            "initial_capital": float(Prompt.ask(
                "Initial capital",
                default="10000"
            )),
            "timeframe": Prompt.ask(
                "Timeframe (1m, 5m, 15m, 1h, 4h, 1d)",
                default="1h",
                choices=["1m", "5m", "15m", "1h", "4h", "1d"]
            )
        }
        
        logger.info(f"User configured backtest: {config}")
        return config

    def load_strategy(self, strategy_name: str):
        """Load a trading strategy.
        
        Args:
            strategy_name: Name of the strategy to load
            
        Returns:
            Strategy instance
        """
        if strategy_name == 'grid':
            from quantframe.strategy.grid_trading import GridTradingStrategy
            return GridTradingStrategy(self.config_manager)
        elif strategy_name == 'momentum':
            from quantframe.strategy.momentum_strategy import MomentumStrategy
            return MomentumStrategy(self.config_manager)
        elif strategy_name == 'mean_reversion':
            from quantframe.strategy.mean_reversion import MeanReversionStrategy
            return MeanReversionStrategy(self.config_manager)
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")

    def display_results_summary(self, results: Dict):
        """Display a summary of backtest results."""
        console.print("\n[bold blue]Backtest Results Summary[/bold blue]")
        console.print("=" * 50)
        
        summary_table = Table(title="Performance Metrics")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="magenta")
        
        metrics = {
            "Total Return": f"{results.get('total_return', 0):.2f}%",
            "Sharpe Ratio": f"{results.get('sharpe_ratio', 0):.2f}",
            "Max Drawdown": f"{results.get('max_drawdown', 0):.2f}%",
            "Win Rate": f"{results.get('win_rate', 0):.2f}%",
            "Total Trades": str(results.get('total_trades', 0))
        }
        
        for metric, value in metrics.items():
            summary_table.add_row(metric, value)
        
        console.print(summary_table)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"backtest_results_{timestamp}.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=4, default=str)
        
        console.print(f"\n[green]Results saved to: {results_file}[/green]")

    def run(self):
        """Main execution flow."""
        try:
            self.display_welcome()
            
            # Strategy selection and configuration
            strategy_name = self.select_strategy()
            strategy_config = self.configure_strategy(strategy_name)
            
            # Load strategy
            strategy = self.load_strategy(strategy_name)
            
            # Backtest configuration (including asset selection)
            backtest_config = self.configure_backtest()
            
            # Asset selection only if not using default configuration
            if 'symbols' not in backtest_config:
                assets = self.select_assets()
                backtest_config['symbols'] = assets
            
            # Load data
            console.print("\n[green]Loading market data...[/green]")
            data = self.data_manager.load_data(
                symbols=backtest_config['symbols'],
                start_date=backtest_config["start_date"],
                end_date=backtest_config["end_date"],
                timeframe=backtest_config["timeframe"]
            )
            
            if data.empty:
                console.print("[red]No data available for the selected period[/red]")
                return
            
            # Debug data before backtest
            console.print("\n[yellow]Debug Information:[/yellow]")
            console.print(f"Data shape: {data.shape}")
            console.print(f"Data columns: {data.columns.tolist()}")
            console.print(f"Data types:\n{data.dtypes}")
            console.print(f"First few rows:\n{data.head()}")
            
            # Run backtest
            console.print("\n[green]Running backtest...[/green]")
            try:
                results = self.backtest_engine.run(
                    strategy=strategy,
                    data=data,
                    initial_capital=backtest_config["initial_capital"]
                )
                
                # Display results
                self.display_results_summary(results)
                
            except Exception as e:
                console.print(f"\n[red]Backtest Error Details:[/red]")
                console.print(f"Error type: {type(e)}")
                console.print(f"Error message: {str(e)}")
                import traceback
                console.print(f"[red]Traceback:[/red]\n{''.join(traceback.format_tb(e.__traceback__))}")
                
                # Additional debugging information
                if hasattr(strategy, '_debug_data'):
                    strategy._debug_data(data, "Error state")
                
        except KeyboardInterrupt:
            console.print("\n[red]Operation cancelled by user[/red]")
        except Exception as e:
            console.print(f"\n[red]System Error:[/red]")
            console.print(f"Error type: {type(e)}")
            console.print(f"Error message: {str(e)}")
            import traceback
            console.print(f"[red]Traceback:[/red]\n{''.join(traceback.format_tb(e.__traceback__))}")

def main():
    """Entry point for the QuantFrame system."""
    logging.basicConfig(level=logging.INFO)
    controller = QuantFrameController()
    controller.run()

if __name__ == "__main__":
    main()
