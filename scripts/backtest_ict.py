"""Backtest ICT strategy on BTCUSDT using test data."""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from quantframe.data.test_data import TestDataGenerator
from quantframe.strategy.ict_strategy import ICTStrategy
from quantframe.strategy.config.ict_config import ICTConfig
from quantframe.backtesting.engine import Backtester

def load_config(config_path: str) -> dict:
    """Load configuration from JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        return json.load(f)

def run_ict_backtest():
    """Run ICT strategy backtest on BTCUSDT test data."""
    try:
        # Load configuration
        config = load_config('config/ict_backtest_config.json')
        
        # Setup test data generator
        data_generator = TestDataGenerator(seed=42)  # Fixed seed for reproducibility
        
        # Calculate date range
        end_date = pd.Timestamp.now()
        start_date = end_date - pd.Timedelta(days=365)
        
        # Generate test data
        print(f"Generating {config['data']['symbol']} test data from {start_date} to {end_date}")
        data = data_generator.generate_btc_data(
            start_date=start_date,
            end_date=end_date,
            interval=config['data']['interval'],
            base_price=40000,  # Starting price around $40k
            volatility=0.02    # 2% daily volatility
        )
        
        if data.empty:
            raise ValueError(f"Failed to generate test data")
            
        print(f"Generated {len(data)} {config['data']['interval']} candles")
        
        # Configure ICT strategy
        strategy_config = ICTConfig(**config['strategy'])
        
        # Validate configuration
        strategy_config.validate()
        print("\nStrategy Configuration:")
        print(strategy_config.to_dataframe().to_string())
        
        # Initialize strategy
        strategy = ICTStrategy(strategy_config)
        
        # Setup backtester
        backtester = Backtester(
            strategy=strategy,
            initial_capital=config['backtest']['initial_capital'],
            commission=config['backtest']['commission'],
            slippage=config['backtest']['slippage']
        )
        
        # Create results directory if it doesn't exist
        results_dir = Path('results')
        results_dir.mkdir(exist_ok=True)
        
        # Run backtest
        print("\nRunning backtest...")
        results = backtester.run(data)
        
        # Print results
        print("\nBacktest Results:")
        print(f"Total Return: {results.metrics['total_return']:.2%}")
        print(f"Sharpe Ratio: {results.metrics['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {results.metrics['max_drawdown']:.2%}")
        print(f"Win Rate: {results.metrics['win_rate']:.2%}")
        print(f"Profit Factor: {results.metrics['profit_factor']:.2f}")
        print(f"Number of Trades: {len(results.trades)}")
        
        # Calculate monthly returns
        monthly_returns = results.equity_curve.resample('M').last().pct_change()
        print("\nMonthly Returns:")
        print(monthly_returns.tail().to_string())
        
        # Save detailed results
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        
        # Save trades
        trades_df = pd.DataFrame([
            {
                'symbol': t.symbol,
                'entry_time': t.entry_time,
                'exit_time': t.exit_time,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'direction': t.direction,
                'size': t.size,
                'pnl': t.pnl,
                'return_pct': t.return_pct,
                'reason': t.reason
            }
            for t in results.trades
        ])
        
        trades_file = results_dir / f"ict_backtest_trades_{timestamp}.csv"
        trades_df.to_csv(trades_file, index=False)
        print(f"\nDetailed trades saved to {trades_file}")
        
        # Save equity curve
        equity_file = results_dir / f"ict_equity_curve_{timestamp}.csv"
        results.equity_curve.to_csv(equity_file)
        print(f"Equity curve saved to {equity_file}")
        
        # Save test data for reference
        data_file = results_dir / f"test_data_{timestamp}.csv"
        data.to_csv(data_file)
        print(f"Test data saved to {data_file}")
        
        # Plot results
        print("\nGenerating performance plots...")
        fig = results.plot_results()
        plot_file = results_dir / f"ict_performance_{timestamp}.png"
        fig.savefig(plot_file)
        print(f"Performance plots saved to {plot_file}")
            
        return results
        
    except FileNotFoundError as e:
        print(f"Error: Configuration file not found - {e}")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in configuration file - {e}")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error during backtest: {e}")
        raise

if __name__ == '__main__':
    run_ict_backtest()
