"""Example of using the quantframe framework for mean reversion strategy"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from quantframe.data.sources.yfinance_source import YFinanceSource
from quantframe.strategy.mean_reversion import MeanReversionStrategy
from quantframe.backtesting.engine import Backtester
from quantframe.utils.indicators import calculate_zscore, calculate_rsi

def main():
    # Configuration
    config = {
        'data_source': {
            'cache_dir': 'data/cache'
        },
        'strategy': {
            'lookback_period': 20,
            'entry_zscore': 1.75,  # Slightly lower for more trades
            'exit_zscore': 0.5,   # Faster exits
            'rsi_period': 14,
            'rsi_entry_threshold': 30,  # Less extreme for more trades
            'rsi_exit_threshold': 70,   # Less extreme for more trades
            'volatility_lookback': 21,
            'position_size_atr_multiple': 0.3,  # Slightly larger positions
            'max_position_size': 0.05,   # Keep conservative max position
            'stop_loss_atr_multiple': 1.5,  # Keep tight stops
            'take_profit_atr_multiple': 3.0   # More realistic take profit
        },
        'backtest': {
            'initial_capital': 1_000_000,
            'commission': 0.001,
            'slippage': 0.001
        }
    }
    
    # Initialize components
    data_source = YFinanceSource(config['data_source'])
    strategy = MeanReversionStrategy(config['strategy'])
    backtester = Backtester(
        strategy=strategy,
        initial_capital=config['backtest']['initial_capital'],
        commission=config['backtest']['commission'],
        slippage=config['backtest']['slippage']
    )
    
    # Download data
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    start_date = pd.Timestamp('2023-01-01', tz='UTC')  # Use a fixed date for testing
    end_date = pd.Timestamp('2024-01-01', tz='UTC')
    
    results = {}
    for symbol in symbols:
        print(f"\nProcessing {symbol}...")
        
        # Get data
        data = data_source.get_data(symbol, start_date, end_date)
        
        # Convert MultiIndex to single index if present
        if isinstance(data.columns, pd.MultiIndex):
            data = data.copy()
            data.columns = data.columns.get_level_values(0)
        
        # Print data info
        print(f"Data shape: {data.shape}")
        print("\nFirst few rows:")
        print(data.head())
        print("\nLast few rows:")
        print(data.tail())
        print("\nColumns:", data.columns.tolist())
        print("\nIndex:", data.index.name)
        
        # Calculate indicators for debugging
        zscore = calculate_zscore(data['close'], strategy.lookback_period)
        rsi = calculate_rsi(data['close'], strategy.rsi_period)
        
        print(f"\nZ-score (last 5 values):")
        print(zscore.tail())
        print(f"\nRSI (last 5 values):")
        print(rsi.tail())
        
        # Run backtest
        result = backtester.run(data)
        results[symbol] = result
        
        # Print performance metrics
        print(f"\nPerformance Metrics for {symbol}:")
        for key, value in result.metrics.items():
            print(f"{key}: {value:.4f}")
        
        print(f"\nTotal trades: {len(result.trades)}")
        if result.trades:
            print("\nLast 5 trades:")
            for trade in result.trades[-5:]:
                print(f"Entry: {trade.entry_time}, Exit: {trade.exit_time}, "
                      f"Direction: {trade.direction}, PnL: ${trade.pnl:.2f}, "
                      f"Return: {trade.return_pct:.2%}")
    
    # Calculate overall performance
    total_pnl = sum(sum(trade.pnl for trade in result.trades) 
                   for result in results.values())
    total_trades = sum(len(result.trades) for result in results.values())
    
    print(f"\nOverall Performance:")
    print(f"Total PnL: ${total_pnl:,.2f}")
    print(f"Total Trades: {total_trades}")
    
    # Plot equity curves
    plt.figure(figsize=(12, 6))
    for symbol, result in results.items():
        plt.plot(result.equity_curve.index, result.equity_curve.values, label=symbol)
    
    plt.title('Portfolio Equity Curves')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True)
    plt.savefig('equity_curves.png')  # Save to file instead of showing
    plt.close()  # Close the figure to free memory

if __name__ == '__main__':
    main()
