"""Example of optimizing mean reversion strategy parameters with 3D visualization"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import pickle
import os

from quantframe.data.sources.yfinance_source import YFinanceSource
from quantframe.strategy.mean_reversion import MeanReversionStrategy
from quantframe.backtesting.engine import Backtester
from quantframe.utils.market_data import get_sp500_symbols, filter_liquid_stocks

def optimize_single_stock(args):
    """Optimize strategy for a single stock."""
    symbol, data, param_ranges = args
    results = []
    
    try:
        if len(data) == 0:
            print(f"Empty data for {symbol}, skipping...")
            return pd.DataFrame()
            
        for lookback in param_ranges['lookback_period']:
            for entry_z in param_ranges['entry_zscore']:
                for rsi_thresh in param_ranges['rsi_entry_threshold']:
                    # Create strategy config with current parameters
                    config = {
                        'lookback_period': lookback,
                        'entry_zscore': entry_z,
                        'exit_zscore': entry_z * 0.3,
                        'rsi_period': 14,
                        'rsi_entry_threshold': rsi_thresh,
                        'rsi_exit_threshold': 100 - rsi_thresh,
                        'volatility_lookback': 21,
                        'position_size_atr_multiple': 0.3,
                        'max_position_size': 0.05,
                        'stop_loss_atr_multiple': 1.5,
                        'take_profit_atr_multiple': 3.0
                    }
                    
                    strategy = MeanReversionStrategy(config)
                    backtester = Backtester(
                        strategy=strategy,
                        initial_capital=1_000_000,
                        commission=0.001,
                        slippage=0.001
                    )
                    
                    result = backtester.run(data)
                    
                    results.append({
                        'symbol': symbol,
                        'lookback': lookback,
                        'entry_zscore': entry_z,
                        'rsi_threshold': rsi_thresh,
                        'total_return': result.metrics['total_return'],
                        'sharpe_ratio': result.metrics['sharpe_ratio'],
                        'profit_factor': result.metrics['profit_factor'],
                        'win_rate': result.metrics['win_rate'],
                        'num_trades': len(result.trades)
                    })
        
        return pd.DataFrame(results)
    except Exception as e:
        print(f"Error optimizing {symbol}: {e}")
        return pd.DataFrame()

def plot_optimization_3d(results_df, metric='total_return', output_dir='optimization_results'):
    """Create 3D scatter plot of optimization results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Group results by symbol
    for symbol in results_df['symbol'].unique():
        symbol_results = results_df[results_df['symbol'] == symbol]
        
        if symbol_results.empty:
            continue
            
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(
            symbol_results['lookback'],
            symbol_results['entry_zscore'],
            symbol_results['rsi_threshold'],
            c=symbol_results[metric],
            cmap='viridis',
            marker='o'
        )
        
        ax.set_xlabel('Lookback Period')
        ax.set_ylabel('Entry Z-Score')
        ax.set_zlabel('RSI Threshold')
        ax.set_title(f'{symbol} Optimization Results ({metric})')
        
        plt.colorbar(scatter, label=metric)
        plt.savefig(f'{output_dir}/{symbol}_{metric}_3d.png')
        plt.close()

def analyze_results(results_df, output_dir='optimization_results'):
    """Analyze and save optimization results."""
    os.makedirs(output_dir, exist_ok=True)
    
    if results_df.empty:
        print("No results to analyze")
        return None, None
    
    # Save full results to CSV
    results_df.to_csv(f'{output_dir}/all_results.csv', index=False)
    
    # Calculate best parameters for each metric and symbol
    metrics = ['total_return', 'sharpe_ratio', 'profit_factor', 'win_rate']
    best_params = pd.DataFrame()
    
    for symbol in results_df['symbol'].unique():
        symbol_results = results_df[results_df['symbol'] == symbol]
        
        for metric in metrics:
            best_idx = symbol_results[metric].idxmax()
            if pd.isna(best_idx):
                continue
                
            best_row = symbol_results.loc[best_idx].copy()
            best_row['optimized_metric'] = metric
            best_params = pd.concat([best_params, pd.DataFrame([best_row])], ignore_index=True)
    
    # Save best parameters
    if not best_params.empty:
        best_params.to_csv(f'{output_dir}/best_parameters.csv', index=False)
    
    # Create summary statistics
    summary_stats = results_df.groupby('symbol').agg({
        'total_return': ['mean', 'std', 'max'],
        'sharpe_ratio': ['mean', 'std', 'max'],
        'profit_factor': ['mean', 'std', 'max'],
        'win_rate': ['mean', 'std', 'max'],
        'num_trades': ['mean', 'sum']
    }).round(4)
    
    if not summary_stats.empty:
        summary_stats.to_csv(f'{output_dir}/summary_statistics.csv')
    
    return best_params, summary_stats

def main():
    # Create output directory
    output_dir = 'optimization_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Parameter ranges to test
    param_ranges = {
        'lookback_period': range(10, 31, 5),
        'entry_zscore': np.arange(1.0, 2.6, 0.5),
        'rsi_entry_threshold': range(20, 41, 5)
    }
    
    # Get S&P 500 symbols and filter for liquid stocks
    print("Fetching S&P 500 symbols...")
    sp500_symbols = get_sp500_symbols()
    print("Filtering for liquid stocks...")
    symbols = filter_liquid_stocks(sp500_symbols)
    print(f"Testing {len(symbols)} liquid stocks")
    
    # Initialize data source
    data_source = YFinanceSource({'cache_dir': 'data/cache'})
    
    # Download data
    start_date = pd.Timestamp('2023-01-01')  # Removed timezone for consistency
    end_date = pd.Timestamp('2024-01-01')  # Removed timezone for consistency
    
    # Prepare optimization tasks
    tasks = []
    for symbol in symbols:
        try:
            data = data_source.get_data(symbol, start_date, end_date)
            if not data.empty:
                tasks.append((symbol, data, param_ranges))
        except Exception as e:
            print(f"Error getting data for {symbol}: {e}")
            continue
    
    if not tasks:
        print("No valid tasks to process")
        return
    
    # Run optimization in parallel
    print("Running optimization...")
    results_dfs = []
    
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(optimize_single_stock, task) for task in tasks]
        
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                results_df = future.result()
                if not results_df.empty:
                    results_dfs.append(results_df)
            except Exception as e:
                print(f"Error processing optimization result: {e}")
                continue
    
    if not results_dfs:
        print("No optimization results to process")
        return
    
    # Combine all results
    all_results = pd.concat(results_dfs, ignore_index=True)
    
    if all_results.empty:
        print("No valid optimization results")
        return
    
    # Create visualizations and analyze results
    print("Creating visualizations...")
    metrics = ['total_return', 'sharpe_ratio', 'profit_factor', 'win_rate']
    for metric in metrics:
        plot_optimization_3d(all_results, metric, output_dir)
    
    # Analyze and save results
    print("Analyzing results...")
    best_params, summary_stats = analyze_results(all_results, output_dir)
    
    # Print summary
    if best_params is not None:
        print("\nBest Parameters Summary:")
        print(best_params.groupby('optimized_metric')[['symbol', 'lookback', 'entry_zscore', 'rsi_threshold']].head())
    
    if summary_stats is not None:
        print("\nPerformance Summary:")
        print(summary_stats.head())

if __name__ == '__main__':
    main()
