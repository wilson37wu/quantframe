import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantframe.strategy.arima_regression_strategy import ARIMARegression
import argparse

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run ARIMA-Regression analysis on a trading pair')
    parser.add_argument('--symbol', type=str, default='BTC/USDT',
                      help='Trading pair symbol (e.g., BTC/USDT, ETH/USDT)')
    parser.add_argument('--timeframe', type=str, default='1d',
                      help='Data timeframe (1d, 4h, etc.)')
    parser.add_argument('--lookback', type=int, default=365,
                      help='Number of days of historical data')
    parser.add_argument('--sma-window', type=int, default=20,
                      help='Window size for moving averages')
    parser.add_argument('--reg-window', type=int, default=30,
                      help='Window size for rolling regression')
    parser.add_argument('--arima-p', type=int, default=1,
                      help='ARIMA p parameter (AR order)')
    parser.add_argument('--arima-d', type=int, default=1,
                      help='ARIMA d parameter (differencing)')
    parser.add_argument('--arima-q', type=int, default=1,
                      help='ARIMA q parameter (MA order)')
    
    args = parser.parse_args()
    
    # Initialize and run strategy
    strategy = ARIMARegression(
        symbol=args.symbol,
        timeframe=args.timeframe,
        lookback_days=args.lookback,
        sma_window=args.sma_window,
        regression_window=args.reg_window,
        arima_order=(args.arima_p, args.arima_d, args.arima_q)
    )
    
    print(f"\nRunning ARIMA-Regression analysis for {args.symbol}")
    print("Parameters:")
    print(f"- Timeframe: {args.timeframe}")
    print(f"- Lookback days: {args.lookback}")
    print(f"- SMA window: {args.sma_window}")
    print(f"- Regression window: {args.reg_window}")
    print(f"- ARIMA order: ({args.arima_p}, {args.arima_d}, {args.arima_q})")
    print("\nStarting analysis...")
    
    strategy.run_analysis()

if __name__ == "__main__":
    main()
