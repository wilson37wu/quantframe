"""Example script demonstrating cryptocurrency data fetching and analysis."""
import yaml
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from quantframe.data.sources.binance_source import BinanceSource

def load_config():
    """Load configuration from YAML file."""
    config_path = Path("config/data_sources.yaml")
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Warning: Config file not found at {config_path}. Using default configuration.")
        return {"binance": {"cache_dir": "data/cache/crypto"}}

def test_market_data():
    """Test basic market data functionality."""
    # Load configuration
    config = load_config()
    
    # Initialize Binance data source
    binance = BinanceSource(config["binance"])
    
    # Define time range
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.Timedelta(days=365)
    
    # Test available pairs
    try:
        print("\nTesting available pairs...")
        pairs = binance.get_available_pairs()
        print(f"Found {len(pairs)} trading pairs")
        print("Sample pairs:", pairs[:5])
    except Exception as e:
        print(f"Warning: Could not fetch available pairs: {e}")
    
    # Fetch data for multiple cryptocurrencies
    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
    crypto_data = {}
    
    for symbol in symbols:
        try:
            print(f"\nFetching data for {symbol}...")
            data = binance.get_data(symbol, start_date, end_date, interval="1d")
            crypto_data[symbol] = data
            
            # Print basic statistics
            print(f"\nStatistics for {symbol}:")
            print(f"Data points: {len(data)}")
            print(f"Date range: {data.index[0]} to {data.index[-1]}")
            print(f"Current Price: {data['close'].iloc[-1]:.2f} USDT")
            print(f"Daily Returns Mean: {data['log_returns'].mean()*100:.2f}%")
            print(f"Daily Returns Std: {data['log_returns'].std()*100:.2f}%")
            print(f"Current Volatility: {data['volatility'].iloc[-1]*100:.2f}%")
            
            # Test market cap (requires API key)
            try:
                market_cap = binance.get_market_cap(symbol)
                print(f"Market Cap: {market_cap:,.2f} USDT")
            except Exception as e:
                print(f"Note: Market cap not available (requires API key)")
                
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
    
    if crypto_data:
        # Plot price and volume data
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        for symbol, data in crypto_data.items():
            # Plot prices
            ax1.plot(data.index, data["close"], label=symbol)
            
            # Plot volumes
            ax2.bar(data.index, data["volume"], alpha=0.3, label=symbol)
        
        ax1.set_title("Cryptocurrency Prices")
        ax1.set_ylabel("Price (USDT)")
        ax1.legend()
        ax1.grid(True)
        
        ax2.set_title("Trading Volume")
        ax2.set_ylabel("Volume")
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig("crypto_analysis.png")
        print("\nSaved price and volume chart to crypto_analysis.png")
        plt.close()

def test_different_timeframes():
    """Test data fetching with different timeframes."""
    config = load_config()
    binance = BinanceSource(config["binance"])
    
    # Test different intervals
    intervals = ["1h", "4h", "1d"]
    symbol = "BTCUSDT"
    end_date = pd.Timestamp.now()
    
    print("\nTesting different timeframes for BTC:")
    for interval in intervals:
        try:
            if interval == "1h":
                start_date = end_date - pd.Timedelta(days=7)
            elif interval == "4h":
                start_date = end_date - pd.Timedelta(days=30)
            else:
                start_date = end_date - pd.Timedelta(days=365)
                
            print(f"\nFetching {interval} data...")
            data = binance.get_data(symbol, start_date, end_date, interval=interval)
            print(f"Retrieved {len(data)} data points")
            print(f"First timestamp: {data.index[0]}")
            print(f"Last timestamp: {data.index[-1]}")
            
        except Exception as e:
            print(f"Error fetching {interval} data: {e}")

if __name__ == "__main__":
    print("Testing Binance Data Source Implementation")
    print("=" * 40)
    
    test_market_data()
    test_different_timeframes()
