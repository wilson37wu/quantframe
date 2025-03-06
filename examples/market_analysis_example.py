"""Example script demonstrating market analysis functionality."""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from quantframe.data.sources.binance_source import BinanceSource
from quantframe.analysis.market_analysis import MarketAnalyzer

def plot_market_analysis(analyzer: MarketAnalyzer, output_dir: str = '.') -> None:
    """Run and plot market analysis for major cryptocurrencies.
    
    Args:
        analyzer: MarketAnalyzer instance
        output_dir: Directory to save plot files
    """
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.Timedelta(days=90)
    
    # Analyze major cryptocurrencies
    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "SOLUSDT"]
    
    # Get cross-market dynamics
    market_data = analyzer.analyze_cross_market_dynamics(
        symbols, start_date, end_date
    )
    
    if market_data and 'correlation' in market_data:
        # Plot correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            market_data['correlation'],
            annot=True,
            cmap='coolwarm',
            center=0,
            fmt='.2f'
        )
        plt.title("Cryptocurrency Market Correlations")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/market_correlations.png")
        plt.close()
        
        # Print market statistics
        print("\nMarket Statistics:")
        print("\nBeta to BTC:")
        for symbol, beta in market_data['betas'].items():
            print(f"{symbol}: {beta:.2f}")
    
    # Analyze intraday patterns for major pairs
    for symbol in ["BTCUSDT", "ETHUSDT"]:
        patterns = analyzer.analyze_intraday_patterns(symbol)
        
        if not patterns.empty:
            # Create intraday pattern plots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            patterns['volatility'].plot(ax=ax1, kind='bar')
            ax1.set_title(f"{symbol} Hourly Volatility Pattern")
            ax1.set_xlabel("Hour")
            ax1.set_ylabel("Volatility (Annualized)")
            
            patterns['volume'].plot(ax=ax2, kind='bar')
            ax2.set_title(f"{symbol} Hourly Volume Pattern")
            ax2.set_xlabel("Hour")
            ax2.set_ylabel("Average Volume")
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{symbol}_intraday_patterns.png")
            plt.close()
            
            print(f"\n{symbol} Hourly Patterns:")
            print("\nTop 3 Most Volatile Hours:")
            print(patterns['volatility'].nlargest(3))
            print("\nTop 3 Highest Volume Hours:")
            print(patterns['volume'].nlargest(3))

def main():
    """Run market analysis example."""
    # Initialize components
    config = {"cache_dir": "data/cache/crypto"}
    binance = BinanceSource(config)
    analyzer = MarketAnalyzer(binance)
    
    # Run analysis
    plot_market_analysis(analyzer)

if __name__ == "__main__":
    main()
