"""Detailed cryptocurrency market analysis focusing on volatility and correlations."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from quantframe.data.sources.binance_source import BinanceSource

class DetailedMarketAnalyzer:
    """Advanced market analysis with focus on volatility and correlations."""
    
    def __init__(self, binance: BinanceSource):
        self.binance = binance
        
    def analyze_volatility_clustering(self, returns: pd.Series) -> Tuple[float, pd.Series]:
        """Analyze volatility clustering using autocorrelation of squared returns."""
        squared_returns = returns ** 2
        # Calculate autocorrelation for different lags
        autocorr = pd.Series({
            lag: squared_returns.autocorr(lag=lag)
            for lag in range(1, 11)
        })
        
        # Calculate persistence (sum of autocorrelations)
        persistence = autocorr.sum()
        
        return persistence, autocorr
    
    def detect_regime_changes(self, returns: pd.Series, window: int = 21) -> pd.DataFrame:
        """Detect volatility regime changes using rolling statistics."""
        vol = returns.rolling(window=window).std() * np.sqrt(252)
        
        # Calculate z-score of volatility
        vol_zscore = (vol - vol.rolling(window=63).mean()) / vol.rolling(window=63).std()
        
        # Identify regime changes
        regimes = pd.DataFrame({
            'volatility': vol,
            'zscore': vol_zscore,
            'regime': pd.cut(vol_zscore, 
                           bins=[-np.inf, -1.5, 1.5, np.inf],
                           labels=['Low', 'Normal', 'High'])
        })
        
        # Calculate regime duration
        regimes['regime_change'] = regimes['regime'] != regimes['regime'].shift(1)
        regimes['regime_duration'] = regimes.groupby(
            (regimes['regime_change']).cumsum()
        )['regime'].transform('count')
        
        return regimes
    
    def analyze_cross_market_dynamics(self, symbols: List[str], 
                                    start_date: pd.Timestamp,
                                    end_date: pd.Timestamp,
                                    interval: str = '1d') -> Dict:
        """Analyze cross-market dynamics including lead-lag relationships."""
        data = {}
        returns_data = {}
        
        # Fetch data for all symbols
        for symbol in symbols:
            try:
                df = self.binance.get_data(symbol, start_date, end_date, interval=interval)
                data[symbol] = df
                returns_data[symbol] = df['log_returns'].fillna(0)  # Fill NaN values
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
                
        if not returns_data:
            return {}
            
        # Create returns DataFrame
        returns_df = pd.DataFrame(returns_data)
        
        # Calculate correlation matrix
        corr_matrix = returns_df.corr()
        
        # Calculate beta to BTC (if available)
        betas = {}
        if 'BTCUSDT' in symbols:
            btc_returns = returns_df['BTCUSDT']
            for symbol in symbols:
                if symbol != 'BTCUSDT':
                    # Calculate beta using covariance method
                    cov = returns_df[symbol].cov(btc_returns)
                    var = btc_returns.var()
                    beta = cov / var if var != 0 else np.nan
                    betas[symbol] = beta
        
        return {
            'returns': returns_df,
            'correlation': corr_matrix,
            'betas': betas
        }
    
    def analyze_intraday_patterns(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Analyze intraday volatility and volume patterns."""
        end_date = pd.Timestamp.now()
        start_date = end_date - pd.Timedelta(days=days)
        
        try:
            # Fetch hourly data
            data = self.binance.get_data(symbol, start_date, end_date, interval='1h')
            
            # Extract hour from timestamp
            data['hour'] = data.index.hour
            
            # Calculate hourly patterns
            hourly_patterns = pd.DataFrame({
                'volatility': data.groupby('hour')['log_returns'].std() * np.sqrt(24),
                'volume': data.groupby('hour')['volume'].mean(),
                'trades': data.groupby('hour')['trades'].mean() if 'trades' in data.columns else None
            }).fillna(0)  # Fill any NaN values
            
            return hourly_patterns
            
        except Exception as e:
            print(f"Error analyzing intraday patterns for {symbol}: {e}")
            return pd.DataFrame()

def main():
    """Run detailed market analysis."""
    config = {"cache_dir": "data/cache/crypto"}
    binance = BinanceSource(config)
    analyzer = DetailedMarketAnalyzer(binance)
    
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.Timedelta(days=90)
    
    # 1. Volatility Clustering Analysis
    print("\n=== Volatility Clustering Analysis ===")
    major_pairs = ["BTCUSDT", "ETHUSDT"]
    
    for symbol in major_pairs:
        try:
            data = binance.get_data(symbol, start_date, end_date)
            persistence, autocorr = analyzer.analyze_volatility_clustering(data['log_returns'])
            
            print(f"\n{symbol} Analysis:")
            print(f"Volatility Persistence: {persistence:.4f}")
            print("\nAutocorrelation of Squared Returns:")
            print(autocorr)
            
            # Plot autocorrelation
            plt.figure(figsize=(10, 6))
            autocorr.plot(kind='bar')
            plt.title(f"{symbol} Volatility Clustering - Return Autocorrelation")
            plt.xlabel("Lag")
            plt.ylabel("Autocorrelation")
            plt.tight_layout()
            plt.savefig(f"{symbol}_volatility_clustering.png")
            plt.close()
            
        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")
    
    # 2. Regime Change Analysis
    print("\n=== Regime Change Analysis ===")
    for symbol in major_pairs:
        try:
            data = binance.get_data(symbol, start_date, end_date)
            regimes = analyzer.detect_regime_changes(data['log_returns'])
            
            print(f"\n{symbol} Regime Statistics:")
            print("\nRegime Distribution:")
            print(regimes['regime'].value_counts())
            print("\nAverage Duration (days):")
            duration_stats = regimes.groupby('regime', observed=True)['regime_duration'].mean()
            print(duration_stats)
            
            # Plot regime changes
            plt.figure(figsize=(12, 6))
            plt.plot(regimes.index, regimes['volatility'], label='Volatility')
            plt.scatter(regimes[regimes['regime'] == 'High'].index,
                       regimes[regimes['regime'] == 'High']['volatility'],
                       color='red', label='High Regime')
            plt.scatter(regimes[regimes['regime'] == 'Low'].index,
                       regimes[regimes['regime'] == 'Low']['volatility'],
                       color='green', label='Low Regime')
            plt.title(f"{symbol} Volatility Regimes")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{symbol}_regimes.png")
            plt.close()
            
        except Exception as e:
            print(f"Error analyzing regimes for {symbol}: {e}")
    
    # 3. Cross-Market Analysis
    print("\n=== Cross-Market Analysis ===")
    market_pairs = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "SOLUSDT"]
    
    results = analyzer.analyze_cross_market_dynamics(
        market_pairs, start_date, end_date
    )
    
    if results and 'correlation' in results:
        # Plot correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(results['correlation'], annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title("Cross-Market Correlations")
        plt.tight_layout()
        plt.savefig("cross_market_correlations.png")
        plt.close()
        
        if 'betas' in results:
            print("\nBeta to BTC:")
            for symbol, beta in results['betas'].items():
                print(f"{symbol}: {beta:.2f}")
    
    # 4. Intraday Pattern Analysis
    print("\n=== Intraday Pattern Analysis ===")
    for symbol in ["BTCUSDT", "ETHUSDT"]:
        patterns = analyzer.analyze_intraday_patterns(symbol)
        
        if not patterns.empty:
            print(f"\n{symbol} Hourly Patterns:")
            print("\nTop 3 Most Volatile Hours:")
            print(patterns['volatility'].nlargest(3))
            print("\nTop 3 Highest Volume Hours:")
            print(patterns['volume'].nlargest(3))
            
            # Plot intraday patterns
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
            plt.savefig(f"{symbol}_intraday_patterns.png")
            plt.close()

if __name__ == "__main__":
    main()
