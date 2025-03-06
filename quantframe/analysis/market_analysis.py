"""Market analysis tools for cryptocurrency and traditional markets.

This module provides advanced market analysis capabilities for both cryptocurrency
and traditional markets. Key features include:
- Volatility analysis and regime detection
- Market microstructure analysis
- Momentum and mean reversion analysis
- Liquidity analysis and market impact estimation

The MarketAnalyzer class serves as the main interface for all analysis tools,
providing a consistent API for different types of market analysis.

Example:
    >>> from quantframe.data import YFinanceSource
    >>> source = YFinanceSource(config={'cache_dir': 'cache'})
    >>> analyzer = MarketAnalyzer(source)
    >>> regimes = analyzer.detect_regime_changes(returns_data)
    >>> print(f"Current regime: {regimes['regime'].iloc[-1]}")
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
from ..data.base import DataSource

class MarketAnalyzer:
    """Advanced market analysis tools for various asset classes.
    
    This class provides a comprehensive suite of market analysis tools,
    focusing on:
    - Volatility patterns and regime changes
    - Market microstructure analysis
    - Price momentum and mean reversion
    - Market liquidity and efficiency
    
    The analyzer requires a data source that implements the DataSource
    interface to fetch market data for analysis.
    
    Attributes:
        data_source (DataSource): Data source for market data retrieval
        
    Example:
        >>> analyzer = MarketAnalyzer(data_source)
        >>> vol_persistence, autocorr = analyzer.analyze_volatility_clustering(returns)
        >>> print(f"Volatility persistence: {vol_persistence:.2f}")
    """
    
    def __init__(self, data_source: DataSource):
        """Initialize market analyzer with a data source.
        
        Args:
            data_source (DataSource): Data source instance implementing
                the DataSource interface. Must support fetching OHLCV data
                with configurable intervals.
        """
        self.data_source = data_source
    
    def analyze_volatility_clustering(self, returns: pd.Series) -> Tuple[float, pd.Series]:
        """Analyze volatility clustering using autocorrelation of squared returns.
        
        Implements the ARCH effect analysis by examining the autocorrelation
        structure of squared returns. High persistence indicates strong
        volatility clustering.
        
        Args:
            returns (pd.Series): Series of log returns. Should be
                calculated as log(price_t/price_t-1)
            
        Returns:
            Tuple[float, pd.Series]: Two-element tuple containing:
                - float: Volatility persistence (sum of autocorrelations)
                - pd.Series: Autocorrelations for lags 1-10
                
        Example:
            >>> returns = np.log(prices).diff()
            >>> persistence, autocorr = analyzer.analyze_volatility_clustering(returns)
            >>> print(f"First-order autocorrelation: {autocorr[1]:.3f}")
        
        Note:
            Volatility persistence > 0.5 typically indicates significant
            clustering. Values > 0.7 suggest strong ARCH effects.
        """
        squared_returns = returns ** 2
        autocorr = pd.Series({
            lag: squared_returns.autocorr(lag=lag)
            for lag in range(1, 11)
        })
        persistence = autocorr.sum()
        return persistence, autocorr
    
    def detect_regime_changes(self, returns: pd.Series, window: int = 21) -> pd.DataFrame:
        """Detect volatility regime changes using rolling statistics.
        
        Identifies distinct volatility regimes (Low, Normal, High) using
        a z-score approach based on rolling volatility. The method uses
        robust statistics to minimize the impact of outliers.
        
        Args:
            returns (pd.Series): Series of log returns
            window (int, optional): Rolling window size for volatility
                calculation. Defaults to 21 (about one trading month)
            
        Returns:
            pd.DataFrame: DataFrame containing:
                - volatility: Annualized rolling volatility
                - zscore: Standardized volatility score
                - regime: Current volatility regime
                - regime_change: Binary indicator of regime changes
                - regime_duration: Length of current regime
                
        Example:
            >>> regimes = analyzer.detect_regime_changes(returns)
            >>> regime_counts = regimes['regime'].value_counts()
            >>> print("Regime distribution:\\n", regime_counts)
            
        Note:
            Regime thresholds are set at ±1.5 standard deviations from
            the mean, using the interquartile range for robustness.
        """
        # Ensure returns are numeric and replace inf/nan with 0
        returns = pd.to_numeric(returns, errors='coerce').fillna(0)
        returns = returns.replace([np.inf, -np.inf], 0)
        
        # Calculate rolling volatility (always non-negative)
        vol = np.sqrt((returns ** 2).rolling(window=window).mean()) * np.sqrt(252)
        vol = vol.fillna(0)  # Replace NaN with 0 for initial window
        
        # Calculate z-score of volatility using robust statistics
        vol_mean = vol.rolling(window=63).median()
        vol_std = vol.rolling(window=63).quantile(0.75) - vol.rolling(window=63).quantile(0.25)
        vol_zscore = (vol - vol_mean) / vol_std.replace(0, np.nan)
        
        # Create initial DataFrame
        regimes = pd.DataFrame({
            'volatility': vol,
            'zscore': vol_zscore
        })
        
        # Assign regimes based on z-score
        regimes['regime'] = pd.cut(
            vol_zscore,
            bins=[-np.inf, -1.5, 1.5, np.inf],
            labels=['Low', 'Normal', 'High']
        )
        
        # Fill NaN regimes with 'Normal'
        regimes['regime'] = regimes['regime'].fillna('Normal')
        
        # Calculate regime changes and durations
        regimes['regime_change'] = (regimes['regime'] != regimes['regime'].shift(1)).astype(int)
        regimes['regime_group'] = regimes['regime_change'].cumsum()
        regimes['regime_duration'] = regimes.groupby('regime_group')['regime'].transform('count')
        
        # Drop the temporary regime group column
        regimes = regimes.drop('regime_group', axis=1)
        
        return regimes
    
    def analyze_market_microstructure(self, symbol: str, start_date: pd.Timestamp,
                                    end_date: pd.Timestamp, interval: str = '1m') -> Dict:
        """Analyze market microstructure metrics.
        
        Computes various microstructure metrics to assess market quality
        and trading costs. Metrics include:
        - Bid-ask bounce estimation
        - Order flow imbalance
        - Price impact coefficients
        - Trade size distribution
        
        Args:
            symbol (str): Trading symbol to analyze
            start_date (pd.Timestamp): Analysis start date
            end_date (pd.Timestamp): Analysis end date
            interval (str, optional): Data interval. Defaults to '1m'
                for minute-level analysis
            
        Returns:
            Dict: Dictionary containing:
                - bid_ask_bounce: Estimate of effective spread
                - flow_imbalance: Buy/sell volume imbalance
                - price_impact: Price sensitivity to trade size
                - trade_size_distribution: Size quantiles
                
        Example:
            >>> metrics = analyzer.analyze_market_microstructure(
            ...     'BTC/USD',
            ...     pd.Timestamp('2023-01-01'),
            ...     pd.Timestamp('2023-12-31')
            ... )
            >>> print(f"Price impact coefficient: {metrics['price_impact']:.3f}")
            
        Note:
            Requires high-frequency data for accurate results.
            Minute-level data is recommended for crypto markets.
        """
        try:
            data = self.data_source.get_data(symbol, start_date, end_date, interval=interval)
            
            # Calculate price changes
            data['price_change'] = data['close'].diff()
            data['direction'] = np.sign(data['price_change'])
            
            # Estimate bid-ask bounce
            bounce_metric = -data['direction'].autocorr(1)
            
            # Calculate order flow imbalance
            data['buy_volume'] = data['volume'] * (data['direction'] > 0)
            data['sell_volume'] = data['volume'] * (data['direction'] < 0)
            flow_imbalance = (data['buy_volume'].sum() - data['sell_volume'].sum()) / data['volume'].sum()
            
            # Estimate price impact
            log_volume = np.log(data['volume'].replace(0, data['volume'].mean()))
            impact_coef = np.abs(data['price_change']).corr(log_volume)
            
            # Analyze trade size distribution
            size_quantiles = data['volume'].quantile([0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
            
            return {
                'bid_ask_bounce': bounce_metric,
                'flow_imbalance': flow_imbalance,
                'price_impact': impact_coef,
                'trade_size_distribution': size_quantiles.to_dict()
            }
            
        except Exception as e:
            print(f"Error in market microstructure analysis: {e}")
            return {}
    
    def analyze_momentum_reversal(self, returns: pd.Series, 
                                lookback_periods: List[int] = [5, 10, 21, 63]) -> Dict:
        """Analyze momentum and mean reversion characteristics.
        
        Examines the presence and strength of momentum and mean reversion
        effects across multiple timeframes. Useful for:
        - Strategy development
        - Market regime identification
        - Risk factor analysis
        
        Args:
            returns (pd.Series): Series of asset returns
            lookback_periods (List[int], optional): Periods to analyze.
                Defaults to [5, 10, 21, 63] for standard trading periods
            
        Returns:
            Dict: Dictionary with results for each period containing:
                - momentum_sharpe: Momentum strategy Sharpe ratio
                - reversal_strength: Mean reversion coefficient
                - momentum_reversal_ratio: Balance between effects
                - hit_rate: Momentum prediction accuracy
                
        Example:
            >>> results = analyzer.analyze_momentum_reversal(returns)
            >>> for period, metrics in results.items():
            ...     print(f"{period}-day momentum Sharpe: "
            ...           f"{metrics['momentum_sharpe']:.2f}")
            
        Note:
            Momentum_reversal_ratio > 1 suggests momentum dominance,
            while ratio < 1 suggests mean reversion dominance.
        """
        results = {}
        
        for period in lookback_periods:
            # Calculate momentum factor
            momentum = returns.rolling(period).sum()
            
            # Calculate reversal factor (negative autocorrelation)
            reversal = -returns.autocorr(period)
            
            # Calculate momentum/reversal ratio
            ratio = momentum.std() / (returns.std() * np.sqrt(period))
            
            # Calculate hit rate (percentage of momentum continuation)
            hit_rate = (np.sign(momentum) == np.sign(returns.shift(-1))).mean()
            
            results[period] = {
                'momentum_sharpe': momentum.mean() / momentum.std() * np.sqrt(252),
                'reversal_strength': reversal,
                'momentum_reversal_ratio': ratio,
                'hit_rate': hit_rate
            }
        
        return results
    
    def analyze_liquidity(self, symbol: str, start_date: pd.Timestamp,
                         end_date: pd.Timestamp, interval: str = '1h') -> Dict:
        """Analyze market liquidity metrics.
        
        Computes comprehensive liquidity metrics to assess market quality
        and trading costs. Includes standard and advanced measures:
        - Amihud illiquidity ratio
        - Turnover analysis
        - VWAP calculations
        - Spread estimates
        
        Args:
            symbol (str): Trading symbol to analyze
            start_date (pd.Timestamp): Analysis start date
            end_date (pd.Timestamp): Analysis end date
            interval (str, optional): Data interval. Defaults to '1h'
                
        Returns:
            Dict: Dictionary containing liquidity metrics:
                - illiquidity: Amihud ratio
                - turnover: Average daily turnover
                - vwap_stats: VWAP analysis results
                - spread_metrics: Various spread estimates
                
        Example:
            >>> liquidity = analyzer.analyze_liquidity(
            ...     'ETH/USD',
            ...     pd.Timestamp('2023-01-01'),
            ...     pd.Timestamp('2023-12-31')
            ... )
            >>> print(f"Illiquidity ratio: {liquidity['illiquidity']:.6f}")
            
        Note:
            Higher Amihud illiquidity ratio indicates lower market liquidity
            and potentially higher trading costs.
        """
        try:
            data = self.data_source.get_data(symbol, start_date, end_date, interval=interval)
            
            # Calculate Amihud illiquidity ratio
            data['returns_to_volume'] = np.abs(data['log_returns']) / data['volume']
            illiquidity = data['returns_to_volume'].mean()
            
            # Calculate turnover ratio
            turnover = data['volume'].mean() * data['close'].mean()
            
            # Calculate volume-weighted average price (VWAP)
            data['vwap'] = (data['volume'] * data['close']).cumsum() / data['volume'].cumsum()
            
            # Calculate spread estimates
            high_low_spread = (data['high'] - data['low']) / data['close']
            
            # Calculate market depth estimates
            typical_volume = data['volume'].quantile(0.5)
            price_impact = data['returns_to_volume'].quantile(0.95)
            
            return {
                'illiquidity_ratio': illiquidity,
                'turnover_ratio': turnover,
                'relative_spread': high_low_spread.mean(),
                'typical_volume': typical_volume,
                'price_impact_95pct': price_impact
            }
            
        except Exception as e:
            print(f"Error in liquidity analysis: {e}")
            return {}
    
    def estimate_market_impact(self, symbol: str, trade_size: float,
                             lookback_days: int = 30) -> Dict:
        """Estimate market impact for a given trade size.
        
        Computes market impact estimates using a square root model.
        Estimates include:
        - Participation rate
        - Temporary impact
        - Permanent impact
        
        Args:
            symbol (str): Trading symbol
            trade_size (float): Size of trade in base currency
            lookback_days (int, optional): Days of historical data to use.
                Defaults to 30
            
        Returns:
            Dict: Dictionary containing impact estimates:
                - participation_rate: Trade size relative to average daily volume
                - total_impact_bps: Total market impact in basis points
                - temporary_impact_bps: Temporary market impact in basis points
                - permanent_impact_bps: Permanent market impact in basis points
                
        Example:
            >>> impact = analyzer.estimate_market_impact(
            ...     'BTC/USD',
            ...     1000.0,
            ...     lookback_days=60
            ... )
            >>> print(f"Total market impact: {impact['total_impact_bps']:.2f} bps")
            
        Note:
            Market impact estimates are sensitive to trade size and
            historical data quality.
        """
        try:
            end_date = pd.Timestamp.now()
            start_date = end_date - pd.Timedelta(days=lookback_days)
            data = self.data_source.get_data(symbol, start_date, end_date, interval='1h')
            
            # Calculate average daily volume
            daily_volume = data['volume'].resample('D').sum().mean()
            
            # Calculate participation rate
            participation = trade_size / daily_volume
            
            # Estimate price impact using square root model
            sqrt_impact = 0.1 * np.sqrt(participation)
            
            # Estimate temporary impact
            temp_impact = sqrt_impact * 0.6
            
            # Estimate permanent impact
            perm_impact = sqrt_impact * 0.4
            
            return {
                'participation_rate': participation,
                'total_impact_bps': sqrt_impact * 10000,
                'temporary_impact_bps': temp_impact * 10000,
                'permanent_impact_bps': perm_impact * 10000,
                'daily_volume': daily_volume,
                'volatility': data['log_returns'].std() * np.sqrt(252)
            }
            
        except Exception as e:
            print(f"Error in market impact estimation: {e}")
            return {}
    
    def analyze_tail_risk(self, returns: pd.Series, confidence_levels: List[float] = [0.99, 0.95]) -> Dict:
        """Analyze tail risk measures.
        
        Computes Value at Risk (VaR) and Expected Shortfall (ES) for
        specified confidence levels. Also analyzes tail dependence using
        extreme value theory.
        
        Args:
            returns (pd.Series): Series of asset returns
            confidence_levels (List[float], optional): Confidence levels
                for VaR and ES calculations. Defaults to [0.99, 0.95]
            
        Returns:
            Dict: Dictionary containing tail risk metrics:
                - VaR: Value at Risk for each confidence level
                - ES: Expected Shortfall for each confidence level
                - tail_metrics: Tail dependence metrics
                
        Example:
            >>> tail_risk = analyzer.analyze_tail_risk(returns)
            >>> print(f"VaR (99%): {tail_risk['VaR'][0.99]:.2f}")
            
        Note:
            VaR and ES estimates are sensitive to the choice of confidence
            levels and historical data quality.
        """
        try:
            results = {}
            
            # Calculate Value at Risk (VaR)
            var_metrics = {}
            for level in confidence_levels:
                var_metrics[level] = np.percentile(returns, (1 - level) * 100)
            results['VaR'] = var_metrics
            
            # Calculate Expected Shortfall (ES)
            es_metrics = {}
            for level in confidence_levels:
                threshold = np.percentile(returns, (1 - level) * 100)
                es_metrics[level] = returns[returns <= threshold].mean()
            results['ES'] = es_metrics
            
            # Calculate tail dependence using extreme value theory
            positive_tail = returns[returns > returns.quantile(0.95)]
            negative_tail = returns[returns < returns.quantile(0.05)]
            
            results['tail_metrics'] = {
                'positive_tail_mean': positive_tail.mean(),
                'negative_tail_mean': negative_tail.mean(),
                'positive_tail_vol': positive_tail.std(),
                'negative_tail_vol': negative_tail.std(),
                'tail_asymmetry': len(positive_tail) / len(negative_tail)
            }
            
            # Calculate maximum drawdown
            cum_returns = (1 + returns).cumprod()
            rolling_max = cum_returns.expanding().max()
            drawdowns = cum_returns / rolling_max - 1
            
            results['drawdown_metrics'] = {
                'max_drawdown': drawdowns.min(),
                'avg_drawdown': drawdowns.mean(),
                'drawdown_vol': drawdowns.std()
            }
            
            return results
            
        except Exception as e:
            print(f"Error in tail risk analysis: {e}")
            return {}
    
    def analyze_cross_market_dynamics(self, symbols: List[str], 
                                    start_date: pd.Timestamp,
                                    end_date: pd.Timestamp,
                                    interval: str = '1d') -> Dict:
        """Analyze cross-market dynamics including correlations and betas.
        
        Computes cross-market correlations and betas for a list of symbols.
        
        Args:
            symbols (List[str]): List of trading symbols
            start_date (pd.Timestamp): Analysis start date
            end_date (pd.Timestamp): Analysis end date
            interval (str, optional): Data interval. Defaults to '1d'
                
        Returns:
            Dict: Dictionary containing:
                - returns: DataFrame of returns for each symbol
                - correlation: Correlation matrix of returns
                - betas: Dictionary of betas for each symbol
                
        Example:
            >>> dynamics = analyzer.analyze_cross_market_dynamics(
            ...     ['BTC/USD', 'ETH/USD'],
            ...     pd.Timestamp('2023-01-01'),
            ...     pd.Timestamp('2023-12-31')
            ... )
            >>> print(f"Correlation (BTC/USD, ETH/USD): {dynamics['correlation'].iloc[0, 1]:.3f}")
            
        Note:
            Betas are calculated relative to the first symbol in the list.
        """
        returns_data = {}
        
        for symbol in symbols:
            try:
                df = self.data_source.get_data(symbol, start_date, end_date, interval=interval)
                returns_data[symbol] = df['log_returns'].fillna(0)
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
                
        if not returns_data:
            return {}
            
        returns_df = pd.DataFrame(returns_data)
        corr_matrix = returns_df.corr()
        
        betas = {}
        if 'BTCUSDT' in symbols:
            btc_returns = returns_df['BTCUSDT']
            for symbol in symbols:
                if symbol != 'BTCUSDT':
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
        """Analyze intraday volatility and volume patterns.
        
        Computes hourly patterns of volatility and volume for a given symbol.
        
        Args:
            symbol (str): Trading symbol
            days (int, optional): Number of days to analyze. Defaults to 30
            
        Returns:
            pd.DataFrame: DataFrame containing hourly patterns of:
                - volatility: Hourly volatility
                - volume: Hourly volume
                
        Example:
            >>> patterns = analyzer.analyze_intraday_patterns('BTC/USD')
            >>> print(f"Average hourly volatility: {patterns['volatility'].mean():.3f}")
            
        Note:
            Hourly patterns are calculated using a rolling window approach.
        """
        end_date = pd.Timestamp.now()
        start_date = end_date - pd.Timedelta(days=days)
        
        try:
            data = self.data_source.get_data(symbol, start_date, end_date, interval='1h')
            data['hour'] = data.index.hour
            
            hourly_patterns = pd.DataFrame({
                'volatility': data.groupby('hour')['log_returns'].std() * np.sqrt(24),
                'volume': data.groupby('hour')['volume'].mean(),
                'trades': data.groupby('hour')['trades'].mean() if 'trades' in data.columns else None
            }).fillna(0)
            
            return hourly_patterns
            
        except Exception as e:
            print(f"Error analyzing intraday patterns for {symbol}: {e}")
            return pd.DataFrame()
