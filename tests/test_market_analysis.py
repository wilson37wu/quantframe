"""Unit tests for market analysis functionality."""
import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from quantframe.analysis.market_analysis import MarketAnalyzer
from quantframe.data.base import DataSource

class MockDataSource(DataSource):
    """Mock data source for testing."""
    
    def __init__(self):
        super().__init__()
        
    def get_data(self, symbol: str, start_date: pd.Timestamp, 
                 end_date: pd.Timestamp, **kwargs) -> pd.DataFrame:
        """Return mock data for testing."""
        dates = pd.date_range(start_date, end_date, freq='D')
        n = len(dates)
        
        if kwargs.get('interval') == '1h':
            dates = pd.date_range(start_date, end_date, freq='h')
            n = len(dates)
        elif kwargs.get('interval') == '1m':
            dates = pd.date_range(start_date, end_date, freq='min')
            n = len(dates)
        
        # Generate random walk prices
        np.random.seed(42)
        returns = np.random.normal(0, 0.01, n)
        prices = 100 * np.exp(np.cumsum(returns))
        
        # Add some volatility clustering
        volatility = np.exp(np.random.normal(0, 0.5, n))
        returns = returns * volatility
        
        data = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.001, n)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.002, n))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.002, n))),
            'close': prices,
            'volume': np.random.lognormal(10, 1, n),
            'trades': np.random.randint(100, 1000, n)
        }, index=dates)
        
        data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        return data

class TestMarketAnalyzer(unittest.TestCase):
    """Test cases for MarketAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.data_source = MockDataSource()
        self.analyzer = MarketAnalyzer(self.data_source)
        
        # Create sample return series with known properties
        np.random.seed(42)
        n = 252  # One year of daily data
        self.returns = pd.Series(
            np.concatenate([
                np.random.normal(0, 0.01, n//3),  # Low volatility
                np.random.normal(0, 0.02, n//3),  # Normal volatility
                np.random.normal(0, 0.04, n//3)   # High volatility
            ])
        )
    
    def test_volatility_clustering(self):
        """Test volatility clustering analysis."""
        persistence, autocorr = self.analyzer.analyze_volatility_clustering(self.returns)
        
        self.assertIsInstance(persistence, float)
        self.assertIsInstance(autocorr, pd.Series)
        self.assertEqual(len(autocorr), 10)  # 10 lags
        
        # Autocorrelations should be between -1 and 1
        self.assertTrue(all((-1 <= x <= 1) for x in autocorr))
    
    def test_regime_changes(self):
        """Test volatility regime detection."""
        regimes = self.analyzer.detect_regime_changes(self.returns)
        
        self.assertIsInstance(regimes, pd.DataFrame)
        self.assertTrue(all(col in regimes.columns for col in 
                          ['volatility', 'zscore', 'regime', 'regime_change', 'regime_duration']))
        
        # Check regime labels
        unique_regimes = regimes['regime'].dropna().unique()
        self.assertTrue(all(regime in ['Low', 'Normal', 'High'] for regime in unique_regimes))
        
        # Check regime properties
        self.assertTrue(len(regimes) > 0)
        self.assertTrue(all(regimes['volatility'] >= 0))
        self.assertTrue(all(regimes['regime_duration'] > 0))
    
    def test_market_microstructure(self):
        """Test market microstructure analysis."""
        start_date = pd.Timestamp('2025-01-01')
        end_date = pd.Timestamp('2025-01-02')
        
        results = self.analyzer.analyze_market_microstructure(
            'BTCUSDT', start_date, end_date
        )
        
        self.assertIsInstance(results, dict)
        self.assertTrue(all(key in results for key in 
                          ['bid_ask_bounce', 'flow_imbalance', 'price_impact',
                           'trade_size_distribution']))
        
        # Check value ranges
        self.assertTrue(-1 <= results['bid_ask_bounce'] <= 1)
        self.assertTrue(-1 <= results['flow_imbalance'] <= 1)
        self.assertTrue(-1 <= results['price_impact'] <= 1)
        
        # Check trade size distribution
        size_dist = results['trade_size_distribution']
        self.assertTrue(all(size_dist[q1] <= size_dist[q2] 
                          for q1, q2 in zip(size_dist.keys(), list(size_dist.keys())[1:])))
    
    def test_momentum_reversal(self):
        """Test momentum and reversal analysis."""
        results = self.analyzer.analyze_momentum_reversal(self.returns)
        
        self.assertIsInstance(results, dict)
        for period in [5, 10, 21, 63]:
            self.assertIn(period, results)
            period_results = results[period]
            
            # Check metric presence
            self.assertTrue(all(key in period_results for key in 
                              ['momentum_sharpe', 'reversal_strength',
                               'momentum_reversal_ratio', 'hit_rate']))
            
            # Check value ranges
            self.assertTrue(-1 <= period_results['reversal_strength'] <= 1)
            self.assertTrue(0 <= period_results['hit_rate'] <= 1)
    
    def test_liquidity(self):
        """Test liquidity analysis."""
        start_date = pd.Timestamp('2025-01-01')
        end_date = pd.Timestamp('2025-01-10')
        
        results = self.analyzer.analyze_liquidity(
            'BTCUSDT', start_date, end_date
        )
        
        self.assertIsInstance(results, dict)
        self.assertTrue(all(key in results for key in 
                          ['illiquidity_ratio', 'turnover_ratio', 'relative_spread',
                           'typical_volume', 'price_impact_95pct']))
        
        # Check value ranges
        self.assertTrue(results['illiquidity_ratio'] >= 0)
        self.assertTrue(results['turnover_ratio'] >= 0)
        self.assertTrue(0 <= results['relative_spread'] <= 1)
        self.assertTrue(results['typical_volume'] >= 0)
    
    def test_market_impact(self):
        """Test market impact estimation."""
        # Use a smaller trade size relative to daily volume
        results = self.analyzer.estimate_market_impact('BTCUSDT', 1000)
        
        self.assertIsInstance(results, dict)
        self.assertTrue(all(key in results for key in 
                          ['participation_rate', 'total_impact_bps',
                           'temporary_impact_bps', 'permanent_impact_bps',
                           'daily_volume', 'volatility']))
        
        # Check value ranges
        self.assertTrue(0 <= float(results['participation_rate']) <= 1)
        self.assertTrue(results['total_impact_bps'] >= 0)
        self.assertTrue(results['temporary_impact_bps'] >= 0)
        self.assertTrue(results['permanent_impact_bps'] >= 0)
        self.assertTrue(results['daily_volume'] > 0)
        self.assertTrue(results['volatility'] >= 0)
    
    def test_tail_risk(self):
        """Test tail risk analysis."""
        results = self.analyzer.analyze_tail_risk(self.returns)
        
        self.assertIsInstance(results, dict)
        self.assertTrue(all(key in results for key in ['VaR', 'ES', 'tail_metrics', 'drawdown_metrics']))
        
        # Check VaR and ES
        for level in [0.99, 0.95]:
            self.assertIn(level, results['VaR'])
            self.assertIn(level, results['ES'])
            self.assertTrue(results['VaR'][level] <= 0)  # VaR should be negative
            self.assertTrue(results['ES'][level] <= results['VaR'][level])  # ES should be more extreme than VaR
        
        # Check tail metrics
        tail_metrics = results['tail_metrics']
        self.assertTrue(all(key in tail_metrics for key in 
                          ['positive_tail_mean', 'negative_tail_mean',
                           'positive_tail_vol', 'negative_tail_vol',
                           'tail_asymmetry']))
        
        # Check drawdown metrics
        drawdown_metrics = results['drawdown_metrics']
        self.assertTrue(all(key in drawdown_metrics for key in 
                          ['max_drawdown', 'avg_drawdown', 'drawdown_vol']))
        self.assertTrue(drawdown_metrics['max_drawdown'] <= 0)
        self.assertTrue(drawdown_metrics['avg_drawdown'] <= 0)
    
    def test_cross_market_dynamics(self):
        """Test cross-market analysis."""
        start_date = pd.Timestamp('2025-01-01')
        end_date = pd.Timestamp('2025-01-10')
        symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
        
        results = self.analyzer.analyze_cross_market_dynamics(
            symbols, start_date, end_date
        )
        
        self.assertIn('returns', results)
        self.assertIn('correlation', results)
        self.assertIn('betas', results)
        
        # Check correlation matrix properties
        corr_matrix = results['correlation']
        self.assertEqual(corr_matrix.shape, (len(symbols), len(symbols)))
        self.assertTrue(all((-1 <= x <= 1) for x in corr_matrix.values.flatten()))
    
    def test_intraday_patterns(self):
        """Test intraday pattern analysis."""
        patterns = self.analyzer.analyze_intraday_patterns('BTCUSDT', days=5)
        
        self.assertIsInstance(patterns, pd.DataFrame)
        self.assertTrue(all(col in patterns.columns for col in ['volatility', 'volume']))
        
        # Check hour range
        self.assertTrue(all(0 <= hour <= 23 for hour in patterns.index))
        
        # Check data validity
        self.assertTrue(all(patterns['volatility'] >= 0))
        self.assertTrue(all(patterns['volume'] >= 0))

if __name__ == '__main__':
    unittest.main()
