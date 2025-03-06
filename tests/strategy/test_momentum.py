"""Unit tests for momentum strategy."""
import pytest
import pandas as pd
import numpy as np
from quantframe.strategy.momentum import MomentumStrategy
from quantframe.strategy.config_manager import StrategyConfigManager

def test_strategy_initialization(config_dir):
    """Test strategy initialization with configuration."""
    config_manager = StrategyConfigManager(str(config_dir))
    strategy = MomentumStrategy(config_manager)
    
    assert strategy.config is not None
    assert isinstance(strategy.config.lookback_periods, list)
    assert strategy.config.breakout_threshold == 2.0
    assert strategy.config.trend_threshold == 0.05

def test_signal_generation(config_dir, sample_market_data):
    """Test signal generation logic."""
    config_manager = StrategyConfigManager(str(config_dir))
    strategy = MomentumStrategy(config_manager)
    
    # Generate signals
    signals = strategy.generate_signals(sample_market_data)
    
    # Verify signals
    assert isinstance(signals, list)
    for signal in signals:
        assert hasattr(signal, 'direction')
        assert hasattr(signal, 'strength')
        assert hasattr(signal, 'timestamp')
        assert hasattr(signal, 'metadata')
        assert 'momentum_score' in signal.metadata
        assert 'volume_score' in signal.metadata
        assert 'trend_score' in signal.metadata
        
        # Verify signal logic
        if signal.direction == 1:  # Long
            assert signal.metadata['momentum_score'] > 0
            assert signal.metadata['trend_score'] > strategy.config.trend_threshold
        elif signal.direction == -1:  # Short
            assert signal.metadata['momentum_score'] < 0
            assert signal.metadata['trend_score'] < -strategy.config.trend_threshold

def test_momentum_calculation(config_dir, sample_market_data):
    """Test momentum calculation across different timeframes."""
    config_manager = StrategyConfigManager(str(config_dir))
    strategy = MomentumStrategy(config_manager)
    
    # Calculate momentum scores
    scores = strategy.calculate_momentum_scores(sample_market_data)
    
    # Verify scores
    assert isinstance(scores, pd.DataFrame)
    assert all(f'momentum_{period}' in scores.columns 
              for period in strategy.config.lookback_periods)
    assert not scores.isnull().any().any()
    assert all(-1 <= scores.iloc[-1] <= 1)  # Normalized scores

def test_volume_analysis(config_dir, sample_market_data):
    """Test volume analysis component."""
    config_manager = StrategyConfigManager(str(config_dir))
    strategy = MomentumStrategy(config_manager)
    
    # Calculate volume scores
    volume_scores = strategy.analyze_volume(sample_market_data)
    
    # Verify volume analysis
    assert isinstance(volume_scores, pd.Series)
    assert not volume_scores.isnull().any()
    assert all(0 <= volume_scores <= strategy.config.volume_factor)

def test_trend_detection(config_dir, sample_market_data):
    """Test trend detection logic."""
    config_manager = StrategyConfigManager(str(config_dir))
    strategy = MomentumStrategy(config_manager)
    
    # Detect trends
    trends = strategy.detect_trends(sample_market_data)
    
    # Verify trends
    assert isinstance(trends, pd.Series)
    assert not trends.isnull().any()
    assert all(abs(trends) <= 1)  # Normalized trend strength

def test_position_sizing(config_dir, sample_market_data):
    """Test position size calculation."""
    config_manager = StrategyConfigManager(str(config_dir))
    strategy = MomentumStrategy(config_manager)
    
    # Generate a signal
    signal = strategy.generate_signals(sample_market_data)[0]
    
    # Calculate position size
    position_size = strategy.calculate_position_size(signal, sample_market_data)
    
    # Verify position size
    assert isinstance(position_size, float)
    assert 0 <= position_size <= strategy.config.max_position_size
    assert not np.isnan(position_size)
    assert not np.isinf(position_size)

def test_correlation_filter(config_dir, sample_market_data):
    """Test correlation-based filtering."""
    config_manager = StrategyConfigManager(str(config_dir))
    strategy = MomentumStrategy(config_manager)
    
    # Create correlated asset data
    correlated_data = sample_market_data.copy()
    correlated_data['symbol'] = 'ETHUSDT'
    combined_data = pd.concat([sample_market_data, correlated_data])
    
    # Generate signals
    signals = strategy.generate_signals(combined_data)
    
    # Verify correlation filtering
    symbols_in_signals = set(s.symbol for s in signals)
    assert len(symbols_in_signals) <= 2  # Should filter highly correlated assets

def test_volatility_scaling(config_dir, sample_market_data):
    """Test volatility-based position scaling."""
    config_manager = StrategyConfigManager(str(config_dir))
    strategy = MomentumStrategy(config_manager)
    
    # Generate signals for different volatility regimes
    high_vol_data = sample_market_data.copy()
    high_vol_data['close'] = high_vol_data['close'] * (1 + np.random.normal(0, 0.02, len(high_vol_data)))
    
    signal_normal = strategy.generate_signals(sample_market_data)[0]
    signal_high_vol = strategy.generate_signals(high_vol_data)[0]
    
    size_normal = strategy.calculate_position_size(signal_normal, sample_market_data)
    size_high_vol = strategy.calculate_position_size(signal_high_vol, high_vol_data)
    
    # Higher volatility should lead to smaller position sizes
    assert size_high_vol < size_normal

def test_error_handling(config_dir, sample_market_data):
    """Test error handling in strategy."""
    config_manager = StrategyConfigManager(str(config_dir))
    strategy = MomentumStrategy(config_manager)
    
    # Test with insufficient data
    short_data = sample_market_data.iloc[-10:]
    with pytest.raises(ValueError):
        strategy.generate_signals(short_data)
    
    # Test with missing columns
    invalid_data = sample_market_data.drop('volume', axis=1)
    with pytest.raises(KeyError):
        strategy.generate_signals(invalid_data)

def test_strategy_consistency(config_dir, sample_market_data):
    """Test strategy consistency with same input."""
    config_manager = StrategyConfigManager(str(config_dir))
    strategy = MomentumStrategy(config_manager)
    
    # Generate signals multiple times
    signals1 = strategy.generate_signals(sample_market_data)
    signals2 = strategy.generate_signals(sample_market_data)
    
    # Verify consistency
    assert len(signals1) == len(signals2)
    for s1, s2 in zip(signals1, signals2):
        assert s1.direction == s2.direction
        assert s1.strength == s2.strength
        assert s1.metadata == s2.metadata
