"""Unit tests for mean reversion strategy."""
import pytest
import pandas as pd
import numpy as np
from quantframe.strategy.mean_reversion import MeanReversionStrategy
from quantframe.strategy.config_manager import StrategyConfigManager

def test_strategy_initialization(config_dir):
    """Test strategy initialization with configuration."""
    config_manager = StrategyConfigManager(str(config_dir))
    strategy = MeanReversionStrategy(config_manager)
    
    assert strategy.config is not None
    assert strategy.config.lookback_period == 20
    assert strategy.config.entry_zscore == 2.0
    assert strategy.config.exit_zscore == 0.5

def test_signal_generation(config_dir, sample_market_data):
    """Test signal generation logic."""
    config_manager = StrategyConfigManager(str(config_dir))
    strategy = MeanReversionStrategy(config_manager)
    
    # Generate signals
    signals = strategy.generate_signals(sample_market_data)
    
    # Verify signals
    assert isinstance(signals, list)
    for signal in signals:
        assert hasattr(signal, 'direction')
        assert hasattr(signal, 'strength')
        assert hasattr(signal, 'timestamp')
        assert hasattr(signal, 'metadata')
        assert 'zscore' in signal.metadata
        assert 'rsi' in signal.metadata
        
        # Verify signal logic
        if signal.direction == 1:  # Long
            assert signal.metadata['zscore'] <= -strategy.config.entry_zscore
            assert signal.metadata['rsi'] >= strategy.config.rsi_exit_threshold
        elif signal.direction == -1:  # Short
            assert signal.metadata['zscore'] >= strategy.config.entry_zscore
            assert signal.metadata['rsi'] <= strategy.config.rsi_entry_threshold
        elif signal.direction == 0:  # Exit
            assert abs(signal.metadata['zscore']) <= strategy.config.exit_zscore

def test_position_sizing(config_dir, sample_market_data):
    """Test position size calculation."""
    config_manager = StrategyConfigManager(str(config_dir))
    strategy = MeanReversionStrategy(config_manager)
    
    # Create a test signal
    signal = strategy.generate_signals(sample_market_data)[0]
    
    # Calculate position size
    position_size = strategy.calculate_position_size(signal, sample_market_data)
    
    # Verify position size
    assert isinstance(position_size, float)
    assert 0 <= position_size <= strategy.config.max_position_size
    assert not np.isnan(position_size)
    assert not np.isinf(position_size)

def test_stop_loss_calculation(config_dir, sample_market_data):
    """Test stop loss price calculation."""
    config_manager = StrategyConfigManager(str(config_dir))
    strategy = MeanReversionStrategy(config_manager)
    
    # Create a test signal
    signal = strategy.generate_signals(sample_market_data)[0]
    
    # Calculate stop loss
    stop_loss = strategy.get_stop_loss(signal, sample_market_data)
    
    # Verify stop loss
    assert isinstance(stop_loss, float)
    assert not np.isnan(stop_loss)
    assert not np.isinf(stop_loss)
    
    current_price = sample_market_data['close'].iloc[-1]
    if signal.direction == 1:  # Long
        assert stop_loss < current_price
    elif signal.direction == -1:  # Short
        assert stop_loss > current_price

def test_take_profit_calculation(config_dir, sample_market_data):
    """Test take profit price calculation."""
    config_manager = StrategyConfigManager(str(config_dir))
    strategy = MeanReversionStrategy(config_manager)
    
    # Create a test signal
    signal = strategy.generate_signals(sample_market_data)[0]
    
    # Calculate take profit
    take_profit = strategy.get_take_profit(signal, sample_market_data)
    
    # Verify take profit
    assert isinstance(take_profit, float)
    assert not np.isnan(take_profit)
    assert not np.isinf(take_profit)
    
    current_price = sample_market_data['close'].iloc[-1]
    if signal.direction == 1:  # Long
        assert take_profit > current_price
    elif signal.direction == -1:  # Short
        assert take_profit < current_price

def test_market_filters(config_dir, sample_market_data):
    """Test market filtering logic."""
    config_manager = StrategyConfigManager(str(config_dir))
    strategy = MeanReversionStrategy(config_manager)
    
    # Test with valid market data
    signals = strategy.generate_signals(sample_market_data)
    assert len(signals) > 0
    
    # Test with invalid volume
    invalid_volume_data = sample_market_data.copy()
    invalid_volume_data['volume'] = 0
    signals = strategy.generate_signals(invalid_volume_data)
    assert len(signals) == 0
    
    # Test with invalid spread
    invalid_spread_data = sample_market_data.copy()
    invalid_spread_data['high'] = invalid_spread_data['close'] * 1.1  # 10% spread
    signals = strategy.generate_signals(invalid_spread_data)
    assert len(signals) == 0

def test_error_handling(config_dir, sample_market_data):
    """Test error handling in strategy."""
    config_manager = StrategyConfigManager(str(config_dir))
    strategy = MeanReversionStrategy(config_manager)
    
    # Test with missing data
    with pytest.raises(KeyError):
        strategy.generate_signals(pd.DataFrame())
    
    # Test with NaN values
    invalid_data = sample_market_data.copy()
    invalid_data.loc[invalid_data.index[0], 'close'] = np.nan
    signals = strategy.generate_signals(invalid_data)
    assert len(signals) > 0  # Strategy should handle NaN values gracefully

def test_strategy_consistency(config_dir, sample_market_data):
    """Test strategy consistency with same input."""
    config_manager = StrategyConfigManager(str(config_dir))
    strategy = MeanReversionStrategy(config_manager)
    
    # Generate signals multiple times
    signals1 = strategy.generate_signals(sample_market_data)
    signals2 = strategy.generate_signals(sample_market_data)
    
    # Verify consistency
    assert len(signals1) == len(signals2)
    for s1, s2 in zip(signals1, signals2):
        assert s1.direction == s2.direction
        assert s1.strength == s2.strength
        assert s1.metadata == s2.metadata
