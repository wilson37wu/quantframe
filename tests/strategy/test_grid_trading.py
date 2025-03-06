"""Unit tests for grid trading strategy."""
import pytest
import pandas as pd
import numpy as np
from quantframe.strategy.grid_trading import GridTradingStrategy
from quantframe.strategy.config_manager import StrategyConfigManager

def test_strategy_initialization(config_dir):
    """Test strategy initialization with configuration."""
    config_manager = StrategyConfigManager(str(config_dir))
    strategy = GridTradingStrategy(config_manager)
    
    assert strategy.config is not None
    assert strategy.config.grid_levels == 10
    assert strategy.config.grid_spacing == 0.01
    assert strategy.config.grid_type == "arithmetic"

def test_grid_creation(config_dir, sample_market_data):
    """Test grid level calculation."""
    config_manager = StrategyConfigManager(str(config_dir))
    strategy = GridTradingStrategy(config_manager)
    
    # Initialize grid
    current_price = sample_market_data['close'].iloc[-1]
    grid = strategy.create_grid(current_price)
    
    # Verify grid properties
    assert len(grid) == strategy.config.grid_levels
    assert all(grid[i] > grid[i+1] for i in range(len(grid)-1))  # Descending order
    
    # Test grid spacing
    if strategy.config.grid_type == "arithmetic":
        diffs = np.diff(grid)
        assert np.allclose(diffs, diffs[0], rtol=1e-10)
    else:  # geometric
        ratios = grid[1:] / grid[:-1]
        assert np.allclose(ratios, ratios[0], rtol=1e-10)

def test_signal_generation(config_dir, sample_market_data):
    """Test signal generation logic."""
    config_manager = StrategyConfigManager(str(config_dir))
    strategy = GridTradingStrategy(config_manager)
    
    # Generate signals
    signals = strategy.generate_signals(sample_market_data)
    
    # Verify signals
    assert isinstance(signals, list)
    for signal in signals:
        assert hasattr(signal, 'direction')
        assert hasattr(signal, 'strength')
        assert hasattr(signal, 'timestamp')
        assert hasattr(signal, 'metadata')
        assert 'grid_level' in signal.metadata
        assert 'grid_price' in signal.metadata
        
        # Verify signal logic
        current_price = sample_market_data['close'].iloc[-1]
        if signal.direction == 1:  # Buy
            assert signal.metadata['grid_price'] < current_price
        elif signal.direction == -1:  # Sell
            assert signal.metadata['grid_price'] > current_price

def test_position_sizing(config_dir, sample_market_data):
    """Test position size calculation."""
    config_manager = StrategyConfigManager(str(config_dir))
    strategy = GridTradingStrategy(config_manager)
    
    # Generate a signal
    signal = strategy.generate_signals(sample_market_data)[0]
    
    # Calculate position size
    position_size = strategy.calculate_position_size(signal, sample_market_data)
    
    # Verify position size
    assert isinstance(position_size, float)
    assert position_size >= strategy.config.base_order_size
    assert position_size <= strategy.config.max_position_size
    assert not np.isnan(position_size)
    assert not np.isinf(position_size)

def test_grid_rebalancing(config_dir, sample_market_data):
    """Test grid rebalancing logic."""
    config_manager = StrategyConfigManager(str(config_dir))
    strategy = GridTradingStrategy(config_manager)
    
    # Initialize grid
    initial_price = sample_market_data['close'].iloc[-1]
    initial_grid = strategy.create_grid(initial_price)
    
    # Test grid rebalancing with price movement
    new_price = initial_price * 1.1  # 10% price increase
    rebalanced_grid = strategy.rebalance_grid(new_price, initial_grid)
    
    # Verify rebalanced grid
    assert len(rebalanced_grid) == len(initial_grid)
    assert abs(new_price - np.mean(rebalanced_grid)) < abs(new_price - np.mean(initial_grid))

def test_dynamic_spacing(config_dir, sample_market_data):
    """Test dynamic grid spacing based on volatility."""
    config_manager = StrategyConfigManager(str(config_dir))
    strategy = GridTradingStrategy(config_manager)
    
    # Calculate spacing for different volatility regimes
    normal_volatility = strategy.calculate_grid_spacing(sample_market_data)
    
    # Create high volatility data
    high_vol_data = sample_market_data.copy()
    high_vol_data['close'] = high_vol_data['close'] * (1 + np.random.normal(0, 0.02, len(high_vol_data)))
    high_volatility = strategy.calculate_grid_spacing(high_vol_data)
    
    # Higher volatility should lead to wider grid spacing
    assert high_volatility > normal_volatility

def test_profit_taking(config_dir, sample_market_data):
    """Test profit-taking logic."""
    config_manager = StrategyConfigManager(str(config_dir))
    strategy = GridTradingStrategy(config_manager)
    
    # Initialize strategy state
    strategy.initialize_state(sample_market_data)
    
    # Simulate profitable position
    profitable_trade = {
        'entry_price': 100,
        'current_price': 102,  # 2% profit
        'position_size': strategy.config.base_order_size
    }
    
    # Check profit-taking signal
    should_take_profit = strategy.check_profit_taking(profitable_trade)
    assert should_take_profit == strategy.config.profit_taking['enabled']

def test_loss_handling(config_dir, sample_market_data):
    """Test loss handling mechanisms."""
    config_manager = StrategyConfigManager(str(config_dir))
    strategy = GridTradingStrategy(config_manager)
    
    # Initialize strategy state
    strategy.initialize_state(sample_market_data)
    
    # Simulate losing position
    losing_trade = {
        'entry_price': 100,
        'current_price': 95,  # 5% loss
        'position_size': strategy.config.base_order_size
    }
    
    # Check loss handling actions
    actions = strategy.handle_losses(losing_trade)
    assert 'stop_and_reset' in actions
    assert actions['stop_and_reset'] == strategy.config.loss_handling['stop_and_reset']

def test_error_handling(config_dir, sample_market_data):
    """Test error handling in strategy."""
    config_manager = StrategyConfigManager(str(config_dir))
    strategy = GridTradingStrategy(config_manager)
    
    # Test with invalid price data
    invalid_data = sample_market_data.copy()
    invalid_data.loc[invalid_data.index[0], 'close'] = -1
    
    with pytest.raises(ValueError):
        strategy.create_grid(invalid_data['close'].iloc[0])
    
    # Test with insufficient data
    with pytest.raises(ValueError):
        strategy.generate_signals(sample_market_data.iloc[:5])

def test_strategy_consistency(config_dir, sample_market_data):
    """Test strategy consistency with same input."""
    config_manager = StrategyConfigManager(str(config_dir))
    strategy = GridTradingStrategy(config_manager)
    
    # Generate signals multiple times
    signals1 = strategy.generate_signals(sample_market_data)
    signals2 = strategy.generate_signals(sample_market_data)
    
    # Verify consistency
    assert len(signals1) == len(signals2)
    for s1, s2 in zip(signals1, signals2):
        assert s1.direction == s2.direction
        assert s1.strength == s2.strength
        assert s1.metadata == s2.metadata

def test_volatility_adjustments(config_dir, sample_market_data):
    """Test volatility-based grid adjustments."""
    config_manager = StrategyConfigManager(str(config_dir))
    strategy = GridTradingStrategy(config_manager)
    
    # Calculate adjustments for different volatility regimes
    normal_adjustments = strategy.calculate_volatility_adjustments(sample_market_data)
    
    # Create high volatility data
    high_vol_data = sample_market_data.copy()
    high_vol_data['close'] = high_vol_data['close'] * (1 + np.random.normal(0, 0.03, len(high_vol_data)))
    high_vol_adjustments = strategy.calculate_volatility_adjustments(high_vol_data)
    
    # Verify adjustments
    assert high_vol_adjustments['spacing_multiplier'] > normal_adjustments['spacing_multiplier']
    assert high_vol_adjustments['grid_levels'] < normal_adjustments['grid_levels']
