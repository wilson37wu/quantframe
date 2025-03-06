"""Test fixtures for strategy tests."""
import os
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

@pytest.fixture
def sample_market_data():
    """Create sample market data for testing."""
    dates = pd.date_range(start='2025-01-01', end='2025-01-10', freq='1H')
    n_samples = len(dates)
    
    # Generate realistic price movement
    np.random.seed(42)
    returns = np.random.normal(0, 0.002, n_samples)
    price = 100 * np.exp(np.cumsum(returns))
    
    # Create sample data
    data = pd.DataFrame({
        'open': price * (1 + np.random.normal(0, 0.0005, n_samples)),
        'high': price * (1 + abs(np.random.normal(0, 0.001, n_samples))),
        'low': price * (1 - abs(np.random.normal(0, 0.001, n_samples))),
        'close': price,
        'volume': np.random.lognormal(10, 1, n_samples),
        'symbol': 'BTCUSDT'
    }, index=dates)
    
    # Ensure high/low are actually high/low
    data['high'] = data[['open', 'close', 'high']].max(axis=1)
    data['low'] = data[['open', 'close', 'low']].min(axis=1)
    
    return data

@pytest.fixture
def config_dir(tmp_path):
    """Create temporary config directory with test configurations."""
    config_dir = tmp_path / "config" / "strategies"
    config_dir.mkdir(parents=True)
    
    # Create test configurations
    mean_reversion_config = {
        "lookback_period": 20,
        "entry_zscore": 2.0,
        "exit_zscore": 0.5,
        "rsi_period": 14,
        "rsi_entry_threshold": 30,
        "rsi_exit_threshold": 70,
        "volatility_lookback": 21,
        "position_size_atr_multiple": 0.5,
        "kelly_fraction": 0.5,
        "max_position_size": 0.1,
        "stop_loss_atr_multiple": 2.0,
        "take_profit_atr_multiple": 3.0,
        "max_drawdown": 0.2,
        "max_leverage": 2.0,
        "min_adv": 1000000,
        "min_price": 1.0,
        "max_spread": 0.005,
        "order_type": "LIMIT",
        "slippage_tolerance": 0.001,
        "execution_timeout": 30,
        "retry_attempts": 3,
        "retry_delay": 5,
        "health_check_interval": 60,
        "logging_level": "INFO"
    }
    
    momentum_config = {
        "lookback_periods": [20, 60, 120],
        "breakout_threshold": 2.0,
        "trend_threshold": 0.05,
        "momentum_smoothing": 10,
        "volume_factor": 1.5,
        "volatility_lookback": 21,
        "position_size_atr_multiple": 0.5,
        "kelly_fraction": 0.5,
        "max_position_size": 0.1,
        "position_update_frequency": "1d",
        "stop_loss_atr_multiple": 2.0,
        "take_profit_atr_multiple": 4.0,
        "max_drawdown": 0.25,
        "max_leverage": 2.0,
        "correlation_threshold": 0.7,
        "min_adv": 2000000,
        "min_price": 1.0,
        "max_spread": 0.003,
        "order_type": "LIMIT",
        "slippage_tolerance": 0.001,
        "execution_timeout": 30,
        "retry_attempts": 3,
        "retry_delay": 5,
        "health_check_interval": 60,
        "logging_level": "INFO",
        "signal_weights": {
            "price_momentum": 0.4,
            "volume_momentum": 0.2,
            "volatility_regime": 0.2,
            "market_regime": 0.2
        }
    }
    
    grid_config = {
        "grid_levels": 10,
        "grid_spacing": 0.01,
        "grid_type": "arithmetic",
        "rebalance_threshold": 0.005,
        "base_order_size": 100,
        "position_step_size": 10,
        "size_scaling_factor": 1.5,
        "max_position_size": 0.1,
        "max_drawdown": 0.15,
        "max_leverage": 3.0,
        "dynamic_spacing": True,
        "volatility_scaling": True,
        "vol_lookback": 20,
        "vol_multiplier": 1.5,
        "order_type": "LIMIT",
        "slippage_tolerance": 0.001,
        "execution_timeout": 30,
        "retry_attempts": 3,
        "retry_delay": 5,
        "health_check_interval": 60,
        "logging_level": "INFO",
        "profit_taking": {
            "enabled": True,
            "threshold": 0.02,
            "reinvest": True
        },
        "loss_handling": {
            "stop_and_reset": True,
            "partial_close": False,
            "hedge_at_extremes": True
        },
        "volatility_adjustments": {
            "enabled": True,
            "vol_window": 24,
            "vol_threshold_high": 0.4,
            "vol_threshold_low": 0.1,
            "spacing_multiplier": 1.5
        }
    }
    
    # Write configurations to files
    import yaml
    with open(config_dir / "mean_reversion.yaml", "w") as f:
        yaml.safe_dump(mean_reversion_config, f)
    with open(config_dir / "momentum.yaml", "w") as f:
        yaml.safe_dump(momentum_config, f)
    with open(config_dir / "grid_trading.yaml", "w") as f:
        yaml.safe_dump(grid_config, f)
    
    return config_dir
