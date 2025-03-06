# Quantframe API Reference

## Project Structure

```
quantframe/
├── assets/           # Project assets and outputs
│   ├── images/      # Generated visualizations
│   │   ├── analysis/    # Market analysis charts
│   │   ├── backtest/    # Strategy backtest results
│   │   └── optimization/# Parameter optimization plots
│   └── logs/        # Application logs
├── config/          # Configuration management
├── data/           # Data source implementations
├── docs/           # Documentation
└── utils/          # Utility modules
```

## Configuration System

### Base Configuration (`quantframe.config.base_config`)
Core configuration infrastructure providing parameter validation and conversion utilities.

#### BaseConfig
Base class for all configuration objects in quantframe.

**Methods:**
- `to_dict()`: Convert configuration to dictionary format
- `to_dataframe()`: Convert configuration to DataFrame format
- `from_dict(config_dict)`: Create configuration from dictionary
- `validate_range(param_name, value, min_val, max_val)`: Validate parameter ranges
- `validate()`: Abstract method for parameter validation

### ICT Strategy Configuration (`quantframe.config.ict_config`)
Configuration parameters for the Inner Circle Trader (ICT) strategy implementation.

#### ICTConfig
Inherits from BaseConfig, providing ICT-specific parameter validation.

**Parameters:**
- FVG Parameters
  - `fvg_threshold`: 0.002 (0.01% - 5%)
  - Gap size detection threshold

- Order Block Parameters
  - `ob_lookback`: 20 bars (10-100)
  - `volume_threshold`: 1.5x (1.1x - 5.0x)

- Risk Management
  - `stop_loss`: 2% (0.5% - 10%)
  - `take_profit`: 3% (0.5% - 20%)
  - `risk_per_trade`: 1% (0.1% - 5%)

- Position Management
  - `min_volume`: 100,000 units
  - `max_positions`: 5 (1-20)

## Data Sources

### Base Data Source (`quantframe.data.sources.base_source`)
Abstract base class providing shared functionality for all data sources.

**Features:**
- Standardized data caching
- Column name mapping
- Error handling
- Logging integration

**Methods:**
- `get_data()`: Abstract method for data retrieval
- `load_cache()`: Load data from cache
- `save_cache()`: Save data to cache
- `standardize_columns()`: Standardize column names

### Binance Source (`quantframe.data.sources.binance_source`)
Implementation for retrieving data from Binance API.

**Features:**
- Real-time and historical data
- OHLCV data retrieval
- Market pair information
- Built-in caching system

**Timeframes:**
- Minutes: 1m, 5m, 15m
- Hours: 1h, 4h
- Days: 1d

### Yahoo Finance Source (`quantframe.data.sources.yfinance_source`)
Implementation for retrieving data from Yahoo Finance.

**Features:**
- Historical price data
- Adjusted price calculations
- Data caching system
- Multiple timeframe support

**Timeframes:**
- Minutes: 1m, 2m, 5m, 15m, 30m
- Hours: 1h
- Days: 1d, 5d
- Weeks/Months: 1wk, 1mo, 3mo

## Utilities

### Log Manager (`quantframe.utils.log_manager`)
Manages log files with rotation and cleanup capabilities.

**Features:**
- Automatic log rotation based on size
- Compression of rotated logs
- Cleanup of expired logs
- Configurable retention policies

**Configuration:**
- `max_size_mb`: Maximum log file size
- `max_age_days`: Log retention period
- `max_backups`: Number of backup files to keep

## Market Analysis (`quantframe.analysis.market_analysis`)

### MarketAnalyzer
Comprehensive market analysis toolkit.

**Methods:**
- `analyze_volatility_clustering()`: ARCH effects analysis
- `detect_regime_changes()`: Market regime detection
- `analyze_market_microstructure()`: Microstructure metrics
- `analyze_momentum_reversal()`: Momentum patterns
- `analyze_liquidity()`: Liquidity metrics
- `estimate_market_impact()`: Impact modeling
- `analyze_tail_risk()`: VaR and ES calculations
- `analyze_cross_market_dynamics()`: Correlation analysis
- `analyze_intraday_patterns()`: Pattern detection

## ICT Strategy (`quantframe.strategy.ict_strategy`)

### Core Components
- Fair Value Gap (FVG) detection
- Order Block identification
- Volume analysis
- Market structure tracking

**Key Features:**
- Configurable FVG detection
- Volume-based institutional analysis
- Position sizing
- Risk management integration
