# QuantFrame

A comprehensive quantitative trading framework designed for research, backtesting, and live trading.

## Features

- **Modular Architecture**: Clean separation of concerns between data, strategy, and execution layers
- **Data Management**: 
  - Multiple data source support (stocks, crypto, forex)
  - Efficient data processing and storage
  - Data validation and quality checks
- **Strategy Development**:
  - Strategy component library
  - Signal generation framework
  - Portfolio management
  - Risk management
- **Backtesting Engine**:
  - Event-driven architecture
  - Transaction cost modeling
  - Multiple timeframe support
  - Walk-forward analysis
- **Performance Analytics**:
  - Comprehensive performance metrics
  - Risk analytics
  - Trade attribution
  - Interactive reporting

## Project Structure

```
quantframe/
├── quantframe/          # Main package
│   ├── core/           # Core functionality
│   ├── data/           # Data management
│   │   ├── sources/    # Data source implementations
│   │   ├── processors/ # Data processing pipelines
│   │   └── storage/    # Data storage implementations
│   ├── strategy/       # Strategy implementations
│   │   ├── components/ # Reusable strategy components
│   │   ├── signals/    # Signal generators
│   │   └── portfolio/  # Portfolio management
│   ├── backtesting/    # Backtesting engine
│   ├── analytics/      # Analysis tools
│   │   ├── performance/# Performance metrics
│   │   ├── risk/       # Risk metrics
│   │   └── reporting/  # Report generation
│   ├── risk/           # Risk management
│   └── utils/          # Utility functions
├── config/             # Configuration files
├── tests/              # Test suite
├── examples/           # Example implementations
├── docs/              # Documentation
└── notebooks/         # Jupyter notebooks
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/quantframe.git
cd quantframe

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package in development mode
pip install -e ".[dev]"
```

## Quick Start

```python
from quantframe.data.sources import YFinanceDataSource
from quantframe.strategy import MeanReversionStrategy
from quantframe.backtesting import Backtester

# Initialize data source
data_source = YFinanceDataSource()
data = data_source.get_data("AAPL", "2020-01-01", "2023-12-31")

# Create strategy
strategy = MeanReversionStrategy(
    lookback_period=20,
    entry_threshold=2.0,
    exit_threshold=0.5
)

# Run backtest
backtester = Backtester(strategy)
results = backtester.run(data)

# Analyze results
results.plot_performance()
results.print_statistics()
```

## Development

1. Install development dependencies:
```bash
pip install -e ".[dev]"
```

2. Install pre-commit hooks:
```bash
pre-commit install
```

3. Run tests:
```bash
pytest tests/
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
