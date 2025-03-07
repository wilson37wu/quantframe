# Quantframe

A quantitative trading framework with a focus on ICT (Inner Circle Trader) strategy implementation and market analysis.

## Features

- **ICT Strategy Implementation**
  - Fair Value Gap (FVG) detection
  - Order Block identification
  - Volume-based institutional activity analysis
  - Market structure tracking

- **Data Sources**
  - Binance cryptocurrency data
  - Yahoo Finance market data
  - Extensible base class for additional sources

- **Configuration System**
  - Parameter validation with ranges
  - Type checking and conversion utilities
  - Strategy-specific configurations

- **Market Analysis**
  - Volatility clustering analysis
  - Regime change detection
  - Market microstructure analysis
  - Cross-market dynamics

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

## Installation

1. Clone the repository:
```bash
git clone https://github.com/wilson37wu/quantframe.git
cd quantframe
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Configure your API keys in `.env`:
```bash
cp .env.template .env
# Edit .env with your API keys
```

2. Run example strategies:
```python
from quantframe.strategy.ict_strategy import ICTStrategy
from quantframe.config.ict_config import ICTConfig

# Create strategy instance
config = ICTConfig()
strategy = ICTStrategy(config)

# Run backtest
results = strategy.backtest('BTCUSDT', '2024-01-01', '2024-03-01')
```

## Documentation

Detailed documentation is available in the `docs/` directory:
- [API Reference](docs/api_reference.md)
- [ICT Strategy Documentation](docs/ict_strategy.md)

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
