"""Unit tests for momentum strategy."""
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from quantframe.strategy.momentum import MomentumStrategy
from quantframe.data.base import DataManager

class TestMomentumStrategy(unittest.TestCase):
    """Test cases for momentum strategy."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.strategy = MomentumStrategy()
        self.data_manager = DataManager()
        
        # Create sample data with unique timestamps for each symbol
        self.dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        self.symbols = ['AAPL', 'MSFT', 'TSLA']
        
        # Create sample technical indicators
        data_list = []
        for symbol in self.symbols:
            symbol_data = pd.DataFrame(index=self.dates)
            symbol_data['symbol'] = symbol
            symbol_data['close'] = np.random.normal(100, 10, len(self.dates))
            symbol_data['RSI'] = np.random.normal(50, 15, len(self.dates))
            symbol_data['MACD'] = np.random.normal(0, 1, len(self.dates))
            symbol_data['Signal'] = np.random.normal(0, 1, len(self.dates))
            symbol_data['SMA_20'] = symbol_data['close'].rolling(20).mean()
            symbol_data['SMA_50'] = symbol_data['close'].rolling(50).mean()
            # Create multi-index with symbol and date
            symbol_data['date'] = symbol_data.index
            symbol_data.set_index(['symbol', 'date'], inplace=True)
            data_list.append(symbol_data)
        
        self.test_data = pd.concat(data_list)
    
    def test_initialization(self):
        """Test strategy initialization."""
        self.assertEqual(self.strategy.rsi_period, 14)
        self.assertEqual(self.strategy.rsi_overbought, 70)
        self.assertEqual(self.strategy.rsi_oversold, 30)
        self.assertEqual(self.strategy.position_size, 0.1)
        self.assertEqual(len(self.strategy.positions), 0)
    
    def test_generate_signals(self):
        """Test signal generation."""
        signals = self.strategy.generate_signals(self.test_data)
        
        # Check signal properties
        self.assertEqual(len(signals), len(self.test_data))
        self.assertTrue(all(s in [-1, 0, 1] for s in signals['signal'].unique()))
        
        # Check that signals are generated for each symbol
        for symbol in self.symbols:
            symbol_signals = signals.loc[symbol]
            self.assertGreater(len(symbol_signals), 0)
    
    def test_position_management(self):
        """Test position opening and closing."""
        symbol = 'AAPL'
        timestamp = pd.Timestamp('2023-01-01')
        price = 100.0
        size = 1.0
        
        # Test opening position
        self.strategy.open_position(symbol, timestamp, price, size)
        self.assertIn(symbol, self.strategy.positions)
        self.assertEqual(self.strategy.positions[symbol].size, size)
        
        # Test closing position
        self.strategy.close_position(symbol, timestamp + pd.Timedelta(days=1), price * 1.1)
        self.assertNotIn(symbol, self.strategy.positions)
    
    def test_update_positions(self):
        """Test position updates with new data."""
        # Create sample data for a single timestamp
        test_date = self.test_data.index.get_level_values('date')[100]  # Use a date with sufficient history
        current_data = self.test_data.xs(test_date, level='date', drop_level=False)
        
        # Update positions
        self.strategy.update(test_date, current_data)
        
        # Verify portfolio state
        portfolio_state = self.strategy.get_portfolio_state()
        self.assertIsInstance(portfolio_state, dict)
        self.assertIn('positions', portfolio_state)
        
        # Check stop loss and take profit
        for symbol in self.symbols:
            if symbol in self.strategy.positions:
                stop_loss = self.strategy.get_stop_loss(symbol)
                take_profit = self.strategy.get_take_profit(symbol)
                self.assertIsNotNone(stop_loss)
                self.assertIsNotNone(take_profit)
                self.assertLess(stop_loss, take_profit)

class TestMomentumStrategyIntegration(unittest.TestCase):
    """Integration tests for momentum strategy with real market data."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.strategy = MomentumStrategy()
        self.data_manager = DataManager()
    
    def test_real_data_backtest(self):
        """Test strategy with real market data."""
        # Test with recent data to avoid yfinance limitations
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
        
        # Test with daily data only since hourly data has limitations
        symbols = ['AAPL']
        timeframe = '1d'
        
        try:
            # Load data
            data = self.data_manager.load_data(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                timeframe=timeframe
            )
            
            self.assertFalse(data.empty, "Data should not be empty")
            
            # Create multi-index with symbol and date
            data['date'] = data.index
            data.set_index(['symbol', 'date'], inplace=True)
            
            # Preprocess data
            data = self.data_manager.preprocess_data(data)
            
            # Generate signals
            signals = self.strategy.generate_signals(data)
            
            # Basic assertions
            self.assertEqual(len(signals), len(data))
            self.assertTrue(all(s in [-1, 0, 1] for s in signals['signal'].unique()))
            
            # Test strategy update
            for timestamp in data.index.get_level_values('date')[-10:]:  # Test last 10 days
                current_data = data.xs(timestamp, level='date', drop_level=False)
                self.strategy.update(timestamp, current_data)
                
                # Check portfolio state
                portfolio_state = self.strategy.get_portfolio_state()
                self.assertIsInstance(portfolio_state, dict)
                self.assertIn('positions', portfolio_state)
                
        except Exception as e:
            self.fail(f"Failed with {symbols}, {timeframe}: {str(e)}")

def run_tests():
    """Run all tests."""
    unittest.main(argv=[''], verbosity=2)
