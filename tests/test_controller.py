"""Unit tests for QuantFrame controller."""
import unittest
from unittest.mock import patch
import io
import sys
from datetime import datetime
from quantframe.controller import QuantFrameController
from quantframe.strategy.momentum import MomentumStrategy
from quantframe.strategy.grid import GridStrategy
from quantframe.strategy.mean_reversion import MeanReversionStrategy

class TestQuantFrameController(unittest.TestCase):
    """Test cases for QuantFrame controller."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.controller = QuantFrameController()
    
    @patch('builtins.input', side_effect=['momentum'])
    def test_select_strategy_momentum(self, mock_input):
        """Test selecting momentum strategy."""
        strategy_name = self.controller.select_strategy()
        self.assertEqual(strategy_name, 'momentum')
    
    @patch('builtins.input', side_effect=['grid'])
    def test_select_strategy_grid(self, mock_input):
        """Test selecting grid strategy."""
        strategy_name = self.controller.select_strategy()
        self.assertEqual(strategy_name, 'grid')
    
    @patch('builtins.input', side_effect=['mean_reversion'])
    def test_select_strategy_mean_reversion(self, mock_input):
        """Test selecting mean reversion strategy."""
        strategy_name = self.controller.select_strategy()
        self.assertEqual(strategy_name, 'mean_reversion')
    
    @patch('builtins.input', side_effect=['invalid', 'momentum'])
    def test_select_strategy_invalid(self, mock_input):
        """Test selecting invalid strategy."""
        strategy_name = self.controller.select_strategy()
        self.assertEqual(strategy_name, 'momentum')
    
    @patch('builtins.input', side_effect=[''])
    def test_select_strategy_default(self, mock_input):
        """Test selecting default strategy."""
        strategy_name = self.controller.select_strategy()
        self.assertEqual(strategy_name, 'grid')
    
    @patch('builtins.input', side_effect=['y', '75', '25', '20', '50', '0.1', '0.02', '0.05'])
    def test_configure_momentum_strategy(self, mock_input):
        """Test configuring momentum strategy with custom parameters."""
        strategy = self.controller.configure_strategy('momentum')
        self.assertIsInstance(strategy, MomentumStrategy)
        self.assertEqual(strategy.rsi_overbought, 75)
        self.assertEqual(strategy.rsi_oversold, 25)
        self.assertEqual(strategy.sma_fast, 20)
        self.assertEqual(strategy.sma_slow, 50)
        self.assertEqual(strategy.position_size, 0.1)
        self.assertEqual(strategy.stop_loss, 0.02)
        self.assertEqual(strategy.take_profit, 0.05)
    
    @patch('builtins.input', side_effect=['n'])
    def test_configure_momentum_strategy_default(self, mock_input):
        """Test configuring momentum strategy with default parameters."""
        strategy = self.controller.configure_strategy('momentum')
        self.assertIsInstance(strategy, MomentumStrategy)
        self.assertEqual(strategy.rsi_overbought, 70)
        self.assertEqual(strategy.rsi_oversold, 30)
    
    @patch('builtins.input', side_effect=['y', '0.01', '0.05', '10'])
    def test_configure_grid_strategy(self, mock_input):
        """Test configuring grid strategy with custom parameters."""
        strategy = self.controller.configure_strategy('grid')
        self.assertIsInstance(strategy, GridStrategy)
        self.assertEqual(strategy.grid_size, 0.01)
        self.assertEqual(strategy.take_profit, 0.05)
        self.assertEqual(strategy.num_grids, 10)
    
    @patch('builtins.input', side_effect=['y', '20', '2.0', '0.02', '0.05'])
    def test_configure_mean_reversion_strategy(self, mock_input):
        """Test configuring mean reversion strategy with custom parameters."""
        strategy = self.controller.configure_strategy('mean_reversion')
        self.assertIsInstance(strategy, MeanReversionStrategy)
        self.assertEqual(strategy.ma_period, 20)
        self.assertEqual(strategy.std_dev, 2.0)
        self.assertEqual(strategy.stop_loss, 0.02)
        self.assertEqual(strategy.take_profit, 0.05)
    
    @patch('builtins.input', side_effect=['AAPL', 'MSFT', 'TSLA', ''])
    def test_get_symbols(self, mock_input):
        """Test getting symbols from user input."""
        symbols = self.controller.get_symbols()
        self.assertEqual(symbols, ['AAPL', 'MSFT', 'TSLA'])
    
    @patch('builtins.input', side_effect=['1d'])
    def test_get_timeframe(self, mock_input):
        """Test getting timeframe from user input."""
        timeframe = self.controller.get_timeframe()
        self.assertEqual(timeframe, '1d')
    
    @patch('builtins.input', side_effect=['2024-01-01'])
    def test_get_start_date(self, mock_input):
        """Test getting start date from user input."""
        start_date = self.controller.get_start_date()
        self.assertEqual(start_date, '2024-01-01')
    
    @patch('builtins.input', side_effect=['2024-12-31'])
    def test_get_end_date(self, mock_input):
        """Test getting end date from user input."""
        end_date = self.controller.get_end_date()
        self.assertEqual(end_date, '2024-12-31')

def run_tests():
    """Run all tests."""
    unittest.main(argv=[''], verbosity=2)
