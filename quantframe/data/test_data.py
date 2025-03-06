"""Test data generation for strategy backtesting.

This module provides utilities for generating realistic market data
for testing trading strategies. It simulates various market conditions
including trends, reversals, and institutional patterns.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple

class TestDataGenerator:
    """Generate test market data with realistic patterns.
    
    This class creates synthetic market data that includes:
    - Trends and reversals
    - Volume patterns
    - Fair value gaps
    - Order blocks
    - Volatility regimes
    
    The generated data is suitable for testing trading strategies,
    particularly those focused on institutional order flow.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize test data generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
            
    def generate_btc_data(self, 
                         start_date: pd.Timestamp,
                         end_date: pd.Timestamp,
                         interval: str = '1h',
                         base_price: float = 40000,
                         volatility: float = 0.02) -> pd.DataFrame:
        """Generate BTC/USDT test data.
        
        Args:
            start_date: Start date for data
            end_date: End date for data
            interval: Time interval ('1h' for hourly)
            base_price: Starting price
            volatility: Annual volatility
            
        Returns:
            DataFrame with OHLCV data and additional metrics
        """
        # Generate timestamps
        timestamps = pd.date_range(start_date, end_date, freq=interval)
        n_periods = len(timestamps)
        
        # Generate returns with trends and volatility clusters
        daily_vol = volatility / np.sqrt(252 * 24 if interval == '1h' else 252)
        returns = np.random.normal(0, daily_vol, n_periods)
        
        # Add trend regimes
        trend_periods = n_periods // 30
        trends = np.random.choice([-1, 1], trend_periods) * daily_vol * 5
        trends = np.repeat(trends, 30)[:n_periods]
        returns += trends
        
        # Calculate prices
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Generate OHLCV data
        data = pd.DataFrame(index=timestamps)
        data['close'] = prices
        
        # Generate realistic open, high, low prices
        data['open'] = data['close'].shift(1) * (1 + np.random.normal(0, daily_vol, n_periods))
        data['high'] = np.maximum(
            data['open'] * (1 + abs(np.random.normal(0, daily_vol * 2, n_periods))),
            data['close'] * (1 + abs(np.random.normal(0, daily_vol * 2, n_periods)))
        )
        data['low'] = np.minimum(
            data['open'] * (1 - abs(np.random.normal(0, daily_vol * 2, n_periods))),
            data['close'] * (1 - abs(np.random.normal(0, daily_vol * 2, n_periods)))
        )
        
        # Generate volume with institutional patterns
        base_volume = 1000000  # Base volume in USDT
        data['volume'] = base_volume * (1 + np.random.normal(0, 0.5, n_periods))
        
        # Add volume spikes for institutional activity
        inst_periods = np.random.choice(n_periods, n_periods // 20)
        data.loc[data.index[inst_periods], 'volume'] *= np.random.uniform(2, 5, len(inst_periods))
        
        # Generate fair value gaps
        fvg_periods = np.random.choice(n_periods - 2, n_periods // 50)
        for i in fvg_periods:
            if np.random.random() > 0.5:  # Bullish FVG
                data.iloc[i+1]['low'] = data.iloc[i]['high'] * (1 + np.random.uniform(0.003, 0.01))
            else:  # Bearish FVG
                data.iloc[i+1]['high'] = data.iloc[i]['low'] * (1 - np.random.uniform(0.003, 0.01))
        
        # Calculate additional metrics
        data['typical_price'] = (data['high'] + data['low'] + data['close']) / 3
        data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        data['volatility'] = data['log_returns'].rolling(window=21).std() * np.sqrt(252 * 24 if interval == '1h' else 252)
        
        # Volume metrics
        data['volume_ma'] = data['volume'].rolling(window=24 if interval == '1h' else 20).mean()
        data['volume_std'] = data['volume'].rolling(window=24 if interval == '1h' else 20).std()
        data['relative_volume'] = data['volume'] / data['volume_ma']
        
        # Price metrics
        data['weighted_price'] = (data['typical_price'] * data['volume']).rolling(window=24 if interval == '1h' else 20).sum() / \
                               data['volume'].rolling(window=24 if interval == '1h' else 20).sum()
        
        # Add quote asset volume and trades
        data['quote_volume'] = data['volume'] * data['typical_price']
        data['trades'] = (data['volume'] / 1000).round()  # Approximate number of trades
        
        return data.fillna(method='bfill')
