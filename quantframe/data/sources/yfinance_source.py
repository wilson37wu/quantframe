"""YFinance data source implementation.

This module provides access to Yahoo Finance market data with built-in caching
and standardized column naming.
"""
from datetime import datetime
from typing import Optional, Dict
import pandas as pd
import yfinance as yf

from .base_source import BaseDataSource

class YFinanceSource(BaseDataSource):
    """Yahoo Finance data source with caching and standardization.
    
    Attributes:
        adjust_ohlc: Whether to use adjusted OHLC values
    """
    
    TIMEFRAME_MAP = {
        '1m': '1m',
        '2m': '2m',
        '5m': '5m',
        '15m': '15m',
        '30m': '30m',
        '60m': '60m',
        '90m': '90m',
        '1h': '1h',
        '1d': '1d',
        '5d': '5d',
        '1wk': '1wk',
        '1mo': '1mo',
        '3mo': '3mo'
    }
    
    COLUMN_MAP = {
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume',
        'Datetime': 'timestamp'
    }

    def __init__(self, cache_dir: str, adjust_ohlc: bool = True):
        """Initialize Yahoo Finance data source.
        
        Args:
            cache_dir: Directory for caching data
            adjust_ohlc: Whether to use adjusted OHLC values
        """
        super().__init__(cache_dir)
        self.adjust_ohlc = adjust_ohlc

    def get_data(self, symbol: str, timeframe: str,
                 start_date: Optional[datetime] = None,
                 end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Get market data from Yahoo Finance with caching.
        
        Args:
            symbol: Trading symbol (e.g. 'AAPL')
            timeframe: Data timeframe (e.g. '1d', '1h')
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            
        Returns:
            DataFrame with market data
        """
        cache_path = self.get_cache_path(symbol, timeframe)
        cached_data = self.load_cache(cache_path)
        
        if cached_data is not None:
            return cached_data
            
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                interval=self.TIMEFRAME_MAP[timeframe],
                start=start_date,
                end=end_date,
                auto_adjust=self.adjust_ohlc
            )
            
            df = self.standardize_columns(df, self.COLUMN_MAP)
            df = df.reset_index()
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
            
            self.save_cache(df, cache_path)
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to fetch data from Yahoo Finance: {e}")
            raise

    def get_available_symbols(self) -> Dict[str, str]:
        """Get information about available symbols.
        
        Returns:
            Dictionary of symbol: company name pairs
        """
        try:
            # This is a simplified implementation
            # In practice, you might want to use a more comprehensive
            # source of symbols or maintain a local database
            return {'AAPL': 'Apple Inc.', 'MSFT': 'Microsoft Corporation'}
        except Exception as e:
            self.logger.error(f"Failed to fetch symbols from Yahoo Finance: {e}")
            raise
