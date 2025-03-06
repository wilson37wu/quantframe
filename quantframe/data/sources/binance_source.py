"""Binance data source implementation.

This module provides access to Binance market data with built-in caching
and standardized column naming.
"""
from datetime import datetime
from typing import Optional, Dict
import pandas as pd
from binance.client import Client

from .base_source import BaseDataSource

class BinanceSource(BaseDataSource):
    """Binance data source with caching and standardization.
    
    Attributes:
        api_key: Binance API key
        api_secret: Binance API secret
        client: Binance API client instance
    """
    
    TIMEFRAME_MAP = {
        '1m': Client.KLINE_INTERVAL_1MINUTE,
        '5m': Client.KLINE_INTERVAL_5MINUTE,
        '15m': Client.KLINE_INTERVAL_15MINUTE,
        '1h': Client.KLINE_INTERVAL_1HOUR,
        '4h': Client.KLINE_INTERVAL_4HOUR,
        '1d': Client.KLINE_INTERVAL_1DAY,
    }
    
    COLUMN_MAP = {
        0: 'timestamp',
        1: 'open',
        2: 'high',
        3: 'low',
        4: 'close',
        5: 'volume'
    }

    def __init__(self, api_key: str, api_secret: str, cache_dir: str):
        """Initialize Binance data source.
        
        Args:
            api_key: Binance API key
            api_secret: Binance API secret
            cache_dir: Directory for caching data
        """
        super().__init__(cache_dir)
        self.api_key = api_key
        self.api_secret = api_secret
        self.client = Client(api_key, api_secret)

    def get_data(self, symbol: str, timeframe: str,
                 start_date: Optional[datetime] = None,
                 end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Get market data from Binance with caching.
        
        Args:
            symbol: Trading symbol (e.g. 'BTCUSDT')
            timeframe: Data timeframe (e.g. '1m', '1h')
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
            klines = self.client.get_historical_klines(
                symbol,
                self.TIMEFRAME_MAP[timeframe],
                start_date.strftime("%d %b %Y %H:%M:%S") if start_date else None,
                end_date.strftime("%d %b %Y %H:%M:%S") if end_date else None
            )
            
            df = pd.DataFrame(klines)
            df = self.standardize_columns(df, self.COLUMN_MAP)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
            self.save_cache(df, cache_path)
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to fetch data from Binance: {e}")
            raise

    def get_available_pairs(self) -> Dict[str, str]:
        """Get available trading pairs.
        
        Returns:
            Dictionary of symbol: base_asset pairs
        """
        try:
            info = self.client.get_exchange_info()
            return {s['symbol']: s['baseAsset'] for s in info['symbols']}
        except Exception as e:
            self.logger.error(f"Failed to fetch pairs from Binance: {e}")
            raise
