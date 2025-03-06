"""Base data source implementation with shared utilities.

This module provides common functionality for all data sources including:
- Data caching
- Standardized column mapping
- Error handling
- Rate limiting
"""
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union
import json
import pandas as pd
import logging

class BaseDataSource(ABC):
    """Abstract base class for all data sources with shared functionality.
    
    Attributes:
        cache_dir: Directory for caching data
        cache_expiry: Hours before cache expires
        logger: Logger instance for this source
    """
    
    STANDARD_COLUMNS = {
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume',
        'timestamp': 'Timestamp'
    }

    def __init__(self, cache_dir: Union[str, Path], cache_expiry: int = 24):
        """Initialize base data source.
        
        Args:
            cache_dir: Directory to store cached data
            cache_expiry: Cache expiry time in hours
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_expiry = timedelta(hours=cache_expiry)
        self.logger = logging.getLogger(self.__class__.__name__)

    def get_cache_path(self, symbol: str, timeframe: str) -> Path:
        """Get cache file path for symbol and timeframe.
        
        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            
        Returns:
            Path to cache file
        """
        return self.cache_dir / f"{symbol}_{timeframe}.parquet"

    def load_cache(self, cache_path: Path) -> Optional[pd.DataFrame]:
        """Load data from cache if valid.
        
        Args:
            cache_path: Path to cache file
            
        Returns:
            DataFrame if valid cache exists, None otherwise
        """
        if not cache_path.exists():
            return None
            
        if datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime) > self.cache_expiry:
            return None
            
        try:
            return pd.read_parquet(cache_path)
        except Exception as e:
            self.logger.warning(f"Failed to load cache: {e}")
            return None

    def save_cache(self, df: pd.DataFrame, cache_path: Path) -> None:
        """Save data to cache.
        
        Args:
            df: DataFrame to cache
            cache_path: Path to cache file
        """
        try:
            df.to_parquet(cache_path)
        except Exception as e:
            self.logger.warning(f"Failed to save cache: {e}")

    def standardize_columns(self, df: pd.DataFrame, column_map: Dict[str, str]) -> pd.DataFrame:
        """Standardize column names using mapping.
        
        Args:
            df: DataFrame to standardize
            column_map: Mapping of source columns to standard names
            
        Returns:
            DataFrame with standardized column names
        """
        df = df.copy()
        df.rename(columns=column_map, inplace=True)
        return df

    @abstractmethod
    def get_data(self, symbol: str, timeframe: str, start_date: Optional[datetime] = None,
                 end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Retrieve market data for symbol.
        
        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            
        Returns:
            DataFrame with market data
        """
        pass
