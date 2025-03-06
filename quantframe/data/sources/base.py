"""Base classes for data sources.

This module provides the foundational classes for implementing market data sources
in the quantframe framework. It includes:
- Abstract base class for data source implementations
- Data validation utilities
- Caching mechanism for efficient data retrieval

The DataSource class defines the core interface that all data sources must implement,
ensuring consistent data handling and validation across different data providers.

Example:
    >>> class YahooFinanceSource(DataSource):
    ...     def fetch_data(self, symbol, start_date, end_date, interval='1d'):
    ...         # Fetch data from Yahoo Finance
    ...         pass
    ...     def validate_data(self, df):
    ...         return DataValidator().check_missing_values(df)
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
import pandas as pd
from datetime import datetime
from pathlib import Path

class DataValidator:
    """Data validation utilities for market data.
    
    Provides a set of validation methods to ensure data quality and consistency
    across different data sources. Checks include:
    - Missing value detection
    - Duplicate data points
    - Time series monotonicity
    - Price validity
    
    Example:
        >>> validator = DataValidator()
        >>> df = pd.DataFrame({'close': [100.0, 101.0, None]})
        >>> validator.check_missing_values(df)  # Returns False
    """
    
    def check_missing_values(self, df: pd.DataFrame) -> bool:
        """Check for missing values in the DataFrame.
        
        Args:
            df (pd.DataFrame): Market data DataFrame to validate
            
        Returns:
            bool: True if no missing values are found, False otherwise
            
        Note:
            This check applies to all columns in the DataFrame.
            For OHLCV data, all price and volume fields should be present.
        """
        return not df.isnull().any().any()
    
    def check_duplicates(self, df: pd.DataFrame) -> bool:
        """Check for duplicate timestamps in the index.
        
        Args:
            df (pd.DataFrame): Market data DataFrame to validate
            
        Returns:
            bool: True if no duplicate timestamps exist, False otherwise
            
        Note:
            Duplicate timestamps can cause issues with calculations
            and should be resolved before using the data.
        """
        return not df.index.duplicated().any()
    
    def check_monotonic(self, df: pd.DataFrame) -> bool:
        """Check if the time index is monotonically increasing.
        
        Args:
            df (pd.DataFrame): Market data DataFrame to validate
            
        Returns:
            bool: True if timestamps are strictly increasing, False otherwise
            
        Note:
            Non-monotonic timestamps can cause issues with time-based
            calculations and should be fixed before processing.
        """
        return df.index.is_monotonic_increasing
    
    def check_price_validity(self, df: pd.DataFrame) -> bool:
        """Check if all price values are positive.
        
        Args:
            df (pd.DataFrame): Market data DataFrame to validate
            
        Returns:
            bool: True if all prices are positive, False otherwise
            
        Note:
            Checks 'open', 'high', 'low', 'close' columns if they exist.
            This is crucial for assets that cannot have negative prices.
        """
        price_cols = ['open', 'high', 'low', 'close']
        return all(df[col].gt(0).all() for col in price_cols if col in df.columns)

class DataSource(ABC):
    """Abstract base class for all market data sources.
    
    This class provides a standardized interface for fetching and managing
    market data from various sources. Features include:
    - Configurable data fetching
    - Data validation
    - Automatic caching
    - Timezone handling
    
    Attributes:
        config (Dict[str, Any]): Configuration parameters for the data source
        cache_dir (Path): Directory for caching downloaded data
        
    Example:
        >>> config = {'cache_dir': 'data/cache', 'api_key': 'your_key'}
        >>> class MyDataSource(DataSource):
        ...     def fetch_data(self, symbol, start, end, interval):
        ...         # Implementation
        ...         pass
        ...     def validate_data(self, df):
        ...         return True
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the data source.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary containing:
                - cache_dir: Path to cache directory (default: 'data/cache')
                - Other source-specific configuration parameters
        """
        self.config = config
        self.cache_dir = Path(config.get('cache_dir', 'data/cache'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    @abstractmethod
    def fetch_data(self, 
                  symbol: str, 
                  start_date: datetime, 
                  end_date: datetime,
                  interval: str = '1d') -> pd.DataFrame:
        """Fetch market data from the source.
        
        Args:
            symbol (str): Trading symbol to fetch
            start_date (datetime): Start of the data range
            end_date (datetime): End of the data range
            interval (str, optional): Data timeframe. Defaults to '1d'
                Common values: '1m', '5m', '1h', '1d', '1w'
            
        Returns:
            pd.DataFrame: OHLCV data with datetime index
            
        Note:
            Implement this method in derived classes to fetch data
            from specific sources (e.g., Yahoo Finance, Binance).
        """
        pass
    
    @abstractmethod
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate the quality of fetched data.
        
        Args:
            df (pd.DataFrame): Market data to validate
            
        Returns:
            bool: True if data passes all quality checks
            
        Note:
            Implement source-specific validation rules in derived classes.
            Use DataValidator methods as needed.
        """
        pass
    
    def get_data(self, 
                 symbol: str, 
                 start_date: datetime, 
                 end_date: datetime,
                 interval: str = '1d',
                 use_cache: bool = True) -> pd.DataFrame:
        """Get market data with automatic caching.
        
        This method handles:
        1. Cache checking and retrieval
        2. Data fetching if needed
        3. Timezone normalization
        4. Data validation
        5. Cache updating
        
        Args:
            symbol (str): Trading symbol to fetch
            start_date (datetime): Start of the data range
            end_date (datetime): End of the data range
            interval (str, optional): Data timeframe. Defaults to '1d'
            use_cache (bool, optional): Whether to use cached data. Defaults to True
            
        Returns:
            pd.DataFrame: Validated OHLCV data with UTC timezone index
            
        Raises:
            ValueError: If data validation fails
            
        Example:
            >>> source = MyDataSource(config={'cache_dir': 'cache'})
            >>> df = source.get_data('AAPL', 
            ...                     datetime(2023, 1, 1),
            ...                     datetime(2023, 12, 31))
        """
        cache_file = self.cache_dir / f"{symbol}_{interval}.parquet"
        
        if use_cache and cache_file.exists():
            df = pd.read_parquet(cache_file)
            mask = (df.index >= pd.Timestamp(start_date)) & (df.index <= pd.Timestamp(end_date))
            cached_df = df[mask]
            
            if not cached_df.empty:
                return cached_df
        
        df = self.fetch_data(symbol, start_date, end_date, interval)
        
        # Convert dates to timezone-naive UTC
        start_ts = pd.Timestamp(start_date).tz_localize('UTC')
        end_ts = pd.Timestamp(end_date).tz_localize('UTC')
        
        # Convert index to UTC and filter
        df.index = df.index.tz_convert('UTC')
        mask = (df.index >= start_ts) & (df.index <= end_ts)
        df = df[mask].copy()
        
        if self.validate_data(df):
            if use_cache:
                df.to_parquet(cache_file)
            return df
        else:
            raise ValueError(f"Data validation failed for {symbol}")
    
    def clear_cache(self, symbol: Optional[str] = None) -> None:
        """Clear cached market data files.
        
        Args:
            symbol (Optional[str]): If provided, only clear cache for this symbol.
                If None, clear all cached data.
                
        Example:
            >>> source = MyDataSource(config={'cache_dir': 'cache'})
            >>> source.clear_cache('AAPL')  # Clear only AAPL data
            >>> source.clear_cache()  # Clear all cached data
        """
        if symbol:
            for file in self.cache_dir.glob(f"{symbol}_*.parquet"):
                file.unlink()
        else:
            for file in self.cache_dir.glob("*.parquet"):
                file.unlink()
