"""Base data management functionality for market data handling.

This module provides core data management capabilities for fetching,
processing, and validating market data from various sources. Key features:
- Unified interface for multiple data sources (Binance, Yahoo Finance)
- Standardized data format and column naming
- Support for various timeframes (1m to 1mo)
- Automatic data source selection based on asset type

Example:
    >>> manager = DataManager()
    >>> data = manager.load_data(
    ...     symbols=['BTC/USDT', 'ETH/USDT'],
    ...     start_date='2023-01-01',
    ...     end_date='2023-12-31',
    ...     timeframe='1d'
    ... )
    >>> print(f"Loaded {len(data)} rows of market data")
"""

import os
from typing import List, Dict, Optional, Any
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
import ta
from .sources.binance_source import BinanceSource

class DataSource:
    """Base class for all data sources.
    
    This abstract class defines the interface that all data sources must
    implement. It provides a consistent way to fetch market data regardless
    of the underlying data provider.
    
    Attributes:
        config (dict): Configuration dictionary containing source-specific
            parameters such as API keys, rate limits, etc.
            
    Example:
        >>> class MyDataSource(DataSource):
        ...     def fetch_data(self, symbol, start_time, end_time, timeframe='1d'):
        ...         # Implementation for fetching data
        ...         return data_frame
    """
    
    def __init__(self, config: dict = None):
        """Initialize data source.
        
        Args:
            config (dict, optional): Configuration dictionary with
                source-specific parameters. Defaults to None.
        """
        self.config = config or {}
        
    def fetch_data(self, symbol: str, start_time: datetime, end_time: datetime, timeframe: str = '1d') -> pd.DataFrame:
        """Fetch market data for given symbol and timeframe.
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTC/USDT', 'AAPL')
            start_time (datetime): Start time for data retrieval
            end_time (datetime): End time for data retrieval
            timeframe (str, optional): Data timeframe. Defaults to '1d'.
                Common values: '1m', '5m', '1h', '1d'
            
        Returns:
            pd.DataFrame: DataFrame containing market data with columns:
                - Open: Opening price
                - High: Highest price
                - Low: Lowest price
                - Close: Closing price
                - Volume: Trading volume
                Additional columns may be present depending on the source
                
        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError("Data sources must implement fetch_data method")
        
    def validate_timeframe(self, timeframe: str) -> bool:
        """Validate if timeframe is supported by the data source.
        
        Args:
            timeframe (str): Timeframe to validate (e.g., '1m', '1h', '1d')
            
        Returns:
            bool: True if timeframe is supported, False otherwise
            
        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError("Data sources must implement validate_timeframe method")


class DataManager:
    """Data management class for loading and preprocessing market data.
    
    This class provides a unified interface for:
    - Loading data from multiple sources
    - Standardizing data format and column names
    - Basic data validation and preprocessing
    - Automatic data source selection
    
    Attributes:
        data (Dict[str, pd.DataFrame]): Dictionary storing loaded market data
        timeframes (dict): Mapping of short timeframe codes to full names
        column_mapping (dict): Standardized column name mappings
        binance_source (BinanceSource): Binance data source instance
        
    Example:
        >>> manager = DataManager()
        >>> data = manager.load_data(
        ...     symbols=['AAPL', 'MSFT'],
        ...     start_date='2023-01-01',
        ...     end_date='2023-12-31'
        ... )
        >>> print(data.head())
    """
    
    def __init__(self):
        """Initialize data manager with default settings."""
        self.data: Dict[str, pd.DataFrame] = {}
        self.timeframes = {
            '1m': '1min',
            '5m': '5min',
            '15m': '15min',
            '30m': '30min',
            '1h': '1h',
            '4h': '4h',
            '1d': '1d',
            '1w': '1w',
            '1mo': '1mo'
        }
        self.column_mapping = {
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume',
            'adj close': 'Adj Close',
            'adj_close': 'Adj Close'
        }
        
        # Initialize data sources
        self.binance_source = BinanceSource()
    
    def get_data_source(self, symbol: str) -> str:
        """Determine the appropriate data source for a symbol.
        
        Analyzes the symbol format to determine whether it's a
        cryptocurrency pair or a traditional market symbol.
        
        Args:
            symbol (str): Trading symbol to analyze
            
        Returns:
            str: Data source identifier ('binance' or 'yahoo')
            
        Example:
            >>> manager = DataManager()
            >>> source = manager.get_data_source('BTC/USDT')
            >>> print(source)  # Output: 'binance'
            >>> source = manager.get_data_source('AAPL')
            >>> print(source)  # Output: 'yahoo'
        """
        # Check if it's a crypto pair (ends in USDT, BTC, ETH, etc.)
        crypto_suffixes = ['USDT', 'BTC', 'ETH', 'BNB', 'BUSD']
        if any(symbol.endswith(suffix) for suffix in crypto_suffixes):
            return 'binance'
        return 'yahoo'

    def standardize_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to use capitalized format.
        
        Ensures consistent column naming across different data sources
        by mapping various common column name formats to a standard set.
        
        Args:
            data (pd.DataFrame): DataFrame with market data
            
        Returns:
            pd.DataFrame: DataFrame with standardized column names
            
        Example:
            >>> df = pd.DataFrame({
            ...     'open': [100],
            ...     'adj_close': [99]
            ... })
            >>> df = manager.standardize_columns(df)
            >>> print(df.columns)  # Output: ['Open', 'Adj Close']
            
        Note:
            This method creates a copy of the input DataFrame to avoid
            modifying the original data.
        """
        # Create a copy to avoid modifying the original DataFrame
        df = data.copy()
        
        # Convert all column names to lowercase for comparison
        columns_lower = {col.lower(): col for col in df.columns}
        
        # Standardize known columns
        for old_name, new_name in self.column_mapping.items():
            if old_name in columns_lower:
                df.rename(columns={columns_lower[old_name]: new_name}, inplace=True)
        
        return df
    
    def load_data(self,
                  symbols: List[str],
                  start_date: str,
                  end_date: str,
                  timeframe: str = '1d') -> pd.DataFrame:
        """Load market data for given symbols and timeframe.
        
        Fetches market data from appropriate sources based on symbol type.
        Automatically selects between Binance for crypto and Yahoo Finance
        for traditional markets.
        
        Args:
            symbols (List[str]): List of trading symbols
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            timeframe (str, optional): Data timeframe. Defaults to '1d'.
                Must be one of the supported timeframes in self.timeframes
            
        Returns:
            pd.DataFrame: Multi-index DataFrame with market data:
                - Level 0 index: symbol
                - Level 1 index: date
                - Columns: Open, High, Low, Close, Volume
                
        Raises:
            ValueError: If no data is loaded or required columns are missing
            
        Example:
            >>> data = manager.load_data(
            ...     symbols=['ETH/USDT', 'BTC/USDT'],
            ...     start_date='2023-01-01',
            ...     end_date='2023-12-31',
            ...     timeframe='1h'
            ... )
            >>> print(f"Loaded {len(data)} rows")
            
        Note:
            Data is automatically standardized and validated before being
            returned. Missing data points are handled gracefully with
            appropriate warnings.
        """
        if timeframe not in self.timeframes:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        
        data_list = []
        for symbol in symbols:
            try:
                data_source = self.get_data_source(symbol)
                
                if data_source == 'binance':
                    # Convert dates to datetime
                    start_time = datetime.strptime(start_date, '%Y-%m-%d')
                    end_time = datetime.strptime(end_date, '%Y-%m-%d')
                    
                    # Get data from Binance
                    df = self.binance_source.fetch_data(
                        symbol=symbol,
                        start_time=start_time,
                        end_time=end_time,
                        timeframe=timeframe
                    )
                else:
                    # Download data from yfinance
                    ticker = yf.Ticker(symbol)
                    df = ticker.history(
                        start=start_date,
                        end=end_date,
                        interval=timeframe
                    )
                
                if df.empty:
                    print(f"No data available for {symbol}")
                    continue
                
                # Rename columns to match our convention
                df = df.rename(columns={
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume'
                })
                
                # Add symbol column and set multi-index
                df['symbol'] = symbol
                df['date'] = df.index
                df.set_index(['symbol', 'date'], inplace=True)
                
                data_list.append(df)
                
            except Exception as e:
                print(f"Error loading data for {symbol}: {str(e)}")
                continue
        
        if not data_list:
            raise ValueError("No data loaded for any symbols")
        
        # Combine all data
        combined_data = pd.concat(data_list)
        
        # Store data
        self.data = combined_data
        
        # Standardize column names before returning
        combined_data = self.standardize_columns(combined_data)
        
        # Ensure required columns exist
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in combined_data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        return combined_data
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess market data by adding technical indicators.
        
        Args:
            df (pd.DataFrame): DataFrame with market data
            
        Returns:
            pd.DataFrame: DataFrame with added technical indicators
        """
        processed_data_list = []
        
        for symbol in df.index.get_level_values('symbol').unique():
            try:
                # Get symbol data
                symbol_data = df.loc[symbol].copy()
                
                # Calculate RSI
                symbol_data['RSI'] = ta.momentum.rsi(symbol_data['close'], window=14)
                
                # Calculate MACD
                macd = ta.trend.macd(symbol_data['close'])
                symbol_data['MACD'] = macd
                symbol_data['Signal'] = ta.trend.macd_signal(symbol_data['close'])
                
                # Calculate Moving Averages
                symbol_data['SMA_20'] = ta.trend.sma_indicator(symbol_data['close'], window=20)
                symbol_data['SMA_50'] = ta.trend.sma_indicator(symbol_data['close'], window=50)
                
                # Add back the symbol level
                symbol_data['symbol'] = symbol
                symbol_data['date'] = symbol_data.index
                symbol_data.set_index(['symbol', 'date'], inplace=True)
                
                processed_data_list.append(symbol_data)
                
            except Exception as e:
                print(f"Error preprocessing data for {symbol}: {str(e)}")
                continue
        
        if not processed_data_list:
            raise ValueError("No data processed for any symbols")
        
        # Combine all processed data
        processed_data = pd.concat(processed_data_list)
        
        return processed_data
    
    def get_latest_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Get latest data for a symbol.
        
        Args:
            symbol (str): Trading symbol
            
        Returns:
            Latest data point or None if not available
        """
        if symbol not in self.data.index.get_level_values('symbol'):
            return None
            
        return self.data.loc[symbol].iloc[-1]
    
    def update_data(self, new_data: pd.DataFrame):
        """
        Update stored data with new data.
        
        Args:
            new_data (pd.DataFrame): New market data
        """
        self.data = pd.concat([self.data, new_data])
        self.data = self.data[~self.data.index.duplicated(keep='last')]
    
    def get_symbols(self) -> List[str]:
        """
        Get list of available symbols.
        
        Returns:
            List of symbols
        """
        return list(self.data.index.get_level_values('symbol').unique())
    
    def get_timeframe(self) -> str:
        """
        Get current timeframe.
        
        Returns:
            Current timeframe
        """
        return self.timeframe
