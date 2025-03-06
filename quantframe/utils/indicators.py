"""Technical analysis indicators"""
import pandas as pd
import numpy as np
from typing import Union

def calculate_zscore(series: pd.Series, window: int) -> pd.Series:
    """Calculate rolling z-score of a series"""
    # Convert to simple Series if MultiIndex
    if isinstance(series.index, pd.MultiIndex):
        series = series.reset_index(level=0, drop=True)
    
    mean = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    return (series - mean) / std

def calculate_rsi(series: pd.Series, window: int) -> pd.Series:
    """Calculate Relative Strength Index"""
    # Convert to simple Series if MultiIndex
    if isinstance(series.index, pd.MultiIndex):
        series = series.reset_index(level=0, drop=True)
    
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_atr(data: pd.DataFrame, window: int) -> pd.Series:
    """Calculate Average True Range"""
    # Convert to simple Series if MultiIndex
    if isinstance(data.columns, pd.MultiIndex):
        data = data.copy()
        data.columns = data.columns.get_level_values(0)
    
    high = data['high']
    low = data['low']
    close = data['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=window).mean()

def calculate_bollinger_bands(series: pd.Series, window: int, num_std: float = 2.0) -> pd.DataFrame:
    """Calculate Bollinger Bands"""
    # Convert to simple Series if MultiIndex
    if isinstance(series.index, pd.MultiIndex):
        series = series.reset_index(level=0, drop=True)
    
    middle = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    
    return pd.DataFrame({
        'middle': middle,
        'upper': upper,
        'lower': lower
    })

def calculate_macd(series: pd.Series, 
                  fast_period: int = 12, 
                  slow_period: int = 26, 
                  signal_period: int = 9) -> pd.DataFrame:
    """Calculate MACD (Moving Average Convergence Divergence)"""
    # Convert to simple Series if MultiIndex
    if isinstance(series.index, pd.MultiIndex):
        series = series.reset_index(level=0, drop=True)
    
    fast_ema = series.ewm(span=fast_period, adjust=False).mean()
    slow_ema = series.ewm(span=slow_period, adjust=False).mean()
    
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line
    
    return pd.DataFrame({
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram
    })

def calculate_volatility(returns: pd.Series, window: int, annualize: bool = True) -> pd.Series:
    """Calculate rolling volatility"""
    # Convert to simple Series if MultiIndex
    if isinstance(returns.index, pd.MultiIndex):
        returns = returns.reset_index(level=0, drop=True)
    
    vol = returns.rolling(window=window).std()
    if annualize:
        vol = vol * np.sqrt(252)  # Annualize assuming daily data
    return vol

def calculate_drawdown(series: pd.Series) -> pd.DataFrame:
    """Calculate drawdown series"""
    # Convert to simple Series if MultiIndex
    if isinstance(series.index, pd.MultiIndex):
        series = series.reset_index(level=0, drop=True)
    
    rolling_max = series.expanding().max()
    drawdown = (series - rolling_max) / rolling_max
    
    return pd.DataFrame({
        'drawdown': drawdown,
        'high_watermark': rolling_max
    })

def calculate_momentum(series: pd.Series, window: int) -> pd.Series:
    """Calculate momentum indicator"""
    # Convert to simple Series if MultiIndex
    if isinstance(series.index, pd.MultiIndex):
        series = series.reset_index(level=0, drop=True)
    
    return series / series.shift(window) - 1
