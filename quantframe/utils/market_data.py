"""Utilities for market data operations"""
import pandas as pd
import yfinance as yf

def get_sp500_symbols():
    """Get list of current S&P 500 symbols using Wikipedia data."""
    try:
        # Read S&P 500 table from Wikipedia
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        sp500_table = tables[0]
        symbols = sp500_table['Symbol'].tolist()
        
        # Clean symbols (remove special characters, etc.)
        symbols = [symbol.replace('.', '-') for symbol in symbols]
        return symbols
    except Exception as e:
        print(f"Error fetching S&P 500 symbols: {e}")
        return []

def filter_liquid_stocks(symbols, min_volume=1000000, min_price=10):
    """Filter stocks based on liquidity criteria."""
    liquid_symbols = []
    
    for symbol in symbols:
        try:
            # Get recent data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1mo")
            
            # Check average volume and price
            avg_volume = hist['Volume'].mean()
            avg_price = hist['Close'].mean()
            
            if avg_volume >= min_volume and avg_price >= min_price:
                liquid_symbols.append(symbol)
        except Exception as e:
            print(f"Error checking liquidity for {symbol}: {e}")
            continue
    
    return liquid_symbols
