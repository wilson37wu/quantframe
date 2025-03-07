import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

def get_btc_data():
    # Initialize Binance exchange
    exchange = ccxt.binance()
    
    # Calculate timestamps
    end_time = datetime.now()
    start_time = end_time - timedelta(days=365)
    
    # Fetch OHLCV data
    ohlcv = exchange.fetch_ohlcv(
        symbol='BTC/USDT',
        timeframe='1d',
        since=int(start_time.timestamp() * 1000),
        limit=365
    )
    
    # Convert to DataFrame
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.rename(columns={
        'timestamp': 'date',
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })
    return df

def prepare_data(df):
    df = df.copy()
    
    # Calculate log returns
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Calculate moving averages
    df['price_sma'] = df['Close'].rolling(window=20).mean()
    df['volume_sma'] = df['Volume'].rolling(window=20).mean()
    
    # Calculate volatility
    df['volatility'] = df['log_return'].rolling(window=20).std()
    
    return df.dropna()

def rolling_regression_analysis(df):
    # Define features and target
    features = ['volume_sma', 'price_sma', 'volatility']
    target = 'log_return'
    
    # Create lagged features for rolling regression
    for feature in features:
        df[f'{feature}_lag1'] = df[feature].shift(1)
    
    # Drop NA values after creating lags
    df = df.dropna()
    
    # Prepare X (features) and y (target)
    X = df[[f'{feature}_lag1' for feature in features]]
    y = df[target]
    
    # Initialize arrays for storing results
    window_size = 30
    n_samples = len(df) - window_size + 1
    r_squared = np.zeros(n_samples)
    coef = np.zeros((n_samples, len(features)))
    
    # Perform rolling regression
    model = LinearRegression()
    for i in range(n_samples):
        X_window = X.iloc[i:i+window_size]
        y_window = y.iloc[i:i+window_size]
        
        model.fit(X_window, y_window)
        r_squared[i] = model.score(X_window, y_window)
        coef[i] = model.coef_
    
    # Create results DataFrame
    results = pd.DataFrame({
        'date': df['date'].iloc[window_size-1:],
        'r.squared': r_squared
    })
    
    for i, feature in enumerate(features):
        results[f'{feature}_coef'] = coef[:, i]
    
    return results

def plot_results(df, rolling_reg):
    # Create figure with secondary y-axis
    fig = go.Figure()

    # Add BTC price
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['Close'],
                  name='BTC Price',
                  line=dict(color='blue'))
    )

    # Add R-squared on secondary y-axis
    fig.add_trace(
        go.Scatter(x=rolling_reg['date'], y=rolling_reg['r.squared'],
                  name='R-squared',
                  yaxis='y2',
                  line=dict(color='red'))
    )

    # Update layout
    fig.update_layout(
        title='BTC-USD Price and Rolling Regression R-squared',
        xaxis_title='Date',
        yaxis_title='BTC Price (USD)',
        yaxis2=dict(
            title='R-squared',
            overlaying='y',
            side='right'
        ),
        showlegend=True
    )

    fig.show()

def main():
    print("Fetching BTC/USDT data...")
    df = get_btc_data()
    
    print("Preparing data and calculating features...")
    df = prepare_data(df)
    
    print("Performing rolling regression analysis...")
    rolling_reg = rolling_regression_analysis(df)
    
    print("Plotting results...")
    plot_results(df, rolling_reg)
    
    # Print latest regression statistics
    latest_stats = rolling_reg.iloc[-1]
    print("\nLatest Rolling Regression Statistics:")
    print(f"R-squared: {latest_stats['r.squared']:.4f}")
    print("\nFeature Coefficients:")
    print(f"Volume SMA: {latest_stats['volume_sma_coef']:.6f}")
    print(f"Price SMA: {latest_stats['price_sma_coef']:.6f}")
    print(f"Volatility: {latest_stats['volatility_coef']:.6f}")

if __name__ == "__main__":
    main()
