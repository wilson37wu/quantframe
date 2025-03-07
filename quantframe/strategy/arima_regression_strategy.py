import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from typing import Optional, Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ARIMARegression:
    """
    A trading strategy that combines ARIMA time series forecasting with rolling regression analysis.
    """
    
    def __init__(self, 
                 symbol: str = 'BTC/USDT',
                 timeframe: str = '1d',
                 lookback_days: int = 365,
                 sma_window: int = 20,
                 regression_window: int = 30,
                 arima_order: Tuple[int, int, int] = (1, 1, 1),
                 bb_window: int = 20,
                 bb_std: float = 2.0,
                 macd_fast: int = 12,
                 macd_slow: int = 26,
                 macd_signal: int = 9,
                 stoch_window: int = 14,
                 stoch_smooth: int = 3,
                 atr_window: int = 14):
        """
        Initialize the strategy.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            timeframe: Data timeframe ('1d', '4h', etc.)
            lookback_days: Number of days of historical data to analyze
            sma_window: Window size for moving averages
            regression_window: Window size for rolling regression
            arima_order: ARIMA model order (p, d, q)
            bb_window: Bollinger Bands window
            bb_std: Bollinger Bands standard deviation
            macd_fast: MACD fast period
            macd_slow: MACD slow period
            macd_signal: MACD signal period
            stoch_window: Stochastic Oscillator window
            stoch_smooth: Stochastic Oscillator smoothing
            atr_window: Average True Range window
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.lookback_days = lookback_days
        self.sma_window = sma_window
        self.regression_window = regression_window
        self.arima_order = arima_order
        self.bb_window = bb_window
        self.bb_std = bb_std
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.stoch_window = stoch_window
        self.stoch_smooth = stoch_smooth
        self.atr_window = atr_window
        
        # Initialize exchange
        self.exchange = ccxt.binance()
    
    def fetch_data(self) -> pd.DataFrame:
        """Fetch historical data from the exchange."""
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=self.lookback_days)
            
            ohlcv = self.exchange.fetch_ohlcv(
                symbol=self.symbol,
                timeframe=self.timeframe,
                since=int(start_time.timestamp() * 1000),
                limit=self.lookback_days
            )
            
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
            
        except Exception as e:
            logger.error(f"Error fetching data: {str(e)}")
            raise
    
    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical features for analysis."""
        df = df.copy()
        
        # Basic features
        df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
        df['price_sma'] = df['Close'].rolling(window=self.sma_window).mean()
        df['volume_sma'] = df['Volume'].rolling(window=self.sma_window).mean()
        df['volatility'] = df['log_return'].rolling(window=self.sma_window).std()
        df['momentum'] = df['Close'] / df['Close'].shift(self.sma_window) - 1
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.sma_window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.sma_window).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_middle'] = df['Close'].rolling(window=self.bb_window).mean()
        bb_std = df['Close'].rolling(window=self.bb_window).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * self.bb_std)
        df['bb_lower'] = df['bb_middle'] - (bb_std * self.bb_std)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # MACD
        exp1 = df['Close'].ewm(span=self.macd_fast, adjust=False).mean()
        exp2 = df['Close'].ewm(span=self.macd_slow, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=self.macd_signal, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        df['macd_divergence'] = df['macd_hist'].diff()
        
        # Stochastic Oscillator
        df['stoch_k'] = ((df['Close'] - df['Low'].rolling(window=self.stoch_window).min()) /
                        (df['High'].rolling(window=self.stoch_window).max() - 
                         df['Low'].rolling(window=self.stoch_window).min())) * 100
        df['stoch_d'] = df['stoch_k'].rolling(window=self.stoch_smooth).mean()
        df['stoch_trend'] = df['stoch_k'] - df['stoch_d']
        
        # Average True Range (ATR)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['atr'] = true_range.rolling(window=self.atr_window).mean()
        df['atr_ratio'] = df['atr'] / df['Close']
        
        # Additional trend indicators
        df['ema_fast'] = df['Close'].ewm(span=self.macd_fast, adjust=False).mean()
        df['ema_slow'] = df['Close'].ewm(span=self.macd_slow, adjust=False).mean()
        df['trend_strength'] = (df['ema_fast'] - df['ema_slow']) / df['Close']
        
        # Volume trend
        df['volume_trend'] = df['Volume'] / df['Volume'].rolling(window=self.sma_window).mean()
        
        # Price patterns
        df['higher_high'] = df['High'] > df['High'].shift(1)
        df['lower_low'] = df['Low'] < df['Low'].shift(1)
        df['trend_consistency'] = df['higher_high'].rolling(window=self.sma_window).mean() - \
                                df['lower_low'].rolling(window=self.sma_window).mean()
        
        return df.dropna()
    
    def perform_rolling_regression(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform rolling regression analysis."""
        features = [
            'volume_sma', 'price_sma', 'volatility', 'momentum', 'rsi',
            'bb_width', 'bb_position', 'macd', 'macd_divergence',
            'stoch_trend', 'atr_ratio', 'trend_strength',
            'volume_trend', 'trend_consistency'
        ]
        target = 'log_return'
        
        # Create lagged features
        for feature in features:
            df[f'{feature}_lag1'] = df[feature].shift(1)
        
        df = df.dropna()
        
        # Prepare X and y
        X = df[[f'{feature}_lag1' for feature in features]]
        y = df[target]
        
        # Initialize arrays for results
        n_samples = len(df) - self.regression_window + 1
        r_squared = np.zeros(n_samples)
        coef = np.zeros((n_samples, len(features)))
        
        # Perform rolling regression
        model = LinearRegression()
        for i in range(n_samples):
            X_window = X.iloc[i:i+self.regression_window]
            y_window = y.iloc[i:i+self.regression_window]
            
            model.fit(X_window, y_window)
            r_squared[i] = model.score(X_window, y_window)
            coef[i] = model.coef_
        
        # Create results DataFrame
        results = pd.DataFrame({
            'date': df['date'].iloc[self.regression_window-1:],
            'r.squared': r_squared
        })
        
        for i, feature in enumerate(features):
            results[f'{feature}_coef'] = coef[:, i]
        
        return results
    
    def fit_arima(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Fit ARIMA model and generate predictions."""
        try:
            # Fit ARIMA model
            model = ARIMA(df['Close'], order=self.arima_order)
            results = model.fit()
            
            # Generate predictions
            forecast = results.get_forecast(steps=5)
            
            # Create forecast DataFrame
            forecast_df = pd.DataFrame({
                'date': pd.date_range(start=df['date'].iloc[-1], periods=6)[1:],
                'forecast': forecast.predicted_mean,
                'lower_ci': forecast.conf_int()['lower Close'],
                'upper_ci': forecast.conf_int()['upper Close']
            })
            
            # Get model metrics
            metrics = {
                'aic': results.aic,
                'bic': results.bic,
                'mse': ((results.resid ** 2).mean()),
                'mae': abs(results.resid).mean()
            }
            
            return forecast_df, metrics
            
        except Exception as e:
            logger.error(f"Error fitting ARIMA model: {str(e)}")
            raise
    
    def plot_analysis(self, df: pd.DataFrame, rolling_reg: pd.DataFrame, 
                     forecast_df: pd.DataFrame, metrics: Dict) -> None:
        """Plot analysis results."""
        # Create figure with subplots
        fig = make_subplots(rows=4, cols=1,
                          subplot_titles=('Price, BB, and ARIMA Forecast',
                                        'Technical Indicators',
                                        'Rolling Regression R-squared',
                                        'Feature Coefficients'),
                          vertical_spacing=0.1,
                          row_heights=[0.4, 0.2, 0.2, 0.2])

        # Plot 1: Price, BB, and Forecast
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['Close'],
                      name='Price',
                      line=dict(color='blue')),
            row=1, col=1
        )
        
        # Add Bollinger Bands
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['bb_upper'],
                      name='BB Upper',
                      line=dict(color='gray', dash='dash'),
                      showlegend=True),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['bb_lower'],
                      name='BB Lower',
                      line=dict(color='gray', dash='dash'),
                      fill='tonexty',
                      showlegend=True),
            row=1, col=1
        )
        
        # Add ARIMA forecast
        fig.add_trace(
            go.Scatter(x=forecast_df['date'], y=forecast_df['forecast'],
                      name='ARIMA Forecast',
                      line=dict(color='red', dash='dash')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=forecast_df['date'], y=forecast_df['upper_ci'],
                      name='Forecast CI',
                      line=dict(color='rgba(255,0,0,0.2)', dash='dot'),
                      showlegend=False),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=forecast_df['date'], y=forecast_df['lower_ci'],
                      line=dict(color='rgba(255,0,0,0.2)', dash='dot'),
                      fill='tonexty',
                      showlegend=False),
            row=1, col=1
        )

        # Plot 2: Technical Indicators
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['macd'],
                      name='MACD',
                      line=dict(color='blue')),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['macd_signal'],
                      name='Signal',
                      line=dict(color='orange')),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Bar(x=df['date'], y=df['macd_hist'],
                  name='MACD Hist',
                  marker_color='gray'),
            row=2, col=1
        )

        # Plot 3: R-squared
        fig.add_trace(
            go.Scatter(x=rolling_reg['date'], y=rolling_reg['r.squared'],
                      name='R-squared',
                      line=dict(color='green')),
            row=3, col=1
        )

        # Plot 4: Feature Coefficients
        features = [
            'volume_sma', 'price_sma', 'volatility', 'momentum', 'rsi',
            'bb_width', 'bb_position', 'macd', 'macd_divergence',
            'stoch_trend', 'atr_ratio', 'trend_strength',
            'volume_trend', 'trend_consistency'
        ]
        
        for feature in features:
            fig.add_trace(
                go.Scatter(x=rolling_reg['date'], 
                          y=rolling_reg[f'{feature}_coef'],
                          name=feature,
                          line=dict(width=1)),
                row=4, col=1
            )

        # Update layout
        fig.update_layout(
            title=f'Advanced Technical Analysis for {self.symbol}',
            height=1600,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.05
            )
        )
        
        # Add metrics annotation
        metrics_text = (
            f"ARIMA Metrics:<br>"
            f"AIC: {metrics['aic']:.2f}<br>"
            f"BIC: {metrics['bic']:.2f}<br>"
            f"MSE: {metrics['mse']:.2f}<br>"
            f"MAE: {metrics['mae']:.2f}<br><br>"
            f"Latest Indicators:<br>"
            f"RSI: {df['rsi'].iloc[-1]:.1f}<br>"
            f"MACD: {df['macd'].iloc[-1]:.1f}<br>"
            f"Stoch K: {df['stoch_k'].iloc[-1]:.1f}<br>"
            f"ATR Ratio: {df['atr_ratio'].iloc[-1]:.4f}<br>"
            f"Trend Strength: {df['trend_strength'].iloc[-1]:.4f}"
        )
        
        fig.add_annotation(
            xref="paper", yref="paper",
            x=1.15, y=0.8,
            text=metrics_text,
            showarrow=False,
            font=dict(size=10),
            align="left"
        )

        fig.show()
    
    def run_analysis(self) -> None:
        """Run the complete analysis pipeline."""
        try:
            logger.info(f"Starting analysis for {self.symbol}")
            
            # Fetch and prepare data
            df = self.fetch_data()
            df = self.calculate_features(df)
            
            # Perform rolling regression
            rolling_reg = self.perform_rolling_regression(df)
            
            # Fit ARIMA and get forecast
            forecast_df, metrics = self.fit_arima(df)
            
            # Plot results
            self.plot_analysis(df, rolling_reg, forecast_df, metrics)
            
            # Print latest statistics
            latest_reg = rolling_reg.iloc[-1]
            print("\nLatest Rolling Regression Statistics:")
            print(f"R-squared: {latest_reg['r.squared']:.4f}")
            print("\nFeature Coefficients:")
            for feature in [
                'volume_sma', 'price_sma', 'volatility', 'momentum', 'rsi',
                'bb_width', 'bb_position', 'macd', 'macd_divergence',
                'stoch_trend', 'atr_ratio', 'trend_strength',
                'volume_trend', 'trend_consistency'
            ]:
                print(f"{feature}: {latest_reg[f'{feature}_coef']:.6f}")
            
            print("\nARIMA Forecast:")
            print(forecast_df)
            
        except Exception as e:
            logger.error(f"Error in analysis pipeline: {str(e)}")
            raise
