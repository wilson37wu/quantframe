"""Performance analytics package"""
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from scipy import stats

def calculate_returns(equity_curve: pd.Series) -> pd.Series:
    """Calculate return series"""
    return equity_curve.pct_change()

def calculate_performance_metrics(equity_curve: pd.Series) -> Dict[str, float]:
    """Calculate comprehensive performance metrics"""
    returns = calculate_returns(equity_curve)
    
    # Basic metrics
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    annual_return = (1 + total_return) ** (252 / len(returns)) - 1
    daily_vol = returns.std()
    annual_vol = daily_vol * np.sqrt(252)
    sharpe_ratio = annual_return / annual_vol if annual_vol != 0 else 0
    
    # Drawdown analysis
    rolling_max = equity_curve.expanding().max()
    drawdown = (equity_curve - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    # Advanced metrics
    positive_returns = returns[returns > 0]
    negative_returns = returns[returns < 0]
    
    win_rate = len(positive_returns) / len(returns) if len(returns) > 0 else 0
    profit_factor = abs(positive_returns.sum() / negative_returns.sum()) if len(negative_returns) > 0 else np.inf
    
    # Risk metrics
    sortino_ratio = calculate_sortino_ratio(returns)
    calmar_ratio = abs(annual_return / max_drawdown) if max_drawdown != 0 else np.inf
    
    # Return distribution metrics
    skewness = stats.skew(returns.dropna()) if len(returns.dropna()) > 0 else np.nan
    kurtosis = stats.kurtosis(returns.dropna()) if len(returns.dropna()) > 0 else np.nan
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'annual_volatility': annual_vol,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'calmar_ratio': calmar_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'skewness': skewness,
        'kurtosis': kurtosis
    }

def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Calculate Sortino ratio"""
    excess_returns = returns - risk_free_rate
    downside_returns = returns[returns < 0]
    downside_std = np.sqrt(252) * np.sqrt(np.mean(downside_returns ** 2))
    
    if downside_std == 0:
        return np.inf if excess_returns.mean() > 0 else -np.inf
    
    return np.sqrt(252) * excess_returns.mean() / downside_std

def analyze_trades(trades: List[Any]) -> Dict[str, Any]:
    """Analyze trade statistics"""
    if not trades:
        return {}
    
    trade_returns = [trade.return_pct for trade in trades]
    trade_pnls = [trade.pnl for trade in trades]
    holding_times = [(trade.exit_time - trade.entry_time).total_seconds() / (24 * 3600)
                    for trade in trades]
    
    winning_trades = [t for t in trades if t.pnl > 0]
    losing_trades = [t for t in trades if t.pnl <= 0]
    
    return {
        'total_trades': len(trades),
        'winning_trades': len(winning_trades),
        'losing_trades': len(losing_trades),
        'win_rate': len(winning_trades) / len(trades),
        'avg_return': np.mean(trade_returns),
        'avg_winning_return': np.mean([t.return_pct for t in winning_trades]) if winning_trades else 0,
        'avg_losing_return': np.mean([t.return_pct for t in losing_trades]) if losing_trades else 0,
        'largest_winner': max(trade_returns),
        'largest_loser': min(trade_returns),
        'avg_holding_time': np.mean(holding_times),
        'total_pnl': sum(trade_pnls),
        'avg_pnl': np.mean(trade_pnls),
        'pnl_std': np.std(trade_pnls),
        'profit_factor': abs(sum(t.pnl for t in winning_trades) / 
                           sum(t.pnl for t in losing_trades)) if losing_trades else np.inf
    }

def create_trade_summary(trades: List[Any]) -> pd.DataFrame:
    """Create detailed trade summary DataFrame"""
    if not trades:
        return pd.DataFrame()
    
    summary = []
    for trade in trades:
        summary.append({
            'symbol': trade.symbol,
            'entry_time': trade.entry_time,
            'exit_time': trade.exit_time,
            'direction': trade.direction,
            'entry_price': trade.entry_price,
            'exit_price': trade.exit_price,
            'size': trade.size,
            'pnl': trade.pnl,
            'return_pct': trade.return_pct,
            'holding_time_days': (trade.exit_time - trade.entry_time).total_seconds() / (24 * 3600),
            'exit_reason': trade.reason
        })
    
    return pd.DataFrame(summary)

def calculate_rolling_metrics(returns: pd.Series, window: int = 252) -> pd.DataFrame:
    """Calculate rolling performance metrics"""
    rolling_return = returns.rolling(window).mean() * 252
    rolling_vol = returns.rolling(window).std() * np.sqrt(252)
    rolling_sharpe = rolling_return / rolling_vol
    rolling_drawdown = (1 + returns).cumprod() / (1 + returns).cumprod().expanding().max() - 1
    
    return pd.DataFrame({
        'rolling_return': rolling_return,
        'rolling_volatility': rolling_vol,
        'rolling_sharpe': rolling_sharpe,
        'rolling_drawdown': rolling_drawdown
    })