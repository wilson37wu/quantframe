"""Advanced cryptocurrency market analysis examples."""
import os
import logging
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from quantframe.utils.config import Config
from quantframe.utils.api_validator import APIValidator
from quantframe.data.sources.binance_source import BinanceSource
from quantframe.analysis.market_analysis import MarketAnalyzer

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('crypto_analysis.log')
        ]
    )
    return logging.getLogger(__name__)

def validate_apis():
    """Validate API credentials."""
    logger = logging.getLogger(__name__)
    config = Config()
    
    # Validate Binance API
    logger.info("Validating Binance API credentials...")
    binance_config = config.get_api_config('binance')
    is_valid, message = APIValidator.validate_config('binance', binance_config)
    logger.info(f"Binance API: {message}")
    
    if not is_valid:
        raise ValueError("Invalid Binance API credentials")
    
    return config

def analyze_crypto_markets():
    """Perform comprehensive cryptocurrency market analysis."""
    logger = logging.getLogger(__name__)
    
    # Setup and validation
    config = validate_apis()
    data_source = BinanceSource(config.get_api_config('binance'))
    analyzer = MarketAnalyzer(data_source)
    
    # Analysis parameters
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'DOGEUSDT']
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.Timedelta(days=90)
    
    # Create results directory
    results_dir = Path('analysis_results')
    results_dir.mkdir(exist_ok=True)
    
    try:
        # 1. Market Microstructure Analysis
        logger.info("Performing market microstructure analysis...")
        micro_results = {}
        for symbol in symbols:
            micro_results[symbol] = analyzer.analyze_market_microstructure(
                symbol, start_date, end_date
            )
        
        # Plot bid-ask bounce and flow imbalance
        plt.figure(figsize=(12, 6))
        x = np.arange(len(symbols))
        width = 0.35
        
        plt.bar(x - width/2, [r['bid_ask_bounce'] for r in micro_results.values()], width, label='Bid-Ask Bounce')
        plt.bar(x + width/2, [r['flow_imbalance'] for r in micro_results.values()], width, label='Flow Imbalance')
        
        plt.xlabel('Symbol')
        plt.ylabel('Metric Value')
        plt.title('Market Microstructure Metrics')
        plt.xticks(x, symbols)
        plt.legend()
        plt.savefig(results_dir / 'microstructure_metrics.png')
        plt.close()
        
        # 2. Volatility Analysis
        logger.info("Analyzing volatility patterns...")
        for symbol in symbols:
            data = data_source.get_data(symbol, start_date, end_date)
            returns = np.log(data['close']).diff().dropna()
            
            # Volatility clustering
            persistence, autocorr = analyzer.analyze_volatility_clustering(returns)
            
            plt.figure(figsize=(10, 6))
            autocorr.plot(kind='bar')
            plt.title(f'{symbol} Volatility Autocorrelation')
            plt.xlabel('Lag')
            plt.ylabel('Autocorrelation')
            plt.savefig(results_dir / f'{symbol}_volatility_clustering.png')
            plt.close()
            
            # Regime detection
            regimes = analyzer.detect_regime_changes(returns)
            
            plt.figure(figsize=(12, 6))
            plt.plot(regimes.index, regimes['volatility'], label='Volatility')
            plt.scatter(regimes.index, regimes['volatility'], 
                       c=regimes['regime'].map({'Low': 'green', 'Normal': 'blue', 'High': 'red'}),
                       alpha=0.5)
            plt.title(f'{symbol} Volatility Regimes')
            plt.xlabel('Date')
            plt.ylabel('Volatility')
            plt.legend()
            plt.savefig(results_dir / f'{symbol}_regimes.png')
            plt.close()
        
        # 3. Liquidity Analysis
        logger.info("Analyzing market liquidity...")
        liquidity_metrics = {}
        for symbol in symbols:
            metrics = analyzer.analyze_liquidity(symbol, start_date, end_date)
            liquidity_metrics[symbol] = metrics
        
        # Plot liquidity metrics
        metrics_to_plot = ['illiquidity_ratio', 'turnover_ratio', 'relative_spread']
        fig, axes = plt.subplots(len(metrics_to_plot), 1, figsize=(12, 15))
        
        for i, metric in enumerate(metrics_to_plot):
            values = [metrics[metric] for metrics in liquidity_metrics.values()]
            axes[i].bar(symbols, values)
            axes[i].set_title(f'{metric.replace("_", " ").title()}')
            axes[i].set_ylabel('Value')
        
        plt.tight_layout()
        plt.savefig(results_dir / 'liquidity_metrics.png')
        plt.close()
        
        # 4. Cross-Market Analysis
        logger.info("Performing cross-market analysis...")
        cross_market = analyzer.analyze_cross_market_dynamics(
            symbols, start_date, end_date
        )
        
        # Plot correlation matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cross_market['correlation'], annot=True, cmap='coolwarm', 
                   xticklabels=symbols, yticklabels=symbols)
        plt.title('Cross-Market Correlations')
        plt.savefig(results_dir / 'cross_market_correlations.png')
        plt.close()
        
        # 5. Intraday Analysis
        logger.info("Analyzing intraday patterns...")
        for symbol in symbols:
            patterns = analyzer.analyze_intraday_patterns(symbol, days=30)
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            patterns['volatility'].plot(ax=ax1)
            ax1.set_title(f'{symbol} Intraday Volatility Pattern')
            ax1.set_xlabel('Hour')
            ax1.set_ylabel('Average Volatility')
            
            patterns['volume'].plot(ax=ax2)
            ax2.set_title(f'{symbol} Intraday Volume Pattern')
            ax2.set_xlabel('Hour')
            ax2.set_ylabel('Average Volume')
            
            plt.tight_layout()
            plt.savefig(results_dir / f'{symbol}_intraday_patterns.png')
            plt.close()
        
        logger.info("Analysis completed successfully. Results saved in 'analysis_results' directory.")
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        raise

def main():
    """Main execution function."""
    logger = setup_logging()
    
    try:
        analyze_crypto_markets()
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise

if __name__ == '__main__':
    main()
