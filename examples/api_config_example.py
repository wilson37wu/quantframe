from quantframe.utils.config import Config
import logging

def main():
    # Initialize configuration
    config = Config()
    
    # Get API credentials
    binance_config = config.get('data.binance')
    polygon_config = config.get('data.polygon')
    
    # Example of accessing API keys (these will be loaded from environment variables)
    try:
        binance_api_key = binance_config['api_key']
        binance_api_secret = binance_config['api_secret']
        polygon_api_key = polygon_config['api_key']
        
        logging.info("Successfully loaded API configurations")
        logging.info(f"Binance Base URL: {binance_config['base_url']}")
        logging.info(f"Polygon Base URL: {polygon_config['base_url']}")
        
    except KeyError as e:
        logging.error(f"Missing required environment variable: {e}")
        return
    
    # Example of accessing other configuration settings
    strategy_config = config.get('strategy.mean_reversion')
    logging.info(f"Strategy Configuration: {strategy_config}")

if __name__ == "__main__":
    main()
