"""Utility script to validate API keys."""
import sys
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from quantframe.utils.config import Config
from quantframe.utils.api_validator import APIValidator

def main():
    """Validate API keys from configuration."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Load configuration
        config = Config()
        
        # Validate Binance API keys
        logging.info("Validating Binance API credentials...")
        binance_config = config.get_api_config('binance')
        
        # Check main API keys
        is_valid, message = APIValidator.validate_config('binance', binance_config)
        logging.info(f"Binance API: {message}")
        
        # Check testnet if enabled
        if binance_config.get('testnet', {}).get('enabled', False):
            logging.info("Validating Binance Testnet API credentials...")
            testnet_config = {
                'api_key': binance_config['testnet']['api_key'],
                'api_secret': binance_config['testnet']['api_secret'],
                'base_url': binance_config['testnet']['base_url']
            }
            is_valid, message = APIValidator.validate_binance(
                testnet_config['api_key'],
                testnet_config['api_secret'],
                testnet_config['base_url']
            )
            logging.info(f"Binance Testnet API: {message}")
        
        # Validate Polygon API key
        logging.info("Validating Polygon.io API key...")
        polygon_config = config.get_api_config('polygon')
        is_valid, message = APIValidator.validate_config('polygon', polygon_config)
        logging.info(f"Polygon API: {message}")
        
    except Exception as e:
        logging.error(f"Error validating API keys: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()
