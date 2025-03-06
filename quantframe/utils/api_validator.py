"""API validation utilities."""
import logging
from typing import Dict, Any, Optional, Tuple
import requests
from binance.client import Client
from binance.exceptions import BinanceAPIException

logger = logging.getLogger(__name__)

class APIValidator:
    """Validator for various API credentials."""
    
    @staticmethod
    def validate_binance(api_key: str, api_secret: str, base_url: Optional[str] = None) -> Tuple[bool, str]:
        """Validate Binance API credentials.
        
        Args:
            api_key: Binance API key
            api_secret: Binance API secret
            base_url: Optional base URL for testnet
            
        Returns:
            Tuple of (is_valid, message)
        """
        try:
            client = Client(api_key, api_secret, base_url=base_url)
            # Try to get account information
            client.get_account()
            return True, "API credentials are valid"
        except BinanceAPIException as e:
            if e.code == -2015:  # Invalid API-key, IP, or permissions for action
                return False, "Invalid API key or secret"
            elif e.code == -2014:  # API-key format invalid
                return False, "API key format is invalid"
            elif e.code == -1022:  # Signature for this request is not valid
                return False, "API secret is invalid"
            else:
                return False, f"Binance API error: {str(e)}"
        except Exception as e:
            return False, f"Error validating API credentials: {str(e)}"
    
    @staticmethod
    def validate_polygon(api_key: str) -> Tuple[bool, str]:
        """Validate Polygon.io API key.
        
        Args:
            api_key: Polygon.io API key
            
        Returns:
            Tuple of (is_valid, message)
        """
        try:
            # Test endpoint that requires authentication
            url = f"https://api.polygon.io/v2/aggs/ticker/AAPL/range/1/day/2023-01-01/2023-01-02?apiKey={api_key}"
            response = requests.get(url)
            
            if response.status_code == 200:
                return True, "API key is valid"
            elif response.status_code == 403:
                return False, "Invalid API key"
            elif response.status_code == 401:
                return False, "Unauthorized: API key is invalid or expired"
            else:
                return False, f"Unexpected response: {response.status_code}"
        except Exception as e:
            return False, f"Error validating API key: {str(e)}"
    
    @classmethod
    def validate_config(cls, provider: str, config: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate API configuration for a specific provider.
        
        Args:
            provider: Name of the API provider (e.g., 'binance', 'polygon')
            config: Configuration dictionary
            
        Returns:
            Tuple of (is_valid, message)
        """
        if provider == 'binance':
            # Check if using testnet
            if config.get('testnet', {}).get('enabled', False):
                api_key = config['testnet']['api_key']
                api_secret = config['testnet']['api_secret']
                base_url = config['testnet']['base_url']
            else:
                api_key = config['api_key']
                api_secret = config['api_secret']
                base_url = config.get('base_url')
                
            return cls.validate_binance(api_key, api_secret, base_url)
            
        elif provider == 'polygon':
            api_key = config['api_key']
            return cls.validate_polygon(api_key)
            
        else:
            return False, f"Unsupported provider: {provider}"
