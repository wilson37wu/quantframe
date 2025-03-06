"""Configuration management utilities."""
import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import logging
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv

class Config:
    """Configuration management class for handling API keys and other settings"""
    
    def __init__(self, config_path: Optional[str] = None, load_env: bool = True):
        """Initialize configuration manager
        
        Args:
            config_path: Optional path to config file. If not provided, uses default
            load_env: Whether to load environment variables from .env file
        """
        if load_env:
            # Load environment variables from .env file if it exists
            env_path = Path(__file__).parent.parent.parent / '.env'
            load_dotenv(env_path if env_path.exists() else None)
            
        self.config_path = config_path or str(Path(__file__).parent.parent.parent / 'config' / 'config.yaml')
        self.config = self._load_config()
        self._setup_logging()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file with environment variable substitution"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found at {self.config_path}")
            
        # Load main config
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Load API config if specified
        api_config_path = self._substitute_env_vars(config['data']['api_config'])
        api_config_path = os.path.join(os.path.dirname(self.config_path), api_config_path)
        
        if os.path.exists(api_config_path):
            with open(api_config_path, 'r') as f:
                api_config = yaml.safe_load(f)
                config['data'].update(api_config)
                
        # Substitute environment variables
        return self._substitute_env_vars(config)
        
    def _substitute_env_vars(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively substitute environment variables in config
        
        Args:
            config: Configuration dictionary to process
            
        Returns:
            Processed configuration with environment variables substituted
        """
        if isinstance(config, dict):
            return {k: self._substitute_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._substitute_env_vars(v) for v in config]
        elif isinstance(config, str) and config.startswith('${') and config.endswith('}'):
            # Handle default values in env vars: ${VAR_NAME:default_value}
            env_var = config[2:-1]
            if ':' in env_var:
                env_var, default = env_var.split(':', 1)
                return os.environ.get(env_var, default)
            return os.environ.get(env_var, '')
        return config
        
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key
        
        Args:
            key: Configuration key to retrieve (dot notation supported)
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default
        return value
    
    def get_api_config(self, provider: str) -> Dict[str, Any]:
        """Get API configuration for a specific provider
        
        Args:
            provider: Name of the API provider (e.g., 'binance', 'polygon')
            
        Returns:
            API configuration dictionary
        """
        return self.get(f'data.{provider}', {})
    
    def _setup_logging(self):
        """Setup logging based on configuration"""
        log_config = self.config.get('logging', {})
        log_level = getattr(logging, log_config.get('level', 'INFO'))
        log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Create formatter
        formatter = logging.Formatter(log_format)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        
        # Clear existing handlers
        root_logger.handlers = []
        
        # Add handlers from config
        handlers = log_config.get('handlers', [])
        for handler_config in handlers:
            handler_type = handler_config.get('type', '').lower()
            if handler_type == 'file':
                handler = RotatingFileHandler(
                    handler_config.get('filename', 'trading.log'),
                    maxBytes=10*1024*1024,  # 10MB
                    backupCount=5
                )
            elif handler_type == 'console':
                handler = logging.StreamHandler()
            else:
                continue
                
            handler.setFormatter(formatter)
            root_logger.addHandler(handler)
