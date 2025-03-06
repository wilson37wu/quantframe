"""Strategy configuration management utilities."""
import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import logging
from pydantic import BaseModel, Field, validator
from enum import Enum

logger = logging.getLogger(__name__)

class OrderType(str, Enum):
    """Valid order types."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"

class BaseStrategyConfig(BaseModel):
    """Base configuration model for all strategies."""
    
    # Risk management
    max_drawdown: float = Field(..., ge=0, le=1)
    max_leverage: float = Field(..., gt=0)
    
    # Position sizing
    max_position_size: float = Field(..., gt=0, le=1)
    
    # Execution
    order_type: OrderType
    slippage_tolerance: float = Field(..., ge=0, le=0.1)
    execution_timeout: int = Field(..., gt=0)
    retry_attempts: int = Field(..., ge=0)
    retry_delay: int = Field(..., ge=0)
    
    # Monitoring
    health_check_interval: int = Field(..., gt=0)
    logging_level: str = Field(...)
    
    @validator('logging_level')
    def validate_logging_level(cls, v):
        """Validate logging level."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid logging level. Must be one of {valid_levels}")
        return v.upper()

class MeanReversionConfig(BaseStrategyConfig):
    """Configuration model for mean reversion strategy."""
    
    # Core parameters
    lookback_period: int = Field(..., gt=0)
    entry_zscore: float = Field(..., gt=0)
    exit_zscore: float = Field(..., gt=0)
    rsi_period: int = Field(..., gt=0)
    rsi_entry_threshold: float = Field(..., ge=0, le=100)
    rsi_exit_threshold: float = Field(..., ge=0, le=100)
    
    # Position sizing
    volatility_lookback: int = Field(..., gt=0)
    position_size_atr_multiple: float = Field(..., gt=0)
    kelly_fraction: float = Field(..., gt=0, le=1)
    
    # Risk management
    stop_loss_atr_multiple: float = Field(..., gt=0)
    take_profit_atr_multiple: float = Field(..., gt=0)
    
    # Market filters
    min_adv: float = Field(..., gt=0)
    min_price: float = Field(..., gt=0)
    max_spread: float = Field(..., gt=0, le=1)

class MomentumConfig(BaseStrategyConfig):
    """Configuration model for momentum strategy."""
    
    # Core parameters
    lookback_periods: list[int] = Field(..., min_items=1)
    breakout_threshold: float = Field(..., gt=0)
    trend_threshold: float = Field(..., gt=0)
    momentum_smoothing: int = Field(..., gt=0)
    volume_factor: float = Field(..., gt=0)
    
    # Position sizing
    volatility_lookback: int = Field(..., gt=0)
    position_size_atr_multiple: float = Field(..., gt=0)
    kelly_fraction: float = Field(..., gt=0, le=1)
    position_update_frequency: str
    
    # Risk management
    stop_loss_atr_multiple: float = Field(..., gt=0)
    take_profit_atr_multiple: float = Field(..., gt=0)
    correlation_threshold: float = Field(..., gt=-1, le=1)
    
    # Signal weights
    signal_weights: Dict[str, float]
    
    @validator('signal_weights')
    def validate_weights(cls, v):
        """Validate signal weights sum to 1."""
        if abs(sum(v.values()) - 1.0) > 1e-6:
            raise ValueError("Signal weights must sum to 1")
        return v

class MomentumConfig(BaseModel):
    """Configuration model for momentum trading strategy."""
    # Signal parameters
    momentum_period: int = Field(default=20, description="Period for momentum calculation")
    rsi_period: int = Field(default=14, description="Period for RSI calculation")
    rsi_overbought: float = Field(default=70, description="RSI overbought threshold")
    rsi_oversold: float = Field(default=30, description="RSI oversold threshold")
    signal_threshold: float = Field(default=0.2, description="Minimum signal strength to generate trade")
    
    # Indicator weights
    roc_weight: float = Field(default=0.4, description="Weight for Rate of Change indicator")
    rsi_weight: float = Field(default=0.3, description="Weight for RSI indicator")
    macd_weight: float = Field(default=0.3, description="Weight for MACD indicator")
    
    # Thresholds
    roc_threshold: float = Field(default=0.02, description="Rate of Change threshold")
    macd_threshold: float = Field(default=0.002, description="MACD threshold")
    
    # Position sizing
    base_order_size: float = Field(default=0.01, description="Base position size")
    max_position_size: float = Field(default=0.1, description="Maximum position size")
    volatility_scaling: bool = Field(default=True, description="Enable volatility-based position sizing")
    volatility_multiplier: float = Field(default=1.0, description="Volatility scaling multiplier")
    
    # Risk management
    stop_loss_pct: float = Field(default=0.02, description="Stop loss percentage")
    take_profit_pct: float = Field(default=0.04, description="Take profit percentage")

class MeanReversionConfig(BaseModel):
    """Configuration model for mean reversion strategy."""
    # Moving average parameters
    ma_short_period: int = Field(default=10, description="Short-term moving average period")
    ma_long_period: int = Field(default=30, description="Long-term moving average period")
    
    # Bollinger Bands parameters
    bb_period: int = Field(default=20, description="Bollinger Bands period")
    bb_std_dev: float = Field(default=2.0, description="Number of standard deviations for Bollinger Bands")
    
    # Signal parameters
    deviation_threshold: float = Field(default=2.0, description="Price deviation threshold for signals")
    signal_threshold: float = Field(default=0.2, description="Minimum signal strength to generate trade")
    momentum_period: int = Field(default=10, description="Period for momentum calculation")
    momentum_threshold: float = Field(default=0.02, description="Momentum threshold to filter signals")
    
    # Position sizing
    base_order_size: float = Field(default=0.01, description="Base position size")
    max_position_size: float = Field(default=0.1, description="Maximum position size")
    volatility_scaling: bool = Field(default=True, description="Enable volatility-based position sizing")
    volatility_multiplier: float = Field(default=1.0, description="Volatility scaling multiplier")
    
    # Risk management
    base_stop_loss_pct: float = Field(default=0.02, description="Base stop loss percentage")
    base_take_profit_pct: float = Field(default=0.04, description="Base take profit percentage")

class GridTradingConfig(BaseModel):
    """Configuration model for grid trading strategy."""
    base_order_size: float = Field(default=0.01, description="Base order size for each grid level")
    max_position_size: float = Field(default=0.1, description="Maximum total position size")
    grid_levels: int = Field(default=10, description="Number of grid levels")
    grid_spacing_percentage: float = Field(default=1.0, description="Percentage spacing between grid levels")
    volatility_lookback: int = Field(default=20, description="Lookback period for volatility calculation")
    volatility_threshold: float = Field(default=2.0, description="Volatility threshold for adjusting grid spacing")
    take_profit_multiplier: float = Field(default=1.5, description="Multiplier for take profit levels")
    stop_loss_multiplier: float = Field(default=0.5, description="Multiplier for stop loss levels")
    size_scaling_factor: float = Field(default=1.0, description="Scaling factor for position size")
    
    @validator('grid_levels')
    def validate_grid_levels(cls, v):
        if v < 2:
            raise ValueError("grid_levels must be at least 2")
        return v

    @validator('grid_spacing_percentage')  # Validator decorator for grid_spacing_percentage field
                                         # This is a Pydantic validator that checks if the grid spacing value
                                         # is valid before allowing it to be set in the GridTradingConfig model
    def validate_grid_spacing(cls, v):
        if v <= 0:
            raise ValueError("grid_spacing_percentage must be positive")
        return v

class StrategyConfigManager:
    """Manager for handling strategy configurations."""
    
    def __init__(self, config_dir: str):
        """Initialize the config manager.
        
        Args:
            config_dir: Directory containing strategy configuration files
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Default configurations for each strategy
        self.default_configs = {
            "grid": GridTradingConfig(),
            "momentum": MomentumConfig(),
            "mean_reversion": MeanReversionConfig()
        }
        
        # Create default config files if they don't exist
        self._create_default_configs()
    
    def _create_default_configs(self):
        """Create default configuration files if they don't exist."""
        for strategy_name, config in self.default_configs.items():
            config_file = self.config_dir / f"{strategy_name}_config.yaml"
            if not config_file.exists():
                with open(config_file, "w") as f:
                    yaml.dump(config.dict(), f)
    
    def get_default_config(self, strategy_name: str) -> Dict[str, Any]:
        """Get default configuration for a strategy.
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            Default configuration dictionary
        """
        if strategy_name not in self.default_configs:
            raise ValueError(f"No default configuration found for strategy: {strategy_name}")
        
        return self.default_configs[strategy_name].dict()
    
    def load_config(self, strategy_name: str) -> BaseModel:
        """Load configuration for a strategy from file.
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            Configuration object
        """
        config_file = self.config_dir / f"{strategy_name}_config.yaml"
        logger.info(f"Loading configuration from {config_file}")
        if not config_file.exists():
            logger.warning(f"Configuration file does not exist. Loading default config for {strategy_name}")
            return self.default_configs[strategy_name]

        try:
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
                logger.info(f"Loaded configuration data: {config_data}")

            # Map the configuration data to the appropriate config class
            if strategy_name == 'grid':
                return GridTradingConfig(**config_data)
            elif strategy_name == 'momentum':
                return MomentumConfig(**config_data)
            elif strategy_name == 'mean_reversion':
                return MeanReversionConfig(**config_data)
            else:
                raise ValueError(f"Unknown strategy: {strategy_name}")
        except Exception as e:
            logger.error(f"Error loading configuration for {strategy_name}: {e}")
            raise

    def save_config(self, strategy_name: str, config: Dict[str, Any]):
        """Save configuration for a strategy to file.
        
        Args:
            strategy_name: Name of the strategy
            config: Configuration dictionary
        """
        # Validate config against the model
        if strategy_name == "grid":
            config = GridTradingConfig(**config)
        elif strategy_name == "momentum":
            config = MomentumConfig(**config)
        elif strategy_name == "mean_reversion":
            config = MeanReversionConfig(**config)
        
        config_file = self.config_dir / f"{strategy_name}_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config.dict(), f)
    
    def update_config(self, strategy_name: str, updates: Dict[str, Any]):
        """Update specific fields in a strategy's configuration.
        
        Args:
            strategy_name: Name of the strategy
            updates: Dictionary of field updates
        """
        config = self.load_config(strategy_name)
        config.update(updates)
        self.save_config(strategy_name, config)
        
    def get_config_descriptions(self, strategy_name: str) -> Dict[str, str]:
        """Get descriptions for all configuration parameters of a strategy.
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            Dictionary mapping parameter names to their descriptions
        """
        if strategy_name == "grid":
            model = GridTradingConfig
        elif strategy_name == "momentum":
            model = MomentumConfig
        elif strategy_name == "mean_reversion":
            model = MeanReversionConfig
        else:
            raise ValueError(f"No configuration found for strategy: {strategy_name}")
            
        return {
            field_name: model.model_fields[field_name].description
            for field_name in model.model_fields
        }
    
    def load_strategy_config(self, strategy_name: str) -> Dict:
        """Load strategy configuration.
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            Strategy configuration dictionary
        """
        return self.load_config(strategy_name).dict()

    def save_strategy_config(self, strategy_name: str, config: Dict):
        """Save strategy configuration.
        
        Args:
            strategy_name: Name of the strategy
            config: Configuration dictionary to save
        """
        self.save_config(strategy_name, config)
