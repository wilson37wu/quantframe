"""ICT Strategy configuration interface.

This module provides a structured interface for configuring and validating
ICT strategy parameters. It includes parameter validation, default values,
and documentation for each parameter.
"""

from dataclasses import dataclass, field
from typing import Dict, Any
import pandas as pd
from ...utils.validation import validate_range

@dataclass
class ICTConfig:
    """Configuration parameters for ICT strategy.
    
    This class provides a structured way to configure the ICT strategy
    with parameter validation and documentation.
    
    Attributes:
        fvg_threshold (float): Minimum size for Fair Value Gaps as percentage
        ob_lookback (int): Number of bars to analyze for Order Blocks
        volume_threshold (float): Volume multiplier for institutional activity
        stop_loss (float): Maximum loss per trade as percentage
        take_profit (float): Target profit per trade as percentage
        risk_per_trade (float): Maximum risk per trade as % of portfolio
        min_volume (float): Minimum volume for valid signals
        max_positions (int): Maximum number of concurrent positions
        
    Example:
        >>> config = ICTConfig(
        ...     fvg_threshold=0.003,  # 0.3% minimum gap
        ...     ob_lookback=30,       # Look back 30 bars
        ...     stop_loss=0.02        # 2% stop loss
        ... )
        >>> config.validate()
        >>> strategy = ICTStrategy(config)
    """
    
    # Fair Value Gap parameters
    fvg_threshold: float = field(
        default=0.002,
        metadata={
            'description': 'Minimum size for Fair Value Gaps',
            'range': (0.0001, 0.05),
            'unit': 'percentage'
        }
    )
    
    # Order Block parameters
    ob_lookback: int = field(
        default=20,
        metadata={
            'description': 'Bars to analyze for Order Blocks',
            'range': (10, 100),
            'unit': 'bars'
        }
    )
    
    volume_threshold: float = field(
        default=1.5,
        metadata={
            'description': 'Volume multiplier for institutional activity',
            'range': (1.1, 5.0),
            'unit': 'multiplier'
        }
    )
    
    # Risk management parameters
    stop_loss: float = field(
        default=0.02,
        metadata={
            'description': 'Maximum loss per trade',
            'range': (0.005, 0.10),
            'unit': 'percentage'
        }
    )
    
    take_profit: float = field(
        default=0.03,
        metadata={
            'description': 'Target profit per trade',
            'range': (0.005, 0.20),
            'unit': 'percentage'
        }
    )
    
    risk_per_trade: float = field(
        default=0.01,
        metadata={
            'description': 'Maximum risk per trade',
            'range': (0.001, 0.05),
            'unit': 'percentage of portfolio'
        }
    )
    
    # Additional filters
    min_volume: float = field(
        default=100000,
        metadata={
            'description': 'Minimum volume for valid signals',
            'range': (1000, 10000000),
            'unit': 'currency units'
        }
    )
    
    max_positions: int = field(
        default=5,
        metadata={
            'description': 'Maximum number of concurrent positions',
            'range': (1, 20),
            'unit': 'positions'
        }
    )
    
    def validate(self) -> bool:
        """Validate all configuration parameters.
        
        Returns:
            bool: True if all parameters are valid
            
        Raises:
            ValueError: If any parameter is invalid
        """
        for field_name, field_value in self.__dataclass_fields__.items():
            value = getattr(self, field_name)
            metadata = field_value.metadata
            
            if 'range' in metadata:
                min_val, max_val = metadata['range']
                if not validate_range(value, min_val, max_val):
                    raise ValueError(
                        f"{field_name} value {value} outside valid range "
                        f"[{min_val}, {max_val}] ({metadata['unit']})"
                    )
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Dict[str, Any]: Configuration as dictionary
        """
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ICTConfig':
        """Create configuration from dictionary.
        
        Args:
            config_dict: Dictionary with configuration values
            
        Returns:
            ICTConfig: New configuration instance
        """
        return cls(**config_dict)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert configuration to DataFrame for display.
        
        Returns:
            pd.DataFrame: Configuration parameters with metadata
        """
        data = []
        for field_name, field_value in self.__dataclass_fields__.items():
            value = getattr(self, field_name)
            metadata = field_value.metadata
            min_val, max_val = metadata.get('range', (None, None))
            
            data.append({
                'Parameter': field_name,
                'Value': value,
                'Description': metadata['description'],
                'Unit': metadata['unit'],
                'Min': min_val,
                'Max': max_val
            })
            
        return pd.DataFrame(data)
