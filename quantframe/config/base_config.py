"""Base configuration system for quantframe.

This module provides the foundational configuration infrastructure used across
the quantframe project, including parameter validation, type checking, and
conversion utilities.

Features:
    - Structured configuration using dataclasses
    - Parameter validation with ranges
    - Type checking and conversion
    - Utilities for dict and DataFrame conversion
"""
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Union
import pandas as pd

@dataclass
class BaseConfig:
    """Base configuration class with validation and conversion methods."""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary format."""
        return asdict(self)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert configuration to DataFrame format."""
        return pd.DataFrame([self.to_dict()])
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BaseConfig':
        """Create configuration instance from dictionary."""
        return cls(**config_dict)

    def validate_range(self, param_name: str, value: Union[int, float], 
                      min_val: Union[int, float], max_val: Union[int, float]) -> None:
        """Validate parameter falls within specified range.
        
        Args:
            param_name: Name of parameter being validated
            value: Parameter value to validate
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            
        Raises:
            ValueError: If value falls outside allowed range
        """
        if not min_val <= value <= max_val:
            raise ValueError(
                f"{param_name} must be between {min_val} and {max_val}. Got: {value}"
            )

    def validate(self) -> None:
        """Validate all configuration parameters.
        
        This method should be implemented by child classes to perform
        specific validation logic.
        
        Raises:
            NotImplementedError: If child class doesn't implement validation
        """
        raise NotImplementedError("Validation must be implemented by child class")
