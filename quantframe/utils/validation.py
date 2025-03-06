"""Validation utilities for quantframe.

This module provides validation functions for checking parameter values,
data quality, and configuration settings across the framework.
"""

from typing import Any, Tuple, Union, Optional
import numpy as np

def validate_range(value: Union[int, float], 
                  min_val: Optional[Union[int, float]] = None,
                  max_val: Optional[Union[int, float]] = None) -> bool:
    """Validate that a value falls within a specified range.
    
    Args:
        value: Value to validate
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)
        
    Returns:
        bool: True if value is within range
        
    Example:
        >>> validate_range(0.5, 0.0, 1.0)
        True
        >>> validate_range(-1, 0.0, 1.0)
        False
    """
    if min_val is not None and value < min_val:
        return False
    if max_val is not None and value > max_val:
        return False
    return True

def validate_positive(value: Union[int, float]) -> bool:
    """Validate that a value is positive.
    
    Args:
        value: Value to validate
        
    Returns:
        bool: True if value is positive
    """
    return value > 0

def validate_non_negative(value: Union[int, float]) -> bool:
    """Validate that a value is non-negative.
    
    Args:
        value: Value to validate
        
    Returns:
        bool: True if value is non-negative
    """
    return value >= 0

def validate_probability(value: float) -> bool:
    """Validate that a value is a valid probability [0, 1].
    
    Args:
        value: Value to validate
        
    Returns:
        bool: True if value is between 0 and 1
    """
    return 0 <= value <= 1

def validate_percentage(value: float, allow_negative: bool = False) -> bool:
    """Validate that a value is a valid percentage.
    
    Args:
        value: Value to validate as percentage (e.g., 0.05 for 5%)
        allow_negative: Whether to allow negative percentages
        
    Returns:
        bool: True if value is a valid percentage
    """
    if allow_negative:
        return -1 <= value <= 1
    return 0 <= value <= 1
