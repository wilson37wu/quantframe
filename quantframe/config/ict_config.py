"""ICT Strategy configuration module.

This module defines the configuration parameters and validation rules for the
ICT (Inner Circle Trader) strategy implementation.

Features:
    - FVG (Fair Value Gap) parameters
    - Order Block detection settings
    - Risk management configuration
    - Position management rules
"""
from dataclasses import dataclass
from typing import Optional

from .base_config import BaseConfig

@dataclass
class ICTConfig(BaseConfig):
    """Configuration for ICT trading strategy.
    
    Attributes:
        fvg_threshold: Minimum gap size for FVG detection (0.01% - 5%)
        ob_lookback: Number of bars to analyze for order blocks (10-100)
        volume_threshold: Volume multiplier for institutional activity (1.1x - 5.0x)
        stop_loss: Maximum loss per trade as decimal (0.5% - 10%)
        take_profit: Profit target per trade as decimal (0.5% - 20%)
        risk_per_trade: Risk per trade as decimal (0.1% - 5%)
        min_volume: Minimum volume for trade entry in base units
        max_positions: Maximum concurrent positions (1-20)
    """
    # FVG Parameters
    fvg_threshold: float = 0.002
    
    # Order Block Parameters
    ob_lookback: int = 20
    volume_threshold: float = 1.5
    
    # Risk Management
    stop_loss: float = 0.02
    take_profit: float = 0.03
    risk_per_trade: float = 0.01
    
    # Position Management
    min_volume: float = 100000.0
    max_positions: int = 5

    def validate(self) -> None:
        """Validate all ICT strategy parameters.
        
        Raises:
            ValueError: If any parameter falls outside its valid range
        """
        # FVG validation
        self.validate_range("fvg_threshold", self.fvg_threshold, 0.0001, 0.05)
        
        # Order Block validation
        self.validate_range("ob_lookback", self.ob_lookback, 10, 100)
        self.validate_range("volume_threshold", self.volume_threshold, 1.1, 5.0)
        
        # Risk Management validation
        self.validate_range("stop_loss", self.stop_loss, 0.005, 0.10)
        self.validate_range("take_profit", self.take_profit, 0.005, 0.20)
        self.validate_range("risk_per_trade", self.risk_per_trade, 0.001, 0.05)
        
        # Position Management validation
        self.validate_range("min_volume", self.min_volume, 0.0, float("inf"))
        self.validate_range("max_positions", self.max_positions, 1, 20)
