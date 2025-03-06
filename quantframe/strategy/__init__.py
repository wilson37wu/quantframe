"""Strategy module."""
from .base import BaseStrategy, Position
from .momentum import MomentumStrategy

__all__ = ['BaseStrategy', 'Position', 'MomentumStrategy']
