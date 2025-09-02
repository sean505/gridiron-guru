"""
Evaluation module for NFL prediction engine performance analysis.

This module provides comprehensive evaluation tools for measuring
model performance, calibration, and betting-relevant metrics.
"""

from .performance_metrics import PerformanceEvaluator
from .model_analysis import ModelAnalyzer

__all__ = [
    'PerformanceEvaluator',
    'ModelAnalyzer'
]
