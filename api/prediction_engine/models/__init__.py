"""
Models module for storing and managing trained prediction models.

This module handles model persistence, loading, and metadata management.
"""

import os
from pathlib import Path

# Model storage paths
MODEL_BASE_PATH = Path(__file__).parent / "trained"
MODEL_BASE_PATH.mkdir(exist_ok=True)

# Model file names
LOGISTIC_REGRESSION_MODEL = "logistic_regression.joblib"
RANDOM_FOREST_MODEL = "random_forest.joblib"
XGBOOST_MODEL = "xgboost.joblib"
ENSEMBLE_MODEL = "ensemble.joblib"
FEATURE_SCALER = "feature_scaler.joblib"
MODEL_METADATA = "model_metadata.json"

__all__ = [
    'MODEL_BASE_PATH',
    'LOGISTIC_REGRESSION_MODEL',
    'RANDOM_FOREST_MODEL', 
    'XGBOOST_MODEL',
    'ENSEMBLE_MODEL',
    'FEATURE_SCALER',
    'MODEL_METADATA'
]
