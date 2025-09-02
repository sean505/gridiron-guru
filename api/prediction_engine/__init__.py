"""
Gridiron Guru Prediction Engine

A comprehensive NFL prediction system with ensemble machine learning models,
advanced feature engineering, and multi-source data integration.

Main Components:
- data_collector: Multi-source data collection and caching
- feature_engineering: Feature creation and preprocessing
- models: Pydantic data models and validation
- prediction_engine: Ensemble ML models and prediction logic

Usage:
    from prediction_engine import PredictionService
    
    # Initialize service
    service = PredictionService()
    
    # Train models (first time setup)
    service.train_models()
    
    # Make predictions
    predictions = service.predict_week(2024, 18)
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

# Import training components
from .training import (
    TrainingDataPreprocessor,
    EnsembleModelTrainer,
    ModelValidator,
    HyperparameterTuner
)
from .evaluation import (
    PerformanceEvaluator,
    ModelAnalyzer
)

logger = logging.getLogger(__name__)


class PredictionService:
    """
    Main service class that provides a unified interface to the prediction engine.
    
    This class coordinates between data collection, feature engineering, and prediction
    to provide a clean API for making NFL game predictions.
    """
    
    def __init__(self):
        """Initialize the prediction service."""
        self.trainer = EnsembleModelTrainer()
        self.preprocessor = TrainingDataPreprocessor()
        self.validator = ModelValidator()
        self.evaluator = PerformanceEvaluator()
        
        # Service status
        self.is_initialized = False
        self.models_trained = False
        
        logger.info("PredictionService initialized")
    
    def initialize(self) -> bool:
        """
        Initialize the prediction service.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            logger.info("Initializing PredictionService...")
            
            # Try to load existing models
            self.models_trained = self.trainer.load_trained_models()
            
            if self.models_trained:
                logger.info("Existing models loaded successfully")
            else:
                logger.info("No existing models found - training required")
            
            self.is_initialized = True
            logger.info("PredictionService initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing PredictionService: {e}")
            return False
    
    def train_models(self, start_season: int = 2008, end_season: int = 2023) -> Dict[str, Any]:
        """
        Train the prediction models.
        
        Args:
            start_season: First season to use for training
            end_season: Last season to use for training
            
        Returns:
            Training results and performance metrics
        """
        try:
            logger.info("Starting model training...")
            
            # Prepare training data
            X_train, X_val, y_train, y_val = self.preprocessor.prepare_training_data(
                start_season=start_season, end_season=end_season
            )
            
            # Train individual models
            training_results = self.trainer.train_individual_models(X_train, y_train, X_val, y_val)
            
            # Create ensemble
            ensemble_results = self.trainer.create_ensemble(X_train, y_train, X_val, y_val)
            
            # Save models
            self.trainer.save_trained_models()
            
            # Update status
            self.models_trained = True
            
            logger.info("Model training completed successfully")
            return {
                'training_results': training_results,
                'ensemble_results': ensemble_results
            }
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
            raise
    
    def get_available_data(self) -> Dict[str, Any]:
        """
        Get information about available data sources and seasons.
        
        Returns:
            Dictionary with available data information
        """
        try:
            return {
                'models_trained': self.models_trained,
                'service_initialized': self.is_initialized,
                'trainer_summary': self.trainer.get_model_summary()
            }
            
        except Exception as e:
            logger.error(f"Error getting available data: {e}")
            return {
                'error': str(e),
                'models_trained': False,
                'service_initialized': False
            }
    



# Global prediction service instance
prediction_service = PredictionService()

# Export main classes and functions
__all__ = [
    'PredictionService',
    'prediction_service',
    'TrainingDataPreprocessor',
    'EnsembleModelTrainer',
    'ModelValidator',
    'HyperparameterTuner',
    'PerformanceEvaluator',
    'ModelAnalyzer'
]
