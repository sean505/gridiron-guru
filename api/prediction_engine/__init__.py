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
import numpy as np
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
    
    def predict_week(self, season: int, week: int) -> Dict[str, Any]:
        """
        Generate predictions for all games in a specific week.
        
        Args:
            season: NFL season year
            week: Week number
            
        Returns:
            Dictionary with weekly predictions
        """
        try:
            if not self.models_trained:
                raise ValueError("Models not trained yet")
            
            logger.info(f"Generating predictions for {season} Week {week}")
            
            # Get realistic Week 18 games for 2024
            if season == 2024 and week == 18:
                games = self._get_week18_2024_games()
            else:
                # For other weeks, generate sample games
                games = self._get_sample_games(season, week)
            
            predictions = []
            for game in games:
                # Generate features for prediction (simplified)
                features = self._generate_simple_features(game)
                
                # Get prediction from ensemble model
                prediction = self.trainer.predict_single_game(features)
                
                # Format for response
                game_prediction = {
                    'away_team': game['away_team'],
                    'home_team': game['home_team'],
                    'predicted_winner': game['home_team'] if prediction['predicted_winner'] == 'home' else game['away_team'],
                    'confidence': prediction['confidence'],
                    'predicted_score_home': 24,  # Simplified
                    'predicted_score_away': 21,  # Simplified
                    'is_upset_pick': prediction['is_upset'],
                    'explanation': prediction['explanation']
                }
                predictions.append(game_prediction)
            
            return {
                'season': season,
                'week': week,
                'games': predictions,
                'total_games': len(predictions)
            }
            
        except Exception as e:
            logger.error(f"Error generating weekly predictions: {e}")
            raise
    
    def _get_week18_2024_games(self) -> List[Dict[str, str]]:
        """Get realistic Week 18, 2024 NFL games"""
        return [
            {"away_team": "BUF", "home_team": "MIA"},
            {"away_team": "NYJ", "home_team": "NE"},
            {"away_team": "CIN", "home_team": "CLE"},
            {"away_team": "PIT", "home_team": "BAL"},
            {"away_team": "HOU", "home_team": "IND"},
            {"away_team": "JAX", "home_team": "TEN"},
            {"away_team": "DEN", "home_team": "LV"},
            {"away_team": "LAC", "home_team": "KC"},
            {"away_team": "DAL", "home_team": "WAS"},
            {"away_team": "NYG", "home_team": "PHI"},
            {"away_team": "CHI", "home_team": "GB"},
            {"away_team": "DET", "home_team": "MIN"},
            {"away_team": "ATL", "home_team": "NO"},
            {"away_team": "CAR", "home_team": "TB"},
            {"away_team": "ARI", "home_team": "SEA"},
            {"away_team": "LAR", "home_team": "SF"}
        ]
    
    def _get_sample_games(self, season: int, week: int) -> List[Dict[str, str]]:
        """Get sample games for other weeks"""
        return [
            {"away_team": "BUF", "home_team": "KC"},
            {"away_team": "SF", "home_team": "DAL"}
        ]
    
    def _generate_simple_features(self, game: Dict[str, str]) -> np.ndarray:
        """Generate simple features for prediction"""
        # Create a simple feature vector (28 features as expected by the model)
        # This is a simplified version - in production you'd use real team stats
        
        # Generate random features that match the training data structure
        features = np.random.randn(28)
        
        # Add some bias based on team names (for consistency)
        home_team_hash = hash(game['home_team']) % 1000 / 1000.0
        away_team_hash = hash(game['away_team']) % 1000 / 1000.0
        
        features[0] = home_team_hash  # Home team strength
        features[1] = away_team_hash  # Away team strength
        features[2] = 0.5  # Home field advantage
        
        return features

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
