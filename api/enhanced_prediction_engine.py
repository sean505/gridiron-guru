"""
Enhanced Prediction Engine with High-Impact NFL Features

This module integrates the new high-impact NFL features into the prediction pipeline
while maintaining compatibility with existing trained models.
"""

import logging
import numpy as np
import pandas as pd
import joblib
import json
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import cross_val_score, train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# Import enhanced feature engine
from prediction_engine.enhanced_feature_engine import enhanced_feature_engine

logger = logging.getLogger(__name__)


class EnhancedPredictionEngine:
    """
    Enhanced prediction engine that uses high-impact NFL features.
    
    This integrates EPA trends, situational efficiency, turnover/explosive plays,
    and rest/travel factors while maintaining 28-feature compatibility.
    """
    
    def __init__(self, model_dir: str = "prediction_engine/models/trained"):
        """Initialize the enhanced prediction engine."""
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        # Models
        self.logistic_model = None
        self.random_forest_model = None
        self.xgboost_model = None
        self.ensemble_weights = None
        self.ensemble_model = None
        
        # Scalers and preprocessing
        self.scaler = StandardScaler()
        self.feature_names = []
        
        # Model metadata
        self.model_metadata = {}
        self.training_results = {}
        
        # Load existing models
        self._load_models()
        
        logger.info("EnhancedPredictionEngine initialized with high-impact features")
    
    def _load_models(self):
        """Load pre-trained models."""
        try:
            # Load ensemble model
            ensemble_path = self.model_dir / "ensemble.joblib"
            if ensemble_path.exists():
                self.ensemble_model = joblib.load(ensemble_path)
                logger.info("Loaded ensemble model")
            
            # Load individual models
            logistic_path = self.model_dir / "logistic_regression.joblib"
            if logistic_path.exists():
                self.logistic_model = joblib.load(logistic_path)
                logger.info("Loaded logistic regression model")
            
            random_forest_path = self.model_dir / "random_forest.joblib"
            if random_forest_path.exists():
                self.random_forest_model = joblib.load(random_forest_path)
                logger.info("Loaded random forest model")
            
            xgboost_path = self.model_dir / "xgboost.joblib"
            if xgboost_path.exists():
                self.xgboost_model = joblib.load(xgboost_path)
                logger.info("Loaded XGBoost model")
            
            # Load scaler
            scaler_path = self.model_dir / "feature_scaler.joblib"
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                logger.info("Loaded feature scaler")
            
            # Load ensemble weights
            weights_path = self.model_dir / "ensemble_weights.joblib"
            if weights_path.exists():
                self.ensemble_weights = joblib.load(weights_path)
                logger.info("Loaded ensemble weights")
            
            # Load metadata
            metadata_path = self.model_dir / "model_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.model_metadata = json.load(f)
                logger.info("Loaded model metadata")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def predict_game(self, home_team: str, away_team: str, season: int, week: int) -> Dict[str, Any]:
        """
        Predict the outcome of a game using enhanced features.
        
        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            season: NFL season year
            week: Week number
            
        Returns:
            Dictionary containing prediction results
        """
        try:
            # Generate enhanced features
            features = enhanced_feature_engine.create_game_features(home_team, away_team, season, week)
            
            if not features or len(features) != 28:
                logger.warning(f"Invalid features for {away_team} @ {home_team}, using fallback")
                return self._create_fallback_prediction(home_team, away_team)
            
            # Convert to numpy array
            feature_array = np.array(features).reshape(1, -1)
            
            # Scale features
            try:
                feature_array_scaled = self.scaler.transform(feature_array)
            except Exception as e:
                logger.warning(f"Error scaling features: {e}, using unscaled features")
                feature_array_scaled = feature_array
            
            # Make predictions
            prediction_results = self._make_ensemble_prediction(feature_array_scaled)
            
            # Add team information
            prediction_results.update({
                'home_team': home_team,
                'away_team': away_team,
                'season': season,
                'week': week,
                'features_used': len(features),
                'feature_engine': 'enhanced'
            })
            
            logger.info(f"Generated enhanced prediction for {away_team} @ {home_team}")
            return prediction_results
            
        except Exception as e:
            logger.error(f"Error predicting game {away_team} @ {home_team}: {e}")
            return self._create_fallback_prediction(home_team, away_team)
    
    def _make_ensemble_prediction(self, features: np.ndarray) -> Dict[str, Any]:
        """Make ensemble prediction using all available models."""
        try:
            predictions = {}
            probabilities = {}
            
            # Individual model predictions
            if self.logistic_model is not None:
                try:
                    lr_pred = self.logistic_model.predict(features)[0]
                    lr_prob = self.logistic_model.predict_proba(features)[0]
                    predictions['logistic_regression'] = int(lr_pred)
                    probabilities['logistic_regression'] = float(lr_prob[1])  # Probability of home team winning
                except Exception as e:
                    logger.warning(f"Error with logistic regression: {e}")
            
            if self.random_forest_model is not None:
                try:
                    rf_pred = self.random_forest_model.predict(features)[0]
                    rf_prob = self.random_forest_model.predict_proba(features)[0]
                    predictions['random_forest'] = int(rf_pred)
                    probabilities['random_forest'] = float(rf_prob[1])
                except Exception as e:
                    logger.warning(f"Error with random forest: {e}")
            
            if self.xgboost_model is not None:
                try:
                    xgb_pred = self.xgboost_model.predict(features)[0]
                    xgb_prob = self.xgboost_model.predict_proba(features)[0]
                    predictions['xgboost'] = int(xgb_pred)
                    probabilities['xgboost'] = float(xgb_prob[1])
                except Exception as e:
                    logger.warning(f"Error with XGBoost: {e}")
            
            # Ensemble prediction
            if self.ensemble_model is not None:
                try:
                    ensemble_pred = self.ensemble_model.predict(features)[0]
                    ensemble_prob = self.ensemble_model.predict_proba(features)[0]
                    predictions['ensemble'] = int(ensemble_pred)
                    probabilities['ensemble'] = float(ensemble_prob[1])
                except Exception as e:
                    logger.warning(f"Error with ensemble model: {e}")
            
            # Calculate weighted average if we have individual predictions
            if probabilities:
                if self.ensemble_weights is not None:
                    # Use learned ensemble weights
                    weighted_prob = 0.0
                    total_weight = 0.0
                    for model_name, prob in probabilities.items():
                        if model_name in self.ensemble_weights:
                            weight = self.ensemble_weights[model_name]
                            weighted_prob += prob * weight
                            total_weight += weight
                    
                    if total_weight > 0:
                        final_probability = weighted_prob / total_weight
                    else:
                        final_probability = np.mean(list(probabilities.values()))
                else:
                    # Use simple average
                    final_probability = np.mean(list(probabilities.values()))
                
                # Determine winner
                home_team_wins = final_probability > 0.5
                confidence = abs(final_probability - 0.5) * 2  # Convert to 0-1 scale
                
                # Calculate upset potential
                upset_potential = 1.0 - confidence
                
            else:
                # Fallback if no models work
                final_probability = 0.5
                home_team_wins = True
                confidence = 0.0
                upset_potential = 1.0
            
            return {
                'home_team_wins': bool(home_team_wins),
                'win_probability': float(final_probability),
                'confidence': float(confidence),
                'upset_potential': float(upset_potential),
                'individual_predictions': predictions,
                'individual_probabilities': probabilities,
                'model_metadata': self.model_metadata
            }
            
        except Exception as e:
            logger.error(f"Error making ensemble prediction: {e}")
            return self._create_fallback_prediction("", "")
    
    def _create_fallback_prediction(self, home_team: str, away_team: str) -> Dict[str, Any]:
        """Create fallback prediction when models fail."""
        return {
            'home_team': home_team,
            'away_team': away_team,
            'home_team_wins': True,
            'win_probability': 0.5,
            'confidence': 0.0,
            'upset_potential': 1.0,
            'individual_predictions': {},
            'individual_probabilities': {},
            'model_metadata': {},
            'features_used': 0,
            'feature_engine': 'fallback'
        }
    
    def get_feature_importance(self) -> Dict[str, Any]:
        """Get feature importance from trained models."""
        try:
            importance_data = {}
            
            if self.random_forest_model is not None and hasattr(self.random_forest_model, 'feature_importances_'):
                importance_data['random_forest'] = self.random_forest_model.feature_importances_.tolist()
            
            if self.xgboost_model is not None and hasattr(self.xgboost_model, 'feature_importances_'):
                importance_data['xgboost'] = self.xgboost_model.feature_importances_.tolist()
            
            # Get feature names from enhanced feature engine
            feature_names = enhanced_feature_engine.get_feature_names()
            importance_data['feature_names'] = feature_names
            
            return importance_data
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return {}
    
    def validate_models(self) -> Dict[str, Any]:
        """Validate that all models are loaded and working."""
        validation_results = {
            'logistic_regression': self.logistic_model is not None,
            'random_forest': self.random_forest_model is not None,
            'xgboost': self.xgboost_model is not None,
            'ensemble': self.ensemble_model is not None,
            'scaler': self.scaler is not None,
            'ensemble_weights': self.ensemble_weights is not None,
            'model_metadata': len(self.model_metadata) > 0
        }
        
        working_models = sum(validation_results.values())
        validation_results['total_models'] = len(validation_results)
        validation_results['working_models'] = working_models
        validation_results['status'] = 'healthy' if working_models >= 3 else 'degraded'
        
        return validation_results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        return {
            'model_metadata': self.model_metadata,
            'feature_count': 28,
            'feature_engine': 'enhanced',
            'models_loaded': {
                'logistic_regression': self.logistic_model is not None,
                'random_forest': self.random_forest_model is not None,
                'xgboost': self.xgboost_model is not None,
                'ensemble': self.ensemble_model is not None
            },
            'validation': self.validate_models()
        }


# Global enhanced prediction engine instance
enhanced_prediction_engine = EnhancedPredictionEngine()
