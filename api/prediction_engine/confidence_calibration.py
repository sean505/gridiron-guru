"""
Confidence Calibration System
Improves prediction confidence scoring using feature context and historical performance
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import json
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


class ConfidenceCalibrator:
    """
    Calibrates prediction confidence scores for better accuracy.
    
    Features:
    - Feature-based confidence adjustment
    - Historical performance calibration
    - Model-specific calibration
    - Uncertainty quantification
    """
    
    def __init__(self, calibration_data_dir: str = "api/data/calibration"):
        """Initialize the confidence calibrator."""
        self.calibration_data_dir = Path(calibration_data_dir)
        self.calibration_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Calibration models
        self.calibration_models = {}
        self.feature_weights = {}
        self.historical_performance = {}
        
        # Calibration data
        self.calibration_data = []
        self.model_performance_history = {}
        
        logger.info("ConfidenceCalibrator initialized")
    
    def calibrate_confidence(self, raw_confidence: float, features: List[float], 
                           model_name: str, game_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calibrate confidence score using feature context and historical performance.
        
        Args:
            raw_confidence: Raw confidence score from model (0-1)
            features: Feature vector used for prediction
            model_name: Name of the model that made the prediction
            game_context: Additional game context information
            
        Returns:
            Dictionary with calibrated confidence and uncertainty metrics
        """
        try:
            # 1. Feature-based adjustment
            feature_adjusted_confidence = self._adjust_confidence_by_features(
                raw_confidence, features, model_name
            )
            
            # 2. Historical performance calibration
            performance_adjusted_confidence = self._adjust_confidence_by_performance(
                feature_adjusted_confidence, model_name, game_context
            )
            
            # 3. Context-based adjustment
            final_confidence = self._adjust_confidence_by_context(
                performance_adjusted_confidence, game_context
            )
            
            # 4. Calculate uncertainty metrics
            uncertainty = self._calculate_uncertainty(
                raw_confidence, final_confidence, features, model_name
            )
            
            # 5. Generate confidence interpretation
            interpretation = self._interpret_confidence(final_confidence, uncertainty)
            
            return {
                'raw_confidence': raw_confidence,
                'calibrated_confidence': final_confidence,
                'confidence_adjustment': final_confidence - raw_confidence,
                'uncertainty': uncertainty,
                'interpretation': interpretation,
                'calibration_factors': {
                    'feature_adjustment': feature_adjusted_confidence - raw_confidence,
                    'performance_adjustment': performance_adjusted_confidence - feature_adjusted_confidence,
                    'context_adjustment': final_confidence - performance_adjusted_confidence
                }
            }
            
        except Exception as e:
            logger.error(f"Error calibrating confidence: {e}")
            return {
                'raw_confidence': raw_confidence,
                'calibrated_confidence': raw_confidence,
                'confidence_adjustment': 0.0,
                'uncertainty': {'total_uncertainty': 0.5},
                'interpretation': 'Calibration failed - using raw confidence',
                'calibration_factors': {}
            }
    
    def _adjust_confidence_by_features(self, confidence: float, features: List[float], 
                                     model_name: str) -> float:
        """Adjust confidence based on feature values and importance."""
        try:
            # Get feature weights for the model
            if model_name not in self.feature_weights:
                self._load_feature_weights(model_name)
            
            weights = self.feature_weights.get(model_name, {})
            if not weights:
                return confidence
            
            # Calculate feature-based adjustment
            adjustment = 0.0
            
            # Check for high-importance features that might affect confidence
            for i, feature_value in enumerate(features):
                if i < len(weights):
                    feature_weight = weights.get(f'feature_{i}', 0.0)
                    
                    # Adjust based on feature value and importance
                    if feature_weight > 0.1:  # High importance feature
                        if feature_value > 0.7:  # Strong positive signal
                            adjustment += 0.05
                        elif feature_value < 0.3:  # Strong negative signal
                            adjustment -= 0.05
            
            # Apply adjustment with bounds
            adjusted_confidence = max(0.0, min(1.0, confidence + adjustment))
            
            return adjusted_confidence
            
        except Exception as e:
            logger.error(f"Error adjusting confidence by features: {e}")
            return confidence
    
    def _adjust_confidence_by_performance(self, confidence: float, model_name: str,
                                        game_context: Dict[str, Any]) -> float:
        """Adjust confidence based on historical model performance."""
        try:
            # Get historical performance for the model
            if model_name not in self.historical_performance:
                self._load_historical_performance(model_name)
            
            performance = self.historical_performance.get(model_name, {})
            if not performance:
                return confidence
            
            # Get base accuracy for the model
            base_accuracy = performance.get('accuracy', 0.6)
            
            # Adjust confidence based on model accuracy
            if base_accuracy > 0.7:
                # High-performing model - increase confidence
                adjustment = (base_accuracy - 0.6) * 0.2
            elif base_accuracy < 0.55:
                # Low-performing model - decrease confidence
                adjustment = (base_accuracy - 0.6) * 0.3
            else:
                adjustment = 0.0
            
            # Apply adjustment
            adjusted_confidence = max(0.0, min(1.0, confidence + adjustment))
            
            return adjusted_confidence
            
        except Exception as e:
            logger.error(f"Error adjusting confidence by performance: {e}")
            return confidence
    
    def _adjust_confidence_by_context(self, confidence: float, game_context: Dict[str, Any]) -> float:
        """Adjust confidence based on game context factors."""
        try:
            adjustment = 0.0
            
            # Weather impact
            weather_factor = game_context.get('weather_factor', 0.0)
            if weather_factor > 0.5:  # Severe weather
                adjustment -= 0.1
            elif weather_factor < 0.2:  # Good weather
                adjustment += 0.05
            
            # Rest advantage
            rest_differential = game_context.get('rest_differential', 0.0)
            if abs(rest_differential) > 3:  # Significant rest advantage
                adjustment += 0.05
            
            # Division game
            is_division_game = game_context.get('is_division_game', False)
            if is_division_game:
                adjustment -= 0.05  # Division games are more unpredictable
            
            # Playoff implications
            playoff_implications = game_context.get('playoff_implications', 0.0)
            if playoff_implications > 0.7:  # High stakes
                adjustment -= 0.1
            
            # Apply adjustment
            adjusted_confidence = max(0.0, min(1.0, confidence + adjustment))
            
            return adjusted_confidence
            
        except Exception as e:
            logger.error(f"Error adjusting confidence by context: {e}")
            return confidence
    
    def _calculate_uncertainty(self, raw_confidence: float, calibrated_confidence: float,
                             features: List[float], model_name: str) -> Dict[str, float]:
        """Calculate uncertainty metrics for the prediction."""
        try:
            uncertainty = {}
            
            # 1. Model uncertainty (based on confidence spread)
            model_uncertainty = abs(calibrated_confidence - raw_confidence)
            uncertainty['model_uncertainty'] = model_uncertainty
            
            # 2. Feature uncertainty (based on feature variance)
            feature_variance = np.var(features) if len(features) > 0 else 0.0
            uncertainty['feature_uncertainty'] = min(0.5, feature_variance)
            
            # 3. Calibration uncertainty (based on historical calibration performance)
            calibration_uncertainty = self._get_calibration_uncertainty(model_name)
            uncertainty['calibration_uncertainty'] = calibration_uncertainty
            
            # 4. Total uncertainty
            total_uncertainty = np.sqrt(
                model_uncertainty**2 + 
                uncertainty['feature_uncertainty']**2 + 
                calibration_uncertainty**2
            )
            uncertainty['total_uncertainty'] = min(0.5, total_uncertainty)
            
            # 5. Confidence interval
            uncertainty['confidence_interval'] = {
                'lower': max(0.0, calibrated_confidence - 1.96 * total_uncertainty),
                'upper': min(1.0, calibrated_confidence + 1.96 * total_uncertainty)
            }
            
            return uncertainty
            
        except Exception as e:
            logger.error(f"Error calculating uncertainty: {e}")
            return {'total_uncertainty': 0.5}
    
    def _interpret_confidence(self, confidence: float, uncertainty: Dict[str, float]) -> str:
        """Generate human-readable confidence interpretation."""
        try:
            total_uncertainty = uncertainty.get('total_uncertainty', 0.5)
            
            if confidence > 0.8 and total_uncertainty < 0.2:
                return "Very High Confidence - Strong evidence supports this prediction"
            elif confidence > 0.7 and total_uncertainty < 0.3:
                return "High Confidence - Good evidence supports this prediction"
            elif confidence > 0.6 and total_uncertainty < 0.4:
                return "Moderate Confidence - Some evidence supports this prediction"
            elif confidence > 0.5 and total_uncertainty < 0.5:
                return "Low Confidence - Limited evidence for this prediction"
            else:
                return "Very Low Confidence - High uncertainty in this prediction"
                
        except Exception as e:
            logger.error(f"Error interpreting confidence: {e}")
            return "Confidence interpretation unavailable"
    
    def _get_calibration_uncertainty(self, model_name: str) -> float:
        """Get calibration uncertainty for a specific model."""
        try:
            # This would be based on historical calibration performance
            # For now, return a default value
            return 0.1
            
        except Exception as e:
            logger.error(f"Error getting calibration uncertainty: {e}")
            return 0.2
    
    def _load_feature_weights(self, model_name: str):
        """Load feature weights for a specific model."""
        try:
            weights_file = self.calibration_data_dir / f"{model_name}_feature_weights.json"
            if weights_file.exists():
                with open(weights_file, 'r') as f:
                    self.feature_weights[model_name] = json.load(f)
            else:
                # Create default weights
                self.feature_weights[model_name] = {}
                
        except Exception as e:
            logger.error(f"Error loading feature weights for {model_name}: {e}")
            self.feature_weights[model_name] = {}
    
    def _load_historical_performance(self, model_name: str):
        """Load historical performance data for a specific model."""
        try:
            performance_file = self.calibration_data_dir / f"{model_name}_performance.json"
            if performance_file.exists():
                with open(performance_file, 'r') as f:
                    self.historical_performance[model_name] = json.load(f)
            else:
                # Create default performance
                self.historical_performance[model_name] = {
                    'accuracy': 0.6,
                    'confidence_calibration': 0.5,
                    'sample_size': 0
                }
                
        except Exception as e:
            logger.error(f"Error loading historical performance for {model_name}: {e}")
            self.historical_performance[model_name] = {'accuracy': 0.6}
    
    def update_calibration_data(self, prediction_id: str, raw_confidence: float,
                              calibrated_confidence: float, actual_result: bool,
                              model_name: str, features: List[float]):
        """Update calibration data with new prediction results."""
        try:
            # Create calibration record
            record = {
                'prediction_id': prediction_id,
                'timestamp': datetime.now().isoformat(),
                'raw_confidence': raw_confidence,
                'calibrated_confidence': calibrated_confidence,
                'actual_result': actual_result,
                'model_name': model_name,
                'features': features,
                'calibration_error': abs(calibrated_confidence - (1.0 if actual_result else 0.0))
            }
            
            # Add to calibration data
            self.calibration_data.append(record)
            
            # Update model performance
            if model_name not in self.model_performance_history:
                self.model_performance_history[model_name] = []
            
            self.model_performance_history[model_name].append(record)
            
            # Keep only recent data (last 1000 records per model)
            if len(self.model_performance_history[model_name]) > 1000:
                self.model_performance_history[model_name] = self.model_performance_history[model_name][-1000:]
            
            # Update historical performance
            self._update_historical_performance(model_name)
            
        except Exception as e:
            logger.error(f"Error updating calibration data: {e}")
    
    def _update_historical_performance(self, model_name: str):
        """Update historical performance metrics for a model."""
        try:
            if model_name not in self.model_performance_history:
                return
            
            records = self.model_performance_history[model_name]
            if not records:
                return
            
            # Calculate accuracy
            correct_predictions = sum(1 for r in records if r['actual_result'])
            total_predictions = len(records)
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
            
            # Calculate confidence calibration (Brier score)
            brier_score = np.mean([r['calibration_error']**2 for r in records])
            
            # Update historical performance
            self.historical_performance[model_name] = {
                'accuracy': accuracy,
                'confidence_calibration': 1.0 - brier_score,  # Higher is better
                'sample_size': total_predictions,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error updating historical performance: {e}")
    
    def get_calibration_report(self) -> Dict[str, Any]:
        """Get comprehensive calibration report."""
        try:
            report = {
                'calibration_summary': {
                    'total_predictions': len(self.calibration_data),
                    'models_tracked': list(self.historical_performance.keys()),
                    'last_updated': datetime.now().isoformat()
                },
                'model_performance': self.historical_performance,
                'calibration_quality': self._assess_calibration_quality(),
                'recommendations': self._get_calibration_recommendations()
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating calibration report: {e}")
            return {}
    
    def _assess_calibration_quality(self) -> Dict[str, Any]:
        """Assess the quality of confidence calibration."""
        try:
            if not self.calibration_data:
                return {'quality': 'insufficient_data', 'brier_score': 1.0}
            
            # Calculate overall Brier score
            brier_scores = [r['calibration_error']**2 for r in self.calibration_data]
            overall_brier_score = np.mean(brier_scores)
            
            # Assess quality
            if overall_brier_score < 0.1:
                quality = 'excellent'
            elif overall_brier_score < 0.2:
                quality = 'good'
            elif overall_brier_score < 0.3:
                quality = 'fair'
            else:
                quality = 'poor'
            
            return {
                'quality': quality,
                'brier_score': overall_brier_score,
                'sample_size': len(self.calibration_data)
            }
            
        except Exception as e:
            logger.error(f"Error assessing calibration quality: {e}")
            return {'quality': 'error', 'brier_score': 1.0}
    
    def _get_calibration_recommendations(self) -> List[str]:
        """Get recommendations for improving calibration."""
        try:
            recommendations = []
            
            # Check sample size
            if len(self.calibration_data) < 100:
                recommendations.append("Collect more prediction data for better calibration")
            
            # Check model performance
            for model_name, performance in self.historical_performance.items():
                if performance.get('confidence_calibration', 0) < 0.5:
                    recommendations.append(f"Improve calibration for {model_name} model")
            
            # Check overall quality
            quality = self._assess_calibration_quality()
            if quality.get('quality') == 'poor':
                recommendations.append("Consider retraining calibration models")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting calibration recommendations: {e}")
            return ["Unable to generate recommendations"]


# Global confidence calibrator instance
confidence_calibrator = ConfidenceCalibrator()
