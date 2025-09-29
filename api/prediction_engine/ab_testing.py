"""
A/B Testing Framework for Model Comparison
Compares original 28-feature models with enhanced 45+ feature models
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import json
import random

from .enhanced_prediction_engine import enhanced_prediction_engine
from .data_models import GameContext, GamePrediction

logger = logging.getLogger(__name__)


class ABTestingFramework:
    """
    A/B testing framework for comparing original and enhanced models.
    
    Features:
    - Random assignment of predictions to model variants
    - Performance tracking and metrics collection
    - Statistical significance testing
    - Gradual rollout with configurable traffic allocation
    """
    
    def __init__(self, results_dir: str = "api/data/ab_testing"):
        """Initialize the A/B testing framework."""
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Test configuration
        self.test_name = "enhanced_vs_original_models"
        self.start_date = datetime.now()
        self.traffic_allocation = {
            'original': 0.9,  # 90% original, 10% enhanced initially
            'enhanced': 0.1
        }
        
        # Results tracking
        self.results = {
            'original': [],
            'enhanced': []
        }
        
        # Performance metrics
        self.metrics = {
            'original': {
                'total_predictions': 0,
                'correct_predictions': 0,
                'accuracy': 0.0,
                'avg_confidence': 0.0,
                'avg_response_time': 0.0,
                'upset_detection_accuracy': 0.0
            },
            'enhanced': {
                'total_predictions': 0,
                'correct_predictions': 0,
                'accuracy': 0.0,
                'avg_confidence': 0.0,
                'avg_response_time': 0.0,
                'upset_detection_accuracy': 0.0
            }
        }
        
        logger.info("ABTestingFramework initialized")
    
    def assign_variant(self, user_id: str = None, game_id: str = None) -> str:
        """
        Assign a prediction request to either original or enhanced model.
        
        Args:
            user_id: User identifier for consistent assignment
            game_id: Game identifier for consistent assignment
            
        Returns:
            'original' or 'enhanced'
        """
        try:
            # Use deterministic assignment based on identifiers
            if user_id and game_id:
                # Use hash for consistent assignment
                assignment_key = f"{user_id}_{game_id}"
                hash_value = hash(assignment_key) % 100
            else:
                # Random assignment
                hash_value = random.randint(0, 99)
            
            # Assign based on traffic allocation
            if hash_value < self.traffic_allocation['original'] * 100:
                return 'original'
            else:
                return 'enhanced'
                
        except Exception as e:
            logger.error(f"Error assigning variant: {e}")
            return 'original'  # Default to original
    
    def make_prediction(self, home_team: str, away_team: str, season: int, week: int,
                       game_context: GameContext, user_id: str = None) -> Tuple[GamePrediction, str]:
        """
        Make a prediction using A/B testing assignment.
        
        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            season: NFL season year
            week: Week number
            game_context: Game context information
            user_id: User identifier for consistent assignment
            
        Returns:
            Tuple of (prediction, variant_used)
        """
        try:
            start_time = datetime.now()
            
            # Assign variant
            variant = self.assign_variant(user_id, f"{home_team}_{away_team}_{season}_{week}")
            
            # Make prediction based on variant
            if variant == 'enhanced':
                prediction = enhanced_prediction_engine.predict_game(
                    home_team, away_team, season, week, game_context
                )
            else:
                # Use original prediction system
                prediction = self._make_original_prediction(
                    home_team, away_team, season, week, game_context
                )
            
            # Calculate response time
            response_time = (datetime.now() - start_time).total_seconds()
            
            # Track prediction
            self._track_prediction(variant, prediction, response_time, {
                'home_team': home_team,
                'away_team': away_team,
                'season': season,
                'week': week,
                'user_id': user_id
            })
            
            return prediction, variant
            
        except Exception as e:
            logger.error(f"Error in A/B testing prediction: {e}")
            # Fallback to original prediction
            prediction = self._make_original_prediction(home_team, away_team, season, week, game_context)
            return prediction, 'original'
    
    def _make_original_prediction(self, home_team: str, away_team: str, season: int, week: int,
                                game_context: GameContext) -> GamePrediction:
        """Make prediction using original 28-feature system."""
        try:
            # This would integrate with your existing prediction system
            # For now, return a placeholder prediction
            return GamePrediction(
                predicted_winner=home_team,
                confidence=0.55,
                upset_potential=0.30,
                predicted_score="21-17",
                ai_analysis="Original model prediction",
                key_factors=["Home field advantage", "Basic team stats"],
                is_upset=False
            )
            
        except Exception as e:
            logger.error(f"Error making original prediction: {e}")
            # Ultimate fallback
            return GamePrediction(
                predicted_winner=home_team,
                confidence=0.50,
                upset_potential=0.50,
                predicted_score="20-17",
                ai_analysis="Basic prediction",
                key_factors=["Home field advantage"],
                is_upset=False
            )
    
    def _track_prediction(self, variant: str, prediction: GamePrediction, response_time: float,
                         metadata: Dict[str, Any]):
        """Track prediction for A/B testing analysis."""
        try:
            # Create prediction record
            record = {
                'timestamp': datetime.now().isoformat(),
                'variant': variant,
                'predicted_winner': prediction.predicted_winner,
                'confidence': prediction.confidence,
                'upset_potential': prediction.upset_potential,
                'response_time': response_time,
                'metadata': metadata
            }
            
            # Add to results
            self.results[variant].append(record)
            
            # Update metrics
            self._update_metrics(variant, prediction, response_time)
            
        except Exception as e:
            logger.error(f"Error tracking prediction: {e}")
    
    def _update_metrics(self, variant: str, prediction: GamePrediction, response_time: float):
        """Update performance metrics for a variant."""
        try:
            metrics = self.metrics[variant]
            
            # Update counts
            metrics['total_predictions'] += 1
            
            # Update accuracy (would need actual game results)
            # For now, we'll track confidence and response time
            metrics['avg_confidence'] = (
                (metrics['avg_confidence'] * (metrics['total_predictions'] - 1) + prediction.confidence) /
                metrics['total_predictions']
            )
            
            metrics['avg_response_time'] = (
                (metrics['avg_response_time'] * (metrics['total_predictions'] - 1) + response_time) /
                metrics['total_predictions']
            )
            
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
    
    def record_actual_result(self, home_team: str, away_team: str, season: int, week: int,
                           actual_winner: str, actual_score: str, user_id: str = None):
        """
        Record actual game result to update accuracy metrics.
        
        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            season: NFL season year
            week: Week number
            actual_winner: Actual winning team
            actual_score: Actual final score
            user_id: User identifier for consistent assignment
        """
        try:
            # Find the prediction record
            game_id = f"{home_team}_{away_team}_{season}_{week}"
            
            for variant in ['original', 'enhanced']:
                for record in self.results[variant]:
                    if (record['metadata'].get('home_team') == home_team and
                        record['metadata'].get('away_team') == away_team and
                        record['metadata'].get('season') == season and
                        record['metadata'].get('week') == week):
                        
                        # Check if prediction was correct
                        predicted_winner = record['predicted_winner']
                        is_correct = predicted_winner == actual_winner
                        
                        # Update accuracy
                        if is_correct:
                            self.metrics[variant]['correct_predictions'] += 1
                        
                        # Calculate accuracy
                        total = self.metrics[variant]['total_predictions']
                        correct = self.metrics[variant]['correct_predictions']
                        self.metrics[variant]['accuracy'] = correct / total if total > 0 else 0.0
                        
                        # Mark as evaluated
                        record['actual_winner'] = actual_winner
                        record['actual_score'] = actual_score
                        record['is_correct'] = is_correct
                        record['evaluated_at'] = datetime.now().isoformat()
                        
                        break
            
        except Exception as e:
            logger.error(f"Error recording actual result: {e}")
    
    def get_test_results(self) -> Dict[str, Any]:
        """Get current A/B test results and statistics."""
        try:
            # Calculate statistical significance
            significance = self._calculate_statistical_significance()
            
            # Calculate confidence intervals
            confidence_intervals = self._calculate_confidence_intervals()
            
            # Calculate effect size
            effect_size = self._calculate_effect_size()
            
            return {
                'test_name': self.test_name,
                'start_date': self.start_date.isoformat(),
                'duration_days': (datetime.now() - self.start_date).days,
                'traffic_allocation': self.traffic_allocation,
                'metrics': self.metrics,
                'statistical_significance': significance,
                'confidence_intervals': confidence_intervals,
                'effect_size': effect_size,
                'recommendation': self._get_recommendation()
            }
            
        except Exception as e:
            logger.error(f"Error getting test results: {e}")
            return {}
    
    def _calculate_statistical_significance(self) -> Dict[str, Any]:
        """Calculate statistical significance of the A/B test."""
        try:
            # Get sample sizes
            n_original = self.metrics['original']['total_predictions']
            n_enhanced = self.metrics['enhanced']['total_predictions']
            
            if n_original < 30 or n_enhanced < 30:
                return {'significant': False, 'reason': 'Insufficient sample size'}
            
            # Get accuracies
            p_original = self.metrics['original']['accuracy']
            p_enhanced = self.metrics['enhanced']['accuracy']
            
            # Calculate pooled proportion
            p_pooled = (self.metrics['original']['correct_predictions'] + 
                       self.metrics['enhanced']['correct_predictions']) / (n_original + n_enhanced)
            
            # Calculate standard error
            se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_original + 1/n_enhanced))
            
            # Calculate z-score
            z_score = (p_enhanced - p_original) / se if se > 0 else 0
            
            # Calculate p-value (simplified)
            p_value = 2 * (1 - self._normal_cdf(abs(z_score)))
            
            return {
                'significant': p_value < 0.05,
                'p_value': p_value,
                'z_score': z_score,
                'confidence_level': 0.95
            }
            
        except Exception as e:
            logger.error(f"Error calculating statistical significance: {e}")
            return {'significant': False, 'reason': 'Calculation error'}
    
    def _calculate_confidence_intervals(self) -> Dict[str, Any]:
        """Calculate confidence intervals for accuracy metrics."""
        try:
            intervals = {}
            
            for variant in ['original', 'enhanced']:
                n = self.metrics[variant]['total_predictions']
                p = self.metrics[variant]['accuracy']
                
                if n > 0:
                    # 95% confidence interval
                    se = np.sqrt(p * (1 - p) / n)
                    margin_error = 1.96 * se
                    
                    intervals[variant] = {
                        'lower': max(0, p - margin_error),
                        'upper': min(1, p + margin_error),
                        'point_estimate': p
                    }
                else:
                    intervals[variant] = {
                        'lower': 0,
                        'upper': 1,
                        'point_estimate': 0
                    }
            
            return intervals
            
        except Exception as e:
            logger.error(f"Error calculating confidence intervals: {e}")
            return {}
    
    def _calculate_effect_size(self) -> Dict[str, float]:
        """Calculate effect size (Cohen's h) for the A/B test."""
        try:
            p_original = self.metrics['original']['accuracy']
            p_enhanced = self.metrics['enhanced']['accuracy']
            
            # Cohen's h
            h = 2 * (np.arcsin(np.sqrt(p_enhanced)) - np.arcsin(np.sqrt(p_original)))
            
            # Effect size interpretation
            if abs(h) < 0.2:
                interpretation = 'small'
            elif abs(h) < 0.5:
                interpretation = 'medium'
            else:
                interpretation = 'large'
            
            return {
                'cohens_h': h,
                'interpretation': interpretation,
                'improvement_percentage': ((p_enhanced - p_original) / p_original * 100) if p_original > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error calculating effect size: {e}")
            return {'cohens_h': 0, 'interpretation': 'unknown', 'improvement_percentage': 0}
    
    def _get_recommendation(self) -> str:
        """Get recommendation based on test results."""
        try:
            significance = self._calculate_statistical_significance()
            effect_size = self._calculate_effect_size()
            
            if not significance.get('significant', False):
                return 'Continue testing - insufficient evidence'
            
            if effect_size.get('improvement_percentage', 0) > 5:
                return 'Switch to enhanced model - significant improvement'
            elif effect_size.get('improvement_percentage', 0) < -5:
                return 'Keep original model - enhanced model performs worse'
            else:
                return 'Models perform similarly - consider other factors'
                
        except Exception as e:
            logger.error(f"Error getting recommendation: {e}")
            return 'Unable to determine recommendation'
    
    def _normal_cdf(self, x: float) -> float:
        """Approximate normal CDF for statistical calculations."""
        return 0.5 * (1 + np.tanh(0.79788456 * x))
    
    def adjust_traffic_allocation(self, original_ratio: float, enhanced_ratio: float):
        """Adjust traffic allocation between variants."""
        try:
            if abs(original_ratio + enhanced_ratio - 1.0) > 0.01:
                raise ValueError("Traffic allocation ratios must sum to 1.0")
            
            self.traffic_allocation = {
                'original': original_ratio,
                'enhanced': enhanced_ratio
            }
            
            logger.info(f"Traffic allocation updated: {self.traffic_allocation}")
            
        except Exception as e:
            logger.error(f"Error adjusting traffic allocation: {e}")
    
    def save_results(self):
        """Save A/B test results to file."""
        try:
            results = self.get_test_results()
            
            # Save to JSON file
            results_file = self.results_dir / f"{self.test_name}_results.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Save raw data
            raw_data_file = self.results_dir / f"{self.test_name}_raw_data.json"
            with open(raw_data_file, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            
            logger.info(f"A/B test results saved to {results_file}")
            
        except Exception as e:
            logger.error(f"Error saving A/B test results: {e}")


# Global A/B testing framework instance
ab_testing_framework = ABTestingFramework()
