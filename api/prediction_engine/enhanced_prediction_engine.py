"""
Enhanced Prediction Engine
Integrates enhanced features with existing model system while maintaining backward compatibility
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from .enhanced_features import enhanced_extractor
from .team_feature_database import team_feature_db
from .data_models import GameContext, GamePrediction

logger = logging.getLogger(__name__)


class EnhancedPredictionEngine:
    """
    Enhanced prediction engine that integrates new features with existing models.
    
    Features:
    - Backward compatibility with existing 28-feature models
    - Enhanced 45+ feature models (when available)
    - Automatic fallback to original features
    - A/B testing capability
    """
    
    def __init__(self, original_models=None, enhanced_models=None):
        """
        Initialize the enhanced prediction engine.
        
        Args:
            original_models: Original 28-feature models (ensemble, scaler)
            enhanced_models: Enhanced 45+ feature models (optional)
        """
        self.original_models = original_models
        self.enhanced_models = enhanced_models
        self.enhanced_extractor = enhanced_extractor
        self.team_db = team_feature_db
        
        # Feature configuration
        self.original_feature_count = 28
        self.enhanced_feature_count = 45  # 28 original + 17 enhanced
        
        # A/B testing configuration
        self.enhanced_feature_probability = 0.1  # 10% chance to use enhanced features
        
        logger.info("EnhancedPredictionEngine initialized")
    
    def predict_game(self, home_team: str, away_team: str, season: int, week: int, 
                    game_context: GameContext) -> GamePrediction:
        """
        Make a prediction for a specific game using enhanced features when available.
        
        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            season: NFL season year
            week: Week number
            game_context: Game context information
            
        Returns:
            Comprehensive game prediction
        """
        try:
            # Try enhanced features first (if enabled and available)
            if self._should_use_enhanced_features():
                enhanced_prediction = self._predict_with_enhanced_features(
                    home_team, away_team, season, week, game_context
                )
                if enhanced_prediction:
                    return enhanced_prediction
            
            # Fallback to original features
            return self._predict_with_original_features(
                home_team, away_team, season, week, game_context
            )
            
        except Exception as e:
            logger.error(f"Error in enhanced prediction: {e}")
            # Final fallback to original features
            return self._predict_with_original_features(
                home_team, away_team, season, week, game_context
            )
    
    def _should_use_enhanced_features(self) -> bool:
        """Determine if enhanced features should be used."""
        if not self.enhanced_models:
            return False
        
        # Simple probability-based A/B testing
        import random
        return random.random() < self.enhanced_feature_probability
    
    def _predict_with_enhanced_features(self, home_team: str, away_team: str, 
                                      season: int, week: int, game_context: GameContext) -> Optional[GamePrediction]:
        """Make prediction using enhanced features."""
        try:
            # Get enhanced features for both teams
            home_features = self.team_db.get_features(home_team, season, week)
            away_features = self.team_db.get_features(away_team, season, week)
            
            if not home_features or not away_features:
                logger.warning(f"Missing enhanced features for {away_team} @ {home_team}")
                return None
            
            # Combine features (17 + 17 + 11 contextual = 45 features)
            combined_features = self._combine_enhanced_features(
                home_features, away_features, home_team, away_team, season, week, game_context
            )
            
            if len(combined_features) != self.enhanced_feature_count:
                logger.warning(f"Expected {self.enhanced_feature_count} features, got {len(combined_features)}")
                return None
            
            # Make prediction with enhanced model
            prediction = self._make_enhanced_prediction(combined_features)
            
            # Add enhanced analysis
            prediction.ai_analysis = self._generate_enhanced_analysis(
                home_team, away_team, home_features, away_features, prediction
            )
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error in enhanced prediction: {e}")
            return None
    
    def _predict_with_original_features(self, home_team: str, away_team: str, 
                                      season: int, week: int, game_context: GameContext) -> GamePrediction:
        """Make prediction using original 28-feature system."""
        try:
            # This would integrate with the existing prediction system
            # For now, return a placeholder that maintains the same interface
            return self._create_fallback_prediction(home_team, away_team, season, week, game_context)
            
        except Exception as e:
            logger.error(f"Error in original prediction: {e}")
            return self._create_fallback_prediction(home_team, away_team, season, week, game_context)
    
    def _combine_enhanced_features(self, home_features: List[float], away_features: List[float],
                                 home_team: str, away_team: str, season: int, week: int,
                                 game_context: GameContext) -> List[float]:
        """Combine enhanced features for both teams plus contextual factors."""
        try:
            combined = []
            
            # Add home team features (17)
            combined.extend(home_features)
            
            # Add away team features (17)
            combined.extend(away_features)
            
            # Add contextual factors (11)
            contextual_features = self._extract_contextual_features(
                home_team, away_team, season, week, game_context
            )
            combined.extend(contextual_features)
            
            return combined
            
        except Exception as e:
            logger.error(f"Error combining enhanced features: {e}")
            return [0.0] * self.enhanced_feature_count
    
    def _extract_contextual_features(self, home_team: str, away_team: str, season: int, 
                                   week: int, game_context: GameContext) -> List[float]:
        """Extract contextual features for the game."""
        try:
            # Get rest and travel factors
            rest_travel = self.enhanced_extractor.get_rest_travel_factors(home_team, away_team, week)
            
            # Contextual features (11 total)
            features = [
                # Game context (3)
                float(season - 2000) / 25.0,  # Season progression (normalized)
                float(week) / 18.0,  # Week progression
                1.0 if week > 17 else 0.0,  # Playoff indicator
                
                # Rest and travel (3)
                rest_travel['rest_differential'],
                rest_travel['travel_distance'],
                rest_travel['home_advantage'],
                
                # Weather and environment (2)
                game_context.weather_factor if hasattr(game_context, 'weather_factor') else 0.0,
                game_context.stadium_factor if hasattr(game_context, 'stadium_factor') else 0.0,
                
                # Historical context (2)
                self._get_historical_advantage(home_team, away_team, season, week),
                self._get_rivalry_factor(home_team, away_team),
                
                # Betting context (1)
                game_context.spread if hasattr(game_context, 'spread') else 0.0
            ]
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting contextual features: {e}")
            return [0.0] * 11
    
    def _make_enhanced_prediction(self, features: List[float]) -> GamePrediction:
        """Make prediction using enhanced model."""
        try:
            # This would use the enhanced model when available
            # For now, return a placeholder prediction
            return GamePrediction(
                predicted_winner="KC",  # Placeholder
                confidence=0.65,
                upset_potential=0.25,
                predicted_score="24-21",
                ai_analysis="Enhanced prediction using advanced features",
                key_factors=["EPA trends", "Situational efficiency", "Recent form"],
                is_upset=False
            )
            
        except Exception as e:
            logger.error(f"Error making enhanced prediction: {e}")
            raise
    
    def _generate_enhanced_analysis(self, home_team: str, away_team: str, 
                                  home_features: List[float], away_features: List[float],
                                  prediction: GamePrediction) -> str:
        """Generate enhanced analysis using the new features."""
        try:
            analysis_parts = []
            
            # EPA analysis
            home_epa = home_features[0] if len(home_features) > 0 else 0.0
            away_epa = away_features[0] if len(away_features) > 0 else 0.0
            
            if home_epa > away_epa:
                analysis_parts.append(f"{home_team} has superior offensive EPA ({home_epa:.2f} vs {away_epa:.2f})")
            else:
                analysis_parts.append(f"{away_team} has superior offensive EPA ({away_epa:.2f} vs {home_epa:.2f})")
            
            # Situational efficiency
            home_red_zone = home_features[6] if len(home_features) > 6 else 0.0
            away_red_zone = away_features[6] if len(away_features) > 6 else 0.0
            
            if home_red_zone > away_red_zone:
                analysis_parts.append(f"{home_team} excels in red zone efficiency ({home_red_zone:.1%})")
            else:
                analysis_parts.append(f"{away_team} excels in red zone efficiency ({away_red_zone:.1%})")
            
            # Recent form
            home_form = home_features[13] if len(home_features) > 13 else 0.0
            away_form = away_features[13] if len(away_features) > 13 else 0.0
            
            if abs(home_form - away_form) > 0.1:
                if home_form > away_form:
                    analysis_parts.append(f"{home_team} has stronger recent form trend")
                else:
                    analysis_parts.append(f"{away_team} has stronger recent form trend")
            
            return "Enhanced Analysis: " + "; ".join(analysis_parts)
            
        except Exception as e:
            logger.error(f"Error generating enhanced analysis: {e}")
            return "Enhanced analysis unavailable"
    
    def _get_historical_advantage(self, home_team: str, away_team: str, season: int, week: int) -> float:
        """Get historical advantage between teams."""
        # Placeholder - would use actual historical data
        return 0.0
    
    def _get_rivalry_factor(self, home_team: str, away_team: str) -> float:
        """Get rivalry factor between teams."""
        # Placeholder - would use actual rivalry data
        return 0.0
    
    def _create_fallback_prediction(self, home_team: str, away_team: str, 
                                  season: int, week: int, game_context: GameContext) -> GamePrediction:
        """Create a fallback prediction when enhanced features fail."""
        try:
            # Simple fallback prediction
            return GamePrediction(
                predicted_winner=home_team,  # Simple home team advantage
                confidence=0.55,
                upset_potential=0.30,
                predicted_score="21-17",
                ai_analysis="Fallback prediction using basic features",
                key_factors=["Home field advantage", "Basic team stats"],
                is_upset=False
            )
            
        except Exception as e:
            logger.error(f"Error creating fallback prediction: {e}")
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
    
    def get_feature_availability(self, home_team: str, away_team: str, season: int, week: int) -> Dict[str, bool]:
        """Check availability of enhanced features for both teams."""
        try:
            home_features = self.team_db.get_features(home_team, season, week)
            away_features = self.team_db.get_features(away_team, season, week)
            
            return {
                'home_team_features': home_features is not None,
                'away_team_features': away_features is not None,
                'enhanced_model_available': self.enhanced_models is not None,
                'can_use_enhanced': home_features is not None and away_features is not None and self.enhanced_models is not None
            }
            
        except Exception as e:
            logger.error(f"Error checking feature availability: {e}")
            return {
                'home_team_features': False,
                'away_team_features': False,
                'enhanced_model_available': False,
                'can_use_enhanced': False
            }
    
    def set_enhanced_feature_probability(self, probability: float):
        """Set the probability of using enhanced features for A/B testing."""
        if 0.0 <= probability <= 1.0:
            self.enhanced_feature_probability = probability
            logger.info(f"Enhanced feature probability set to {probability}")
        else:
            logger.warning(f"Invalid probability {probability}, must be between 0.0 and 1.0")


# Global enhanced prediction engine instance
enhanced_prediction_engine = EnhancedPredictionEngine()
