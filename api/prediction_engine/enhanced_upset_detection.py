"""
Enhanced Upset Detection System
Uses advanced features to better identify potential upsets
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

logger = logging.getLogger(__name__)


class EnhancedUpsetDetector:
    """
    Enhanced upset detection using advanced features and machine learning.
    
    Features:
    - Multi-feature upset prediction
    - Confidence-based upset scoring
    - Historical upset pattern analysis
    - Real-time upset risk assessment
    """
    
    def __init__(self, models_dir: str = "api/prediction_engine/models/upset"):
        """Initialize the enhanced upset detector."""
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Upset detection models
        self.upset_models = {}
        self.feature_scaler = None
        self.upset_thresholds = {
            'low': 0.3,
            'medium': 0.5,
            'high': 0.7
        }
        
        # Historical upset data
        self.historical_upsets = []
        self.upset_patterns = {}
        
        # Feature importance for upset detection
        self.upset_feature_importance = {}
        
        logger.info("EnhancedUpsetDetector initialized")
    
    def detect_upset_potential(self, home_team: str, away_team: str, season: int, week: int,
                             game_context: Dict[str, Any], features: List[float]) -> Dict[str, Any]:
        """
        Detect potential upset using enhanced features.
        
        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            season: NFL season year
            week: Week number
            game_context: Game context information
            features: Enhanced feature vector
            
        Returns:
            Dictionary with upset detection results
        """
        try:
            # 1. Calculate base upset probability
            base_upset_prob = self._calculate_base_upset_probability(features)
            
            # 2. Apply contextual adjustments
            contextual_upset_prob = self._apply_contextual_adjustments(
                base_upset_prob, game_context, home_team, away_team
            )
            
            # 3. Apply historical pattern adjustments
            pattern_adjusted_prob = self._apply_historical_patterns(
                contextual_upset_prob, home_team, away_team, season, week
            )
            
            # 4. Calculate confidence in upset prediction
            upset_confidence = self._calculate_upset_confidence(
                pattern_adjusted_prob, features, game_context
            )
            
            # 5. Determine upset risk level
            risk_level = self._determine_upset_risk_level(pattern_adjusted_prob)
            
            # 6. Generate upset factors
            upset_factors = self._identify_upset_factors(
                features, game_context, home_team, away_team
            )
            
            # 7. Calculate upset score (0-100)
            upset_score = int(pattern_adjusted_prob * 100)
            
            return {
                'upset_probability': pattern_adjusted_prob,
                'upset_score': upset_score,
                'risk_level': risk_level,
                'confidence': upset_confidence,
                'upset_factors': upset_factors,
                'is_upset_likely': pattern_adjusted_prob > self.upset_thresholds['medium'],
                'upset_alert': pattern_adjusted_prob > self.upset_thresholds['high'],
                'analysis': self._generate_upset_analysis(
                    pattern_adjusted_prob, risk_level, upset_factors, home_team, away_team
                )
            }
            
        except Exception as e:
            logger.error(f"Error detecting upset potential: {e}")
            return self._create_fallback_upset_detection()
    
    def _calculate_base_upset_probability(self, features: List[float]) -> float:
        """Calculate base upset probability using feature analysis."""
        try:
            if len(features) < 17:
                return 0.3  # Default moderate risk
            
            # Extract key upset indicators from features
            # Features: [home_epa, away_epa, home_success_rate, away_success_rate, ...]
            
            # 1. EPA differential (indicates team strength difference)
            home_epa = features[0] if len(features) > 0 else 0.0
            away_epa = features[17] if len(features) > 17 else 0.0  # Away team EPA
            epa_differential = abs(home_epa - away_epa)
            
            # 2. Success rate differential
            home_success = features[2] if len(features) > 2 else 0.5
            away_success = features[19] if len(features) > 19 else 0.5  # Away team success rate
            success_differential = abs(home_success - away_success)
            
            # 3. Recent form trends
            home_form = features[13] if len(features) > 13 else 0.0
            away_form = features[30] if len(features) > 30 else 0.0  # Away team form
            form_differential = abs(home_form - away_form)
            
            # 4. Situational efficiency differences
            home_red_zone = features[6] if len(features) > 6 else 0.5
            away_red_zone = features[23] if len(features) > 23 else 0.5  # Away team red zone
            red_zone_differential = abs(home_red_zone - away_red_zone)
            
            # Calculate base upset probability
            # Higher differentials indicate more potential for upsets
            base_prob = 0.2  # Base 20% upset probability
            
            # EPA differential contribution (0-0.3)
            if epa_differential < 0.1:  # Teams are close in EPA
                base_prob += 0.15
            elif epa_differential < 0.2:
                base_prob += 0.1
            
            # Success rate differential contribution (0-0.2)
            if success_differential < 0.1:  # Teams are close in success rate
                base_prob += 0.1
            elif success_differential < 0.2:
                base_prob += 0.05
            
            # Form differential contribution (0-0.2)
            if form_differential > 0.2:  # One team trending up, other down
                base_prob += 0.1
            elif form_differential > 0.1:
                base_prob += 0.05
            
            # Red zone differential contribution (0-0.1)
            if red_zone_differential < 0.1:  # Teams similar in red zone
                base_prob += 0.05
            
            return min(0.8, max(0.1, base_prob))  # Clamp between 10% and 80%
            
        except Exception as e:
            logger.error(f"Error calculating base upset probability: {e}")
            return 0.3
    
    def _apply_contextual_adjustments(self, base_prob: float, game_context: Dict[str, Any],
                                    home_team: str, away_team: str) -> float:
        """Apply contextual adjustments to upset probability."""
        try:
            adjusted_prob = base_prob
            
            # Weather impact
            weather_factor = game_context.get('weather_factor', 0.0)
            if weather_factor > 0.7:  # Severe weather
                adjusted_prob += 0.1  # Weather creates more unpredictability
            elif weather_factor < 0.3:  # Good weather
                adjusted_prob -= 0.05  # Good weather favors favorites
            
            # Rest advantage
            rest_differential = game_context.get('rest_differential', 0.0)
            if abs(rest_differential) > 3:  # Significant rest advantage
                adjusted_prob += 0.05  # Rest advantage can lead to upsets
            
            # Division game
            is_division_game = game_context.get('is_division_game', False)
            if is_division_game:
                adjusted_prob += 0.1  # Division games are more unpredictable
            
            # Playoff implications
            playoff_implications = game_context.get('playoff_implications', 0.0)
            if playoff_implications > 0.7:  # High stakes
                adjusted_prob += 0.05  # High stakes can lead to upsets
            
            # Home field advantage
            home_advantage = game_context.get('home_advantage', 0.1)
            if home_advantage < 0.05:  # Weak home field advantage
                adjusted_prob += 0.05
            
            # Travel distance
            travel_distance = game_context.get('travel_distance', 0.0)
            if travel_distance > 2000:  # Long travel
                adjusted_prob += 0.03  # Travel fatigue can lead to upsets
            
            return min(0.9, max(0.05, adjusted_prob))  # Clamp between 5% and 90%
            
        except Exception as e:
            logger.error(f"Error applying contextual adjustments: {e}")
            return base_prob
    
    def _apply_historical_patterns(self, contextual_prob: float, home_team: str, away_team: str,
                                 season: int, week: int) -> float:
        """Apply historical upset patterns to adjust probability."""
        try:
            adjusted_prob = contextual_prob
            
            # Check historical upset patterns
            team_upset_rate = self._get_team_upset_rate(home_team, away_team)
            if team_upset_rate > 0.4:  # High upset rate between these teams
                adjusted_prob += 0.1
            elif team_upset_rate < 0.2:  # Low upset rate
                adjusted_prob -= 0.05
            
            # Check week-specific patterns
            week_upset_rate = self._get_week_upset_rate(week)
            if week_upset_rate > 0.4:  # High upset rate in this week
                adjusted_prob += 0.05
            elif week_upset_rate < 0.2:  # Low upset rate
                adjusted_prob -= 0.03
            
            # Check season progression patterns
            season_upset_rate = self._get_season_upset_rate(season, week)
            if season_upset_rate > 0.4:  # High upset rate this season
                adjusted_prob += 0.05
            
            return min(0.9, max(0.05, adjusted_prob))
            
        except Exception as e:
            logger.error(f"Error applying historical patterns: {e}")
            return contextual_prob
    
    def _calculate_upset_confidence(self, upset_prob: float, features: List[float],
                                  game_context: Dict[str, Any]) -> float:
        """Calculate confidence in the upset prediction."""
        try:
            confidence = 0.5  # Base confidence
            
            # Feature consistency
            feature_variance = np.var(features) if len(features) > 0 else 0.0
            if feature_variance < 0.1:  # Low variance - consistent features
                confidence += 0.2
            elif feature_variance > 0.3:  # High variance - inconsistent features
                confidence -= 0.1
            
            # Upset probability magnitude
            if 0.3 <= upset_prob <= 0.7:  # Moderate probability - more confident
                confidence += 0.1
            elif upset_prob < 0.2 or upset_prob > 0.8:  # Extreme probabilities - less confident
                confidence -= 0.1
            
            # Context clarity
            context_factors = len([v for v in game_context.values() if v is not None])
            if context_factors > 5:  # Rich context
                confidence += 0.1
            elif context_factors < 3:  # Limited context
                confidence -= 0.1
            
            return min(1.0, max(0.0, confidence))
            
        except Exception as e:
            logger.error(f"Error calculating upset confidence: {e}")
            return 0.5
    
    def _determine_upset_risk_level(self, upset_prob: float) -> str:
        """Determine the risk level for an upset."""
        try:
            if upset_prob >= self.upset_thresholds['high']:
                return 'high'
            elif upset_prob >= self.upset_thresholds['medium']:
                return 'medium'
            elif upset_prob >= self.upset_thresholds['low']:
                return 'low'
            else:
                return 'minimal'
                
        except Exception as e:
            logger.error(f"Error determining upset risk level: {e}")
            return 'unknown'
    
    def _identify_upset_factors(self, features: List[float], game_context: Dict[str, Any],
                              home_team: str, away_team: str) -> List[str]:
        """Identify specific factors that contribute to upset potential."""
        try:
            factors = []
            
            # Check feature-based factors
            if len(features) >= 17:
                # Close EPA differential
                home_epa = features[0]
                away_epa = features[17] if len(features) > 17 else 0.0
                if abs(home_epa - away_epa) < 0.1:
                    factors.append("Teams have similar offensive efficiency")
                
                # Close success rates
                home_success = features[2]
                away_success = features[19] if len(features) > 19 else 0.5
                if abs(home_success - away_success) < 0.1:
                    factors.append("Teams have similar success rates")
                
                # Form trends
                home_form = features[13]
                away_form = features[30] if len(features) > 30 else 0.0
                if abs(home_form - away_form) > 0.2:
                    factors.append("Significant difference in recent form")
            
            # Check contextual factors
            weather_factor = game_context.get('weather_factor', 0.0)
            if weather_factor > 0.7:
                factors.append("Severe weather conditions")
            
            is_division_game = game_context.get('is_division_game', False)
            if is_division_game:
                factors.append("Division rivalry game")
            
            playoff_implications = game_context.get('playoff_implications', 0.0)
            if playoff_implications > 0.7:
                factors.append("High playoff implications")
            
            rest_differential = game_context.get('rest_differential', 0.0)
            if abs(rest_differential) > 3:
                factors.append("Significant rest advantage")
            
            return factors if factors else ["Standard game factors"]
            
        except Exception as e:
            logger.error(f"Error identifying upset factors: {e}")
            return ["Unable to identify specific factors"]
    
    def _generate_upset_analysis(self, upset_prob: float, risk_level: str, upset_factors: List[str],
                               home_team: str, away_team: str) -> str:
        """Generate human-readable upset analysis."""
        try:
            analysis_parts = []
            
            # Risk level analysis
            if risk_level == 'high':
                analysis_parts.append(f"🚨 HIGH UPSET RISK: {upset_prob:.1%} chance of upset")
            elif risk_level == 'medium':
                analysis_parts.append(f"⚠️ MODERATE UPSET RISK: {upset_prob:.1%} chance of upset")
            elif risk_level == 'low':
                analysis_parts.append(f"📊 LOW UPSET RISK: {upset_prob:.1%} chance of upset")
            else:
                analysis_parts.append(f"✅ MINIMAL UPSET RISK: {upset_prob:.1%} chance of upset")
            
            # Key factors
            if upset_factors:
                analysis_parts.append(f"Key factors: {', '.join(upset_factors[:3])}")
            
            # Team-specific analysis
            if upset_prob > 0.5:
                analysis_parts.append(f"{away_team} has significant upset potential against {home_team}")
            else:
                analysis_parts.append(f"{home_team} is favored to win at home")
            
            return " | ".join(analysis_parts)
            
        except Exception as e:
            logger.error(f"Error generating upset analysis: {e}")
            return f"Upset probability: {upset_prob:.1%}"
    
    def _get_team_upset_rate(self, home_team: str, away_team: str) -> float:
        """Get historical upset rate between specific teams."""
        try:
            # This would query historical data
            # For now, return a default value
            return 0.3
            
        except Exception as e:
            logger.error(f"Error getting team upset rate: {e}")
            return 0.3
    
    def _get_week_upset_rate(self, week: int) -> float:
        """Get historical upset rate for a specific week."""
        try:
            # This would query historical data
            # For now, return a default value
            return 0.3
            
        except Exception as e:
            logger.error(f"Error getting week upset rate: {e}")
            return 0.3
    
    def _get_season_upset_rate(self, season: int, week: int) -> float:
        """Get historical upset rate for a specific season and week."""
        try:
            # This would query historical data
            # For now, return a default value
            return 0.3
            
        except Exception as e:
            logger.error(f"Error getting season upset rate: {e}")
            return 0.3
    
    def _create_fallback_upset_detection(self) -> Dict[str, Any]:
        """Create fallback upset detection when main system fails."""
        return {
            'upset_probability': 0.3,
            'upset_score': 30,
            'risk_level': 'medium',
            'confidence': 0.5,
            'upset_factors': ['Standard game factors'],
            'is_upset_likely': False,
            'upset_alert': False,
            'analysis': 'Fallback upset detection - limited analysis available'
        }
    
    def train_upset_models(self, training_data: List[Dict[str, Any]]) -> bool:
        """Train upset detection models using historical data."""
        try:
            if not training_data:
                logger.warning("No training data provided for upset models")
                return False
            
            logger.info(f"Training upset models with {len(training_data)} samples")
            
            # Prepare training data
            X = []
            y = []
            
            for record in training_data:
                features = record.get('features', [])
                is_upset = record.get('is_upset', False)
                
                if len(features) >= 17:  # Ensure we have enough features
                    X.append(features)
                    y.append(1 if is_upset else 0)
            
            if len(X) < 100:
                logger.warning("Insufficient training data for upset models")
                return False
            
            X = np.array(X)
            y = np.array(y)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            self.feature_scaler = StandardScaler()
            X_train_scaled = self.feature_scaler.fit_transform(X_train)
            X_test_scaled = self.feature_scaler.transform(X_test)
            
            # Train models
            models = {
                'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
                'random_forest': RandomForestClassifier(random_state=42, n_estimators=100)
            }
            
            for name, model in models.items():
                model.fit(X_train_scaled, y_train)
                self.upset_models[name] = model
                
                # Evaluate model
                y_pred = model.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                
                logger.info(f"Upset {name} - Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, "
                          f"Recall: {recall:.3f}, F1: {f1:.3f}")
            
            # Save models
            self._save_upset_models()
            
            logger.info("Upset models trained successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error training upset models: {e}")
            return False
    
    def _save_upset_models(self):
        """Save trained upset models."""
        try:
            import joblib
            
            for name, model in self.upset_models.items():
                model_path = self.models_dir / f"upset_{name}.joblib"
                joblib.dump(model, model_path)
                logger.info(f"Saved upset {name} model to {model_path}")
            
            if self.feature_scaler is not None:
                scaler_path = self.models_dir / "upset_feature_scaler.joblib"
                joblib.dump(self.feature_scaler, scaler_path)
                logger.info(f"Saved upset feature scaler to {scaler_path}")
            
        except Exception as e:
            logger.error(f"Error saving upset models: {e}")


# Global enhanced upset detector instance
enhanced_upset_detector = EnhancedUpsetDetector()
