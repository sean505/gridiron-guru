"""
Prediction Pipeline for Gridiron Guru
Generates predictions for 2025 games using 2024 baseline + 2025 schedule
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import joblib
from datetime import datetime

from prediction_engine.data_models import GamePrediction, GameContext
from temporal_pipeline.baseline_pipeline import baseline_pipeline
from temporal_pipeline.temporal_feature_engine import temporal_feature_engine
from temporal_pipeline.temporal_data_collector import temporal_data_collector

logger = logging.getLogger(__name__)


class PredictionPipeline:
    """
    Prediction pipeline that generates predictions for 2025 games.
    
    Features:
    - Loads 2025 schedule
    - Uses 2024 team strength baseline
    - Engineers prediction features
    - Generates predictions using trained models
    """
    
    def __init__(self, model_dir: str = "prediction_engine/models/trained"):
        """Initialize the prediction pipeline."""
        self.model_dir = Path(model_dir)
        self.baseline_pipeline = baseline_pipeline
        self.feature_engine = temporal_feature_engine
        self.data_collector = temporal_data_collector
        
        # Model storage
        self.models = {}
        self.scaler = None
        self.feature_names = []
        self.models_loaded = False
        
        logger.info(f"PredictionPipeline initialized with model directory: {self.model_dir}")
    
    def load_trained_models(self) -> bool:
        """
        Load trained models from disk.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.models_loaded:
                return True
            
            logger.info("Loading trained models")
            
            # Load ensemble model
            ensemble_path = self.model_dir / "ensemble.joblib"
            if ensemble_path.exists():
                self.models['ensemble'] = joblib.load(ensemble_path)
                logger.info("Loaded ensemble model")
            else:
                logger.warning("Ensemble model not found")
                return False
            
            # Load scaler
            scaler_path = self.model_dir / "feature_scaler.joblib"
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                logger.info("Loaded feature scaler")
            else:
                logger.warning("Feature scaler not found")
                return False
            
            # Load feature names
            feature_names_path = self.model_dir / "feature_names.json"
            if feature_names_path.exists():
                import json
                with open(feature_names_path, 'r') as f:
                    self.feature_names = json.load(f)
                logger.info(f"Loaded feature names: {len(self.feature_names)} features")
            else:
                logger.warning("Feature names not found")
                return False
            
            self.models_loaded = True
            return True
            
        except Exception as e:
            logger.error(f"Error loading trained models: {e}")
            return False
    
    def get_2025_games(self, week: Optional[int] = None) -> pd.DataFrame:
        """
        Get 2025 games for predictions.
        
        Args:
            week: Specific week to get (None for all weeks)
            
        Returns:
            DataFrame containing 2025 games
        """
        try:
            schedule = self.data_collector.get_2025_schedule()
            
            if week is not None:
                schedule = schedule[schedule['week'] == week]
            
            logger.info(f"Retrieved {len(schedule)} games for 2025")
            return schedule
            
        except Exception as e:
            logger.error(f"Error getting 2025 games: {e}")
            return pd.DataFrame()
    
    def generate_prediction(self, home_team: str, away_team: str, 
                          season: int, week: int) -> GamePrediction:
        """
        Generate prediction for a specific game.
        
        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            season: Season year (should be 2025)
            week: Week number
            
        Returns:
            GamePrediction object
        """
        try:
            # Load models if not already loaded
            if not self.models_loaded:
                if not self.load_trained_models():
                    raise ValueError("Failed to load trained models")
            
            # Get team strength from 2024 baseline
            home_strength = self.baseline_pipeline.get_team_strength_by_abbr(home_team)
            away_strength = self.baseline_pipeline.get_team_strength_by_abbr(away_team)
            
            if not home_strength or not away_strength:
                raise ValueError(f"Team strength not found for {home_team} or {away_team}")
            
            # Create game context
            game_context = self._create_game_context(home_team, away_team, season, week)
            
            # Engineer features
            features = self.feature_engine.create_prediction_features(
                home_strength, away_strength, game_context
            )
            
            # Prepare features for model
            feature_array = self._prepare_features_for_model(features)
            
            # Generate prediction
            prediction_proba = self.models['ensemble'].predict_proba(feature_array)
            home_win_prob = prediction_proba[0][1]  # Probability of home team winning
            
            # Create prediction object
            predicted_winner = home_team if home_win_prob > 0.5 else away_team
            confidence = max(home_win_prob, 1 - home_win_prob)
            upset_prob = 1 - confidence
            
            prediction = GamePrediction(
                game_id=f"{season}_{week}_{home_team}_{away_team}",
                home_team=home_team,
                away_team=away_team,
                season=season,
                week=week,
                predicted_winner=predicted_winner,
                confidence=confidence,  # Use 0-1 range
                win_probability=home_win_prob,
                predicted_home_score=24.0,  # Placeholder
                predicted_away_score=21.0,  # Placeholder
                predicted_total=45.0,  # Placeholder
                predicted_spread=3.0,  # Placeholder
                is_upset_pick=upset_prob > 0.25,  # Consider upset if > 25% chance
                upset_probability=upset_prob,
                upset_factors=self._generate_upset_factors(features, home_strength, away_strength),
                key_factors=self._generate_key_factors(features, home_strength, away_strength),
                explanation=self._generate_robust_ai_analysis(predicted_winner, home_team, away_team, confidence, upset_prob, features),
                data_freshness=datetime.now()
            )
            
            logger.info(f"Generated prediction: {home_team} vs {away_team} - {prediction.predicted_winner} ({prediction.confidence}%)")
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error generating prediction: {e}")
            # Return default prediction
            return GamePrediction(
                game_id=f"{season}_{week}_{home_team}_{away_team}",
                home_team=home_team,
                away_team=away_team,
                season=season,
                week=week,
                predicted_winner="N/A",
                confidence=0.0,
                win_probability=0.0,
                predicted_home_score=0.0,
                predicted_away_score=0.0,
                predicted_total=0.0,
                predicted_spread=0.0,
                key_factors=["N/A - Model unavailable"],
                explanation="N/A - Analysis not available",
                data_freshness=datetime.now()
            )
    
    def generate_predictions(self, games_df: pd.DataFrame) -> List[GamePrediction]:
        """
        Generate predictions for multiple games.
        
        Args:
            games_df: DataFrame containing games to predict
            
        Returns:
            List of GamePrediction objects
        """
        try:
            predictions = []
            
            for _, game in games_df.iterrows():
                home_team = game.get('home_team', '')
                away_team = game.get('away_team', '')
                season = game.get('season', 2025)
                week = game.get('week', 1)
                
                if home_team and away_team:
                    prediction = self.generate_prediction(home_team, away_team, season, week)
                    predictions.append(prediction)
            
            logger.info(f"Generated {len(predictions)} predictions")
            return predictions
            
        except Exception as e:
            logger.error(f"Error generating predictions: {e}")
            return []
    
    def _create_game_context(self, home_team: str, away_team: str, 
                           season: int, week: int) -> GameContext:
        """Create game context for prediction."""
        return GameContext(
            game_id=f"{season}_{week}_{home_team}_{away_team}",
            season=season,
            week=week,
            home_team=home_team,
            away_team=away_team,
            game_date=datetime.now(),
            game_type='REG',
            spread=None,
            total=None,
            temperature=None,
            wind_speed=None,
            precipitation=None,
            roof='outdoors',
            surface='grass'
        )
    
    def _prepare_features_for_model(self, features: Dict[str, float]) -> np.ndarray:
        """Prepare features for model prediction."""
        try:
            # Convert to array in correct order
            feature_array = np.array([features.get(name, 0.0) for name in self.feature_names])
            
            # Handle missing values
            feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Scale features
            if self.scaler:
                feature_array = self.scaler.transform(feature_array.reshape(1, -1))
            else:
                feature_array = feature_array.reshape(1, -1)
            
            return feature_array
            
        except Exception as e:
            logger.error(f"Error preparing features for model: {e}")
            return np.zeros((1, len(self.feature_names)))
    
    def _generate_key_factors(self, features: Dict[str, float], 
                            home_strength: Any, away_strength: Any) -> List[str]:
        """Generate key factors for the prediction."""
        try:
            factors = []
            
            # Win percentage difference
            win_pct_diff = features.get('win_pct_diff', 0)
            if abs(win_pct_diff) > 0.1:
                if win_pct_diff > 0:
                    factors.append(f"{home_strength.team_name} has significantly better win percentage")
                else:
                    factors.append(f"{away_strength.team_name} has significantly better win percentage")
            
            # Point differential difference
            point_diff_diff = features.get('point_diff_diff', 0)
            if abs(point_diff_diff) > 5:
                if point_diff_diff > 0:
                    factors.append(f"{home_strength.team_name} has much better point differential")
                else:
                    factors.append(f"{away_strength.team_name} has much better point differential")
            
            # Home field advantage
            if features.get('home_field_advantage', 0) > 0:
                factors.append("Home field advantage")
            
            # Recent form
            recent_form_diff = features.get('recent_form_diff', 0)
            if abs(recent_form_diff) > 0.2:
                if recent_form_diff > 0:
                    factors.append(f"{home_strength.team_name} has better recent form")
                else:
                    factors.append(f"{away_strength.team_name} has better recent form")
            
            # Head-to-head history
            h2h_win_pct = features.get('h2h_win_pct', 0.5)
            if h2h_win_pct > 0.6:
                factors.append(f"{home_strength.team_name} has strong head-to-head record")
            elif h2h_win_pct < 0.4:
                factors.append(f"{away_strength.team_name} has strong head-to-head record")
            
            return factors[:5]  # Return top 5 factors
            
        except Exception as e:
            logger.error(f"Error generating key factors: {e}")
            return ["Analysis unavailable"]
    
    def _generate_upset_factors(self, features: Dict[str, float], 
                              home_strength: Any, away_strength: Any) -> List[str]:
        """Generate upset factors for the prediction."""
        try:
            factors = []
            
            # Close strength difference suggests potential upset
            strength_diff = abs(features.get('strength_diff', 0))
            if strength_diff < 0.1:
                factors.append("Teams are closely matched")
            
            # High upset probability factors
            if features.get('upset_potential', 0) > 0.3:
                factors.append("High upset potential based on historical data")
            
            # Weather or situational factors
            if features.get('weather_impact', 0) > 0.2:
                factors.append("Weather conditions could be a factor")
            
            # Division rivalry
            if features.get('division_game', 0) > 0:
                factors.append("Divisional rivalry adds unpredictability")
            
            return factors[:3]  # Return top 3 upset factors
            
        except Exception as e:
            logger.error(f"Error generating upset factors: {e}")
            return ["Upset potential based on team performance"]
    
    def _generate_robust_ai_analysis(self, predicted_winner: str, home_team: str, away_team: str, 
                                   confidence: float, upset_prob: float, features: Dict[str, float]) -> str:
        """Generate comprehensive AI analysis similar to the image examples."""
        try:
            # Determine confidence level
            if confidence >= 0.75:
                confidence_level = "High Confidence"
                confidence_desc = "Strong evidence supports this prediction"
            elif confidence >= 0.60:
                confidence_level = "Moderate Confidence" 
                confidence_desc = "Solid evidence supports this prediction"
            else:
                confidence_level = "Low Confidence"
                confidence_desc = "Close matchup with multiple variables"
            
            # Determine upset risk level
            if upset_prob >= 0.40:
                upset_risk = "High Upset Risk"
                upset_desc = "Significant uncertainty in this prediction"
            elif upset_prob >= 0.25:
                upset_risk = "Moderate Upset Risk"
                upset_desc = "Some uncertainty in this prediction"
            else:
                upset_risk = "Low Upset Risk"
                upset_desc = "Prediction appears stable"
            
            # Generate team-specific strengths
            team_strengths = {
                'CIN': "Bengals' explosive offense and Burrow's precision",
                'DAL': "Cowboys' offensive firepower and home field advantage",
                'KC': "Chiefs' championship experience and Mahomes' playmaking",
                'BUF': "Bills' balanced attack and Allen's dual-threat ability",
                'SF': "49ers' physical defense and McCaffrey's versatility",
                'PHI': "Eagles' dominant offensive line and Hurts' mobility",
                'BAL': "Ravens' innovative offense and Jackson's athleticism",
                'MIA': "Dolphins' speed and Tua's accuracy",
                'GB': "Packers' tradition and Love's development",
                'DET': "Lions' resurgence and Goff's leadership",
                'LAR': "Rams' star power and McVay's creativity",
                'TB': "Buccaneers' veteran leadership and Mayfield's grit",
                'ATL': "Falcons' young talent and Ridder's potential",
                'NO': "Saints' defensive prowess and Carr's experience",
                'CAR': "Panthers' rebuilding effort and Young's promise",
                'NYG': "Giants' defensive identity and Jones' mobility",
                'WAS': "Commanders' defensive front and Howell's growth",
                'CHI': "Bears' young core and Fields' athleticism",
                'MIN': "Vikings' offensive weapons and Cousins' experience",
                'PIT': "Steelers' defensive tradition and Pickett's development",
                'CLE': "Browns' defensive strength and Watson's potential",
                'HOU': "Texans' young talent and Stroud's accuracy",
                'IND': "Colts' balanced approach and Richardson's upside",
                'JAX': "Jaguars' offensive firepower and Lawrence's growth",
                'TEN': "Titans' physical style and Tannehill's experience",
                'DEN': "Broncos' defensive identity and Wilson's leadership",
                'LV': "Raiders' offensive weapons and Garoppolo's experience",
                'LAC': "Chargers' offensive talent and Herbert's arm",
                'NYJ': "Jets' defensive dominance and Rodgers' experience",
                'NE': "Patriots' defensive tradition and Jones' development",
                'ARI': "Cardinals' young talent and Murray's athleticism",
                'SEA': "Seahawks' offensive creativity and Smith's accuracy"
            }
            
            # Get team strength description
            team_strength = team_strengths.get(predicted_winner, f"{predicted_winner}'s overall team performance")
            
            # Build comprehensive analysis
            analysis_parts = [
                f"ML Model Prediction: {predicted_winner} predicted to win with {int(confidence * 100)}% confidence",
                f"Upset Potential: {int(upset_prob * 100)}% chance of upset",
                f"{confidence_level}: {confidence_desc}",
                f"{upset_risk}: {upset_desc}",
                f"{predicted_winner} Strength: {team_strength}"
            ]
            
            return " • ".join(analysis_parts)
            
        except Exception as e:
            logger.error(f"Error generating robust AI analysis: {e}")
            return f"ML Model Prediction: {predicted_winner} predicted to win with {int(confidence * 100)}% confidence • Upset Potential: {int(upset_prob * 100)}% chance of upset • Analysis based on team performance and historical data"


# Global instance
prediction_pipeline = PredictionPipeline()
