"""
Feature engineering module for the Gridiron Guru prediction engine.

This module transforms raw NFL data into predictive features for machine learning models.
It creates derived metrics, handles missing data, and prepares features for prediction algorithms.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

# Import data models
from .data_models import GameContext, TeamStats
from .data_collector import data_collector

# Simple feature engineering without complex model dependencies

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering class that transforms raw NFL data into ML-ready features.
    
    Features include:
    - Team performance metrics
    - Head-to-head historical data
    - Situational factors (weather, rest, travel)
    - Advanced analytics (EPA, DVOA-style metrics)
    - Trend analysis (recent form, momentum)
    """
    
    def __init__(self):
        """Initialize the feature engineer."""
        self.scalers = {}
        self.imputers = {}
        self.feature_names = []
        self.feature_importance = {}
        
        logger.info("FeatureEngineer initialized")
    
    def create_game_features(self, home_team: str, away_team: str, 
                           season: int, week: int, game_context: GameContext) -> Dict[str, float]:
        """
        Create comprehensive features for a specific game.
        
        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            season: NFL season year
            week: Week number
            game_context: Game context information
            
        Returns:
            Dictionary of feature names and values
        """
        try:
            features = {}
            
            # Get team stats
            home_stats = data_collector.get_team_stats(home_team, season, week)
            away_stats = data_collector.get_team_stats(away_team, season, week)
            
            # 1. Team Performance Features
            features.update(self._create_team_performance_features(home_stats, away_stats))
            
            # 2. Head-to-Head Features
            features.update(self._create_head_to_head_features(home_team, away_team, season, week))
            
            # 3. Situational Features
            features.update(self._create_situational_features(game_context, home_stats, away_stats))
            
            # 4. Advanced Analytics Features
            features.update(self._create_advanced_analytics_features(home_stats, away_stats))
            
            # 5. Trend and Momentum Features
            features.update(self._create_trend_features(home_team, away_team, season, week))
            
            # 6. Betting Market Features
            features.update(self._create_betting_features(game_context))
            
            # 7. Weather and Environmental Features
            features.update(self._create_weather_features(game_context))
            
            # Store feature names for reference
            self.feature_names = list(features.keys())
            
            return features
            
        except Exception as e:
            logger.error(f"Error creating game features: {e}")
            return {}
    
    def _create_team_performance_features(self, home_stats: TeamStats, away_stats: TeamStats) -> Dict[str, float]:
        """Create team performance comparison features."""
        features = {}
        
        try:
            # Create realistic team differentiation using team names and historical patterns
            # This provides meaningful features even when data loading fails
            
            # Use neutral baseline features - let the ML model learn from real data
            # No hardcoded team strengths - use actual team stats when available
            
            # Win percentage differential (use actual team stats)
            features['win_pct_diff'] = home_stats.win_percentage - away_stats.win_percentage
            
            # Point differential (use actual team stats)
            features['point_diff_diff'] = home_stats.point_differential - away_stats.point_differential
            
            # Offensive EPA comparison (use actual team stats)
            features['off_epa_diff'] = home_stats.offensive_epa - away_stats.offensive_epa
            features['off_epa_ratio'] = (home_stats.offensive_epa + 0.1) / (away_stats.offensive_epa + 0.1)
            
            # Defensive EPA comparison (use actual team stats)
            features['def_epa_diff'] = home_stats.defensive_epa - away_stats.defensive_epa
            features['def_epa_ratio'] = (home_stats.defensive_epa + 0.1) / (away_stats.defensive_epa + 0.1)
            
            # Passing game comparison (use actual team stats)
            features['pass_epa_diff'] = home_stats.passing_epa - away_stats.passing_epa
            features['pass_def_epa_diff'] = home_stats.pass_defense_epa - away_stats.pass_defense_epa
            
            # Rushing game comparison (use actual team stats)
            features['rush_epa_diff'] = home_stats.rushing_epa - away_stats.rushing_epa
            features['rush_def_epa_diff'] = home_stats.rush_defense_epa - away_stats.rush_defense_epa
            
            # Turnover margin (use actual team stats)
            features['turnover_margin_diff'] = home_stats.turnover_margin - away_stats.turnover_margin
            
            # Red zone efficiency (use actual team stats)
            features['red_zone_eff_diff'] = home_stats.red_zone_efficiency - away_stats.red_zone_efficiency
            
            # Third down conversion (use actual team stats)
            features['third_down_diff'] = home_stats.third_down_conversion - away_stats.third_down_conversion
            
            # Sack rate (use actual team stats)
            features['sack_rate_diff'] = home_stats.sack_rate - away_stats.sack_rate
            
            # Strength of schedule (use actual team stats)
            features['sos_diff'] = home_stats.strength_of_schedule - away_stats.strength_of_schedule
            
            # Recent form (use actual team stats)
            features['recent_form_diff'] = home_stats.recent_form - away_stats.recent_form
            
            # Pythagorean wins (use actual team stats)
            features['pythagorean_diff'] = home_stats.pythagorean_wins - away_stats.pythagorean_wins
            
            # Luck factor (use actual team stats)
            features['luck_diff'] = home_stats.luck_factor - away_stats.luck_factor
            
        except Exception as e:
            logger.error(f"Error creating team performance features: {e}")
        
        return features
    
    def _create_head_to_head_features(self, home_team: str, away_team: str, 
                                    season: int, week: int) -> Dict[str, float]:
        """Create head-to-head historical features."""
        features = {}
        
        try:
            game_log = data_collector.get_game_log_data()
            
            # Look for recent head-to-head games
            recent_h2h_games = []
            for check_season in range(max(2020, season - 3), season + 1):
                for check_week in range(1, 19):
                    game_key = f"{check_season}_{check_week}_{home_team}_{away_team}"
                    reverse_key = f"{check_season}_{check_week}_{away_team}_{home_team}"
                    
                    if game_key in game_log:
                        recent_h2h_games.append(game_log[game_key])
                    elif reverse_key in game_log:
                        recent_h2h_games.append(game_log[reverse_key])
            
            if recent_h2h_games:
                # Recent head-to-head record
                home_wins = sum(1 for game in recent_h2h_games 
                              if game.get('home_team') == home_team and game.get('won', False))
                total_games = len(recent_h2h_games)
                features['h2h_win_pct'] = home_wins / total_games if total_games > 0 else 0.5
                
                # Average point differential in head-to-head
                point_diffs = []
                for game in recent_h2h_games:
                    if game.get('home_team') == home_team:
                        diff = game.get('home_score', 0) - game.get('away_score', 0)
                    else:
                        diff = game.get('away_score', 0) - game.get('home_score', 0)
                    point_diffs.append(diff)
                
                features['h2h_avg_point_diff'] = np.mean(point_diffs) if point_diffs else 0.0
                features['h2h_games_count'] = len(recent_h2h_games)
            else:
                features['h2h_win_pct'] = 0.5  # Neutral if no history
                features['h2h_avg_point_diff'] = 0.0
                features['h2h_games_count'] = 0
            
        except Exception as e:
            logger.error(f"Error creating head-to-head features: {e}")
            features['h2h_win_pct'] = 0.5
            features['h2h_avg_point_diff'] = 0.0
            features['h2h_games_count'] = 0
        
        return features
    
    def _create_situational_features(self, game_context: GameContext, 
                                   home_stats: TeamStats, away_stats: TeamStats) -> Dict[str, float]:
        """Create situational features (rest, travel, etc.)."""
        features = {}
        
        try:
            # Home field advantage (reduced to realistic level)
            features['home_field_advantage'] = 0.03  # ~3 points advantage (realistic)
            
            # Rest advantage (if available in game context)
            if hasattr(game_context, 'home_rest') and hasattr(game_context, 'away_rest'):
                features['rest_advantage'] = game_context.home_rest - game_context.away_rest
            else:
                features['rest_advantage'] = 0.0
            
            # Division game
            features['division_game'] = 1.0 if game_context.game_type == 'REG' else 0.0
            
            # Playoff implications (simplified)
            features['playoff_implications'] = 1.0 if game_context.week >= 15 else 0.0
            
            # Season stage
            if game_context.week <= 4:
                features['early_season'] = 1.0
                features['mid_season'] = 0.0
                features['late_season'] = 0.0
            elif game_context.week <= 12:
                features['early_season'] = 0.0
                features['mid_season'] = 1.0
                features['late_season'] = 0.0
            else:
                features['early_season'] = 0.0
                features['mid_season'] = 0.0
                features['late_season'] = 1.0
            
            # Home/away record comparison
            home_home_record = home_stats.home_record.get('win_percentage', 0.5)
            away_away_record = away_stats.away_record.get('win_percentage', 0.5)
            features['home_away_record_diff'] = home_home_record - away_away_record
            
        except Exception as e:
            logger.error(f"Error creating situational features: {e}")
        
        return features
    
    def _create_advanced_analytics_features(self, home_stats: TeamStats, away_stats: TeamStats) -> Dict[str, float]:
        """Create advanced analytics features."""
        features = {}
        
        try:
            # DVOA-style efficiency metrics
            features['offensive_efficiency_diff'] = (
                home_stats.offensive_epa - away_stats.defensive_epa
            ) - (away_stats.offensive_epa - home_stats.defensive_epa)
            
            # Defensive efficiency
            features['defensive_efficiency_diff'] = (
                home_stats.defensive_epa - away_stats.offensive_epa
            ) - (away_stats.defensive_epa - home_stats.offensive_epa)
            
            # Special teams (placeholder - would need special teams data)
            features['special_teams_diff'] = 0.0
            
            # Coaching advantage (placeholder - would need coaching data)
            features['coaching_advantage'] = 0.0
            
            # Injury impact
            features['injury_impact_diff'] = home_stats.injury_impact - away_stats.injury_impact
            
            # Momentum (recent form weighted)
            features['momentum_diff'] = home_stats.recent_form - away_stats.recent_form
            
            # Consistency (inverse of luck factor variance)
            home_consistency = 1.0 - abs(home_stats.luck_factor)
            away_consistency = 1.0 - abs(away_stats.luck_factor)
            features['consistency_diff'] = home_consistency - away_consistency
            
        except Exception as e:
            logger.error(f"Error creating advanced analytics features: {e}")
        
        return features
    
    def _create_trend_features(self, home_team: str, away_team: str, 
                             season: int, week: int) -> Dict[str, float]:
        """Create trend and momentum features."""
        features = {}
        
        try:
            game_log = data_collector.get_game_log_data()
            
            # Get recent games for both teams
            home_recent = self._get_recent_team_games(home_team, season, week, game_log, 4)
            away_recent = self._get_recent_team_games(away_team, season, week, game_log, 4)
            
            # Home team trends
            if home_recent:
                features['home_trend_wins'] = sum(1 for game in home_recent if game.get('won', False)) / len(home_recent)
                features['home_trend_points'] = np.mean([game.get('points_scored', 0) for game in home_recent])
                features['home_trend_points_allowed'] = np.mean([game.get('points_allowed', 0) for game in home_recent])
            else:
                features['home_trend_wins'] = 0.5
                features['home_trend_points'] = 0.0
                features['home_trend_points_allowed'] = 0.0
            
            # Away team trends
            if away_recent:
                features['away_trend_wins'] = sum(1 for game in away_recent if game.get('won', False)) / len(away_recent)
                features['away_trend_points'] = np.mean([game.get('points_scored', 0) for game in away_recent])
                features['away_trend_points_allowed'] = np.mean([game.get('points_allowed', 0) for game in away_recent])
            else:
                features['away_trend_wins'] = 0.5
                features['away_trend_points'] = 0.0
                features['away_trend_points_allowed'] = 0.0
            
            # Trend differentials
            features['trend_wins_diff'] = features['home_trend_wins'] - features['away_trend_wins']
            features['trend_points_diff'] = features['home_trend_points'] - features['away_trend_points']
            features['trend_defense_diff'] = features['away_trend_points_allowed'] - features['home_trend_points_allowed']
            
        except Exception as e:
            logger.error(f"Error creating trend features: {e}")
            # Set default values
            features.update({
                'home_trend_wins': 0.5, 'away_trend_wins': 0.5,
                'home_trend_points': 0.0, 'away_trend_points': 0.0,
                'home_trend_points_allowed': 0.0, 'away_trend_points_allowed': 0.0,
                'trend_wins_diff': 0.0, 'trend_points_diff': 0.0, 'trend_defense_diff': 0.0
            })
        
        return features
    
    def _create_betting_features(self, game_context: GameContext) -> Dict[str, float]:
        """Create betting market features."""
        features = {}
        
        try:
            # Spread features
            if game_context.spread is not None:
                features['spread'] = game_context.spread
                features['spread_abs'] = abs(game_context.spread)
                features['home_underdog'] = 1.0 if game_context.spread > 0 else 0.0
            else:
                features['spread'] = 0.0
                features['spread_abs'] = 0.0
                features['home_underdog'] = 0.0
            
            # Total features
            if game_context.total is not None:
                features['total'] = game_context.total
                features['high_total'] = 1.0 if game_context.total > 50 else 0.0
            else:
                features['total'] = 45.0  # Default total
                features['high_total'] = 0.0
            
            # Moneyline features
            if game_context.home_moneyline is not None and game_context.away_moneyline is not None:
                # Convert moneylines to implied probabilities
                home_prob = self._moneyline_to_probability(game_context.home_moneyline)
                away_prob = self._moneyline_to_probability(game_context.away_moneyline)
                features['home_ml_prob'] = home_prob
                features['away_ml_prob'] = away_prob
                features['ml_prob_diff'] = home_prob - away_prob
            else:
                features['home_ml_prob'] = 0.5
                features['away_ml_prob'] = 0.5
                features['ml_prob_diff'] = 0.0
            
        except Exception as e:
            logger.error(f"Error creating betting features: {e}")
        
        return features
    
    def _create_weather_features(self, game_context: GameContext) -> Dict[str, float]:
        """Create weather and environmental features."""
        features = {}
        
        try:
            # Temperature features
            if game_context.temperature is not None:
                features['temperature'] = game_context.temperature
                features['cold_weather'] = 1.0 if game_context.temperature < 32 else 0.0
                features['hot_weather'] = 1.0 if game_context.temperature > 80 else 0.0
            else:
                features['temperature'] = 70.0  # Default temperature
                features['cold_weather'] = 0.0
                features['hot_weather'] = 0.0
            
            # Wind features
            if game_context.wind_speed is not None:
                features['wind_speed'] = game_context.wind_speed
                features['high_wind'] = 1.0 if game_context.wind_speed > 15 else 0.0
            else:
                features['wind_speed'] = 0.0
                features['high_wind'] = 0.0
            
            # Precipitation
            if game_context.precipitation is not None:
                features['precipitation'] = game_context.precipitation
                features['rain_game'] = 1.0 if game_context.precipitation > 0.3 else 0.0
            else:
                features['precipitation'] = 0.0
                features['rain_game'] = 0.0
            
            # Stadium features
            features['dome'] = 1.0 if game_context.roof == 'dome' else 0.0
            features['retractable_roof'] = 1.0 if game_context.roof == 'retractable' else 0.0
            features['grass_surface'] = 1.0 if game_context.surface == 'grass' else 0.0
            
        except Exception as e:
            logger.error(f"Error creating weather features: {e}")
        
        return features
    
    def _get_recent_team_games(self, team: str, season: int, week: int, 
                             game_log: Dict, count: int) -> List[Dict]:
        """Get recent games for a specific team."""
        try:
            recent_games = []
            for check_week in range(max(1, week - count), week):
                game_key = f"{season}_{check_week}_{team}"
                if game_key in game_log:
                    recent_games.append(game_log[game_key])
            return recent_games
        except Exception as e:
            logger.error(f"Error getting recent team games: {e}")
            return []
    
    def _moneyline_to_probability(self, moneyline: int) -> float:
        """Convert moneyline odds to implied probability."""
        try:
            if moneyline > 0:
                return 100 / (moneyline + 100)
            else:
                return abs(moneyline) / (abs(moneyline) + 100)
        except Exception as e:
            logger.error(f"Error converting moneyline: {e}")
            return 0.5
    
    def prepare_features_for_ml(self, features: Dict[str, float]) -> np.ndarray:
        """
        Prepare features for machine learning models.
        
        Args:
            features: Dictionary of feature names and values
            
        Returns:
            Numpy array of feature values
        """
        try:
            # Convert to array in consistent order
            feature_array = np.array([features.get(name, 0.0) for name in self.feature_names])
            
            # Handle missing values
            feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=0.0, neginf=0.0)
            
            return feature_array.reshape(1, -1)  # Reshape for single prediction
            
        except Exception as e:
            logger.error(f"Error preparing features for ML: {e}")
            return np.zeros((1, len(self.feature_names)))
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        return self.feature_importance
    
    def set_feature_importance(self, importance_scores: Dict[str, float]) -> None:
        """Set feature importance scores from trained model."""
        self.feature_importance = importance_scores
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return self.feature_names.copy()


# Global feature engineer instance
feature_engineer = FeatureEngineer()
