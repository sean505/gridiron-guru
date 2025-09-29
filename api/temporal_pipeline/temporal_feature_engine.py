"""
Temporal Feature Engine
Handles feature engineering for both training (historical) and prediction (2024 baseline + 2025 context)
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from prediction_engine.data_models import TeamStats, GameContext
from temporal_pipeline.temporal_data_collector import temporal_data_collector

logger = logging.getLogger(__name__)


class TemporalFeatureEngine:
    """
    Temporal feature engine that creates features for both training and prediction.
    
    Features:
    - Training features: Uses historical data (2008-2024)
    - Prediction features: Uses 2024 baseline + 2025 context
    - Proper temporal separation
    - Team differentiation based on real performance
    """
    
    def __init__(self):
        """Initialize the temporal feature engine."""
        self.data_collector = temporal_data_collector
        self.feature_names = []
        
        logger.info("TemporalFeatureEngine initialized")
    
    def create_training_features(self, home_stats: TeamStats, away_stats: TeamStats, 
                                game_context: GameContext) -> Dict[str, float]:
        """
        Create features for training using historical data.
        
        Args:
            home_stats: Home team stats from historical data
            away_stats: Away team stats from historical data
            game_context: Game context information
            
        Returns:
            Dictionary of feature names and values
        """
        try:
            features = {}
            
            # 1. Team Performance Features (historical)
            features.update(self._create_team_performance_features(home_stats, away_stats))
            
            # 2. Head-to-Head Features (historical)
            features.update(self._create_head_to_head_features(
                home_stats.team_abbr, away_stats.team_abbr, 
                home_stats.season, home_stats.week
            ))
            
            # 3. Situational Features (historical)
            features.update(self._create_situational_features(game_context, home_stats, away_stats))
            
            # 4. Advanced Analytics Features (historical)
            features.update(self._create_advanced_analytics_features(home_stats, away_stats))
            
            # 5. Trend Features (historical)
            features.update(self._create_trend_features(
                home_stats.team_abbr, away_stats.team_abbr,
                home_stats.season, home_stats.week
            ))
            
            # 6. Betting Market Features (historical)
            features.update(self._create_betting_features(game_context))
            
            # 7. Weather Features (historical)
            features.update(self._create_weather_features(game_context))
            
            return features
            
        except Exception as e:
            logger.error(f"Error creating training features: {e}")
            return {}
    
    def create_prediction_features(self, home_strength: TeamStats, away_strength: TeamStats,
                                 game_context: GameContext) -> Dict[str, float]:
        """
        Create features for prediction using 2024 baseline + 2025 context.
        
        Args:
            home_strength: Home team strength from 2024 baseline
            away_strength: Away team strength from 2024 baseline
            game_context: 2025 game context information
            
        Returns:
            Dictionary of feature names and values (only first 28 features for model compatibility)
        """
        try:
            features = {}
            
            # 1. Team Performance Features (2024 baseline)
            features.update(self._create_team_performance_features(home_strength, away_strength))
            
            # 2. Head-to-Head Features (historical 2008-2024)
            features.update(self._create_head_to_head_features(
                home_strength.team_abbr, away_strength.team_abbr,
                2025, game_context.week  # Use 2025 context but historical H2H
            ))
            
            # 3. Situational Features (2025 context) - only first 7 features
            situational_features = self._create_situational_features(game_context, home_strength, away_strength)
            # Only include the first 7 situational features to match model expectations
            situational_keys = list(situational_features.keys())[:7]
            for key in situational_keys:
                features[key] = situational_features[key]
            
            # Return only the first 28 features that the model expects
            feature_names = [
                "win_pct_diff", "point_diff_diff", "off_epa_diff", "off_epa_ratio",
                "def_epa_diff", "def_epa_ratio", "pass_epa_diff", "pass_def_epa_diff",
                "rush_epa_diff", "rush_def_epa_diff", "turnover_margin_diff", "red_zone_eff_diff",
                "third_down_diff", "sack_rate_diff", "sos_diff", "recent_form_diff",
                "pythagorean_diff", "luck_diff", "h2h_win_pct", "h2h_avg_point_diff",
                "h2h_games_count", "home_field_advantage", "rest_advantage", "division_game",
                "playoff_implications", "early_season", "mid_season", "late_season"
            ]
            
            # Create final features dict with only the expected features
            final_features = {}
            for name in feature_names:
                final_features[name] = features.get(name, 0.0)
            
            return final_features
            
        except Exception as e:
            logger.error(f"Error creating prediction features: {e}")
            return {}
    
    def _create_team_performance_features(self, home_stats: TeamStats, away_stats: TeamStats) -> Dict[str, float]:
        """Create team performance comparison features."""
        features = {}
        
        try:
            # Win percentage differential
            features['win_pct_diff'] = home_stats.win_percentage - away_stats.win_percentage
            
            # Point differential
            features['point_diff_diff'] = home_stats.point_differential - away_stats.point_differential
            
            # Offensive EPA comparison
            features['off_epa_diff'] = home_stats.offensive_epa - away_stats.offensive_epa
            features['off_epa_ratio'] = (home_stats.offensive_epa + 0.1) / (away_stats.offensive_epa + 0.1)
            
            # Defensive EPA comparison
            features['def_epa_diff'] = home_stats.defensive_epa - away_stats.defensive_epa
            features['def_epa_ratio'] = (home_stats.defensive_epa + 0.1) / (away_stats.defensive_epa + 0.1)
            
            # Passing game comparison
            features['pass_epa_diff'] = home_stats.passing_epa - away_stats.passing_epa
            features['pass_def_epa_diff'] = home_stats.pass_defense_epa - away_stats.pass_defense_epa
            
            # Rushing game comparison
            features['rush_epa_diff'] = home_stats.rushing_epa - away_stats.rushing_epa
            features['rush_def_epa_diff'] = home_stats.rush_defense_epa - away_stats.rush_defense_epa
            
            # Turnover margin
            features['turnover_margin_diff'] = home_stats.turnover_margin - away_stats.turnover_margin
            
            # Red zone efficiency
            features['red_zone_eff_diff'] = home_stats.red_zone_efficiency - away_stats.red_zone_efficiency
            
            # Third down conversion
            features['third_down_diff'] = home_stats.third_down_conversion - away_stats.third_down_conversion
            
            # Sack rate
            features['sack_rate_diff'] = home_stats.sack_rate - away_stats.sack_rate
            
            # Strength of schedule
            features['sos_diff'] = home_stats.strength_of_schedule - away_stats.strength_of_schedule
            
            # Recent form
            features['recent_form_diff'] = home_stats.recent_form - away_stats.recent_form
            
            # Pythagorean wins
            features['pythagorean_diff'] = home_stats.pythagorean_wins - away_stats.pythagorean_wins
            
            # Luck factor
            features['luck_diff'] = home_stats.luck_factor - away_stats.luck_factor
            
        except Exception as e:
            logger.error(f"Error creating team performance features: {e}")
        
        return features
    
    def _create_head_to_head_features(self, home_team: str, away_team: str, 
                                    season: int, week: int) -> Dict[str, float]:
        """Create head-to-head historical features."""
        features = {}
        
        try:
            from temporal_pipeline.team_name_mapper import team_name_mapper
            
            game_log = self.data_collector._load_game_log_data()
            
            # Get all possible historical names for both teams
            home_historical_names = team_name_mapper.get_historical_team_name(home_team)
            away_historical_names = team_name_mapper.get_historical_team_name(away_team)
            
            # Look for recent head-to-head games (2008-2024)
            recent_h2h_games = []
            if isinstance(game_log, dict):
                # game_log is a dictionary, iterate through values
                for game in game_log.values():
                    if isinstance(game, dict):
                        game_season = game.get('season', 0)
                        if max(2008, season - 3) <= game_season <= min(2024, season + 1):
                            game_home = game.get('homeTeam', '')
                            game_away = game.get('awayTeam', '')
                            
                            # Check if this is a head-to-head matchup using historical names
                            home_match = any(team_name_mapper.is_team_name_change(game_home, h_name) for h_name in home_historical_names)
                            away_match = any(team_name_mapper.is_team_name_change(game_away, a_name) for a_name in away_historical_names)
                            
                            if home_match and away_match:
                                recent_h2h_games.append(game)
            elif isinstance(game_log, list):
                # game_log is a list, iterate directly
                for game in game_log:
                    if isinstance(game, dict):
                        game_season = game.get('season', 0)
                        if max(2008, season - 3) <= game_season <= min(2024, season + 1):
                            game_home = game.get('homeTeam', '')
                            game_away = game.get('awayTeam', '')
                            
                            # Check if this is a head-to-head matchup using historical names
                            home_match = any(team_name_mapper.is_team_name_change(game_home, h_name) for h_name in home_historical_names)
                            away_match = any(team_name_mapper.is_team_name_change(game_away, a_name) for a_name in away_historical_names)
                            
                            if home_match and away_match:
                                recent_h2h_games.append(game)
            
            if recent_h2h_games:
                # Recent head-to-head record
                home_wins = 0
                for game in recent_h2h_games:
                    game_home = game.get('homeTeam', '')
                    # Check if the home team in the game matches our home team (accounting for name changes)
                    if any(team_name_mapper.is_team_name_change(game_home, h_name) for h_name in home_historical_names):
                        if game.get('Winner') == 1:  # Home team won
                            home_wins += 1
                    else:
                        if game.get('Winner') == 0:  # Away team won (which is our home team)
                            home_wins += 1
                
                total_games = len(recent_h2h_games)
                features['h2h_win_pct'] = home_wins / total_games if total_games > 0 else 0.5
                
                # Average point differential in head-to-head
                point_diffs = []
                for game in recent_h2h_games:
                    game_home = game.get('homeTeam', '')
                    home_score = game.get('HomeScore', 0)
                    away_score = game.get('AwayScore', 0)
                    
                    # Check if our home team was the home team in this game
                    if any(team_name_mapper.is_team_name_change(game_home, h_name) for h_name in home_historical_names):
                        diff = home_score - away_score
                    else:
                        diff = away_score - home_score
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
        """Create situational features."""
        features = {}
        
        try:
            # Home field advantage
            features['home_field_advantage'] = 0.03  # ~3 points advantage
            
            # Rest advantage (placeholder)
            features['rest_advantage'] = 0.0
            
            # Division game
            features['division_game'] = 1.0 if game_context.game_type == 'REG' else 0.0
            
            # Playoff implications
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
            
            # Special teams (placeholder)
            features['special_teams_diff'] = 0.0
            
            # Coaching advantage (placeholder)
            features['coaching_advantage'] = 0.0
            
            # Injury impact
            features['injury_impact_diff'] = home_stats.injury_impact - away_stats.injury_impact
            
            # Momentum
            features['momentum_diff'] = home_stats.recent_form - away_stats.recent_form
            
            # Consistency
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
            game_log = self.data_collector._load_game_log_data()
            
            # Get recent games for both teams
            home_recent = self._get_recent_team_games(home_team, season, week, game_log, 4)
            away_recent = self._get_recent_team_games(away_team, season, week, game_log, 4)
            
            # Home team trends
            if home_recent:
                features['home_trend_wins'] = sum(1 for game in home_recent if game.get('Winner') == 1) / len(home_recent)
                features['home_trend_points'] = np.mean([game.get('HomeScore', 0) for game in home_recent])
                features['home_trend_points_allowed'] = np.mean([game.get('AwayScore', 0) for game in home_recent])
            else:
                features['home_trend_wins'] = 0.5
                features['home_trend_points'] = 0.0
                features['home_trend_points_allowed'] = 0.0
            
            # Away team trends
            if away_recent:
                features['away_trend_wins'] = sum(1 for game in away_recent if game.get('Winner') == 1) / len(away_recent)
                features['away_trend_points'] = np.mean([game.get('AwayScore', 0) for game in away_recent])
                features['away_trend_points_allowed'] = np.mean([game.get('HomeScore', 0) for game in away_recent])
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
            
            # Moneyline features (placeholder)
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
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return self.feature_names.copy()
    
    def set_feature_names(self, names: List[str]) -> None:
        """Set feature names."""
        self.feature_names = names.copy()


# Global instance
temporal_feature_engine = TemporalFeatureEngine()
