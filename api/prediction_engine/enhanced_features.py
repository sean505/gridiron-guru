"""
Enhanced Feature Extractor for Gridiron Guru
High-impact features from nfl_data_py for improved prediction accuracy
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from pathlib import Path
import sqlite3
import json

# Import nfl_data_py
try:
    import nfl_data_py as nfl
    NFL_DATA_AVAILABLE = True
except ImportError:
    NFL_DATA_AVAILABLE = False
    logging.warning("nfl_data_py not available. Enhanced features will use fallback data.")

logger = logging.getLogger(__name__)


class EnhancedFeatureExtractor:
    """
    Extracts high-impact features from nfl_data_py for NFL predictions.
    
    Features include:
    - EPA (Expected Points Added) metrics
    - Situational efficiency (red zone, third down, etc.)
    - Advanced analytics (success rate, explosive plays)
    - Team-specific trends and context
    """
    
    def __init__(self, cache_dir: str = "api/data/cache"):
        """Initialize the enhanced feature extractor."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Cache settings
        self.cache_duration = timedelta(hours=24)
        self.pbp_cache = {}
        self.team_stats_cache = {}
        
        # Feature names for reference
        self.feature_names = [
            # EPA Metrics (6 features)
            'off_epa_per_play', 'def_epa_per_play', 'off_success_rate', 
            'def_success_rate', 'explosive_play_rate', 'explosive_play_allowed_rate',
            
            # Situational Efficiency (4 features)
            'red_zone_td_pct', 'third_down_pct', 'turnover_rate', 'drive_success_rate',
            
            # Advanced Analytics (4 features)
            'field_position_advantage', 'time_of_possession_pct', 
            'home_away_epa_diff', 'recent_form_trend',
            
            # Contextual Factors (3 features)
            'rest_days', 'travel_distance', 'weather_factor'
        ]
        
        logger.info("EnhancedFeatureExtractor initialized")
    
    def extract_enhanced_features(self, team: str, opponent: str, season: int, week: int) -> List[float]:
        """
        Extract comprehensive enhanced features for a team.
        
        Args:
            team: Team abbreviation (e.g., 'KC')
            opponent: Opponent team abbreviation (e.g., 'BUF')
            season: NFL season year
            week: Week number
            
        Returns:
            List of 17 enhanced feature values
        """
        try:
            features = []
            
            # 1. EPA Metrics (6 features)
            epa_features = self._extract_epa_metrics(team, season, week)
            features.extend(epa_features)
            
            # 2. Situational Efficiency (4 features)
            situational_features = self._extract_situational_efficiency(team, season, week)
            features.extend(situational_features)
            
            # 3. Advanced Analytics (4 features)
            analytics_features = self._extract_advanced_analytics(team, season, week)
            features.extend(analytics_features)
            
            # 4. Contextual Factors (3 features)
            context_features = self._extract_contextual_factors(team, opponent, season, week)
            features.extend(context_features)
            
            # Ensure we have exactly 17 features
            if len(features) != 17:
                logger.warning(f"Expected 17 features, got {len(features)}. Padding with zeros.")
                while len(features) < 17:
                    features.append(0.0)
                features = features[:17]
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting enhanced features for {team}: {e}")
            # Return fallback features (all zeros)
            return [0.0] * 17
    
    def _extract_epa_metrics(self, team: str, season: int, week: int) -> List[float]:
        """Extract EPA-based metrics for a team."""
        try:
            if not NFL_DATA_AVAILABLE:
                return [0.0] * 6
            
            # Get play-by-play data
            pbp_data = self._get_pbp_data(season)
            if pbp_data.empty:
                return [0.0] * 6
            
            # Filter for team's games up to the current week
            team_games = pbp_data[
                ((pbp_data['posteam'] == team) | (pbp_data['defteam'] == team)) &
                (pbp_data['week'] <= week)
            ].copy()
            
            if team_games.empty:
                return [0.0] * 6
            
            # Calculate EPA metrics
            off_plays = team_games[team_games['posteam'] == team]
            def_plays = team_games[team_games['defteam'] == team]
            
            # Offensive EPA per play
            off_epa_per_play = off_plays['epa'].mean() if not off_plays.empty else 0.0
            
            # Defensive EPA per play (negative because we want to stop opponent)
            def_epa_per_play = -def_plays['epa'].mean() if not def_plays.empty else 0.0
            
            # Success rate (EPA > 0)
            off_success_rate = (off_plays['epa'] > 0).mean() if not off_plays.empty else 0.0
            def_success_rate = (def_plays['epa'] < 0).mean() if not def_plays.empty else 0.0
            
            # Explosive play rate (EPA > 1.5)
            explosive_play_rate = (off_plays['epa'] > 1.5).mean() if not off_plays.empty else 0.0
            explosive_play_allowed_rate = (def_plays['epa'] > 1.5).mean() if not def_plays.empty else 0.0
            
            return [
                float(off_epa_per_play),
                float(def_epa_per_play),
                float(off_success_rate),
                float(def_success_rate),
                float(explosive_play_rate),
                float(explosive_play_allowed_rate)
            ]
            
        except Exception as e:
            logger.error(f"Error extracting EPA metrics for {team}: {e}")
            return [0.0] * 6
    
    def _extract_situational_efficiency(self, team: str, season: int, week: int) -> List[float]:
        """Extract situational efficiency metrics."""
        try:
            if not NFL_DATA_AVAILABLE:
                return [0.0] * 4
            
            pbp_data = self._get_pbp_data(season)
            if pbp_data.empty:
                return [0.0] * 4
            
            # Filter for team's games
            team_games = pbp_data[
                ((pbp_data['posteam'] == team) | (pbp_data['defteam'] == team)) &
                (pbp_data['week'] <= week)
            ].copy()
            
            if team_games.empty:
                return [0.0] * 4
            
            # Red zone TD percentage
            red_zone_plays = team_games[
                (team_games['posteam'] == team) & 
                (team_games['yardline_100'] <= 20) &
                (team_games['play_type'] == 'pass')
            ]
            red_zone_td_pct = (red_zone_plays['touchdown'] == 1).mean() if not red_zone_plays.empty else 0.0
            
            # Third down conversion rate
            third_down_plays = team_games[
                (team_games['posteam'] == team) & 
                (team_games['down'] == 3)
            ]
            third_down_pct = (third_down_plays['first_down'] == 1).mean() if not third_down_plays.empty else 0.0
            
            # Turnover rate (per drive)
            drives = team_games[team_games['posteam'] == team].groupby('drive')
            turnover_rate = (drives['interception'].sum() + drives['fumble_lost'].sum()) / len(drives) if len(drives) > 0 else 0.0
            
            # Drive success rate (TD or FG)
            drive_success_rate = (drives['touchdown'].sum() + drives['field_goal_attempt'].sum()) / len(drives) if len(drives) > 0 else 0.0
            
            return [
                float(red_zone_td_pct),
                float(third_down_pct),
                float(turnover_rate),
                float(drive_success_rate)
            ]
            
        except Exception as e:
            logger.error(f"Error extracting situational efficiency for {team}: {e}")
            return [0.0] * 4
    
    def _extract_advanced_analytics(self, team: str, season: int, week: int) -> List[float]:
        """Extract advanced analytics metrics."""
        try:
            if not NFL_DATA_AVAILABLE:
                return [0.0] * 4
            
            pbp_data = self._get_pbp_data(season)
            if pbp_data.empty:
                return [0.0] * 4
            
            # Filter for team's games
            team_games = pbp_data[
                ((pbp_data['posteam'] == team) | (pbp_data['defteam'] == team)) &
                (pbp_data['week'] <= week)
            ].copy()
            
            if team_games.empty:
                return [0.0] * 4
            
            # Field position advantage (average starting field position)
            off_plays = team_games[team_games['posteam'] == team]
            field_position_advantage = off_plays['yardline_100'].mean() if not off_plays.empty else 50.0
            
            # Time of possession percentage (simplified)
            time_of_possession_pct = 0.5  # Placeholder - would need drive data
            
            # Home/Away EPA difference
            home_games = team_games[team_games['home_team'] == team]
            away_games = team_games[team_games['away_team'] == team]
            
            home_epa = home_games[home_games['posteam'] == team]['epa'].mean() if not home_games.empty else 0.0
            away_epa = away_games[away_games['posteam'] == team]['epa'].mean() if not away_games.empty else 0.0
            home_away_epa_diff = home_epa - away_epa
            
            # Recent form trend (last 4 games EPA trend)
            recent_games = team_games[team_games['week'] > week - 4]
            if not recent_games.empty:
                recent_epa = recent_games[recent_games['posteam'] == team]['epa'].mean()
                recent_form_trend = recent_epa
            else:
                recent_form_trend = 0.0
            
            return [
                float(field_position_advantage),
                float(time_of_possession_pct),
                float(home_away_epa_diff),
                float(recent_form_trend)
            ]
            
        except Exception as e:
            logger.error(f"Error extracting advanced analytics for {team}: {e}")
            return [0.0] * 4
    
    def _extract_contextual_factors(self, team: str, opponent: str, season: int, week: int) -> List[float]:
        """Extract contextual factors like rest, travel, weather."""
        try:
            # Rest days (simplified - would need actual schedule data)
            rest_days = 7.0  # Default to 7 days
            
            # Travel distance (simplified - would need actual distance calculation)
            travel_distance = 0.0  # Placeholder
            
            # Weather factor (simplified - would need actual weather data)
            weather_factor = 0.0  # Placeholder
            
            return [
                float(rest_days),
                float(travel_distance),
                float(weather_factor)
            ]
            
        except Exception as e:
            logger.error(f"Error extracting contextual factors for {team}: {e}")
            return [0.0] * 3
    
    def _get_pbp_data(self, season: int) -> pd.DataFrame:
        """Get play-by-play data with caching."""
        cache_key = f"pbp_{season}"
        
        if cache_key in self.pbp_cache:
            return self.pbp_cache[cache_key]
        
        try:
            if NFL_DATA_AVAILABLE:
                # Use nfl_data_py with caching
                pbp_data = nfl.import_pbp_data([season], cache=True)
                self.pbp_cache[cache_key] = pbp_data
                return pbp_data
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching PBP data for season {season}: {e}")
            return pd.DataFrame()
    
    def get_rest_travel_factors(self, home_team: str, away_team: str, week: int) -> Dict[str, float]:
        """Get rest and travel factors for both teams."""
        return {
            'rest_differential': 0.0,  # Home team rest - Away team rest
            'travel_distance': 0.0,    # Away team travel distance
            'home_advantage': 0.1      # Standard home field advantage
        }
    
    def get_all_nfl_teams(self) -> List[str]:
        """Get list of all NFL team abbreviations."""
        return [
            'ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE', 'DAL', 'DEN',
            'DET', 'GB', 'HOU', 'IND', 'JAX', 'KC', 'LAC', 'LAR', 'LV', 'MIA',
            'MIN', 'NE', 'NO', 'NYG', 'NYJ', 'PHI', 'PIT', 'SEA', 'SF', 'TB', 'TEN', 'WAS'
        ]


# Global enhanced feature extractor instance
enhanced_extractor = EnhancedFeatureExtractor()
