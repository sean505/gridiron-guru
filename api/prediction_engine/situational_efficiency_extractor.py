"""
Situational Efficiency Feature Extractor

This module extracts situational efficiency metrics for NFL teams,
including red zone efficiency, third/fourth down conversion rates,
and two-minute drill performance.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from pathlib import Path
import json

# Import nfl_data_py
try:
    import nfl_data_py as nfl
    NFL_DATA_AVAILABLE = True
except ImportError:
    NFL_DATA_AVAILABLE = False
    logging.warning("nfl_data_py not available. Situational efficiency will use fallback data.")

logger = logging.getLogger(__name__)


class SituationalEfficiencyExtractor:
    """
    Extracts situational efficiency features for NFL teams.
    
    Features:
    - Red zone touchdown percentage (last 8 games)
    - Third down conversion rate (last 8 games)
    - Fourth down conversion rate (when attempted)
    - Two-minute drill efficiency
    - Goal line efficiency
    - Short yardage efficiency
    """
    
    def __init__(self, cache_dir: str = "api/data/cache"):
        """Initialize the situational efficiency extractor."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache settings
        self.cache_duration = timedelta(hours=6)  # Cache for 6 hours
        self.cache_file = self.cache_dir / "situational_efficiency_cache.json"
        
        # Load existing cache
        self.cache = self._load_cache()
        
        logger.info("SituationalEfficiencyExtractor initialized")
    
    def _load_cache(self) -> Dict[str, Any]:
        """Load situational efficiency cache from file."""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.warning(f"Error loading situational efficiency cache: {e}")
            return {}
    
    def _save_cache(self):
        """Save situational efficiency cache to file."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, default=str)
        except Exception as e:
            logger.warning(f"Error saving situational efficiency cache: {e}")
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid."""
        if cache_key not in self.cache:
            return False
        
        cached_time = datetime.fromisoformat(self.cache[cache_key]['timestamp'])
        return datetime.now() - cached_time < self.cache_duration
    
    def extract_situational_efficiency(self, team: str, season: int, week: int) -> Dict[str, float]:
        """
        Extract situational efficiency features for a team.
        
        Args:
            team: Team abbreviation (e.g., 'KC')
            season: NFL season year
            week: Week number
            
        Returns:
            Dictionary of situational efficiency features
        """
        try:
            cache_key = f"{team}_{season}_{week}"
            
            # Check cache first
            if self._is_cache_valid(cache_key):
                logger.info(f"Using cached situational efficiency for {team}")
                return self.cache[cache_key]['features']
            
            # Extract fresh data
            features = self._calculate_situational_efficiency(team, season, week)
            
            # Cache the results
            self.cache[cache_key] = {
                'features': features,
                'timestamp': datetime.now().isoformat()
            }
            self._save_cache()
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting situational efficiency for {team}: {e}")
            return self._get_fallback_features()
    
    def _calculate_situational_efficiency(self, team: str, season: int, week: int) -> Dict[str, float]:
        """Calculate situational efficiency features from play-by-play data."""
        try:
            if not NFL_DATA_AVAILABLE:
                logger.warning("nfl_data_py not available, using fallback situational efficiency features")
                return self._get_fallback_features()
            
            # Get play-by-play data for recent games
            recent_games = self._get_recent_games(team, season, week, 8)
            
            if not recent_games:
                logger.warning(f"No recent games found for {team}, using fallback")
                return self._get_fallback_features()
            
            # Calculate situational efficiency metrics
            red_zone_efficiency = self._calculate_red_zone_efficiency(team, recent_games)
            third_down_efficiency = self._calculate_third_down_efficiency(team, recent_games)
            fourth_down_efficiency = self._calculate_fourth_down_efficiency(team, recent_games)
            two_minute_efficiency = self._calculate_two_minute_efficiency(team, recent_games)
            goal_line_efficiency = self._calculate_goal_line_efficiency(team, recent_games)
            short_yardage_efficiency = self._calculate_short_yardage_efficiency(team, recent_games)
            
            # Calculate trends
            red_zone_trend = self._calculate_efficiency_trend(team, recent_games, 'red_zone')
            third_down_trend = self._calculate_efficiency_trend(team, recent_games, 'third_down')
            
            features = {
                'red_zone_td_percentage': red_zone_efficiency,
                'third_down_conversion_rate': third_down_efficiency,
                'fourth_down_conversion_rate': fourth_down_efficiency,
                'two_minute_drill_efficiency': two_minute_efficiency,
                'goal_line_efficiency': goal_line_efficiency,
                'short_yardage_efficiency': short_yardage_efficiency,
                'red_zone_trend': red_zone_trend,
                'third_down_trend': third_down_trend,
                'situational_consistency': self._calculate_situational_consistency(team, recent_games)
            }
            
            logger.info(f"Calculated situational efficiency for {team}: {len(features)} features")
            return features
            
        except Exception as e:
            logger.error(f"Error calculating situational efficiency for {team}: {e}")
            return self._get_fallback_features()
    
    def _get_recent_games(self, team: str, season: int, week: int, count: int) -> List[Dict]:
        """Get recent games for a team."""
        try:
            # Get play-by-play data for the season
            pbp_data = nfl.import_pbp_data([season])
            
            if pbp_data.empty:
                return []
            
            # Filter for team's games
            team_games = pbp_data[
                (pbp_data['posteam'] == team) | (pbp_data['defteam'] == team)
            ].copy()
            
            # Get unique games and sort by week
            games = team_games.groupby(['game_id', 'week']).first().reset_index()
            games = games[games['week'] < week].sort_values('week', ascending=False)
            
            # Get the most recent games
            recent_games = []
            for _, game in games.head(count).iterrows():
                game_data = team_games[team_games['game_id'] == game['game_id']]
                recent_games.append({
                    'game_id': game['game_id'],
                    'week': game['week'],
                    'plays': game_data
                })
            
            return recent_games
            
        except Exception as e:
            logger.error(f"Error getting recent games for {team}: {e}")
            return []
    
    def _calculate_red_zone_efficiency(self, team: str, games: List[Dict]) -> float:
        """Calculate red zone touchdown percentage."""
        try:
            total_red_zone_touchdowns = 0
            total_red_zone_attempts = 0
            
            for game in games:
                plays = game['plays']
                offensive_plays = plays[plays['posteam'] == team]
                
                if not offensive_plays.empty:
                    # Red zone plays (inside 20 yard line)
                    red_zone_plays = offensive_plays[
                        (offensive_plays['yardline_100'] <= 20) & 
                        (offensive_plays['yardline_100'] > 0)
                    ]
                    
                    if not red_zone_plays.empty:
                        # Count touchdowns in red zone
                        red_zone_tds = red_zone_plays[red_zone_plays['touchdown'] == 1]
                        total_red_zone_touchdowns += len(red_zone_tds)
                        
                        # Count red zone drives (simplified - any red zone play counts as attempt)
                        total_red_zone_attempts += len(red_zone_plays)
            
            return total_red_zone_touchdowns / total_red_zone_attempts if total_red_zone_attempts > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating red zone efficiency for {team}: {e}")
            return 0.0
    
    def _calculate_third_down_efficiency(self, team: str, games: List[Dict]) -> float:
        """Calculate third down conversion rate."""
        try:
            total_third_down_conversions = 0
            total_third_down_attempts = 0
            
            for game in games:
                plays = game['plays']
                offensive_plays = plays[plays['posteam'] == team]
                
                if not offensive_plays.empty:
                    # Third down plays
                    third_down_plays = offensive_plays[offensive_plays['down'] == 3]
                    
                    if not third_down_plays.empty:
                        # Count conversions (first down gained)
                        conversions = third_down_plays[third_down_plays['first_down'] == 1]
                        total_third_down_conversions += len(conversions)
                        total_third_down_attempts += len(third_down_plays)
            
            return total_third_down_conversions / total_third_down_attempts if total_third_down_attempts > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating third down efficiency for {team}: {e}")
            return 0.0
    
    def _calculate_fourth_down_efficiency(self, team: str, games: List[Dict]) -> float:
        """Calculate fourth down conversion rate."""
        try:
            total_fourth_down_conversions = 0
            total_fourth_down_attempts = 0
            
            for game in games:
                plays = game['plays']
                offensive_plays = plays[plays['posteam'] == team]
                
                if not offensive_plays.empty:
                    # Fourth down plays
                    fourth_down_plays = offensive_plays[offensive_plays['down'] == 4]
                    
                    if not fourth_down_plays.empty:
                        # Count conversions (first down gained)
                        conversions = fourth_down_plays[fourth_down_plays['first_down'] == 1]
                        total_fourth_down_conversions += len(conversions)
                        total_fourth_down_attempts += len(fourth_down_plays)
            
            return total_fourth_down_conversions / total_fourth_down_attempts if total_fourth_down_attempts > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating fourth down efficiency for {team}: {e}")
            return 0.0
    
    def _calculate_two_minute_efficiency(self, team: str, games: List[Dict]) -> float:
        """Calculate two-minute drill efficiency."""
        try:
            total_two_minute_scores = 0
            total_two_minute_attempts = 0
            
            for game in games:
                plays = game['plays']
                offensive_plays = plays[plays['posteam'] == team]
                
                if not offensive_plays.empty:
                    # Two-minute drill plays (last 2 minutes of half)
                    two_minute_plays = offensive_plays[
                        (offensive_plays['quarter_seconds_remaining'] <= 120) |
                        (offensive_plays['half_seconds_remaining'] <= 120)
                    ]
                    
                    if not two_minute_plays.empty:
                        # Count scoring plays
                        scores = two_minute_plays[two_minute_plays['touchdown'] == 1]
                        total_two_minute_scores += len(scores)
                        
                        # Count two-minute drives (simplified)
                        total_two_minute_attempts += len(two_minute_plays)
            
            return total_two_minute_scores / total_two_minute_attempts if total_two_minute_attempts > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating two-minute efficiency for {team}: {e}")
            return 0.0
    
    def _calculate_goal_line_efficiency(self, team: str, games: List[Dict]) -> float:
        """Calculate goal line efficiency (inside 5 yard line)."""
        try:
            total_goal_line_touchdowns = 0
            total_goal_line_attempts = 0
            
            for game in games:
                plays = game['plays']
                offensive_plays = plays[plays['posteam'] == team]
                
                if not offensive_plays.empty:
                    # Goal line plays (inside 5 yard line)
                    goal_line_plays = offensive_plays[
                        (offensive_plays['yardline_100'] <= 5) & 
                        (offensive_plays['yardline_100'] > 0)
                    ]
                    
                    if not goal_line_plays.empty:
                        # Count touchdowns
                        goal_line_tds = goal_line_plays[goal_line_plays['touchdown'] == 1]
                        total_goal_line_touchdowns += len(goal_line_tds)
                        total_goal_line_attempts += len(goal_line_plays)
            
            return total_goal_line_touchdowns / total_goal_line_attempts if total_goal_line_attempts > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating goal line efficiency for {team}: {e}")
            return 0.0
    
    def _calculate_short_yardage_efficiency(self, team: str, games: List[Dict]) -> float:
        """Calculate short yardage efficiency (3rd/4th and short)."""
        try:
            total_short_yardage_conversions = 0
            total_short_yardage_attempts = 0
            
            for game in games:
                plays = game['plays']
                offensive_plays = plays[plays['posteam'] == team]
                
                if not offensive_plays.empty:
                    # Short yardage plays (3rd/4th and 3 or less)
                    short_yardage_plays = offensive_plays[
                        ((offensive_plays['down'] == 3) | (offensive_plays['down'] == 4)) &
                        (offensive_plays['ydstogo'] <= 3)
                    ]
                    
                    if not short_yardage_plays.empty:
                        # Count conversions
                        conversions = short_yardage_plays[short_yardage_plays['first_down'] == 1]
                        total_short_yardage_conversions += len(conversions)
                        total_short_yardage_attempts += len(short_yardage_plays)
            
            return total_short_yardage_conversions / total_short_yardage_attempts if total_short_yardage_attempts > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating short yardage efficiency for {team}: {e}")
            return 0.0
    
    def _calculate_efficiency_trend(self, team: str, games: List[Dict], efficiency_type: str) -> float:
        """Calculate trend in efficiency over recent games."""
        try:
            if len(games) < 3:
                return 0.0
            
            # Calculate efficiency for each game
            game_efficiencies = []
            for game in games:
                if efficiency_type == 'red_zone':
                    efficiency = self._calculate_red_zone_efficiency(team, [game])
                elif efficiency_type == 'third_down':
                    efficiency = self._calculate_third_down_efficiency(team, [game])
                else:
                    efficiency = 0.0
                
                game_efficiencies.append(efficiency)
            
            if len(game_efficiencies) < 2:
                return 0.0
            
            # Calculate trend using linear regression
            x = np.arange(len(game_efficiencies))
            y = np.array(game_efficiencies)
            
            # Simple linear trend calculation
            trend = np.polyfit(x, y, 1)[0]  # Slope of the line
            
            return trend
            
        except Exception as e:
            logger.error(f"Error calculating {efficiency_type} trend for {team}: {e}")
            return 0.0
    
    def _calculate_situational_consistency(self, team: str, games: List[Dict]) -> float:
        """Calculate consistency across different situational metrics."""
        try:
            if len(games) < 2:
                return 1.0
            
            # Calculate efficiency for each game
            game_red_zone = []
            game_third_down = []
            
            for game in games:
                red_zone = self._calculate_red_zone_efficiency(team, [game])
                third_down = self._calculate_third_down_efficiency(team, [game])
                
                game_red_zone.append(red_zone)
                game_third_down.append(third_down)
            
            if len(game_red_zone) < 2:
                return 1.0
            
            # Calculate consistency as inverse of coefficient of variation
            red_zone_consistency = self._calculate_consistency(game_red_zone)
            third_down_consistency = self._calculate_consistency(game_third_down)
            
            # Average consistency
            return (red_zone_consistency + third_down_consistency) / 2
            
        except Exception as e:
            logger.error(f"Error calculating situational consistency for {team}: {e}")
            return 1.0
    
    def _calculate_consistency(self, values: List[float]) -> float:
        """Calculate consistency as inverse of coefficient of variation."""
        try:
            if len(values) < 2:
                return 1.0
            
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            if std_val == 0:
                return 1.0
            
            consistency = 1.0 / (1.0 + std_val / abs(mean_val))
            return consistency
            
        except Exception as e:
            logger.error(f"Error calculating consistency: {e}")
            return 1.0
    
    def _get_fallback_features(self) -> Dict[str, float]:
        """Get fallback situational efficiency features when data is unavailable."""
        return {
            'red_zone_td_percentage': 0.0,
            'third_down_conversion_rate': 0.0,
            'fourth_down_conversion_rate': 0.0,
            'two_minute_drill_efficiency': 0.0,
            'goal_line_efficiency': 0.0,
            'short_yardage_efficiency': 0.0,
            'red_zone_trend': 0.0,
            'third_down_trend': 0.0,
            'situational_consistency': 1.0
        }
    
    def get_team_situational_comparison(self, home_team: str, away_team: str, season: int, week: int) -> Dict[str, float]:
        """Get situational efficiency comparison features between two teams."""
        try:
            home_features = self.extract_situational_efficiency(home_team, season, week)
            away_features = self.extract_situational_efficiency(away_team, season, week)
            
            comparison_features = {}
            
            # Calculate differentials
            for key in home_features:
                if key in away_features:
                    comparison_features[f"{key}_diff"] = home_features[key] - away_features[key]
                    comparison_features[f"{key}_ratio"] = (
                        (home_features[key] + 0.1) / (away_features[key] + 0.1)
                        if away_features[key] != -0.1 else 1.0
                    )
            
            return comparison_features
            
        except Exception as e:
            logger.error(f"Error getting situational efficiency comparison for {home_team} vs {away_team}: {e}")
            return {}


# Global situational efficiency extractor instance
situational_efficiency_extractor = SituationalEfficiencyExtractor()
