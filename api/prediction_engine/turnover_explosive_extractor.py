"""
Turnover and Explosive Play Feature Extractor

This module extracts turnover and explosive play features for NFL teams,
including turnover differentials, explosive play rates, and momentum indicators.
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
    logging.warning("nfl_data_py not available. Turnover and explosive play features will use fallback data.")

logger = logging.getLogger(__name__)


class TurnoverExplosiveExtractor:
    """
    Extracts turnover and explosive play features for NFL teams.
    
    Features:
    - Turnover differential per game (last 6 games)
    - Explosive play rate (20+ yard plays per game)
    - Explosive plays allowed per game
    - Turnover margin trends
    - Explosive play consistency
    - Big play differential
    """
    
    def __init__(self, cache_dir: str = "api/data/cache"):
        """Initialize the turnover and explosive play extractor."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache settings
        self.cache_duration = timedelta(hours=6)  # Cache for 6 hours
        self.cache_file = self.cache_dir / "turnover_explosive_cache.json"
        
        # Load existing cache
        self.cache = self._load_cache()
        
        logger.info("TurnoverExplosiveExtractor initialized")
    
    def _load_cache(self) -> Dict[str, Any]:
        """Load turnover and explosive play cache from file."""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.warning(f"Error loading turnover explosive cache: {e}")
            return {}
    
    def _save_cache(self):
        """Save turnover and explosive play cache to file."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, default=str)
        except Exception as e:
            logger.warning(f"Error saving turnover explosive cache: {e}")
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid."""
        if cache_key not in self.cache:
            return False
        
        cached_time = datetime.fromisoformat(self.cache[cache_key]['timestamp'])
        return datetime.now() - cached_time < self.cache_duration
    
    def extract_turnover_explosive_features(self, team: str, season: int, week: int) -> Dict[str, float]:
        """
        Extract turnover and explosive play features for a team.
        
        Args:
            team: Team abbreviation (e.g., 'KC')
            season: NFL season year
            week: Week number
            
        Returns:
            Dictionary of turnover and explosive play features
        """
        try:
            cache_key = f"{team}_{season}_{week}"
            
            # Check cache first
            if self._is_cache_valid(cache_key):
                logger.info(f"Using cached turnover explosive features for {team}")
                return self.cache[cache_key]['features']
            
            # Extract fresh data
            features = self._calculate_turnover_explosive_features(team, season, week)
            
            # Cache the results
            self.cache[cache_key] = {
                'features': features,
                'timestamp': datetime.now().isoformat()
            }
            self._save_cache()
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting turnover explosive features for {team}: {e}")
            return self._get_fallback_features()
    
    def _calculate_turnover_explosive_features(self, team: str, season: int, week: int) -> Dict[str, float]:
        """Calculate turnover and explosive play features from play-by-play data."""
        try:
            if not NFL_DATA_AVAILABLE:
                logger.warning("nfl_data_py not available, using fallback turnover explosive features")
                return self._get_fallback_features()
            
            # Get play-by-play data for recent games
            recent_games = self._get_recent_games(team, season, week, 6)
            
            if not recent_games:
                logger.warning(f"No recent games found for {team}, using fallback")
                return self._get_fallback_features()
            
            # Calculate turnover features
            turnover_differential = self._calculate_turnover_differential(team, recent_games)
            turnover_margin = self._calculate_turnover_margin(team, recent_games)
            turnover_trend = self._calculate_turnover_trend(team, recent_games)
            
            # Calculate explosive play features
            explosive_play_rate = self._calculate_explosive_play_rate(team, recent_games)
            explosive_plays_allowed = self._calculate_explosive_plays_allowed(team, recent_games)
            explosive_play_differential = explosive_play_rate - explosive_plays_allowed
            
            # Calculate big play features
            big_play_rate = self._calculate_big_play_rate(team, recent_games)
            big_plays_allowed = self._calculate_big_plays_allowed(team, recent_games)
            
            # Calculate consistency and momentum
            explosive_consistency = self._calculate_explosive_consistency(team, recent_games)
            turnover_consistency = self._calculate_turnover_consistency(team, recent_games)
            
            features = {
                'turnover_differential_per_game': turnover_differential,
                'turnover_margin': turnover_margin,
                'turnover_trend': turnover_trend,
                'explosive_play_rate': explosive_play_rate,
                'explosive_plays_allowed': explosive_plays_allowed,
                'explosive_play_differential': explosive_play_differential,
                'big_play_rate': big_play_rate,
                'big_plays_allowed': big_plays_allowed,
                'explosive_consistency': explosive_consistency,
                'turnover_consistency': turnover_consistency
            }
            
            logger.info(f"Calculated turnover explosive features for {team}: {len(features)} features")
            return features
            
        except Exception as e:
            logger.error(f"Error calculating turnover explosive features for {team}: {e}")
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
    
    def _calculate_turnover_differential(self, team: str, games: List[Dict]) -> float:
        """Calculate turnover differential per game."""
        try:
            total_turnovers_forced = 0
            total_turnovers_committed = 0
            total_games = len(games)
            
            for game in games:
                plays = game['plays']
                
                # Offensive turnovers (team committed)
                offensive_plays = plays[plays['posteam'] == team]
                if not offensive_plays.empty:
                    turnovers_committed = len(offensive_plays[
                        (offensive_plays['interception'] == 1) | 
                        (offensive_plays['fumble_lost'] == 1)
                    ])
                    total_turnovers_committed += turnovers_committed
                
                # Defensive turnovers (team forced)
                defensive_plays = plays[plays['defteam'] == team]
                if not defensive_plays.empty:
                    turnovers_forced = len(defensive_plays[
                        (defensive_plays['interception'] == 1) | 
                        (defensive_plays['fumble_lost'] == 1)
                    ])
                    total_turnovers_forced += turnovers_forced
            
            return (total_turnovers_forced - total_turnovers_committed) / total_games if total_games > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating turnover differential for {team}: {e}")
            return 0.0
    
    def _calculate_turnover_margin(self, team: str, games: List[Dict]) -> float:
        """Calculate total turnover margin."""
        try:
            total_turnovers_forced = 0
            total_turnovers_committed = 0
            
            for game in games:
                plays = game['plays']
                
                # Offensive turnovers
                offensive_plays = plays[plays['posteam'] == team]
                if not offensive_plays.empty:
                    turnovers_committed = len(offensive_plays[
                        (offensive_plays['interception'] == 1) | 
                        (offensive_plays['fumble_lost'] == 1)
                    ])
                    total_turnovers_committed += turnovers_committed
                
                # Defensive turnovers
                defensive_plays = plays[plays['defteam'] == team]
                if not defensive_plays.empty:
                    turnovers_forced = len(defensive_plays[
                        (defensive_plays['interception'] == 1) | 
                        (defensive_plays['fumble_lost'] == 1)
                    ])
                    total_turnovers_forced += turnovers_forced
            
            return total_turnovers_forced - total_turnovers_committed
            
        except Exception as e:
            logger.error(f"Error calculating turnover margin for {team}: {e}")
            return 0.0
    
    def _calculate_turnover_trend(self, team: str, games: List[Dict]) -> float:
        """Calculate turnover trend over recent games."""
        try:
            if len(games) < 2:
                return 0.0
            
            # Calculate turnover differential for each game
            game_turnover_diffs = []
            for game in games:
                plays = game['plays']
                
                # Offensive turnovers
                offensive_plays = plays[plays['posteam'] == team]
                turnovers_committed = 0
                if not offensive_plays.empty:
                    turnovers_committed = len(offensive_plays[
                        (offensive_plays['interception'] == 1) | 
                        (offensive_plays['fumble_lost'] == 1)
                    ])
                
                # Defensive turnovers
                defensive_plays = plays[plays['defteam'] == team]
                turnovers_forced = 0
                if not defensive_plays.empty:
                    turnovers_forced = len(defensive_plays[
                        (defensive_plays['interception'] == 1) | 
                        (defensive_plays['fumble_lost'] == 1)
                    ])
                
                game_turnover_diffs.append(turnovers_forced - turnovers_committed)
            
            if len(game_turnover_diffs) < 2:
                return 0.0
            
            # Calculate trend using linear regression
            x = np.arange(len(game_turnover_diffs))
            y = np.array(game_turnover_diffs)
            
            # Simple linear trend calculation
            trend = np.polyfit(x, y, 1)[0]  # Slope of the line
            
            return trend
            
        except Exception as e:
            logger.error(f"Error calculating turnover trend for {team}: {e}")
            return 0.0
    
    def _calculate_explosive_play_rate(self, team: str, games: List[Dict]) -> float:
        """Calculate explosive play rate (20+ yard plays per game)."""
        try:
            total_explosive_plays = 0
            total_games = len(games)
            
            for game in games:
                plays = game['plays']
                offensive_plays = plays[plays['posteam'] == team]
                
                if not offensive_plays.empty:
                    # Explosive plays (20+ yards)
                    explosive_plays = offensive_plays[
                        (offensive_plays['yards_gained'] >= 20) |
                        (offensive_plays['pass_length'] >= 20) |
                        (offensive_plays['rush_distance'] >= 20)
                    ]
                    total_explosive_plays += len(explosive_plays)
            
            return total_explosive_plays / total_games if total_games > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating explosive play rate for {team}: {e}")
            return 0.0
    
    def _calculate_explosive_plays_allowed(self, team: str, games: List[Dict]) -> float:
        """Calculate explosive plays allowed per game."""
        try:
            total_explosive_plays_allowed = 0
            total_games = len(games)
            
            for game in games:
                plays = game['plays']
                defensive_plays = plays[plays['defteam'] == team]
                
                if not defensive_plays.empty:
                    # Explosive plays allowed (20+ yards)
                    explosive_plays_allowed = defensive_plays[
                        (defensive_plays['yards_gained'] >= 20) |
                        (defensive_plays['pass_length'] >= 20) |
                        (defensive_plays['rush_distance'] >= 20)
                    ]
                    total_explosive_plays_allowed += len(explosive_plays_allowed)
            
            return total_explosive_plays_allowed / total_games if total_games > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating explosive plays allowed for {team}: {e}")
            return 0.0
    
    def _calculate_big_play_rate(self, team: str, games: List[Dict]) -> float:
        """Calculate big play rate (40+ yard plays per game)."""
        try:
            total_big_plays = 0
            total_games = len(games)
            
            for game in games:
                plays = game['plays']
                offensive_plays = plays[plays['posteam'] == team]
                
                if not offensive_plays.empty:
                    # Big plays (40+ yards)
                    big_plays = offensive_plays[
                        (offensive_plays['yards_gained'] >= 40) |
                        (offensive_plays['pass_length'] >= 40) |
                        (offensive_plays['rush_distance'] >= 40)
                    ]
                    total_big_plays += len(big_plays)
            
            return total_big_plays / total_games if total_games > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating big play rate for {team}: {e}")
            return 0.0
    
    def _calculate_big_plays_allowed(self, team: str, games: List[Dict]) -> float:
        """Calculate big plays allowed per game."""
        try:
            total_big_plays_allowed = 0
            total_games = len(games)
            
            for game in games:
                plays = game['plays']
                defensive_plays = plays[plays['defteam'] == team]
                
                if not defensive_plays.empty:
                    # Big plays allowed (40+ yards)
                    big_plays_allowed = defensive_plays[
                        (defensive_plays['yards_gained'] >= 40) |
                        (defensive_plays['pass_length'] >= 40) |
                        (defensive_plays['rush_distance'] >= 40)
                    ]
                    total_big_plays_allowed += len(big_plays_allowed)
            
            return total_big_plays_allowed / total_games if total_games > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating big plays allowed for {team}: {e}")
            return 0.0
    
    def _calculate_explosive_consistency(self, team: str, games: List[Dict]) -> float:
        """Calculate explosive play consistency."""
        try:
            if len(games) < 2:
                return 1.0
            
            # Calculate explosive play rate for each game
            game_explosive_rates = []
            for game in games:
                plays = game['plays']
                offensive_plays = plays[plays['posteam'] == team]
                
                if not offensive_plays.empty:
                    explosive_plays = offensive_plays[
                        (offensive_plays['yards_gained'] >= 20) |
                        (offensive_plays['pass_length'] >= 20) |
                        (offensive_plays['rush_distance'] >= 20)
                    ]
                    game_explosive_rates.append(len(explosive_plays))
                else:
                    game_explosive_rates.append(0)
            
            if len(game_explosive_rates) < 2:
                return 1.0
            
            # Calculate consistency as inverse of coefficient of variation
            mean_rate = np.mean(game_explosive_rates)
            std_rate = np.std(game_explosive_rates)
            
            if std_rate == 0:
                return 1.0
            
            consistency = 1.0 / (1.0 + std_rate / abs(mean_rate))
            return consistency
            
        except Exception as e:
            logger.error(f"Error calculating explosive consistency for {team}: {e}")
            return 1.0
    
    def _calculate_turnover_consistency(self, team: str, games: List[Dict]) -> float:
        """Calculate turnover consistency."""
        try:
            if len(games) < 2:
                return 1.0
            
            # Calculate turnover differential for each game
            game_turnover_diffs = []
            for game in games:
                plays = game['plays']
                
                # Offensive turnovers
                offensive_plays = plays[plays['posteam'] == team]
                turnovers_committed = 0
                if not offensive_plays.empty:
                    turnovers_committed = len(offensive_plays[
                        (offensive_plays['interception'] == 1) | 
                        (offensive_plays['fumble_lost'] == 1)
                    ])
                
                # Defensive turnovers
                defensive_plays = plays[plays['defteam'] == team]
                turnovers_forced = 0
                if not defensive_plays.empty:
                    turnovers_forced = len(defensive_plays[
                        (defensive_plays['interception'] == 1) | 
                        (defensive_plays['fumble_lost'] == 1)
                    ])
                
                game_turnover_diffs.append(turnovers_forced - turnovers_committed)
            
            if len(game_turnover_diffs) < 2:
                return 1.0
            
            # Calculate consistency as inverse of coefficient of variation
            mean_diff = np.mean(game_turnover_diffs)
            std_diff = np.std(game_turnover_diffs)
            
            if std_diff == 0:
                return 1.0
            
            consistency = 1.0 / (1.0 + std_diff / abs(mean_diff))
            return consistency
            
        except Exception as e:
            logger.error(f"Error calculating turnover consistency for {team}: {e}")
            return 1.0
    
    def _get_fallback_features(self) -> Dict[str, float]:
        """Get fallback turnover and explosive play features when data is unavailable."""
        return {
            'turnover_differential_per_game': 0.0,
            'turnover_margin': 0.0,
            'turnover_trend': 0.0,
            'explosive_play_rate': 0.0,
            'explosive_plays_allowed': 0.0,
            'explosive_play_differential': 0.0,
            'big_play_rate': 0.0,
            'big_plays_allowed': 0.0,
            'explosive_consistency': 1.0,
            'turnover_consistency': 1.0
        }
    
    def get_team_turnover_explosive_comparison(self, home_team: str, away_team: str, season: int, week: int) -> Dict[str, float]:
        """Get turnover and explosive play comparison features between two teams."""
        try:
            home_features = self.extract_turnover_explosive_features(home_team, season, week)
            away_features = self.extract_turnover_explosive_features(away_team, season, week)
            
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
            logger.error(f"Error getting turnover explosive comparison for {home_team} vs {away_team}: {e}")
            return {}


# Global turnover and explosive play extractor instance
turnover_explosive_extractor = TurnoverExplosiveExtractor()
