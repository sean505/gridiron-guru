"""
EPA Trends Feature Extractor

This module extracts EPA (Expected Points Added) trend features for NFL teams,
including rolling averages, differentials, and trend analysis for the last 4 games.
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
    logging.warning("nfl_data_py not available. EPA trends will use fallback data.")

logger = logging.getLogger(__name__)


class EPATrendsExtractor:
    """
    Extracts EPA trend features for NFL teams.
    
    Features:
    - Team offensive EPA per play (last 4 games)
    - Team defensive EPA per play allowed (last 4 games)
    - EPA differential trends (improving/declining)
    - EPA momentum indicators
    """
    
    def __init__(self, cache_dir: str = "api/data/cache"):
        """Initialize the EPA trends extractor."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache settings
        self.cache_duration = timedelta(hours=6)  # Cache for 6 hours
        self.cache_file = self.cache_dir / "epa_trends_cache.json"
        
        # Load existing cache
        self.cache = self._load_cache()
        
        logger.info("EPATrendsExtractor initialized")
    
    def _load_cache(self) -> Dict[str, Any]:
        """Load EPA trends cache from file."""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.warning(f"Error loading EPA cache: {e}")
            return {}
    
    def _save_cache(self):
        """Save EPA trends cache to file."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, default=str)
        except Exception as e:
            logger.warning(f"Error saving EPA cache: {e}")
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid."""
        if cache_key not in self.cache:
            return False
        
        cached_time = datetime.fromisoformat(self.cache[cache_key]['timestamp'])
        return datetime.now() - cached_time < self.cache_duration
    
    def extract_epa_trends(self, team: str, season: int, week: int) -> Dict[str, float]:
        """
        Extract EPA trend features for a team.
        
        Args:
            team: Team abbreviation (e.g., 'KC')
            season: NFL season year
            week: Week number
            
        Returns:
            Dictionary of EPA trend features
        """
        try:
            cache_key = f"{team}_{season}_{week}"
            
            # Check cache first
            if self._is_cache_valid(cache_key):
                logger.info(f"Using cached EPA trends for {team}")
                return self.cache[cache_key]['features']
            
            # Extract fresh data
            features = self._calculate_epa_trends(team, season, week)
            
            # Cache the results
            self.cache[cache_key] = {
                'features': features,
                'timestamp': datetime.now().isoformat()
            }
            self._save_cache()
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting EPA trends for {team}: {e}")
            return self._get_fallback_features()
    
    def _calculate_epa_trends(self, team: str, season: int, week: int) -> Dict[str, float]:
        """Calculate EPA trend features from play-by-play data."""
        try:
            if not NFL_DATA_AVAILABLE:
                logger.warning("nfl_data_py not available, using fallback EPA features")
                return self._get_fallback_features()
            
            # Get play-by-play data for recent games
            recent_games = self._get_recent_games(team, season, week, 4)
            
            if not recent_games:
                logger.warning(f"No recent games found for {team}, using fallback")
                return self._get_fallback_features()
            
            # Calculate EPA metrics
            offensive_epa = self._calculate_offensive_epa(team, recent_games)
            defensive_epa = self._calculate_defensive_epa(team, recent_games)
            epa_differential = offensive_epa - defensive_epa
            
            # Calculate trends
            epa_trend = self._calculate_epa_trend(team, recent_games)
            momentum = self._calculate_epa_momentum(team, recent_games)
            
            # Calculate consistency
            consistency = self._calculate_epa_consistency(team, recent_games)
            
            features = {
                'offensive_epa_per_play': offensive_epa,
                'defensive_epa_per_play': defensive_epa,
                'epa_differential': epa_differential,
                'epa_trend': epa_trend,
                'epa_momentum': momentum,
                'epa_consistency': consistency,
                'offensive_epa_improvement': self._calculate_improvement(team, recent_games, 'offensive'),
                'defensive_epa_improvement': self._calculate_improvement(team, recent_games, 'defensive')
            }
            
            logger.info(f"Calculated EPA trends for {team}: {len(features)} features")
            return features
            
        except Exception as e:
            logger.error(f"Error calculating EPA trends for {team}: {e}")
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
    
    def _calculate_offensive_epa(self, team: str, games: List[Dict]) -> float:
        """Calculate offensive EPA per play for recent games."""
        try:
            total_epa = 0.0
            total_plays = 0
            
            for game in games:
                plays = game['plays']
                offensive_plays = plays[plays['posteam'] == team]
                
                if not offensive_plays.empty and 'epa' in offensive_plays.columns:
                    total_epa += offensive_plays['epa'].sum()
                    total_plays += len(offensive_plays)
            
            return total_epa / total_plays if total_plays > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating offensive EPA for {team}: {e}")
            return 0.0
    
    def _calculate_defensive_epa(self, team: str, games: List[Dict]) -> float:
        """Calculate defensive EPA per play allowed for recent games."""
        try:
            total_epa_allowed = 0.0
            total_plays_faced = 0
            
            for game in games:
                plays = game['plays']
                defensive_plays = plays[plays['defteam'] == team]
                
                if not defensive_plays.empty and 'epa' in defensive_plays.columns:
                    # EPA allowed is negative of defensive EPA
                    total_epa_allowed += (-defensive_plays['epa']).sum()
                    total_plays_faced += len(defensive_plays)
            
            return total_epa_allowed / total_plays_faced if total_plays_faced > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating defensive EPA for {team}: {e}")
            return 0.0
    
    def _calculate_epa_trend(self, team: str, games: List[Dict]) -> float:
        """Calculate EPA trend (improving/declining) over recent games."""
        try:
            if len(games) < 2:
                return 0.0
            
            # Calculate EPA for each game
            game_epas = []
            for game in games:
                plays = game['plays']
                offensive_plays = plays[plays['posteam'] == team]
                defensive_plays = plays[plays['defteam'] == team]
                
                if not offensive_plays.empty and not defensive_plays.empty and 'epa' in plays.columns:
                    off_epa = offensive_plays['epa'].sum() / len(offensive_plays) if len(offensive_plays) > 0 else 0
                    def_epa = (-defensive_plays['epa']).sum() / len(defensive_plays) if len(defensive_plays) > 0 else 0
                    game_epas.append(off_epa - def_epa)
            
            if len(game_epas) < 2:
                return 0.0
            
            # Calculate trend using linear regression
            x = np.arange(len(game_epas))
            y = np.array(game_epas)
            
            # Simple linear trend calculation
            trend = np.polyfit(x, y, 1)[0]  # Slope of the line
            
            return trend
            
        except Exception as e:
            logger.error(f"Error calculating EPA trend for {team}: {e}")
            return 0.0
    
    def _calculate_epa_momentum(self, team: str, games: List[Dict]) -> float:
        """Calculate EPA momentum (recent performance vs earlier performance)."""
        try:
            if len(games) < 3:
                return 0.0
            
            # Split games into recent and earlier
            mid_point = len(games) // 2
            recent_games = games[:mid_point]
            earlier_games = games[mid_point:]
            
            recent_epa = self._calculate_offensive_epa(team, recent_games) - self._calculate_defensive_epa(team, recent_games)
            earlier_epa = self._calculate_offensive_epa(team, earlier_games) - self._calculate_defensive_epa(team, earlier_games)
            
            return recent_epa - earlier_epa
            
        except Exception as e:
            logger.error(f"Error calculating EPA momentum for {team}: {e}")
            return 0.0
    
    def _calculate_epa_consistency(self, team: str, games: List[Dict]) -> float:
        """Calculate EPA consistency (inverse of variance)."""
        try:
            if len(games) < 2:
                return 1.0
            
            # Calculate EPA for each game
            game_epas = []
            for game in games:
                plays = game['plays']
                offensive_plays = plays[plays['posteam'] == team]
                defensive_plays = plays[plays['defteam'] == team]
                
                if not offensive_plays.empty and not defensive_plays.empty and 'epa' in plays.columns:
                    off_epa = offensive_plays['epa'].sum() / len(offensive_plays) if len(offensive_plays) > 0 else 0
                    def_epa = (-defensive_plays['epa']).sum() / len(defensive_plays) if len(defensive_plays) > 0 else 0
                    game_epas.append(off_epa - def_epa)
            
            if len(game_epas) < 2:
                return 1.0
            
            # Calculate consistency as inverse of coefficient of variation
            mean_epa = np.mean(game_epas)
            std_epa = np.std(game_epas)
            
            if std_epa == 0:
                return 1.0
            
            consistency = 1.0 / (1.0 + std_epa / abs(mean_epa))
            return consistency
            
        except Exception as e:
            logger.error(f"Error calculating EPA consistency for {team}: {e}")
            return 1.0
    
    def _calculate_improvement(self, team: str, games: List[Dict], side: str) -> float:
        """Calculate improvement in EPA over recent games."""
        try:
            if len(games) < 2:
                return 0.0
            
            # Split games into halves
            mid_point = len(games) // 2
            recent_games = games[:mid_point]
            earlier_games = games[mid_point:]
            
            if side == 'offensive':
                recent_epa = self._calculate_offensive_epa(team, recent_games)
                earlier_epa = self._calculate_offensive_epa(team, earlier_games)
            else:  # defensive
                recent_epa = self._calculate_defensive_epa(team, recent_games)
                earlier_epa = self._calculate_defensive_epa(team, earlier_games)
            
            return recent_epa - earlier_epa
            
        except Exception as e:
            logger.error(f"Error calculating {side} improvement for {team}: {e}")
            return 0.0
    
    def _get_fallback_features(self) -> Dict[str, float]:
        """Get fallback EPA features when data is unavailable."""
        return {
            'offensive_epa_per_play': 0.0,
            'defensive_epa_per_play': 0.0,
            'epa_differential': 0.0,
            'epa_trend': 0.0,
            'epa_momentum': 0.0,
            'epa_consistency': 1.0,
            'offensive_epa_improvement': 0.0,
            'defensive_epa_improvement': 0.0
        }
    
    def get_team_epa_comparison(self, home_team: str, away_team: str, season: int, week: int) -> Dict[str, float]:
        """Get EPA comparison features between two teams."""
        try:
            home_features = self.extract_epa_trends(home_team, season, week)
            away_features = self.extract_epa_trends(away_team, season, week)
            
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
            logger.error(f"Error getting EPA comparison for {home_team} vs {away_team}: {e}")
            return {}


# Global EPA trends extractor instance
epa_trends_extractor = EPATrendsExtractor()
