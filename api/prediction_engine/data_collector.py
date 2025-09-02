"""
Data collection module for the Gridiron Guru prediction engine.

This module handles fetching, caching, and processing NFL data from multiple sources:
- nfl_data_py (real-time data)
- Custom JSON files (historical data)
- Caching and error handling
"""

import json
import os
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path

# Import nfl_data_py
try:
    import nfl_data_py as nfl
    NFL_DATA_AVAILABLE = True
except ImportError:
    NFL_DATA_AVAILABLE = False
    logging.warning("nfl_data_py not available. Using fallback data only.")

from .data_models import TeamStats, GameContext, GamePrediction

logger = logging.getLogger(__name__)


class DataCollector:
    """
    Main data collection class that handles multiple data sources.
    
    Sources:
    1. nfl_data_py - Real-time NFL data
    2. game_log.json - Historical game-by-game data (2008-2024)
    3. season_data_by_team.json - Season-level team aggregates (2008-2024)
    """
    
    def __init__(self, data_dir: str = "api/data"):
        """Initialize the data collector with data directory."""
        self.data_dir = Path(data_dir)
        self.cache_dir = self.data_dir / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        
        # Cache settings
        self.cache_duration = timedelta(hours=6)  # Cache for 6 hours
        
        # Data sources
        self.game_log_file = self.data_dir / "game_log.json"
        self.season_data_file = self.data_dir / "season_data_by_team.json"
        
        # In-memory cache
        self._game_log_cache: Optional[Dict] = None
        self._season_data_cache: Optional[Dict] = None
        self._nfl_data_cache: Dict[str, Any] = {}
        
        logger.info(f"DataCollector initialized with data directory: {self.data_dir}")
    
    def _load_json_file(self, file_path: Path) -> Dict:
        """Load and cache a JSON file."""
        try:
            if file_path.exists():
                with open(file_path, 'r') as f:
                    data = json.load(f)
                logger.info(f"Loaded {file_path.name} with {len(data)} records")
                return data
            else:
                logger.warning(f"File not found: {file_path}")
                return {}
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return {}
    
    def _get_cached_data(self, cache_key: str) -> Optional[Any]:
        """Get data from cache if it's still valid."""
        if cache_key in self._nfl_data_cache:
            cached_data, timestamp = self._nfl_data_cache[cache_key]
            if datetime.now() - timestamp < self.cache_duration:
                return cached_data
            else:
                # Remove expired cache
                del self._nfl_data_cache[cache_key]
        return None
    
    def _set_cached_data(self, cache_key: str, data: Any) -> None:
        """Cache data with timestamp."""
        self._nfl_data_cache[cache_key] = (data, datetime.now())
    
    def get_game_log_data(self) -> Dict:
        """Get historical game-by-game data from JSON file."""
        if self._game_log_cache is None:
            self._game_log_cache = self._load_json_file(self.game_log_file)
        return self._game_log_cache
    
    def get_season_data(self) -> Dict:
        """Get season-level team data from JSON file."""
        if self._season_data_cache is None:
            self._season_data_cache = self._load_json_file(self.season_data_file)
        return self._season_data_cache
    
    def get_nfl_data(self, data_type: str, **kwargs) -> pd.DataFrame:
        """
        Get data from nfl_data_py with caching.
        
        Args:
            data_type: Type of data to fetch ('schedules', 'team_stats', 'player_stats', etc.)
            **kwargs: Additional parameters for nfl_data_py functions
        """
        if not NFL_DATA_AVAILABLE:
            logger.warning("nfl_data_py not available, returning empty DataFrame")
            return pd.DataFrame()
        
        # Create cache key
        cache_key = f"{data_type}_{hash(str(kwargs))}"
        
        # Check cache first
        cached_data = self._get_cached_data(cache_key)
        if cached_data is not None:
            logger.info(f"Using cached {data_type} data")
            return cached_data
        
        try:
            # Fetch data based on type
            if data_type == "schedules":
                seasons = kwargs.get('seasons', [datetime.now().year])
                data = nfl.import_schedules(seasons)
            elif data_type == "team_stats":
                seasons = kwargs.get('seasons', [datetime.now().year])
                data = nfl.import_team_desc(seasons)
            elif data_type == "player_stats":
                seasons = kwargs.get('seasons', [datetime.now().year])
                data = nfl.import_player_stats(seasons)
            elif data_type == "play_by_play":
                seasons = kwargs.get('seasons', [datetime.now().year])
                data = nfl.import_pbp_data(seasons)
            else:
                logger.error(f"Unknown data type: {data_type}")
                return pd.DataFrame()
            
            # Cache the data
            self._set_cached_data(cache_key, data)
            logger.info(f"Fetched and cached {data_type} data: {len(data)} records")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching {data_type} data: {e}")
            return pd.DataFrame()
    
    def get_team_stats(self, team: str, season: int, week: int) -> TeamStats:
        """
        Get comprehensive team statistics for a specific team, season, and week.
        
        Combines data from multiple sources:
        1. Historical JSON data
        2. nfl_data_py real-time data
        3. Calculated metrics
        """
        try:
            # Get historical data
            game_log = self.get_game_log_data()
            season_data = self.get_season_data()
            
            # Get real-time data
            nfl_data = self.get_nfl_data("team_stats", seasons=[season])
            
            # Initialize with defaults
            team_stats = TeamStats(
                team_abbr=team,
                team_name=self._get_team_name(team),
                season=season,
                week=week
            )
            
            # Fill in data from historical sources
            self._populate_from_historical_data(team_stats, game_log, season_data, season, week)
            
            # Fill in data from nfl_data_py
            self._populate_from_nfl_data(team_stats, nfl_data, season, week)
            
            # Calculate derived metrics
            self._calculate_derived_metrics(team_stats, game_log, season, week)
            
            return team_stats
            
        except Exception as e:
            logger.error(f"Error getting team stats for {team} {season} week {week}: {e}")
            # Return basic stats with defaults
            return TeamStats(
                team_abbr=team,
                team_name=self._get_team_name(team),
                season=season,
                week=week
            )
    
    def _get_team_name(self, team_abbr: str) -> str:
        """Get full team name from abbreviation."""
        team_names = {
            'KC': 'Kansas City Chiefs', 'BUF': 'Buffalo Bills', 'MIA': 'Miami Dolphins',
            'NE': 'New England Patriots', 'NYJ': 'New York Jets', 'BAL': 'Baltimore Ravens',
            'CIN': 'Cincinnati Bengals', 'CLE': 'Cleveland Browns', 'PIT': 'Pittsburgh Steelers',
            'HOU': 'Houston Texans', 'IND': 'Indianapolis Colts', 'JAX': 'Jacksonville Jaguars',
            'TEN': 'Tennessee Titans', 'DEN': 'Denver Broncos', 'LV': 'Las Vegas Raiders',
            'LAC': 'Los Angeles Chargers', 'DAL': 'Dallas Cowboys', 'NYG': 'New York Giants',
            'PHI': 'Philadelphia Eagles', 'WAS': 'Washington Commanders', 'CHI': 'Chicago Bears',
            'DET': 'Detroit Lions', 'GB': 'Green Bay Packers', 'MIN': 'Minnesota Vikings',
            'ATL': 'Atlanta Falcons', 'CAR': 'Carolina Panthers', 'NO': 'New Orleans Saints',
            'TB': 'Tampa Bay Buccaneers', 'ARI': 'Arizona Cardinals', 'LAR': 'Los Angeles Rams',
            'SF': 'San Francisco 49ers', 'SEA': 'Seattle Seahawks'
        }
        return team_names.get(team_abbr, team_abbr)
    
    def _populate_from_historical_data(self, team_stats: TeamStats, game_log: Dict, 
                                     season_data: Dict, season: int, week: int) -> None:
        """Populate team stats from historical JSON data."""
        try:
            # Get season data for the team
            season_key = f"{season}_{team_stats.team_abbr}"
            if season_key in season_data:
                season_info = season_data[season_key]
                
                # Basic stats
                team_stats.win_percentage = season_info.get('win_percentage', 0.0)
                team_stats.point_differential = season_info.get('point_differential', 0.0)
                team_stats.strength_of_schedule = season_info.get('strength_of_schedule', 0.0)
                
                # Offensive stats
                team_stats.offensive_epa = season_info.get('offensive_epa', 0.0)
                team_stats.passing_epa = season_info.get('passing_epa', 0.0)
                team_stats.rushing_epa = season_info.get('rushing_epa', 0.0)
                team_stats.red_zone_efficiency = season_info.get('red_zone_efficiency', 0.0)
                team_stats.third_down_conversion = season_info.get('third_down_conversion', 0.0)
                
                # Defensive stats
                team_stats.defensive_epa = season_info.get('defensive_epa', 0.0)
                team_stats.pass_defense_epa = season_info.get('pass_defense_epa', 0.0)
                team_stats.rush_defense_epa = season_info.get('rush_defense_epa', 0.0)
                team_stats.turnover_margin = season_info.get('turnover_margin', 0.0)
                team_stats.sack_rate = season_info.get('sack_rate', 0.0)
                
                # Records
                team_stats.home_record = season_info.get('home_record', {})
                team_stats.away_record = season_info.get('away_record', {})
                team_stats.division_record = season_info.get('division_record', {})
                
        except Exception as e:
            logger.error(f"Error populating historical data: {e}")
    
    def _populate_from_nfl_data(self, team_stats: TeamStats, nfl_data: pd.DataFrame, 
                               season: int, week: int) -> None:
        """Populate team stats from nfl_data_py data."""
        try:
            if nfl_data.empty:
                return
            
            # Filter for the specific team and season
            team_data = nfl_data[
                (nfl_data['team_abbr'] == team_stats.team_abbr) & 
                (nfl_data['season'] == season)
            ]
            
            if not team_data.empty:
                # Update with real-time data if available
                latest_data = team_data.iloc[-1]  # Get most recent data
                
                # Update metrics that might be more current
                if 'win_percentage' in latest_data:
                    team_stats.win_percentage = float(latest_data['win_percentage'])
                if 'point_differential' in latest_data:
                    team_stats.point_differential = float(latest_data['point_differential'])
                
        except Exception as e:
            logger.error(f"Error populating nfl_data: {e}")
    
    def _calculate_derived_metrics(self, team_stats: TeamStats, game_log: Dict, 
                                 season: int, week: int) -> None:
        """Calculate derived metrics like recent form and injury impact."""
        try:
            # Calculate recent form (last 4 games)
            recent_games = self._get_recent_games(team_stats.team_abbr, season, week, game_log, 4)
            if recent_games:
                wins = sum(1 for game in recent_games if game.get('won', False))
                team_stats.recent_form = wins / len(recent_games)
            
            # Calculate Pythagorean wins
            if team_stats.point_differential != 0:
                team_stats.pythagorean_wins = self._calculate_pythagorean_wins(
                    team_stats.point_differential, season, week
                )
                team_stats.luck_factor = team_stats.win_percentage - team_stats.pythagorean_wins
            
            # Calculate injury impact (placeholder - would need injury data)
            team_stats.injury_impact = 0.0  # TODO: Implement injury analysis
            
        except Exception as e:
            logger.error(f"Error calculating derived metrics: {e}")
    
    def _get_recent_games(self, team: str, season: int, week: int, game_log: Dict, count: int) -> List[Dict]:
        """Get recent games for a team."""
        try:
            recent_games = []
            for week_num in range(max(1, week - count), week):
                game_key = f"{season}_{week_num}_{team}"
                if game_key in game_log:
                    recent_games.append(game_log[game_key])
            return recent_games
        except Exception as e:
            logger.error(f"Error getting recent games: {e}")
            return []
    
    def _calculate_pythagorean_wins(self, point_diff: float, season: int, week: int) -> float:
        """Calculate expected wins using Pythagorean theorem."""
        try:
            # Simple Pythagorean calculation
            # This would be more sophisticated in a real implementation
            games_played = week - 1
            if games_played <= 0:
                return 0.0
            
            # Basic Pythagorean formula (simplified)
            expected_wins = games_played * (0.5 + (point_diff / (games_played * 20)))
            return max(0.0, min(1.0, expected_wins / games_played))
        except Exception as e:
            logger.error(f"Error calculating Pythagorean wins: {e}")
            return 0.0
    
    def get_game_context(self, game_id: str, season: int, week: int) -> GameContext:
        """Get comprehensive game context for a specific game."""
        try:
            # Get game data from multiple sources
            nfl_schedules = self.get_nfl_data("schedules", seasons=[season])
            game_log = self.get_game_log_data()
            
            # Initialize with defaults
            game_context = GameContext(
                game_id=game_id,
                season=season,
                week=week,
                home_team="",
                away_team="",
                game_date=datetime.now()
            )
            
            # Fill in from nfl_data_py
            if not nfl_schedules.empty:
                game_data = nfl_schedules[nfl_schedules['game_id'] == game_id]
                if not game_data.empty:
                    game = game_data.iloc[0]
                    game_context.home_team = game.get('home_team', '')
                    game_context.away_team = game.get('away_team', '')
                    game_context.stadium = game.get('stadium', '')
                    game_context.surface = game.get('surface', 'grass')
                    game_context.roof = game.get('roof', 'outdoors')
                    game_context.temperature = game.get('temp', None)
                    game_context.wind_speed = game.get('wind', None)
                    game_context.spread = game.get('spread_line', None)
                    game_context.total = game.get('total_line', None)
                    game_context.home_moneyline = game.get('home_moneyline', None)
                    game_context.away_moneyline = game.get('away_moneyline', None)
            
            # Fill in from historical data
            game_key = f"{season}_{week}_{game_context.home_team}_{game_context.away_team}"
            if game_key in game_log:
                historical_game = game_log[game_key]
                # Update with any additional context from historical data
                if 'weather' in historical_game:
                    weather = historical_game['weather']
                    game_context.temperature = weather.get('temperature', game_context.temperature)
                    game_context.wind_speed = weather.get('wind_speed', game_context.wind_speed)
                    game_context.precipitation = weather.get('precipitation', game_context.precipitation)
            
            return game_context
            
        except Exception as e:
            logger.error(f"Error getting game context for {game_id}: {e}")
            return GameContext(
                game_id=game_id,
                season=season,
                week=week,
                home_team="",
                away_team="",
                game_date=datetime.now()
            )
    
    def get_available_seasons(self) -> List[int]:
        """Get list of available seasons from all data sources."""
        seasons = set()
        
        # From historical data
        game_log = self.get_game_log_data()
        for key in game_log.keys():
            if '_' in key:
                season = int(key.split('_')[0])
                seasons.add(season)
        
        # From nfl_data_py (if available)
        if NFL_DATA_AVAILABLE:
            try:
                # Get available seasons from nfl_data_py
                current_year = datetime.now().year
                for year in range(2008, current_year + 1):
                    seasons.add(year)
            except Exception as e:
                logger.error(f"Error getting available seasons: {e}")
        
        return sorted(list(seasons))
    
    def get_available_weeks(self, season: int) -> List[int]:
        """Get list of available weeks for a specific season."""
        weeks = set()
        
        # From historical data
        game_log = self.get_game_log_data()
        for key in game_log.keys():
            if key.startswith(f"{season}_"):
                parts = key.split('_')
                if len(parts) >= 2:
                    try:
                        week = int(parts[1])
                        weeks.add(week)
                    except ValueError:
                        continue
        
        # From nfl_data_py
        if NFL_DATA_AVAILABLE:
            try:
                schedules = self.get_nfl_data("schedules", seasons=[season])
                if not schedules.empty and 'week' in schedules.columns:
                    weeks.update(schedules['week'].unique())
            except Exception as e:
                logger.error(f"Error getting available weeks: {e}")
        
        return sorted(list(weeks))
    
    def validate_data_integrity(self) -> Dict[str, Any]:
        """Validate the integrity of all data sources."""
        validation_results = {
            'nfl_data_py_available': NFL_DATA_AVAILABLE,
            'game_log_available': self.game_log_file.exists(),
            'season_data_available': self.season_data_file.exists(),
            'data_quality': {},
            'errors': []
        }
        
        try:
            # Validate game log data
            if validation_results['game_log_available']:
                game_log = self.get_game_log_data()
                validation_results['data_quality']['game_log_records'] = len(game_log)
                validation_results['data_quality']['game_log_seasons'] = len(set(
                    key.split('_')[0] for key in game_log.keys() if '_' in key
                ))
            
            # Validate season data
            if validation_results['season_data_available']:
                season_data = self.get_season_data()
                validation_results['data_quality']['season_data_records'] = len(season_data)
                validation_results['data_quality']['season_data_seasons'] = len(set(
                    key.split('_')[0] for key in season_data.keys() if '_' in key
                ))
            
            # Test nfl_data_py if available
            if NFL_DATA_AVAILABLE:
                try:
                    test_data = self.get_nfl_data("schedules", seasons=[2024])
                    validation_results['data_quality']['nfl_data_records'] = len(test_data)
                except Exception as e:
                    validation_results['errors'].append(f"nfl_data_py test failed: {e}")
            
        except Exception as e:
            validation_results['errors'].append(f"Validation error: {e}")
        
        return validation_results


# Global data collector instance
data_collector = DataCollector()
