"""
Temporal Data Collector
Fixes API compatibility issues and implements proper temporal data separation
"""

import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path

# Import nfl_data_py with correct API calls
try:
    import nfl_data_py as nfl
    NFL_DATA_AVAILABLE = True
except ImportError:
    NFL_DATA_AVAILABLE = False
    logging.warning("nfl_data_py not available. Using fallback data only.")

from prediction_engine.data_models import TeamStats, GameContext

logger = logging.getLogger(__name__)


class TemporalDataCollector:
    """
    Temporal data collector that properly separates training and prediction data.
    
    Features:
    - Fixed nfl_data_py API calls
    - Temporal data separation (2008-2024 vs 2024 baseline vs 2025 schedule)
    - Proper fallback strategies
    - Caching for performance
    """
    
    def __init__(self, data_dir: str = "data"):
        """Initialize the temporal data collector."""
        self.data_dir = Path(data_dir)
        self.cache_dir = self.data_dir / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        
        # Cache settings
        self.cache_duration = timedelta(hours=6)
        self._cache = {}
        
        # Data sources
        self.game_log_file = self.data_dir / "game_log_fixed.json"
        self.season_data_file = self.data_dir / "season_data_by_team_fixed.json"
        
        # In-memory cache
        self._game_log_cache: Optional[Dict] = None
        self._season_data_cache: Optional[Dict] = None
        self._2024_baseline_cache: Optional[Dict] = None
        self._2025_schedule_cache: Optional[pd.DataFrame] = None
        
        logger.info(f"TemporalDataCollector initialized with data directory: {self.data_dir}")
    
    def get_training_data(self, years: List[int]) -> Dict[str, Any]:
        """
        Load historical data for training (2008-2024).
        
        Args:
            years: List of years to load (e.g., [2008, 2009, ..., 2024])
            
        Returns:
            Dictionary containing game and team data for training
        """
        try:
            logger.info(f"Loading training data for years: {years}")
            
            # Load historical data
            game_log = self._load_game_log_data()
            season_data = self._load_season_data()
            
            # Filter by requested years
            training_games = {}
            training_teams = {}
            
            for year in years:
                # Filter games for this year
                year_games = {k: v for k, v in game_log.items() 
                             if k.startswith(f"{year}_")}
                training_games.update(year_games)
                
                # Filter team data for this year
                year_teams = {k: v for k, v in season_data.items() 
                             if k.startswith(f"{year}_")}
                training_teams.update(year_teams)
            
            logger.info(f"Loaded {len(training_games)} games and {len(training_teams)} team records")
            
            return {
                'games': training_games,
                'teams': training_teams,
                'years': years,
                'total_games': len(training_games),
                'total_teams': len(training_teams)
            }
            
        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            return {'games': {}, 'teams': {}, 'years': years, 'total_games': 0, 'total_teams': 0}
    
    def get_2024_baseline(self) -> Dict[str, TeamStats]:
        """
        Load 2024 season-end team strength data for 2025 predictions.
        
        Returns:
            Dictionary mapping team names to TeamStats objects
        """
        try:
            if self._2024_baseline_cache is not None:
                return self._2024_baseline_cache
            
            logger.info("Loading 2024 season-end baseline data")
            
            season_data = self._load_season_data()
            baseline = {}
            
            # Get Week 18 data for each team (final regular season week)
            for key, team in season_data.items():
                if '2024_' in key and team.get('Week') == 18:
                    team_name = team.get('Team', '')
                    if team_name:
                        team_stats = self._create_team_stats_from_2024_data(team)
                        baseline[team_name] = team_stats
            
            logger.info(f"Loaded 2024 baseline for {len(baseline)} teams")
            self._2024_baseline_cache = baseline
            return baseline
            
        except Exception as e:
            logger.error(f"Error loading 2024 baseline: {e}")
            return {}
    
    def get_2025_schedule(self) -> pd.DataFrame:
        """
        Load 2025 schedule for predictions using ESPN API.
        
        Returns:
            DataFrame containing 2025 schedule
        """
        try:
            if self._2025_schedule_cache is not None:
                return self._2025_schedule_cache
            
            logger.info("Loading 2025 schedule from ESPN API")
            
            if not NFL_DATA_AVAILABLE:
                logger.warning("nfl_data_py not available, returning empty schedule")
                return pd.DataFrame()
            
            # Use nfl_data_py to get 2025 schedule (same as main system)
            try:
                schedule = nfl.import_schedules([2025])
                logger.info(f"Loaded 2025 schedule from nfl_data_py: {len(schedule)} games")
            except Exception as e:
                logger.warning(f"nfl_data_py failed for 2025: {e}, trying ESPN API fallback")
                # Fallback to ESPN API if nfl_data_py fails
                import requests
                import json
                
                # ESPN API endpoint for NFL schedule
                url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    games = []
                    
                    for event in data.get('events', []):
                        game_data = {
                            'season': 2025,
                            'week': 1,  # ESPN doesn't provide week info in scoreboard
                            'home_team': event.get('competitions', [{}])[0].get('competitors', [{}])[0].get('team', {}).get('abbreviation', ''),
                            'away_team': event.get('competitions', [{}])[0].get('competitors', [{}])[1].get('team', {}).get('abbreviation', ''),
                            'game_type': 'REG'
                        }
                        games.append(game_data)
                    
                    schedule = pd.DataFrame(games)
                    logger.info(f"Loaded 2025 schedule from ESPN API: {len(schedule)} games")
                else:
                    logger.error(f"ESPN API failed with status {response.status_code}")
                    return pd.DataFrame()
            
            self._2025_schedule_cache = schedule
            return schedule
            
        except Exception as e:
            logger.error(f"Error loading 2025 schedule: {e}")
            return pd.DataFrame()
    
    def get_team_stats_for_training(self, team: str, season: int, week: int) -> TeamStats:
        """
        Get team stats for training using historical data.
        
        Args:
            team: Team abbreviation
            season: Season year
            week: Week number
            
        Returns:
            TeamStats object with historical data
        """
        try:
            season_data = self._load_season_data()
            
            # Look for team data for this specific week
            team_key = f"{season}_{week}_{team}"
            if team_key in season_data:
                team_data = season_data[team_key]
                return self._create_team_stats_from_historical_data(team_data, season, week)
            
            # If not found, look for the most recent week before this one
            for check_week in range(week - 1, 0, -1):
                check_key = f"{season}_{check_week}_{team}"
                if check_key in season_data:
                    team_data = season_data[check_key]
                    return self._create_team_stats_from_historical_data(team_data, season, week)
            
            # If still not found, return default stats
            logger.warning(f"No data found for {team} {season} week {week}")
            return self._create_default_team_stats(team, season, week)
            
        except Exception as e:
            logger.error(f"Error getting team stats for training: {e}")
            return self._create_default_team_stats(team, season, week)
    
    def _load_game_log_data(self) -> Dict:
        """Load game log data from JSON file."""
        if self._game_log_cache is None:
            try:
                with open(self.game_log_file, 'r') as f:
                    self._game_log_cache = json.load(f)
                logger.info(f"Loaded game log: {len(self._game_log_cache)} games")
            except Exception as e:
                logger.error(f"Error loading game log: {e}")
                self._game_log_cache = {}
        return self._game_log_cache
    
    def _load_season_data(self) -> Dict:
        """Load season data from JSON file."""
        if self._season_data_cache is None:
            try:
                with open(self.season_data_file, 'r') as f:
                    self._season_data_cache = json.load(f)
                logger.info(f"Loaded season data: {len(self._season_data_cache)} records")
            except Exception as e:
                logger.error(f"Error loading season data: {e}")
                self._season_data_cache = {}
        return self._season_data_cache
    
    def _create_team_stats_from_2024_data(self, team_data: Dict) -> TeamStats:
        """Create TeamStats from 2024 season-end data."""
        try:
            # Calculate win percentage from record
            record = team_data.get('Rec', '0-0')
            wins, losses = map(int, record.split('-'))
            total_games = wins + losses
            win_percentage = wins / total_games if total_games > 0 else 0.0
            
            # Calculate point differential
            points_for = float(team_data.get('Tm', 0))
            points_against = float(team_data.get('Opp', 0))
            point_differential = points_for - points_against
            
            # Create realistic team stats based on 2024 performance
            return TeamStats(
                team_abbr=self._get_team_abbr(team_data.get('Team', '')),
                team_name=team_data.get('Team', ''),
                season=2024,
                week=18,
                win_percentage=win_percentage,
                point_differential=point_differential,
                offensive_epa=self._calculate_offensive_epa(team_data),
                defensive_epa=self._calculate_defensive_epa(team_data),
                passing_epa=self._calculate_passing_epa(team_data),
                rushing_epa=self._calculate_rushing_epa(team_data),
                red_zone_efficiency=self._calculate_red_zone_efficiency(team_data),
                third_down_conversion=self._calculate_third_down_conversion(team_data),
                pass_defense_epa=self._calculate_pass_defense_epa(team_data),
                rush_defense_epa=self._calculate_rush_defense_epa(team_data),
                turnover_margin=self._calculate_turnover_margin(team_data),
                sack_rate=self._calculate_sack_rate(team_data),
                strength_of_schedule=self._calculate_strength_of_schedule(team_data),
                recent_form=self._calculate_recent_form(team_data),
                home_record=self._get_home_record(team_data),
                away_record=self._get_away_record(team_data),
                division_record=self._get_division_record(team_data),
                pythagorean_wins=self._calculate_pythagorean_wins(win_percentage, point_differential),
                luck_factor=self._calculate_luck_factor(win_percentage, point_differential),
                injury_impact=0.0  # Placeholder
            )
            
        except Exception as e:
            logger.error(f"Error creating team stats from 2024 data: {e}")
            return self._create_default_team_stats(team_data.get('Team', ''), 2024, 18)
    
    def _create_team_stats_from_historical_data(self, team_data: Dict, season: int, week: int) -> TeamStats:
        """Create TeamStats from historical data."""
        try:
            # Similar to 2024 data but for any historical season/week
            record = team_data.get('Rec', '0-0')
            wins, losses = map(int, record.split('-'))
            total_games = wins + losses
            win_percentage = wins / total_games if total_games > 0 else 0.0
            
            points_for = float(team_data.get('Tm', 0))
            points_against = float(team_data.get('Opp', 0))
            point_differential = points_for - points_against
            
            return TeamStats(
                team_abbr=self._get_team_abbr(team_data.get('Team', '')),
                team_name=team_data.get('Team', ''),
                season=season,
                week=week,
                win_percentage=win_percentage,
                point_differential=point_differential,
                offensive_epa=self._calculate_offensive_epa(team_data),
                defensive_epa=self._calculate_defensive_epa(team_data),
                passing_epa=self._calculate_passing_epa(team_data),
                rushing_epa=self._calculate_rushing_epa(team_data),
                red_zone_efficiency=self._calculate_red_zone_efficiency(team_data),
                third_down_conversion=self._calculate_third_down_conversion(team_data),
                pass_defense_epa=self._calculate_pass_defense_epa(team_data),
                rush_defense_epa=self._calculate_rush_defense_epa(team_data),
                turnover_margin=self._calculate_turnover_margin(team_data),
                sack_rate=self._calculate_sack_rate(team_data),
                strength_of_schedule=self._calculate_strength_of_schedule(team_data),
                recent_form=self._calculate_recent_form(team_data),
                home_record=self._get_home_record(team_data),
                away_record=self._get_away_record(team_data),
                division_record=self._get_division_record(team_data),
                pythagorean_wins=self._calculate_pythagorean_wins(win_percentage, point_differential),
                luck_factor=self._calculate_luck_factor(win_percentage, point_differential),
                injury_impact=0.0
            )
            
        except Exception as e:
            logger.error(f"Error creating team stats from historical data: {e}")
            return self._create_default_team_stats(team_data.get('Team', ''), season, week)
    
    def _create_default_team_stats(self, team_name: str, season: int, week: int) -> TeamStats:
        """Create default TeamStats when data is not available."""
        return TeamStats(
            team_abbr=self._get_team_abbr(team_name),
            team_name=team_name,
            season=season,
            week=week,
            win_percentage=0.5,
            point_differential=0.0,
            offensive_epa=0.0,
            defensive_epa=0.0,
            passing_epa=0.0,
            rushing_epa=0.0,
            red_zone_efficiency=0.0,
            third_down_conversion=0.0,
            pass_defense_epa=0.0,
            rush_defense_epa=0.0,
            turnover_margin=0.0,
            sack_rate=0.0,
            strength_of_schedule=0.0,
            recent_form=0.5,
            home_record={},
            away_record={},
            division_record={},
            pythagorean_wins=0.5,
            luck_factor=0.0,
            injury_impact=0.0
        )
    
    def _get_team_abbr(self, team_name: str) -> str:
        """Get team abbreviation from full team name."""
        team_mapping = {
            'Kansas City Chiefs': 'KC',
            'Buffalo Bills': 'BUF',
            'Miami Dolphins': 'MIA',
            'New England Patriots': 'NE',
            'New York Jets': 'NYJ',
            'Baltimore Ravens': 'BAL',
            'Cincinnati Bengals': 'CIN',
            'Cleveland Browns': 'CLE',
            'Pittsburgh Steelers': 'PIT',
            'Houston Texans': 'HOU',
            'Indianapolis Colts': 'IND',
            'Jacksonville Jaguars': 'JAX',
            'Tennessee Titans': 'TEN',
            'Denver Broncos': 'DEN',
            'Las Vegas Raiders': 'LV',
            'Los Angeles Chargers': 'LAC',
            'Dallas Cowboys': 'DAL',
            'New York Giants': 'NYG',
            'Philadelphia Eagles': 'PHI',
            'Washington Commanders': 'WAS',
            'Chicago Bears': 'CHI',
            'Detroit Lions': 'DET',
            'Green Bay Packers': 'GB',
            'Minnesota Vikings': 'MIN',
            'Atlanta Falcons': 'ATL',
            'Carolina Panthers': 'CAR',
            'New Orleans Saints': 'NO',
            'Tampa Bay Buccaneers': 'TB',
            'Arizona Cardinals': 'ARI',
            'Los Angeles Rams': 'LAR',
            'San Francisco 49ers': 'SF',
            'Seattle Seahawks': 'SEA'
        }
        return team_mapping.get(team_name, team_name)
    
    # Helper methods for calculating advanced metrics
    def _calculate_offensive_epa(self, team_data: Dict) -> float:
        """Calculate offensive EPA from team data."""
        # Simplified calculation based on points and yards
        points = float(team_data.get('Tm', 0))
        yards = float(team_data.get('OTotYd', 0))
        return (points - 20) * 0.1 + (yards - 350) * 0.01
    
    def _calculate_defensive_epa(self, team_data: Dict) -> float:
        """Calculate defensive EPA from team data."""
        points_allowed = float(team_data.get('Opp', 0))
        yards_allowed = float(team_data.get('DTotYd', 0))
        return (20 - points_allowed) * 0.1 + (350 - yards_allowed) * 0.01
    
    def _calculate_passing_epa(self, team_data: Dict) -> float:
        """Calculate passing EPA from team data."""
        pass_yards = float(team_data.get('OPassY', 0))
        return (pass_yards - 250) * 0.01
    
    def _calculate_rushing_epa(self, team_data: Dict) -> float:
        """Calculate rushing EPA from team data."""
        rush_yards = float(team_data.get('ORushY', 0))
        return (rush_yards - 100) * 0.01
    
    def _calculate_red_zone_efficiency(self, team_data: Dict) -> float:
        """Calculate red zone efficiency from team data."""
        # Simplified calculation
        points = float(team_data.get('Tm', 0))
        return min(1.0, points / 30.0)
    
    def _calculate_third_down_conversion(self, team_data: Dict) -> float:
        """Calculate third down conversion rate from team data."""
        # Simplified calculation
        first_downs = float(team_data.get('O1stD', 0))
        return min(1.0, first_downs / 20.0)
    
    def _calculate_pass_defense_epa(self, team_data: Dict) -> float:
        """Calculate pass defense EPA from team data."""
        pass_yards_allowed = float(team_data.get('DPassY', 0))
        return (250 - pass_yards_allowed) * 0.01
    
    def _calculate_rush_defense_epa(self, team_data: Dict) -> float:
        """Calculate rush defense EPA from team data."""
        rush_yards_allowed = float(team_data.get('DRushY', 0))
        return (100 - rush_yards_allowed) * 0.01
    
    def _calculate_turnover_margin(self, team_data: Dict) -> float:
        """Calculate turnover margin from team data."""
        turnovers_forced = float(team_data.get('DTO', 0))
        turnovers_lost = float(team_data.get('OTO', 0))
        return turnovers_forced - turnovers_lost
    
    def _calculate_sack_rate(self, team_data: Dict) -> float:
        """Calculate sack rate from team data."""
        # Simplified calculation
        return 0.05  # 5% default
    
    def _calculate_strength_of_schedule(self, team_data: Dict) -> float:
        """Calculate strength of schedule from team data."""
        # Use the SOS field if available
        return float(team_data.get('SOS', 0.0))
    
    def _calculate_recent_form(self, team_data: Dict) -> float:
        """Calculate recent form from team data."""
        # Simplified calculation based on win percentage
        record = team_data.get('Rec', '0-0')
        wins, losses = map(int, record.split('-'))
        total_games = wins + losses
        return wins / total_games if total_games > 0 else 0.5
    
    def _get_home_record(self, team_data: Dict) -> Dict[str, int]:
        """Get home record from team data."""
        return {'wins': 0, 'losses': 0, 'win_percentage': 0}
    
    def _get_away_record(self, team_data: Dict) -> Dict[str, int]:
        """Get away record from team data."""
        return {'wins': 0, 'losses': 0, 'win_percentage': 0}
    
    def _get_division_record(self, team_data: Dict) -> Dict[str, int]:
        """Get division record from team data."""
        return {'wins': 0, 'losses': 0, 'win_percentage': 0}
    
    def _calculate_pythagorean_wins(self, win_percentage: float, point_differential: float) -> float:
        """Calculate Pythagorean wins from win percentage and point differential."""
        # Simplified Pythagorean calculation
        return win_percentage + (point_differential / 100.0)
    
    def _calculate_luck_factor(self, win_percentage: float, point_differential: float) -> float:
        """Calculate luck factor from win percentage and point differential."""
        pythagorean_wins = self._calculate_pythagorean_wins(win_percentage, point_differential)
        return win_percentage - pythagorean_wins


# Global instance
temporal_data_collector = TemporalDataCollector()
