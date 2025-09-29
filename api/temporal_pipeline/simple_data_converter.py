"""
Simple Data Converter
Converts the actual data structure to what the ML training pipeline expects
"""

import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)


class SimpleDataConverter:
    """
    Converts actual data structure to ML training format.
    
    The actual data structure:
    - game_log: List of games with rich team stats per game
    - Each game has homeTeam, awayTeam, season, week, and team performance data
    
    Converts to:
    - Team stats by season/week for training
    - Game records with proper team data
    """
    
    def __init__(self, data_dir: str = "data"):
        """Initialize the data converter."""
        self.data_dir = Path(data_dir)
        self.game_log_file = self.data_dir / "game_log.json"
        self.season_data_file = self.data_dir / "season_data_by_team.json"
        
        # Cache for loaded data
        self._game_log_cache = None
        self._season_data_cache = None
        
        logger.info(f"SimpleDataConverter initialized with data directory: {self.data_dir}")
    
    def load_training_data(self, years: List[int]) -> Dict[str, Any]:
        """
        Load and convert training data for specified years.
        
        Args:
            years: List of years to load (e.g., [2008, 2009, ..., 2024])
            
        Returns:
            Dictionary containing converted training data
        """
        try:
            logger.info(f"Loading training data for years: {years}")
            
            # Load raw data
            game_log = self._load_game_log_data()
            
            # Filter games for requested years
            training_games = [game for game in game_log if game.get('season') in years]
            
            # Convert to team stats format
            team_stats = self._extract_team_stats(training_games, years)
            
            # Convert games to training format
            converted_games = self._convert_games_to_training_format(training_games)
            
            logger.info(f"Loaded {len(training_games)} games and {len(team_stats)} team records")
            
            return {
                'games': converted_games,
                'teams': team_stats,
                'years': years,
                'total_games': len(converted_games),
                'total_teams': len(team_stats)
            }
            
        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            return {'games': [], 'teams': {}, 'years': years, 'total_games': 0, 'total_teams': 0}
    
    def _load_game_log_data(self) -> List[Dict]:
        """Load game log data from JSON file."""
        if self._game_log_cache is None:
            try:
                with open(self.game_log_file, 'r') as f:
                    self._game_log_cache = json.load(f)
                logger.info(f"Loaded game log: {len(self._game_log_cache)} games")
            except Exception as e:
                logger.error(f"Error loading game log: {e}")
                self._game_log_cache = []
        return self._game_log_cache
    
    def _extract_team_stats(self, games: List[Dict], years: List[int]) -> Dict[str, Dict]:
        """
        Extract team statistics from game data.
        
        Args:
            games: List of game records
            years: Years to process
            
        Returns:
            Dictionary of team stats by team_season_week
        """
        team_stats = {}
        
        # Group games by team and season
        team_games = defaultdict(lambda: defaultdict(list))
        
        for game in games:
            season = game.get('season', 0)
            week = game.get('week', 0)
            home_team = game.get('homeTeam', '')
            away_team = game.get('awayTeam', '')
            
            if not home_team or not away_team or season not in years:
                continue
            
            # Process home team
            team_games[home_team][season].append({
                'week': week,
                'is_home': True,
                'game': game
            })
            
            # Process away team
            team_games[away_team][season].append({
                'week': week,
                'is_home': False,
                'game': game
            })
        
        # Calculate cumulative stats for each team/season/week
        for team, seasons in team_games.items():
            for season, team_games_list in seasons.items():
                # Sort by week
                team_games_list.sort(key=lambda x: x['week'])
                
                # Calculate cumulative stats
                wins = 0
                losses = 0
                points_for = 0
                points_against = 0
                total_games = 0
                
                for game_data in team_games_list:
                    week = game_data['week']
                    game = game_data['game']
                    is_home = game_data['is_home']
                    
                    # Get team's score
                    if is_home:
                        team_score = game.get('HomeScore', 0)
                        opp_score = game.get('AwayScore', 0)
                    else:
                        team_score = game.get('AwayScore', 0)
                        opp_score = game.get('HomeScore', 0)
                    
                    # Update cumulative stats
                    points_for += team_score
                    points_against += opp_score
                    total_games += 1
                    
                    if team_score > opp_score:
                        wins += 1
                    else:
                        losses += 1
                    
                    # Calculate stats up to this week
                    win_pct = wins / total_games if total_games > 0 else 0.5
                    point_diff = points_for - points_against
                    avg_points_for = points_for / total_games if total_games > 0 else 0
                    avg_points_against = points_against / total_games if total_games > 0 else 0
                    
                    # Create team stats record
                    team_key = f"{season}_{week}_{team}"
                    team_stats[team_key] = {
                        'team': team,
                        'season': season,
                        'week': week,
                        'wins': wins,
                        'losses': losses,
                        'win_percentage': win_pct,
                        'points_for': points_for,
                        'points_against': points_against,
                        'point_differential': point_diff,
                        'avg_points_for': avg_points_for,
                        'avg_points_against': avg_points_against,
                        'total_games': total_games,
                        # Add more advanced stats from game data
                        'offensive_epa': self._calculate_offensive_epa(game, is_home),
                        'defensive_epa': self._calculate_defensive_epa(game, is_home),
                        'passing_epa': self._calculate_passing_epa(game, is_home),
                        'rushing_epa': self._calculate_rushing_epa(game, is_home),
                        'turnover_margin': self._calculate_turnover_margin(game, is_home),
                        'red_zone_efficiency': self._calculate_red_zone_efficiency(game, is_home),
                        'third_down_conversion': self._calculate_third_down_conversion(game, is_home),
                        'sack_rate': self._calculate_sack_rate(game, is_home),
                        'strength_of_schedule': self._calculate_sos(game, is_home),
                        'recent_form': self._calculate_recent_form(team_games_list, len(team_games_list) - 1),
                        'pythagorean_wins': self._calculate_pythagorean_wins(avg_points_for, avg_points_against, total_games),
                        'luck_factor': self._calculate_luck_factor(wins, total_games, avg_points_for, avg_points_against),
                        'injury_impact': 0.0,  # Placeholder
                        'home_record': {'win_percentage': win_pct},  # Simplified
                        'away_record': {'win_percentage': win_pct}   # Simplified
                    }
        
        return team_stats
    
    def _convert_games_to_training_format(self, games: List[Dict]) -> List[Dict]:
        """
        Convert games to training format.
        
        Args:
            games: List of game records
            
        Returns:
            List of converted game records
        """
        converted_games = []
        
        for game in games:
            converted_game = {
                'game_id': f"{game.get('season', 0)}_{game.get('week', 0)}_{game.get('homeTeam', '')}_{game.get('awayTeam', '')}",
                'homeTeam': game.get('homeTeam', ''),
                'awayTeam': game.get('awayTeam', ''),
                'season': game.get('season', 0),
                'week': game.get('week', 0),
                'Winner': game.get('Winner', 0),
                'HomeScore': game.get('HomeScore', 0),
                'AwayScore': game.get('AwayScore', 0),
                'Date': game.get('Date', ''),
                'VegasLine': game.get('VegasLine', ''),
                'actualSpread': game.get('actualSpread', 0),
                'homeRecordAgainstOpp': game.get('homeRecordAgainstOpp', 0.5),
                'awayRecordAgainstOpp': game.get('awayRecordAgainstOpp', 0.5)
            }
            converted_games.append(converted_game)
        
        return converted_games
    
    def _calculate_offensive_epa(self, game: Dict, is_home: bool) -> float:
        """Calculate offensive EPA from game data."""
        if is_home:
            points = game.get('homeavgScore', 0)
            yards = game.get('homeavgOffensiveYards', 0)
        else:
            points = game.get('awayavgScore', 0)
            yards = game.get('awayavgOffensiveYards', 0)
        
        return (points - 20) * 0.1 + (yards - 350) * 0.01
    
    def _calculate_defensive_epa(self, game: Dict, is_home: bool) -> float:
        """Calculate defensive EPA from game data."""
        if is_home:
            points_allowed = game.get('homeavgOppScore', 0)
            yards_allowed = game.get('homeavgYardsAllowed', 0)
        else:
            points_allowed = game.get('awayavgOppScore', 0)
            yards_allowed = game.get('awayavgYardsAllowed', 0)
        
        return (points_allowed - 20) * 0.1 + (yards_allowed - 350) * 0.01
    
    def _calculate_passing_epa(self, game: Dict, is_home: bool) -> float:
        """Calculate passing EPA from game data."""
        if is_home:
            pass_yards = game.get('homeavgPassingYards', 0)
        else:
            pass_yards = game.get('awayavgPassingYards', 0)
        
        return (pass_yards - 250) * 0.01
    
    def _calculate_rushing_epa(self, game: Dict, is_home: bool) -> float:
        """Calculate rushing EPA from game data."""
        if is_home:
            rush_yards = game.get('homeavgRushingYards', 0)
        else:
            rush_yards = game.get('awayavgRushingYards', 0)
        
        return (rush_yards - 120) * 0.01
    
    def _calculate_turnover_margin(self, game: Dict, is_home: bool) -> float:
        """Calculate turnover margin from game data."""
        if is_home:
            turnovers_lost = game.get('homeavgTurnoversLost', 0)
            turnovers_forced = game.get('homeavgTurnoversForced', 0)
        else:
            turnovers_lost = game.get('awayavgTurnoversLost', 0)
            turnovers_forced = game.get('awayavgTurnoversForced', 0)
        
        return turnovers_forced - turnovers_lost
    
    def _calculate_red_zone_efficiency(self, game: Dict, is_home: bool) -> float:
        """Calculate red zone efficiency (placeholder)."""
        return 0.5  # Placeholder
    
    def _calculate_third_down_conversion(self, game: Dict, is_home: bool) -> float:
        """Calculate third down conversion rate (placeholder)."""
        return 0.4  # Placeholder
    
    def _calculate_sack_rate(self, game: Dict, is_home: bool) -> float:
        """Calculate sack rate (placeholder)."""
        return 0.05  # Placeholder
    
    def _calculate_sos(self, game: Dict, is_home: bool) -> float:
        """Calculate strength of schedule."""
        if is_home:
            return game.get('homeSOS', 0.0)
        else:
            return game.get('awaySOS', 0.0)
    
    def _calculate_recent_form(self, team_games: List[Dict], current_index: int) -> float:
        """Calculate recent form (last 4 games)."""
        if current_index < 3:
            return 0.5  # Not enough games
        
        recent_games = team_games[max(0, current_index - 3):current_index + 1]
        wins = 0
        for game_data in recent_games:
            game = game_data['game']
            is_home = game_data['is_home']
            
            if is_home:
                team_score = game.get('HomeScore', 0)
                opp_score = game.get('AwayScore', 0)
            else:
                team_score = game.get('AwayScore', 0)
                opp_score = game.get('HomeScore', 0)
            
            if team_score > opp_score:
                wins += 1
        
        return wins / len(recent_games)
    
    def _calculate_pythagorean_wins(self, avg_points_for: float, avg_points_against: float, games: int) -> float:
        """Calculate Pythagorean wins."""
        if avg_points_against == 0:
            return games * 0.5
        
        pythagorean_pct = (avg_points_for ** 2.37) / (avg_points_for ** 2.37 + avg_points_against ** 2.37)
        return pythagorean_pct * games
    
    def _calculate_luck_factor(self, wins: int, games: int, avg_points_for: float, avg_points_against: float) -> float:
        """Calculate luck factor."""
        if games == 0:
            return 0.0
        
        actual_win_pct = wins / games
        pythagorean_pct = self._calculate_pythagorean_wins(avg_points_for, avg_points_against, games) / games
        
        return actual_win_pct - pythagorean_pct


# Global instance
simple_data_converter = SimpleDataConverter()



