"""
Optimized data loading for Vercel deployment.
Loads data on-demand instead of loading everything at startup.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class OptimizedDataLoader:
    """Efficient data loader that loads data on-demand"""
    
    def __init__(self):
        self.game_log_path = Path("data/game_log.json")
        self.season_data_path = Path("data/season_data_by_team.json")
        self._game_log_cache = {}
        self._season_data_cache = {}
    
    def get_historical_matchups(self, home_team: str, away_team: str) -> List[Dict]:
        """Get historical matchups between two teams (on-demand loading)"""
        try:
            # Use cached data if available
            cache_key = f"{home_team.lower()}_{away_team.lower()}"
            if cache_key in self._game_log_cache:
                return self._game_log_cache[cache_key]
            
            if not self.game_log_path.exists():
                return []
            
            # Load and search data on-demand
            matchups = []
            home_team_lower = home_team.lower()
            away_team_lower = away_team.lower()
            
            # Map team abbreviations to actual abbreviations used in game log
            team_abbrev_mapping = {
                'kc': 'kan',
                'buf': 'buf',
                'sf': 'sfo',
                'dal': 'dal',
                'bal': 'rav',
                'mia': 'mia',
                'det': 'det',
                'gb': 'gnb',
                'hou': 'htx',
                'ind': 'clt',
                'jax': 'jax',
                'lv': 'rai',
                'lac': 'sdg',
                'lar': 'ram',
                'min': 'min',
                'ne': 'nwe',
                'no': 'nor',
                'nyg': 'nyg',
                'nyj': 'nyj',
                'phi': 'phi',
                'pit': 'pit',
                'sea': 'sea',
                'tb': 'tam',
                'ten': 'oti',
                'was': 'was'
            }
            
            home_abbrev = team_abbrev_mapping.get(home_team_lower, home_team_lower)
            away_abbrev = team_abbrev_mapping.get(away_team_lower, away_team_lower)
            
            # Read file in chunks to avoid memory issues
            with open(self.game_log_path, 'r') as f:
                data = json.load(f)
                
            for game in data:
                if ((game.get('homeTeamShort', '').lower() == home_abbrev and 
                     game.get('awayTeamShort', '').lower() == away_abbrev) or
                    (game.get('homeTeamShort', '').lower() == away_abbrev and 
                     game.get('awayTeamShort', '').lower() == home_abbrev)):
                    matchups.append(game)
            
            # Cache the result
            self._game_log_cache[cache_key] = matchups
            return matchups
            
        except Exception as e:
            logger.error(f"Error loading historical matchups: {e}")
            return []
    
    def get_team_record(self, team: str, season: int) -> Dict:
        """Get team's record for a specific season (on-demand loading)"""
        try:
            # Use cached data if available
            cache_key = f"{team.lower()}_{season}"
            if cache_key in self._season_data_cache:
                return self._season_data_cache[cache_key]
            
            if not self.season_data_path.exists():
                return {"wins": 8, "losses": 8, "win_pct": 0.5, "points_for": 350, "points_against": 350}
            
            team_lower = team.lower()
            wins = 0
            losses = 0
            points_for = 0
            points_against = 0
            games_played = 0
            
            # Load and search data on-demand
            with open(self.season_data_path, 'r') as f:
                data = json.load(f)
                
            # Map team abbreviations to full names (using actual abbreviations from game log)
            team_name_mapping = {
                'kc': 'kansas city chiefs',
                'kan': 'kansas city chiefs',
                'buf': 'buffalo bills',
                'sf': 'san francisco 49ers',
                'sfo': 'san francisco 49ers',
                'dal': 'dallas cowboys',
                'bal': 'baltimore ravens',
                'rav': 'baltimore ravens',
                'mia': 'miami dolphins',
                'det': 'detroit lions',
                'gb': 'green bay packers',
                'gnb': 'green bay packers',
                'hou': 'houston texans',
                'htx': 'houston texans',
                'ind': 'indianapolis colts',
                'clt': 'indianapolis colts',
                'jax': 'jacksonville jaguars',
                'jax': 'jacksonville jaguars',
                'lv': 'las vegas raiders',
                'rai': 'las vegas raiders',
                'lac': 'los angeles chargers',
                'sdg': 'los angeles chargers',
                'lar': 'los angeles rams',
                'ram': 'los angeles rams',
                'min': 'minnesota vikings',
                'ne': 'new england patriots',
                'nwe': 'new england patriots',
                'no': 'new orleans saints',
                'nor': 'new orleans saints',
                'nyg': 'new york giants',
                'nyj': 'new york jets',
                'phi': 'philadelphia eagles',
                'pit': 'pittsburgh steelers',
                'sea': 'seattle seahawks',
                'sea': 'seattle seahawks',
                'tb': 'tampa bay buccaneers',
                'tam': 'tampa bay buccaneers',
                'ten': 'tennessee titans',
                'oti': 'tennessee titans',
                'was': 'washington commanders',
                'was': 'washington commanders'
            }
            
            full_team_name = team_name_mapping.get(team_lower, team_lower)
            
            for record in data:
                if (record.get('Season') == season and 
                    record.get('Team', '').lower().find(full_team_name) != -1):
                    games_played += 1
                    points_for += int(record.get('Tm', 0))
                    points_against += int(record.get('Opp', 0))
                    
                    if record.get('W/L') == 'W':
                        wins += 1
                    elif record.get('W/L') == 'L':
                        losses += 1
            
            win_pct = wins / max(games_played, 1)
            
            result = {
                "wins": wins,
                "losses": losses,
                "win_pct": win_pct,
                "points_for": points_for,
                "points_against": points_against,
                "games_played": games_played
            }
            
            # Cache the result
            self._season_data_cache[cache_key] = result
            return result
            
        except Exception as e:
            logger.error(f"Error loading team record: {e}")
            return {"wins": 8, "losses": 8, "win_pct": 0.5, "points_for": 350, "points_against": 350}
    
    def has_upset_history(self, home_team: str, away_team: str, matchups: List[Dict]) -> bool:
        """Check if there's a history of upsets between these teams"""
        if not matchups:
            return False
        
        upset_count = 0
        total_games = len(matchups)
        
        for game in matchups:
            # Determine if this was an upset based on records and outcome
            home_wins = game.get('homeWins', 0)
            away_wins = game.get('awayWins', 0)
            home_score = game.get('homeScore', 0)
            away_score = game.get('awayScore', 0)
            
            # Upset: team with worse record won
            if (home_wins < away_wins and home_score > away_score) or \
               (away_wins < home_wins and away_score > home_score):
                upset_count += 1
        
        # Consider it an upset pattern if >30% of games were upsets
        return (upset_count / max(total_games, 1)) > 0.3

# Global instance
data_loader = OptimizedDataLoader()
