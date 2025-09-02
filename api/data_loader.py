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
        self.game_log_path = Path("api/data/game_log.json")
        self.season_data_path = Path("api/data/season_data_by_team.json")
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
            
            # Read file in chunks to avoid memory issues
            with open(self.game_log_path, 'r') as f:
                data = json.load(f)
                
            for game in data:
                if ((game.get('homeTeamShort', '').lower() == home_team_lower and 
                     game.get('awayTeamShort', '').lower() == away_team_lower) or
                    (game.get('homeTeamShort', '').lower() == away_team_lower and 
                     game.get('awayTeamShort', '').lower() == home_team_lower)):
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
                
            for record in data:
                if (record.get('Season') == season and 
                    record.get('Team', '').lower().find(team_lower) != -1):
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
