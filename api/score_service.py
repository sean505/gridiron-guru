"""
Score Service for fetching real NFL game scores from ESPN API
"""

import requests
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json

# Import team name mapper for historical team name handling
try:
    from temporal_pipeline.team_name_mapper import team_name_mapper
    TEAM_MAPPER_AVAILABLE = True
except ImportError:
    TEAM_MAPPER_AVAILABLE = False
    logging.warning("Team name mapper not available")

logger = logging.getLogger(__name__)

class ScoreService:
    """Service for fetching real NFL scores from ESPN API"""
    
    def __init__(self):
        self.base_url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"
        self.cache = {}
        self.cache_duration = 300  # 5 minutes cache
        
    def get_current_week_scores(self, season: int = 2025, week: int = None) -> Dict[str, Dict]:
        """
        Get scores for current week games from ESPN API
        
        Returns:
            Dict mapping game_id to score data
        """
        try:
            # ESPN API endpoint for current week
            url = f"{self.base_url}/scoreboard"
            if week:
                url += f"?week={week}"
                
            logger.info(f"Fetching scores from ESPN API: {url}")
            response = requests.get(url, timeout=10)
            
            if response.status_code != 200:
                logger.error(f"ESPN API failed with status {response.status_code}")
                return {}
                
            data = response.json()
            
            # Extract week information from ESPN API response
            week_info = data.get('week', {})
            api_week = week_info.get('number', week)
            if api_week and not week:
                logger.info(f"ESPN API returned week {api_week}")
                week = api_week
            
            scores = {}
            
            for event in data.get('events', []):
                competition = event.get('competitions', [{}])[0]
                competitors = competition.get('competitors', [])
                
                if len(competitors) >= 2:
                    home_team = competitors[0]
                    away_team = competitors[1]
                    
                    # Get team abbreviations
                    home_abbr = home_team.get('team', {}).get('abbreviation', '')
                    away_abbr = away_team.get('team', {}).get('abbreviation', '')
                    
                    # Normalize team abbreviations using team name mapper
                    if TEAM_MAPPER_AVAILABLE:
                        home_abbr = team_name_mapper.get_current_abbreviation(home_abbr)
                        away_abbr = team_name_mapper.get_current_abbreviation(away_abbr)
                    
                    # Get scores
                    home_score = home_team.get('score', 0)
                    away_score = away_team.get('score', 0)
                    
                    # Get team records (ESPN uses 'records' field)
                    home_records = home_team.get('records', [])
                    away_records = away_team.get('records', [])
                    
                    # Extract wins and losses from records array
                    home_wins = 0
                    home_losses = 0
                    away_wins = 0
                    away_losses = 0
                    
                    # Look for overall record in the records array
                    for record in home_records:
                        if record.get('type') == 'total':
                            summary = record.get('summary', '0-0')
                            try:
                                home_wins, home_losses = map(int, summary.split('-'))
                            except (ValueError, AttributeError):
                                home_wins, home_losses = 0, 0
                            break
                    
                    for record in away_records:
                        if record.get('type') == 'total':
                            summary = record.get('summary', '0-0')
                            try:
                                away_wins, away_losses = map(int, summary.split('-'))
                            except (ValueError, AttributeError):
                                away_wins, away_losses = 0, 0
                            break
                    
                    # Get game status
                    status = competition.get('status', {})
                    game_status = status.get('type', {}).get('name', 'scheduled')
                    
                    # Get game date and time from ESPN API
                    game_date_raw = event.get('date', '')
                    game_time_raw = event.get('date', '')
                    
                    # Parse date and time
                    game_date = 'TBD'
                    game_time = 'TBD'
                    
                    if game_date_raw:
                        try:
                            from datetime import datetime
                            import pytz
                            import tzlocal
                            # Parse ISO format date from ESPN (UTC)
                            dt_utc = datetime.fromisoformat(game_date_raw.replace('Z', '+00:00'))
                            
                            # Get user's local timezone
                            try:
                                local_tz = tzlocal.get_localzone()
                                dt_local = dt_utc.astimezone(local_tz)
                            except Exception:
                                # Fallback to system timezone detection
                                import time
                                local_offset = time.timezone if (time.daylight == 0) else time.altzone
                                local_tz = pytz.timezone(time.tzname[0]) if hasattr(time, 'tzname') else pytz.timezone('US/Eastern')
                                dt_local = dt_utc.astimezone(local_tz)
                            
                            # Format date and time in user's timezone
                            game_date = dt_local.strftime('%Y-%m-%d')
                            game_time = dt_local.strftime('%I:%M %p')
                            
                        except Exception as e:
                            logger.warning(f"Error parsing date {game_date_raw}: {e}")
                            # Fallback to UTC if timezone conversion fails
                            try:
                                dt = datetime.fromisoformat(game_date_raw.replace('Z', '+00:00'))
                                game_date = dt.strftime('%Y-%m-%d')
                                game_time = dt.strftime('%I:%M %p')
                            except:
                                game_date = 'TBD'
                                game_time = 'TBD'
                    
                    # Create game ID to match temporal pipeline format
                    game_id = f"{season}_{week or 1}_{home_abbr}_{away_abbr}"
                    
                    # Determine if game is completed
                    is_completed = game_status in ['STATUS_FINAL', 'STATUS_POSTGAME']
                    
                    scores[game_id] = {
                        'game_id': game_id,
                        'home_team': home_abbr,
                        'away_team': away_abbr,
                        'home_score': home_score,
                        'away_score': away_score,
                        'actual_score': f"{away_score}-{home_score}" if is_completed else None,
                        'game_status': 'completed' if is_completed else 'scheduled',
                        'game_date': game_date,
                        'game_time': game_time,
                        'week': week or 1,
                        'season': season,
                        'is_completed': is_completed,
                        'home_record': f"{home_wins}-{home_losses}",
                        'away_record': f"{away_wins}-{away_losses}"
                    }
                    
            logger.info(f"Fetched {len(scores)} games from ESPN API")
            return scores
            
        except Exception as e:
            logger.error(f"Error fetching scores from ESPN API: {e}")
            return {}
    
    def get_historical_scores(self, season: int, week: int) -> Dict[str, Dict]:
        """
        Get historical scores for a specific week
        
        For now, this will simulate completed games since ESPN API
        doesn't provide historical week data easily
        """
        try:
            # For historical data, we'll simulate completed games
            # In a real implementation, you'd use a different API or database
            logger.info(f"Simulating historical scores for {season} Week {week}")
            
            # This is a placeholder - in reality you'd fetch from a historical API
            # or database that stores completed game results
            return {}
            
        except Exception as e:
            logger.error(f"Error fetching historical scores: {e}")
            return {}
    
    def get_game_score(self, game_id: str, season: int, week: int) -> Optional[Dict]:
        """
        Get score for a specific game
        """
        try:
            # Try current week first
            current_scores = self.get_current_week_scores(season, week)
            if game_id in current_scores:
                return current_scores[game_id]
                
            # Try historical
            historical_scores = self.get_historical_scores(season, week)
            if game_id in historical_scores:
                return historical_scores[game_id]
                
            return None
            
        except Exception as e:
            logger.error(f"Error fetching score for game {game_id}: {e}")
            return None

# Initialize score service
score_service = ScoreService()
