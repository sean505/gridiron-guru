"""
Rest and Travel Factors Feature Extractor

This module extracts rest and travel factor features for NFL teams,
including days of rest differentials, travel distances, and home field advantage.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from pathlib import Path
import json
import math

# Import nfl_data_py
try:
    import nfl_data_py as nfl
    NFL_DATA_AVAILABLE = True
except ImportError:
    NFL_DATA_AVAILABLE = False
    logging.warning("nfl_data_py not available. Rest and travel features will use fallback data.")

logger = logging.getLogger(__name__)


class RestTravelExtractor:
    """
    Extracts rest and travel factor features for NFL teams.
    
    Features:
    - Days of rest differential between teams
    - Travel distance for away team
    - Home field advantage strength (by stadium/team)
    - Rest advantage indicators
    - Travel fatigue factors
    - Stadium-specific advantages
    """
    
    def __init__(self, cache_dir: str = "api/data/cache"):
        """Initialize the rest and travel extractor."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache settings
        self.cache_duration = timedelta(hours=6)  # Cache for 6 hours
        self.cache_file = self.cache_dir / "rest_travel_cache.json"
        
        # Load existing cache
        self.cache = self._load_cache()
        
        # Stadium coordinates for distance calculations
        self.stadium_coordinates = self._load_stadium_coordinates()
        
        logger.info("RestTravelExtractor initialized")
    
    def _load_cache(self) -> Dict[str, Any]:
        """Load rest and travel cache from file."""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.warning(f"Error loading rest travel cache: {e}")
            return {}
    
    def _save_cache(self):
        """Save rest and travel cache to file."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, default=str)
        except Exception as e:
            logger.warning(f"Error saving rest travel cache: {e}")
    
    def _load_stadium_coordinates(self) -> Dict[str, Tuple[float, float]]:
        """Load stadium coordinates for distance calculations."""
        return {
            # AFC East
            'BUF': (42.7738, -78.7869),  # Highmark Stadium
            'MIA': (25.9581, -80.2389),  # Hard Rock Stadium
            'NE': (42.0909, -71.2643),   # Gillette Stadium
            'NYJ': (40.8136, -74.0744),  # MetLife Stadium
            
            # AFC North
            'BAL': (39.2780, -76.6227),  # M&T Bank Stadium
            'CIN': (39.0950, -84.5160),  # Paycor Stadium
            'CLE': (41.5061, -81.6996),  # FirstEnergy Stadium
            'PIT': (40.4468, -80.0157),  # Acrisure Stadium
            
            # AFC South
            'HOU': (29.6847, -95.4107),  # NRG Stadium
            'IND': (39.7601, -86.1639),  # Lucas Oil Stadium
            'JAX': (30.3239, -81.6371),  # TIAA Bank Field
            'TEN': (36.1660, -86.7713),  # Nissan Stadium
            
            # AFC West
            'DEN': (39.7439, -105.0200), # Empower Field at Mile High
            'KC': (39.0489, -94.4839),   # Arrowhead Stadium
            'LV': (36.0908, -115.1836),  # Allegiant Stadium
            'LAC': (33.9533, -118.3387), # SoFi Stadium
            
            # NFC East
            'DAL': (32.7473, -97.0945),  # AT&T Stadium
            'NYG': (40.8136, -74.0744),  # MetLife Stadium
            'PHI': (39.9008, -75.1674),  # Lincoln Financial Field
            'WAS': (38.9076, -76.8645),  # FedExField
            
            # NFC North
            'CHI': (41.8625, -87.6167),  # Soldier Field
            'DET': (42.3400, -83.0456),  # Ford Field
            'GB': (44.5013, -88.0622),   # Lambeau Field
            'MIN': (44.9740, -93.2581),  # U.S. Bank Stadium
            
            # NFC South
            'ATL': (33.7550, -84.4010),  # Mercedes-Benz Stadium
            'CAR': (35.2258, -80.8528),  # Bank of America Stadium
            'NO': (29.9508, -90.0811),   # Caesars Superdome
            'TB': (27.9759, -82.5033),   # Raymond James Stadium
            
            # NFC West
            'ARI': (33.5275, -112.2625), # State Farm Stadium
            'LAR': (33.9533, -118.3387), # SoFi Stadium
            'SF': (37.7133, -122.3860),  # Levi's Stadium
            'SEA': (47.5952, -122.3316)  # Lumen Field
        }
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid."""
        if cache_key not in self.cache:
            return False
        
        cached_time = datetime.fromisoformat(self.cache[cache_key]['timestamp'])
        return datetime.now() - cached_time < self.cache_duration
    
    def extract_rest_travel_factors(self, home_team: str, away_team: str, season: int, week: int) -> Dict[str, float]:
        """
        Extract rest and travel factor features for a game.
        
        Args:
            home_team: Home team abbreviation (e.g., 'KC')
            away_team: Away team abbreviation (e.g., 'BUF')
            season: NFL season year
            week: Week number
            
        Returns:
            Dictionary of rest and travel factor features
        """
        try:
            cache_key = f"{home_team}_{away_team}_{season}_{week}"
            
            # Check cache first
            if self._is_cache_valid(cache_key):
                logger.info(f"Using cached rest travel factors for {away_team} @ {home_team}")
                return self.cache[cache_key]['features']
            
            # Extract fresh data
            features = self._calculate_rest_travel_factors(home_team, away_team, season, week)
            
            # Cache the results
            self.cache[cache_key] = {
                'features': features,
                'timestamp': datetime.now().isoformat()
            }
            self._save_cache()
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting rest travel factors for {away_team} @ {home_team}: {e}")
            return self._get_fallback_features()
    
    def _calculate_rest_travel_factors(self, home_team: str, away_team: str, season: int, week: int) -> Dict[str, float]:
        """Calculate rest and travel factor features."""
        try:
            if not NFL_DATA_AVAILABLE:
                logger.warning("nfl_data_py not available, using fallback rest travel features")
                return self._get_fallback_features()
            
            # Get schedule data
            schedule_data = nfl.import_schedules([season])
            
            if schedule_data.empty:
                logger.warning("No schedule data available, using fallback")
                return self._get_fallback_features()
            
            # Calculate rest days
            home_rest_days = self._calculate_rest_days(home_team, schedule_data, season, week)
            away_rest_days = self._calculate_rest_days(away_team, schedule_data, season, week)
            rest_differential = home_rest_days - away_rest_days
            
            # Calculate travel distance
            travel_distance = self._calculate_travel_distance(away_team, home_team)
            
            # Calculate home field advantage
            home_field_advantage = self._calculate_home_field_advantage(home_team)
            
            # Calculate rest advantage indicators
            rest_advantage = self._calculate_rest_advantage(rest_differential)
            travel_fatigue = self._calculate_travel_fatigue(travel_distance)
            
            # Calculate stadium-specific factors
            stadium_factor = self._calculate_stadium_factor(home_team)
            weather_factor = self._calculate_weather_factor(home_team, season, week)
            
            features = {
                'rest_differential': rest_differential,
                'travel_distance': travel_distance,
                'home_field_advantage': home_field_advantage,
                'rest_advantage': rest_advantage,
                'travel_fatigue': travel_fatigue,
                'stadium_factor': stadium_factor,
                'weather_factor': weather_factor,
                'home_rest_days': home_rest_days,
                'away_rest_days': away_rest_days
            }
            
            logger.info(f"Calculated rest travel factors for {away_team} @ {home_team}: {len(features)} features")
            return features
            
        except Exception as e:
            logger.error(f"Error calculating rest travel factors for {away_team} @ {home_team}: {e}")
            return self._get_fallback_features()
    
    def _calculate_rest_days(self, team: str, schedule_data: pd.DataFrame, season: int, week: int) -> float:
        """Calculate days of rest for a team before the current week."""
        try:
            # Get team's games
            team_games = schedule_data[
                ((schedule_data['home_team'] == team) | (schedule_data['away_team'] == team)) &
                (schedule_data['season'] == season) &
                (schedule_data['week'] < week)
            ].copy()
            
            if team_games.empty:
                return 7.0  # Default to 7 days if no previous games
            
            # Get the most recent game
            latest_game = team_games.iloc[-1]
            game_date = pd.to_datetime(latest_game['gameday'])
            
            # Calculate days since last game
            current_date = datetime.now()
            rest_days = (current_date - game_date).days
            
            # Cap at reasonable values
            return max(3.0, min(14.0, rest_days))
            
        except Exception as e:
            logger.error(f"Error calculating rest days for {team}: {e}")
            return 7.0
    
    def _calculate_travel_distance(self, away_team: str, home_team: str) -> float:
        """Calculate travel distance for the away team."""
        try:
            if away_team not in self.stadium_coordinates or home_team not in self.stadium_coordinates:
                return 0.0  # No travel if coordinates not available
            
            away_coords = self.stadium_coordinates[away_team]
            home_coords = self.stadium_coordinates[home_team]
            
            # Calculate distance using Haversine formula
            distance = self._haversine_distance(away_coords, home_coords)
            
            return distance
            
        except Exception as e:
            logger.error(f"Error calculating travel distance for {away_team} to {home_team}: {e}")
            return 0.0
    
    def _haversine_distance(self, coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
        """Calculate distance between two coordinates using Haversine formula."""
        try:
            lat1, lon1 = coord1
            lat2, lon2 = coord2
            
            # Convert to radians
            lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
            
            # Haversine formula
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
            c = 2 * math.asin(math.sqrt(a))
            
            # Radius of earth in miles
            r = 3959
            
            return c * r
            
        except Exception as e:
            logger.error(f"Error calculating Haversine distance: {e}")
            return 0.0
    
    def _calculate_home_field_advantage(self, home_team: str) -> float:
        """Calculate home field advantage strength for a team."""
        try:
            # Historical home field advantage by team (simplified)
            home_advantage = {
                'KC': 0.08,   # Arrowhead Stadium
                'BUF': 0.06,  # Highmark Stadium
                'MIA': 0.05,  # Hard Rock Stadium
                'NE': 0.07,   # Gillette Stadium
                'NYJ': 0.04,  # MetLife Stadium
                'BAL': 0.06,  # M&T Bank Stadium
                'CIN': 0.05,  # Paycor Stadium
                'CLE': 0.05,  # FirstEnergy Stadium
                'PIT': 0.07,  # Acrisure Stadium
                'HOU': 0.05,  # NRG Stadium
                'IND': 0.04,  # Lucas Oil Stadium
                'JAX': 0.05,  # TIAA Bank Field
                'TEN': 0.05,  # Nissan Stadium
                'DEN': 0.08,  # Mile High
                'LV': 0.03,   # Allegiant Stadium
                'LAC': 0.04,  # SoFi Stadium
                'DAL': 0.06,  # AT&T Stadium
                'NYG': 0.04,  # MetLife Stadium
                'PHI': 0.07,  # Lincoln Financial Field
                'WAS': 0.05,  # FedExField
                'CHI': 0.06,  # Soldier Field
                'DET': 0.04,  # Ford Field
                'GB': 0.08,   # Lambeau Field
                'MIN': 0.05,  # U.S. Bank Stadium
                'ATL': 0.05,  # Mercedes-Benz Stadium
                'CAR': 0.05,  # Bank of America Stadium
                'NO': 0.07,   # Caesars Superdome
                'TB': 0.05,   # Raymond James Stadium
                'ARI': 0.05,  # State Farm Stadium
                'LAR': 0.04,  # SoFi Stadium
                'SF': 0.06,   # Levi's Stadium
                'SEA': 0.08    # Lumen Field
            }
            
            return home_advantage.get(home_team, 0.05)  # Default 5% advantage
            
        except Exception as e:
            logger.error(f"Error calculating home field advantage for {home_team}: {e}")
            return 0.05
    
    def _calculate_rest_advantage(self, rest_differential: float) -> float:
        """Calculate rest advantage based on differential."""
        try:
            # Normalize rest differential to advantage score
            if rest_differential > 0:
                # Home team has more rest
                return min(1.0, rest_differential / 7.0)  # Cap at 1.0
            else:
                # Away team has more rest
                return max(-1.0, rest_differential / 7.0)  # Cap at -1.0
            
        except Exception as e:
            logger.error(f"Error calculating rest advantage: {e}")
            return 0.0
    
    def _calculate_travel_fatigue(self, travel_distance: float) -> float:
        """Calculate travel fatigue factor based on distance."""
        try:
            # Normalize travel distance to fatigue score
            if travel_distance == 0:
                return 0.0  # No travel
            
            # Fatigue increases with distance, but caps at reasonable level
            fatigue = min(1.0, travel_distance / 2000.0)  # Cap at 2000 miles
            
            return fatigue
            
        except Exception as e:
            logger.error(f"Error calculating travel fatigue: {e}")
            return 0.0
    
    def _calculate_stadium_factor(self, home_team: str) -> float:
        """Calculate stadium-specific factors."""
        try:
            # Stadium-specific factors (noise, weather, etc.)
            stadium_factors = {
                'KC': 0.1,    # Arrowhead Stadium - loud
                'BUF': 0.08,  # Highmark Stadium - cold weather
                'MIA': 0.05,  # Hard Rock Stadium - heat
                'NE': 0.08,   # Gillette Stadium - cold weather
                'NYJ': 0.04,  # MetLife Stadium - neutral
                'BAL': 0.06,  # M&T Bank Stadium - loud
                'CIN': 0.05,  # Paycor Stadium - neutral
                'CLE': 0.05,  # FirstEnergy Stadium - cold weather
                'PIT': 0.08,  # Acrisure Stadium - loud
                'HOU': 0.05,  # NRG Stadium - dome
                'IND': 0.04,  # Lucas Oil Stadium - dome
                'JAX': 0.05,  # TIAA Bank Field - heat
                'TEN': 0.05,  # Nissan Stadium - neutral
                'DEN': 0.1,   # Mile High - altitude
                'LV': 0.03,   # Allegiant Stadium - dome
                'LAC': 0.04,  # SoFi Stadium - neutral
                'DAL': 0.06,  # AT&T Stadium - dome
                'NYG': 0.04,  # MetLife Stadium - neutral
                'PHI': 0.08,  # Lincoln Financial Field - loud
                'WAS': 0.05,  # FedExField - neutral
                'CHI': 0.07,  # Soldier Field - cold weather
                'DET': 0.04,  # Ford Field - dome
                'GB': 0.1,    # Lambeau Field - cold weather
                'MIN': 0.05,  # U.S. Bank Stadium - dome
                'ATL': 0.05,  # Mercedes-Benz Stadium - dome
                'CAR': 0.05,  # Bank of America Stadium - neutral
                'NO': 0.07,   # Caesars Superdome - loud
                'TB': 0.05,   # Raymond James Stadium - heat
                'ARI': 0.05,  # State Farm Stadium - dome
                'LAR': 0.04,  # SoFi Stadium - neutral
                'SF': 0.06,   # Levi's Stadium - neutral
                'SEA': 0.08    # Lumen Field - loud
            }
            
            return stadium_factors.get(home_team, 0.05)  # Default factor
            
        except Exception as e:
            logger.error(f"Error calculating stadium factor for {home_team}: {e}")
            return 0.05
    
    def _calculate_weather_factor(self, home_team: str, season: int, week: int) -> float:
        """Calculate weather factor based on team and time of year."""
        try:
            # Weather factors by team and time of year
            if week <= 4 or week >= 15:  # Early season or late season
                weather_teams = {
                    'BUF': 0.1,   # Cold weather advantage
                    'NE': 0.1,    # Cold weather advantage
                    'GB': 0.1,    # Cold weather advantage
                    'CHI': 0.08,  # Cold weather advantage
                    'CLE': 0.08,  # Cold weather advantage
                    'PIT': 0.08,  # Cold weather advantage
                    'DEN': 0.06,  # Altitude advantage
                    'MIA': 0.05,  # Heat advantage
                    'TB': 0.05,   # Heat advantage
                    'JAX': 0.05,  # Heat advantage
                    'HOU': 0.05,  # Heat advantage
                }
                
                return weather_teams.get(home_team, 0.0)
            else:
                return 0.0  # No weather advantage in mid-season
                
        except Exception as e:
            logger.error(f"Error calculating weather factor for {home_team}: {e}")
            return 0.0
    
    def _get_fallback_features(self) -> Dict[str, float]:
        """Get fallback rest and travel features when data is unavailable."""
        return {
            'rest_differential': 0.0,
            'travel_distance': 0.0,
            'home_field_advantage': 0.05,
            'rest_advantage': 0.0,
            'travel_fatigue': 0.0,
            'stadium_factor': 0.05,
            'weather_factor': 0.0,
            'home_rest_days': 7.0,
            'away_rest_days': 7.0
        }
    
    def get_team_rest_travel_comparison(self, home_team: str, away_team: str, season: int, week: int) -> Dict[str, float]:
        """Get rest and travel comparison features between two teams."""
        try:
            features = self.extract_rest_travel_factors(home_team, away_team, season, week)
            
            # Add comparison-specific features
            comparison_features = features.copy()
            
            # Rest advantage indicators
            comparison_features['rest_advantage_home'] = 1.0 if features['rest_differential'] > 0 else 0.0
            comparison_features['rest_advantage_away'] = 1.0 if features['rest_differential'] < 0 else 0.0
            
            # Travel fatigue indicators
            comparison_features['significant_travel'] = 1.0 if features['travel_distance'] > 1000 else 0.0
            comparison_features['minimal_travel'] = 1.0 if features['travel_distance'] < 500 else 0.0
            
            return comparison_features
            
        except Exception as e:
            logger.error(f"Error getting rest travel comparison for {home_team} vs {away_team}: {e}")
            return {}


# Global rest and travel extractor instance
rest_travel_extractor = RestTravelExtractor()
