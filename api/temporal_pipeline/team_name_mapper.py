"""
Team Name Mapper for Historical Data
Handles team name changes and abbreviations across different time periods
"""

import logging
from typing import Dict, Optional, List

logger = logging.getLogger(__name__)


class TeamNameMapper:
    """
    Maps team names and abbreviations across different time periods.
    
    Handles team relocations and name changes:
    - Oakland Raiders -> Las Vegas Raiders
    - San Diego Chargers -> Los Angeles Chargers  
    - Washington Redskins -> Washington Commanders
    - St. Louis Rams -> Los Angeles Rams
    """
    
    def __init__(self):
        """Initialize the team name mapper."""
        self.team_mappings = self._create_team_mappings()
        self.abbreviation_mappings = self._create_abbreviation_mappings()
        
        logger.info("TeamNameMapper initialized")
    
    def _create_team_mappings(self) -> Dict[str, str]:
        """Create mappings from historical team names to current names."""
        return {
            # Team relocations and name changes (2004+)
            'Oakland Raiders': 'Las Vegas Raiders',
            'San Diego Chargers': 'Los Angeles Chargers',
            'Washington Redskins': 'Washington Commanders',
            'Washington Football Team': 'Washington Commanders',  # 2020-2022 intermediate name
            'St. Louis Rams': 'Los Angeles Rams',
            
            # Alternative name variations
            'Oakland': 'Las Vegas Raiders',
            'San Diego': 'Los Angeles Chargers',
            'St. Louis': 'Los Angeles Rams',
            'LA': 'Los Angeles Rams',  # Some systems use LA for Los Angeles Rams
            'Washington': 'Washington Commanders',
            'Las Vegas': 'Las Vegas Raiders',
            
            # Common variations and partial matches
            'Washington Redskins': 'Washington Commanders',
            'Washington Football Team': 'Washington Commanders',
            'St. Louis Rams': 'Los Angeles Rams',
            'Los Angeles Chargers': 'Los Angeles Chargers',
            'Los Angeles Rams': 'Los Angeles Rams',
            'Las Vegas Raiders': 'Las Vegas Raiders',
            'Washington Commanders': 'Washington Commanders',
            
            # Additional historical variations
            'Oakland Raiders': 'Las Vegas Raiders',
            'San Diego Chargers': 'Los Angeles Chargers',
            'St. Louis Rams': 'Los Angeles Rams',
        }
    
    def _create_abbreviation_mappings(self) -> Dict[str, str]:
        """Create mappings from historical abbreviations to current abbreviations."""
        return {
            # Team abbreviation changes (2004+)
            'OAK': 'LV',  # Oakland Raiders -> Las Vegas Raiders (2020)
            'SD': 'LAC',  # San Diego Chargers -> Los Angeles Chargers (2017)
            'STL': 'LAR',  # St. Louis Rams -> Los Angeles Rams (2016)
            'LA': 'LAR',   # Los Angeles Rams (some systems use LA instead of LAR)
            'WAS': 'WAS',  # Washington (consistent abbreviation)
            'WSH': 'WAS',  # Some systems use WSH for Washington
            
            # Ensure consistency with current abbreviations
            'LV': 'LV',    # Las Vegas Raiders
            'LAC': 'LAC',  # Los Angeles Chargers
            'LAR': 'LAR',  # Los Angeles Rams
            'WAS': 'WAS',  # Washington Commanders
            
            # Additional historical abbreviations
            'OAK': 'LV',   # Oakland (multiple entries for robustness)
            'SD': 'LAC',   # San Diego (multiple entries for robustness)
            'STL': 'LAR',  # St. Louis (multiple entries for robustness)
            'LA': 'LAR',   # Los Angeles (multiple entries for robustness)
        }
    
    def get_current_team_name(self, historical_name: str) -> str:
        """
        Convert historical team name to current team name.
        
        Args:
            historical_name: Historical team name
            
        Returns:
            Current team name
        """
        # Direct mapping
        if historical_name in self.team_mappings:
            return self.team_mappings[historical_name]
        
        # Partial matching for variations
        historical_lower = historical_name.lower()
        for old_name, new_name in self.team_mappings.items():
            if old_name.lower() in historical_lower or historical_lower in old_name.lower():
                return new_name
        
        # If no mapping found, return original name
        logger.warning(f"No mapping found for team name: {historical_name}")
        return historical_name
    
    def get_current_abbreviation(self, historical_abbr: str) -> str:
        """
        Convert historical team abbreviation to current abbreviation.
        
        Args:
            historical_abbr: Historical team abbreviation
            
        Returns:
            Current team abbreviation
        """
        return self.abbreviation_mappings.get(historical_abbr.upper(), historical_abbr.upper())
    
    def _abbreviation_to_full_name(self, team_input: str) -> str:
        """Convert team abbreviation to full team name."""
        abbreviation_mapping = {
            'KC': 'Kansas City Chiefs',
            'BUF': 'Buffalo Bills',
            'MIA': 'Miami Dolphins',
            'NE': 'New England Patriots',
            'NYJ': 'New York Jets',
            'BAL': 'Baltimore Ravens',
            'CIN': 'Cincinnati Bengals',
            'CLE': 'Cleveland Browns',
            'PIT': 'Pittsburgh Steelers',
            'HOU': 'Houston Texans',
            'IND': 'Indianapolis Colts',
            'JAX': 'Jacksonville Jaguars',
            'TEN': 'Tennessee Titans',
            'DEN': 'Denver Broncos',
            'LV': 'Las Vegas Raiders',
            'LAC': 'Los Angeles Chargers',
            'DAL': 'Dallas Cowboys',
            'NYG': 'New York Giants',
            'PHI': 'Philadelphia Eagles',
            'WAS': 'Washington Commanders',
            'CHI': 'Chicago Bears',
            'DET': 'Detroit Lions',
            'GB': 'Green Bay Packers',
            'MIN': 'Minnesota Vikings',
            'ATL': 'Atlanta Falcons',
            'CAR': 'Carolina Panthers',
            'NO': 'New Orleans Saints',
            'TB': 'Tampa Bay Buccaneers',
            'ARI': 'Arizona Cardinals',
            'LAR': 'Los Angeles Rams',
            'LA': 'Los Angeles Rams',  # Some systems use LA instead of LAR
            'SF': 'San Francisco 49ers',
            'SEA': 'Seattle Seahawks'
        }
        
        # If it's already a full name, return as is
        if len(team_input) > 3:
            return team_input
        
        # Convert abbreviation to full name
        return abbreviation_mapping.get(team_input.upper(), team_input)
    
    def get_historical_team_name(self, current_name: str) -> List[str]:
        """
        Get all possible historical names for a current team.
        
        Args:
            current_name: Current team name or abbreviation
            
        Returns:
            List of historical names that map to this current name
        """
        historical_names = []
        
        # First, convert abbreviation to full name if needed
        full_name = self._abbreviation_to_full_name(current_name)
        
        for old_name, new_name in self.team_mappings.items():
            if new_name == full_name:
                historical_names.append(old_name)
        
        # Add the full name itself
        historical_names.append(full_name)
        
        return historical_names
    
    def get_team_abbreviation_from_name(self, team_name: str) -> str:
        """
        Get team abbreviation from team name.
        
        Args:
            team_name: Full team name
            
        Returns:
            Team abbreviation
        """
        # Use the existing team mapping that's already working
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
        
        # Normalize the name first
        normalized_name = self.normalize_team_name(team_name)
        
        # Return abbreviation
        return team_mapping.get(normalized_name, 'UNK')
    
    def normalize_team_name(self, team_name: str) -> str:
        """
        Normalize team name to a standard format.
        
        Args:
            team_name: Any team name variation
            
        Returns:
            Normalized team name
        """
        # Convert to current name first
        current_name = self.get_current_team_name(team_name)
        
        # Standardize common variations
        name_lower = current_name.lower()
        
        if 'las vegas' in name_lower and 'raiders' not in name_lower:
            return 'Las Vegas Raiders'
        elif 'los angeles' in name_lower and 'chargers' in name_lower:
            return 'Los Angeles Chargers'
        elif 'los angeles' in name_lower and 'rams' in name_lower:
            return 'Los Angeles Rams'
        elif 'washington' in name_lower and 'commanders' not in name_lower:
            return 'Washington Commanders'
        
        return current_name
    
    def is_team_name_change(self, team1: str, team2: str) -> bool:
        """
        Check if two team names represent the same team (accounting for name changes).
        
        Args:
            team1: First team name
            team2: Second team name
            
        Returns:
            True if they represent the same team
        """
        normalized1 = self.normalize_team_name(team1)
        normalized2 = self.normalize_team_name(team2)
        return normalized1 == normalized2


# Global instance
team_name_mapper = TeamNameMapper()
