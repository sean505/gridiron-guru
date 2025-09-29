"""
Prediction Baseline Pipeline
Loads 2024 season-end team strength data for 2025 predictions
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

from prediction_engine.data_models import TeamStats
from temporal_pipeline.temporal_data_collector import temporal_data_collector

logger = logging.getLogger(__name__)


class PredictionBaselinePipeline:
    """
    Prediction baseline pipeline that loads 2024 season-end team strength data.
    
    Features:
    - Loads 2024 Week 18 team records
    - Calculates team strength from final 2024 performance
    - Creates baseline features for 2025 predictions
    - Provides team differentiation data
    """
    
    def __init__(self):
        """Initialize the prediction baseline pipeline."""
        self.data_collector = temporal_data_collector
        self.baseline_cache = None
        
        logger.info("PredictionBaselinePipeline initialized")
    
    def load_2024_baseline(self) -> Dict[str, TeamStats]:
        """
        Load 2024 season-end team strength data.
        
        Returns:
            Dictionary mapping team names to TeamStats objects
        """
        try:
            if self.baseline_cache is not None:
                return self.baseline_cache
            
            logger.info("Loading 2024 season-end baseline data")
            
            baseline = self.data_collector.get_2024_baseline()
            
            logger.info(f"Loaded 2024 baseline for {len(baseline)} teams")
            self.baseline_cache = baseline
            return baseline
            
        except Exception as e:
            logger.error(f"Error loading 2024 baseline: {e}")
            return {}
    
    def get_team_strength(self, team_name: str) -> Optional[TeamStats]:
        """
        Get team strength for a specific team.
        
        Args:
            team_name: Full team name (e.g., "Kansas City Chiefs")
            
        Returns:
            TeamStats object or None if not found
        """
        try:
            baseline = self.load_2024_baseline()
            return baseline.get(team_name)
            
        except Exception as e:
            logger.error(f"Error getting team strength for {team_name}: {e}")
            return None
    
    def get_team_strength_by_abbr(self, team_abbr: str) -> Optional[TeamStats]:
        """
        Get team strength by team abbreviation.
        
        Args:
            team_abbr: Team abbreviation (e.g., "KC")
            
        Returns:
            TeamStats object or None if not found
        """
        try:
            baseline = self.load_2024_baseline()
            
            # Find team by abbreviation
            for team_name, team_stats in baseline.items():
                if team_stats.team_abbr == team_abbr:
                    return team_stats
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting team strength for {team_abbr}: {e}")
            return None
    
    def get_all_teams(self) -> List[str]:
        """
        Get list of all available teams.
        
        Returns:
            List of team names
        """
        try:
            baseline = self.load_2024_baseline()
            return list(baseline.keys())
            
        except Exception as e:
            logger.error(f"Error getting team list: {e}")
            return []
    
    def get_team_rankings(self) -> List[Tuple[str, float]]:
        """
        Get team rankings by win percentage.
        
        Returns:
            List of (team_name, win_percentage) tuples sorted by win percentage
        """
        try:
            baseline = self.load_2024_baseline()
            
            rankings = []
            for team_name, team_stats in baseline.items():
                rankings.append((team_name, team_stats.win_percentage))
            
            # Sort by win percentage (descending)
            rankings.sort(key=lambda x: x[1], reverse=True)
            
            return rankings
            
        except Exception as e:
            logger.error(f"Error getting team rankings: {e}")
            return []
    
    def validate_baseline_data(self) -> Dict[str, Any]:
        """
        Validate the quality of baseline data.
        
        Returns:
            Dictionary containing validation results
        """
        try:
            baseline = self.load_2024_baseline()
            
            if not baseline:
                return {
                    'valid': False,
                    'error': 'No baseline data loaded',
                    'team_count': 0
                }
            
            # Check data quality
            team_count = len(baseline)
            teams_with_data = 0
            teams_with_differentiation = 0
            
            win_percentages = []
            point_differentials = []
            
            for team_name, team_stats in baseline.items():
                if team_stats.win_percentage > 0:
                    teams_with_data += 1
                    win_percentages.append(team_stats.win_percentage)
                    point_differentials.append(team_stats.point_differential)
            
            # Check for team differentiation
            if len(set(win_percentages)) > 1:
                teams_with_differentiation = len(set(win_percentages))
            
            return {
                'valid': True,
                'team_count': team_count,
                'teams_with_data': teams_with_data,
                'teams_with_differentiation': teams_with_differentiation,
                'win_percentage_range': (min(win_percentages), max(win_percentages)) if win_percentages else (0, 0),
                'point_differential_range': (min(point_differentials), max(point_differentials)) if point_differentials else (0, 0)
            }
            
        except Exception as e:
            logger.error(f"Error validating baseline data: {e}")
            return {
                'valid': False,
                'error': str(e),
                'team_count': 0
            }


# Global instance
baseline_pipeline = PredictionBaselinePipeline()
