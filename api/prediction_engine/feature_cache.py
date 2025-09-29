#!/usr/bin/env python3
"""
Enhanced Feature Cache System
Pre-calculates and caches enhanced features to avoid on-demand computation
"""

import json
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from .enhanced_features import EnhancedFeatureExtractor

logger = logging.getLogger(__name__)

class FeatureCache:
    """Manages caching of enhanced features for all teams"""
    
    def __init__(self, cache_dir: Path = None):
        self.cache_dir = cache_dir or Path(__file__).parent / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        
        self.extractor = EnhancedFeatureExtractor()
        self.cache_file = self.cache_dir / "enhanced_features_cache.json"
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        
        # Cache expiration (24 hours)
        self.cache_duration = timedelta(hours=24)
        
    def is_cache_valid(self) -> bool:
        """Check if cache is still valid"""
        if not self.metadata_file.exists():
            return False
            
        try:
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
            
            cache_time = datetime.fromisoformat(metadata['created_at'])
            return datetime.now() - cache_time < self.cache_duration
        except Exception as e:
            logger.warning(f"Error checking cache validity: {e}")
            return False
    
    def get_team_features(self, team: str, week: int) -> Optional[List[float]]:
        """Get cached features for a team and week"""
        if not self.is_cache_valid():
            return None
            
        try:
            with open(self.cache_file, 'r') as f:
                cache_data = json.load(f)
            
            team_key = f"{team}_{week}"
            return cache_data.get(team_key)
        except Exception as e:
            logger.warning(f"Error reading team features for {team}: {e}")
            return None
    
    def cache_team_features(self, team: str, week: int, features: List[float]):
        """Cache features for a team and week"""
        try:
            # Load existing cache
            cache_data = {}
            if self.cache_file.exists():
                with open(self.cache_file, 'r') as f:
                    cache_data = json.load(f)
            
            # Update with new features
            team_key = f"{team}_{week}"
            cache_data[team_key] = features
            
            # Save updated cache
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
                
            logger.info(f"Cached features for {team} week {week}")
        except Exception as e:
            logger.error(f"Error caching features for {team}: {e}")
    
    def precompute_all_teams(self, week: int, teams: List[str] = None):
        """Pre-compute features for all teams for a given week"""
        if teams is None:
            teams = [
                'ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE', 'DAL', 'DEN',
                'DET', 'GB', 'HOU', 'IND', 'JAX', 'KC', 'LAC', 'LAR', 'LV', 'MIA',
                'MIN', 'NE', 'NO', 'NYG', 'NYJ', 'PHI', 'PIT', 'SEA', 'SF', 'TB', 'TEN', 'WAS'
            ]
        
        logger.info(f"Pre-computing enhanced features for {len(teams)} teams, week {week}")
        
        cached_count = 0
        for team in teams:
            try:
                # Check if already cached
                if self.get_team_features(team, week) is not None:
                    cached_count += 1
                    continue
                
                # Generate features for the team
                features = self.extractor.extract_enhanced_features(team, 'DAL', week)  # Use dummy opponent
                
                # Cache the features
                self.cache_team_features(team, week, features)
                cached_count += 1
                
                logger.info(f"Pre-computed features for {team} ({cached_count}/{len(teams)})")
                
            except Exception as e:
                logger.error(f"Error pre-computing features for {team}: {e}")
                continue
        
        # Update metadata
        self._update_cache_metadata(week, cached_count)
        logger.info(f"Pre-computation complete: {cached_count}/{len(teams)} teams cached")
    
    def _update_cache_metadata(self, week: int, team_count: int):
        """Update cache metadata"""
        metadata = {
            'created_at': datetime.now().isoformat(),
            'week': week,
            'team_count': team_count,
            'cache_version': '1.0'
        }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def get_cached_features(self, home_team: str, away_team: str, week: int) -> List[float]:
        """Get cached features for both teams and combine them"""
        try:
            home_features = self.get_team_features(home_team, week)
            away_features = self.get_team_features(away_team, week)
            
            if home_features is None or away_features is None:
                logger.warning(f"Missing cached features for {away_team} @ {home_team}")
                return []
            
            # Combine features (home + away + rest/travel factors)
            combined_features = home_features + away_features
            
            # Add rest/travel factors (last 3 features)
            rest_travel = self.extractor.get_rest_travel_factors(home_team, away_team, week)
            combined_features.extend([
                rest_travel['rest_differential'],
                rest_travel['travel_distance'],
                rest_travel['home_advantage']
            ])
            
            return combined_features
            
        except Exception as e:
            logger.error(f"Error getting cached features: {e}")
            return []
    
    def clear_cache(self):
        """Clear all cached features"""
        try:
            if self.cache_file.exists():
                self.cache_file.unlink()
            if self.metadata_file.exists():
                self.metadata_file.unlink()
            logger.info("Cache cleared")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")

# Global cache instance
feature_cache = FeatureCache()
