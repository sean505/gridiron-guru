"""
NFL Data Cache System
Manages caching of nfl_data_py datasets for improved performance
"""

import logging
import pickle
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import pandas as pd

# Import nfl_data_py
try:
    import nfl_data_py as nfl
    NFL_DATA_AVAILABLE = True
except ImportError:
    NFL_DATA_AVAILABLE = False
    logging.warning("nfl_data_py not available. NFL data caching will be disabled.")

logger = logging.getLogger(__name__)


class NFLDataCache:
    """
    Manages caching of nfl_data_py datasets to avoid repeated API calls.
    
    Features:
    - Automatic cache expiration
    - Compressed storage for large datasets
    - Metadata tracking for cache management
    - Fallback to live data if cache fails
    """
    
    def __init__(self, cache_dir: str = "api/data/cache/nfl_data"):
        """Initialize the NFL data cache."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache settings
        self.cache_duration = timedelta(days=7)  # Weekly cache refresh
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        
        # Load existing metadata
        self.metadata = self._load_metadata()
        
        logger.info(f"NFLDataCache initialized at {self.cache_dir}")
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata from file."""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.warning(f"Error loading cache metadata: {e}")
            return {}
    
    def _save_metadata(self):
        """Save cache metadata to file."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving cache metadata: {e}")
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get the file path for a cache key."""
        return self.cache_dir / f"{cache_key}.pkl"
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is still valid."""
        if cache_key not in self.metadata:
            return False
        
        cache_time = datetime.fromisoformat(self.metadata[cache_key]['created_at'])
        return datetime.now() - cache_time < self.cache_duration
    
    def cache_weekly_data(self, season: int, week: int) -> bool:
        """
        Cache all required nfl_data_py datasets for a specific week.
        
        Args:
            season: NFL season year
            week: Week number
            
        Returns:
            True if successful, False otherwise
        """
        if not NFL_DATA_AVAILABLE:
            logger.warning("nfl_data_py not available, skipping cache")
            return False
        
        try:
            logger.info(f"Caching NFL data for season {season}, week {week}")
            
            # Cache play-by-play data (most expensive)
            pbp_success = self._cache_pbp_data(season)
            
            # Cache team stats
            team_stats_success = self._cache_team_stats(season)
            
            # Cache schedules
            schedules_success = self._cache_schedules(season)
            
            # Cache player stats
            player_stats_success = self._cache_player_stats(season)
            
            # Update metadata
            self.metadata[f"weekly_{season}_{week}"] = {
                'created_at': datetime.now().isoformat(),
                'season': season,
                'week': week,
                'pbp_success': pbp_success,
                'team_stats_success': team_stats_success,
                'schedules_success': schedules_success,
                'player_stats_success': player_stats_success
            }
            self._save_metadata()
            
            success = all([pbp_success, team_stats_success, schedules_success, player_stats_success])
            logger.info(f"Cache operation {'successful' if success else 'partially successful'}")
            return success
            
        except Exception as e:
            logger.error(f"Error caching weekly data: {e}")
            return False
    
    def _cache_pbp_data(self, season: int) -> bool:
        """Cache play-by-play data for a season."""
        try:
            cache_key = f"pbp_{season}"
            if self._is_cache_valid(cache_key):
                logger.info(f"PBP data for season {season} already cached and valid")
                return True
            
            logger.info(f"Fetching PBP data for season {season}")
            pbp_data = nfl.import_pbp_data([season], cache=True)
            
            if pbp_data.empty:
                logger.warning(f"No PBP data available for season {season}")
                return False
            
            # Save to cache
            cache_path = self._get_cache_path(cache_key)
            with open(cache_path, 'wb') as f:
                pickle.dump(pbp_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Update metadata
            self.metadata[cache_key] = {
                'created_at': datetime.now().isoformat(),
                'season': season,
                'record_count': len(pbp_data),
                'file_size_mb': cache_path.stat().st_size / (1024 * 1024)
            }
            
            logger.info(f"Cached PBP data for season {season}: {len(pbp_data)} records")
            return True
            
        except Exception as e:
            logger.error(f"Error caching PBP data for season {season}: {e}")
            return False
    
    def _cache_team_stats(self, season: int) -> bool:
        """Cache team statistics for a season."""
        try:
            cache_key = f"team_stats_{season}"
            if self._is_cache_valid(cache_key):
                logger.info(f"Team stats for season {season} already cached and valid")
                return True
            
            logger.info(f"Fetching team stats for season {season}")
            team_stats = nfl.import_team_desc([season])
            
            if team_stats.empty:
                logger.warning(f"No team stats available for season {season}")
                return False
            
            # Save to cache
            cache_path = self._get_cache_path(cache_key)
            with open(cache_path, 'wb') as f:
                pickle.dump(team_stats, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Update metadata
            self.metadata[cache_key] = {
                'created_at': datetime.now().isoformat(),
                'season': season,
                'record_count': len(team_stats),
                'file_size_mb': cache_path.stat().st_size / (1024 * 1024)
            }
            
            logger.info(f"Cached team stats for season {season}: {len(team_stats)} records")
            return True
            
        except Exception as e:
            logger.error(f"Error caching team stats for season {season}: {e}")
            return False
    
    def _cache_schedules(self, season: int) -> bool:
        """Cache schedules for a season."""
        try:
            cache_key = f"schedules_{season}"
            if self._is_cache_valid(cache_key):
                logger.info(f"Schedules for season {season} already cached and valid")
                return True
            
            logger.info(f"Fetching schedules for season {season}")
            schedules = nfl.import_schedules([season])
            
            if schedules.empty:
                logger.warning(f"No schedules available for season {season}")
                return False
            
            # Save to cache
            cache_path = self._get_cache_path(cache_key)
            with open(cache_path, 'wb') as f:
                pickle.dump(schedules, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Update metadata
            self.metadata[cache_key] = {
                'created_at': datetime.now().isoformat(),
                'season': season,
                'record_count': len(schedules),
                'file_size_mb': cache_path.stat().st_size / (1024 * 1024)
            }
            
            logger.info(f"Cached schedules for season {season}: {len(schedules)} records")
            return True
            
        except Exception as e:
            logger.error(f"Error caching schedules for season {season}: {e}")
            return False
    
    def _cache_player_stats(self, season: int) -> bool:
        """Cache player statistics for a season."""
        try:
            cache_key = f"player_stats_{season}"
            if self._is_cache_valid(cache_key):
                logger.info(f"Player stats for season {season} already cached and valid")
                return True
            
            logger.info(f"Fetching player stats for season {season}")
            player_stats = nfl.import_seasonal_data([season])
            
            if player_stats.empty:
                logger.warning(f"No player stats available for season {season}")
                return False
            
            # Save to cache
            cache_path = self._get_cache_path(cache_key)
            with open(cache_path, 'wb') as f:
                pickle.dump(player_stats, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Update metadata
            self.metadata[cache_key] = {
                'created_at': datetime.now().isoformat(),
                'season': season,
                'record_count': len(player_stats),
                'file_size_mb': cache_path.stat().st_size / (1024 * 1024)
            }
            
            logger.info(f"Cached player stats for season {season}: {len(player_stats)} records")
            return True
            
        except Exception as e:
            logger.error(f"Error caching player stats for season {season}: {e}")
            return False
    
    def get_cached_data(self, data_type: str, season: int) -> Optional[pd.DataFrame]:
        """
        Retrieve cached data for a specific type and season.
        
        Args:
            data_type: Type of data ('pbp', 'team_stats', 'schedules', 'player_stats')
            season: NFL season year
            
        Returns:
            Cached DataFrame or None if not available
        """
        try:
            cache_key = f"{data_type}_{season}"
            
            if not self._is_cache_valid(cache_key):
                logger.info(f"Cache for {data_type} season {season} is invalid or expired")
                return None
            
            cache_path = self._get_cache_path(cache_key)
            if not cache_path.exists():
                logger.warning(f"Cache file not found for {cache_key}")
                return None
            
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            
            logger.debug(f"Retrieved cached {data_type} data for season {season}")
            return data
            
        except Exception as e:
            logger.error(f"Error retrieving cached data for {data_type} season {season}: {e}")
            return None
    
    def get_fresh_data(self, data_type: str, season: int) -> Optional[pd.DataFrame]:
        """
        Get fresh data from nfl_data_py (bypassing cache).
        
        Args:
            data_type: Type of data to fetch
            season: NFL season year
            
        Returns:
            Fresh DataFrame or None if error
        """
        if not NFL_DATA_AVAILABLE:
            logger.warning("nfl_data_py not available")
            return None
        
        try:
            if data_type == "pbp":
                return nfl.import_pbp_data([season], cache=True)
            elif data_type == "team_stats":
                return nfl.import_team_desc([season])
            elif data_type == "schedules":
                return nfl.import_schedules([season])
            elif data_type == "player_stats":
                return nfl.import_seasonal_data([season])
            else:
                logger.error(f"Unknown data type: {data_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching fresh data for {data_type} season {season}: {e}")
            return None
    
    def clear_cache(self, data_type: str = None, season: int = None) -> int:
        """
        Clear cache entries.
        
        Args:
            data_type: Specific data type to clear (None for all)
            season: Specific season to clear (None for all)
            
        Returns:
            Number of files deleted
        """
        try:
            deleted_count = 0
            
            if data_type and season:
                # Clear specific data type and season
                cache_key = f"{data_type}_{season}"
                cache_path = self._get_cache_path(cache_key)
                if cache_path.exists():
                    cache_path.unlink()
                    deleted_count += 1
                    if cache_key in self.metadata:
                        del self.metadata[cache_key]
            else:
                # Clear all cache files
                for cache_file in self.cache_dir.glob("*.pkl"):
                    cache_file.unlink()
                    deleted_count += 1
                
                # Clear metadata
                self.metadata = {}
            
            self._save_metadata()
            logger.info(f"Cleared {deleted_count} cache files")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return 0
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring."""
        try:
            total_size = 0
            file_count = 0
            
            for cache_file in self.cache_dir.glob("*.pkl"):
                total_size += cache_file.stat().st_size
                file_count += 1
            
            return {
                'total_files': file_count,
                'total_size_mb': total_size / (1024 * 1024),
                'cache_entries': len(self.metadata),
                'cache_dir': str(self.cache_dir)
            }
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {}


# Global cache instance
nfl_data_cache = NFLDataCache()
