"""
Weekly Feature Processor
Handles batch processing of enhanced features for all teams
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path

from .enhanced_features import enhanced_extractor
from .team_feature_database import team_feature_db
from .nfl_data_cache import nfl_data_cache

logger = logging.getLogger(__name__)


class WeeklyFeatureProcessor:
    """
    Processes enhanced features for all teams on a weekly basis.
    
    This class handles:
    - Caching nfl_data_py datasets
    - Extracting features for all teams
    - Storing features in the database
    - Managing incremental updates
    """
    
    def __init__(self):
        """Initialize the weekly feature processor."""
        self.enhanced_extractor = enhanced_extractor
        self.team_db = team_feature_db
        self.nfl_cache = nfl_data_cache
        
        # NFL team abbreviations
        self.nfl_teams = [
            'ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE', 'DAL', 'DEN',
            'DET', 'GB', 'HOU', 'IND', 'JAX', 'KC', 'LAC', 'LAR', 'LV', 'MIA',
            'MIN', 'NE', 'NO', 'NYG', 'NYJ', 'PHI', 'PIT', 'SEA', 'SF', 'TB', 'TEN', 'WAS'
        ]
        
        logger.info("WeeklyFeatureProcessor initialized")
    
    async def process_week(self, season: int, week: int, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Process all teams for a given week.
        
        Args:
            season: NFL season year
            week: Week number
            force_refresh: Force refresh even if cache is valid
            
        Returns:
            Dictionary with processing results
        """
        try:
            logger.info(f"Starting weekly processing for season {season}, week {week}")
            start_time = datetime.now()
            
            # Check if processing is needed
            if not force_refresh and self._is_processing_needed(season, week):
                logger.info(f"Processing not needed for season {season}, week {week}")
                return {
                    'status': 'skipped',
                    'reason': 'cache_valid',
                    'season': season,
                    'week': week
                }
            
            # Step 1: Cache NFL data
            logger.info("Step 1: Caching NFL data")
            cache_success = await self._cache_nfl_data(season, week)
            if not cache_success:
                logger.warning("NFL data caching failed, proceeding with limited features")
            
            # Step 2: Extract features for all teams
            logger.info("Step 2: Extracting features for all teams")
            processing_results = await self._extract_all_team_features(season, week)
            
            # Step 3: Update cache metadata
            logger.info("Step 3: Updating cache metadata")
            self.team_db.update_cache_metadata(
                season=season,
                week=week,
                team_count=processing_results['successful_teams'],
                status='completed'
            )
            
            # Calculate processing time
            processing_time = datetime.now() - start_time
            
            logger.info(f"Weekly processing completed in {processing_time.total_seconds():.2f} seconds")
            
            return {
                'status': 'completed',
                'season': season,
                'week': week,
                'processing_time_seconds': processing_time.total_seconds(),
                'successful_teams': processing_results['successful_teams'],
                'failed_teams': processing_results['failed_teams'],
                'cache_success': cache_success,
                'total_teams': len(self.nfl_teams)
            }
            
        except Exception as e:
            logger.error(f"Error in weekly processing: {e}")
            return {
                'status': 'error',
                'season': season,
                'week': week,
                'error': str(e)
            }
    
    def _is_processing_needed(self, season: int, week: int) -> bool:
        """Check if processing is needed for this week."""
        try:
            # Check if cache is valid
            if self.team_db.is_cache_valid(season, week):
                return False
            
            # Check if we have features for most teams
            team_features = self.team_db.get_features_batch(self.nfl_teams, season, week)
            if len(team_features) >= len(self.nfl_teams) * 0.8:  # 80% threshold
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking if processing needed: {e}")
            return True
    
    async def _cache_nfl_data(self, season: int, week: int) -> bool:
        """Cache NFL data for the season and week."""
        try:
            # Cache all required datasets
            cache_success = self.nfl_cache.cache_weekly_data(season, week)
            
            if cache_success:
                logger.info(f"Successfully cached NFL data for season {season}, week {week}")
            else:
                logger.warning(f"Failed to cache some NFL data for season {season}, week {week}")
            
            return cache_success
            
        except Exception as e:
            logger.error(f"Error caching NFL data: {e}")
            return False
    
    async def _extract_all_team_features(self, season: int, week: int) -> Dict[str, Any]:
        """Extract features for all teams."""
        try:
            successful_teams = []
            failed_teams = []
            
            # Process teams in batches to avoid overwhelming the system
            batch_size = 8  # Process 8 teams at a time
            teams_batches = [self.nfl_teams[i:i + batch_size] for i in range(0, len(self.nfl_teams), batch_size)]
            
            for batch_idx, team_batch in enumerate(teams_batches):
                logger.info(f"Processing team batch {batch_idx + 1}/{len(teams_batches)}: {team_batch}")
                
                # Process batch concurrently
                batch_tasks = []
                for team in team_batch:
                    task = self._extract_team_features(team, season, week)
                    batch_tasks.append(task)
                
                # Wait for batch to complete
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                # Process results
                for team, result in zip(team_batch, batch_results):
                    if isinstance(result, Exception):
                        logger.error(f"Error processing {team}: {result}")
                        failed_teams.append(team)
                    elif result:
                        successful_teams.append(team)
                    else:
                        failed_teams.append(team)
            
            logger.info(f"Feature extraction complete: {len(successful_teams)} successful, {len(failed_teams)} failed")
            
            return {
                'successful_teams': successful_teams,
                'failed_teams': failed_teams
            }
            
        except Exception as e:
            logger.error(f"Error extracting team features: {e}")
            return {
                'successful_teams': [],
                'failed_teams': self.nfl_teams
            }
    
    async def _extract_team_features(self, team: str, season: int, week: int) -> bool:
        """Extract features for a single team."""
        try:
            # Check if features already exist and are recent
            existing_features = self.team_db.get_features(team, season, week)
            if existing_features:
                logger.debug(f"Features already exist for {team} season {season} week {week}")
                return True
            
            # Extract enhanced features
            features = self.enhanced_extractor.extract_enhanced_features(
                team=team,
                opponent='DAL',  # Use dummy opponent for individual team features
                season=season,
                week=week
            )
            
            if not features or len(features) != 17:
                logger.warning(f"Invalid features for {team}: {len(features) if features else 0} features")
                return False
            
            # Store features in database
            success = self.team_db.store_features(
                team=team,
                season=season,
                week=week,
                features=features
            )
            
            if success:
                logger.debug(f"Successfully processed features for {team}")
            else:
                logger.warning(f"Failed to store features for {team}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error extracting features for {team}: {e}")
            return False
    
    def get_processing_status(self, season: int, week: int) -> Dict[str, Any]:
        """Get the processing status for a specific week."""
        try:
            cache_status = self.team_db.get_cache_status(season, week)
            db_stats = self.team_db.get_database_stats()
            cache_stats = self.nfl_cache.get_cache_stats()
            
            return {
                'season': season,
                'week': week,
                'cache_status': cache_status,
                'database_stats': db_stats,
                'nfl_cache_stats': cache_stats,
                'is_cache_valid': self.team_db.is_cache_valid(season, week)
            }
            
        except Exception as e:
            logger.error(f"Error getting processing status: {e}")
            return {
                'season': season,
                'week': week,
                'error': str(e)
            }
    
    def cleanup_old_data(self, max_age_days: int = 30) -> Dict[str, Any]:
        """Clean up old data to free up space."""
        try:
            logger.info(f"Cleaning up data older than {max_age_days} days")
            
            # Clean up team features database
            db_cleaned = self.team_db.clear_old_cache(max_age_days)
            
            # Clean up NFL data cache
            cache_cleaned = self.nfl_cache.clear_cache()
            
            return {
                'database_cleaned': db_cleaned,
                'cache_cleaned': cache_cleaned,
                'total_cleaned': db_cleaned + cache_cleaned
            }
            
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
            return {
                'error': str(e),
                'database_cleaned': 0,
                'cache_cleaned': 0,
                'total_cleaned': 0
            }


# Global processor instance
weekly_processor = WeeklyFeatureProcessor()
