"""
Team Feature Database
Manages storage and retrieval of pre-computed team features
"""

import sqlite3
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd

logger = logging.getLogger(__name__)


class TeamFeatureDatabase:
    """
    Manages storage and retrieval of pre-computed team features.
    
    Uses SQLite for simplicity and performance. Can be upgraded to PostgreSQL
    for production use with multiple instances.
    """
    
    def __init__(self, db_path: str = "api/data/team_features.db"):
        """Initialize the team feature database."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database schema
        self._init_database()
        
        logger.info(f"TeamFeatureDatabase initialized at {self.db_path}")
    
    def _init_database(self):
        """Initialize the database schema."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create team_weekly_features table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS team_weekly_features (
                        team_name VARCHAR(3) NOT NULL,
                        season INTEGER NOT NULL,
                        week INTEGER NOT NULL,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        
                        -- EPA Metrics
                        off_epa_per_play REAL,
                        def_epa_per_play REAL,
                        off_success_rate REAL,
                        def_success_rate REAL,
                        explosive_play_rate REAL,
                        explosive_play_allowed_rate REAL,
                        
                        -- Situational Efficiency
                        red_zone_td_pct REAL,
                        third_down_pct REAL,
                        turnover_rate REAL,
                        drive_success_rate REAL,
                        
                        -- Advanced Analytics
                        field_position_advantage REAL,
                        time_of_possession_pct REAL,
                        home_away_epa_diff REAL,
                        recent_form_trend REAL,
                        
                        -- Contextual Factors
                        rest_days INTEGER,
                        travel_distance REAL,
                        weather_factor REAL,
                        
                        -- Raw feature data (JSON for flexibility)
                        raw_features TEXT,
                        
                        PRIMARY KEY (team_name, season, week)
                    )
                """)
                
                # Create index for faster lookups
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_team_season_week 
                    ON team_weekly_features(team_name, season, week)
                """)
                
                # Create cache_metadata table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS cache_metadata (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        season INTEGER NOT NULL,
                        week INTEGER NOT NULL,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        team_count INTEGER,
                        cache_version VARCHAR(10) DEFAULT '1.0',
                        status VARCHAR(20) DEFAULT 'active'
                    )
                """)
                
                conn.commit()
                logger.info("Database schema initialized successfully")
                
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    def store_features(self, team: str, season: int, week: int, features: List[float], 
                      raw_features: Dict = None) -> bool:
        """
        Store pre-computed features for a team and week.
        
        Args:
            team: Team abbreviation (e.g., 'KC')
            season: NFL season year
            week: Week number
            features: List of 17 feature values
            raw_features: Optional raw feature data as dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if len(features) != 17:
                logger.warning(f"Expected 17 features, got {len(features)}")
                return False
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Prepare feature data
                feature_data = {
                    'team_name': team,
                    'season': season,
                    'week': week,
                    'off_epa_per_play': features[0],
                    'def_epa_per_play': features[1],
                    'off_success_rate': features[2],
                    'def_success_rate': features[3],
                    'explosive_play_rate': features[4],
                    'explosive_play_allowed_rate': features[5],
                    'red_zone_td_pct': features[6],
                    'third_down_pct': features[7],
                    'turnover_rate': features[8],
                    'drive_success_rate': features[9],
                    'field_position_advantage': features[10],
                    'time_of_possession_pct': features[11],
                    'home_away_epa_diff': features[12],
                    'recent_form_trend': features[13],
                    'rest_days': int(features[14]),
                    'travel_distance': features[15],
                    'weather_factor': features[16],
                    'raw_features': json.dumps(raw_features) if raw_features else None
                }
                
                # Insert or replace features
                cursor.execute("""
                    INSERT OR REPLACE INTO team_weekly_features 
                    (team_name, season, week, off_epa_per_play, def_epa_per_play, 
                     off_success_rate, def_success_rate, explosive_play_rate, 
                     explosive_play_allowed_rate, red_zone_td_pct, third_down_pct, 
                     turnover_rate, drive_success_rate, field_position_advantage, 
                     time_of_possession_pct, home_away_epa_diff, recent_form_trend, 
                     rest_days, travel_distance, weather_factor, raw_features)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, tuple(feature_data.values()))
                
                conn.commit()
                logger.debug(f"Stored features for {team} season {season} week {week}")
                return True
                
        except Exception as e:
            logger.error(f"Error storing features for {team}: {e}")
            return False
    
    def get_features(self, team: str, season: int, week: int) -> Optional[List[float]]:
        """
        Retrieve cached features for a team and week.
        
        Args:
            team: Team abbreviation
            season: NFL season year
            week: Week number
            
        Returns:
            List of 17 feature values or None if not found
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT off_epa_per_play, def_epa_per_play, off_success_rate, 
                           def_success_rate, explosive_play_rate, explosive_play_allowed_rate,
                           red_zone_td_pct, third_down_pct, turnover_rate, drive_success_rate,
                           field_position_advantage, time_of_possession_pct, home_away_epa_diff,
                           recent_form_trend, rest_days, travel_distance, weather_factor
                    FROM team_weekly_features 
                    WHERE team_name = ? AND season = ? AND week = ?
                """, (team, season, week))
                
                result = cursor.fetchone()
                if result:
                    return [float(x) for x in result]
                else:
                    return None
                    
        except Exception as e:
            logger.error(f"Error retrieving features for {team}: {e}")
            return None
    
    def get_features_batch(self, teams: List[str], season: int, week: int) -> Dict[str, List[float]]:
        """
        Retrieve features for multiple teams in a single query.
        
        Args:
            teams: List of team abbreviations
            season: NFL season year
            week: Week number
            
        Returns:
            Dictionary mapping team names to feature lists
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                placeholders = ','.join('?' for _ in teams)
                cursor.execute(f"""
                    SELECT team_name, off_epa_per_play, def_epa_per_play, off_success_rate, 
                           def_success_rate, explosive_play_rate, explosive_play_allowed_rate,
                           red_zone_td_pct, third_down_pct, turnover_rate, drive_success_rate,
                           field_position_advantage, time_of_possession_pct, home_away_epa_diff,
                           recent_form_trend, rest_days, travel_distance, weather_factor
                    FROM team_weekly_features 
                    WHERE team_name IN ({placeholders}) AND season = ? AND week = ?
                """, teams + [season, week])
                
                results = {}
                for row in cursor.fetchall():
                    team_name = row[0]
                    features = [float(x) for x in row[1:]]
                    results[team_name] = features
                
                return results
                
        except Exception as e:
            logger.error(f"Error retrieving batch features: {e}")
            return {}
    
    def get_last_update(self, season: int, week: int) -> Optional[datetime]:
        """Get the last update time for a specific season and week."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT MAX(last_updated) FROM team_weekly_features 
                    WHERE season = ? AND week = ?
                """, (season, week))
                
                result = cursor.fetchone()
                if result and result[0]:
                    return datetime.fromisoformat(result[0])
                return None
                
        except Exception as e:
            logger.error(f"Error getting last update time: {e}")
            return None
    
    def update_cache_metadata(self, season: int, week: int, team_count: int, 
                            status: str = 'active') -> bool:
        """Update cache metadata for tracking."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO cache_metadata 
                    (season, week, team_count, status, last_updated)
                    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (season, week, team_count, status))
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Error updating cache metadata: {e}")
            return False
    
    def get_cache_status(self, season: int, week: int) -> Dict[str, Any]:
        """Get cache status for a specific season and week."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT team_count, status, last_updated FROM cache_metadata 
                    WHERE season = ? AND week = ?
                    ORDER BY last_updated DESC LIMIT 1
                """, (season, week))
                
                result = cursor.fetchone()
                if result:
                    return {
                        'team_count': result[0],
                        'status': result[1],
                        'last_updated': datetime.fromisoformat(result[2])
                    }
                return {}
                
        except Exception as e:
            logger.error(f"Error getting cache status: {e}")
            return {}
    
    def is_cache_valid(self, season: int, week: int, max_age_hours: int = 24) -> bool:
        """Check if cache is still valid for a season and week."""
        try:
            last_update = self.get_last_update(season, week)
            if not last_update:
                return False
            
            max_age = timedelta(hours=max_age_hours)
            return datetime.now() - last_update < max_age
            
        except Exception as e:
            logger.error(f"Error checking cache validity: {e}")
            return False
    
    def clear_old_cache(self, max_age_days: int = 30) -> int:
        """Clear old cache entries to free up space."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cutoff_date = datetime.now() - timedelta(days=max_age_days)
                
                cursor.execute("""
                    DELETE FROM team_weekly_features 
                    WHERE last_updated < ?
                """, (cutoff_date.isoformat(),))
                
                deleted_count = cursor.rowcount
                conn.commit()
                
                logger.info(f"Cleared {deleted_count} old cache entries")
                return deleted_count
                
        except Exception as e:
            logger.error(f"Error clearing old cache: {e}")
            return 0
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics for monitoring."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get total records
                cursor.execute("SELECT COUNT(*) FROM team_weekly_features")
                total_records = cursor.fetchone()[0]
                
                # Get unique teams
                cursor.execute("SELECT COUNT(DISTINCT team_name) FROM team_weekly_features")
                unique_teams = cursor.fetchone()[0]
                
                # Get date range
                cursor.execute("""
                    SELECT MIN(last_updated), MAX(last_updated) 
                    FROM team_weekly_features
                """)
                date_range = cursor.fetchone()
                
                return {
                    'total_records': total_records,
                    'unique_teams': unique_teams,
                    'date_range': date_range,
                    'database_size_mb': self.db_path.stat().st_size / (1024 * 1024)
                }
                
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {}


# Global database instance
team_feature_db = TeamFeatureDatabase()
