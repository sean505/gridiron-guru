"""
Training data preprocessor for NFL prediction engine.

This module prepares historical NFL data for model training by loading
game logs and season data, engineering features, and creating training sets.
"""

import json
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
from pathlib import Path
import logging
from datetime import datetime

# Simple data preprocessing without complex dependencies

logger = logging.getLogger(__name__)

class TrainingDataPreprocessor:
    """Prepares historical data for model training"""
    
    def __init__(self, data_path: str = "data"):
        self.data_path = Path(data_path)
        
        # Load historical data
        self.game_logs = self._load_game_logs()
        self.season_data = self._load_season_data()
        
        logger.info(f"Loaded {len(self.game_logs)} game logs and {len(self.season_data)} season records")
    
    def _load_game_logs(self) -> pd.DataFrame:
        """Load game-by-game historical data"""
        try:
            game_log_path = self.data_path / "game_log.json"
            if not game_log_path.exists():
                logger.warning(f"Game log file not found at {game_log_path}")
                return pd.DataFrame()
            
            with open(game_log_path, 'r') as f:
                game_data = json.load(f)
            
            # Convert to DataFrame
            df = pd.DataFrame(game_data)
            
            # Ensure proper data types and normalize column names
            if 'season' in df.columns:
                df['season'] = df['season'].astype(int)
            if 'week' in df.columns:
                df['week'] = df['week'].astype(int)
            
            # Normalize team column names
            if 'awayTeam' in df.columns:
                df['away_team'] = df['awayTeam']
            if 'homeTeam' in df.columns:
                df['home_team'] = df['homeTeam']
            if 'awayTeamShort' in df.columns:
                df['away_team_short'] = df['awayTeamShort']
            if 'homeTeamShort' in df.columns:
                df['home_team_short'] = df['homeTeamShort']
            
            # Normalize score column names
            if 'AwayScore' in df.columns:
                df['away_score'] = df['AwayScore']
            if 'HomeScore' in df.columns:
                df['home_score'] = df['HomeScore']
            
            # Normalize other columns
            if 'Winner' in df.columns:
                df['winner'] = df['Winner']
            if 'VegasLine' in df.columns:
                df['vegas_line'] = df['VegasLine']
            
            logger.info(f"Loaded {len(df)} games from {df['season'].min()}-{df['season'].max()}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading game logs: {e}")
            return pd.DataFrame()
    
    def _load_season_data(self) -> pd.DataFrame:
        """Load season-level team statistics"""
        try:
            season_path = self.data_path / "season_data_by_team.json"
            if not season_path.exists():
                logger.warning(f"Season data file not found at {season_path}")
                return pd.DataFrame()
            
            with open(season_path, 'r') as f:
                season_data = json.load(f)
            
            # Convert to DataFrame
            df = pd.DataFrame(season_data)
            
            # Ensure proper data types and normalize column names
            if 'Season' in df.columns:
                df['season'] = df['Season'].astype(int)
            elif 'season' in df.columns:
                df['season'] = df['season'].astype(int)
            
            # Normalize team column names
            if 'Team' in df.columns:
                df['team'] = df['Team']
            elif 'team' not in df.columns:
                logger.warning("No team column found in season data")
            
            logger.info(f"Loaded {len(df)} season records from {df['season'].min()}-{df['season'].max()}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading season data: {e}")
            return pd.DataFrame()
    
    def prepare_training_data(
        self, 
        start_season: int = 2008, 
        end_season: int = 2023,
        validation_split: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare training data from historical games.
        
        Args:
            start_season: First season to include in training
            end_season: Last season to include in training
            validation_split: Fraction of data to use for validation
            
        Returns:
            X_train, X_val, y_train, y_val
        """
        logger.info(f"Preparing training data for seasons {start_season}-{end_season}")
        
        # Filter games for training period
        training_games = self.game_logs[
            (self.game_logs['season'] >= start_season) & 
            (self.game_logs['season'] <= end_season)
        ].copy()
        
        if training_games.empty:
            raise ValueError(f"No training data found for seasons {start_season}-{end_season}")
        
        logger.info(f"Found {len(training_games)} games for training")
        
        # Create training examples
        training_examples = []
        
        for _, game in training_games.iterrows():
            try:
                # Extract features for this game
                features = self._extract_game_features(game)
                if features is not None:
                    training_examples.append(features)
            except Exception as e:
                logger.warning(f"Error processing game {game.get('game_id', 'unknown')}: {e}")
                continue
        
        if not training_examples:
            raise ValueError("No valid training examples could be created")
        
        # Convert to DataFrame
        X = pd.DataFrame(training_examples)
        
        # Create target variable (1 if home team wins, 0 if away team wins)
        y = self._create_target_variable(training_games)
        
        # Ensure X and y have same length
        min_length = min(len(X), len(y))
        X = X.iloc[:min_length]
        y = y.iloc[:min_length]
        
        # Split into train/validation
        split_idx = int(len(X) * (1 - validation_split))
        
        X_train = X.iloc[:split_idx]
        X_val = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_val = y.iloc[split_idx:]
        
        logger.info(f"Training set: {len(X_train)} examples")
        logger.info(f"Validation set: {len(X_val)} examples")
        logger.info(f"Feature count: {X_train.shape[1]}")
        
        return X_train, X_val, y_train, y_val
    
    def _extract_game_features(self, game: pd.Series) -> Optional[Dict]:
        """Extract features for a single game"""
        try:
            # Basic game info
            season = game.get('season', 2024)
            week = game.get('week', 1)
            home_team = game.get('home_team', '')
            away_team = game.get('away_team', '')
            
            if not home_team or not away_team:
                return None
            
            # Get team stats going into this game
            home_stats = self._get_team_stats_at_game(home_team, season, week)
            away_stats = self._get_team_stats_at_game(away_team, season, week)
            
            if home_stats is None or away_stats is None:
                return None
            
            # Create feature vector
            features = {
                # Game context
                'season': season,
                'week': week,
                'is_playoffs': 1 if week > 17 else 0,
                
                # Home team features
                'home_wins': home_stats.get('wins', 0),
                'home_losses': home_stats.get('losses', 0),
                'home_win_pct': home_stats.get('win_pct', 0.0),
                'home_points_for': home_stats.get('points_for', 0.0),
                'home_points_against': home_stats.get('points_against', 0.0),
                'home_point_diff': home_stats.get('point_diff', 0.0),
                'home_offensive_rank': home_stats.get('offensive_rank', 16),
                'home_defensive_rank': home_stats.get('defensive_rank', 16),
                
                # Away team features
                'away_wins': away_stats.get('wins', 0),
                'away_losses': away_stats.get('losses', 0),
                'away_win_pct': away_stats.get('win_pct', 0.0),
                'away_points_for': away_stats.get('points_for', 0.0),
                'away_points_against': away_stats.get('points_against', 0.0),
                'away_point_diff': away_stats.get('point_diff', 0.0),
                'away_offensive_rank': away_stats.get('offensive_rank', 16),
                'away_defensive_rank': away_stats.get('defensive_rank', 16),
                
                # Head-to-head features
                'h2h_home_wins': self._get_h2h_record(home_team, away_team, season, week, 'home'),
                'h2h_away_wins': self._get_h2h_record(home_team, away_team, season, week, 'away'),
                
                # Vegas line features (if available)
                'spread': game.get('spread', 0.0),
                'total': game.get('total', 45.0),
                'home_favorite': 1 if game.get('spread', 0) < 0 else 0,
                
                # Recent form features
                'home_recent_form': self._get_recent_form(home_team, season, week),
                'away_recent_form': self._get_recent_form(away_team, season, week),
                
                # Rest advantage
                'home_rest_days': self._get_rest_days(home_team, season, week),
                'away_rest_days': self._get_rest_days(away_team, season, week),
            }
            
            return features
            
        except Exception as e:
            logger.warning(f"Error extracting features for game: {e}")
            return None
    
    def _get_team_stats_at_game(self, team: str, season: int, week: int) -> Optional[Dict]:
        """Get team statistics as they were going into a specific game"""
        try:
            # Get season data for this team
            team_season_data = self.season_data[
                (self.season_data['team'] == team) & 
                (self.season_data['season'] == season)
            ]
            
            if team_season_data.empty:
                # Fallback: use previous season data
                prev_season_data = self.season_data[
                    (self.season_data['team'] == team) & 
                    (self.season_data['season'] == season - 1)
                ]
                if not prev_season_data.empty:
                    return prev_season_data.iloc[0].to_dict()
                return None
            
            # Get games played up to this week
            games_played = self.game_logs[
                ((self.game_logs['home_team'] == team) | (self.game_logs['away_team'] == team)) &
                (self.game_logs['season'] == season) &
                (self.game_logs['week'] < week)
            ]
            
            # Calculate current season stats
            wins = 0
            losses = 0
            points_for = 0
            points_against = 0
            
            for _, game in games_played.iterrows():
                if game['home_team'] == team:
                    # Home game
                    home_score = game.get('home_score', 0)
                    away_score = game.get('away_score', 0)
                    points_for += home_score
                    points_against += away_score
                    if home_score > away_score:
                        wins += 1
                    else:
                        losses += 1
                else:
                    # Away game
                    home_score = game.get('home_score', 0)
                    away_score = game.get('away_score', 0)
                    points_for += away_score
                    points_against += home_score
                    if away_score > home_score:
                        wins += 1
                    else:
                        losses += 1
            
            # Use season data as fallback if no games played yet
            if len(games_played) == 0:
                season_record = team_season_data.iloc[0]
                return {
                    'wins': 0,
                    'losses': 0,
                    'win_pct': 0.0,
                    'points_for': season_record.get('points_for', 0.0),
                    'points_against': season_record.get('points_against', 0.0),
                    'point_diff': season_record.get('point_diff', 0.0),
                    'offensive_rank': season_record.get('offensive_rank', 16),
                    'defensive_rank': season_record.get('defensive_rank', 16),
                }
            
            # Calculate current stats
            total_games = wins + losses
            win_pct = wins / total_games if total_games > 0 else 0.0
            point_diff = points_for - points_against
            
            return {
                'wins': wins,
                'losses': losses,
                'win_pct': win_pct,
                'points_for': points_for,
                'points_against': points_against,
                'point_diff': point_diff,
                'offensive_rank': 16,  # Placeholder - would need more complex calculation
                'defensive_rank': 16,  # Placeholder - would need more complex calculation
            }
            
        except Exception as e:
            logger.warning(f"Error getting team stats for {team}: {e}")
            return None
    
    def _get_h2h_record(self, home_team: str, away_team: str, season: int, week: int, perspective: str) -> int:
        """Get head-to-head record between teams"""
        try:
            # Look at previous meetings between these teams
            h2h_games = self.game_logs[
                ((self.game_logs['home_team'] == home_team) & (self.game_logs['away_team'] == away_team)) |
                ((self.game_logs['home_team'] == away_team) & (self.game_logs['away_team'] == home_team))
            ]
            
            # Filter to games before current week
            h2h_games = h2h_games[
                (h2h_games['season'] < season) | 
                ((h2h_games['season'] == season) & (h2h_games['week'] < week))
            ]
            
            wins = 0
            for _, game in h2h_games.iterrows():
                if perspective == 'home':
                    # Count wins for the home team in this matchup
                    if game['home_team'] == home_team and game.get('home_score', 0) > game.get('away_score', 0):
                        wins += 1
                    elif game['away_team'] == home_team and game.get('away_score', 0) > game.get('home_score', 0):
                        wins += 1
                else:
                    # Count wins for the away team in this matchup
                    if game['home_team'] == away_team and game.get('home_score', 0) > game.get('away_score', 0):
                        wins += 1
                    elif game['away_team'] == away_team and game.get('away_score', 0) > game.get('home_score', 0):
                        wins += 1
            
            return wins
            
        except Exception as e:
            logger.warning(f"Error getting H2H record: {e}")
            return 0
    
    def _get_recent_form(self, team: str, season: int, week: int, games: int = 5) -> float:
        """Get team's recent form (win percentage in last N games)"""
        try:
            recent_games = self.game_logs[
                ((self.game_logs['home_team'] == team) | (self.game_logs['away_team'] == team)) &
                (self.game_logs['season'] == season) &
                (self.game_logs['week'] < week)
            ].tail(games)
            
            if recent_games.empty:
                return 0.5  # Neutral form
            
            wins = 0
            for _, game in recent_games.iterrows():
                if game['home_team'] == team:
                    if game.get('home_score', 0) > game.get('away_score', 0):
                        wins += 1
                else:
                    if game.get('away_score', 0) > game.get('home_score', 0):
                        wins += 1
            
            return wins / len(recent_games)
            
        except Exception as e:
            logger.warning(f"Error getting recent form for {team}: {e}")
            return 0.5
    
    def _get_rest_days(self, team: str, season: int, week: int) -> int:
        """Get days of rest for team before this game"""
        try:
            # Find team's previous game
            prev_game = self.game_logs[
                ((self.game_logs['home_team'] == team) | (self.game_logs['away_team'] == team)) &
                (self.game_logs['season'] == season) &
                (self.game_logs['week'] < week)
            ].tail(1)
            
            if prev_game.empty:
                return 7  # Default to 7 days if no previous game
            
            # Calculate days between games (simplified)
            return 7  # Placeholder - would need actual game dates
            
        except Exception as e:
            logger.warning(f"Error getting rest days for {team}: {e}")
            return 7
    
    def _create_target_variable(self, games: pd.DataFrame) -> pd.Series:
        """Create target variable (1 if home team wins, 0 if away team wins)"""
        targets = []
        
        for _, game in games.iterrows():
            home_score = game.get('home_score', 0)
            away_score = game.get('away_score', 0)
            
            # 1 if home team wins, 0 if away team wins
            target = 1 if home_score > away_score else 0
            targets.append(target)
        
        return pd.Series(targets)
    
    def create_holdout_test_set(self, test_season: int = 2024) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create separate test set from specified season.
        This data is never used during training.
        """
        logger.info(f"Creating holdout test set for season {test_season}")
        
        # Filter games for test season
        test_games = self.game_logs[self.game_logs['season'] == test_season].copy()
        
        if test_games.empty:
            logger.warning(f"No test data found for season {test_season}")
            return pd.DataFrame(), pd.Series()
        
        # Create test examples
        test_examples = []
        
        for _, game in test_games.iterrows():
            try:
                features = self._extract_game_features(game)
                if features is not None:
                    test_examples.append(features)
            except Exception as e:
                logger.warning(f"Error processing test game {game.get('game_id', 'unknown')}: {e}")
                continue
        
        if not test_examples:
            logger.warning("No valid test examples could be created")
            return pd.DataFrame(), pd.Series()
        
        # Convert to DataFrame
        X_test = pd.DataFrame(test_examples)
        
        # Create target variable
        y_test = self._create_target_variable(test_games)
        
        # Ensure same length
        min_length = min(len(X_test), len(y_test))
        X_test = X_test.iloc[:min_length]
        y_test = y_test.iloc[:min_length]
        
        logger.info(f"Created test set with {len(X_test)} examples")
        
        return X_test, y_test
    
    def balance_dataset(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Balance the dataset to handle class imbalance.
        Ensures equal representation of home/away wins.
        """
        logger.info("Balancing dataset...")
        
        # Count class distribution
        class_counts = y.value_counts()
        logger.info(f"Class distribution: {class_counts.to_dict()}")
        
        # If classes are balanced (within 10%), return as-is
        if abs(class_counts[0] - class_counts[1]) / len(y) < 0.1:
            logger.info("Dataset is already balanced")
            return X, y
        
        # Undersample majority class
        minority_class = class_counts.idxmin()
        majority_class = class_counts.idxmax()
        
        minority_indices = y[y == minority_class].index
        majority_indices = y[y == majority_class].index
        
        # Sample majority class to match minority class size
        n_minority = len(minority_indices)
        majority_sample = np.random.choice(majority_indices, size=n_minority, replace=False)
        
        # Combine balanced indices
        balanced_indices = np.concatenate([minority_indices, majority_sample])
        
        # Create balanced dataset
        X_balanced = X.loc[balanced_indices].reset_index(drop=True)
        y_balanced = y.loc[balanced_indices].reset_index(drop=True)
        
        logger.info(f"Balanced dataset: {len(X_balanced)} examples")
        logger.info(f"New class distribution: {y_balanced.value_counts().to_dict()}")
        
        return X_balanced, y_balanced
