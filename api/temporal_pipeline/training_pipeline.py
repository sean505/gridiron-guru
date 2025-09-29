"""
Training Pipeline for Gridiron Guru
Uses 2008-2024 historical data to train ML models
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb

from temporal_pipeline.temporal_data_collector import temporal_data_collector
from temporal_pipeline.temporal_feature_engine import TemporalFeatureEngine

logger = logging.getLogger(__name__)


class TrainingDataPipeline:
    """
    Training pipeline that uses 2008-2024 historical data to train ML models.
    
    Features:
    - Loads historical game and team data
    - Engineers features for training
    - Trains ensemble ML models
    - Validates model performance
    - Saves trained models for prediction use
    """
    
    def __init__(self, model_dir: str = "prediction_engine/models/trained"):
        """Initialize the training pipeline."""
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.feature_engine = TemporalFeatureEngine()
        self.data_collector = temporal_data_collector
        
        # Training configuration
        self.training_years = list(range(2008, 2025))  # 2008-2024
        self.validation_years = [2024]  # Use 2024 for validation
        
        # Model storage
        self.models = {}
        self.scaler = None
        self.feature_names = []
        
        logger.info(f"TrainingDataPipeline initialized with model directory: {self.model_dir}")
    
    def load_training_data(self, years: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Load historical data for training.
        
        Args:
            years: List of years to load (default: 2008-2024)
            
        Returns:
            Dictionary containing training data
        """
        try:
            if years is None:
                years = self.training_years
            
            logger.info(f"Loading training data for years: {years}")
            
            # Load historical data
            training_data = self.data_collector.get_training_data(years)
            
            logger.info(f"Loaded {training_data['total_games']} games and {training_data['total_teams']} team records")
            
            return training_data
            
        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            return {'games': {}, 'teams': {}, 'years': years, 'total_games': 0, 'total_teams': 0}
    
    def engineer_training_features(self, training_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Engineer features for training using historical data.
        
        Args:
            training_data: Dictionary containing games and team data
            
        Returns:
            DataFrame with features and target variable
        """
        try:
            logger.info("Engineering training features from historical data")
            
            features = []
            games = training_data['games']
            
            for game in games:
                try:
                    # Extract game information
                    home_team = game.get('homeTeam', '')
                    away_team = game.get('awayTeam', '')
                    season = game.get('season', 0)
                    week = game.get('week', 0)
                    
                    if not home_team or not away_team or season < 2008:
                        continue
                    
                    # Get team stats for this game
                    home_stats = self.data_collector.get_team_stats_for_training(
                        self._get_team_abbr(home_team), season, week
                    )
                    away_stats = self.data_collector.get_team_stats_for_training(
                        self._get_team_abbr(away_team), season, week
                    )
                    
                    # Create game context
                    game_context = self._create_game_context(game)
                    
                    # Engineer features
                    game_features = self.feature_engine.create_training_features(
                        home_stats, away_stats, game_context
                    )
                    
                    # Add target variable (home team win)
                    winner = game.get('Winner', 0)
                    home_won = 1 if winner == 1 else 0
                    
                    # Add game metadata
                    game_features['home_team'] = home_team
                    game_features['away_team'] = away_team
                    game_features['season'] = season
                    game_features['week'] = week
                    game_features['home_won'] = home_won
                    
                    features.append(game_features)
                    
                except Exception as e:
                    logger.warning(f"Error processing game {game_key}: {e}")
                    continue
            
            # Convert to DataFrame
            df = pd.DataFrame(features)
            
            if len(df) == 0:
                logger.error("No features generated from training data")
                return pd.DataFrame()
            
            # Store feature names
            self.feature_names = [col for col in df.columns if col not in ['home_team', 'away_team', 'season', 'week', 'home_won']]
            
            logger.info(f"Generated {len(df)} training samples with {len(self.feature_names)} features")
            logger.info(f"Feature names: {self.feature_names[:10]}...")
            
            return df
            
        except Exception as e:
            logger.error(f"Error engineering training features: {e}")
            return pd.DataFrame()
    
    def train_models(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train ensemble ML models on historical features.
        
        Args:
            features_df: DataFrame with features and target variable
            
        Returns:
            Dictionary containing trained models and performance metrics
        """
        try:
            if len(features_df) == 0:
                raise ValueError("No training data available")
            
            logger.info("Training ensemble ML models")
            
            # Prepare features and target
            X = features_df[self.feature_names].fillna(0)
            y = features_df['home_won']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train individual models
            models = {}
            
            # 1. Logistic Regression
            logger.info("Training Logistic Regression...")
            lr_model = LogisticRegression(random_state=42, max_iter=1000)
            lr_model.fit(X_train_scaled, y_train)
            lr_pred = lr_model.predict(X_test_scaled)
            lr_accuracy = accuracy_score(y_test, lr_pred)
            models['logistic_regression'] = lr_model
            logger.info(f"Logistic Regression accuracy: {lr_accuracy:.3f}")
            
            # 2. Random Forest
            logger.info("Training Random Forest...")
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_model.fit(X_train_scaled, y_train)
            rf_pred = rf_model.predict(X_test_scaled)
            rf_accuracy = accuracy_score(y_test, rf_pred)
            models['random_forest'] = rf_model
            logger.info(f"Random Forest accuracy: {rf_accuracy:.3f}")
            
            # 3. XGBoost
            logger.info("Training XGBoost...")
            xgb_model = xgb.XGBClassifier(random_state=42)
            xgb_model.fit(X_train_scaled, y_train)
            xgb_pred = xgb_model.predict(X_test_scaled)
            xgb_accuracy = accuracy_score(y_test, xgb_pred)
            models['xgboost'] = xgb_model
            logger.info(f"XGBoost accuracy: {xgb_accuracy:.3f}")
            
            # 4. Ensemble (Voting Classifier)
            logger.info("Training Ensemble...")
            ensemble = VotingClassifier([
                ('lr', lr_model),
                ('rf', rf_model),
                ('xgb', xgb_model)
            ], voting='soft')
            ensemble.fit(X_train_scaled, y_train)
            ensemble_pred = ensemble.predict(X_test_scaled)
            ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
            models['ensemble'] = ensemble
            logger.info(f"Ensemble accuracy: {ensemble_accuracy:.3f}")
            
            # Calculate performance metrics
            performance = {
                'logistic_regression': {
                    'accuracy': lr_accuracy,
                    'precision': precision_score(y_test, lr_pred),
                    'recall': recall_score(y_test, lr_pred),
                    'f1': f1_score(y_test, lr_pred)
                },
                'random_forest': {
                    'accuracy': rf_accuracy,
                    'precision': precision_score(y_test, rf_pred),
                    'recall': recall_score(y_test, rf_pred),
                    'f1': f1_score(y_test, rf_pred)
                },
                'xgboost': {
                    'accuracy': xgb_accuracy,
                    'precision': precision_score(y_test, xgb_pred),
                    'recall': recall_score(y_test, xgb_pred),
                    'f1': f1_score(y_test, xgb_pred)
                },
                'ensemble': {
                    'accuracy': ensemble_accuracy,
                    'precision': precision_score(y_test, ensemble_pred),
                    'recall': recall_score(y_test, ensemble_pred),
                    'f1': f1_score(y_test, ensemble_pred)
                }
            }
            
            self.models = models
            
            logger.info("Model training completed successfully")
            return {
                'models': models,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'performance': performance,
                'training_samples': len(X_train),
                'test_samples': len(X_test)
            }
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
            return {}
    
    def save_models(self, training_results: Dict[str, Any]) -> bool:
        """
        Save trained models to disk.
        
        Args:
            training_results: Dictionary containing trained models and metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Saving trained models to disk")
            
            models = training_results['models']
            scaler = training_results['scaler']
            feature_names = training_results['feature_names']
            
            # Save individual models
            for name, model in models.items():
                model_path = self.model_dir / f"{name}.joblib"
                joblib.dump(model, model_path)
                logger.info(f"Saved {name} model to {model_path}")
            
            # Save scaler
            scaler_path = self.model_dir / "feature_scaler.joblib"
            joblib.dump(scaler, scaler_path)
            logger.info(f"Saved scaler to {scaler_path}")
            
            # Save feature names
            feature_names_path = self.model_dir / "feature_names.json"
            import json
            with open(feature_names_path, 'w') as f:
                json.dump(feature_names, f)
            logger.info(f"Saved feature names to {feature_names_path}")
            
            # Save model metadata
            metadata = {
                'model_version': '2.0.0',
                'training_years': self.training_years,
                'feature_count': len(feature_names),
                'training_samples': training_results.get('training_samples', 0),
                'test_samples': training_results.get('test_samples', 0),
                'performance': training_results.get('performance', {}),
                'created_at': pd.Timestamp.now().isoformat()
            }
            
            metadata_path = self.model_dir / "model_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Saved model metadata to {metadata_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
            return False
    
    def run_full_training(self, years: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Run the complete training pipeline.
        
        Args:
            years: List of years to train on (default: 2008-2024)
            
        Returns:
            Dictionary containing training results
        """
        try:
            logger.info("Starting full training pipeline")
            
            # 1. Load training data
            training_data = self.load_training_data(years)
            if training_data['total_games'] == 0:
                raise ValueError("No training data loaded")
            
            # 2. Engineer features
            features_df = self.engineer_training_features(training_data)
            if len(features_df) == 0:
                raise ValueError("No features generated")
            
            # 3. Train models
            training_results = self.train_models(features_df)
            if not training_results:
                raise ValueError("Model training failed")
            
            # 4. Save models
            save_success = self.save_models(training_results)
            if not save_success:
                raise ValueError("Model saving failed")
            
            logger.info("Full training pipeline completed successfully")
            
            return {
                'success': True,
                'training_data': training_data,
                'features': features_df,
                'models': training_results,
                'saved': save_success
            }
            
        except Exception as e:
            logger.error(f"Error in full training pipeline: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _get_team_abbr(self, team_name: str) -> str:
        """Get team abbreviation from full team name."""
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
        return team_mapping.get(team_name, team_name)
    
    def _create_game_context(self, game: Dict) -> Any:
        """Create game context from game data."""
        # Create a simple game context object
        class GameContext:
            def __init__(self, game_data):
                self.home_team = game_data.get('homeTeam', '')
                self.away_team = game_data.get('awayTeam', '')
                self.season = game_data.get('season', 0)
                self.week = game_data.get('week', 0)
                self.game_type = 'REG'
                self.spread = None
                self.total = None
                self.temperature = None
                self.wind_speed = None
                self.precipitation = None
                self.roof = 'outdoors'
                self.surface = 'grass'
        
        return GameContext(game)


# Global instance
training_pipeline = TrainingDataPipeline()
