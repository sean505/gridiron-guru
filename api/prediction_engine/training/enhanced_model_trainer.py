"""
Enhanced Model Trainer
Trains models with enhanced features for improved prediction accuracy
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from pathlib import Path
import joblib
import json

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif, RFE
import xgboost as xgb

from ..data_models import GameContext, TeamStats
from ..enhanced_features import enhanced_extractor
from ..team_feature_database import team_feature_db
from ..nfl_data_cache import nfl_data_cache

logger = logging.getLogger(__name__)


class EnhancedModelTrainer:
    """
    Trains enhanced models with 45+ features for improved prediction accuracy.
    
    Features:
    - Feature selection and importance analysis
    - Hyperparameter tuning
    - Model ensemble creation
    - Cross-validation and performance metrics
    - Confidence calibration
    """
    
    def __init__(self, models_dir: str = "api/prediction_engine/models/trained"):
        """Initialize the enhanced model trainer."""
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Feature configuration
        self.original_feature_count = 28
        self.enhanced_feature_count = 45  # 28 original + 17 enhanced
        
        # Model components
        self.enhanced_models = {}
        self.feature_scaler = None
        self.feature_selector = None
        self.feature_importance = {}
        
        # Training data
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = []
        
        logger.info("EnhancedModelTrainer initialized")
    
    def prepare_training_data(self, start_season: int = 2008, end_season: int = 2023) -> bool:
        """
        Prepare training data with enhanced features.
        
        Args:
            start_season: Starting season for training data
            end_season: Ending season for training data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Preparing enhanced training data for seasons {start_season}-{end_season}")
            
            # Load historical game data
            games_data = self._load_historical_games(start_season, end_season)
            if games_data.empty:
                logger.error("No historical games data found")
                return False
            
            # Extract features for each game
            enhanced_features = []
            labels = []
            game_contexts = []
            
            logger.info(f"Processing {len(games_data)} historical games")
            
            for idx, game in games_data.iterrows():
                try:
                    # Extract enhanced features
                    home_features = self._extract_team_features(
                        game['home_team'], game['season'], game['week']
                    )
                    away_features = self._extract_team_features(
                        game['away_team'], game['season'], game['week']
                    )
                    
                    if home_features is None or away_features is None:
                        continue
                    
                    # Combine features (17 + 17 + 11 contextual = 45 features)
                    combined_features = self._combine_enhanced_features(
                        home_features, away_features, game
                    )
                    
                    if len(combined_features) != self.enhanced_feature_count:
                        logger.warning(f"Expected {self.enhanced_feature_count} features, got {len(combined_features)}")
                        continue
                    
                    enhanced_features.append(combined_features)
                    
                    # Create label (1 if home team won, 0 if away team won)
                    home_won = 1 if game['home_score'] > game['away_score'] else 0
                    labels.append(home_won)
                    
                    # Store game context
                    game_contexts.append({
                        'home_team': game['home_team'],
                        'away_team': game['away_team'],
                        'season': game['season'],
                        'week': game['week'],
                        'home_score': game['home_score'],
                        'away_score': game['away_score']
                    })
                    
                except Exception as e:
                    logger.warning(f"Error processing game {idx}: {e}")
                    continue
            
            if not enhanced_features:
                logger.error("No valid enhanced features extracted")
                return False
            
            # Convert to numpy arrays
            X = np.array(enhanced_features)
            y = np.array(labels)
            
            # Split data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Create feature names
            self.feature_names = self._create_feature_names()
            
            logger.info(f"Training data prepared: {len(self.X_train)} train, {len(self.X_test)} test samples")
            logger.info(f"Feature count: {X.shape[1]}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return False
    
    def train_enhanced_models(self) -> Dict[str, Any]:
        """
        Train enhanced models with feature selection and hyperparameter tuning.
        
        Returns:
            Dictionary with training results and model performance
        """
        try:
            if self.X_train is None:
                logger.error("Training data not prepared. Call prepare_training_data() first.")
                return {}
            
            logger.info("Starting enhanced model training")
            
            # Feature selection
            logger.info("Performing feature selection")
            self._perform_feature_selection()
            
            # Scale features
            logger.info("Scaling features")
            self.feature_scaler = StandardScaler()
            X_train_scaled = self.feature_scaler.fit_transform(self.X_train)
            X_test_scaled = self.feature_scaler.transform(self.X_test)
            
            # Train individual models
            models = {}
            model_performance = {}
            
            # 1. Logistic Regression
            logger.info("Training Logistic Regression")
            lr_model = self._train_logistic_regression(X_train_scaled, self.y_train)
            models['logistic_regression'] = lr_model
            model_performance['logistic_regression'] = self._evaluate_model(
                lr_model, X_test_scaled, self.y_test
            )
            
            # 2. Random Forest
            logger.info("Training Random Forest")
            rf_model = self._train_random_forest(X_train_scaled, self.y_train)
            models['random_forest'] = rf_model
            model_performance['random_forest'] = self._evaluate_model(
                rf_model, X_test_scaled, self.y_test
            )
            
            # 3. XGBoost
            logger.info("Training XGBoost")
            xgb_model = self._train_xgboost(X_train_scaled, self.y_train)
            models['xgboost'] = xgb_model
            model_performance['xgboost'] = self._evaluate_model(
                xgb_model, X_test_scaled, self.y_test
            )
            
            # 4. Create ensemble
            logger.info("Creating enhanced ensemble")
            ensemble_model = self._create_enhanced_ensemble(models)
            models['ensemble'] = ensemble_model
            model_performance['ensemble'] = self._evaluate_model(
                ensemble_model, X_test_scaled, self.y_test
            )
            
            # Store models
            self.enhanced_models = models
            
            # Save models and metadata
            self._save_enhanced_models()
            
            # Calculate feature importance
            self._calculate_feature_importance()
            
            logger.info("Enhanced model training completed successfully")
            
            return {
                'models_trained': list(models.keys()),
                'model_performance': model_performance,
                'feature_count': X_train_scaled.shape[1],
                'training_samples': len(self.X_train),
                'test_samples': len(self.X_test),
                'feature_importance': self.feature_importance
            }
            
        except Exception as e:
            logger.error(f"Error training enhanced models: {e}")
            return {}
    
    def _load_historical_games(self, start_season: int, end_season: int) -> pd.DataFrame:
        """Load historical game data for training."""
        try:
            # This would load from your existing historical data
            # For now, return empty DataFrame as placeholder
            logger.info(f"Loading historical games for seasons {start_season}-{end_season}")
            
            # Placeholder - would load from actual historical data
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error loading historical games: {e}")
            return pd.DataFrame()
    
    def _extract_team_features(self, team: str, season: int, week: int) -> Optional[List[float]]:
        """Extract enhanced features for a team."""
        try:
            # Try to get from database first
            features = team_feature_db.get_features(team, season, week)
            if features:
                return features
            
            # If not in database, extract live
            features = enhanced_extractor.extract_enhanced_features(team, 'DAL', season, week)
            return features
            
        except Exception as e:
            logger.warning(f"Error extracting features for {team}: {e}")
            return None
    
    def _combine_enhanced_features(self, home_features: List[float], away_features: List[float], 
                                 game: pd.Series) -> List[float]:
        """Combine enhanced features for both teams plus contextual factors."""
        try:
            combined = []
            
            # Add home team features (17)
            combined.extend(home_features)
            
            # Add away team features (17)
            combined.extend(away_features)
            
            # Add contextual factors (11)
            contextual_features = self._extract_contextual_features(game)
            combined.extend(contextual_features)
            
            return combined
            
        except Exception as e:
            logger.error(f"Error combining enhanced features: {e}")
            return [0.0] * self.enhanced_feature_count
    
    def _extract_contextual_features(self, game: pd.Series) -> List[float]:
        """Extract contextual features for the game."""
        try:
            # Contextual features (11 total)
            features = [
                # Game context (3)
                float(game['season'] - 2000) / 25.0,  # Season progression
                float(game['week']) / 18.0,  # Week progression
                1.0 if game['week'] > 17 else 0.0,  # Playoff indicator
                
                # Rest and travel (3) - placeholders
                0.0,  # Rest differential
                0.0,  # Travel distance
                0.1,  # Home advantage
                
                # Weather and environment (2) - placeholders
                0.0,  # Weather factor
                0.0,  # Stadium factor
                
                # Historical context (2) - placeholders
                0.0,  # Historical advantage
                0.0,  # Rivalry factor
                
                # Betting context (1) - placeholder
                0.0  # Spread
            ]
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting contextual features: {e}")
            return [0.0] * 11
    
    def _create_feature_names(self) -> List[str]:
        """Create feature names for the enhanced feature set."""
        try:
            feature_names = []
            
            # Home team features (17)
            for i, name in enumerate(enhanced_extractor.feature_names):
                feature_names.append(f"home_{name}")
            
            # Away team features (17)
            for i, name in enumerate(enhanced_extractor.feature_names):
                feature_names.append(f"away_{name}")
            
            # Contextual features (11)
            contextual_names = [
                'season_progression', 'week_progression', 'playoff_indicator',
                'rest_differential', 'travel_distance', 'home_advantage',
                'weather_factor', 'stadium_factor', 'historical_advantage',
                'rivalry_factor', 'spread'
            ]
            feature_names.extend(contextual_names)
            
            return feature_names
            
        except Exception as e:
            logger.error(f"Error creating feature names: {e}")
            return [f"feature_{i}" for i in range(self.enhanced_feature_count)]
    
    def _perform_feature_selection(self):
        """Perform feature selection to identify most important features."""
        try:
            # Use SelectKBest with f_classif
            k_best = SelectKBest(score_func=f_classif, k=30)  # Select top 30 features
            X_selected = k_best.fit_transform(self.X_train, self.y_train)
            
            # Store feature selector
            self.feature_selector = k_best
            
            # Update training data
            self.X_train = X_selected
            self.X_test = k_best.transform(self.X_test)
            
            # Get selected feature names
            selected_indices = k_best.get_support(indices=True)
            selected_features = [self.feature_names[i] for i in selected_indices]
            
            logger.info(f"Selected {len(selected_features)} most important features")
            logger.info(f"Selected features: {selected_features[:10]}...")  # Show first 10
            
        except Exception as e:
            logger.error(f"Error in feature selection: {e}")
    
    def _train_logistic_regression(self, X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
        """Train logistic regression model with hyperparameter tuning."""
        try:
            # Define parameter grid
            param_grid = {
                'C': [0.1, 1.0, 10.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga'],
                'class_weight': ['balanced', None]
            }
            
            # Grid search
            grid_search = GridSearchCV(
                LogisticRegression(random_state=42, max_iter=1000),
                param_grid,
                cv=5,
                scoring='accuracy',
                n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            
            logger.info(f"Best LR parameters: {grid_search.best_params_}")
            logger.info(f"Best LR score: {grid_search.best_score_:.4f}")
            
            return grid_search.best_estimator_
            
        except Exception as e:
            logger.error(f"Error training logistic regression: {e}")
            return LogisticRegression(random_state=42, max_iter=1000)
    
    def _train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
        """Train random forest model with hyperparameter tuning."""
        try:
            # Define parameter grid
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'class_weight': ['balanced', None]
            }
            
            # Grid search
            grid_search = GridSearchCV(
                RandomForestClassifier(random_state=42, n_jobs=-1),
                param_grid,
                cv=5,
                scoring='accuracy',
                n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            
            logger.info(f"Best RF parameters: {grid_search.best_params_}")
            logger.info(f"Best RF score: {grid_search.best_score_:.4f}")
            
            return grid_search.best_estimator_
            
        except Exception as e:
            logger.error(f"Error training random forest: {e}")
            return RandomForestClassifier(random_state=42, n_jobs=-1)
    
    def _train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray) -> xgb.XGBClassifier:
        """Train XGBoost model with hyperparameter tuning."""
        try:
            # Define parameter grid
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
            
            # Grid search
            grid_search = GridSearchCV(
                xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
                param_grid,
                cv=5,
                scoring='accuracy',
                n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            
            logger.info(f"Best XGB parameters: {grid_search.best_params_}")
            logger.info(f"Best XGB score: {grid_search.best_score_:.4f}")
            
            return grid_search.best_estimator_
            
        except Exception as e:
            logger.error(f"Error training XGBoost: {e}")
            return xgb.XGBClassifier(random_state=42, eval_metric='logloss')
    
    def _create_enhanced_ensemble(self, models: Dict[str, Any]) -> VotingClassifier:
        """Create enhanced ensemble model."""
        try:
            # Create voting classifier
            ensemble = VotingClassifier(
                estimators=[
                    ('lr', models['logistic_regression']),
                    ('rf', models['random_forest']),
                    ('xgb', models['xgboost'])
                ],
                voting='soft'  # Use predicted probabilities
            )
            
            # Fit ensemble
            ensemble.fit(self.X_train, self.y_train)
            
            logger.info("Enhanced ensemble created successfully")
            return ensemble
            
        except Exception as e:
            logger.error(f"Error creating enhanced ensemble: {e}")
            return models['logistic_regression']  # Fallback to single model
    
    def _evaluate_model(self, model: Any, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance."""
        try:
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_test, y_test, cv=5, scoring='accuracy')
            
            return {
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred.tolist(),
                'probabilities': y_pred_proba.tolist() if y_pred_proba is not None else None
            }
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return {'accuracy': 0.0, 'cv_mean': 0.0, 'cv_std': 0.0}
    
    def _calculate_feature_importance(self):
        """Calculate feature importance for all models."""
        try:
            self.feature_importance = {}
            
            for name, model in self.enhanced_models.items():
                if hasattr(model, 'feature_importances_'):
                    # Tree-based models
                    importances = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    # Linear models
                    importances = np.abs(model.coef_[0])
                else:
                    continue
                
                # Get feature names (if feature selection was used)
                if self.feature_selector is not None:
                    selected_indices = self.feature_selector.get_support(indices=True)
                    feature_names = [self.feature_names[i] for i in selected_indices]
                else:
                    feature_names = self.feature_names
                
                # Create importance dictionary
                importance_dict = dict(zip(feature_names, importances))
                self.feature_importance[name] = importance_dict
                
                # Log top features
                top_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:10]
                logger.info(f"Top 10 features for {name}: {top_features}")
            
        except Exception as e:
            logger.error(f"Error calculating feature importance: {e}")
    
    def _save_enhanced_models(self):
        """Save enhanced models and metadata."""
        try:
            # Save individual models
            for name, model in self.enhanced_models.items():
                model_path = self.models_dir / f"enhanced_{name}.joblib"
                joblib.dump(model, model_path)
                logger.info(f"Saved enhanced {name} model to {model_path}")
            
            # Save feature scaler
            if self.feature_scaler is not None:
                scaler_path = self.models_dir / "enhanced_feature_scaler.joblib"
                joblib.dump(self.feature_scaler, scaler_path)
                logger.info(f"Saved enhanced feature scaler to {scaler_path}")
            
            # Save feature selector
            if self.feature_selector is not None:
                selector_path = self.models_dir / "enhanced_feature_selector.joblib"
                joblib.dump(self.feature_selector, selector_path)
                logger.info(f"Saved enhanced feature selector to {selector_path}")
            
            # Save metadata
            metadata = {
                'training_timestamp': datetime.now().isoformat(),
                'feature_count': self.enhanced_feature_count,
                'feature_names': self.feature_names,
                'model_performance': self._get_model_performance_summary(),
                'feature_importance': self.feature_importance
            }
            
            metadata_path = self.models_dir / "enhanced_model_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"Saved enhanced model metadata to {metadata_path}")
            
        except Exception as e:
            logger.error(f"Error saving enhanced models: {e}")
    
    def _get_model_performance_summary(self) -> Dict[str, Any]:
        """Get summary of model performance."""
        try:
            summary = {}
            
            for name, model in self.enhanced_models.items():
                if hasattr(model, 'score'):
                    train_score = model.score(self.X_train, self.y_train)
                    test_score = model.score(self.X_test, self.y_test)
                    
                    summary[name] = {
                        'train_accuracy': train_score,
                        'test_accuracy': test_score,
                        'overfitting': train_score - test_score
                    }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting model performance summary: {e}")
            return {}


# Global enhanced model trainer instance
enhanced_model_trainer = EnhancedModelTrainer()
