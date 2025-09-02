"""
Main prediction engine for the Gridiron Guru system.

This module implements a comprehensive ensemble prediction system with multiple ML models,
confidence scoring, upset detection, and model training pipelines.
"""

import logging
import numpy as np
import pandas as pd
import joblib
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from .data_models import TeamStats, GameContext, GamePrediction, WeeklyPredictions, PredictionRequest, PredictionResponse
from .data_collector import data_collector
from .feature_engineering import feature_engineer

logger = logging.getLogger(__name__)


class PredictionEngine:
    """
    Main prediction engine with ensemble modeling and advanced analytics.
    
    Features:
    - Multiple ML models (Logistic Regression, Random Forest, XGBoost)
    - Ensemble prediction with confidence scoring
    - Upset detection and analysis
    - Model training and validation pipeline
    - Historical performance tracking
    """
    
    def __init__(self, model_dir: str = "api/prediction_engine/models"):
        """Initialize the prediction engine."""
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        # Models
        self.logistic_model = None
        self.random_forest_model = None
        self.xgboost_model = None
        self.ensemble_weights = None
        
        # Scalers and preprocessing
        self.scaler = StandardScaler()
        self.feature_names = []
        
        # Performance tracking
        self.model_performance = {}
        self.upset_threshold = 0.3  # Confidence threshold for upset alerts
        
        # Model metadata
        self.model_version = "1.0.0"
        self.last_trained = None
        self.training_data_size = 0
        
        logger.info("PredictionEngine initialized")
    
    def train_models(self, train_years: List[int] = None, val_year: int = 2024) -> Dict[str, Any]:
        """
        Train all prediction models on historical data.
        
        Args:
            train_years: Years to use for training (default: 2008-2023)
            val_year: Year to use for validation
            
        Returns:
            Training results and performance metrics
        """
        try:
            if train_years is None:
                train_years = list(range(2008, 2024))
            
            logger.info(f"Training models on years {train_years}, validating on {val_year}")
            
            # Prepare training data
            X_train, y_train, X_val, y_val = self._prepare_training_data(train_years, val_year)
            
            if X_train is None or len(X_train) == 0:
                raise ValueError("No training data available")
            
            # Train individual models
            training_results = {}
            
            # 1. Logistic Regression
            logger.info("Training Logistic Regression model...")
            self.logistic_model = LogisticRegression(
                random_state=42, 
                max_iter=1000,
                class_weight='balanced'
            )
            self.logistic_model.fit(X_train, y_train)
            lr_performance = self._evaluate_model(self.logistic_model, X_val, y_val, "Logistic Regression")
            training_results['logistic_regression'] = lr_performance
            
            # 2. Random Forest
            logger.info("Training Random Forest model...")
            self.random_forest_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            )
            self.random_forest_model.fit(X_train, y_train)
            rf_performance = self._evaluate_model(self.random_forest_model, X_val, y_val, "Random Forest")
            training_results['random_forest'] = rf_performance
            
            # 3. XGBoost
            logger.info("Training XGBoost model...")
            self.xgboost_model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss'
            )
            self.xgboost_model.fit(X_train, y_train)
            xgb_performance = self._evaluate_model(self.xgboost_model, X_val, y_val, "XGBoost")
            training_results['xgboost'] = xgb_performance
            
            # 4. Calculate ensemble weights based on validation performance
            self._calculate_ensemble_weights(training_results)
            
            # 5. Cross-validation for robust performance estimates
            cv_results = self._cross_validate_models(X_train, y_train)
            training_results['cross_validation'] = cv_results
            
            # 6. Save models
            self._save_models()
            
            # Update metadata
            self.last_trained = datetime.now()
            self.training_data_size = len(X_train)
            self.model_performance = training_results
            
            logger.info("Model training completed successfully")
            return training_results
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
            raise
    
    def _prepare_training_data(self, train_years: List[int], val_year: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare training and validation data from historical sources."""
        try:
            all_features = []
            all_labels = []
            all_metadata = []
            
            # Get historical data
            game_log = data_collector.get_game_log_data()
            season_data = data_collector.get_season_data()
            
            logger.info(f"Processing {len(game_log)} historical games for training")
            
            # Process each year
            for year in train_years + [val_year]:
                year_games = [key for key in game_log.keys() if key.startswith(f"{year}_")]
                
                for game_key in year_games:
                    try:
                        game_data = game_log[game_key]
                        
                        # Extract game information
                        parts = game_key.split('_')
                        if len(parts) < 4:
                            continue
                        
                        season = int(parts[0])
                        week = int(parts[1])
                        home_team = parts[2]
                        away_team = parts[3]
                        
                        # Skip if we don't have enough data
                        if week < 3:  # Need some season context
                            continue
                        
                        # Create game context
                        game_context = GameContext(
                            game_id=game_key,
                            season=season,
                            week=week,
                            home_team=home_team,
                            away_team=away_team,
                            game_date=datetime.now()  # Placeholder
                        )
                        
                        # Create features
                        features = feature_engineer.create_game_features(
                            home_team, away_team, season, week, game_context
                        )
                        
                        if not features:
                            continue
                        
                        # Get actual outcome
                        home_won = game_data.get('home_won', False)
                        label = 1 if home_won else 0
                        
                        # Store data
                        all_features.append(features)
                        all_labels.append(label)
                        all_metadata.append({
                            'season': season,
                            'week': week,
                            'home_team': home_team,
                            'away_team': away_team,
                            'game_key': game_key
                        })
                        
                    except Exception as e:
                        logger.warning(f"Error processing game {game_key}: {e}")
                        continue
            
            if not all_features:
                logger.error("No training data could be prepared")
                return None, None, None, None
            
            # Convert to arrays
            feature_df = pd.DataFrame(all_features)
            self.feature_names = list(feature_df.columns)
            
            # Fill missing values
            feature_df = feature_df.fillna(0.0)
            
            # Split into train/validation
            train_mask = [meta['season'] in train_years for meta in all_metadata]
            val_mask = [meta['season'] == val_year for meta in all_metadata]
            
            X_train = feature_df[train_mask].values
            y_train = np.array(all_labels)[train_mask]
            X_val = feature_df[val_mask].values
            y_val = np.array(all_labels)[val_mask]
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            
            logger.info(f"Prepared training data: {len(X_train)} samples, {len(self.feature_names)} features")
            logger.info(f"Prepared validation data: {len(X_val)} samples")
            
            return X_train_scaled, y_train, X_val_scaled, y_val
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return None, None, None, None
    
    def _evaluate_model(self, model, X_val: np.ndarray, y_val: np.ndarray, model_name: str) -> Dict[str, float]:
        """Evaluate a single model and return performance metrics."""
        try:
            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else y_pred
            
            metrics = {
                'accuracy': accuracy_score(y_val, y_pred),
                'precision': precision_score(y_val, y_pred, average='weighted'),
                'recall': recall_score(y_val, y_pred, average='weighted'),
                'f1_score': f1_score(y_val, y_pred, average='weighted'),
                'roc_auc': roc_auc_score(y_val, y_pred_proba) if len(np.unique(y_val)) > 1 else 0.0
            }
            
            logger.info(f"{model_name} performance: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {e}")
            return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'roc_auc': 0.0}
    
    def _calculate_ensemble_weights(self, performance_results: Dict[str, Dict[str, float]]) -> None:
        """Calculate ensemble weights based on model performance."""
        try:
            # Use ROC AUC as the primary metric for weighting
            weights = {}
            total_score = 0
            
            for model_name, metrics in performance_results.items():
                if 'roc_auc' in metrics:
                    score = max(0.5, metrics['roc_auc'])  # Minimum weight of 0.5
                    weights[model_name] = score
                    total_score += score
            
            # Normalize weights
            if total_score > 0:
                self.ensemble_weights = {
                    'logistic_regression': weights.get('logistic_regression', 0.33) / total_score,
                    'random_forest': weights.get('random_forest', 0.33) / total_score,
                    'xgboost': weights.get('xgboost', 0.33) / total_score
                }
            else:
                # Equal weights if no performance data
                self.ensemble_weights = {
                    'logistic_regression': 0.33,
                    'random_forest': 0.33,
                    'xgboost': 0.33
                }
            
            logger.info(f"Ensemble weights: {self.ensemble_weights}")
            
        except Exception as e:
            logger.error(f"Error calculating ensemble weights: {e}")
            self.ensemble_weights = {'logistic_regression': 0.33, 'random_forest': 0.33, 'xgboost': 0.33}
    
    def _cross_validate_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """Perform cross-validation for robust performance estimates."""
        try:
            cv_results = {}
            
            # Time series split for temporal data
            tscv = TimeSeriesSplit(n_splits=5)
            
            models = {
                'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
                'random_forest': RandomForestClassifier(n_estimators=50, random_state=42),
                'xgboost': xgb.XGBClassifier(n_estimators=50, random_state=42, eval_metric='logloss')
            }
            
            for model_name, model in models.items():
                try:
                    scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring='roc_auc')
                    cv_results[model_name] = {
                        'mean_score': np.mean(scores),
                        'std_score': np.std(scores),
                        'scores': scores.tolist()
                    }
                except Exception as e:
                    logger.warning(f"CV failed for {model_name}: {e}")
                    cv_results[model_name] = {'mean_score': 0.5, 'std_score': 0.0, 'scores': []}
            
            return cv_results
            
        except Exception as e:
            logger.error(f"Error in cross-validation: {e}")
            return {}
    
    def predict_game(self, home_team: str, away_team: str, season: int, week: int, 
                    game_context: GameContext) -> GamePrediction:
        """
        Make a prediction for a specific game.
        
        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            season: NFL season year
            week: Week number
            game_context: Game context information
            
        Returns:
            Comprehensive game prediction
        """
        try:
            if not self._models_loaded():
                logger.warning("Models not loaded, using fallback prediction")
                return self._fallback_prediction(home_team, away_team, season, week, game_context)
            
            # Create features
            features = feature_engineer.create_game_features(
                home_team, away_team, season, week, game_context
            )
            
            if not features:
                return self._fallback_prediction(home_team, away_team, season, week, game_context)
            
            # Prepare features for ML
            X = feature_engineer.prepare_features_for_ml(features)
            X_scaled = self.scaler.transform(X)
            
            # Get individual model predictions
            predictions = {}
            probabilities = {}
            
            # Logistic Regression
            if self.logistic_model:
                lr_pred = self.logistic_model.predict(X_scaled)[0]
                lr_proba = self.logistic_model.predict_proba(X_scaled)[0]
                predictions['logistic_regression'] = lr_pred
                probabilities['logistic_regression'] = lr_proba[1]  # Home team win probability
            
            # Random Forest
            if self.random_forest_model:
                rf_pred = self.random_forest_model.predict(X_scaled)[0]
                rf_proba = self.random_forest_model.predict_proba(X_scaled)[0]
                predictions['random_forest'] = rf_pred
                probabilities['random_forest'] = rf_proba[1]
            
            # XGBoost
            if self.xgboost_model:
                xgb_pred = self.xgboost_model.predict(X_scaled)[0]
                xgb_proba = self.xgboost_model.predict_proba(X_scaled)[0]
                predictions['xgboost'] = xgb_pred
                probabilities['xgboost'] = xgb_proba[1]
            
            # Ensemble prediction
            ensemble_prob = self._ensemble_prediction(probabilities)
            predicted_winner = home_team if ensemble_prob > 0.5 else away_team
            
            # Calculate confidence
            confidence = self._calculate_confidence(predictions, probabilities, features, game_context)
            
            # Upset detection
            is_upset, upset_probability, upset_factors = self._detect_upset(
                predicted_winner, confidence, game_context, features
            )
            
            # Generate explanation
            explanation = self._generate_explanation(
                home_team, away_team, predicted_winner, confidence, features, game_context
            )
            
            # Create prediction object
            prediction = GamePrediction(
                game_id=game_context.game_id,
                home_team=home_team,
                away_team=away_team,
                predicted_winner=predicted_winner,
                confidence=confidence,
                win_probability=ensemble_prob if predicted_winner == home_team else 1 - ensemble_prob,
                predicted_home_score=self._predict_score(home_team, season, week, features, 'home'),
                predicted_away_score=self._predict_score(away_team, season, week, features, 'away'),
                predicted_total=self._predict_total(features),
                predicted_spread=self._predict_spread(features),
                is_upset_pick=is_upset,
                upset_probability=upset_probability,
                upset_factors=upset_factors,
                key_factors=self._extract_key_factors(features),
                explanation=explanation,
                model_version=self.model_version,
                data_freshness=datetime.now()
            )
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return self._fallback_prediction(home_team, away_team, season, week, game_context)
    
    def _models_loaded(self) -> bool:
        """Check if all models are loaded."""
        return (self.logistic_model is not None and 
                self.random_forest_model is not None and 
                self.xgboost_model is not None)
    
    def _ensemble_prediction(self, probabilities: Dict[str, float]) -> float:
        """Calculate ensemble prediction from individual model probabilities."""
        try:
            if not self.ensemble_weights:
                # Equal weights if no weights calculated
                return np.mean(list(probabilities.values()))
            
            ensemble_prob = 0.0
            for model_name, prob in probabilities.items():
                weight = self.ensemble_weights.get(model_name, 0.33)
                ensemble_prob += weight * prob
            
            return ensemble_prob
            
        except Exception as e:
            logger.error(f"Error in ensemble prediction: {e}")
            return 0.5
    
    def _calculate_confidence(self, predictions: Dict[str, int], probabilities: Dict[str, float], 
                            features: Dict[str, float], game_context: GameContext) -> float:
        """Calculate prediction confidence based on multiple factors."""
        try:
            confidence_factors = []
            
            # 1. Model agreement
            if len(predictions) > 1:
                agreement = len(set(predictions.values())) == 1
                confidence_factors.append(0.8 if agreement else 0.4)
            else:
                confidence_factors.append(0.5)
            
            # 2. Probability spread
            if len(probabilities) > 1:
                prob_std = np.std(list(probabilities.values()))
                confidence_factors.append(max(0.1, 1.0 - prob_std))
            else:
                confidence_factors.append(0.5)
            
            # 3. Feature quality
            feature_quality = self._assess_feature_quality(features)
            confidence_factors.append(feature_quality)
            
            # 4. Historical accuracy in similar situations
            historical_confidence = self._assess_historical_accuracy(features, game_context)
            confidence_factors.append(historical_confidence)
            
            # Weighted average
            weights = [0.3, 0.2, 0.3, 0.2]  # Model agreement, probability spread, feature quality, historical
            confidence = sum(w * f for w, f in zip(weights, confidence_factors))
            
            return min(1.0, max(0.0, confidence))
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5
    
    def _detect_upset(self, predicted_winner: str, confidence: float, 
                     game_context: GameContext, features: Dict[str, float]) -> Tuple[bool, float, List[str]]:
        """Detect if this is a potential upset pick."""
        try:
            upset_factors = []
            upset_score = 0.0
            
            # 1. Betting line deviation
            if game_context.spread is not None:
                if predicted_winner == game_context.home_team and game_context.spread > 0:
                    # Predicted home team to win but they're underdogs
                    upset_score += 0.3
                    upset_factors.append("Home underdog predicted to win")
                elif predicted_winner == game_context.away_team and game_context.spread < 0:
                    # Predicted away team to win but they're underdogs
                    upset_score += 0.3
                    upset_factors.append("Away underdog predicted to win")
            
            # 2. High confidence in underdog
            if confidence > 0.7 and upset_score > 0:
                upset_score += 0.2
                upset_factors.append("High confidence in underdog")
            
            # 3. Historical upset patterns
            historical_upset_score = self._assess_upset_patterns(features, game_context)
            upset_score += historical_upset_score
            
            # 4. Weather/environmental factors
            if features.get('cold_weather', 0) > 0 or features.get('high_wind', 0) > 0:
                upset_score += 0.1
                upset_factors.append("Adverse weather conditions")
            
            # 5. Recent form discrepancy
            if abs(features.get('recent_form_diff', 0)) > 0.3:
                upset_score += 0.1
                upset_factors.append("Significant recent form difference")
            
            is_upset = upset_score > self.upset_threshold
            upset_probability = min(1.0, upset_score)
            
            return is_upset, upset_probability, upset_factors
            
        except Exception as e:
            logger.error(f"Error detecting upset: {e}")
            return False, 0.0, []
    
    def _assess_feature_quality(self, features: Dict[str, float]) -> float:
        """Assess the quality of available features."""
        try:
            # Check for missing or default values
            quality_score = 1.0
            
            # Penalize for too many zero/default values
            zero_count = sum(1 for v in features.values() if v == 0.0)
            total_features = len(features)
            
            if total_features > 0:
                zero_ratio = zero_count / total_features
                quality_score -= zero_ratio * 0.5
            
            return max(0.1, quality_score)
            
        except Exception as e:
            logger.error(f"Error assessing feature quality: {e}")
            return 0.5
    
    def _assess_historical_accuracy(self, features: Dict[str, float], game_context: GameContext) -> float:
        """Assess historical accuracy in similar situations."""
        try:
            # This would typically look at historical performance in similar game contexts
            # For now, return a baseline confidence
            return 0.6
            
        except Exception as e:
            logger.error(f"Error assessing historical accuracy: {e}")
            return 0.5
    
    def _assess_upset_patterns(self, features: Dict[str, float], game_context: GameContext) -> float:
        """Assess historical upset patterns."""
        try:
            # This would analyze historical upsets with similar characteristics
            # For now, return a baseline score
            return 0.1
            
        except Exception as e:
            logger.error(f"Error assessing upset patterns: {e}")
            return 0.0
    
    def _generate_explanation(self, home_team: str, away_team: str, predicted_winner: str, 
                            confidence: float, features: Dict[str, float], game_context: GameContext) -> str:
        """Generate human-readable explanation for the prediction."""
        try:
            explanations = []
            
            # Key factors
            if features.get('win_pct_diff', 0) > 0.1:
                explanations.append(f"{predicted_winner} has a significantly better win percentage")
            elif features.get('win_pct_diff', 0) < -0.1:
                explanations.append(f"{predicted_winner} has a significantly better win percentage")
            
            if features.get('off_epa_diff', 0) > 0.5:
                explanations.append(f"{predicted_winner} has superior offensive efficiency")
            
            if features.get('def_epa_diff', 0) > 0.5:
                explanations.append(f"{predicted_winner} has superior defensive efficiency")
            
            if features.get('home_field_advantage', 0) > 0 and predicted_winner == home_team:
                explanations.append("Home field advantage favors the home team")
            
            if features.get('recent_form_diff', 0) > 0.2:
                explanations.append(f"{predicted_winner} has better recent form")
            
            # Confidence level
            if confidence > 0.8:
                explanations.append("High confidence prediction based on strong indicators")
            elif confidence > 0.6:
                explanations.append("Moderate confidence prediction")
            else:
                explanations.append("Low confidence prediction - game could go either way")
            
            if not explanations:
                explanations.append(f"Prediction based on overall team performance and situational factors")
            
            return ". ".join(explanations) + "."
            
        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            return f"Prediction based on team performance analysis."
    
    def _extract_key_factors(self, features: Dict[str, float]) -> List[str]:
        """Extract key factors influencing the prediction."""
        try:
            key_factors = []
            
            # Sort features by absolute value to find most important
            sorted_features = sorted(features.items(), key=lambda x: abs(x[1]), reverse=True)
            
            factor_mapping = {
                'win_pct_diff': 'Win percentage differential',
                'off_epa_diff': 'Offensive efficiency differential',
                'def_epa_diff': 'Defensive efficiency differential',
                'recent_form_diff': 'Recent form differential',
                'home_field_advantage': 'Home field advantage',
                'turnover_margin_diff': 'Turnover margin differential',
                'red_zone_eff_diff': 'Red zone efficiency differential',
                'third_down_diff': 'Third down conversion differential'
            }
            
            for feature_name, value in sorted_features[:5]:  # Top 5 factors
                if abs(value) > 0.1 and feature_name in factor_mapping:
                    key_factors.append(factor_mapping[feature_name])
            
            return key_factors[:3]  # Return top 3 factors
            
        except Exception as e:
            logger.error(f"Error extracting key factors: {e}")
            return ["Team performance analysis"]
    
    def _predict_score(self, team: str, season: int, week: int, features: Dict[str, float], side: str) -> float:
        """Predict team score (simplified model)."""
        try:
            # Base score prediction (would be more sophisticated in practice)
            base_score = 24.0
            
            # Adjust based on offensive features
            if side == 'home':
                off_epa = features.get('off_epa_diff', 0)
                def_epa = features.get('def_epa_diff', 0)
            else:
                off_epa = -features.get('off_epa_diff', 0)
                def_epa = -features.get('def_epa_diff', 0)
            
            # Adjust score based on EPA
            score_adjustment = (off_epa - def_epa) * 2
            predicted_score = base_score + score_adjustment
            
            return max(0, min(50, predicted_score))  # Clamp between 0 and 50
            
        except Exception as e:
            logger.error(f"Error predicting score: {e}")
            return 24.0
    
    def _predict_total(self, features: Dict[str, float]) -> float:
        """Predict total points for the game."""
        try:
            base_total = 45.0
            
            # Adjust based on offensive features
            off_total = features.get('off_epa_diff', 0) * 4
            total_adjustment = off_total
            
            predicted_total = base_total + total_adjustment
            return max(20, min(70, predicted_total))  # Clamp between 20 and 70
            
        except Exception as e:
            logger.error(f"Error predicting total: {e}")
            return 45.0
    
    def _predict_spread(self, features: Dict[str, float]) -> float:
        """Predict point spread."""
        try:
            # Spread is roughly the difference in predicted scores
            spread = features.get('win_pct_diff', 0) * 10 + features.get('off_epa_diff', 0) * 3
            return round(spread, 1)
            
        except Exception as e:
            logger.error(f"Error predicting spread: {e}")
            return 0.0
    
    def _fallback_prediction(self, home_team: str, away_team: str, season: int, week: int, 
                           game_context: GameContext) -> GamePrediction:
        """Create a fallback prediction when models are not available."""
        try:
            # Simple fallback based on home field advantage
            predicted_winner = home_team
            confidence = 0.5
            
            return GamePrediction(
                game_id=game_context.game_id,
                home_team=home_team,
                away_team=away_team,
                predicted_winner=predicted_winner,
                confidence=confidence,
                win_probability=0.55,  # Slight home field advantage
                predicted_home_score=24.0,
                predicted_away_score=21.0,
                predicted_total=45.0,
                predicted_spread=3.0,
                is_upset_pick=False,
                upset_probability=0.0,
                upset_factors=[],
                key_factors=["Home field advantage"],
                explanation="Fallback prediction based on home field advantage.",
                model_version="fallback",
                data_freshness=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error creating fallback prediction: {e}")
            # Return minimal prediction
            return GamePrediction(
                game_id=game_context.game_id,
                home_team=home_team,
                away_team=away_team,
                predicted_winner=home_team,
                confidence=0.5,
                win_probability=0.5,
                predicted_home_score=24.0,
                predicted_away_score=24.0,
                predicted_total=48.0,
                predicted_spread=0.0,
                is_upset_pick=False,
                upset_probability=0.0,
                upset_factors=[],
                key_factors=["Default prediction"],
                explanation="Default prediction due to system limitations.",
                model_version="default",
                data_freshness=datetime.now()
            )
    
    def _save_models(self) -> None:
        """Save trained models to disk."""
        try:
            if self.logistic_model:
                joblib.dump(self.logistic_model, self.model_dir / "logistic_model.pkl")
            if self.random_forest_model:
                joblib.dump(self.random_forest_model, self.model_dir / "random_forest_model.pkl")
            if self.xgboost_model:
                joblib.dump(self.xgboost_model, self.model_dir / "xgboost_model.pkl")
            
            # Save scaler and metadata
            joblib.dump(self.scaler, self.model_dir / "scaler.pkl")
            joblib.dump(self.ensemble_weights, self.model_dir / "ensemble_weights.pkl")
            joblib.dump(self.feature_names, self.model_dir / "feature_names.pkl")
            
            logger.info("Models saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def load_models(self) -> bool:
        """Load trained models from disk."""
        try:
            model_files = {
                'logistic_model': 'logistic_model.pkl',
                'random_forest_model': 'random_forest_model.pkl',
                'xgboost_model': 'xgboost_model.pkl',
                'scaler': 'scaler.pkl',
                'ensemble_weights': 'ensemble_weights.pkl',
                'feature_names': 'feature_names.pkl'
            }
            
            for attr_name, filename in model_files.items():
                file_path = self.model_dir / filename
                if file_path.exists():
                    setattr(self, attr_name, joblib.load(file_path))
                else:
                    logger.warning(f"Model file not found: {filename}")
                    return False
            
            logger.info("Models loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Get current model performance metrics."""
        return self.model_performance.copy()
    
    def predict_weekly_games(self, season: int, week: int) -> WeeklyPredictions:
        """Predict all games for a specific week."""
        try:
            # Get games for the week
            nfl_schedules = data_collector.get_nfl_data("schedules", seasons=[season])
            
            if nfl_schedules.empty:
                logger.warning(f"No games found for {season} week {week}")
                return WeeklyPredictions(
                    season=season,
                    week=week,
                    games=[],
                    total_games=0
                )
            
            # Filter for the specific week
            week_games = nfl_schedules[nfl_schedules['week'] == week]
            
            predictions = []
            for _, game in week_games.iterrows():
                try:
                    game_context = data_collector.get_game_context(
                        str(game.get('game_id', '')),
                        season,
                        week
                    )
                    
                    prediction = self.predict_game(
                        game_context.home_team,
                        game_context.away_team,
                        season,
                        week,
                        game_context
                    )
                    
                    predictions.append(prediction)
                    
                except Exception as e:
                    logger.warning(f"Error predicting game {game.get('game_id', '')}: {e}")
                    continue
            
            # Calculate summary statistics
            upset_picks = sum(1 for p in predictions if p.is_upset_pick)
            avg_confidence = np.mean([p.confidence for p in predictions]) if predictions else 0.0
            
            return WeeklyPredictions(
                season=season,
                week=week,
                games=predictions,
                total_games=len(predictions),
                upset_picks=upset_picks,
                average_confidence=avg_confidence
            )
            
        except Exception as e:
            logger.error(f"Error predicting weekly games: {e}")
            return WeeklyPredictions(
                season=season,
                week=week,
                games=[],
                total_games=0
            )


# Global prediction engine instance
prediction_engine = PredictionEngine()
