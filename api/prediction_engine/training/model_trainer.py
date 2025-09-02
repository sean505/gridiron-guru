"""
Ensemble model trainer for NFL prediction engine.

This module trains individual models and creates an ensemble
that combines their predictions for improved accuracy.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
import joblib
from pathlib import Path
from datetime import datetime
import json

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

from ..models import (
    MODEL_BASE_PATH, 
    LOGISTIC_REGRESSION_MODEL,
    RANDOM_FOREST_MODEL,
    XGBOOST_MODEL,
    ENSEMBLE_MODEL,
    FEATURE_SCALER,
    MODEL_METADATA
)

logger = logging.getLogger(__name__)

class EnsembleModelTrainer:
    """Trains and manages the ensemble prediction models"""
    
    def __init__(self):
        self.models = {
            'logistic_regression': LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight='balanced'
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss'
            )
        }
        
        self.ensemble_model = None
        self.feature_scaler = StandardScaler()
        self.feature_importance = {}
        self.training_metrics = {}
        self.model_metadata = {}
        
        logger.info("Initialized EnsembleModelTrainer")
    
    def train_individual_models(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_val: pd.DataFrame, 
        y_val: pd.Series
    ) -> Dict[str, Dict[str, float]]:
        """
        Train each model in the ensemble individually.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Dictionary of training metrics for each model
        """
        logger.info("Training individual models...")
        
        # Scale features
        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        X_val_scaled = self.feature_scaler.transform(X_val)
        
        training_results = {}
        
        for model_name, model in self.models.items():
            logger.info(f"Training {model_name}...")
            
            try:
                # Train model
                if model_name == 'xgboost':
                    # XGBoost needs special handling
                    model.fit(
                        X_train_scaled, y_train,
                        eval_set=[(X_val_scaled, y_val)],
                        verbose=False
                    )
                else:
                    model.fit(X_train_scaled, y_train)
                
                # Make predictions
                train_pred = model.predict(X_train_scaled)
                val_pred = model.predict(X_val_scaled)
                
                # Calculate metrics
                train_acc = accuracy_score(y_train, train_pred)
                val_acc = accuracy_score(y_val, val_pred)
                
                # Store feature importance if available
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[model_name] = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    # For logistic regression, use absolute coefficients
                    self.feature_importance[model_name] = np.abs(model.coef_[0])
                
                # Store results
                training_results[model_name] = {
                    'train_accuracy': train_acc,
                    'val_accuracy': val_acc,
                    'train_samples': len(X_train),
                    'val_samples': len(X_val)
                }
                
                logger.info(f"{model_name} - Train Acc: {train_acc:.3f}, Val Acc: {val_acc:.3f}")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                training_results[model_name] = {
                    'train_accuracy': 0.0,
                    'val_accuracy': 0.0,
                    'error': str(e)
                }
        
        self.training_metrics = training_results
        return training_results
    
    def create_ensemble(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> Dict[str, float]:
        """
        Create ensemble model that combines individual model predictions.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Ensemble performance metrics
        """
        logger.info("Creating ensemble model...")
        
        # Scale features
        X_train_scaled = self.feature_scaler.transform(X_train)
        X_val_scaled = self.feature_scaler.transform(X_val)
        
        # Create voting classifier
        self.ensemble_model = VotingClassifier(
            estimators=[
                ('lr', self.models['logistic_regression']),
                ('rf', self.models['random_forest']),
                ('xgb', self.models['xgboost'])
            ],
            voting='soft'  # Use predicted probabilities
        )
        
        # Train ensemble
        self.ensemble_model.fit(X_train_scaled, y_train)
        
        # Evaluate ensemble
        train_pred = self.ensemble_model.predict(X_train_scaled)
        val_pred = self.ensemble_model.predict(X_val_scaled)
        
        train_acc = accuracy_score(y_train, train_pred)
        val_acc = accuracy_score(y_val, val_pred)
        
        ensemble_results = {
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'train_samples': len(X_train),
            'val_samples': len(X_val)
        }
        
        logger.info(f"Ensemble - Train Acc: {train_acc:.3f}, Val Acc: {val_acc:.3f}")
        
        return ensemble_results
    
    def optimize_ensemble_weights(
        self, 
        X_val: pd.DataFrame, 
        y_val: pd.Series
    ) -> Dict[str, float]:
        """
        Optimize ensemble weights based on validation performance.
        
        Args:
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Optimized weights for each model
        """
        logger.info("Optimizing ensemble weights...")
        
        X_val_scaled = self.feature_scaler.transform(X_val)
        
        # Get individual model predictions
        model_predictions = {}
        for model_name, model in self.models.items():
            if model_name == 'xgboost':
                # XGBoost returns probabilities
                pred_proba = model.predict_proba(X_val_scaled)[:, 1]
            else:
                pred_proba = model.predict_proba(X_val_scaled)[:, 1]
            model_predictions[model_name] = pred_proba
        
        # Simple grid search for optimal weights
        best_accuracy = 0.0
        best_weights = {'logistic_regression': 0.33, 'random_forest': 0.33, 'xgboost': 0.34}
        
        # Test different weight combinations
        weight_combinations = [
            [0.4, 0.3, 0.3],  # LR heavy
            [0.3, 0.4, 0.3],  # RF heavy
            [0.3, 0.3, 0.4],  # XGB heavy
            [0.5, 0.25, 0.25], # LR very heavy
            [0.25, 0.5, 0.25], # RF very heavy
            [0.25, 0.25, 0.5], # XGB very heavy
        ]
        
        for weights in weight_combinations:
            # Calculate weighted ensemble prediction
            ensemble_pred = (
                weights[0] * model_predictions['logistic_regression'] +
                weights[1] * model_predictions['random_forest'] +
                weights[2] * model_predictions['xgboost']
            )
            
            # Convert to binary predictions
            ensemble_binary = (ensemble_pred > 0.5).astype(int)
            
            # Calculate accuracy
            accuracy = accuracy_score(y_val, ensemble_binary)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_weights = {
                    'logistic_regression': weights[0],
                    'random_forest': weights[1],
                    'xgboost': weights[2]
                }
        
        logger.info(f"Best ensemble weights: {best_weights}")
        logger.info(f"Best ensemble accuracy: {best_accuracy:.3f}")
        
        return best_weights
    
    def save_trained_models(self, model_path: Optional[str] = None) -> Dict[str, str]:
        """
        Save all trained models and metadata.
        
        Args:
            model_path: Path to save models (defaults to configured path)
            
        Returns:
            Dictionary of saved model file paths
        """
        if model_path is None:
            model_path = MODEL_BASE_PATH
        else:
            model_path = Path(model_path)
        
        model_path.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        try:
            # Save individual models
            for model_name, model in self.models.items():
                filename = {
                    'logistic_regression': LOGISTIC_REGRESSION_MODEL,
                    'random_forest': RANDOM_FOREST_MODEL,
                    'xgboost': XGBOOST_MODEL
                }[model_name]
                
                filepath = model_path / filename
                joblib.dump(model, filepath)
                saved_files[model_name] = str(filepath)
                logger.info(f"Saved {model_name} to {filepath}")
            
            # Save ensemble model
            if self.ensemble_model is not None:
                ensemble_path = model_path / ENSEMBLE_MODEL
                joblib.dump(self.ensemble_model, ensemble_path)
                saved_files['ensemble'] = str(ensemble_path)
                logger.info(f"Saved ensemble model to {ensemble_path}")
            
            # Save feature scaler
            scaler_path = model_path / FEATURE_SCALER
            joblib.dump(self.feature_scaler, scaler_path)
            saved_files['scaler'] = str(scaler_path)
            logger.info(f"Saved feature scaler to {scaler_path}")
            
            # Save model metadata
            metadata = {
                'training_timestamp': datetime.now().isoformat(),
                'training_metrics': self.training_metrics,
                'feature_importance': {
                    name: importance.tolist() 
                    for name, importance in self.feature_importance.items()
                },
                'model_parameters': {
                    name: model.get_params() 
                    for name, model in self.models.items()
                }
            }
            
            metadata_path = model_path / MODEL_METADATA
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            saved_files['metadata'] = str(metadata_path)
            logger.info(f"Saved model metadata to {metadata_path}")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
            raise
        
        return saved_files
    
    def load_trained_models(self, model_path: Optional[str] = None) -> bool:
        """
        Load previously trained models.
        
        Args:
            model_path: Path to load models from (defaults to configured path)
            
        Returns:
            True if models loaded successfully, False otherwise
        """
        if model_path is None:
            model_path = MODEL_BASE_PATH
        else:
            model_path = Path(model_path)
        
        try:
            # Load individual models
            model_files = {
                'logistic_regression': LOGISTIC_REGRESSION_MODEL,
                'random_forest': RANDOM_FOREST_MODEL,
                'xgboost': XGBOOST_MODEL
            }
            
            for model_name, filename in model_files.items():
                filepath = model_path / filename
                if filepath.exists():
                    self.models[model_name] = joblib.load(filepath)
                    logger.info(f"Loaded {model_name} from {filepath}")
                else:
                    logger.warning(f"Model file not found: {filepath}")
                    return False
            
            # Load ensemble model
            ensemble_path = model_path / ENSEMBLE_MODEL
            if ensemble_path.exists():
                self.ensemble_model = joblib.load(ensemble_path)
                logger.info(f"Loaded ensemble model from {ensemble_path}")
            
            # Load feature scaler
            scaler_path = model_path / FEATURE_SCALER
            if scaler_path.exists():
                self.feature_scaler = joblib.load(scaler_path)
                logger.info(f"Loaded feature scaler from {scaler_path}")
            
            # Load metadata
            metadata_path = model_path / MODEL_METADATA
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                self.model_metadata = metadata
                logger.info(f"Loaded model metadata from {metadata_path}")
            
            logger.info("Successfully loaded all trained models")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using the ensemble model.
        
        Args:
            X: Features for prediction
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        if self.ensemble_model is None:
            raise ValueError("No trained ensemble model available")
        
        # Scale features
        X_scaled = self.feature_scaler.transform(X)
        
        # Make predictions
        predictions = self.ensemble_model.predict(X_scaled)
        probabilities = self.ensemble_model.predict_proba(X_scaled)
        
        return predictions, probabilities
    
    def get_feature_importance(self) -> Dict[str, np.ndarray]:
        """Get feature importance from all models"""
        return self.feature_importance.copy()
    
    def get_training_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get training metrics for all models"""
        return self.training_metrics.copy()
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary"""
        return {
            'models_trained': list(self.models.keys()),
            'ensemble_available': self.ensemble_model is not None,
            'training_metrics': self.training_metrics,
            'feature_importance_available': len(self.feature_importance) > 0,
            'metadata': self.model_metadata
        }
