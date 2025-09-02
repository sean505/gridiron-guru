"""
Model validator for NFL prediction engine.

This module provides comprehensive validation methods including
cross-validation, upset detection testing, and seasonal performance analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

logger = logging.getLogger(__name__)

class ModelValidator:
    """Validates model performance using multiple methods"""
    
    def __init__(self):
        self.validation_results = {}
        self.upset_detection_results = {}
        self.seasonal_performance = {}
        
        logger.info("Initialized ModelValidator")
    
    def cross_validate_models(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        models: Dict[str, Any],
        cv_folds: int = 5,
        feature_scaler: Optional[Any] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Perform time-series aware cross-validation.
        Ensures no data leakage by training on past, testing on future.
        
        Args:
            X: Features
            y: Targets
            models: Dictionary of models to validate
            cv_folds: Number of cross-validation folds
            feature_scaler: Optional feature scaler
            
        Returns:
            Cross-validation results for each model
        """
        logger.info(f"Performing {cv_folds}-fold time-series cross-validation...")
        
        # Use TimeSeriesSplit to ensure temporal order
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        cv_results = {}
        
        for model_name, model in models.items():
            logger.info(f"Cross-validating {model_name}...")
            
            fold_scores = {
                'accuracy': [],
                'precision': [],
                'recall': [],
                'f1': [],
                'auc': []
            }
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                logger.info(f"  Fold {fold + 1}/{cv_folds}")
                
                # Split data
                if hasattr(X, 'iloc'):
                    X_train_fold = X.iloc[train_idx]
                    X_val_fold = X.iloc[val_idx]
                else:
                    X_train_fold = X[train_idx]
                    X_val_fold = X[val_idx]
                
                if hasattr(y, 'iloc'):
                    y_train_fold = y.iloc[train_idx]
                    y_val_fold = y.iloc[val_idx]
                else:
                    y_train_fold = y[train_idx]
                    y_val_fold = y[val_idx]
                
                # Scale features if scaler provided
                if feature_scaler is not None:
                    X_train_fold_scaled = feature_scaler.fit_transform(X_train_fold)
                    X_val_fold_scaled = feature_scaler.transform(X_val_fold)
                else:
                    X_train_fold_scaled = X_train_fold
                    X_val_fold_scaled = X_val_fold
                
                try:
                    # Train model
                    model.fit(X_train_fold_scaled, y_train_fold)
                    
                    # Make predictions
                    y_pred = model.predict(X_val_fold_scaled)
                    
                    # Get probabilities if available
                    if hasattr(model, 'predict_proba'):
                        y_prob = model.predict_proba(X_val_fold_scaled)[:, 1]
                    else:
                        y_prob = y_pred
                    
                    # Calculate metrics
                    accuracy = accuracy_score(y_val_fold, y_pred)
                    precision = precision_score(y_val_fold, y_pred, average='weighted')
                    recall = recall_score(y_val_fold, y_pred, average='weighted')
                    f1 = f1_score(y_val_fold, y_pred, average='weighted')
                    
                    try:
                        auc = roc_auc_score(y_val_fold, y_prob)
                    except ValueError:
                        auc = 0.5  # Default if only one class in fold
                    
                    # Store fold results
                    fold_scores['accuracy'].append(accuracy)
                    fold_scores['precision'].append(precision)
                    fold_scores['recall'].append(recall)
                    fold_scores['f1'].append(f1)
                    fold_scores['auc'].append(auc)
                    
                except Exception as e:
                    logger.warning(f"Error in fold {fold + 1} for {model_name}: {e}")
                    continue
            
            # Calculate mean and std for each metric
            cv_results[model_name] = {}
            for metric, scores in fold_scores.items():
                if scores:  # Only if we have valid scores
                    cv_results[model_name][f'{metric}_mean'] = np.mean(scores)
                    cv_results[model_name][f'{metric}_std'] = np.std(scores)
                    cv_results[model_name][f'{metric}_scores'] = scores
                else:
                    cv_results[model_name][f'{metric}_mean'] = 0.0
                    cv_results[model_name][f'{metric}_std'] = 0.0
                    cv_results[model_name][f'{metric}_scores'] = []
            
            logger.info(f"{model_name} CV Results:")
            logger.info(f"  Accuracy: {cv_results[model_name]['accuracy_mean']:.3f} ± {cv_results[model_name]['accuracy_std']:.3f}")
            logger.info(f"  F1 Score: {cv_results[model_name]['f1_mean']:.3f} ± {cv_results[model_name]['f1_std']:.3f}")
            logger.info(f"  AUC: {cv_results[model_name]['auc_mean']:.3f} ± {cv_results[model_name]['auc_std']:.3f}")
        
        self.validation_results = cv_results
        return cv_results
    
    def validate_upset_detection(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        vegas_lines: pd.Series,
        model: Any,
        feature_scaler: Optional[Any] = None,
        upset_threshold: float = 0.3
    ) -> Dict[str, float]:
        """
        Specifically test upset prediction accuracy.
        Compare model predictions vs Vegas favorites.
        
        Args:
            X: Features
            y: Actual outcomes
            vegas_lines: Vegas point spreads (negative = home favorite)
            model: Trained model
            feature_scaler: Optional feature scaler
            upset_threshold: Threshold for considering a prediction an upset
            
        Returns:
            Upset detection performance metrics
        """
        logger.info("Validating upset detection performance...")
        
        # Scale features if scaler provided
        if feature_scaler is not None:
            X_scaled = feature_scaler.transform(X)
        else:
            X_scaled = X
        
        # Get model predictions
        y_pred = model.predict(X_scaled)
        y_prob = model.predict_proba(X_scaled)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        # Identify Vegas favorites (negative spread = home favorite)
        vegas_favorites = (vegas_lines < 0).astype(int)  # 1 if home favorite, 0 if away favorite
        
        # Identify upsets (Vegas favorite lost)
        actual_upsets = (vegas_favorites != y).astype(int)
        
        # Identify model upset predictions (model disagrees with Vegas)
        model_upset_predictions = (vegas_favorites != y_pred).astype(int)
        
        # Calculate upset detection metrics
        upset_precision = precision_score(actual_upsets, model_upset_predictions, zero_division=0)
        upset_recall = recall_score(actual_upsets, model_upset_predictions, zero_division=0)
        upset_f1 = f1_score(actual_upsets, model_upset_predictions, zero_division=0)
        
        # Calculate accuracy when model predicts upset
        upset_predictions_mask = model_upset_predictions == 1
        if upset_predictions_mask.sum() > 0:
            upset_accuracy = accuracy_score(
                actual_upsets[upset_predictions_mask], 
                y_pred[upset_predictions_mask]
            )
        else:
            upset_accuracy = 0.0
        
        # Calculate overall accuracy
        overall_accuracy = accuracy_score(y, y_pred)
        
        # Calculate accuracy vs Vegas
        vegas_accuracy = accuracy_score(y, vegas_favorites)
        model_vs_vegas = overall_accuracy - vegas_accuracy
        
        upset_results = {
            'upset_precision': upset_precision,
            'upset_recall': upset_recall,
            'upset_f1': upset_f1,
            'upset_accuracy': upset_accuracy,
            'overall_accuracy': overall_accuracy,
            'vegas_accuracy': vegas_accuracy,
            'model_vs_vegas_improvement': model_vs_vegas,
            'total_upsets': actual_upsets.sum(),
            'predicted_upsets': model_upset_predictions.sum(),
            'correct_upset_predictions': ((actual_upsets == 1) & (model_upset_predictions == 1)).sum()
        }
        
        logger.info("Upset Detection Results:")
        logger.info(f"  Upset Precision: {upset_precision:.3f}")
        logger.info(f"  Upset Recall: {upset_recall:.3f}")
        logger.info(f"  Upset F1: {upset_f1:.3f}")
        logger.info(f"  Upset Accuracy: {upset_accuracy:.3f}")
        logger.info(f"  Model vs Vegas Improvement: {model_vs_vegas:.3f}")
        
        self.upset_detection_results = upset_results
        return upset_results
    
    def seasonal_performance_analysis(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        seasons: pd.Series,
        weeks: pd.Series,
        model: Any,
        feature_scaler: Optional[Any] = None,
        test_seasons: List[int] = [2022, 2023, 2024]
    ) -> Dict[str, Dict[str, float]]:
        """
        Test model performance on recent complete seasons.
        Analyze performance by week and season.
        
        Args:
            X: Features
            y: Targets
            seasons: Season for each sample
            weeks: Week for each sample
            model: Trained model
            feature_scaler: Optional feature scaler
            test_seasons: Seasons to test on
            
        Returns:
            Performance breakdown by season and week
        """
        logger.info(f"Analyzing seasonal performance for seasons: {test_seasons}")
        
        # Scale features if scaler provided
        if feature_scaler is not None:
            X_scaled = feature_scaler.transform(X)
        else:
            X_scaled = X
        
        seasonal_results = {}
        
        for season in test_seasons:
            logger.info(f"Analyzing season {season}...")
            
            # Filter data for this season
            season_mask = seasons == season
            if not season_mask.any():
                logger.warning(f"No data found for season {season}")
                continue
            
            X_season = X_scaled[season_mask]
            y_season = y[season_mask]
            weeks_season = weeks[season_mask]
            
            # Make predictions
            y_pred_season = model.predict(X_season)
            y_prob_season = model.predict_proba(X_season)[:, 1] if hasattr(model, 'predict_proba') else y_pred_season
            
            # Calculate overall season metrics
            season_accuracy = accuracy_score(y_season, y_pred_season)
            season_precision = precision_score(y_season, y_pred_season, average='weighted')
            season_recall = recall_score(y_season, y_pred_season, average='weighted')
            season_f1 = f1_score(y_season, y_pred_season, average='weighted')
            
            # Analyze by week
            week_performance = {}
            for week in sorted(weeks_season.unique()):
                week_mask = weeks_season == week
                if week_mask.sum() > 0:
                    week_accuracy = accuracy_score(y_season[week_mask], y_pred_season[week_mask])
                    week_performance[f'week_{week}'] = {
                        'accuracy': week_accuracy,
                        'games': week_mask.sum()
                    }
            
            seasonal_results[f'season_{season}'] = {
                'overall_accuracy': season_accuracy,
                'precision': season_precision,
                'recall': season_recall,
                'f1': season_f1,
                'total_games': len(y_season),
                'week_performance': week_performance
            }
            
            logger.info(f"Season {season} - Accuracy: {season_accuracy:.3f}, Games: {len(y_season)}")
        
        # Analyze early vs late season performance
        early_season_accuracy = []
        late_season_accuracy = []
        
        for season in test_seasons:
            season_mask = seasons == season
            if not season_mask.any():
                continue
            
            weeks_season = weeks[season_mask]
            y_season = y[season_mask]
            y_pred_season = model.predict(X_scaled[season_mask])
            
            # Early season (weeks 1-8)
            early_mask = (weeks_season >= 1) & (weeks_season <= 8)
            if early_mask.any():
                early_acc = accuracy_score(y_season[early_mask], y_pred_season[early_mask])
                early_season_accuracy.append(early_acc)
            
            # Late season (weeks 9-17)
            late_mask = (weeks_season >= 9) & (weeks_season <= 17)
            if late_mask.any():
                late_acc = accuracy_score(y_season[late_mask], y_pred_season[late_mask])
                late_season_accuracy.append(late_acc)
        
        # Add early vs late season analysis
        if early_season_accuracy and late_season_accuracy:
            seasonal_results['early_vs_late'] = {
                'early_season_accuracy': np.mean(early_season_accuracy),
                'late_season_accuracy': np.mean(late_season_accuracy),
                'early_season_std': np.std(early_season_accuracy),
                'late_season_std': np.std(late_season_accuracy),
                'improvement': np.mean(late_season_accuracy) - np.mean(early_season_accuracy)
            }
        
        self.seasonal_performance = seasonal_results
        return seasonal_results
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """
        Create comprehensive performance summary.
        Include accuracy metrics, feature importance, confusion matrices.
        """
        logger.info("Generating comprehensive performance report...")
        
        report = {
            'validation_timestamp': datetime.now().isoformat(),
            'cross_validation_results': self.validation_results,
            'upset_detection_results': self.upset_detection_results,
            'seasonal_performance': self.seasonal_performance,
            'summary': {}
        }
        
        # Generate summary statistics
        if self.validation_results:
            # Find best performing model
            best_model = None
            best_accuracy = 0.0
            
            for model_name, results in self.validation_results.items():
                if 'accuracy_mean' in results and results['accuracy_mean'] > best_accuracy:
                    best_accuracy = results['accuracy_mean']
                    best_model = model_name
            
            report['summary']['best_model'] = best_model
            report['summary']['best_accuracy'] = best_accuracy
        
        # Upset detection summary
        if self.upset_detection_results:
            report['summary']['upset_detection'] = {
                'precision': self.upset_detection_results.get('upset_precision', 0.0),
                'recall': self.upset_detection_results.get('upset_recall', 0.0),
                'f1': self.upset_detection_results.get('upset_f1', 0.0),
                'vs_vegas_improvement': self.upset_detection_results.get('model_vs_vegas_improvement', 0.0)
            }
        
        # Seasonal performance summary
        if self.seasonal_performance:
            season_accuracies = []
            for key, results in self.seasonal_performance.items():
                if key.startswith('season_') and 'overall_accuracy' in results:
                    season_accuracies.append(results['overall_accuracy'])
            
            if season_accuracies:
                report['summary']['seasonal_performance'] = {
                    'mean_accuracy': np.mean(season_accuracies),
                    'std_accuracy': np.std(season_accuracies),
                    'min_accuracy': np.min(season_accuracies),
                    'max_accuracy': np.max(season_accuracies)
                }
        
        # Performance assessment
        report['summary']['performance_assessment'] = self._assess_performance(report['summary'])
        
        logger.info("Performance report generated")
        return report
    
    def _assess_performance(self, summary: Dict[str, Any]) -> Dict[str, str]:
        """Assess overall model performance and provide recommendations"""
        assessment = {}
        
        # Overall accuracy assessment
        best_accuracy = summary.get('best_accuracy', 0.0)
        if best_accuracy >= 0.65:
            assessment['overall'] = "Excellent - Competitive with expert analysts"
        elif best_accuracy >= 0.60:
            assessment['overall'] = "Good - Above random chance, room for improvement"
        elif best_accuracy >= 0.55:
            assessment['overall'] = "Fair - Slightly better than random"
        else:
            assessment['overall'] = "Poor - Needs significant improvement"
        
        # Upset detection assessment
        upset_f1 = summary.get('upset_detection', {}).get('f1', 0.0)
        if upset_f1 >= 0.6:
            assessment['upset_detection'] = "Excellent - Strong upset prediction capability"
        elif upset_f1 >= 0.5:
            assessment['upset_detection'] = "Good - Decent upset detection"
        elif upset_f1 >= 0.4:
            assessment['upset_detection'] = "Fair - Some upset detection ability"
        else:
            assessment['upset_detection'] = "Poor - Limited upset detection"
        
        # Seasonal consistency assessment
        seasonal_std = summary.get('seasonal_performance', {}).get('std_accuracy', 1.0)
        if seasonal_std <= 0.05:
            assessment['consistency'] = "Excellent - Very consistent across seasons"
        elif seasonal_std <= 0.10:
            assessment['consistency'] = "Good - Reasonably consistent"
        else:
            assessment['consistency'] = "Fair - Some seasonal variation"
        
        return assessment
