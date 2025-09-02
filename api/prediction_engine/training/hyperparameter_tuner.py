"""
Hyperparameter tuner for NFL prediction engine.

This module optimizes model hyperparameters using grid search
and random search techniques for best performance.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

logger = logging.getLogger(__name__)

class HyperparameterTuner:
    """Optimizes model hyperparameters for best performance"""
    
    def __init__(self, cv_folds: int = 3, n_jobs: int = -1, random_state: int = 42):
        self.cv_folds = cv_folds
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.tuning_results = {}
        
        logger.info(f"Initialized HyperparameterTuner with {cv_folds} CV folds")
    
    def tune_random_forest(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        use_grid_search: bool = True
    ) -> Dict[str, Any]:
        """
        Tune Random Forest hyperparameters.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            use_grid_search: Use grid search (True) or random search (False)
            
        Returns:
            Best parameters and performance metrics
        """
        logger.info("Tuning Random Forest hyperparameters...")
        
        # Define parameter grid
        if use_grid_search:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None],
                'class_weight': ['balanced', 'balanced_subsample', None]
            }
            
            # Use GridSearchCV
            rf = RandomForestClassifier(random_state=self.random_state, n_jobs=self.n_jobs)
            search = GridSearchCV(
                rf, 
                param_grid, 
                cv=self.cv_folds,
                scoring='accuracy',
                n_jobs=self.n_jobs,
                verbose=1
            )
        else:
            # Use RandomizedSearchCV for faster tuning
            param_dist = {
                'n_estimators': [50, 100, 150, 200, 250],
                'max_depth': [3, 5, 7, 10, 15, None],
                'min_samples_split': [2, 3, 5, 7, 10],
                'min_samples_leaf': [1, 2, 3, 4, 5],
                'max_features': ['sqrt', 'log2', 0.5, 0.7, None],
                'class_weight': ['balanced', 'balanced_subsample', None]
            }
            
            rf = RandomForestClassifier(random_state=self.random_state, n_jobs=self.n_jobs)
            search = RandomizedSearchCV(
                rf,
                param_dist,
                n_iter=50,  # Number of parameter settings sampled
                cv=self.cv_folds,
                scoring='accuracy',
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                verbose=1
            )
        
        # Perform search
        search.fit(X_train, y_train)
        
        # Evaluate on validation set
        best_model = search.best_estimator_
        val_pred = best_model.predict(X_val)
        val_accuracy = accuracy_score(y_val, val_pred)
        
        results = {
            'best_params': search.best_params_,
            'best_cv_score': search.best_score_,
            'val_accuracy': val_accuracy,
            'cv_results': search.cv_results_,
            'search_type': 'grid' if use_grid_search else 'random'
        }
        
        logger.info(f"Random Forest tuning complete:")
        logger.info(f"  Best CV Score: {search.best_score_:.3f}")
        logger.info(f"  Validation Accuracy: {val_accuracy:.3f}")
        logger.info(f"  Best Parameters: {search.best_params_}")
        
        self.tuning_results['random_forest'] = results
        return results
    
    def tune_xgboost(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        use_grid_search: bool = False
    ) -> Dict[str, Any]:
        """
        Tune XGBoost hyperparameters.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            use_grid_search: Use grid search (True) or random search (False)
            
        Returns:
            Best parameters and performance metrics
        """
        logger.info("Tuning XGBoost hyperparameters...")
        
        # Define parameter grid/distribution
        if use_grid_search:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'reg_alpha': [0, 0.1, 1],
                'reg_lambda': [0, 0.1, 1]
            }
            
            xgb_model = xgb.XGBClassifier(
                random_state=self.random_state,
                eval_metric='logloss',
                early_stopping_rounds=10
            )
            
            search = GridSearchCV(
                xgb_model,
                param_grid,
                cv=self.cv_folds,
                scoring='accuracy',
                n_jobs=self.n_jobs,
                verbose=1
            )
        else:
            # Use RandomizedSearchCV for XGBoost (recommended due to complexity)
            param_dist = {
                'n_estimators': [50, 100, 150, 200, 300],
                'max_depth': [3, 4, 5, 6, 7, 8, 9],
                'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2, 0.25],
                'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
                'reg_alpha': [0, 0.01, 0.1, 0.5, 1],
                'reg_lambda': [0, 0.01, 0.1, 0.5, 1],
                'gamma': [0, 0.1, 0.5, 1, 2]
            }
            
            xgb_model = xgb.XGBClassifier(
                random_state=self.random_state,
                eval_metric='logloss'
            )
            
            search = RandomizedSearchCV(
                xgb_model,
                param_dist,
                n_iter=30,  # Fewer iterations for XGBoost due to complexity
                cv=self.cv_folds,
                scoring='accuracy',
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                verbose=1
            )
        
        # Perform search
        search.fit(X_train, y_train)
        
        # Evaluate on validation set
        best_model = search.best_estimator_
        val_pred = best_model.predict(X_val)
        val_accuracy = accuracy_score(y_val, val_pred)
        
        results = {
            'best_params': search.best_params_,
            'best_cv_score': search.best_score_,
            'val_accuracy': val_accuracy,
            'cv_results': search.cv_results_,
            'search_type': 'grid' if use_grid_search else 'random'
        }
        
        logger.info(f"XGBoost tuning complete:")
        logger.info(f"  Best CV Score: {search.best_score_:.3f}")
        logger.info(f"  Validation Accuracy: {val_accuracy:.3f}")
        logger.info(f"  Best Parameters: {search.best_params_}")
        
        self.tuning_results['xgboost'] = results
        return results
    
    def tune_logistic_regression(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> Dict[str, Any]:
        """
        Tune Logistic Regression hyperparameters.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Best parameters and performance metrics
        """
        logger.info("Tuning Logistic Regression hyperparameters...")
        
        # Define parameter grid
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2', 'elasticnet'],
            'solver': ['liblinear', 'saga'],
            'max_iter': [1000, 2000, 5000],
            'class_weight': ['balanced', None]
        }
        
        # Note: Not all combinations are valid (e.g., l1 penalty with liblinear)
        # We'll use a more targeted approach
        valid_combinations = [
            {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'penalty': ['l2'], 'solver': ['liblinear'], 'max_iter': [1000], 'class_weight': ['balanced', None]},
            {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'penalty': ['l1'], 'solver': ['liblinear'], 'max_iter': [1000], 'class_weight': ['balanced', None]},
            {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'penalty': ['elasticnet'], 'solver': ['saga'], 'max_iter': [1000], 'class_weight': ['balanced', None]}
        ]
        
        best_score = 0
        best_params = None
        best_model = None
        all_results = []
        
        for param_set in valid_combinations:
            try:
                lr = LogisticRegression(random_state=self.random_state)
                search = GridSearchCV(
                    lr,
                    param_set,
                    cv=self.cv_folds,
                    scoring='accuracy',
                    n_jobs=self.n_jobs,
                    verbose=0
                )
                
                search.fit(X_train, y_train)
                
                if search.best_score_ > best_score:
                    best_score = search.best_score_
                    best_params = search.best_params_
                    best_model = search.best_estimator_
                
                all_results.extend(search.cv_results_['params'])
                
            except Exception as e:
                logger.warning(f"Error with parameter combination {param_set}: {e}")
                continue
        
        # Evaluate best model on validation set
        if best_model is not None:
            val_pred = best_model.predict(X_val)
            val_accuracy = accuracy_score(y_val, val_pred)
        else:
            val_accuracy = 0.0
            best_params = {}
        
        results = {
            'best_params': best_params,
            'best_cv_score': best_score,
            'val_accuracy': val_accuracy,
            'search_type': 'grid'
        }
        
        logger.info(f"Logistic Regression tuning complete:")
        logger.info(f"  Best CV Score: {best_score:.3f}")
        logger.info(f"  Validation Accuracy: {val_accuracy:.3f}")
        logger.info(f"  Best Parameters: {best_params}")
        
        self.tuning_results['logistic_regression'] = results
        return results
    
    def tune_ensemble_weights(
        self, 
        X_val: pd.DataFrame, 
        y_val: pd.Series,
        model_predictions: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """
        Optimize ensemble weights based on validation performance.
        
        Args:
            X_val: Validation features
            y_val: Validation targets
            model_predictions: Dictionary of model predictions (probabilities)
            
        Returns:
            Optimized weights for each model
        """
        logger.info("Optimizing ensemble weights...")
        
        # Define weight search space
        weight_combinations = []
        
        # Generate systematic weight combinations
        for lr_weight in np.arange(0.1, 0.8, 0.1):
            for rf_weight in np.arange(0.1, 0.8, 0.1):
                xgb_weight = 1.0 - lr_weight - rf_weight
                if xgb_weight > 0:
                    weight_combinations.append([lr_weight, rf_weight, xgb_weight])
        
        # Add some random combinations
        np.random.seed(self.random_state)
        for _ in range(20):
            weights = np.random.dirichlet([1, 1, 1])  # Random weights that sum to 1
            weight_combinations.append(weights.tolist())
        
        best_accuracy = 0.0
        best_weights = {'logistic_regression': 0.33, 'random_forest': 0.33, 'xgboost': 0.34}
        
        model_names = list(model_predictions.keys())
        
        for weights in weight_combinations:
            # Calculate weighted ensemble prediction
            ensemble_pred = np.zeros(len(y_val))
            for i, model_name in enumerate(model_names):
                if model_name in model_predictions:
                    ensemble_pred += weights[i] * model_predictions[model_name]
            
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
        
        results = {
            'best_weights': best_weights,
            'best_accuracy': best_accuracy,
            'weight_combinations_tested': len(weight_combinations)
        }
        
        logger.info(f"Ensemble weight optimization complete:")
        logger.info(f"  Best Accuracy: {best_accuracy:.3f}")
        logger.info(f"  Best Weights: {best_weights}")
        
        self.tuning_results['ensemble_weights'] = results
        return results
    
    def comprehensive_tuning(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        tune_individual_models: bool = True,
        tune_ensemble: bool = True
    ) -> Dict[str, Any]:
        """
        Perform comprehensive hyperparameter tuning for all models.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            tune_individual_models: Whether to tune individual models
            tune_ensemble: Whether to tune ensemble weights
            
        Returns:
            Complete tuning results
        """
        logger.info("Starting comprehensive hyperparameter tuning...")
        
        tuning_start_time = datetime.now()
        comprehensive_results = {}
        
        if tune_individual_models:
            # Tune individual models
            logger.info("Tuning individual models...")
            
            # Tune Random Forest
            rf_results = self.tune_random_forest(X_train, y_train, X_val, y_val, use_grid_search=False)
            comprehensive_results['random_forest'] = rf_results
            
            # Tune XGBoost
            xgb_results = self.tune_xgboost(X_train, y_train, X_val, y_val, use_grid_search=False)
            comprehensive_results['xgboost'] = xgb_results
            
            # Tune Logistic Regression
            lr_results = self.tune_logistic_regression(X_train, y_train, X_val, y_val)
            comprehensive_results['logistic_regression'] = lr_results
        
        if tune_ensemble:
            # For ensemble tuning, we need model predictions
            # This would typically be done after training with tuned parameters
            logger.info("Ensemble weight tuning requires trained models with tuned parameters")
            logger.info("This should be done after retraining models with optimal parameters")
        
        tuning_end_time = datetime.now()
        tuning_duration = (tuning_end_time - tuning_start_time).total_seconds()
        
        comprehensive_results['tuning_summary'] = {
            'tuning_duration_seconds': tuning_duration,
            'models_tuned': list(comprehensive_results.keys()),
            'tuning_timestamp': tuning_start_time.isoformat()
        }
        
        logger.info(f"Comprehensive tuning complete in {tuning_duration:.1f} seconds")
        
        return comprehensive_results
    
    def get_tuning_results(self) -> Dict[str, Any]:
        """Get all tuning results"""
        return self.tuning_results.copy()
    
    def get_best_parameters(self) -> Dict[str, Dict[str, Any]]:
        """Get best parameters for each model"""
        best_params = {}
        
        for model_name, results in self.tuning_results.items():
            if 'best_params' in results:
                best_params[model_name] = results['best_params']
        
        return best_params
