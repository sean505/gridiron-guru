#!/usr/bin/env python3
"""
Main training script for NFL prediction engine.

This script runs the complete training pipeline:
1. Load and preprocess historical data
2. Train individual models
3. Create ensemble model
4. Validate performance
5. Save trained models

Usage:
    python train_models.py [--tune-hyperparameters] [--quick-test]
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
import json
import pandas as pd

# Add the api directory to the path
sys.path.append(str(Path(__file__).parent))

from prediction_engine.training import (
    TrainingDataPreprocessor,
    EnsembleModelTrainer,
    ModelValidator,
    HyperparameterTuner
)
from prediction_engine.evaluation import (
    PerformanceEvaluator,
    ModelAnalyzer
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Main training pipeline"""
    parser = argparse.ArgumentParser(description='Train NFL prediction models')
    parser.add_argument('--tune-hyperparameters', action='store_true',
                       help='Perform hyperparameter tuning (slower)')
    parser.add_argument('--quick-test', action='store_true',
                       help='Quick test with limited data')
    parser.add_argument('--data-path', default='data',
                       help='Path to data directory')
    parser.add_argument('--output-dir', default='api/prediction_engine/models/trained',
                       help='Output directory for trained models')
    parser.add_argument('--start-season', type=int, default=2008,
                       help='Start season for training data')
    parser.add_argument('--end-season', type=int, default=2023,
                       help='End season for training data')
    parser.add_argument('--test-season', type=int, default=2024,
                       help='Season to use for holdout testing')
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("NFL PREDICTION ENGINE TRAINING PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Start time: {datetime.now()}")
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Training seasons: {args.start_season}-{args.end_season}")
    logger.info(f"Test season: {args.test_season}")
    logger.info(f"Hyperparameter tuning: {args.tune_hyperparameters}")
    logger.info(f"Quick test mode: {args.quick_test}")
    
    try:
        # Step 1: Data Preprocessing
        logger.info("\n" + "=" * 40)
        logger.info("STEP 1: DATA PREPROCESSING")
        logger.info("=" * 40)
        
        preprocessor = TrainingDataPreprocessor(data_path=args.data_path)
        
        # Prepare training data
        logger.info("Preparing training data...")
        X_train, X_val, y_train, y_val = preprocessor.prepare_training_data(
            start_season=args.start_season,
            end_season=args.end_season,
            validation_split=0.2
        )
        
        # Create holdout test set
        logger.info("Creating holdout test set...")
        X_test, y_test = preprocessor.create_holdout_test_set(test_season=args.test_season)
        
        if args.quick_test:
            # Use smaller subset for quick testing
            X_train = X_train.head(1000)
            y_train = y_train.head(1000)
            X_val = X_val.head(200)
            y_val = y_val.head(200)
            logger.info("Quick test mode: Using reduced dataset")
        
        logger.info(f"Training set: {X_train.shape}")
        logger.info(f"Validation set: {X_val.shape}")
        logger.info(f"Test set: {X_test.shape}")
        
        # Step 2: Hyperparameter Tuning (Optional)
        if args.tune_hyperparameters:
            logger.info("\n" + "=" * 40)
            logger.info("STEP 2: HYPERPARAMETER TUNING")
            logger.info("=" * 40)
            
            tuner = HyperparameterTuner(cv_folds=3)  # Reduced for speed
            
            # Tune individual models
            tuning_results = tuner.comprehensive_tuning(
                X_train, y_train, X_val, y_val,
                tune_individual_models=True,
                tune_ensemble=False  # Will tune ensemble after retraining
            )
            
            # Save tuning results
            tuning_file = Path(args.output_dir) / "hyperparameter_tuning_results.json"
            tuning_file.parent.mkdir(parents=True, exist_ok=True)
            with open(tuning_file, 'w') as f:
                json.dump(tuning_results, f, indent=2, default=str)
            
            logger.info(f"Hyperparameter tuning results saved to {tuning_file}")
        
        # Step 3: Model Training
        logger.info("\n" + "=" * 40)
        logger.info("STEP 3: MODEL TRAINING")
        logger.info("=" * 40)
        
        trainer = EnsembleModelTrainer()
        
        # Train individual models
        logger.info("Training individual models...")
        training_results = trainer.train_individual_models(X_train, y_train, X_val, y_val)
        
        # Create ensemble
        logger.info("Creating ensemble model...")
        ensemble_results = trainer.create_ensemble(X_train, y_train, X_val, y_val)
        
        # Optimize ensemble weights
        logger.info("Optimizing ensemble weights...")
        ensemble_weights = trainer.optimize_ensemble_weights(X_val, y_val)
        
        logger.info("Model training completed successfully!")
        
        # Step 4: Model Validation
        logger.info("\n" + "=" * 40)
        logger.info("STEP 4: MODEL VALIDATION")
        logger.info("=" * 40)
        
        validator = ModelValidator()
        
        # Cross-validation
        logger.info("Performing cross-validation...")
        cv_results = validator.cross_validate_models(
            X_train, y_train, trainer.models, cv_folds=3
        )
        
        # Seasonal performance analysis
        if not X_test.empty:
            logger.info("Analyzing seasonal performance...")
            # Extract season and week from test data (assuming they're in the features)
            seasons = X_test.get('season', pd.Series([args.test_season] * len(X_test)))
            weeks = X_test.get('week', pd.Series([1] * len(X_test)))
            
            seasonal_results = validator.seasonal_performance_analysis(
                X_test, y_test, seasons, weeks, trainer.ensemble_model
            )
        else:
            seasonal_results = {}
        
        # Generate performance report
        logger.info("Generating performance report...")
        performance_report = validator.generate_performance_report()
        
        # Step 5: Comprehensive Evaluation
        logger.info("\n" + "=" * 40)
        logger.info("STEP 5: COMPREHENSIVE EVALUATION")
        logger.info("=" * 40)
        
        evaluator = PerformanceEvaluator()
        
        # Make predictions on test set
        if not X_test.empty:
            test_predictions, test_probabilities = trainer.predict(X_test)
            
            # Calculate comprehensive metrics
            logger.info("Calculating performance metrics...")
            evaluation_report = evaluator.generate_performance_report(
                y_test, test_predictions, test_probabilities
            )
            
            # Save evaluation results
            eval_file = Path(args.output_dir) / "evaluation_results.json"
            evaluator.save_evaluation_results(str(eval_file), evaluation_report)
            logger.info(f"Evaluation results saved to {eval_file}")
        else:
            evaluation_report = {}
            logger.warning("No test data available for evaluation")
        
        # Step 6: Model Analysis
        logger.info("\n" + "=" * 40)
        logger.info("STEP 6: MODEL ANALYSIS")
        logger.info("=" * 40)
        
        analyzer = ModelAnalyzer()
        
        # Analyze feature importance
        feature_names = list(X_train.columns)
        feature_importance = trainer.get_feature_importance()
        feature_analysis = analyzer.analyze_feature_importance(feature_importance, feature_names)
        
        # Analyze prediction patterns
        if not X_test.empty:
            pattern_analysis = analyzer.analyze_prediction_patterns(
                test_predictions, test_probabilities, X_test, y_test
            )
            
            # Analyze seasonal trends
            seasonal_analysis = analyzer.analyze_seasonal_trends(
                test_predictions, y_test, seasons, weeks
            )
            
            # Generate model insights
            model_insights = analyzer.generate_model_insights(
                feature_analysis, pattern_analysis, seasonal_analysis, {}
            )
            
            # Save model insights
            insights_file = Path(args.output_dir) / "model_insights.json"
            with open(insights_file, 'w') as f:
                json.dump(model_insights, f, indent=2, default=str)
            logger.info(f"Model insights saved to {insights_file}")
        else:
            model_insights = {}
        
        # Step 7: Save Models
        logger.info("\n" + "=" * 40)
        logger.info("STEP 7: SAVING MODELS")
        logger.info("=" * 40)
        
        # Create output directory
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save trained models
        saved_files = trainer.save_trained_models(str(output_path))
        logger.info("Models saved successfully!")
        for model_name, filepath in saved_files.items():
            logger.info(f"  {model_name}: {filepath}")
        
        # Step 8: Generate Training Summary
        logger.info("\n" + "=" * 40)
        logger.info("STEP 8: TRAINING SUMMARY")
        logger.info("=" * 40)
        
        training_summary = {
            'training_timestamp': datetime.now().isoformat(),
            'training_parameters': {
                'start_season': args.start_season,
                'end_season': args.end_season,
                'test_season': args.test_season,
                'hyperparameter_tuning': args.tune_hyperparameters,
                'quick_test': args.quick_test
            },
            'data_summary': {
                'training_samples': len(X_train),
                'validation_samples': len(X_val),
                'test_samples': len(X_test),
                'features': len(feature_names)
            },
            'training_results': training_results,
            'ensemble_results': ensemble_results,
            'ensemble_weights': ensemble_weights,
            'cross_validation_results': cv_results,
            'performance_report': performance_report,
            'evaluation_report': evaluation_report,
            'model_insights': model_insights,
            'saved_files': saved_files
        }
        
        # Save training summary
        summary_file = output_path / "training_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(training_summary, f, indent=2, default=str)
        
        logger.info(f"Training summary saved to {summary_file}")
        
        # Print final results
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        
        if evaluation_report:
            accuracy = evaluation_report.get('accuracy_metrics', {}).get('accuracy', 0.0)
            logger.info(f"Final Test Accuracy: {accuracy:.3f}")
        
        if performance_report.get('summary', {}).get('best_model'):
            best_model = performance_report['summary']['best_model']
            best_accuracy = performance_report['summary'].get('best_accuracy', 0.0)
            logger.info(f"Best Model: {best_model} (CV Accuracy: {best_accuracy:.3f})")
        
        logger.info(f"Models saved to: {output_path}")
        logger.info(f"Training completed at: {datetime.now()}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        logger.error("Check training.log for detailed error information")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
