#!/usr/bin/env python3
"""
Test script for NFL prediction engine training pipeline.

This script performs a quick test of the training pipeline
to ensure all components work correctly.
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Add the api directory to the path
sys.path.append(str(Path(__file__).parent))

from prediction_engine.training import (
    TrainingDataPreprocessor,
    EnsembleModelTrainer,
    ModelValidator
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_mock_data():
    """Create mock training data for testing"""
    logger.info("Creating mock training data...")
    
    # Create mock features
    n_samples = 1000
    n_features = 20
    
    # Generate random features
    np.random.seed(42)
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Add some realistic NFL-like features
    X['season'] = np.random.choice([2020, 2021, 2022, 2023], n_samples)
    X['week'] = np.random.randint(1, 18, n_samples)
    X['home_wins'] = np.random.randint(0, 16, n_samples)
    X['away_wins'] = np.random.randint(0, 16, n_samples)
    X['home_win_pct'] = X['home_wins'] / (X['home_wins'] + X['away_wins'] + 1)
    X['away_win_pct'] = X['away_wins'] / (X['home_wins'] + X['away_wins'] + 1)
    X['spread'] = np.random.normal(0, 7, n_samples)
    
    # Generate realistic targets (home team wins)
    # Make it somewhat predictable based on features
    home_advantage = 0.1
    win_pct_diff = X['home_win_pct'] - X['away_win_pct']
    spread_effect = -X['spread'] / 10  # Negative spread = home favorite
    
    win_probability = 0.5 + home_advantage + win_pct_diff * 0.3 + spread_effect * 0.1
    win_probability = np.clip(win_probability, 0.1, 0.9)
    
    y = np.random.binomial(1, win_probability, n_samples)
    
    return X, y

def test_training_pipeline():
    """Test the complete training pipeline"""
    logger.info("=" * 50)
    logger.info("TESTING NFL PREDICTION ENGINE TRAINING PIPELINE")
    logger.info("=" * 50)
    
    try:
        # Step 1: Create mock data
        logger.info("\nStep 1: Creating mock data...")
        X, y = create_mock_data()
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train = X.iloc[:split_idx]
        X_val = X.iloc[split_idx:]
        y_train = y[:split_idx]
        y_val = y[split_idx:]
        
        logger.info(f"Training set: {X_train.shape}")
        logger.info(f"Validation set: {X_val.shape}")
        
        # Step 2: Test model trainer
        logger.info("\nStep 2: Testing model trainer...")
        trainer = EnsembleModelTrainer()
        
        # Train individual models
        training_results = trainer.train_individual_models(X_train, y_train, X_val, y_val)
        logger.info("Individual models trained successfully!")
        
        # Create ensemble
        ensemble_results = trainer.create_ensemble(X_train, y_train, X_val, y_val)
        logger.info("Ensemble model created successfully!")
        
        # Step 3: Test model validator
        logger.info("\nStep 3: Testing model validator...")
        validator = ModelValidator()
        
        # Cross-validation
        cv_results = validator.cross_validate_models(
            X_train, y_train, trainer.models, cv_folds=3
        )
        logger.info("Cross-validation completed successfully!")
        
        # Step 4: Test predictions
        logger.info("\nStep 4: Testing predictions...")
        predictions, probabilities = trainer.predict(X_val)
        
        # Calculate accuracy
        accuracy = np.mean(predictions == y_val)
        logger.info(f"Validation accuracy: {accuracy:.3f}")
        
        # Step 5: Test model saving/loading
        logger.info("\nStep 5: Testing model persistence...")
        
        # Save models to temporary directory
        temp_dir = Path("temp_models")
        temp_dir.mkdir(exist_ok=True)
        
        saved_files = trainer.save_trained_models(str(temp_dir))
        logger.info("Models saved successfully!")
        
        # Test loading
        new_trainer = EnsembleModelTrainer()
        load_success = new_trainer.load_trained_models(str(temp_dir))
        
        if load_success:
            logger.info("Models loaded successfully!")
            
            # Test predictions with loaded models
            loaded_predictions, loaded_probabilities = new_trainer.predict(X_val)
            loaded_accuracy = np.mean(loaded_predictions == y_val)
            logger.info(f"Loaded model accuracy: {loaded_accuracy:.3f}")
        else:
            logger.error("Failed to load models!")
            return False
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)
        logger.info("Temporary files cleaned up")
        
        # Step 6: Print results summary
        logger.info("\n" + "=" * 50)
        logger.info("TEST RESULTS SUMMARY")
        logger.info("=" * 50)
        
        logger.info("✅ All components working correctly!")
        logger.info(f"✅ Training accuracy: {training_results.get('ensemble', {}).get('train_accuracy', 0):.3f}")
        logger.info(f"✅ Validation accuracy: {accuracy:.3f}")
        logger.info(f"✅ Cross-validation completed: {len(cv_results)} models")
        logger.info(f"✅ Model persistence: Working")
        logger.info(f"✅ Ensemble weights: {trainer.optimize_ensemble_weights(X_val, y_val)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_data_preprocessor():
    """Test the data preprocessor with mock data"""
    logger.info("\n" + "=" * 50)
    logger.info("TESTING DATA PREPROCESSOR")
    logger.info("=" * 50)
    
    try:
        # Create mock game logs
        mock_game_logs = pd.DataFrame({
            'season': [2023, 2023, 2023, 2023],
            'week': [1, 1, 2, 2],
            'home_team': ['KC', 'BUF', 'SF', 'DAL'],
            'away_team': ['DET', 'NYJ', 'SEA', 'NYG'],
            'home_score': [21, 16, 30, 40],
            'away_score': [20, 22, 23, 0],
            'spread': [-6.5, -3.0, -7.0, -10.0]
        })
        
        # Create mock season data
        mock_season_data = pd.DataFrame({
            'team': ['KC', 'BUF', 'SF', 'DAL', 'DET', 'NYJ', 'SEA', 'NYG'],
            'season': [2023, 2023, 2023, 2023, 2023, 2023, 2023, 2023],
            'wins': [12, 11, 13, 12, 9, 7, 9, 6],
            'losses': [5, 6, 4, 5, 8, 10, 8, 11],
            'points_for': [400, 380, 420, 410, 320, 280, 350, 290],
            'points_against': [350, 360, 300, 320, 380, 400, 370, 420]
        })
        
        # Save mock data
        data_dir = Path("temp_data")
        data_dir.mkdir(exist_ok=True)
        
        mock_game_logs.to_json(data_dir / "game_log.json", orient='records')
        mock_season_data.to_json(data_dir / "season_data_by_team.json", orient='records')
        
        # Test preprocessor
        preprocessor = TrainingDataPreprocessor(data_path=str(data_dir))
        
        # Test data preparation
        X_train, X_val, y_train, y_val = preprocessor.prepare_training_data(
            start_season=2023, end_season=2023, validation_split=0.5
        )
        
        logger.info(f"✅ Data preprocessor working!")
        logger.info(f"✅ Training features: {X_train.shape}")
        logger.info(f"✅ Training targets: {y_train.shape}")
        
        # Cleanup
        import shutil
        shutil.rmtree(data_dir)
        
        return True
        
    except Exception as e:
        logger.error(f"Data preprocessor test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    logger.info("Starting NFL Prediction Engine Tests...")
    
    # Test data preprocessor
    preprocessor_success = test_data_preprocessor()
    
    # Test training pipeline
    pipeline_success = test_training_pipeline()
    
    if preprocessor_success and pipeline_success:
        logger.info("\n🎉 ALL TESTS PASSED! Training pipeline is ready to use.")
        sys.exit(0)
    else:
        logger.error("\n❌ SOME TESTS FAILED! Check the logs above.")
        sys.exit(1)
