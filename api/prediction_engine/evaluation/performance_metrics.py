"""
Performance evaluator for NFL prediction engine.

This module provides comprehensive evaluation tools for measuring
model performance, calibration, and betting-relevant metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report, log_loss
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class PerformanceEvaluator:
    """Comprehensive evaluation of prediction performance"""
    
    def __init__(self):
        self.evaluation_results = {}
        self.calibration_results = {}
        self.betting_metrics = {}
        
        logger.info("Initialized PerformanceEvaluator")
    
    def calculate_accuracy_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_prob: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate standard classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities
            
        Returns:
            Dictionary of accuracy metrics
        """
        logger.info("Calculating accuracy metrics...")
        
        # Basic classification metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        # AUC metrics
        try:
            auc_roc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc_roc = 0.5  # Default if only one class
        
        try:
            auc_pr = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc_pr = 0.5
        
        # Log loss
        try:
            logloss = log_loss(y_true, y_prob)
        except ValueError:
            logloss = float('inf')
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        # Additional metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc_roc,
            'auc_pr': auc_pr,
            'log_loss': logloss,
            'specificity': specificity,
            'sensitivity': sensitivity,
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn
        }
        
        logger.info(f"Accuracy: {accuracy:.3f}, F1: {f1:.3f}, AUC: {auc_roc:.3f}")
        
        return metrics
    
    def calculate_betting_metrics(
        self, 
        predictions: np.ndarray, 
        actual_results: np.ndarray, 
        vegas_lines: np.ndarray,
        confidence_threshold: float = 0.6
    ) -> Dict[str, float]:
        """
        Calculate metrics relevant to sports betting.
        
        Args:
            predictions: Model predictions (0/1)
            actual_results: Actual game outcomes (0/1)
            vegas_lines: Vegas point spreads (negative = home favorite)
            confidence_threshold: Minimum confidence for betting decisions
            
        Returns:
            Dictionary of betting-relevant metrics
        """
        logger.info("Calculating betting metrics...")
        
        # Vegas favorites (negative spread = home favorite)
        vegas_favorites = (vegas_lines < 0).astype(int)
        
        # Against the spread (ATS) accuracy
        # Model prediction vs actual outcome
        ats_accuracy = accuracy_score(actual_results, predictions)
        
        # Vegas accuracy
        vegas_accuracy = accuracy_score(actual_results, vegas_favorites)
        
        # Model vs Vegas improvement
        model_vs_vegas = ats_accuracy - vegas_accuracy
        
        # Upset detection metrics
        actual_upsets = (vegas_favorites != actual_results).astype(int)
        predicted_upsets = (vegas_favorites != predictions).astype(int)
        
        upset_precision = precision_score(actual_upsets, predicted_upsets, zero_division=0)
        upset_recall = recall_score(actual_upsets, predicted_upsets, zero_division=0)
        upset_f1 = f1_score(actual_upsets, predicted_upsets, zero_division=0)
        
        # Confidence-based betting simulation
        # This would require probability predictions for full implementation
        # For now, we'll use a simplified approach
        
        betting_metrics = {
            'ats_accuracy': ats_accuracy,
            'vegas_accuracy': vegas_accuracy,
            'model_vs_vegas_improvement': model_vs_vegas,
            'upset_precision': upset_precision,
            'upset_recall': upset_recall,
            'upset_f1': upset_f1,
            'total_upsets': actual_upsets.sum(),
            'predicted_upsets': predicted_upsets.sum(),
            'correct_upset_predictions': ((actual_upsets == 1) & (predicted_upsets == 1)).sum()
        }
        
        logger.info(f"ATS Accuracy: {ats_accuracy:.3f}")
        logger.info(f"Model vs Vegas: {model_vs_vegas:+.3f}")
        logger.info(f"Upset F1: {upset_f1:.3f}")
        
        return betting_metrics
    
    def weekly_performance_breakdown(
        self, 
        predictions_by_week: Dict[int, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze performance by NFL week.
        
        Args:
            predictions_by_week: Dictionary with week as key and prediction data as value
            
        Returns:
            Performance breakdown by week
        """
        logger.info("Analyzing weekly performance breakdown...")
        
        weekly_metrics = {}
        
        for week, week_data in predictions_by_week.items():
            y_true = week_data.get('y_true', np.array([]))
            y_pred = week_data.get('y_pred', np.array([]))
            
            if len(y_true) > 0 and len(y_pred) > 0:
                accuracy = accuracy_score(y_true, y_pred)
                weekly_metrics[f'week_{week}'] = {
                    'accuracy': accuracy,
                    'games': len(y_true),
                    'correct_predictions': (y_true == y_pred).sum()
                }
        
        # Calculate trends
        if weekly_metrics:
            accuracies = [metrics['accuracy'] for metrics in weekly_metrics.values()]
            weeks = list(weekly_metrics.keys())
            
            # Early season (weeks 1-8) vs late season (weeks 9-17)
            early_weeks = [w for w in weeks if int(w.split('_')[1]) <= 8]
            late_weeks = [w for w in weeks if int(w.split('_')[1]) > 8]
            
            early_accuracy = np.mean([weekly_metrics[w]['accuracy'] for w in early_weeks]) if early_weeks else 0.0
            late_accuracy = np.mean([weekly_metrics[w]['accuracy'] for w in late_weeks]) if late_weeks else 0.0
            
            weekly_metrics['trends'] = {
                'early_season_accuracy': early_accuracy,
                'late_season_accuracy': late_accuracy,
                'seasonal_improvement': late_accuracy - early_accuracy,
                'overall_accuracy': np.mean(accuracies),
                'accuracy_std': np.std(accuracies),
                'best_week': weeks[np.argmax(accuracies)],
                'worst_week': weeks[np.argmin(accuracies)]
            }
        
        logger.info(f"Analyzed {len(weekly_metrics)} weeks")
        
        return weekly_metrics
    
    def confidence_calibration_analysis(
        self, 
        predictions: np.ndarray, 
        actuals: np.ndarray,
        n_bins: int = 10
    ) -> Dict[str, Any]:
        """
        Check if model confidence scores are well-calibrated.
        
        Args:
            predictions: Predicted probabilities
            actuals: Actual outcomes
            n_bins: Number of bins for calibration analysis
            
        Returns:
            Calibration analysis results
        """
        logger.info("Analyzing confidence calibration...")
        
        # Calculate calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            actuals, predictions, n_bins=n_bins
        )
        
        # Calculate Expected Calibration Error (ECE)
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (predictions > bin_lower) & (predictions <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = actuals[in_bin].mean()
                avg_confidence_in_bin = predictions[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        # Calculate reliability diagram data
        reliability_data = []
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            in_bin = (predictions > bin_lower) & (predictions <= bin_upper)
            
            if in_bin.sum() > 0:
                reliability_data.append({
                    'bin_center': (bin_lower + bin_upper) / 2,
                    'fraction_of_positives': fraction_of_positives[i],
                    'mean_predicted_value': mean_predicted_value[i],
                    'count': in_bin.sum()
                })
        
        calibration_results = {
            'expected_calibration_error': ece,
            'fraction_of_positives': fraction_of_positives.tolist(),
            'mean_predicted_value': mean_predicted_value.tolist(),
            'reliability_data': reliability_data,
            'n_bins': n_bins,
            'total_samples': len(predictions)
        }
        
        logger.info(f"Expected Calibration Error: {ece:.3f}")
        
        self.calibration_results = calibration_results
        return calibration_results
    
    def generate_performance_report(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_prob: np.ndarray,
        vegas_lines: Optional[np.ndarray] = None,
        weekly_data: Optional[Dict[int, Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive performance report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities
            vegas_lines: Optional Vegas lines for betting metrics
            weekly_data: Optional weekly breakdown data
            
        Returns:
            Comprehensive performance report
        """
        logger.info("Generating comprehensive performance report...")
        
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'sample_size': len(y_true)
        }
        
        # Basic accuracy metrics
        accuracy_metrics = self.calculate_accuracy_metrics(y_true, y_pred, y_prob)
        report['accuracy_metrics'] = accuracy_metrics
        
        # Betting metrics (if Vegas lines provided)
        if vegas_lines is not None:
            betting_metrics = self.calculate_betting_metrics(y_pred, y_true, vegas_lines)
            report['betting_metrics'] = betting_metrics
        
        # Weekly breakdown (if provided)
        if weekly_data is not None:
            weekly_breakdown = self.weekly_performance_breakdown(weekly_data)
            report['weekly_breakdown'] = weekly_breakdown
        
        # Calibration analysis
        calibration_analysis = self.confidence_calibration_analysis(y_true, y_prob)
        report['calibration_analysis'] = calibration_analysis
        
        # Performance assessment
        report['performance_assessment'] = self._assess_overall_performance(report)
        
        # Recommendations
        report['recommendations'] = self._generate_recommendations(report)
        
        logger.info("Performance report generated")
        
        return report
    
    def _assess_overall_performance(self, report: Dict[str, Any]) -> Dict[str, str]:
        """Assess overall model performance and provide ratings"""
        assessment = {}
        
        accuracy_metrics = report.get('accuracy_metrics', {})
        betting_metrics = report.get('betting_metrics', {})
        calibration_analysis = report.get('calibration_analysis', {})
        
        # Overall accuracy assessment
        accuracy = accuracy_metrics.get('accuracy', 0.0)
        if accuracy >= 0.70:
            assessment['overall_accuracy'] = "Excellent - Competitive with expert analysts"
        elif accuracy >= 0.65:
            assessment['overall_accuracy'] = "Very Good - Strong predictive performance"
        elif accuracy >= 0.60:
            assessment['overall_accuracy'] = "Good - Above random chance"
        elif accuracy >= 0.55:
            assessment['overall_accuracy'] = "Fair - Slightly better than random"
        else:
            assessment['overall_accuracy'] = "Poor - Needs significant improvement"
        
        # Betting performance assessment
        if betting_metrics:
            model_vs_vegas = betting_metrics.get('model_vs_vegas_improvement', 0.0)
            if model_vs_vegas >= 0.05:
                assessment['betting_performance'] = "Excellent - Significant edge over Vegas"
            elif model_vs_vegas >= 0.02:
                assessment['betting_performance'] = "Good - Modest edge over Vegas"
            elif model_vs_vegas >= 0.0:
                assessment['betting_performance'] = "Fair - Matches Vegas performance"
            else:
                assessment['betting_performance'] = "Poor - Underperforms Vegas"
        
        # Calibration assessment
        ece = calibration_analysis.get('expected_calibration_error', 1.0)
        if ece <= 0.05:
            assessment['calibration'] = "Excellent - Well-calibrated confidence scores"
        elif ece <= 0.10:
            assessment['calibration'] = "Good - Reasonably well-calibrated"
        elif ece <= 0.15:
            assessment['calibration'] = "Fair - Some calibration issues"
        else:
            assessment['calibration'] = "Poor - Poorly calibrated confidence scores"
        
        return assessment
    
    def _generate_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on performance analysis"""
        recommendations = []
        
        accuracy_metrics = report.get('accuracy_metrics', {})
        betting_metrics = report.get('betting_metrics', {})
        calibration_analysis = report.get('calibration_analysis', {})
        
        # Accuracy-based recommendations
        accuracy = accuracy_metrics.get('accuracy', 0.0)
        if accuracy < 0.60:
            recommendations.append("Consider feature engineering improvements to increase accuracy")
            recommendations.append("Evaluate model ensemble weights and hyperparameters")
        
        # Betting performance recommendations
        if betting_metrics:
            model_vs_vegas = betting_metrics.get('model_vs_vegas_improvement', 0.0)
            if model_vs_vegas < 0.0:
                recommendations.append("Model underperforms Vegas - consider incorporating Vegas lines as features")
            
            upset_f1 = betting_metrics.get('upset_f1', 0.0)
            if upset_f1 < 0.4:
                recommendations.append("Improve upset detection - focus on underdog prediction features")
        
        # Calibration recommendations
        ece = calibration_analysis.get('expected_calibration_error', 1.0)
        if ece > 0.10:
            recommendations.append("Implement calibration techniques (Platt scaling, isotonic regression)")
        
        # General recommendations
        if not recommendations:
            recommendations.append("Model performance is satisfactory - consider production deployment")
        
        return recommendations
    
    def save_evaluation_results(self, filepath: str, results: Dict[str, Any]) -> None:
        """Save evaluation results to file"""
        try:
            import json
            
            # Convert numpy arrays to lists for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                return obj
            
            # Recursively convert numpy objects
            def recursive_convert(obj):
                if isinstance(obj, dict):
                    return {k: recursive_convert(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [recursive_convert(item) for item in obj]
                else:
                    return convert_numpy(obj)
            
            converted_results = recursive_convert(results)
            
            with open(filepath, 'w') as f:
                json.dump(converted_results, f, indent=2)
            
            logger.info(f"Evaluation results saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving evaluation results: {e}")
            raise
