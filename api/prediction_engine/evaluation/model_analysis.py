"""
Model analyzer for NFL prediction engine.

This module provides detailed analysis of model behavior,
feature importance, and prediction patterns.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class ModelAnalyzer:
    """Analyzes model behavior and prediction patterns"""
    
    def __init__(self):
        self.analysis_results = {}
        
        logger.info("Initialized ModelAnalyzer")
    
    def analyze_feature_importance(
        self, 
        feature_importance: Dict[str, np.ndarray],
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """
        Analyze feature importance across different models.
        
        Args:
            feature_importance: Dictionary of feature importance arrays
            feature_names: List of feature names
            
        Returns:
            Feature importance analysis
        """
        logger.info("Analyzing feature importance...")
        
        analysis = {}
        
        # Create feature importance DataFrame
        importance_df = pd.DataFrame(feature_importance, index=feature_names)
        
        # Calculate average importance across models
        importance_df['average_importance'] = importance_df.mean(axis=1)
        importance_df['std_importance'] = importance_df.std(axis=1)
        
        # Sort by average importance
        importance_df = importance_df.sort_values('average_importance', ascending=False)
        
        # Top features
        top_features = importance_df.head(10)
        
        # Feature categories (if we can identify them)
        feature_categories = self._categorize_features(feature_names)
        
        # Category importance
        category_importance = {}
        for category, features in feature_categories.items():
            category_features = [f for f in features if f in importance_df.index]
            if category_features:
                category_importance[category] = importance_df.loc[category_features, 'average_importance'].mean()
        
        analysis = {
            'feature_importance_df': importance_df,
            'top_features': top_features.to_dict(),
            'feature_categories': feature_categories,
            'category_importance': category_importance,
            'total_features': len(feature_names),
            'high_importance_features': len(importance_df[importance_df['average_importance'] > 0.01])
        }
        
        logger.info(f"Analyzed {len(feature_names)} features")
        logger.info(f"Top 5 features: {list(top_features.index[:5])}")
        
        return analysis
    
    def analyze_prediction_patterns(
        self, 
        predictions: np.ndarray, 
        probabilities: np.ndarray,
        features: pd.DataFrame,
        actuals: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyze prediction patterns and model behavior.
        
        Args:
            predictions: Model predictions
            probabilities: Prediction probabilities
            features: Feature matrix
            actuals: Actual outcomes
            
        Returns:
            Prediction pattern analysis
        """
        logger.info("Analyzing prediction patterns...")
        
        analysis = {}
        
        # Confidence distribution
        confidence_stats = {
            'mean_confidence': np.mean(probabilities),
            'std_confidence': np.std(probabilities),
            'min_confidence': np.min(probabilities),
            'max_confidence': np.max(probabilities),
            'high_confidence_predictions': np.sum(probabilities > 0.8),
            'low_confidence_predictions': np.sum(probabilities < 0.2)
        }
        
        # Prediction accuracy by confidence level
        confidence_bins = np.linspace(0, 1, 11)
        accuracy_by_confidence = []
        
        for i in range(len(confidence_bins) - 1):
            bin_lower = confidence_bins[i]
            bin_upper = confidence_bins[i + 1]
            
            mask = (probabilities >= bin_lower) & (probabilities < bin_upper)
            if mask.sum() > 0:
                bin_accuracy = accuracy_score(actuals[mask], predictions[mask])
                accuracy_by_confidence.append({
                    'confidence_range': f"{bin_lower:.1f}-{bin_upper:.1f}",
                    'accuracy': bin_accuracy,
                    'count': mask.sum()
                })
        
        # Error analysis
        errors = predictions != actuals
        error_analysis = {
            'total_errors': errors.sum(),
            'error_rate': errors.mean(),
            'false_positives': ((predictions == 1) & (actuals == 0)).sum(),
            'false_negatives': ((predictions == 0) & (actuals == 1)).sum()
        }
        
        # Feature patterns for correct vs incorrect predictions
        correct_mask = predictions == actuals
        if correct_mask.sum() > 0 and (~correct_mask).sum() > 0:
            correct_features = features[correct_mask].mean()
            incorrect_features = features[~correct_mask].mean()
            
            feature_differences = (correct_features - incorrect_features).abs().sort_values(ascending=False)
            top_differentiating_features = feature_differences.head(10)
        else:
            top_differentiating_features = pd.Series()
        
        analysis = {
            'confidence_stats': confidence_stats,
            'accuracy_by_confidence': accuracy_by_confidence,
            'error_analysis': error_analysis,
            'top_differentiating_features': top_differentiating_features.to_dict(),
            'total_predictions': len(predictions)
        }
        
        logger.info(f"Analyzed {len(predictions)} predictions")
        logger.info(f"Mean confidence: {confidence_stats['mean_confidence']:.3f}")
        logger.info(f"Error rate: {error_analysis['error_rate']:.3f}")
        
        return analysis
    
    def analyze_seasonal_trends(
        self, 
        predictions: np.ndarray, 
        actuals: np.ndarray,
        seasons: np.ndarray,
        weeks: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyze seasonal and weekly trends in model performance.
        
        Args:
            predictions: Model predictions
            actuals: Actual outcomes
            seasons: Season for each prediction
            weeks: Week for each prediction
            
        Returns:
            Seasonal trend analysis
        """
        logger.info("Analyzing seasonal trends...")
        
        analysis = {}
        
        # Performance by season
        seasonal_performance = {}
        for season in np.unique(seasons):
            season_mask = seasons == season
            if season_mask.sum() > 0:
                season_accuracy = accuracy_score(actuals[season_mask], predictions[season_mask])
                seasonal_performance[int(season)] = {
                    'accuracy': season_accuracy,
                    'games': season_mask.sum()
                }
        
        # Performance by week
        weekly_performance = {}
        for week in np.unique(weeks):
            week_mask = weeks == week
            if week_mask.sum() > 0:
                week_accuracy = accuracy_score(actuals[week_mask], predictions[week_mask])
                weekly_performance[int(week)] = {
                    'accuracy': week_accuracy,
                    'games': week_mask.sum()
                }
        
        # Early vs late season analysis
        early_season_mask = weeks <= 8
        late_season_mask = weeks > 8
        
        early_accuracy = accuracy_score(actuals[early_season_mask], predictions[early_season_mask]) if early_season_mask.sum() > 0 else 0.0
        late_accuracy = accuracy_score(actuals[late_season_mask], predictions[late_season_mask]) if late_season_mask.sum() > 0 else 0.0
        
        # Playoff vs regular season (if applicable)
        playoff_mask = weeks > 17
        regular_mask = weeks <= 17
        
        playoff_accuracy = accuracy_score(actuals[playoff_mask], predictions[playoff_mask]) if playoff_mask.sum() > 0 else 0.0
        regular_accuracy = accuracy_score(actuals[regular_mask], predictions[regular_mask]) if regular_mask.sum() > 0 else 0.0
        
        analysis = {
            'seasonal_performance': seasonal_performance,
            'weekly_performance': weekly_performance,
            'early_vs_late_season': {
                'early_season_accuracy': early_accuracy,
                'late_season_accuracy': late_accuracy,
                'improvement': late_accuracy - early_accuracy
            },
            'playoff_vs_regular': {
                'playoff_accuracy': playoff_accuracy,
                'regular_season_accuracy': regular_accuracy,
                'difference': playoff_accuracy - regular_accuracy
            }
        }
        
        logger.info(f"Analyzed {len(np.unique(seasons))} seasons and {len(np.unique(weeks))} weeks")
        
        return analysis
    
    def analyze_team_performance(
        self, 
        predictions: np.ndarray, 
        actuals: np.ndarray,
        home_teams: np.ndarray,
        away_teams: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyze model performance by team.
        
        Args:
            predictions: Model predictions
            actuals: Actual outcomes
            home_teams: Home team for each game
            away_teams: Away team for each game
            
        Returns:
            Team performance analysis
        """
        logger.info("Analyzing team performance...")
        
        analysis = {}
        
        # Performance when predicting home team wins
        home_win_predictions = predictions == 1
        home_win_accuracy = accuracy_score(actuals[home_win_predictions], predictions[home_win_predictions]) if home_win_predictions.sum() > 0 else 0.0
        
        # Performance when predicting away team wins
        away_win_predictions = predictions == 0
        away_win_accuracy = accuracy_score(actuals[away_win_predictions], predictions[away_win_predictions]) if away_win_predictions.sum() > 0 else 0.0
        
        # Team-specific performance
        all_teams = np.unique(np.concatenate([home_teams, away_teams]))
        team_performance = {}
        
        for team in all_teams:
            # Games where this team is home
            home_games = home_teams == team
            # Games where this team is away
            away_games = away_teams == team
            
            team_accuracy = 0.0
            team_games = 0
            
            if home_games.sum() > 0:
                home_actuals = actuals[home_games]
                home_predictions = predictions[home_games]
                home_accuracy = accuracy_score(home_actuals, home_predictions)
                team_accuracy += home_accuracy * home_games.sum()
                team_games += home_games.sum()
            
            if away_games.sum() > 0:
                away_actuals = actuals[away_games]
                away_predictions = predictions[away_games]
                # For away games, we need to flip the prediction logic
                away_predictions_flipped = 1 - away_predictions
                away_accuracy = accuracy_score(away_actuals, away_predictions_flipped)
                team_accuracy += away_accuracy * away_games.sum()
                team_games += away_games.sum()
            
            if team_games > 0:
                team_performance[team] = {
                    'accuracy': team_accuracy / team_games,
                    'total_games': team_games
                }
        
        # Sort teams by accuracy
        sorted_teams = sorted(team_performance.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        
        analysis = {
            'home_win_accuracy': home_win_accuracy,
            'away_win_accuracy': away_win_accuracy,
            'team_performance': dict(sorted_teams),
            'best_predicted_teams': [team for team, _ in sorted_teams[:5]],
            'worst_predicted_teams': [team for team, _ in sorted_teams[-5:]]
        }
        
        logger.info(f"Analyzed performance for {len(all_teams)} teams")
        logger.info(f"Home win accuracy: {home_win_accuracy:.3f}")
        logger.info(f"Away win accuracy: {away_win_accuracy:.3f}")
        
        return analysis
    
    def generate_model_insights(
        self, 
        feature_analysis: Dict[str, Any],
        pattern_analysis: Dict[str, Any],
        seasonal_analysis: Dict[str, Any],
        team_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate comprehensive model insights and recommendations.
        
        Args:
            feature_analysis: Feature importance analysis
            pattern_analysis: Prediction pattern analysis
            seasonal_analysis: Seasonal trend analysis
            team_analysis: Team performance analysis
            
        Returns:
            Comprehensive model insights
        """
        logger.info("Generating model insights...")
        
        insights = {
            'insights_timestamp': datetime.now().isoformat(),
            'feature_insights': self._generate_feature_insights(feature_analysis),
            'pattern_insights': self._generate_pattern_insights(pattern_analysis),
            'seasonal_insights': self._generate_seasonal_insights(seasonal_analysis),
            'team_insights': self._generate_team_insights(team_analysis),
            'overall_recommendations': []
        }
        
        # Generate overall recommendations
        recommendations = []
        
        # Feature-based recommendations
        if feature_analysis.get('high_importance_features', 0) < 5:
            recommendations.append("Consider adding more predictive features")
        
        # Pattern-based recommendations
        if pattern_analysis.get('error_analysis', {}).get('error_rate', 0) > 0.4:
            recommendations.append("High error rate - consider model retraining or feature engineering")
        
        # Seasonal recommendations
        seasonal_improvement = seasonal_analysis.get('early_vs_late_season', {}).get('improvement', 0)
        if seasonal_improvement > 0.05:
            recommendations.append("Model improves significantly during late season - consider seasonal adjustments")
        
        # Team-based recommendations
        team_accuracies = [perf['accuracy'] for perf in team_analysis.get('team_performance', {}).values()]
        if team_accuracies and np.std(team_accuracies) > 0.1:
            recommendations.append("High variance in team prediction accuracy - consider team-specific features")
        
        insights['overall_recommendations'] = recommendations
        
        logger.info("Model insights generated")
        
        return insights
    
    def _categorize_features(self, feature_names: List[str]) -> Dict[str, List[str]]:
        """Categorize features by type"""
        categories = {
            'game_context': [],
            'home_team': [],
            'away_team': [],
            'head_to_head': [],
            'vegas': [],
            'recent_form': [],
            'rest': []
        }
        
        for feature in feature_names:
            if any(keyword in feature.lower() for keyword in ['season', 'week', 'playoff']):
                categories['game_context'].append(feature)
            elif feature.startswith('home_'):
                categories['home_team'].append(feature)
            elif feature.startswith('away_'):
                categories['away_team'].append(feature)
            elif 'h2h' in feature.lower():
                categories['head_to_head'].append(feature)
            elif any(keyword in feature.lower() for keyword in ['spread', 'total', 'vegas', 'favorite']):
                categories['vegas'].append(feature)
            elif 'recent' in feature.lower() or 'form' in feature.lower():
                categories['recent_form'].append(feature)
            elif 'rest' in feature.lower():
                categories['rest'].append(feature)
            else:
                categories['game_context'].append(feature)
        
        return {k: v for k, v in categories.items() if v}
    
    def _generate_feature_insights(self, feature_analysis: Dict[str, Any]) -> List[str]:
        """Generate insights from feature analysis"""
        insights = []
        
        top_features = feature_analysis.get('top_features', {})
        if top_features:
            top_5 = list(top_features.get('average_importance', {}).keys())[:5]
            insights.append(f"Top 5 most important features: {', '.join(top_5)}")
        
        category_importance = feature_analysis.get('category_importance', {})
        if category_importance:
            most_important_category = max(category_importance.items(), key=lambda x: x[1])
            insights.append(f"Most important feature category: {most_important_category[0]}")
        
        return insights
    
    def _generate_pattern_insights(self, pattern_analysis: Dict[str, Any]) -> List[str]:
        """Generate insights from pattern analysis"""
        insights = []
        
        confidence_stats = pattern_analysis.get('confidence_stats', {})
        if confidence_stats:
            mean_conf = confidence_stats.get('mean_confidence', 0)
            if mean_conf > 0.7:
                insights.append("Model tends to make high-confidence predictions")
            elif mean_conf < 0.5:
                insights.append("Model tends to make low-confidence predictions")
        
        error_analysis = pattern_analysis.get('error_analysis', {})
        if error_analysis:
            error_rate = error_analysis.get('error_rate', 0)
            if error_rate > 0.4:
                insights.append("High error rate suggests model needs improvement")
            elif error_rate < 0.3:
                insights.append("Low error rate indicates strong model performance")
        
        return insights
    
    def _generate_seasonal_insights(self, seasonal_analysis: Dict[str, Any]) -> List[str]:
        """Generate insights from seasonal analysis"""
        insights = []
        
        early_vs_late = seasonal_analysis.get('early_vs_late_season', {})
        if early_vs_late:
            improvement = early_vs_late.get('improvement', 0)
            if improvement > 0.05:
                insights.append("Model performance improves significantly in late season")
            elif improvement < -0.05:
                insights.append("Model performance declines in late season")
        
        return insights
    
    def _generate_team_insights(self, team_analysis: Dict[str, Any]) -> List[str]:
        """Generate insights from team analysis"""
        insights = []
        
        best_teams = team_analysis.get('best_predicted_teams', [])
        worst_teams = team_analysis.get('worst_predicted_teams', [])
        
        if best_teams:
            insights.append(f"Best predicted teams: {', '.join(best_teams[:3])}")
        
        if worst_teams:
            insights.append(f"Worst predicted teams: {', '.join(worst_teams[:3])}")
        
        return insights
