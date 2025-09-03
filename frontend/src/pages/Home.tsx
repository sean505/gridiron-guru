import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { Trophy, TrendingUp, Database, Brain, Users, ArrowRight, Target, AlertTriangle } from 'lucide-react';
import { nflApi } from '../api/nflApi';

export default function HomePage() {
  const [featuredPredictions, setFeaturedPredictions] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchFeaturedPredictions = async () => {
      try {
        // Fetch current week's games
        const gamesResponse = await nflApi.getGames();
        const games = gamesResponse.games || [];
        
        // Take first 3 games and create predictions
        const predictions = games.slice(0, 3).map((game, index) => {
          // Use real ML predictions from the API if available
          if (game.ai_prediction) {
            return {
              id: index + 1,
              away_team: game.away_team,
              home_team: game.home_team,
              ai_pick: game.ai_prediction.predicted_winner,
              confidence: game.ai_prediction.confidence,
              upset_potential: game.ai_prediction.upset_potential,
              is_upset: game.ai_prediction.is_upset,
              game_date: game.game_date,
              game_time: game.game_time
            };
          }
          
          // If no API prediction data, skip this game (don't use hardcoded fallbacks)
          console.warn(`No AI prediction data for featured game: ${game.away_team} @ ${game.home_team}`);
          return null;
        }).filter(prediction => prediction !== null);
        
        setFeaturedPredictions(predictions);
      } catch (error) {
        console.error('Error fetching games:', error);
        // No fallback data - only use live API data
        setFeaturedPredictions([]);
      } finally {
        setLoading(false);
      }
    };

    fetchFeaturedPredictions();
  }, []);

  const getTeamName = (teamCode: string) => {
    const teamNames: { [key: string]: string } = {
      'BUF': 'Buffalo Bills', 'MIA': 'Miami Dolphins', 'NE': 'New England Patriots', 'NYJ': 'New York Jets',
      'BAL': 'Baltimore Ravens', 'CIN': 'Cincinnati Bengals', 'CLE': 'Cleveland Browns', 'PIT': 'Pittsburgh Steelers',
      'HOU': 'Houston Texans', 'IND': 'Indianapolis Colts', 'JAX': 'Jacksonville Jaguars', 'TEN': 'Tennessee Titans',
      'DEN': 'Denver Broncos', 'KC': 'Kansas City Chiefs', 'LV': 'Las Vegas Raiders', 'LAC': 'Los Angeles Chargers',
      'DAL': 'Dallas Cowboys', 'NYG': 'New York Giants', 'PHI': 'Philadelphia Eagles', 'WAS': 'Washington Commanders',
      'CHI': 'Chicago Bears', 'DET': 'Detroit Lions', 'GB': 'Green Bay Packers', 'MIN': 'Minnesota Vikings',
      'ATL': 'Atlanta Falcons', 'CAR': 'Carolina Panthers', 'NO': 'New Orleans Saints', 'TB': 'Tampa Bay Buccaneers',
      'ARI': 'Arizona Cardinals', 'LAR': 'Los Angeles Rams', 'SF': 'San Francisco 49ers', 'SEA': 'Seattle Seahawks'
    };
    return teamNames[teamCode] || teamCode;
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 80) return 'text-green-600 bg-green-100';
    if (confidence >= 65) return 'text-blue-600 bg-blue-100';
    if (confidence >= 50) return 'text-yellow-600 bg-yellow-100';
    return 'text-red-600 bg-red-100';
  };

  const getUpsetPotentialColor = (potential: number) => {
    if (potential >= 35) return 'text-red-600 bg-red-100';
    if (potential >= 25) return 'text-yellow-600 bg-yellow-100';
    return 'text-green-600 bg-green-100';
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Hero Section */}
      <div className="bg-gradient-to-br from-blue-600 via-blue-700 to-blue-800 text-white">
        <div className="max-w-7xl mx-auto px-6 py-20">
          <div className="text-center">
            <div className="flex justify-center mb-6">
              <Trophy className="w-20 h-20 text-yellow-400" />
            </div>
            <h1 className="text-5xl md:text-6xl font-bold mb-6">
              Gridiron Guru
            </h1>
            <p className="text-xl md:text-2xl mb-8 text-blue-100 max-w-3xl mx-auto">
              AI-powered NFL predictions backed by comprehensive data analysis from the most reliable sources in football analytics.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link
                to="/predictor"
                className="bg-yellow-500 text-blue-900 px-8 py-4 rounded-lg font-bold text-lg hover:bg-yellow-400 transition-colors shadow-lg hover:shadow-xl"
              >
                View AI Predictions
              </Link>
              <Link
                to="/predictor"
                className="border-2 border-white text-white px-8 py-4 rounded-lg font-bold text-lg hover:bg-white hover:text-blue-900 transition-colors"
              >
                Make Your Picks
              </Link>
              <Link
                to="/previous-week"
                className="bg-green-600 text-white px-8 py-4 rounded-lg font-bold text-lg hover:bg-green-700 transition-colors shadow-lg hover:shadow-xl"
              >
                Previous Week Results
              </Link>
            </div>
          </div>
        </div>
      </div>

      {/* Featured AI Predictions */}
      <div className="max-w-7xl mx-auto px-6 py-16">
        <div className="text-center mb-12">
          <h2 className="text-3xl font-bold text-gray-900 mb-4">
            This Week's AI Predictions
          </h2>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Our AI analyzes thousands of data points to predict game outcomes with remarkable accuracy.
          </p>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          {loading ? (
            // Loading skeleton
            Array.from({ length: 3 }).map((_, index) => (
              <div key={index} className="bg-white rounded-lg shadow-lg p-6 border border-gray-100 animate-pulse">
                <div className="h-4 bg-gray-200 rounded mb-3"></div>
                <div className="h-6 bg-gray-200 rounded mb-4"></div>
                <div className="h-4 bg-gray-200 rounded mb-4"></div>
                <div className="flex justify-center space-x-3 mb-4">
                  <div className="h-6 bg-gray-200 rounded w-24"></div>
                  <div className="h-6 bg-gray-200 rounded w-24"></div>
                </div>
                <div className="h-4 bg-gray-200 rounded"></div>
              </div>
            ))
          ) : (
            featuredPredictions.map((prediction) => (
            <div
              key={prediction.id}
              className={`bg-white rounded-lg shadow-lg p-6 border ${
                prediction.is_upset ? 'border-red-200 bg-red-50' : 'border-gray-100'
              }`}
            >
              {prediction.is_upset && (
                <div className="flex items-center justify-center mb-3">
                  <AlertTriangle className="w-4 h-4 text-red-500 mr-2" />
                  <span className="text-sm font-medium text-red-700 bg-red-200 px-2 py-1 rounded-full">
                    UPSET PICK
                  </span>
                </div>
              )}
              
              <h3 className="text-lg font-semibold text-gray-800 mb-3 text-center">
                {getTeamName(prediction.away_team)} @ {getTeamName(prediction.home_team)}
              </h3>
              
              <div className="text-center mb-4">
                <span className="text-sm text-gray-600">AI Pick: </span>
                <span className="font-bold text-blue-700">{getTeamName(prediction.ai_pick)}</span>
              </div>
              
              <div className="flex justify-center space-x-3 text-sm mb-4">
                <span className={`px-2 py-1 rounded-full ${getConfidenceColor(prediction.confidence)}`}>
                  {prediction.confidence}% confidence
                </span>
                <span className={`px-2 py-1 rounded-full ${getUpsetPotentialColor(prediction.upset_potential)}`}>
                  {prediction.upset_potential}% upset potential
                </span>
              </div>
              
              <div className="text-center">
                <Link
                  to="/predictor"
                  className="inline-flex items-center text-blue-600 hover:text-blue-800 font-medium text-sm"
                >
                  View Full Analysis
                  <ArrowRight className="w-4 h-4 ml-1" />
                </Link>
              </div>
            </div>
            ))
          )}
        </div>
        
        <div className="text-center">
          <Link
            to="/predictor"
            className="inline-flex items-center bg-blue-600 text-white px-6 py-3 rounded-lg font-medium hover:bg-blue-700 transition-colors shadow-md hover:shadow-lg"
          >
            See All Predictions
            <ArrowRight className="w-4 h-4 ml-2" />
          </Link>
        </div>
      </div>

      {/* Features Section */}
      <div className="bg-white py-16">
        <div className="max-w-7xl mx-auto px-6">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">
              Why Choose Gridiron Guru?
            </h2>
            <p className="text-lg text-gray-600 max-w-2xl mx-auto">
              We combine cutting-edge AI with the most comprehensive NFL data sources to deliver predictions you can trust.
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            <div className="text-center">
              <div className="bg-blue-100 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
                <Brain className="w-8 h-8 text-blue-600" />
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-2">AI-Powered Analysis</h3>
              <p className="text-gray-600">
                Our advanced AI analyzes thousands of data points including player stats, team performance, weather, and historical matchups.
              </p>
            </div>
            
            <div className="text-center">
              <div className="bg-green-100 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
                <Database className="w-8 h-8 text-green-600" />
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-2">Comprehensive Data</h3>
              <p className="text-gray-600">
                Access to nflfastR, nfldata, DynastyProcess, and Draft Scout data for the most complete analysis available.
              </p>
            </div>
            
            <div className="text-center">
              <div className="bg-yellow-100 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
                <TrendingUp className="w-8 h-8 text-yellow-600" />
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-2">Proven Accuracy</h3>
              <p className="text-gray-600">
                Track record of successful predictions with detailed confidence levels and upset potential analysis.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Data Sources Section */}
      <div className="bg-gray-100 py-16">
        <div className="max-w-7xl mx-auto px-6">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">
              Powered by Industry-Leading Data Sources
            </h2>
            <p className="text-lg text-gray-600 max-w-2xl mx-auto">
              We integrate data from the most trusted names in football analytics to ensure our predictions are based on the best available information.
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
            <div className="text-center">
              <div className="bg-white rounded-lg p-6 shadow-md">
                <h3 className="text-lg font-semibold text-gray-900 mb-2">nflfastR</h3>
                <p className="text-gray-600 text-sm">
                  Play-by-play data and advanced statistics for comprehensive game analysis.
                </p>
              </div>
            </div>
            
            <div className="text-center">
              <div className="bg-white rounded-lg p-6 shadow-md">
                <h3 className="text-lg font-semibold text-gray-900 mb-2">nfldata</h3>
                <p className="text-gray-600 text-sm">
                  Official NFL statistics and historical performance data.
                </p>
              </div>
            </div>
            
            <div className="text-center">
              <div className="bg-white rounded-lg p-6 shadow-md">
                <h3 className="text-lg font-semibold text-gray-900 mb-2">DynastyProcess</h3>
                <p className="text-gray-600 text-sm">
                  Player performance metrics and dynasty fantasy football insights.
                </p>
              </div>
            </div>
            
            <div className="text-center">
              <div className="bg-white rounded-lg p-6 shadow-md">
                <h3 className="text-lg font-semibold text-gray-900 mb-2">Draft Scout</h3>
                <p className="text-gray-600 text-sm">
                  College player evaluation and draft prospect analysis.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* CTA Section */}
      <div className="bg-blue-600 text-white py-16">
        <div className="max-w-4xl mx-auto text-center px-6">
          <h2 className="text-3xl font-bold mb-4">
            Ready to Get Started?
          </h2>
          <p className="text-xl mb-8 text-blue-100">
            Join thousands of users who trust Gridiron Guru for their NFL predictions.
          </p>
          <Link
            to="/predictor"
            className="bg-yellow-500 text-blue-900 px-8 py-4 rounded-lg font-bold text-lg hover:bg-yellow-400 transition-colors shadow-lg hover:shadow-xl inline-block"
          >
            Start Predicting Now
          </Link>
        </div>
      </div>

      {/* Footer */}
      <footer className="bg-gray-900 text-white py-12">
        <div className="max-w-7xl mx-auto px-6 text-center">
          <div className="flex justify-center mb-6">
            <Trophy className="w-12 h-12 text-yellow-400" />
          </div>
          <h3 className="text-2xl font-bold mb-4">Gridiron Guru</h3>
          <p className="text-gray-400 mb-6">
            AI-powered NFL predictions for the modern football fan.
          </p>
          <div className="text-gray-500 text-sm">
            © 2024 Gridiron Guru. All rights reserved.
          </div>
        </div>
      </footer>
    </div>
  );
}
