import React, { useState, useEffect } from 'react';
import { ArrowLeft, Trophy, Target, BarChart3, Calendar, Clock, Users } from 'lucide-react';
import PredictionCard from '../components/PredictionCard';
import Header from '../components/Header';
import { getApiUrl } from '../utils/simpleApiConfig';
import { getCurrentNFLWeek, getWeekForSection, getWeekDisplayName } from '../utils/weekUtils';

interface GameResult {
  game_id: string;
  away_team: string;
  home_team: string;
  game_date: string;
  game_time: string;
  week: number;
  season: number;
  away_score?: number;
  home_score?: number;
  actual_result: {
    home_score: number;
    away_score: number;
    winner: string;
    final_score: string;
  };
  ai_prediction: {
    predicted_winner: string;
    confidence: number;
    predicted_score: string;
    key_factors: string[];
    upset_potential: number;
    ai_analysis: string;
    is_upset: boolean;
    model_accuracy: number;
  };
  prediction_accuracy: {
    correct: boolean;
    status: string;
  };
}

interface PreviousWeekResponse {
  games: GameResult[];
  season: number;
  week: number;
  total_games: number;
  prediction_accuracy: {
    correct: number;
    total: number;
    percentage: number;
  };
  is_previous_week: boolean;
}

const PreviousWeek: React.FC = () => {
  const [games, setGames] = useState<GameResult[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [predictionStats, setPredictionStats] = useState<PreviousWeekResponse['prediction_accuracy'] | null>(null);
  const [season, setSeason] = useState(2025);
  const [week, setWeek] = useState(1);
  const [userPicks, setUserPicks] = useState<{[gameId: string]: string}>({});
  const [userPredictionStats, setUserPredictionStats] = useState<{
    total: number;
    correct: number;
    percentage: number;
  } | null>(null);

  // Utility functions for localStorage operations
  const getLocalStorageKey = (season: number, week: number) => `userPicks_${season}_${week}`;
  
  const loadUserPicksFromLocalStorage = (season: number, week: number): {[gameId: string]: string} => {
    try {
      const key = getLocalStorageKey(season, week);
      const savedPicks = localStorage.getItem(key);
      return savedPicks ? JSON.parse(savedPicks) : {};
    } catch (error) {
      console.warn('Error loading user picks from localStorage:', error);
      return {};
    }
  };

  const calculateUserPredictionStats = (games: GameResult[], picks: {[gameId: string]: string}) => {
    const gamesWithPicks = games.filter(game => picks[game.game_id]);
    const correctPicks = gamesWithPicks.filter(game => {
      const userPick = picks[game.game_id];
      const actualWinner = game.actual_result?.winner;
      return userPick === actualWinner;
    });

    const total = gamesWithPicks.length;
    const correct = correctPicks.length;
    const percentage = total > 0 ? Math.round((correct / total) * 100) : 0;

    return { total, correct, percentage };
  };

  useEffect(() => {
    fetchPreviousWeekData();
  }, []);

  const fetchPreviousWeekData = async () => {
    try {
      setLoading(true);
      setError(null);
      
      // Get dynamic week info
      const weekInfo = getCurrentNFLWeek();
      const previousWeek = getWeekForSection('previous');
      
      // Update state with dynamic week info
      setSeason(weekInfo.season);
      setWeek(previousWeek);
      
      // Load user picks from localStorage for this week
      const savedPicks = loadUserPicksFromLocalStorage(weekInfo.season, previousWeek);
      setUserPicks(savedPicks);
      console.log(`📱 Loaded ${Object.keys(savedPicks).length} user picks from localStorage for ${weekInfo.season} ${getWeekDisplayName(previousWeek)}`);
      
      const response = await fetch(`${getApiUrl()}/api/games/previous?season=${weekInfo.season}&week=${previousWeek}`);
      
      if (!response.ok) {
        throw new Error('Failed to fetch previous week data');
      }
      
      const data: PreviousWeekResponse = await response.json();
      setGames(data.games);
      setPredictionStats(data.prediction_accuracy);
      
      // Calculate user prediction stats
      const userStats = calculateUserPredictionStats(data.games, savedPicks);
      setUserPredictionStats(userStats);
      console.log(`👤 User prediction stats: ${userStats.correct}/${userStats.total} (${userStats.percentage}%)`);
      
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 py-8">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
            <p className="text-gray-600 text-lg">Loading Previous Week Results...</p>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-50 py-8">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center">
            <p className="text-red-600 text-lg mb-4">Error: {error}</p>
            <button 
              onClick={fetchPreviousWeekData}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
            >
              Try Again
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <Header 
        title="Gridiron Guru"
      />
      
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Back Button and Page Info */}
        <div className="mb-8">
          <button 
            onClick={() => window.history.back()} 
            className="flex items-center text-blue-600 hover:text-blue-800 font-medium mb-6"
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back
          </button>
          
          <div className="text-center mb-8">
            <div className="flex items-center justify-center mb-4">
              <Trophy className="h-12 w-12 text-yellow-500 mr-3" />
              <h1 className="text-4xl font-bold text-gray-900 font-rubik-dirt">
                Previous Week Results
              </h1>
            </div>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Week {week}, {season}: AI Predictions vs Actual Results
            </p>
          </div>
        </div>

        {/* Prediction Accuracy Summary */}
        {predictionStats && (
          <div className="bg-white rounded-lg shadow-lg p-6 border border-gray-100 mb-8">
            <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center">
              <Target className="w-6 h-6 mr-2 text-blue-600" />
              AI Prediction Accuracy Summary
            </h2>
            <div className="grid md:grid-cols-3 gap-6">
              <div className="text-center">
                <div className="text-4xl font-bold text-blue-600 mb-2">
                  {predictionStats.percentage}%
                </div>
                <div className="text-gray-600 text-sm">Overall Accuracy</div>
              </div>
              <div className="text-center">
                <div className="text-4xl font-bold text-green-600 mb-2">
                  {predictionStats.correct}
                </div>
                <div className="text-gray-600 text-sm">Correct Predictions</div>
              </div>
              <div className="text-center">
                <div className="text-4xl font-bold text-red-600 mb-2">
                  {predictionStats.total - predictionStats.correct}
                </div>
                <div className="text-gray-600 text-sm">Incorrect Predictions</div>
              </div>
            </div>
          </div>
        )}

        {/* User Prediction Accuracy Summary */}
        {userPredictionStats && userPredictionStats.total > 0 && (
          <div className="bg-gradient-to-r from-purple-50 to-pink-50 rounded-lg shadow-lg p-6 border border-purple-200 mb-8">
            <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center">
              <Users className="w-6 h-6 mr-2 text-purple-600" />
              Your Prediction Accuracy
            </h2>
            <div className="grid md:grid-cols-3 gap-6">
              <div className="text-center">
                <div className="text-4xl font-bold text-purple-600 mb-2">
                  {userPredictionStats.percentage}%
                </div>
                <div className="text-gray-600 text-sm">Your Accuracy</div>
              </div>
              <div className="text-center">
                <div className="text-4xl font-bold text-green-600 mb-2">
                  {userPredictionStats.correct}
                </div>
                <div className="text-gray-600 text-sm">Correct Picks</div>
              </div>
              <div className="text-center">
                <div className="text-4xl font-bold text-red-600 mb-2">
                  {userPredictionStats.total - userPredictionStats.correct}
                </div>
                <div className="text-gray-600 text-sm">Incorrect Picks</div>
              </div>
            </div>
          </div>
        )}

        {/* Games Grid */}
        <div className="space-y-8">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {games.map((game, index) => (
              <div
                key={game.game_id || index}
                className="cursor-pointer"
              >
                <PredictionCard 
                  game={{
                    ...game,
                    away_score: game.actual_result?.away_score,
                    home_score: game.actual_result?.home_score,
                  }} 
                  showScores={true}
                  userPick={userPicks[game.game_id]}
                />
              </div>
            ))}
          </div>
        </div>

        {games.length === 0 && (
          <div className="bg-white rounded-lg shadow-lg p-12 text-center">
            <p className="text-gray-600 text-lg">No games found for Week {week}, {season}</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default PreviousWeek;