import { useState, useEffect } from 'react';
import { Game, UserPrediction } from '@/shared/types';
import GameCard from './GameCard';
import { Lock, Trophy, Target, Brain, TrendingUp, Loader2 } from 'lucide-react';

export default function WeeklyPredictor() {
  const [games, setGames] = useState<Game[]>([]);
  const [userPredictions, setUserPredictions] = useState<UserPrediction[]>([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [userId] = useState(() => {
    // Use demo user for testing with dummy data
    let id = localStorage.getItem('user_id');
    if (!id) {
      id = 'user_demo123'; // Use demo user to see dummy predictions
      localStorage.setItem('user_id', id);
    }
    return id;
  });

  const fetchGames = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/games/current-week');
      const data = await response.json();
      
      if (data.error) {
        setError(data.error);
      } else {
        setGames(data.games || []);
      }
    } catch (err) {
      setError('Failed to load games');
    } finally {
      setLoading(false);
    }
  };

  const fetchUserPredictions = async () => {
    try {
      const response = await fetch(`/api/predictions/${userId}`);
      const data = await response.json();
      setUserPredictions(data.predictions || []);
    } catch (err) {
      console.error('Failed to load user predictions:', err);
    }
  };

  useEffect(() => {
    fetchGames();
    fetchUserPredictions();
  }, [userId]);

  const handlePredictionChange = async (gameId: string, winner: string) => {
    try {
      setSaving(true);
      const response = await fetch('/api/predictions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          gameId,
          predictedWinner: winner,
          userId,
        }),
      });

      if (response.ok) {
        // Update local state
        setUserPredictions(prev => {
          const existing = prev.find(p => p.game_id === gameId);
          if (existing) {
            if (winner === '') {
              // If deselecting, remove the prediction from local state
              return prev.filter(p => p.game_id !== gameId);
            } else {
              // Update existing prediction
              return prev.map(p => 
                p.game_id === gameId 
                  ? { ...p, predicted_winner: winner }
                  : p
              );
            }
          } else if (winner !== '') {
            // Only create a new prediction if winner is not empty
            const newPrediction: UserPrediction = {
              id: Date.now(), // temporary ID
              user_id: userId,
              game_id: gameId,
              predicted_winner: winner,
              is_upset_pick: false,
              created_at: new Date().toISOString(),
              updated_at: new Date().toISOString(),
            };
            return [...prev, newPrediction];
          }
          return prev;
        });
      }
    } catch (err) {
      console.error('Failed to save prediction:', err);
    } finally {
      setSaving(false);
    }
  };

  const getUserPrediction = (gameId: string) => {
    return userPredictions.find(p => p.game_id === gameId);
  };

  const getStats = () => {
    const totalPredictions = userPredictions.length;
    const upsetPicks = userPredictions.filter(p => p.is_upset_pick).length;
    const correctPredictions = userPredictions.filter(p => p.user_correct === true).length;
    const completedGames = userPredictions.filter(p => p.actual_winner).length;
    
    // Get current week and season for weekly record calculation
    const currentWeek = games.length > 0 ? games[0].week : 1;
    const currentSeason = games.length > 0 ? games[0].season : 2024;
    
    // Weekly record - predictions for current week only
    const weeklyPredictions = userPredictions.filter(p => {
      const game = games.find(g => g.id === p.game_id);
      return game && game.week === currentWeek && game.season === currentSeason;
    });
    const weeklyCorrect = weeklyPredictions.filter(p => p.user_correct === true).length;
    const weeklyCompleted = weeklyPredictions.filter(p => p.actual_winner).length;
    const weeklyWrong = weeklyCompleted - weeklyCorrect;
    
    // Season record - all predictions for current season
    const seasonPredictions = userPredictions.filter(p => {
      const game = games.find(g => g.id === p.game_id);
      return game && game.season === currentSeason;
    });
    const seasonCorrect = seasonPredictions.filter(p => p.user_correct === true).length;
    const seasonCompleted = seasonPredictions.filter(p => p.actual_winner).length;
    const seasonWrong = seasonCompleted - seasonCorrect;
    
    return {
      totalPredictions,
      upsetPicks,
      accuracy: completedGames > 0 ? Math.round((correctPredictions / completedGames) * 100) : 0,
      completedGames,
      weeklyRecord: `${weeklyCorrect}-${weeklyWrong}`,
      weeklyCompleted,
      seasonRecord: `${seasonCorrect}-${seasonWrong}`,
      seasonCompleted,
    };
  };

  const stats = getStats();

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[400px] space-y-4">
        <Loader2 className="w-8 h-8 animate-spin text-purple-600" />
        <p className="text-gray-600">Loading this week's games...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-center py-12">
        <div className="bg-red-50 border border-red-200 rounded-lg p-6 max-w-md mx-auto">
          <p className="text-red-800">{error}</p>
          <button 
            onClick={fetchGames}
            className="mt-4 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
          >
            Try Again
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto px-4 py-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-4xl font-bold text-gray-900 mb-2">Weekly Game Predictor</h1>
        <p className="text-lg text-gray-600">
          Make your picks and compete against our AI predictions
        </p>
      </div>

      {/* Stats Dashboard */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-4 mb-8">
        <div className="bg-gradient-to-br from-blue-500 to-blue-600 rounded-xl p-6 text-white">
          <div className="flex items-center gap-3">
            <Target className="w-8 h-8" />
            <div>
              <div className="text-2xl font-bold">{stats.totalPredictions}</div>
              <div className="text-blue-100 text-sm">Total Picks</div>
            </div>
          </div>
        </div>

        <div className="bg-gradient-to-br from-orange-500 to-orange-600 rounded-xl p-6 text-white">
          <div className="flex items-center gap-3">
            <TrendingUp className="w-8 h-8" />
            <div>
              <div className="text-2xl font-bold">{stats.upsetPicks}</div>
              <div className="text-orange-100 text-sm">Upset Picks</div>
            </div>
          </div>
        </div>

        <div className="bg-gradient-to-br from-green-500 to-green-600 rounded-xl p-6 text-white">
          <div className="flex items-center gap-3">
            <Trophy className="w-8 h-8" />
            <div>
              <div className="text-2xl font-bold">{stats.accuracy}%</div>
              <div className="text-green-100 text-sm">Accuracy</div>
            </div>
          </div>
        </div>

        <div className="bg-gradient-to-br from-purple-500 to-purple-600 rounded-xl p-6 text-white">
          <div className="flex items-center gap-3">
            <Brain className="w-8 h-8" />
            <div>
              <div className="text-2xl font-bold">VS AI</div>
              <div className="text-purple-100 text-sm">Challenge</div>
            </div>
          </div>
        </div>

        <div className="bg-gradient-to-br from-indigo-500 to-indigo-600 rounded-xl p-6 text-white">
          <div className="flex items-center gap-3">
            <Target className="w-8 h-8" />
            <div>
              <div className="text-2xl font-bold">{stats.weeklyRecord}</div>
              <div className="text-indigo-100 text-sm">Weekly Record</div>
            </div>
          </div>
        </div>

        <div className="bg-gradient-to-br from-cyan-500 to-cyan-600 rounded-xl p-6 text-white">
          <div className="flex items-center gap-3">
            <Trophy className="w-8 h-8" />
            <div>
              <div className="text-2xl font-bold">{stats.seasonRecord}</div>
              <div className="text-cyan-100 text-sm">Season Record</div>
            </div>
          </div>
        </div>
      </div>

      {/* Loading indicator when saving */}
      {saving && (
        <div className="mb-4 bg-blue-50 border border-blue-200 rounded-lg p-3">
          <div className="flex items-center gap-2 text-blue-800">
            <Loader2 className="w-4 h-4 animate-spin" />
            <span className="text-sm">Saving your prediction...</span>
          </div>
        </div>
      )}

      {/* Games Grid */}
      {games.length > 0 ? (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {games.map((game) => (
            <GameCard
              key={game.id}
              game={game}
              userPrediction={getUserPrediction(game.id)}
              onPredictionChange={handlePredictionChange}
              disabled={saving}
            />
          ))}
        </div>
      ) : (
        <div className="text-center py-12">
          <div className="bg-gray-50 rounded-lg p-8 max-w-md mx-auto">
            <Lock className="w-12 h-12 text-gray-400 mx-auto mb-4" />
            <h3 className="text-lg font-semibold text-gray-900 mb-2">No Games Available</h3>
            <p className="text-gray-600">
              Games are updated every Tuesday. Check back soon for this week's matchups!
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
