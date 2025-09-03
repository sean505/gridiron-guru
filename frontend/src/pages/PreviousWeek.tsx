import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';
import { Badge } from '../components/ui/badge';
import { Button } from '../components/ui/button';
import { ArrowLeft, Trophy, Target, TrendingUp, CheckCircle, XCircle } from 'lucide-react';

interface GameResult {
  game_id: string;
  away_team: string;
  home_team: string;
  game_date: string;
  week: number;
  season: number;
  actual_result: {
    home_score: number;
    away_score: number;
    winner: string;
    final_score: string;
  };
  ai_prediction: {
    predicted_winner: string;
    confidence: number;
    upset_potential: number;
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
  const [season, setSeason] = useState(2024);
  const [week, setWeek] = useState(18);

  useEffect(() => {
    fetchPreviousWeekData();
  }, [season, week]);

  const fetchPreviousWeekData = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await fetch(`/api/games/previous?season=${season}&week=${week}`);
      
      if (!response.ok) {
        throw new Error('Failed to fetch previous week data');
      }
      
      const data: PreviousWeekResponse = await response.json();
      setGames(data.games);
      setPredictionStats(data.prediction_accuracy);
      
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const getTeamName = (abbr: string) => {
    const teams: { [key: string]: string } = {
      'DEN': 'Denver Broncos',
      'KC': 'Kansas City Chiefs',
      'BUF': 'Buffalo Bills',
      'MIA': 'Miami Dolphins',
      'BAL': 'Baltimore Ravens',
      'SF': 'San Francisco 49ers',
      'DAL': 'Dallas Cowboys',
      'DET': 'Detroit Lions',
      'GB': 'Green Bay Packers',
      'HOU': 'Houston Texans',
      'IND': 'Indianapolis Colts',
      'JAX': 'Jacksonville Jaguars',
      'LV': 'Las Vegas Raiders',
      'LAC': 'Los Angeles Chargers',
      'LAR': 'Los Angeles Rams',
      'SEA': 'Seattle Seahawks',
      'MIN': 'Minnesota Vikings',
      'CHI': 'Chicago Bears',
      'NE': 'New England Patriots',
      'NYJ': 'New York Jets',
      'NO': 'New Orleans Saints',
      'ATL': 'Atlanta Falcons',
      'NYG': 'New York Giants',
      'PHI': 'Philadelphia Eagles',
      'PIT': 'Pittsburgh Steelers',
      'TB': 'Tampa Bay Buccaneers',
      'CAR': 'Carolina Panthers',
      'WAS': 'Washington Commanders',
      'ARI': 'Arizona Cardinals',
      'CIN': 'Cincinnati Bengals',
      'CLE': 'Cleveland Browns',
      'TEN': 'Tennessee Titans'
    };
    return teams[abbr] || abbr;
  };

  const getTeamColor = (abbr: string) => {
    const colors: { [key: string]: string } = {
      'DEN': 'bg-orange-500',
      'KC': 'bg-red-500',
      'BUF': 'bg-blue-500',
      'MIA': 'bg-teal-500',
      'BAL': 'bg-purple-500',
      'SF': 'bg-red-600',
      'DAL': 'bg-blue-600',
      'DET': 'bg-blue-700',
      'GB': 'bg-green-500',
      'HOU': 'bg-blue-800',
      'IND': 'bg-blue-400',
      'JAX': 'bg-teal-600',
      'LV': 'bg-gray-600',
      'LAC': 'bg-blue-500',
      'LAR': 'bg-blue-900',
      'SEA': 'bg-green-600',
      'MIN': 'bg-purple-600',
      'CHI': 'bg-orange-600',
      'NE': 'bg-blue-700',
      'NYJ': 'bg-green-700',
      'NO': 'bg-gold-500',
      'ATL': 'bg-red-700',
      'NYG': 'bg-blue-800',
      'PHI': 'bg-green-800',
      'PIT': 'bg-yellow-600',
      'TB': 'bg-red-800',
      'CAR': 'bg-blue-500',
      'WAS': 'bg-red-600',
      'ARI': 'bg-red-500',
      'CIN': 'bg-orange-500',
      'CLE': 'bg-orange-600',
      'TEN': 'bg-blue-600'
    };
    return colors[abbr] || 'bg-gray-500';
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-white mx-auto mb-4"></div>
          <p className="text-white text-lg">Loading Previous Week Results...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 flex items-center justify-center">
        <div className="text-center">
          <p className="text-red-400 text-lg mb-4">Error: {error}</p>
          <Button onClick={fetchPreviousWeekData} className="bg-purple-600 hover:bg-purple-700">
            Try Again
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="mb-8">
          <Button 
            onClick={() => window.history.back()} 
            variant="ghost" 
            className="text-white hover:bg-white/10 mb-4"
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back
          </Button>
          <h1 className="text-4xl font-bold text-white mb-2">Previous Week Results</h1>
          <p className="text-gray-300 text-lg">
            Week {week}, {season}: AI Predictions vs Actual Results
          </p>
        </div>

        {/* Prediction Accuracy Summary */}
        {predictionStats && (
          <Card className="bg-white/10 backdrop-blur-sm border-white/20 mb-8">
            <CardHeader>
              <CardTitle className="text-white flex items-center">
                <Target className="w-5 h-5 mr-2" />
                Prediction Accuracy Summary
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid md:grid-cols-3 gap-6">
                <div className="text-center">
                  <div className="text-3xl font-bold text-white mb-2">
                    {predictionStats.percentage}%
                  </div>
                  <div className="text-gray-400 text-sm">Overall Accuracy</div>
                </div>
                <div className="text-center">
                  <div className="text-3xl font-bold text-green-400 mb-2">
                    {predictionStats.correct}
                  </div>
                  <div className="text-gray-400 text-sm">Correct Predictions</div>
                </div>
                <div className="text-center">
                  <div className="text-3xl font-bold text-red-400 mb-2">
                    {predictionStats.total - predictionStats.correct}
                  </div>
                  <div className="text-gray-400 text-sm">Incorrect Predictions</div>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Games Grid */}
        <div className="grid gap-6">
          {games.map((game) => (
            <Card key={game.game_id} className="bg-white/10 backdrop-blur-sm border-white/20">
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle className="text-white">
                    {getTeamName(game.away_team)} @ {getTeamName(game.home_team)}
                  </CardTitle>
                  <Badge 
                    variant={game.prediction_accuracy.correct ? "default" : "destructive"}
                    className="text-sm"
                  >
                    {game.prediction_accuracy.status}
                  </Badge>
                </div>
              </CardHeader>
              <CardContent>
                <div className="grid md:grid-cols-2 gap-6">
                  {/* AI Prediction */}
                  <div className="space-y-4">
                    <h3 className="text-lg font-semibold text-white flex items-center">
                      <Target className="w-4 h-4 mr-2" />
                      AI Prediction
                    </h3>
                    <div className="space-y-3">
                      <div className="flex items-center justify-between">
                        <span className="text-gray-400">Predicted Winner:</span>
                        <div className={`inline-flex items-center px-3 py-1 rounded-full ${getTeamColor(game.ai_prediction.predicted_winner)} text-white text-sm font-semibold`}>
                          <Trophy className="w-3 h-3 mr-1" />
                          {getTeamName(game.ai_prediction.predicted_winner)}
                        </div>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-gray-400">Confidence:</span>
                        <span className="text-white font-semibold">{game.ai_prediction.confidence}%</span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-gray-400">Upset Potential:</span>
                        <span className="text-white font-semibold">{game.ai_prediction.upset_potential}%</span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-gray-400">Model Accuracy:</span>
                        <span className="text-white font-semibold">{game.ai_prediction.model_accuracy}%</span>
                      </div>
                    </div>
                  </div>

                  {/* Actual Result */}
                  <div className="space-y-4">
                    <h3 className="text-lg font-semibold text-white flex items-center">
                      <TrendingUp className="w-4 h-4 mr-2" />
                      Actual Result
                    </h3>
                    <div className="space-y-3">
                      <div className="flex items-center justify-between">
                        <span className="text-gray-400">Winner:</span>
                        <div className={`inline-flex items-center px-3 py-1 rounded-full ${getTeamColor(game.actual_result.winner)} text-white text-sm font-semibold`}>
                          <Trophy className="w-3 h-3 mr-1" />
                          {getTeamName(game.actual_result.winner)}
                        </div>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-gray-400">Final Score:</span>
                        <span className="text-white font-semibold text-lg">{game.actual_result.final_score}</span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-gray-400">Home Score:</span>
                        <span className="text-white font-semibold">{game.actual_result.home_score}</span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-gray-400">Away Score:</span>
                        <span className="text-white font-semibold">{game.actual_result.away_score}</span>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Analysis */}
                <div className="mt-6 pt-4 border-t border-white/20">
                  <div className="flex items-center justify-center">
                    {game.prediction_accuracy.correct ? (
                      <div className="flex items-center text-green-400">
                        <CheckCircle className="w-5 h-5 mr-2" />
                        <span className="font-semibold">AI Prediction Correct!</span>
                      </div>
                    ) : (
                      <div className="flex items-center text-red-400">
                        <XCircle className="w-5 h-5 mr-2" />
                        <span className="font-semibold">AI Prediction Incorrect</span>
                      </div>
                    )}
                  </div>
                  <p className="text-gray-300 text-center mt-2 text-sm">
                    The AI predicted <strong className="text-white">{getTeamName(game.ai_prediction.predicted_winner)}</strong> to win with <strong className="text-white">{game.ai_prediction.confidence}%</strong> confidence, 
                    but <strong className="text-white">{getTeamName(game.actual_result.winner)}</strong> won with a final score of <strong className="text-white">{game.actual_result.final_score}</strong>.
                  </p>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>

        {games.length === 0 && (
          <Card className="bg-white/10 backdrop-blur-sm border-white/20">
            <CardContent className="text-center py-12">
              <p className="text-gray-300 text-lg">No games found for Week {week}, {season}</p>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
};

export default PreviousWeek;
