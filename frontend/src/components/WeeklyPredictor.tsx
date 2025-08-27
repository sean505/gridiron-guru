import React, { useState, useEffect } from 'react';
import { Trophy, TrendingUp, AlertTriangle, Target, BarChart3, Calendar, Clock, Users } from 'lucide-react';
import { mockNflApi } from '../api/mockApi';

interface Game {
  game_id: string;
  away_team: string;
  home_team: string;
  game_date: string;
  game_time: string;
}

interface AIPrediction {
  predicted_winner: string;
  confidence: number;
  predicted_score: string;
  key_factors: string[];
  upset_potential: number;
  ai_analysis: string;
}

interface GameWithPrediction extends Game {
  ai_prediction: AIPrediction;
  is_upset_pick: boolean;
}

export default function WeeklyPredictor() {
  const [games, setGames] = useState<GameWithPrediction[]>([]);
  const [selectedGame, setSelectedGame] = useState<GameWithPrediction | null>(null);
  const [userPrediction, setUserPrediction] = useState('');
  const [confidence, setConfidence] = useState(5);
  const [loading, setLoading] = useState(false);
  const [aiPrediction, setAiPrediction] = useState<AIPrediction | null>(null);

  useEffect(() => {
    loadGamesWithPredictions();
  }, []);

  const loadGamesWithPredictions = async () => {
    try {
      const gamesData = await mockNflApi.getGames();
      const gamesWithPredictions: GameWithPrediction[] = gamesData.map((game, index) => {
        // Generate mock AI predictions for each game
        const isHomeWin = Math.random() > 0.5;
        const predictedWinner = isHomeWin ? game.home_team : game.away_team;
        const confidence = Math.floor(Math.random() * 30) + 60; // 60-90%
        const upsetPotential = Math.floor(Math.random() * 40) + 10; // 10-50%
        
        // Mark some games as upset picks (lower confidence, higher upset potential)
        const isUpsetPick = confidence < 70 && upsetPotential > 30;
        
        return {
          ...game,
          ai_prediction: {
            predicted_winner: predictedWinner,
            confidence,
            predicted_score: `${Math.floor(Math.random() * 14) + 17}-${Math.floor(Math.random() * 14) + 17}`,
            key_factors: generateKeyFactors(),
            upset_potential: upsetPotential,
            ai_analysis: generateAIAnalysis(predictedWinner, game.away_team, game.home_team, confidence)
          },
          is_upset_pick: isUpsetPick
        };
      });
      
      setGames(gamesWithPredictions);
    } catch (error) {
      console.error('Error loading games:', error);
    }
  };

  const generateKeyFactors = (): string[] => {
    const factors = [
      'Home field advantage',
      'Recent form trends',
      'Head-to-head history',
      'Injury impact',
      'Weather conditions',
      'Rest advantage',
      'Matchup advantages',
      'Momentum factors'
    ];
    return factors.sort(() => Math.random() - 0.5).slice(0, 3);
  };

  const generateAIAnalysis = (winner: string, away: string, home: string, confidence: number): string => {
    const analyses = [
      `${winner} has shown consistent performance in recent weeks, particularly in key situations. Their defensive unit has been dominant against the run, which could neutralize ${winner === home ? away : home}'s offensive strategy.`,
      `The data suggests ${winner} has a significant advantage in this matchup. Their offensive efficiency in the red zone and ability to control time of possession gives them the edge.`,
      `${winner} has historically performed well in similar game conditions. Their coaching staff has demonstrated excellent game planning against this type of opponent.`
    ];
    return analyses[Math.floor(Math.random() * analyses.length)];
  };

  const handleGameSelect = (game: GameWithPrediction) => {
    setSelectedGame(game);
    setAiPrediction(game.ai_prediction);
    setUserPrediction('');
    setConfidence(5);
  };

  const handlePredictionSubmit = async () => {
    if (!selectedGame || !userPrediction) return;
    
    setLoading(true);
    
    // Simulate API call delay
    setTimeout(() => {
      setLoading(false);
      // Here you would normally send the prediction to your backend
      console.log('Prediction submitted:', {
        game: selectedGame,
        user_prediction: userPrediction,
        confidence
      });
    }, 2000);
  };

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

  const getTeamLogo = (teamCode: string) => {
    // Mock team logos - in production, you'd use real team logo URLs
    return null;
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 80) return 'text-green-600 bg-green-100';
    if (confidence >= 65) return 'text-blue-600 bg-blue-100';
    if (confidence >= 50) return 'text-yellow-600 bg-yellow-100';
    return 'text-red-600 bg-red-100';
  };

  const getUpsetPotentialColor = (potential: number) => {
    if (potential >= 40) return 'text-red-600 bg-red-100';
    if (potential >= 25) return 'text-yellow-600 bg-yellow-100';
    return 'text-green-600 bg-green-100';
  };

  if (selectedGame) {
    return (
      <div className="min-h-screen bg-gray-50">
        <div className="max-w-6xl mx-auto p-6 space-y-8">
          {/* Back Button */}
          <button
            onClick={() => setSelectedGame(null)}
            className="flex items-center text-blue-600 hover:text-blue-800 font-medium mb-6"
          >
            ← Back to All Games
          </button>

          {/* Game Header */}
          <div className="bg-white rounded-lg shadow-lg p-6 border border-gray-100">
            <div className="text-center mb-6">
              <h1 className="text-3xl font-bold text-gray-900 mb-2">
                {getTeamName(selectedGame.away_team)} @ {getTeamName(selectedGame.home_team)}
              </h1>
              <div className="flex items-center justify-center space-x-4 text-gray-600">
                <Calendar className="w-4 h-4" />
                <span>{selectedGame.game_date}</span>
                <Clock className="w-4 h-4" />
                <span>{selectedGame.game_time}</span>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              {/* AI Prediction */}
              <div className="space-y-4">
                <h3 className="text-xl font-semibold text-gray-800 mb-4 flex items-center">
                  <Target className="w-5 h-5 mr-2 text-blue-600" />
                  AI Prediction
                </h3>
                
                <div className="bg-blue-50 rounded-lg p-4 border border-blue-200">
                  <div className="flex items-center justify-between mb-3">
                    <span className="font-medium text-blue-900">Predicted Winner:</span>
                    <span className="font-bold text-blue-900">{getTeamName(selectedGame.ai_prediction.predicted_winner)}</span>
                  </div>
                  
                  <div className="flex items-center justify-between mb-3">
                    <span className="font-medium text-blue-900">Confidence:</span>
                    <span className={`px-2 py-1 rounded-full text-sm font-medium ${getConfidenceColor(selectedGame.ai_prediction.confidence)}`}>
                      {selectedGame.ai_prediction.confidence}%
                    </span>
                  </div>
                  
                  <div className="flex items-center justify-between mb-3">
                    <span className="font-medium text-blue-900">Predicted Score:</span>
                    <span className="font-bold text-blue-900">{selectedGame.ai_prediction.predicted_score}</span>
                  </div>
                  
                  <div className="flex items-center justify-between mb-3">
                    <span className="font-medium text-blue-900">Upset Potential:</span>
                    <span className={`px-2 py-1 rounded-full text-sm font-medium ${getUpsetPotentialColor(selectedGame.ai_prediction.upset_potential)}`}>
                      {selectedGame.ai_prediction.upset_potential}%
                    </span>
                  </div>
                </div>

                <div className="space-y-3">
                  <h4 className="font-medium text-gray-800">Key Factors:</h4>
                  <ul className="space-y-2">
                    {selectedGame.ai_prediction.key_factors.map((factor, index) => (
                      <li key={index} className="flex items-start">
                        <div className="w-2 h-2 bg-blue-500 rounded-full mt-2 mr-3 flex-shrink-0"></div>
                        <span className="text-gray-700">{factor}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                <div className="space-y-3">
                  <h4 className="font-medium text-gray-800">AI Analysis:</h4>
                  <p className="text-gray-700 bg-gray-50 rounded-lg p-3 text-sm leading-relaxed">
                    {selectedGame.ai_prediction.ai_analysis}
                  </p>
                </div>
              </div>

              {/* User Prediction Form */}
              <div className="space-y-4">
                <h3 className="text-xl font-semibold text-gray-800 mb-4 flex items-center">
                  <Users className="w-5 h-5 mr-2 text-green-600" />
                  Make Your Prediction
                </h3>
                
                <div className="bg-green-50 rounded-lg p-4 border border-green-200">
                  <div>
                    <label className="block text-sm font-medium text-green-900 mb-2">
                      Winner
                    </label>
                    <select
                      value={userPrediction}
                      onChange={(e) => setUserPrediction(e.target.value)}
                      className="w-full border border-green-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-green-500 bg-white"
                    >
                      <option value="">Select winner...</option>
                      <option value={selectedGame.away_team}>
                        {getTeamName(selectedGame.away_team)} (Away)
                      </option>
                      <option value={selectedGame.home_team}>
                        {getTeamName(selectedGame.home_team)} (Home)
                      </option>
                    </select>
                  </div>
                  
                  <div className="mt-4">
                    <label className="block text-sm font-medium text-green-900 mb-2">
                      Confidence Level (1-10)
                    </label>
                    <input
                      type="range"
                      min="1"
                      max="10"
                      value={confidence}
                      onChange={(e) => setConfidence(Number(e.target.value))}
                      className="w-full h-2 bg-green-200 rounded-lg appearance-none cursor-pointer slider"
                    />
                    <div className="flex justify-between text-xs text-green-600 mt-1">
                      <span>Low</span>
                      <span className="font-medium">{confidence}</span>
                      <span>High</span>
                    </div>
                  </div>
                  
                  <button
                    onClick={handlePredictionSubmit}
                    disabled={!userPrediction || loading}
                    className="w-full mt-4 bg-green-600 text-white py-3 px-6 rounded-md font-medium hover:bg-green-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors shadow-md hover:shadow-lg"
                  >
                    {loading ? 'Submitting...' : 'Submit Prediction'}
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto p-6 space-y-8">
        {/* Header */}
        <div className="text-center">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            <Trophy className="inline-block w-10 h-10 text-yellow-500 mr-3" />
            Gridiron Guru Weekly Predictor
          </h1>
          <p className="text-xl text-gray-600">
            AI-powered NFL predictions backed by comprehensive data analysis
          </p>
          <div className="mt-4 p-3 bg-blue-100 rounded-lg inline-block">
            <p className="text-sm text-blue-800">
              🚧 Currently using mock data for development. Backend integration coming soon!
            </p>
          </div>
        </div>

        {/* Upset Picks Section */}
        {games.filter(g => g.is_upset_pick).length > 0 && (
          <div className="space-y-4">
            <h2 className="text-2xl font-bold text-gray-800 flex items-center">
              <AlertTriangle className="w-6 h-6 mr-2 text-red-500" />
              AI Upset Picks
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {games.filter(g => g.is_upset_pick).map((game) => (
                <div
                  key={game.game_id}
                  onClick={() => handleGameSelect(game)}
                  className="bg-red-50 border-2 border-red-200 rounded-lg shadow-lg p-6 cursor-pointer transition-all hover:shadow-xl hover:bg-red-100"
                >
                  <div className="text-center mb-4">
                    <div className="flex items-center justify-center mb-2">
                      <AlertTriangle className="w-5 h-5 text-red-500 mr-2" />
                      <span className="text-sm font-medium text-red-700 bg-red-200 px-2 py-1 rounded-full">
                        UPSET PICK
                      </span>
                    </div>
                    <h3 className="text-lg font-semibold text-gray-800 mb-2">
                      {getTeamName(game.away_team)} @ {getTeamName(game.home_team)}
                    </h3>
                    <div className="text-center">
                      <span className="text-sm text-gray-600">AI Pick: </span>
                      <span className="font-bold text-red-700">{getTeamName(game.ai_prediction.predicted_winner)}</span>
                    </div>
                    <div className="mt-2 flex justify-center space-x-4 text-sm">
                      <span className={`px-2 py-1 rounded-full ${getConfidenceColor(game.ai_prediction.confidence)}`}>
                        {game.ai_prediction.confidence}% confidence
                      </span>
                      <span className={`px-2 py-1 rounded-full ${getUpsetPotentialColor(game.ai_prediction.upset_potential)}`}>
                        {game.ai_prediction.upset_potential}% upset potential
                      </span>
                    </div>
                  </div>
                  <div className="text-center text-sm text-gray-500 bg-white rounded-md py-2">
                    {game.game_date} • {game.game_time}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* All Games Grid */}
        <div className="space-y-4">
          <h2 className="text-2xl font-bold text-gray-800 flex items-center">
            <BarChart3 className="w-6 h-6 mr-2 text-blue-500" />
            All Games This Week
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {games.map((game) => (
              <div
                key={game.game_id}
                onClick={() => handleGameSelect(game)}
                className={`bg-white rounded-lg shadow-lg p-6 cursor-pointer transition-all hover:shadow-xl border border-gray-100 ${
                  game.is_upset_pick ? 'ring-2 ring-red-300' : ''
                }`}
              >
                <div className="text-center mb-4">
                  <h3 className="text-lg font-semibold text-gray-800 mb-2">
                    {getTeamName(game.away_team)} @ {getTeamName(game.home_team)}
                  </h3>
                  
                  <div className="text-center mb-3">
                    <span className="text-sm text-gray-600">AI Pick: </span>
                    <span className="font-bold text-blue-700">{getTeamName(game.ai_prediction.predicted_winner)}</span>
                  </div>
                  
                  <div className="flex justify-center space-x-3 text-sm">
                    <span className={`px-2 py-1 rounded-full ${getConfidenceColor(game.ai_prediction.confidence)}`}>
                      {game.ai_prediction.confidence}%
                    </span>
                    <span className={`px-2 py-1 rounded-full ${getUpsetPotentialColor(game.ai_prediction.upset_potential)}`}>
                      {game.ai_prediction.upset_potential}%
                    </span>
                  </div>
                </div>
                
                <div className="text-center text-sm text-gray-500 bg-gray-50 rounded-md py-2">
                  {game.game_date} • {game.game_time}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
