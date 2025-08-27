import { useState } from 'react';
import { Game, NFL_TEAMS, TeamCode, UserPrediction } from '@/shared/types';
import { Check, Clock, TrendingUp } from 'lucide-react';

interface GameCardProps {
  game: Game;
  userPrediction?: UserPrediction;
  onPredictionChange: (gameId: string, winner: string) => void;
  disabled?: boolean;
}

export default function GameCard({ game, userPrediction, onPredictionChange, disabled = false }: GameCardProps) {
  const [selectedWinner, setSelectedWinner] = useState<string>(
    userPrediction?.predicted_winner || ''
  );

  const homeTeam = NFL_TEAMS[game.home_team as TeamCode];
  const awayTeam = NFL_TEAMS[game.away_team as TeamCode];
  
  const isUpsetPrediction = game.ai_predicted_winner && 
    game.ai_confidence && 
    game.ai_confidence > 0.6 && 
    game.ai_predicted_winner !== game.home_team;

  const handleWinnerSelect = (winner: string) => {
    if (disabled) return;
    
    // If clicking on already selected team, deselect it
    if (selectedWinner === winner) {
      setSelectedWinner('');
      onPredictionChange(game.id, '');
    } else {
      setSelectedWinner(winner);
      onPredictionChange(game.id, winner);
    }
  };

  const formatGameTime = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', { 
      weekday: 'short', 
      month: 'short', 
      day: 'numeric',
      hour: 'numeric',
      minute: '2-digit'
    });
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-green-600';
    if (confidence >= 0.7) return 'text-yellow-600';
    return 'text-orange-600';
  };

  return (
    <div className="bg-white rounded-xl shadow-lg border border-gray-100 overflow-hidden hover:shadow-xl transition-all duration-300">
      {/* Header with game time and upset indicator */}
      <div className="bg-gradient-to-r from-slate-800 to-slate-700 text-white px-4 py-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Clock className="w-4 h-4" />
            <span className="text-sm font-medium">{formatGameTime(game.game_date)}</span>
          </div>
          {isUpsetPrediction && (
            <div className="flex items-center gap-1 bg-orange-500 px-2 py-1 rounded-full">
              <TrendingUp className="w-3 h-3" />
              <span className="text-xs font-bold">UPSET ALERT</span>
            </div>
          )}
        </div>
      </div>

      {/* Teams section */}
      <div className="p-6">
        <div className="grid grid-cols-3 gap-4 items-center mb-6">
          {/* Away team */}
          <button
            onClick={() => handleWinnerSelect(game.away_team)}
            disabled={disabled}
            className={`group relative p-4 rounded-xl border-2 transition-all duration-200 ${
              selectedWinner === game.away_team
                ? 'border-blue-500 bg-blue-50 shadow-lg scale-105'
                : 'border-gray-200 hover:border-gray-300 hover:shadow-md'
            } ${disabled ? 'cursor-not-allowed opacity-60' : 'cursor-pointer'}`}
          >
            <div className="text-center">
              <div className="text-lg font-bold text-gray-900">{game.away_team}</div>
              <div className="text-sm text-gray-600">{awayTeam?.name || 'Unknown'}</div>
              {game.away_team_record && (
                <div className="text-xs text-gray-500 mt-1">{game.away_team_record}</div>
              )}
              {game.is_completed && game.away_score && (
                <div className="text-2xl font-bold text-gray-900 mt-2">{game.away_score}</div>
              )}
            </div>
            {selectedWinner === game.away_team && (
              <div className="absolute -top-2 -right-2 bg-blue-500 rounded-full p-1">
                <Check className="w-4 h-4 text-white" />
              </div>
            )}
          </button>

          {/* VS separator */}
          <div className="text-center">
            <div className="text-gray-400 font-bold text-lg">@</div>
          </div>

          {/* Home team */}
          <button
            onClick={() => handleWinnerSelect(game.home_team)}
            disabled={disabled}
            className={`group relative p-4 rounded-xl border-2 transition-all duration-200 ${
              selectedWinner === game.home_team
                ? 'border-blue-500 bg-blue-50 shadow-lg scale-105'
                : 'border-gray-200 hover:border-gray-300 hover:shadow-md'
            } ${disabled ? 'cursor-not-allowed opacity-60' : 'cursor-pointer'}`}
          >
            <div className="text-center">
              <div className="text-lg font-bold text-gray-900">{game.home_team}</div>
              <div className="text-sm text-gray-600">{homeTeam?.name || 'Unknown'}</div>
              {game.home_team_record && (
                <div className="text-xs text-gray-500 mt-1">{game.home_team_record}</div>
              )}
              {game.is_completed && game.home_score && (
                <div className="text-2xl font-bold text-gray-900 mt-2">{game.home_score}</div>
              )}
            </div>
            {selectedWinner === game.home_team && (
              <div className="absolute -top-2 -right-2 bg-blue-500 rounded-full p-1">
                <Check className="w-4 h-4 text-white" />
              </div>
            )}
          </button>
        </div>

        {/* AI Prediction */}
        {game.ai_predicted_winner && game.ai_confidence && (
          <div className="bg-gradient-to-r from-purple-50 to-indigo-50 rounded-lg p-4 border border-purple-100">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-purple-500 rounded-full"></div>
                <span className="text-sm font-semibold text-purple-900">AI Prediction</span>
              </div>
              <div className={`text-sm font-bold ${getConfidenceColor(game.ai_confidence)}`}>
                {Math.round(game.ai_confidence * 100)}% confidence
              </div>
            </div>
            
            <div className="text-sm text-purple-800 mb-2">
              <span className="font-semibold">{game.ai_predicted_winner}</span> predicted to win
            </div>
            
            {game.ai_explanation && (
              <div className="text-xs text-purple-700 leading-relaxed">
                {game.ai_explanation}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
