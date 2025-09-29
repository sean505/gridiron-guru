import React from 'react';
import { Lightbulb, Circle, Flame } from 'lucide-react';
import { categorizeUpset, getUpsetPotentialColor } from '../utils/upsetUtils';

interface PredictionCardProps {
  game: {
    game_id: string;
    away_team: string;
    home_team: string;
    game_date: string;
    game_time: string;
    game_status: string;
    week: number;
    season: number;
    away_score?: number;
    home_score?: number;
    actual_score?: string; // Format: "away_score-home_score" (e.g., "20-24")
    home_record?: string;
    away_record?: string;
    ai_prediction: {
      predicted_winner: string;
      confidence: number;
      predicted_score: string;
      key_factors: string[];
      upset_potential: number;
      ai_analysis: string;
      is_upset: boolean;
      model_accuracy?: number;
    };
    actual_result?: {
      home_score: number;
      away_score: number;
      winner: string;
      final_score: string;
    };
    prediction_accuracy?: {
      correct: boolean;
      status: string;
    };
  };
  userPick?: string;
  showScores?: boolean;
}

const PredictionCard: React.FC<PredictionCardProps> = ({ game, userPick, showScores = false }) => {
  // Categorize upset potential
  const upsetInfo = categorizeUpset(game.ai_prediction.upset_potential);
  

  const getTeamCity = (teamCode: string) => {
    const teamCities: { [key: string]: string } = {
      'BUF': 'Buffalo', 'MIA': 'Miami', 'NE': 'New England', 'NYJ': 'New York',
      'BAL': 'Baltimore', 'CIN': 'Cincinnati', 'CLE': 'Cleveland', 'PIT': 'Pittsburgh',
      'HOU': 'Houston', 'IND': 'Indianapolis', 'JAX': 'Jacksonville', 'TEN': 'Tennessee',
      'DEN': 'Denver', 'KC': 'Kansas City', 'LV': 'Las Vegas', 'LAC': 'Los Angeles',
      'DAL': 'Dallas', 'NYG': 'New York', 'PHI': 'Philadelphia', 'WAS': 'Washington',
      'CHI': 'Chicago', 'DET': 'Detroit', 'GB': 'Green Bay', 'MIN': 'Minnesota',
      'ATL': 'Atlanta', 'CAR': 'Carolina', 'NO': 'New Orleans', 'TB': 'Tampa Bay',
      'ARI': 'Arizona', 'LAR': 'Los Angeles', 'LA': 'Los Angeles', 'SF': 'San Francisco', 'SEA': 'Seattle'
    };
    return teamCities[teamCode] || teamCode;
  };

  const getTeamNickname = (teamCode: string) => {
    const teamNicknames: { [key: string]: string } = {
      'BUF': 'Bills', 'MIA': 'Dolphins', 'NE': 'Patriots', 'NYJ': 'Jets',
      'BAL': 'Ravens', 'CIN': 'Bengals', 'CLE': 'Browns', 'PIT': 'Steelers',
      'HOU': 'Texans', 'IND': 'Colts', 'JAX': 'Jaguars', 'TEN': 'Titans',
      'DEN': 'Broncos', 'KC': 'Chiefs', 'LV': 'Raiders', 'LAC': 'Chargers',
      'DAL': 'Cowboys', 'NYG': 'Giants', 'PHI': 'Eagles', 'WAS': 'Commanders',
      'CHI': 'Bears', 'DET': 'Lions', 'GB': 'Packers', 'MIN': 'Vikings',
      'ATL': 'Falcons', 'CAR': 'Panthers', 'NO': 'Saints', 'TB': 'Buccaneers',
      'ARI': 'Cardinals', 'LAR': 'Rams', 'LA': 'Rams', 'SF': '49ers', 'SEA': 'Seahawks'
    };
    return teamNicknames[teamCode] || teamCode;
  };

  const formatDate = (dateString: string) => {
    if (!dateString) return 'TBD';
    try {
      // Parse the date string and ensure it's treated as local time
      // Add 'T00:00:00' to avoid timezone interpretation issues
      const date = new Date(dateString + 'T00:00:00');
      return date.toLocaleDateString('en-US', { 
        weekday: 'short', 
        month: 'short', 
        day: 'numeric' 
      });
    } catch (error) {
      console.error('Error formatting date:', error);
      return 'TBD';
    }
  };

  const formatTime = (timeString: string) => {
    if (!timeString) return 'TBD';
    
    try {
      // Handle different time formats
      if (timeString.includes('AM') || timeString.includes('PM')) {
        const match = timeString.match(/(\d{1,2}):(\d{2})\s*(AM|PM)/i);
        if (match) {
          const hours = parseInt(match[1]);
          const minutes = match[2];
          const ampm = match[3].toLowerCase();
          // Remove leading zero from hours
          const formattedHours = hours.toString();
          const time = `${formattedHours}:${minutes}${ampm}`;
          // Add timezone indicator
          const timezoneAbbr = new Date().toLocaleTimeString('en-US', { timeZoneName: 'short' }).split(' ')[1];
          return `${time} ${timezoneAbbr}`;
        }
      }
      
      // Handle 24-hour format (e.g., "13:00" or "20:20")
      if (timeString.includes(':')) {
        const [hours, minutes] = timeString.split(':');
        const hour24 = parseInt(hours);
        const hour12 = hour24 === 0 ? 12 : hour24 > 12 ? hour24 - 12 : hour24;
        const ampm = hour24 >= 12 ? 'pm' : 'am';
        // Remove leading zero from hours
        const formattedHours = hour12.toString();
        const time = `${formattedHours}:${minutes}${ampm}`;
        // Add timezone indicator
        const timezoneAbbr = new Date().toLocaleTimeString('en-US', { timeZoneName: 'short' }).split(' ')[1];
        return `${time} ${timezoneAbbr}`;
      }
      
      return timeString;
    } catch (error) {
      console.error('Error formatting time:', error);
      return 'TBD';
    }
  };

  // Determine banner style based on upset level
  const isHighUpset = upsetInfo.level === 'pick';
  const bannerText = isHighUpset ? "UPSET PICK" : "LOCK IT IN";
  const bannerBgColor = isHighUpset ? "bg-yellow-100 border-yellow-300" : "bg-blue-50 border-blue-200";
  const bannerTextColor = isHighUpset ? "text-orange-700" : "text-blue-600";
  const bannerIcon = isHighUpset ? "text-orange-500" : "text-blue-500";
  const bannerBorderColor = isHighUpset ? "border-orange-300" : "border-blue-300";

  return (
    <div className="w-full max-w-4xl mx-auto lg:max-w-none transition-all hover:scale-[1.02] hover:shadow-xl">
      {/* Banner Section */}
      <div className={`${bannerBgColor} border border-b-0 rounded-t-xl p-3 lg:p-4 shadow-sm`}>
        <div className="flex items-center gap-4">
          {isHighUpset ? (
            <Flame className={`w-6 h-6 ${bannerIcon}`} />
          ) : (
            <Lightbulb className={`w-6 h-6 ${bannerIcon}`} />
          )}
          <div className={`flex-1 border-l ${bannerBorderColor} pl-4`}>
            <h3 className={`text-xl font-extrabold ${bannerTextColor}`}>
              {bannerText}
            </h3>
          </div>
        </div>
      </div>

      {/* Main Card */}
      <div className="bg-white border border-t-0 border-gray-200 rounded-b-xl shadow-sm p-4 lg:p-6">
        {/* Desktop Layout - Horizontal */}
        <div className="hidden sm:flex justify-center gap-4 lg:gap-6 mb-6 lg:mb-8">
          {/* Away Team */}
          <div className={`flex flex-col items-center gap-1 w-40 lg:w-48 px-0 py-2 ${upsetInfo.level === 'pick' && game.ai_prediction?.predicted_winner !== game.away_team ? 'bg-yellow-50 border border-yellow-200 rounded-lg' : ''}`}>
            <p className="text-gray-600 text-base text-center">{getTeamCity(game.away_team)}</p>
            <div className="flex items-center gap-2">
              <h4 className="text-2xl font-bold text-gray-900 text-center tracking-tight">{getTeamNickname(game.away_team)}</h4>
              {game.ai_prediction.predicted_winner === game.away_team && (
                <div className="w-3 h-3 rounded-full bg-blue-500"></div>
              )}
            </div>
            <p className="text-gray-600 text-base text-center">
              {game.away_record || '0-0'}
            </p>
            {showScores && game.game_status === 'completed' && game.actual_score && (
              <p className={`text-6xl font-bold text-center tracking-tight leading-tight ${
                (() => {
                  // Parse actual_score in format "away_score-home_score" (e.g., "20-24")
                  if (game.actual_score) {
                    const [awayScore, homeScore] = game.actual_score.split('-');
                    const awayScoreNum = parseInt(awayScore) || 0;
                    const homeScoreNum = parseInt(homeScore) || 0;
                    // If away team lost, use grey-400, otherwise use gray-900
                    return awayScoreNum < homeScoreNum ? 'text-gray-400' : 'text-gray-900';
                  }
                  return 'text-gray-900';
                })()
              }`}>
                {(() => {
                  // Parse actual_score in format "away_score-home_score" (e.g., "20-24")
                  if (game.actual_score) {
                    const [awayScore] = game.actual_score.split('-');
                    return awayScore || '00';
                  }
                  return game.actual_result?.away_score ?? game.away_score ?? '00';
                })()}
              </p>
            )}
            {game.ai_prediction.predicted_winner === game.away_team ? (
              <p className="text-base text-center text-blue-600">Predict to Win</p>
            ) : upsetInfo.level === 'threat' ? (
              <p className="text-base text-center text-orange-600">Upset Threat!</p>
            ) : upsetInfo.level === 'pick' ? (
              <p className="text-base text-center text-red-600">Upset Pick!</p>
            ) : null}
            {userPick === game.away_team && (
              <div className="bg-[#fbe1f7] border border-[#fcb2f4] rounded-xl px-2 py-1 mt-2">
                <span className="text-[#8c3e79] text-sm font-bold">MY PICK</span>
              </div>
            )}
          </div>

          {/* Game Info */}
          <div className="flex flex-col items-center gap-1 w-40 lg:w-48">
            <h4 className="text-xl font-bold text-gray-600 text-center tracking-tight">
              {formatDate(game.game_date)}
            </h4>
            <p className="text-gray-600 text-base text-center">
              {game.game_status === 'completed' || game.game_status === 'STATUS_FINAL' ? 'FINAL' : formatTime(game.game_time)}
            </p>
            <div className="bg-green-100 border border-green-300 rounded-xl px-3 py-1 flex items-center gap-2 min-w-fit">
              <span className="text-green-700 text-sm font-normal whitespace-nowrap">AI PREDICTION</span>
              <div className="w-px h-3 bg-green-300"></div>
              <span className="text-green-700 text-sm font-extrabold whitespace-nowrap">{game.ai_prediction.predicted_winner === "N/A" ? "N/A" : game.ai_prediction.predicted_winner}</span>
            </div>
          </div>

          {/* Home Team */}
          <div className={`flex flex-col items-center gap-1 w-40 lg:w-48 px-0 py-2 ${upsetInfo.level === 'pick' && game.ai_prediction?.predicted_winner !== game.home_team ? 'bg-yellow-50 border border-yellow-200 rounded-lg' : ''}`}>
            <p className="text-gray-600 text-base text-center">{getTeamCity(game.home_team)}</p>
            <div className="flex items-center gap-2">
              <h4 className="text-2xl font-bold text-gray-900 text-center tracking-tight">{getTeamNickname(game.home_team)}</h4>
              {game.ai_prediction.predicted_winner === game.home_team && (
                <div className="w-3 h-3 rounded-full bg-blue-500"></div>
              )}
            </div>
            <p className="text-gray-600 text-base text-center">
              {game.home_record || '0-0'}
            </p>
            {showScores && game.game_status === 'completed' && game.actual_score && (
              <p className={`text-6xl font-bold text-center tracking-tight leading-tight ${
                (() => {
                  // Parse actual_score in format "away_score-home_score" (e.g., "20-24")
                  if (game.actual_score) {
                    const [awayScore, homeScore] = game.actual_score.split('-');
                    const awayScoreNum = parseInt(awayScore) || 0;
                    const homeScoreNum = parseInt(homeScore) || 0;
                    // If home team lost, use grey-400, otherwise use gray-900
                    return homeScoreNum < awayScoreNum ? 'text-gray-400' : 'text-gray-900';
                  }
                  return 'text-gray-900';
                })()
              }`}>
                {(() => {
                  // Parse actual_score in format "away_score-home_score" (e.g., "20-24")
                  if (game.actual_score) {
                    const [, homeScore] = game.actual_score.split('-');
                    return homeScore || '00';
                  }
                  return game.actual_result?.home_score ?? game.home_score ?? '00';
                })()}
              </p>
            )}
            {game.ai_prediction.predicted_winner === game.home_team ? (
              <p className="text-base text-center text-blue-600">Predict to Win</p>
            ) : upsetInfo.level === 'threat' ? (
              <p className="text-base text-center text-orange-600">Upset Threat!</p>
            ) : upsetInfo.level === 'pick' ? (
              <p className="text-base text-center text-red-600">Upset Pick!</p>
            ) : null}
            {userPick === game.home_team && (
              <div className="bg-[#fbe1f7] border border-[#fcb2f4] rounded-xl px-2 py-1 mt-2">
                <span className="text-[#8c3e79] text-sm font-bold">MY PICK</span>
              </div>
            )}
          </div>
        </div>

        {/* Mobile Layout - Vertical Stack */}
        <div className="sm:hidden flex flex-col gap-4 mb-6">
          {/* Away Team - Top */}
          <div className={`flex flex-col items-center gap-2 px-4 py-3 ${upsetInfo.level === 'pick' && game.ai_prediction?.predicted_winner !== game.away_team ? 'bg-yellow-50 border border-yellow-200 rounded-lg' : ''}`}>
            <p className="text-gray-600 text-sm text-center">{getTeamCity(game.away_team)}</p>
            <div className="flex items-center gap-2">
              <h4 className="text-xl font-bold text-gray-900 text-center tracking-tight">{getTeamNickname(game.away_team)}</h4>
              {game.ai_prediction.predicted_winner === game.away_team && (
                <div className="w-3 h-3 rounded-full bg-blue-500"></div>
              )}
            </div>
            <p className="text-gray-600 text-sm text-center">
              {game.away_record || '0-0'}
            </p>
            {showScores && game.game_status === 'completed' && game.actual_score && (
              <p className={`text-6xl font-bold text-center tracking-tight leading-tight ${
                (() => {
                  // Parse actual_score in format "away_score-home_score" (e.g., "20-24")
                  if (game.actual_score) {
                    const [awayScore, homeScore] = game.actual_score.split('-');
                    const awayScoreNum = parseInt(awayScore) || 0;
                    const homeScoreNum = parseInt(homeScore) || 0;
                    // If away team lost, use grey-400, otherwise use gray-900
                    return awayScoreNum < homeScoreNum ? 'text-gray-400' : 'text-gray-900';
                  }
                  return 'text-gray-900';
                })()
              }`}>
                {(() => {
                  // Parse actual_score in format "away_score-home_score" (e.g., "20-24")
                  if (game.actual_score) {
                    const [awayScore] = game.actual_score.split('-');
                    return awayScore || '00';
                  }
                  return game.actual_result?.away_score ?? game.away_score ?? '00';
                })()}
              </p>
            )}
            {game.ai_prediction.predicted_winner === game.away_team ? (
              <p className="text-sm text-center text-blue-600">Predict to Win</p>
            ) : upsetInfo.level === 'threat' ? (
              <p className="text-sm text-center text-orange-600">Upset Threat!</p>
            ) : upsetInfo.level === 'pick' ? (
              <p className="text-sm text-center text-red-600">Upset Pick!</p>
            ) : null}
            {userPick === game.away_team && (
              <div className="bg-[#fbe1f7] border border-[#fcb2f4] rounded-xl px-2 py-1 mt-1">
                <span className="text-[#8c3e79] text-xs font-bold">MY PICK</span>
              </div>
            )}
          </div>

          {/* Game Info - Middle */}
          <div className="flex flex-col items-center gap-2 px-4 py-3 bg-gray-50 rounded-lg">
            <h4 className="text-lg font-bold text-gray-600 text-center tracking-tight">
              {formatDate(game.game_date)}
            </h4>
            <p className="text-gray-600 text-sm text-center">
              {game.game_status === 'completed' || game.game_status === 'STATUS_FINAL' ? 'FINAL' : formatTime(game.game_time)}
            </p>
            <div className="bg-green-100 border border-green-300 rounded-xl px-3 py-1 flex items-center gap-2">
              <span className="text-green-700 text-xs font-normal">AI PREDICTION</span>
              <div className="w-px h-3 bg-green-300"></div>
              <span className="text-green-700 text-xs font-extrabold">{game.ai_prediction.predicted_winner === "N/A" ? "N/A" : game.ai_prediction.predicted_winner}</span>
            </div>
          </div>

          {/* Home Team - Bottom */}
          <div className={`flex flex-col items-center gap-2 px-4 py-3 ${upsetInfo.level === 'pick' && game.ai_prediction?.predicted_winner !== game.home_team ? 'bg-yellow-50 border border-yellow-200 rounded-lg' : ''}`}>
            <p className="text-gray-600 text-sm text-center">{getTeamCity(game.home_team)}</p>
            <div className="flex items-center gap-2">
              <h4 className="text-xl font-bold text-gray-900 text-center tracking-tight">{getTeamNickname(game.home_team)}</h4>
              {game.ai_prediction.predicted_winner === game.home_team && (
                <div className="w-3 h-3 rounded-full bg-blue-500"></div>
              )}
            </div>
            <p className="text-gray-600 text-sm text-center">
              {game.home_record || '0-0'}
            </p>
            {showScores && game.game_status === 'completed' && game.actual_score && (
              <p className={`text-6xl font-bold text-center tracking-tight leading-tight ${
                (() => {
                  // Parse actual_score in format "away_score-home_score" (e.g., "20-24")
                  if (game.actual_score) {
                    const [awayScore, homeScore] = game.actual_score.split('-');
                    const awayScoreNum = parseInt(awayScore) || 0;
                    const homeScoreNum = parseInt(homeScore) || 0;
                    // If home team lost, use grey-400, otherwise use gray-900
                    return homeScoreNum < awayScoreNum ? 'text-gray-400' : 'text-gray-900';
                  }
                  return 'text-gray-900';
                })()
              }`}>
                {(() => {
                  // Parse actual_score in format "away_score-home_score" (e.g., "20-24")
                  if (game.actual_score) {
                    const [, homeScore] = game.actual_score.split('-');
                    return homeScore || '00';
                  }
                  return game.actual_result?.home_score ?? game.home_score ?? '00';
                })()}
              </p>
            )}
            {game.ai_prediction.predicted_winner === game.home_team ? (
              <p className="text-sm text-center text-blue-600">Predict to Win</p>
            ) : upsetInfo.level === 'threat' ? (
              <p className="text-sm text-center text-orange-600">Upset Threat!</p>
            ) : upsetInfo.level === 'pick' ? (
              <p className="text-sm text-center text-red-600">Upset Pick!</p>
            ) : null}
            {userPick === game.home_team && (
              <div className="bg-[#fbe1f7] border border-[#fcb2f4] rounded-xl px-2 py-1 mt-1">
                <span className="text-[#8c3e79] text-xs font-bold">MY PICK</span>
              </div>
            )}
          </div>
        </div>

        {/* Prediction Accuracy Status (for previous week) */}
        {game.prediction_accuracy && (
          <div className={`border border-b-0 rounded-t-xl p-3 shadow-sm ${
            game.prediction_accuracy.correct 
              ? 'bg-green-50 border-green-200' 
              : 'bg-red-50 border-red-200'
          }`}>
            <div className="flex items-center gap-2">
              <div className={`w-3 h-3 rounded-full ${
                game.prediction_accuracy.correct ? 'bg-green-500' : 'bg-red-500'
              }`}></div>
              <p className={`text-base font-medium ${
                game.prediction_accuracy.correct ? 'text-green-700' : 'text-red-700'
              }`}>
                {game.prediction_accuracy.status}
              </p>
            </div>
          </div>
        )}

        {/* AI Analysis Section - Always visible */}
        <div className="border border-t-0 rounded-b-xl p-4 shadow-sm bg-purple-50 border-purple-200 rounded-t-xl">
          <div className="flex flex-col gap-2 mb-4">
            <div className="flex items-center gap-2">
              <Circle className="w-3 h-3 fill-purple-600 text-purple-600" />
              <p className="text-purple-700 text-base font-medium flex-1">
                <span className="font-medium">AI Prediction:</span>
                <span className="font-bold"> {game.ai_prediction.predicted_winner === "N/A" ? "N/A - No prediction available" : `${game.ai_prediction.predicted_winner} predicted to win`}</span>
              </p>
            </div>
            <p className="text-purple-700 text-sm leading-5">
              {upsetInfo.message && (
                <span className={`block mb-2 font-medium ${upsetInfo.colorClass}`}>
                  {upsetInfo.message}
                </span>
              )}
              {game.ai_prediction.ai_analysis}
            </p>
          </div>
          
          <div className="flex justify-between items-center">
            <div className="bg-blue-100 border border-blue-300 rounded-xl px-2 py-1 flex items-center gap-2">
              <span className="text-blue-700 text-sm font-medium">{game.ai_prediction.confidence === 0 ? "N/A" : Math.round(game.ai_prediction.confidence)}</span>
              <div className="w-px h-3 bg-blue-300"></div>
              <span className="text-blue-700 text-sm font-normal">Confidence</span>
            </div>
            
            <div className={`rounded-xl px-2 py-1 flex items-center gap-2 ${getUpsetPotentialColor(game.ai_prediction.upset_potential)}`}>
              <span className="text-sm font-medium">{game.ai_prediction.upset_potential === 0 ? "N/A" : Math.round(game.ai_prediction.upset_potential)}</span>
              <div className="w-px h-3 bg-current opacity-30"></div>
              <span className="text-sm font-normal">Upset Potential</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PredictionCard;
