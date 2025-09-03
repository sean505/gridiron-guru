import React from 'react';
import { Lightbulb, Circle, Flame } from 'lucide-react';

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
    ai_prediction: {
      predicted_winner: string;
      confidence: number;
      upset_potential: number;
      is_upset: boolean;
      model_accuracy: number;
    };
  };
  userPick?: string;
}

const PredictionCard: React.FC<PredictionCardProps> = ({ game, userPick }) => {
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

  const getTeamCity = (teamCode: string) => {
    const teamCities: { [key: string]: string } = {
      'BUF': 'Buffalo', 'MIA': 'Miami', 'NE': 'New England', 'NYJ': 'New York',
      'BAL': 'Baltimore', 'CIN': 'Cincinnati', 'CLE': 'Cleveland', 'PIT': 'Pittsburgh',
      'HOU': 'Houston', 'IND': 'Indianapolis', 'JAX': 'Jacksonville', 'TEN': 'Tennessee',
      'DEN': 'Denver', 'KC': 'Kansas City', 'LV': 'Las Vegas', 'LAC': 'Los Angeles',
      'DAL': 'Dallas', 'NYG': 'New York', 'PHI': 'Philadelphia', 'WAS': 'Washington',
      'CHI': 'Chicago', 'DET': 'Detroit', 'GB': 'Green Bay', 'MIN': 'Minnesota',
      'ATL': 'Atlanta', 'CAR': 'Carolina', 'NO': 'New Orleans', 'TB': 'Tampa Bay',
      'ARI': 'Arizona', 'LAR': 'Los Angeles', 'SF': 'San Francisco', 'SEA': 'Seattle'
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
      'ARI': 'Cardinals', 'LAR': 'Rams', 'SF': '49ers', 'SEA': 'Seahawks'
    };
    return teamNicknames[teamCode] || teamCode;
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', { 
      weekday: 'short', 
      month: 'short', 
      day: 'numeric' 
    });
  };

  const formatTime = (timeString: string) => {
    return timeString.replace(/\s*(AM|PM)/i, '').toLowerCase() + timeString.match(/\s*(AM|PM)/i)?.[1]?.toLowerCase();
  };

  const bannerText = game.ai_prediction.is_upset ? "UPSET PICK" : "LOCK IT IN";
  const bannerBgColor = game.ai_prediction.is_upset ? "bg-yellow-100 border-yellow-300" : "bg-blue-50 border-blue-200";
  const bannerTextColor = game.ai_prediction.is_upset ? "text-orange-700" : "text-blue-600";
  const bannerIcon = game.ai_prediction.is_upset ? "text-orange-500" : "text-blue-500";
  const bannerBorderColor = game.ai_prediction.is_upset ? "border-orange-300" : "border-blue-300";

  return (
    <div className="w-full max-w-4xl mx-auto lg:max-w-none">
      {/* Banner Section */}
      <div className={`${bannerBgColor} border border-b-0 rounded-t-xl p-3 lg:p-4 shadow-sm`}>
        <div className="flex items-center gap-4">
          {game.ai_prediction.is_upset ? (
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
      <div className="bg-white border border-t-0 border-gray-200 rounded-b-xl shadow-sm p-4 lg:p-6 relative">
        {/* MY PICK Badge - Top Right */}
        {userPick && (
          <div className="absolute top-4 right-4 bg-[#fbe1f7] border border-[#fcb2f4] rounded-xl px-2 py-1">
            <span className="text-[#8c3e79] text-sm font-bold">MY PICK</span>
          </div>
        )}
        {/* Stats Section */}
        <div className="flex justify-center gap-4 lg:gap-6 mb-6 lg:mb-8">
          {/* Away Team */}
          <div className="flex flex-col items-center gap-1 w-40 lg:w-48">
            <p className="text-gray-600 text-base text-center">{getTeamCity(game.away_team)}</p>
            <div className="flex items-center gap-2">
              <h4 className="text-2xl font-bold text-gray-900 text-center tracking-tight">{getTeamNickname(game.away_team)}</h4>
              {game.ai_prediction.is_upset && game.ai_prediction.predicted_winner === game.away_team && (
                <div className="w-3 h-3 bg-orange-500 rounded-full"></div>
              )}
            </div>
            <p className="text-gray-600 text-base text-center">0-0</p>
            <p className="text-6xl font-bold text-gray-900 text-center tracking-tight leading-tight">00</p>
          </div>

          {/* Game Info */}
          <div className="flex flex-col items-center gap-1 w-40 lg:w-48">
            <h4 className="text-xl font-bold text-gray-600 text-center tracking-tight">
              {formatDate(game.game_date)}
            </h4>
            <p className="text-gray-600 text-base text-center">{formatTime(game.game_time)}</p>
            <div className="bg-green-100 border border-green-300 rounded-xl px-2 py-1 flex items-center gap-2">
              <span className="text-green-700 text-sm font-normal">AI PREDICTION</span>
              <div className="w-px h-3 bg-green-300"></div>
              <span className="text-green-700 text-sm font-extrabold">{game.ai_prediction.predicted_winner}</span>
            </div>
          </div>

          {/* Home Team */}
          <div className="flex flex-col items-center gap-1 w-40 lg:w-48">
            <p className="text-gray-600 text-base text-center">{getTeamCity(game.home_team)}</p>
            <div className="flex items-center gap-2">
              <h4 className="text-2xl font-bold text-gray-900 text-center tracking-tight">{getTeamNickname(game.home_team)}</h4>
              {game.ai_prediction.is_upset && game.ai_prediction.predicted_winner === game.home_team && (
                <div className="w-3 h-3 bg-orange-500 rounded-full"></div>
              )}
            </div>
            <p className="text-gray-600 text-base text-center">0-0</p>
            <p className="text-6xl font-bold text-gray-900 text-center tracking-tight leading-tight">00</p>
          </div>
        </div>

        {/* AI Analysis Section */}
        <div className="bg-purple-50 border border-purple-200 rounded-xl p-4 shadow-sm">
          <div className="flex flex-col gap-2 mb-4">
            <div className="flex items-center gap-2">
              <Circle className="w-3 h-3 fill-purple-600 text-purple-600" />
              <p className="text-purple-700 text-base font-medium flex-1">
                <span className="font-medium">AI Prediction:</span>
                <span className="font-bold"> {game.ai_prediction.predicted_winner} predicted to win</span>
              </p>
            </div>
            <p className="text-purple-700 text-sm leading-5">
              {game.ai_prediction.predicted_winner} has historically performed well in similar game conditions. 
              Their coaching staff has demonstrated excellent game planning against this type of opponent.
            </p>
          </div>
          
          <div className="flex justify-between items-center">
            <div className="bg-blue-100 border border-blue-300 rounded-xl px-2 py-1 flex items-center gap-2">
              <span className="text-blue-700 text-sm font-medium">{Math.round(game.ai_prediction.confidence)}</span>
              <div className="w-px h-3 bg-blue-300"></div>
              <span className="text-blue-700 text-sm font-normal">Confidence</span>
            </div>
            
            <div className="bg-yellow-100 border border-yellow-300 rounded-xl px-2 py-1 flex items-center gap-2">
              <span className="text-red-700 text-sm font-medium">{Math.round(game.ai_prediction.upset_potential)}</span>
              <div className="w-px h-3 bg-yellow-300"></div>
              <span className="text-red-700 text-sm font-normal">Upset Potential</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PredictionCard;
