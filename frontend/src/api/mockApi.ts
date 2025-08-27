// Mock API service for local development
// This allows the frontend to work without the backend running

export interface Team {
  team_abbr: string;
  team_name: string;
  team_conf: string;
  team_division: string;
  team_color: string;
  team_color2: string;
  team_color3: string;
  team_color4: string;
  team_logo_wikipedia: string;
  team_logo_espn: string;
}

export interface TeamStats {
  team_abbr: string;
  team_name: string;
  wins: number;
  losses: number;
  ties: number;
  points_for: number;
  points_against: number;
  offensive_rank: number;
  defensive_rank: number;
}

export interface Game {
  game_id: string;
  season: number;
  week: number;
  season_type: string;
  home_team: string;
  away_team: string;
  home_score: number;
  away_score: number;
  game_date: string;
  game_time: string;
  game_status: string;
}

export interface Player {
  player_id: string;
  player_name: string;
  position: string;
  team: string;
  season: number;
  games_played: number;
  passing_yards: number;
  rushing_yards: number;
  receiving_yards: number;
  touchdowns: number;
}

export interface GamePrediction {
  home_team: string;
  away_team: string;
  user_prediction: string;
  confidence: number;
}

export interface PredictionResponse {
  prediction: string;
  confidence: number;
  reasoning: string;
  historical_data: any;
}

// Mock data
const mockTeams: Team[] = [
  {
    team_abbr: "KC",
    team_name: "Kansas City Chiefs",
    team_conf: "AFC",
    team_division: "West",
    team_color: "#E31837",
    team_color2: "#FFB81C",
    team_color3: "#000000",
    team_color4: "#FFFFFF",
    team_logo_wikipedia: "",
    team_logo_espn: ""
  },
  {
    team_abbr: "BUF",
    team_name: "Buffalo Bills",
    team_conf: "AFC",
    team_division: "East",
    team_color: "#00338D",
    team_color2: "#C60C30",
    team_color3: "#000000",
    team_color4: "#FFFFFF",
    team_logo_wikipedia: "",
    team_logo_espn: ""
  },
  {
    team_abbr: "SF",
    team_name: "San Francisco 49ers",
    team_conf: "NFC",
    team_division: "West",
    team_color: "#AA0000",
    team_color2: "#B3995D",
    team_color3: "#000000",
    team_color4: "#FFFFFF",
    team_logo_wikipedia: "",
    team_logo_espn: ""
  }
];

const mockGames: Game[] = [
  {
    game_id: "1",
    season: 2024,
    week: 1,
    season_type: "REG",
    home_team: "KC",
    away_team: "BUF",
    home_score: 0,
    away_score: 0,
    game_date: "2024-09-08",
    game_time: "8:20 PM",
    game_status: "SCHEDULED"
  },
  {
    game_id: "2",
    season: 2024,
    week: 1,
    season_type: "REG",
    home_team: "SF",
    away_team: "KC",
    home_score: 0,
    away_score: 0,
    game_date: "2024-09-09",
    game_time: "4:25 PM",
    game_status: "SCHEDULED"
  }
];

const mockTeamStats: TeamStats[] = [
  {
    team_abbr: "KC",
    team_name: "Kansas City Chiefs",
    wins: 11,
    losses: 6,
    ties: 0,
    points_for: 371,
    points_against: 294,
    offensive_rank: 8,
    defensive_rank: 4
  },
  {
    team_abbr: "BUF",
    team_name: "Buffalo Bills",
    wins: 11,
    losses: 6,
    ties: 0,
    points_for: 451,
    points_against: 311,
    offensive_rank: 4,
    defensive_rank: 7
  }
];

// Mock API functions
export const mockNflApi = {
  // Teams
  getTeams: async (): Promise<Team[]> => {
    console.log('🏈 Mock API: getTeams called');
    await new Promise(resolve => setTimeout(resolve, 500)); // Simulate network delay
    console.log('🏈 Mock API: Returning teams:', mockTeams);
    return mockTeams;
  },

  getTeamStats: async (team: string, season?: number): Promise<TeamStats> => {
    console.log('📊 Mock API: getTeamStats called for', team);
    await new Promise(resolve => setTimeout(resolve, 300));
    const stats = mockTeamStats.find(t => t.team_abbr === team.toUpperCase());
    if (!stats) throw new Error(`Team ${team} not found`);
    return stats;
  },

  // Games
  getGames: async (season?: number, week?: number): Promise<Game[]> => {
    console.log('🎮 Mock API: getGames called for season', season, 'week', week);
    await new Promise(resolve => setTimeout(resolve, 400));
    console.log('🎮 Mock API: Returning games:', mockGames);
    return mockGames;
  },

  // Players
  getPlayers: async (season?: number, position?: string): Promise<Player[]> => {
    await new Promise(resolve => setTimeout(resolve, 400));
    return [];
  },

  // Standings
  getStandings: async (season?: number): Promise<TeamStats[]> => {
    await new Promise(resolve => setTimeout(resolve, 400));
    return mockTeamStats;
  },

  // Predictions
  predictGame: async (prediction: GamePrediction): Promise<PredictionResponse> => {
    await new Promise(resolve => setTimeout(resolve, 1000)); // Simulate AI processing
    
    const mockReasoning = `Based on the data analysis, ${prediction.user_prediction} has a strong advantage in this matchup. 
    The team's recent performance, head-to-head history, and current form suggest they should come out on top. 
    Key factors include offensive efficiency, defensive strength, and home/away performance trends.`;
    
    return {
      prediction: prediction.user_prediction,
      confidence: Math.floor(Math.random() * 3) + 7, // Random confidence 7-9
      reasoning: mockReasoning,
      historical_data: {
        home_team: { team: prediction.home_team, stats: {} },
        away_team: { team: prediction.away_team, stats: {} },
        user_prediction: prediction.user_prediction,
        user_confidence: prediction.confidence
      }
    };
  },

  // Health check
  healthCheck: async (): Promise<{ status: string; service: string }> => {
    await new Promise(resolve => setTimeout(resolve, 100));
    return { status: "healthy", service: "Mock NFL API" };
  },
};

export default mockNflApi;
