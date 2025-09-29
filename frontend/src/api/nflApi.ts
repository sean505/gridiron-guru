import axios from 'axios';
import { getApiUrl } from '../utils/simpleApiConfig';

// Create axios instance with simple API URL detection
const createApiInstance = () => {
  const baseURL = getApiUrl();
  console.log('🔄 Creating new API instance with baseURL:', baseURL);
  
  return axios.create({
    baseURL,
    timeout: 10000,
    headers: {
      'Content-Type': 'application/json',
    },
  });
};

// Create initial instance
let api = createApiInstance();

// Add request interceptor for debugging
api.interceptors.request.use(
  (config) => {
    console.log('🚀 Making API request to:', config.baseURL + config.url);
    console.log('🔧 Full URL:', config.baseURL + config.url);
    return config;
  },
  (error) => {
    console.error('❌ Request interceptor error:', error);
    return Promise.reject(error);
  }
);

// API response types
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

// API functions
export const nflApi = {
  // Teams
  getTeams: async (): Promise<Team[]> => {
    const apiInstance = createApiInstance();
    const response = await apiInstance.get('/api/teams');
    return response.data.teams;
  },

  getTeamStats: async (team: string, season?: number): Promise<TeamStats> => {
    const params = season ? { season } : {};
    const response = await api.get(`/api/teams/${team}/stats`, { params });
    return response.data.team_stats;
  },

  // Games
  getGames: async (season?: number, week?: number): Promise<{games: Game[], season: number, week: number, total_games: number, is_upcoming_week?: boolean}> => {
    const apiInstance = createApiInstance();
    const params: any = {};
    if (season) params.season = season;
    if (week) params.week = week;
    const response = await apiInstance.get('/api/games', { params });
    return response.data;
  },

  getUpcomingGames: async (): Promise<{games: Game[], season: number, week: number, total_games: number, is_upcoming_week: boolean, message: string}> => {
    const response = await api.get('/api/games/upcoming');
    return response.data;
  },

  getPreviousGames: async (season?: number, week?: number): Promise<{games: Game[], season: number, week: number, total_games: number, prediction_accuracy?: {correct: number, total: number, percentage: number}, is_previous_week: boolean, message: string}> => {
    const apiInstance = createApiInstance();
    const params: any = {};
    if (season) params.season = season;
    if (week) params.week = week;
    const response = await apiInstance.get('/api/games/previous', { params });
    return response.data;
  },

  getWeek1_2025: async (): Promise<{games: Game[], message?: string, season: number, week: number, total_games: number}> => {
    const response = await api.get('/api/games/week1-2025');
    return response.data;
  },

  getWeek18_2024: async (): Promise<{games: Game[], message?: string, season: number, week: number, total_games: number}> => {
    const response = await api.get('/api/games/week18-2024');
    return response.data;
  },

  // Players
  getPlayers: async (season?: number, position?: string): Promise<Player[]> => {
    const params: any = {};
    if (season) params.season = season;
    if (position) params.position = position;
    
    const response = await api.get('/api/players', { params });
    return response.data.players;
  },

  // Standings
  getStandings: async (season?: number): Promise<TeamStats[]> => {
    const params = season ? { season } : {};
    const response = await api.get('/api/standings', { params });
    return response.data.standings;
  },

  // Predictions
  predictGame: async (prediction: GamePrediction): Promise<PredictionResponse> => {
    const response = await api.post('/api/predict', prediction);
    return response.data;
  },

  // Health check
  healthCheck: async (): Promise<{ status: string; service: string }> => {
    const response = await api.get('/health');
    return response.data;
  },
};

// Error handling interceptor
api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error);
    
    if (error.response) {
      // Server responded with error status
      console.error('Error data:', error.response.data);
      console.error('Error status:', error.response.status);
    } else if (error.request) {
      // Request made but no response
      console.error('No response received:', error.request);
    } else {
      // Something else happened
      console.error('Error setting up request:', error.message);
    }
    
    return Promise.reject(error);
  }
);

export default nflApi;
