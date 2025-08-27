import z from "zod";

/**
 * Types shared between the client and server go here.
 */

export const GameSchema = z.object({
  id: z.string(),
  week: z.number(),
  season: z.number(),
  home_team: z.string(),
  away_team: z.string(),
  home_team_record: z.string().optional(),
  away_team_record: z.string().optional(),
  spread: z.number().optional(),
  game_date: z.string(),
  home_score: z.number().optional(),
  away_score: z.number().optional(),
  is_completed: z.boolean(),
  ai_predicted_winner: z.string().optional(),
  ai_confidence: z.number().optional(),
  ai_explanation: z.string().optional(),
  actual_winner: z.string().optional(),
});

export type Game = z.infer<typeof GameSchema>;

export const UserPredictionSchema = z.object({
  id: z.number(),
  user_id: z.string(),
  game_id: z.string(),
  predicted_winner: z.string(),
  ai_predicted_winner: z.string().optional(),
  ai_confidence: z.number().optional(),
  is_upset_pick: z.boolean(),
  locked_at: z.string().optional(),
  actual_winner: z.string().optional(),
  user_correct: z.boolean().optional(),
  ai_correct: z.boolean().optional(),
  created_at: z.string(),
  updated_at: z.string(),
});

export type UserPrediction = z.infer<typeof UserPredictionSchema>;

// NFL team info for display
export const NFL_TEAMS = {
  'ARI': { name: 'Cardinals', colors: ['#97233F', '#000000', '#FFB612'] },
  'ATL': { name: 'Falcons', colors: ['#A71930', '#000000', '#A5ACAF'] },
  'BAL': { name: 'Ravens', colors: ['#241773', '#000000', '#9E7C0C'] },
  'BUF': { name: 'Bills', colors: ['#00338D', '#C60C30'] },
  'CAR': { name: 'Panthers', colors: ['#0085CA', '#101820', '#BFC0BF'] },
  'CHI': { name: 'Bears', colors: ['#0B162A', '#C83803'] },
  'CIN': { name: 'Bengals', colors: ['#FB4F14', '#000000'] },
  'CLE': { name: 'Browns', colors: ['#311D00', '#FF3C00'] },
  'DAL': { name: 'Cowboys', colors: ['#003594', '#041E42', '#869397'] },
  'DEN': { name: 'Broncos', colors: ['#FB4F14', '#002244'] },
  'DET': { name: 'Lions', colors: ['#0076B6', '#B0B7BC', '#000000'] },
  'GB': { name: 'Packers', colors: ['#203731', '#FFB612'] },
  'HOU': { name: 'Texans', colors: ['#03202F', '#A71930'] },
  'IND': { name: 'Colts', colors: ['#002C5F', '#A2AAAD'] },
  'JAX': { name: 'Jaguars', colors: ['#006778', '#9F792C', '#000000'] },
  'KC': { name: 'Chiefs', colors: ['#E31837', '#FFB612'] },
  'LV': { name: 'Raiders', colors: ['#000000', '#A5ACAF'] },
  'LAC': { name: 'Chargers', colors: ['#0080C6', '#FFC20E', '#FFFFFF'] },
  'LAR': { name: 'Rams', colors: ['#003594', '#FFA300', '#FF8200'] },
  'MIA': { name: 'Dolphins', colors: ['#008E97', '#FC4C02', '#005778'] },
  'MIN': { name: 'Vikings', colors: ['#4F2683', '#FFC62F'] },
  'NE': { name: 'Patriots', colors: ['#002244', '#C60C30', '#B0B7BC'] },
  'NO': { name: 'Saints', colors: ['#101820', '#D3BC8D'] },
  'NYG': { name: 'Giants', colors: ['#0B2265', '#A71930', '#A5ACAF'] },
  'NYJ': { name: 'Jets', colors: ['#125740', '#000000', '#FFFFFF'] },
  'PHI': { name: 'Eagles', colors: ['#004C54', '#A5ACAF', '#ACC0C6'] },
  'PIT': { name: 'Steelers', colors: ['#FFB612', '#101820'] },
  'SF': { name: '49ers', colors: ['#AA0000', '#B3995D'] },
  'SEA': { name: 'Seahawks', colors: ['#002244', '#69BE28', '#A5ACAF'] },
  'TB': { name: 'Buccaneers', colors: ['#D50A0A', '#FF7900', '#0A0A08'] },
  'TEN': { name: 'Titans', colors: ['#0C2340', '#4B92DB', '#C8102E'] },
  'WAS': { name: 'Commanders', colors: ['#773141', '#FFB612'] },
} as const;

export type TeamCode = keyof typeof NFL_TEAMS;
