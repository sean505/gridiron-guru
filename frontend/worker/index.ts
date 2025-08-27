import { Hono } from "hono";
import { zValidator } from "@hono/zod-validator";
import OpenAI from "openai";
import { z } from "zod";

const app = new Hono<{ Bindings: Env }>();

// Initialize OpenAI client
const getOpenAIClient = (env: Env) => {
  return new OpenAI({
    apiKey: env.OPENAI_API_KEY,
  });
};

// Types for ESPN API responses
interface ESPNEvent {
  id: string;
  date: string;
  week?: { number: number };
  season?: { year: number };
  competitions?: Array<{
    competitors: Array<{
      team: {
        abbreviation: string;
        displayName: string;
        logo?: string;
      };
      homeAway: string;
      score?: string;
      records?: Array<{ summary: string }>;
    }>;
    status: {
      type: {
        completed: boolean;
      };
    };
  }>;
}

interface ESPNResponse {
  events?: ESPNEvent[];
}

// Fetch current week's games from ESPN API
async function fetchWeeklyGames() {
  try {
    const response = await fetch(
      'http://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard'
    );
    const data = await response.json() as ESPNResponse;
    
    return data.events?.map((event: ESPNEvent) => {
      const competition = event.competitions?.[0];
      const competitors = competition?.competitors || [];
      
      const homeTeam = competitors.find((c: any) => c.homeAway === 'home');
      const awayTeam = competitors.find((c: any) => c.homeAway === 'away');
      
      return {
        id: event.id,
        week: event.week?.number || 1,
        season: event.season?.year || 2024,
        home_team: homeTeam?.team?.abbreviation || '',
        away_team: awayTeam?.team?.abbreviation || '',
        home_team_record: homeTeam?.records?.[0]?.summary || '',
        away_team_record: awayTeam?.records?.[0]?.summary || '',
        game_date: event.date,
        home_score: homeTeam?.score ? parseInt(homeTeam.score) : null,
        away_score: awayTeam?.score ? parseInt(awayTeam.score) : null,
        is_completed: competition?.status?.type?.completed || false,
        actual_winner: competition?.status?.type?.completed 
          ? (parseInt(homeTeam?.score || '0') > parseInt(awayTeam?.score || '0') 
            ? homeTeam?.team?.abbreviation 
            : awayTeam?.team?.abbreviation)
          : null,
      };
    }) || [];
  } catch (error) {
    console.error('Error fetching ESPN data:', error);
    return [];
  }
}

// Generate AI predictions for a game
async function generateGamePrediction(openai: OpenAI, homeTeam: string, awayTeam: string, homeRecord: string, awayRecord: string) {
  try {
    const completion = await openai.chat.completions.create({
      model: 'gpt-4o-mini',
      messages: [
        {
          role: 'system',
          content: `You are an expert NFL analyst. Analyze matchups and provide win probability predictions with clear reasoning. Keep explanations concise but insightful, focusing on key factors that influence the outcome.`
        },
        {
          role: 'user',
          content: `Predict the winner for this NFL game: ${awayTeam} (${awayRecord}) @ ${homeTeam} (${homeRecord}). 
          
          Provide your analysis in the following format:
          Winner: [TEAM_ABBREVIATION]
          Confidence: [0.50-0.95]
          Reasoning: [2-3 sentence explanation focusing on key matchup factors]`
        }
      ],
      temperature: 0.3,
      max_tokens: 200,
    });

    const response = completion.choices[0]?.message?.content || '';
    
    // Parse the structured response
    const winnerMatch = response.match(/Winner:\s*([A-Z]{2,4})/);
    const confidenceMatch = response.match(/Confidence:\s*(0\.\d+)/);
    const reasoningMatch = response.match(/Reasoning:\s*(.+)/s);
    
    return {
      predicted_winner: winnerMatch?.[1] || homeTeam,
      confidence: parseFloat(confidenceMatch?.[1] || '0.6'),
      explanation: reasoningMatch?.[1]?.trim() || 'Analysis pending.',
    };
  } catch (error) {
    console.error('Error generating prediction:', error);
    return {
      predicted_winner: homeTeam,
      confidence: 0.6,
      explanation: 'Analysis pending.',
    };
  }
}

// API Routes

// Get current week's games with AI predictions
app.get('/api/games/current-week', async (c) => {
  const db = c.env.DB;
  const openai = getOpenAIClient(c.env);

  try {
    // First check if we have games in the database (for demo/development)
    const dbGames = await db.prepare(
      'SELECT * FROM weekly_games ORDER BY game_date ASC'
    ).all();

    if (dbGames.results && dbGames.results.length > 0) {
      // Return database games if we have them
      return c.json({ games: dbGames.results });
    }

    // Fallback to ESPN API if no database games
    const espnGames = await fetchWeeklyGames();
    
    if (espnGames.length === 0) {
      return c.json({ games: [], error: 'No games found' });
    }

    // Check which games need AI predictions
    const games = [];
    for (const game of espnGames) {
      // Check if we already have this game with predictions
      const existingGame = await db.prepare(
        'SELECT * FROM weekly_games WHERE id = ?'
      ).bind(game.id).first();

      if (existingGame && existingGame.ai_predicted_winner) {
        // Use existing prediction
        games.push(existingGame);
      } else {
        // Generate new prediction
        const prediction = await generateGamePrediction(
          openai,
          game.home_team,
          game.away_team,
          game.home_team_record,
          game.away_team_record
        );

        const gameWithPrediction = {
          ...game,
          ai_predicted_winner: prediction.predicted_winner,
          ai_confidence: prediction.confidence,
          ai_explanation: prediction.explanation,
        };

        // Upsert game to database
        await db.prepare(`
          INSERT OR REPLACE INTO weekly_games 
          (id, week, season, home_team, away_team, home_team_record, away_team_record, 
           game_date, home_score, away_score, is_completed, ai_predicted_winner, 
           ai_confidence, ai_explanation, actual_winner, updated_at)
          VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        `).bind(
          gameWithPrediction.id,
          gameWithPrediction.week,
          gameWithPrediction.season,
          gameWithPrediction.home_team,
          gameWithPrediction.away_team,
          gameWithPrediction.home_team_record,
          gameWithPrediction.away_team_record,
          gameWithPrediction.game_date,
          gameWithPrediction.home_score,
          gameWithPrediction.away_score,
          gameWithPrediction.is_completed,
          gameWithPrediction.ai_predicted_winner,
          gameWithPrediction.ai_confidence,
          gameWithPrediction.ai_explanation,
          gameWithPrediction.actual_winner
        ).run();

        games.push(gameWithPrediction);
      }
    }

    return c.json({ games });
  } catch (error) {
    console.error('Error fetching games:', error);
    return c.json({ error: 'Failed to fetch games' }, 500);
  }
});

// Save user prediction
const UserPredictionSchema = z.object({
  gameId: z.string(),
  predictedWinner: z.string(),
  userId: z.string(),
});

app.post('/api/predictions', zValidator('json', UserPredictionSchema), async (c) => {
  const { gameId, predictedWinner, userId } = c.req.valid('json');
  const db = c.env.DB;

  try {
    // Get game data for context
    const game = await db.prepare(
      'SELECT * FROM weekly_games WHERE id = ?'
    ).bind(gameId).first();

    if (!game) {
      return c.json({ error: 'Game not found' }, 404);
    }

    // Determine if this is an upset pick (user picked underdog)
    const gameConfidence = typeof game.ai_confidence === 'number' ? game.ai_confidence : 0.5;
    const homeTeamFavored = gameConfidence > 0.5 && game.ai_predicted_winner === game.home_team;
    const isUpsetPick = homeTeamFavored ? predictedWinner === game.away_team : predictedWinner === game.home_team;

    // Upsert user prediction
    await db.prepare(`
      INSERT OR REPLACE INTO user_predictions 
      (user_id, game_id, predicted_winner, ai_predicted_winner, ai_confidence, is_upset_pick, updated_at)
      VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
    `).bind(
      userId,
      gameId,
      predictedWinner,
      game.ai_predicted_winner,
      game.ai_confidence,
      isUpsetPick
    ).run();

    return c.json({ success: true });
  } catch (error) {
    console.error('Error saving prediction:', error);
    return c.json({ error: 'Failed to save prediction' }, 500);
  }
});

// Get user predictions for current week
app.get('/api/predictions/:userId', async (c) => {
  const userId = c.req.param('userId');
  const db = c.env.DB;

  try {
    const predictions = await db.prepare(`
      SELECT up.*, wg.home_team, wg.away_team, wg.game_date, wg.is_completed
      FROM user_predictions up
      JOIN weekly_games wg ON up.game_id = wg.id
      WHERE up.user_id = ?
      ORDER BY wg.game_date ASC
    `).bind(userId).all();

    return c.json({ predictions: predictions.results || [] });
  } catch (error) {
    console.error('Error fetching predictions:', error);
    return c.json({ error: 'Failed to fetch predictions' }, 500);
  }
});

export default app;
