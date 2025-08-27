from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import nfl_data_py as nfl
import pandas as pd
import openai
import os
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Gridiron Guru API",
    description="NFL data and AI-powered predictions API",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Pydantic models
class GamePrediction(BaseModel):
    home_team: str
    away_team: str
    user_prediction: str
    confidence: int  # 1-10 scale

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    reasoning: str
    historical_data: Dict[str, Any]

class TeamStats(BaseModel):
    team: str
    wins: int
    losses: int
    points_for: float
    points_against: float
    offensive_rank: int
    defensive_rank: int

# NFL Data Service
class NFLDataService:
    def __init__(self):
        self.current_season = 2024
        
    def get_team_stats(self, season: int = None) -> pd.DataFrame:
        """Get team statistics for a given season"""
        if season is None:
            season = self.current_season
        try:
            stats = nfl.import_team_data([season])
            return stats
        except Exception as e:
            logger.error(f"Error fetching team stats: {e}")
            raise HTTPException(status_code=500, detail="Failed to fetch team statistics")
    
    def get_game_data(self, season: int = None) -> pd.DataFrame:
        """Get game data for a given season"""
        if season is None:
            season = self.current_season
        try:
            games = nfl.import_pbp_data([season])
            return games
        except Exception as e:
            logger.error(f"Error fetching game data: {e}")
            raise HTTPException(status_code=500, detail="Failed to fetch game data")
    
    def get_player_stats(self, season: int = None) -> pd.DataFrame:
        """Get player statistics for a given season"""
        if season is None:
            season = self.current_season
        try:
            players = nfl.import_seasonal_data([season])
            return players
        except Exception as e:
            logger.error(f"Error fetching player stats: {e}")
            raise HTTPException(status_code=500, detail="Failed to fetch player statistics")

# Initialize NFL data service
nfl_service = NFLDataService()

# AI Prediction Service
class AIPredictionService:
    def __init__(self):
        self.client = openai_client
        
    async def predict_game(self, home_team: str, away_team: str, historical_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate AI-powered game prediction"""
        try:
            # Create prompt with historical data
            prompt = f"""
            Based on the following NFL data, predict the winner of {home_team} vs {away_team}:
            
            Historical Data:
            {historical_data}
            
            Provide your prediction in the following format:
            - Winner: [Team Name]
            - Confidence: [1-10 scale]
            - Reasoning: [Detailed explanation based on the data]
            """
            
            response = await self.client.chat.completions.acreate(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert NFL analyst with deep knowledge of team statistics, player performance, and game dynamics."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            # Parse the response
            prediction_text = response.choices[0].message.content
            
            # Extract prediction components (simplified parsing)
            winner = home_team if "home" in prediction_text.lower() else away_team
            confidence = 7  # Default confidence
            
            return {
                "prediction": winner,
                "confidence": confidence,
                "reasoning": prediction_text,
                "historical_data": historical_data
            }
            
        except Exception as e:
            logger.error(f"Error generating AI prediction: {e}")
            raise HTTPException(status_code=500, detail="Failed to generate AI prediction")

# Initialize AI prediction service
ai_service = AIPredictionService()

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Gridiron Guru API - NFL Data & AI Predictions"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Gridiron Guru API"}

@app.get("/api/teams")
async def get_teams():
    """Get all NFL teams"""
    try:
        teams = nfl.import_team_desc()
        return {"teams": teams.to_dict('records')}
    except Exception as e:
        logger.error(f"Error fetching teams: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch teams")

@app.get("/api/teams/{team}/stats")
async def get_team_stats(team: str, season: Optional[int] = None):
    """Get statistics for a specific team"""
    try:
        stats = nfl_service.get_team_stats(season)
        team_stats = stats[stats['team_abbr'] == team.upper()]
        
        if team_stats.empty:
            raise HTTPException(status_code=404, detail=f"Team {team} not found")
            
        return {"team_stats": team_stats.to_dict('records')[0]}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching team stats for {team}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch team statistics")

@app.get("/api/games")
async def get_games(season: Optional[int] = None, week: Optional[int] = None):
    """Get games for a season/week"""
    try:
        if season is None:
            season = nfl_service.current_season
            
        games = nfl.import_schedules([season])
        
        if week:
            games = games[games['week'] == week]
            
        return {"games": games.to_dict('records')}
    except Exception as e:
        logger.error(f"Error fetching games: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch games")

@app.get("/api/players")
async def get_players(season: Optional[int] = None, position: Optional[str] = None):
    """Get player statistics"""
    try:
        players = nfl_service.get_player_stats(season)
        
        if position:
            players = players[players['position'] == position.upper()]
            
        return {"players": players.to_dict('records')}
    except Exception as e:
        logger.error(f"Error fetching players: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch player data")

@app.post("/api/predict", response_model=PredictionResponse)
async def predict_game(prediction: GamePrediction):
    """Generate AI-powered game prediction"""
    try:
        # Get historical data for both teams
        home_stats = nfl_service.get_team_stats()
        away_stats = nfl_service.get_team_stats()
        
        home_team_data = home_stats[home_stats['team_abbr'] == prediction.home_team.upper()]
        away_team_data = away_stats[away_stats['team_abbr'] == prediction.away_team.upper()]
        
        if home_team_data.empty or away_team_data.empty:
            raise HTTPException(status_code=404, detail="One or both teams not found")
        
        # Prepare historical data for AI
        historical_data = {
            "home_team": {
                "team": prediction.home_team,
                "stats": home_team_data.to_dict('records')[0]
            },
            "away_team": {
                "team": prediction.away_team,
                "stats": away_team_data.to_dict('records')[0]
            },
            "user_prediction": prediction.user_prediction,
            "user_confidence": prediction.confidence
        }
        
        # Generate AI prediction
        ai_prediction = await ai_service.predict_game(
            prediction.home_team,
            prediction.away_team,
            historical_data
        )
        
        return PredictionResponse(**ai_prediction)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating prediction: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate prediction")

@app.get("/api/standings")
async def get_standings(season: Optional[int] = None):
    """Get current standings"""
    try:
        if season is None:
            season = nfl_service.current_season
            
        standings = nfl.import_team_data([season])
        return {"standings": standings.to_dict('records')}
    except Exception as e:
        logger.error(f"Error fetching standings: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch standings")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
