from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import nfl_data_py as nfl
import pandas as pd
import os
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Gridiron Guru API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Frontend dev server
        "http://localhost:3000",  # Alternative frontend port
        "https://gridiron-guru.vercel.app",  # Production frontend
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

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
        
    def get_teams(self) -> List[Dict[str, Any]]:
        """Get all NFL teams"""
        try:
            teams = nfl.import_team_desc()
            return teams.to_dict('records')
        except Exception as e:
            logger.error(f"Error fetching teams: {e}")
            # Fallback to basic team list
            return [
                {"team_abbr": "BUF", "team_name": "Buffalo Bills", "team_conf": "AFC", "team_division": "East"},
                {"team_abbr": "KC", "team_name": "Kansas City Chiefs", "team_conf": "AFC", "team_division": "West"},
                {"team_abbr": "SF", "team_name": "San Francisco 49ers", "team_conf": "NFC", "team_division": "West"},
                {"team_abbr": "DAL", "team_name": "Dallas Cowboys", "team_conf": "NFC", "team_division": "East"},
            ]
    
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
            logger.info(f"Attempting to fetch game data for season {season}")
            games = nfl.import_schedules([season])
            logger.info(f"Successfully fetched {len(games)} games for season {season}")
            logger.info(f"Columns available: {list(games.columns)}")
            return games
        except Exception as e:
            logger.error(f"Error fetching game data for season {season}: {e}")
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=['week', 'home_team', 'away_team', 'game_date', 'game_time'])
    
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

@app.get("/")
async def root():
    return {"message": "Gridiron Guru API is running!", "status": "healthy"}

@app.get("/api/teams")
async def get_teams():
    """Get all NFL teams"""
    try:
        teams = nfl_service.get_teams()
        return {"teams": teams}
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
            
        games = nfl_service.get_game_data(season)
        
        if week:
            games = games[games['week'] == week]
            
        return {"games": games.to_dict('records')}
    except Exception as e:
        logger.error(f"Error fetching games: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch games")

@app.get("/api/simple-test")
async def simple_test():
    """Super simple test endpoint without nfl_data_py"""
    try:
        logger.info("Simple test endpoint called")
        return {
            "status": "success",
            "message": "Simple endpoint works",
            "data": "No nfl_data_py used"
        }
    except Exception as e:
        logger.error(f"Error in simple test: {e}")
        return {"error": str(e)}

@app.get("/api/games/week1-2025")
async def get_week1_2025_games():
    """Get specifically Week 1 games for 2025 season"""
    try:
        logger.info("Attempting to fetch Week 1, 2025 games from nfl_data_py")
        
        # Get real 2025 game data
        games = nfl_service.get_game_data(2025)
        logger.info(f"Fetched {len(games)} games for 2025 season")
        
        if games.empty:
            logger.warning("No 2025 games data available")
            return {
                "games": [],
                "message": "2025 season data not yet available",
                "season": 2025,
                "week": 1,
                "total_games": 0
            }
        
        # Clean the data to remove NaN and Infinity values
        def clean_value(value):
            """Clean a value to be JSON serializable"""
            import math
            if pd.isna(value) or (isinstance(value, float) and math.isinf(value)):
                return None
            return value
        
        # Convert to clean, JSON-safe format
        clean_games = []
        for _, game in games.head(5).iterrows():
            clean_game = {}
            for column, value in game.items():
                clean_game[str(column)] = clean_value(value)
            clean_games.append(clean_game)
        
        logger.info(f"Returning {len(clean_games)} cleaned games")
        
        return {
            "games": clean_games,
            "season": 2025,
            "total_games": len(clean_games),
            "note": "Sample games from 2025 season (cleaned for JSON)",
            "raw_columns": list(games.columns) if not games.empty else []
        }
        
    except Exception as e:
        logger.error(f"Error fetching Week 1, 2025 games: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        # Return mock data as fallback if real data fails
        return {
            "games": [
                {
                    "game_id": "fallback",
                    "away_team": "BUF",
                    "home_team": "KC",
                    "game_date": "2025-09-07",
                    "game_time": "8:20 PM",
                    "week": 1,
                    "season": 2025
                }
            ],
            "season": 2025,
            "week": 1,
            "total_games": 1,
            "note": "Fallback data due to error",
            "error": str(e)
        }

@app.get("/api/games/week18-2024")
async def get_week18_2024_games():
    """Get Week 18 (final regular season week) games for 2024 season"""
    try:
        logger.info("Attempting to fetch Week 18, 2024 games from nfl_data_py")
        
        # Get 2024 game data
        games = nfl_service.get_game_data(2024)
        logger.info(f"Fetched {len(games)} games for 2024 season")
        
        if games.empty:
            logger.warning("No 2024 games data available")
            return {
                "games": [],
                "message": "2024 season data not available",
                "season": 2024,
                "week": 18,
                "total_games": 0
            }
        
        # Filter for Week 18 games
        week18_games = games[games['week'] == 18]
        logger.info(f"Found {len(week18_games)} Week 18 games")
        
        if week18_games.empty:
            logger.info("No Week 18 games found, returning sample games from 2024")
            # Return first few games if no Week 18 data
            sample_games = games.head(5).to_dict('records')
            return {
                "games": sample_games,
                "message": "Week 18 data not found. Showing first 5 games from 2024 season.",
                "season": 2024,
                "total_games": len(sample_games),
                "note": "Week 18 may not be available in current dataset"
            }
        
        # Clean the data to remove NaN and Infinity values
        def clean_value(value):
            """Clean a value to be JSON serializable"""
            import math
            if pd.isna(value) or (isinstance(value, float) and math.isinf(value)):
                return None
            return value
        
        # Convert to clean, JSON-safe format
        clean_games = []
        for _, game in week18_games.iterrows():
            clean_game = {}
            for column, value in game.items():
                clean_game[str(column)] = clean_value(value)
            clean_games.append(clean_game)
        
        logger.info(f"Returning {len(clean_games)} cleaned Week 18 games")
        
        return {
            "games": clean_games,
            "season": 2024,
            "week": 18,
            "total_games": len(clean_games),
            "note": "Week 18, 2024 regular season finale (cleaned for JSON)"
        }
        
    except Exception as e:
        logger.error(f"Error fetching Week 18, 2024 games: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        # Return mock data as fallback if real data fails
        return {
            "games": [
                {
                    "game_id": "fallback",
                    "away_team": "BUF",
                    "home_team": "KC",
                    "game_date": "2024-01-07",
                    "game_time": "8:20 PM",
                    "week": 18,
                    "season": 2024
                }
            ],
            "season": 2024,
            "week": 18,
            "total_games": 1,
            "note": "Fallback data due to error",
            "error": str(e)
        }

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

@app.get("/api/standings")
async def get_standings(season: Optional[int] = None):
    """Get current standings"""
    try:
        if season is None:
            season = nfl_service.current_season
            
        standings = nfl_service.get_team_stats(season)
        return {"standings": standings.to_dict('records')}
    except Exception as e:
        logger.error(f"Error fetching standings: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch standings")

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "message": "Gridiron Guru API is running"}

@app.get("/api/test-data")
async def test_nfl_data():
    """Test endpoint to see what data nfl_data_py actually contains"""
    try:
        logger.info("Testing basic endpoint functionality")
        
        # Simple test without nfl_data_py
        return {
            "status": "success",
            "message": "Basic endpoint is working",
            "test_data": {
                "teams": ["BUF", "KC", "SF", "DAL"],
                "seasons": [2024, 2025],
                "backend_status": "running"
            }
        }
        
    except Exception as e:
        logger.error(f"Error in test endpoint: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return {
            "error": str(e),
            "message": "Test failed",
            "traceback": traceback.format_exc()
        }

# Vercel requires this for Python functions
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
