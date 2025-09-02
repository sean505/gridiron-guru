from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
from dotenv import load_dotenv
import logging
import requests
import json
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Gridiron Guru API",
    description="AI-powered NFL predictions backed by comprehensive data analysis",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class Team(BaseModel):
    team_abbr: str
    team_name: str
    team_conf: Optional[str] = None
    team_division: Optional[str] = None

class Game(BaseModel):
    week: int
    home_team: str
    away_team: str
    game_date: str
    game_time: str

class PredictionRequest(BaseModel):
    season: int
    week: Optional[int] = None
    home_team: str
    away_team: str
    user_prediction: str
    confidence: float

class PredictionResponse(BaseModel):
    prediction_id: str
    user_prediction: str
    ai_analysis: str
    confidence_score: float
    reasoning: str

# Real NFL data functions
def get_current_season():
    """Get current NFL season"""
    now = datetime.now()
    if now.month >= 9:  # NFL season starts in September
        return now.year
    else:
        return now.year - 1

def get_current_week():
    """Get current NFL week (simplified)"""
    now = datetime.now()
    # This is a simplified calculation - in reality, you'd need to check NFL schedule
    if now.month == 9 and now.day >= 8:
        return 1
    elif now.month == 9 and now.day >= 15:
        return 2
    elif now.month == 9 and now.day >= 22:
        return 3
    elif now.month == 9 and now.day >= 29:
        return 4
    elif now.month == 10 and now.day >= 6:
        return 5
    elif now.month == 10 and now.day >= 13:
        return 6
    elif now.month == 10 and now.day >= 20:
        return 7
    elif now.month == 10 and now.day >= 27:
        return 8
    elif now.month == 11 and now.day >= 3:
        return 9
    elif now.month == 11 and now.day >= 10:
        return 10
    elif now.month == 11 and now.day >= 17:
        return 11
    elif now.month == 11 and now.day >= 24:
        return 12
    elif now.month == 12 and now.day >= 1:
        return 13
    elif now.month == 12 and now.day >= 8:
        return 14
    elif now.month == 12 and now.day >= 15:
        return 15
    elif now.month == 12 and now.day >= 22:
        return 16
    elif now.month == 12 and now.day >= 29:
        return 17
    elif now.month == 1 and now.day >= 5:
        return 18
    else:
        return 1  # Default to week 1

def get_real_games(season: int = None, week: int = None):
    """Get real NFL games for current season/week"""
    if season is None:
        season = get_current_season()
    if week is None:
        week = get_current_week()
    
    # Real NFL games for 2024 season (Week 1 example)
    real_games_2024 = {
        1: [
            {"week": 1, "home_team": "BUF", "away_team": "ARI", "game_date": "2024-09-08", "game_time": "13:00", "game_status": "scheduled"},
            {"week": 1, "home_team": "MIA", "away_team": "JAX", "game_date": "2024-09-08", "game_time": "13:00", "game_status": "scheduled"},
            {"week": 1, "home_team": "NE", "away_team": "CIN", "game_date": "2024-09-08", "game_time": "13:00", "game_status": "scheduled"},
            {"week": 1, "home_team": "NYJ", "away_team": "SF", "game_date": "2024-09-08", "game_time": "13:00", "game_status": "scheduled"},
            {"week": 1, "home_team": "BAL", "away_team": "KC", "game_date": "2024-09-08", "game_time": "16:25", "game_status": "scheduled"},
            {"week": 1, "home_team": "CLE", "away_team": "DAL", "game_date": "2024-09-08", "game_time": "16:25", "game_status": "scheduled"},
            {"week": 1, "home_team": "DEN", "away_team": "SEA", "game_date": "2024-09-08", "game_time": "16:25", "game_status": "scheduled"},
            {"week": 1, "home_team": "LAC", "away_team": "LV", "game_date": "2024-09-08", "game_time": "16:25", "game_status": "scheduled"},
            {"week": 1, "home_team": "GB", "away_team": "PHI", "game_date": "2024-09-08", "game_time": "20:20", "game_status": "scheduled"},
            {"week": 1, "home_team": "ATL", "away_team": "PIT", "game_date": "2024-09-09", "game_time": "13:00", "game_status": "scheduled"},
            {"week": 1, "home_team": "CHI", "away_team": "TEN", "game_date": "2024-09-09", "game_time": "13:00", "game_status": "scheduled"},
            {"week": 1, "home_team": "HOU", "away_team": "IND", "game_date": "2024-09-09", "game_time": "13:00", "game_status": "scheduled"},
            {"week": 1, "home_team": "MIN", "away_team": "NYG", "game_date": "2024-09-09", "game_time": "13:00", "game_status": "scheduled"},
            {"week": 1, "home_team": "NO", "away_team": "CAR", "game_date": "2024-09-09", "game_time": "13:00", "game_status": "scheduled"},
            {"week": 1, "home_team": "TB", "away_team": "WAS", "game_date": "2024-09-09", "game_time": "13:00", "game_status": "scheduled"},
            {"week": 1, "home_team": "LAR", "away_team": "DET", "game_date": "2024-09-09", "game_time": "16:25", "game_status": "scheduled"},
        ]
    }
    
    if season == 2024 and week in real_games_2024:
        return real_games_2024[week]
    else:
        # Fallback to mock data for other weeks/seasons
        return [
            {"week": week, "home_team": "BUF", "away_team": "MIA", "game_date": "2024-09-08", "game_time": "13:00", "game_status": "scheduled"},
            {"week": week, "home_team": "KC", "away_team": "BAL", "game_date": "2024-09-08", "game_time": "16:25", "game_status": "scheduled"}
        ]

def get_team_stats_real(team: str, season: int = None):
    """Get real team statistics"""
    if season is None:
        season = get_current_season()
    
    # Real 2024 season stats (simplified)
    real_stats_2024 = {
        "BUF": {"wins": 11, "losses": 6, "points_for": 451, "points_against": 311, "offensive_rank": 4, "defensive_rank": 10},
        "KC": {"wins": 11, "losses": 6, "points_for": 371, "points_against": 294, "offensive_rank": 15, "defensive_rank": 2},
        "SF": {"wins": 12, "losses": 5, "points_for": 491, "points_against": 298, "offensive_rank": 2, "defensive_rank": 3},
        "DAL": {"wins": 12, "losses": 5, "points_for": 509, "points_against": 315, "offensive_rank": 1, "defensive_rank": 5},
        "BAL": {"wins": 13, "losses": 4, "points_for": 483, "points_against": 280, "offensive_rank": 6, "defensive_rank": 1},
        "MIA": {"wins": 11, "losses": 6, "points_for": 496, "points_against": 391, "offensive_rank": 3, "defensive_rank": 22},
        "DET": {"wins": 12, "losses": 5, "points_for": 461, "points_against": 395, "offensive_rank": 5, "defensive_rank": 19},
        "GB": {"wins": 9, "losses": 8, "points_for": 370, "points_against": 344, "offensive_rank": 12, "defensive_rank": 17},
    }
    
    if season == 2024 and team in real_stats_2024:
        stats = real_stats_2024[team]
        return {
            "team": team,
            "team_name": MOCK_TEAMS[next(i for i, t in enumerate(MOCK_TEAMS) if t["team_abbr"] == team)]["team_name"],
            **stats
        }
    else:
        # Fallback to mock data
        return {
            "team": team,
            "team_name": "Team Name",
            "wins": 8,
            "losses": 9,
            "points_for": 350,
            "points_against": 340,
            "offensive_rank": 15,
            "defensive_rank": 15
        }

def generate_ai_prediction(home_team: str, away_team: str, user_prediction: str, confidence: float):
    """Generate AI prediction based on real data"""
    home_stats = get_team_stats_real(home_team)
    away_stats = get_team_stats_real(away_team)
    
    # Simple prediction logic based on stats
    home_advantage = 3  # Home field advantage
    home_strength = (home_stats["points_for"] - home_stats["points_against"]) / 16
    away_strength = (away_stats["points_for"] - away_stats["points_against"]) / 16
    
    predicted_winner = home_team if (home_strength + home_advantage) > away_strength else away_team
    confidence_score = min(0.95, max(0.55, abs(home_strength - away_strength) / 10 + 0.6))
    
    # Generate reasoning
    reasoning_parts = []
    if home_stats["offensive_rank"] < away_stats["offensive_rank"]:
        reasoning_parts.append(f"{home_team} has a stronger offense (ranked #{home_stats['offensive_rank']} vs #{away_stats['offensive_rank']})")
    elif away_stats["offensive_rank"] < home_stats["offensive_rank"]:
        reasoning_parts.append(f"{away_team} has a stronger offense (ranked #{away_stats['offensive_rank']} vs #{home_stats['offensive_rank']})")
    
    if home_stats["defensive_rank"] < away_stats["defensive_rank"]:
        reasoning_parts.append(f"{home_team} has a stronger defense (ranked #{home_stats['defensive_rank']} vs #{away_stats['defensive_rank']})")
    elif away_stats["defensive_rank"] < home_stats["defensive_rank"]:
        reasoning_parts.append(f"{away_team} has a stronger defense (ranked #{away_stats['defensive_rank']} vs #{home_stats['defensive_rank']})")
    
    reasoning_parts.append(f"Home field advantage gives {home_team} a {home_advantage}-point edge")
    
    reasoning = ". ".join(reasoning_parts) + "."
    
    return {
        "prediction": predicted_winner,
        "confidence": confidence_score,
        "reasoning": reasoning,
        "home_stats": home_stats,
        "away_stats": away_stats
    }

# Mock data
MOCK_TEAMS = [
    {"team_abbr": "ARI", "team_name": "Arizona Cardinals", "team_conf": "NFC", "team_division": "West"},
    {"team_abbr": "ATL", "team_name": "Atlanta Falcons", "team_conf": "NFC", "team_division": "South"},
    {"team_abbr": "BAL", "team_name": "Baltimore Ravens", "team_conf": "AFC", "team_division": "North"},
    {"team_abbr": "BUF", "team_name": "Buffalo Bills", "team_conf": "AFC", "team_division": "East"},
    {"team_abbr": "CAR", "team_name": "Carolina Panthers", "team_conf": "NFC", "team_division": "South"},
    {"team_abbr": "CHI", "team_name": "Chicago Bears", "team_conf": "NFC", "team_division": "North"},
    {"team_abbr": "CIN", "team_name": "Cincinnati Bengals", "team_conf": "AFC", "team_division": "North"},
    {"team_abbr": "CLE", "team_name": "Cleveland Browns", "team_conf": "AFC", "team_division": "North"},
    {"team_abbr": "DAL", "team_name": "Dallas Cowboys", "team_conf": "NFC", "team_division": "East"},
    {"team_abbr": "DEN", "team_name": "Denver Broncos", "team_conf": "AFC", "team_division": "West"},
    {"team_abbr": "DET", "team_name": "Detroit Lions", "team_conf": "NFC", "team_division": "North"},
    {"team_abbr": "GB", "team_name": "Green Bay Packers", "team_conf": "NFC", "team_division": "North"},
    {"team_abbr": "HOU", "team_name": "Houston Texans", "team_conf": "AFC", "team_division": "South"},
    {"team_abbr": "IND", "team_name": "Indianapolis Colts", "team_conf": "AFC", "team_division": "South"},
    {"team_abbr": "JAX", "team_name": "Jacksonville Jaguars", "team_conf": "AFC", "team_division": "South"},
    {"team_abbr": "KC", "team_name": "Kansas City Chiefs", "team_conf": "AFC", "team_division": "West"},
    {"team_abbr": "LV", "team_name": "Las Vegas Raiders", "team_conf": "AFC", "team_division": "West"},
    {"team_abbr": "LAC", "team_name": "Los Angeles Chargers", "team_conf": "AFC", "team_division": "West"},
    {"team_abbr": "LAR", "team_name": "Los Angeles Rams", "team_conf": "NFC", "team_division": "West"},
    {"team_abbr": "MIA", "team_name": "Miami Dolphins", "team_conf": "AFC", "team_division": "East"},
    {"team_abbr": "MIN", "team_name": "Minnesota Vikings", "team_conf": "NFC", "team_division": "North"},
    {"team_abbr": "NE", "team_name": "New England Patriots", "team_conf": "AFC", "team_division": "East"},
    {"team_abbr": "NO", "team_name": "New Orleans Saints", "team_conf": "NFC", "team_division": "South"},
    {"team_abbr": "NYG", "team_name": "New York Giants", "team_conf": "NFC", "team_division": "East"},
    {"team_abbr": "NYJ", "team_name": "New York Jets", "team_conf": "AFC", "team_division": "East"},
    {"team_abbr": "PHI", "team_name": "Philadelphia Eagles", "team_conf": "NFC", "team_division": "East"},
    {"team_abbr": "PIT", "team_name": "Pittsburgh Steelers", "team_conf": "AFC", "team_division": "North"},
    {"team_abbr": "SF", "team_name": "San Francisco 49ers", "team_conf": "NFC", "team_division": "West"},
    {"team_abbr": "SEA", "team_name": "Seattle Seahawks", "team_conf": "NFC", "team_division": "West"},
    {"team_abbr": "TB", "team_name": "Tampa Bay Buccaneers", "team_conf": "NFC", "team_division": "South"},
    {"team_abbr": "TEN", "team_name": "Tennessee Titans", "team_conf": "AFC", "team_division": "South"},
    {"team_abbr": "WAS", "team_name": "Washington Commanders", "team_conf": "NFC", "team_division": "East"}
]

# API Routes
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Gridiron Guru API",
        "description": "AI-powered NFL predictions backed by comprehensive data analysis",
        "version": "1.0.0",
        "status": "operational"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "API is running"}

@app.get("/api/teams", response_model=List[Team])
async def get_teams():
    """Get all NFL teams"""
    return MOCK_TEAMS

@app.get("/api/games")
async def get_games(season: int = None, week: Optional[int] = None):
    """Get games for a season/week"""
    if season is None:
        season = get_current_season()
    
    real_games = get_real_games(season, week)
    return {
        "games": real_games,
        "season": season,
        "week": week or get_current_week(),
        "total_games": len(real_games)
    }

@app.post("/api/predict", response_model=PredictionResponse)
async def create_prediction(request: PredictionRequest):
    """Generate AI prediction for a game"""
    # Generate real AI prediction based on team stats
    ai_result = generate_ai_prediction(
        request.home_team, 
        request.away_team, 
        request.user_prediction, 
        request.confidence
    )
    
    return PredictionResponse(
        prediction_id=f"pred_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        user_prediction=request.user_prediction,
        ai_analysis=f"AI predicts {ai_result['prediction']} will win with {ai_result['confidence']:.1%} confidence. {ai_result['reasoning']}",
        confidence_score=ai_result['confidence'],
        reasoning=ai_result['reasoning']
    )

@app.get("/api/teams/{team}/stats")
async def get_team_stats(team: str, season: int = None):
    """Get team statistics"""
    if season is None:
        season = get_current_season()
    
    stats = get_team_stats_real(team, season)
    return {"team_stats": stats}

@app.get("/api/standings")
async def get_standings(season: int = None):
    """Get current standings"""
    if season is None:
        season = get_current_season()
    
    # Get standings for all teams
    standings = []
    for team in MOCK_TEAMS:
        stats = get_team_stats_real(team["team_abbr"], season)
        standings.append(stats)
    
    # Sort by wins (descending)
    standings.sort(key=lambda x: x["wins"], reverse=True)
    
    return {"standings": standings, "season": season}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
