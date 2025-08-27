from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import json

app = FastAPI(title="Gridiron Guru API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mock data for now (we'll integrate nfl_data_py later)
MOCK_TEAMS = [
    {"team_abbr": "BUF", "team_name": "Buffalo Bills", "team_conf": "AFC", "team_division": "East"},
    {"team_abbr": "KC", "team_name": "Kansas City Chiefs", "team_conf": "AFC", "team_division": "West"},
    {"team_abbr": "SF", "team_name": "San Francisco 49ers", "team_conf": "NFC", "team_division": "West"},
    {"team_abbr": "DAL", "team_name": "Dallas Cowboys", "team_conf": "NFC", "team_division": "East"},
]

MOCK_GAMES = [
    {"game_id": "1", "away_team": "BUF", "home_team": "KC", "game_date": "2024-01-15", "game_time": "8:00 PM"},
    {"game_id": "2", "away_team": "SF", "home_team": "DAL", "game_date": "2024-01-16", "game_time": "8:00 PM"},
]

@app.get("/")
async def root():
    return {"message": "Gridiron Guru API is running!", "status": "healthy"}

@app.get("/api/teams")
async def get_teams():
    return {"teams": MOCK_TEAMS}

@app.get("/api/games")
async def get_games():
    return {"games": MOCK_GAMES}

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "message": "Gridiron Guru API is running"}

# Vercel requires this for Python functions
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
