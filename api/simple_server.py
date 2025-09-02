from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="Gridiron Guru API - Simple", version="1.0.0")

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

@app.get("/")
async def root():
    return {"message": "Gridiron Guru API is running!", "status": "healthy"}

@app.get("/api/games/week18-2024")
async def get_week18_2024_games():
    """Get Week 18 (final regular season week) games for 2024 season"""
    return {
        "games": [
            {
                "game_id": "2024_18_001",
                "away_team": "BUF",
                "home_team": "MIA",
                "game_date": "2024-01-07",
                "game_time": "1:00 PM",
                "week": 18,
                "season": 2024
            },
            {
                "game_id": "2024_18_002",
                "away_team": "NYJ",
                "home_team": "NE",
                "game_date": "2024-01-07",
                "game_time": "1:00 PM",
                "week": 18,
                "season": 2024
            },
            {
                "game_id": "2024_18_003",
                "away_team": "CIN",
                "home_team": "CLE",
                "game_date": "2024-01-07",
                "game_time": "1:00 PM",
                "week": 18,
                "season": 2024
            },
            {
                "game_id": "2024_18_004",
                "away_team": "PIT",
                "home_team": "BAL",
                "game_date": "2024-01-07",
                "game_time": "1:00 PM",
                "week": 18,
                "season": 2024
            },
            {
                "game_id": "2024_18_005",
                "away_team": "HOU",
                "home_team": "IND",
                "game_date": "2024-01-07",
                "game_time": "1:00 PM",
                "week": 18,
                "season": 2024
            },
            {
                "game_id": "2024_18_006",
                "away_team": "JAX",
                "home_team": "TEN",
                "game_date": "2024-01-07",
                "game_time": "1:00 PM",
                "week": 18,
                "season": 2024
            },
            {
                "game_id": "2024_18_007",
                "away_team": "DEN",
                "home_team": "LV",
                "game_date": "2024-01-07",
                "game_time": "4:25 PM",
                "week": 18,
                "season": 2024
            },
            {
                "game_id": "2024_18_008",
                "away_team": "LAC",
                "home_team": "KC",
                "game_date": "2024-01-07",
                "game_time": "4:25 PM",
                "week": 18,
                "season": 2024
            },
            {
                "game_id": "2024_18_009",
                "away_team": "DAL",
                "home_team": "WAS",
                "game_date": "2024-01-07",
                "game_time": "1:00 PM",
                "week": 18,
                "season": 2024
            },
            {
                "game_id": "2024_18_010",
                "away_team": "NYG",
                "home_team": "PHI",
                "game_date": "2024-01-07",
                "game_time": "1:00 PM",
                "week": 18,
                "season": 2024
            },
            {
                "game_id": "2024_18_011",
                "away_team": "CHI",
                "home_team": "GB",
                "game_date": "2024-01-07",
                "game_time": "4:25 PM",
                "week": 18,
                "season": 2024
            },
            {
                "game_id": "2024_18_012",
                "away_team": "DET",
                "home_team": "MIN",
                "game_date": "2024-01-07",
                "game_time": "1:00 PM",
                "week": 18,
                "season": 2024
            },
            {
                "game_id": "2024_18_013",
                "away_team": "ATL",
                "home_team": "NO",
                "game_date": "2024-01-07",
                "game_time": "1:00 PM",
                "week": 18,
                "season": 2024
            },
            {
                "game_id": "2024_18_014",
                "away_team": "CAR",
                "home_team": "TB",
                "game_date": "2024-01-07",
                "game_time": "1:00 PM",
                "week": 18,
                "season": 2024
            },
            {
                "game_id": "2024_18_015",
                "away_team": "ARI",
                "home_team": "SEA",
                "game_date": "2024-01-07",
                "game_time": "4:25 PM",
                "week": 18,
                "season": 2024
            },
            {
                "game_id": "2024_18_016",
                "away_team": "LAR",
                "home_team": "SF",
                "game_date": "2024-01-07",
                "game_time": "4:25 PM",
                "week": 18,
                "season": 2024
            }
        ],
        "season": 2024,
        "week": 18,
        "total_games": 16,
        "note": "Complete Week 18, 2024 regular season finale"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Gridiron Guru API is running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
