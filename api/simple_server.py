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
                "game_id": "test1",
                "away_team": "BUF",
                "home_team": "KC",
                "game_date": "2024-01-07",
                "game_time": "8:20 PM",
                "week": 18,
                "season": 2024
            },
            {
                "game_id": "test2", 
                "away_team": "SF",
                "home_team": "DAL",
                "game_date": "2024-01-07",
                "game_time": "4:25 PM",
                "week": 18,
                "season": 2024
            }
        ],
        "season": 2024,
        "week": 18,
        "total_games": 2,
        "note": "Test data for Week 18, 2024"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Gridiron Guru API is running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
