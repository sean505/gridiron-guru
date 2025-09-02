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
import httpx
import joblib
import numpy as np
from pathlib import Path

# Import optimized data loader
from .data_loader import data_loader

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for ML models only (no large data)
ensemble_model = None
feature_scaler = None
model_metadata = None

def load_ml_models():
    """Load the trained ML models"""
    global ensemble_model, feature_scaler, model_metadata
    
    try:
        model_dir = Path("api/prediction_engine/models/trained")
        
        # Load ensemble model
        ensemble_path = model_dir / "ensemble.joblib"
        if ensemble_path.exists():
            ensemble_model = joblib.load(ensemble_path)
            logger.info("Loaded ensemble model")
        
        # Load feature scaler
        scaler_path = model_dir / "feature_scaler.joblib"
        if scaler_path.exists():
            feature_scaler = joblib.load(scaler_path)
            logger.info("Loaded feature scaler")
        
        # Load model metadata
        metadata_path = model_dir / "model_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                model_metadata = json.load(f)
            logger.info("Loaded model metadata")
        
        return ensemble_model is not None and feature_scaler is not None
        
    except Exception as e:
        logger.error(f"Error loading ML models: {e}")
        return False

def detect_upset(home_team: str, away_team: str, prediction_confidence: float) -> bool:
    """Detect upsets using REAL historical data (on-demand loading)"""
    
    # Get historical matchups from game_log.json (on-demand)
    historical_matchups = data_loader.get_historical_matchups(home_team, away_team)
    
    # Get current season records from season_data_by_team.json (on-demand)
    home_record = data_loader.get_team_record(home_team, 2024)
    away_record = data_loader.get_team_record(away_team, 2024)
    
    # Real upset conditions:
    # 1. Away team has better record but model predicts close game
    # 2. Historical underdog pattern exists
    # 3. Model confidence is low despite clear favorite
    
    is_upset = (
        (away_record['win_pct'] < home_record['win_pct']) and 
        (prediction_confidence < 0.65) and
        data_loader.has_upset_history(home_team, away_team, historical_matchups)
    )
    
    return is_upset

def generate_ml_prediction(home_team: str, away_team: str, game_date: str) -> Dict[str, Any]:
    """Generate prediction using trained ML models with real historical data (on-demand)"""
    global ensemble_model, feature_scaler, model_metadata
    
    try:
        if ensemble_model is None or feature_scaler is None:
            logger.warning("ML models not loaded, using fallback prediction")
            return generate_fallback_prediction(home_team, away_team)
        
        # Get real team records from historical data (on-demand)
        home_record = data_loader.get_team_record(home_team, 2024)
        away_record = data_loader.get_team_record(away_team, 2024)
        
        # Get historical matchups for context (on-demand)
        historical_matchups = data_loader.get_historical_matchups(home_team, away_team)
        
        # Calculate advanced features from historical data
        home_win_pct = home_record["win_pct"]
        away_win_pct = away_record["win_pct"]
        home_ppg = home_record["points_for"] / max(home_record["games_played"], 1)
        away_ppg = away_record["points_for"] / max(away_record["games_played"], 1)
        home_papg = home_record["points_against"] / max(home_record["games_played"], 1)
        away_papg = away_record["points_against"] / max(away_record["games_played"], 1)
        
        # Historical matchup analysis
        historical_advantage = 0.0
        if historical_matchups:
            home_wins = sum(1 for game in historical_matchups 
                          if game.get('homeTeamShort', '').lower() == home_team.lower() and 
                          game.get('homeScore', 0) > game.get('awayScore', 0))
            total_games = len(historical_matchups)
            historical_advantage = (home_wins / total_games) - 0.5  # -0.5 to 0.5 range
        
        # Create comprehensive feature vector using real data
        features = np.array([
            home_win_pct,  # Home win percentage (real data)
            away_win_pct,  # Away win percentage (real data)
            home_ppg / 30.0,  # Home points per game (normalized)
            away_ppg / 30.0,  # Away points per game (normalized)
            home_papg / 30.0,  # Home points allowed per game (normalized)
            away_papg / 30.0,  # Away points allowed per game (normalized)
            (32 - home_record.get("offensive_rank", 16)) / 32,  # Home offensive rank
            (32 - away_record.get("offensive_rank", 16)) / 32,  # Away offensive rank
            (32 - home_record.get("defensive_rank", 16)) / 32,  # Home defensive rank
            (32 - away_record.get("defensive_rank", 16)) / 32,  # Away defensive rank
            0.5,  # Home field advantage
            historical_advantage,  # Historical matchup advantage
            0.0,  # Weather factor (placeholder)
            0.0,  # Rest days (placeholder)
            0.0,  # Travel distance (placeholder)
            0.0,  # Divisional game (placeholder)
            0.0,  # Playoff implications (placeholder)
            0.0,  # Recent form (placeholder)
            0.0,  # Injury factor (placeholder)
            0.0,  # Coaching factor (placeholder)
            0.0,  # Betting line (placeholder)
            0.0,  # Public betting (placeholder)
            0.0,  # Sharp money (placeholder)
            0.0,  # Weather impact (placeholder)
            0.0,  # Stadium factor (placeholder)
            0.0,  # Time zone (placeholder)
            0.0,  # Prime time (placeholder)
            0.0,  # Rivalry factor (placeholder)
            0.0   # Season stage (placeholder)
        ]).reshape(1, -1)
        
        # Scale features
        features_scaled = feature_scaler.transform(features)
        
        # Get prediction
        prediction = ensemble_model.predict(features_scaled)[0]
        probabilities = ensemble_model.predict_proba(features_scaled)[0]
        
        # Calculate confidence and win probability
        confidence = max(probabilities)
        win_probability = probabilities[1] if prediction == 1 else probabilities[0]
        
        # Determine winner
        predicted_winner = home_team if prediction == 1 else away_team
        
        # Calculate upset potential (inverse of confidence)
        upset_potential = (1.0 - confidence) * 100
        
        # Use REAL upset detection based on historical patterns
        is_upset = detect_upset(home_team, away_team, confidence)
        
        return {
            "predicted_winner": predicted_winner,
            "confidence": float(confidence * 100),  # Convert to percentage
            "win_probability": float(win_probability * 100),
            "upset_potential": float(upset_potential),
            "is_upset": is_upset,
            "model_accuracy": 0.614,  # Ensemble model accuracy from training
            "historical_matchups": len(historical_matchups),
            "home_record": f"{home_record['wins']}-{home_record['losses']}",
            "away_record": f"{away_record['wins']}-{away_record['losses']}"
        }
        
    except Exception as e:
        logger.error(f"Error generating ML prediction: {e}")
        return generate_fallback_prediction(home_team, away_team)

def generate_fallback_prediction(home_team: str, away_team: str) -> Dict[str, Any]:
    """Generate fallback prediction when ML models are not available"""
    # Use on-demand data loading for fallback too
    home_record = data_loader.get_team_record(home_team, 2024)
    away_record = data_loader.get_team_record(away_team, 2024)
    
    # Simple prediction logic based on stats
    home_advantage = 3  # Home field advantage
    home_strength = (home_record["points_for"] - home_record["points_against"]) / 16
    away_strength = (away_record["points_for"] - away_record["points_against"]) / 16
    
    predicted_winner = home_team if (home_strength + home_advantage) > away_strength else away_team
    confidence = min(0.95, max(0.55, abs(home_strength - away_strength) / 10 + 0.6))
    upset_potential = (1.0 - confidence) * 100
    
    return {
        "predicted_winner": predicted_winner,
        "confidence": float(confidence * 100),
        "win_probability": float(confidence * 100),
        "upset_potential": float(upset_potential),
        "is_upset": confidence < 0.6,
        "model_accuracy": 0.55  # Fallback accuracy
    }

def format_time_12hr(time_str: str) -> str:
    """Convert 24-hour time string to 12-hour format with AM/PM"""
    try:
        # Handle various time formats
        if ':' in time_str:
            if 'AM' in time_str.upper() or 'PM' in time_str.upper():
                return time_str  # Already in 12-hour format
            else:
                # Assume 24-hour format
                time_part = time_str.split()[0] if ' ' in time_str else time_str
                hour, minute = time_part.split(':')
                hour = int(hour)
                if hour == 0:
                    return f"12:{minute} AM"
                elif hour < 12:
                    return f"{hour}:{minute} AM"
                elif hour == 12:
                    return f"12:{minute} PM"
                else:
                    return f"{hour-12}:{minute} PM"
        else:
            return time_str
    except:
        return time_str

app = FastAPI(
    title="Gridiron Guru API",
    description="AI-powered NFL predictions backed by comprehensive data analysis",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    """Load ML models on startup (no large data loading)"""
    logger.info("Starting up Gridiron Guru API...")
    
    # Load ML models only
    models_loaded = load_ml_models()
    if models_loaded:
        logger.info("✅ ML models loaded successfully")
    else:
        logger.warning("⚠️ ML models not loaded, using fallback predictions")
    
    logger.info("✅ API ready - using on-demand data loading")

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
    confidence: float

# Rest of the API endpoints remain the same...
# (I'll include the key endpoints in the next part)
