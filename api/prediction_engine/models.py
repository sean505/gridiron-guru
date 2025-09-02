"""
Pydantic models for the Gridiron Guru prediction engine.

This module defines the data structures used throughout the prediction system,
ensuring type safety and validation for all NFL data and predictions.
"""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class GameStatus(str, Enum):
    """Enumeration for game statuses."""
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    FINAL = "final"
    POSTPONED = "postponed"
    CANCELLED = "cancelled"


class TeamStats(BaseModel):
    """Comprehensive team statistics model."""
    
    # Basic team info
    team_abbr: str = Field(..., description="Team abbreviation (e.g., 'KC', 'BUF')")
    team_name: str = Field(..., description="Full team name")
    season: int = Field(..., description="NFL season year")
    week: int = Field(..., description="Current week number")
    
    # Offensive metrics
    offensive_epa: float = Field(default=0.0, description="Expected Points Added on offense")
    passing_epa: float = Field(default=0.0, description="Expected Points Added through passing")
    rushing_epa: float = Field(default=0.0, description="Expected Points Added through rushing")
    red_zone_efficiency: float = Field(default=0.0, description="Red zone touchdown percentage")
    third_down_conversion: float = Field(default=0.0, description="Third down conversion rate")
    
    # Defensive metrics
    defensive_epa: float = Field(default=0.0, description="Expected Points Added on defense")
    pass_defense_epa: float = Field(default=0.0, description="Expected Points Added against pass")
    rush_defense_epa: float = Field(default=0.0, description="Expected Points Added against rush")
    turnover_margin: float = Field(default=0.0, description="Turnover differential")
    sack_rate: float = Field(default=0.0, description="Sack rate percentage")
    
    # Team performance metrics
    win_percentage: float = Field(default=0.0, description="Season win percentage")
    point_differential: float = Field(default=0.0, description="Points scored minus points allowed")
    strength_of_schedule: float = Field(default=0.0, description="Opponent strength rating")
    recent_form: float = Field(default=0.0, description="Performance in last 4 games")
    
    # Situational metrics
    home_record: Dict[str, int] = Field(default_factory=dict, description="Home game record")
    away_record: Dict[str, int] = Field(default_factory=dict, description="Away game record")
    division_record: Dict[str, int] = Field(default_factory=dict, description="Division game record")
    
    # Advanced metrics
    pythagorean_wins: float = Field(default=0.0, description="Expected wins based on point differential")
    luck_factor: float = Field(default=0.0, description="Difference between actual and expected wins")
    injury_impact: float = Field(default=0.0, description="Impact of key player injuries")
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class GameContext(BaseModel):
    """Contextual information for a specific game."""
    
    game_id: str = Field(..., description="Unique game identifier")
    season: int = Field(..., description="NFL season year")
    week: int = Field(..., description="Week number")
    game_type: str = Field(default="REG", description="Game type (REG, WC, DIV, CONF, SB)")
    
    # Teams
    home_team: str = Field(..., description="Home team abbreviation")
    away_team: str = Field(..., description="Away team abbreviation")
    
    # Game details
    game_date: datetime = Field(..., description="Game date and time")
    stadium: str = Field(default="", description="Stadium name")
    surface: str = Field(default="grass", description="Playing surface type")
    roof: str = Field(default="outdoors", description="Stadium roof type")
    
    # Weather conditions
    temperature: Optional[float] = Field(default=None, description="Temperature in Fahrenheit")
    wind_speed: Optional[float] = Field(default=None, description="Wind speed in mph")
    precipitation: Optional[float] = Field(default=None, description="Precipitation probability")
    
    # Betting lines
    spread: Optional[float] = Field(default=None, description="Point spread")
    total: Optional[float] = Field(default=None, description="Over/under total")
    home_moneyline: Optional[int] = Field(default=None, description="Home team moneyline")
    away_moneyline: Optional[int] = Field(default=None, description="Away team moneyline")
    
    # Game status
    status: GameStatus = Field(default=GameStatus.SCHEDULED, description="Current game status")
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class GamePrediction(BaseModel):
    """Comprehensive game prediction model."""
    
    # Game identification
    game_id: str = Field(..., description="Unique game identifier")
    home_team: str = Field(..., description="Home team abbreviation")
    away_team: str = Field(..., description="Away team abbreviation")
    
    # Prediction results
    predicted_winner: str = Field(..., description="Predicted winning team abbreviation")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence (0-1)")
    win_probability: float = Field(..., ge=0.0, le=1.0, description="Win probability for predicted winner")
    
    # Score prediction
    predicted_home_score: float = Field(..., description="Predicted home team score")
    predicted_away_score: float = Field(..., description="Predicted away team score")
    predicted_total: float = Field(..., description="Predicted total points")
    predicted_spread: float = Field(..., description="Predicted point spread")
    
    # Upset analysis
    is_upset_pick: bool = Field(default=False, description="Whether this is considered an upset")
    upset_probability: float = Field(default=0.0, ge=0.0, le=1.0, description="Probability of upset occurring")
    upset_factors: List[str] = Field(default_factory=list, description="Factors contributing to upset potential")
    
    # Key factors
    key_factors: List[str] = Field(default_factory=list, description="Key factors influencing prediction")
    explanation: str = Field(..., description="Human-readable explanation of prediction")
    
    # Model metadata
    model_version: str = Field(default="1.0.0", description="Prediction model version")
    prediction_timestamp: datetime = Field(default_factory=datetime.now, description="When prediction was made")
    data_freshness: datetime = Field(..., description="When underlying data was last updated")
    
    # Confidence breakdown
    offensive_advantage: float = Field(default=0.0, description="Offensive advantage score")
    defensive_advantage: float = Field(default=0.0, description="Defensive advantage score")
    situational_advantage: float = Field(default=0.0, description="Situational advantage score")
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class WeeklyPredictions(BaseModel):
    """Collection of predictions for a specific week."""
    
    season: int = Field(..., description="NFL season year")
    week: int = Field(..., description="Week number")
    prediction_date: datetime = Field(default_factory=datetime.now, description="When predictions were generated")
    
    # Predictions
    games: List[GamePrediction] = Field(default_factory=list, description="List of game predictions")
    
    # Summary statistics
    total_games: int = Field(..., description="Total number of games predicted")
    upset_picks: int = Field(default=0, description="Number of upset picks")
    average_confidence: float = Field(default=0.0, description="Average prediction confidence")
    
    # Model performance (if available)
    historical_accuracy: Optional[float] = Field(default=None, description="Historical accuracy of model")
    last_updated: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class PredictionRequest(BaseModel):
    """Request model for generating predictions."""
    
    season: int = Field(..., description="NFL season year")
    week: Optional[int] = Field(default=None, description="Specific week (None for current week)")
    include_upsets: bool = Field(default=True, description="Whether to include upset analysis")
    confidence_threshold: float = Field(default=0.6, ge=0.0, le=1.0, description="Minimum confidence threshold")
    
    # Optional filters
    teams: Optional[List[str]] = Field(default=None, description="Filter by specific teams")
    game_types: Optional[List[str]] = Field(default=None, description="Filter by game types")
    
    class Config:
        """Pydantic configuration."""
        schema_extra = {
            "example": {
                "season": 2024,
                "week": 18,
                "include_upsets": True,
                "confidence_threshold": 0.7
            }
        }


class PredictionResponse(BaseModel):
    """Response model for prediction API endpoints."""
    
    success: bool = Field(..., description="Whether the prediction was successful")
    predictions: Optional[WeeklyPredictions] = Field(default=None, description="Generated predictions")
    error_message: Optional[str] = Field(default=None, description="Error message if prediction failed")
    processing_time: float = Field(default=0.0, description="Time taken to generate predictions")
    
    class Config:
        """Pydantic configuration."""
        schema_extra = {
            "example": {
                "success": True,
                "predictions": {
                    "season": 2024,
                    "week": 18,
                    "total_games": 16,
                    "upset_picks": 3,
                    "average_confidence": 0.72
                },
                "processing_time": 2.34
            }
        }
