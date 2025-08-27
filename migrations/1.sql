
CREATE TABLE user_predictions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id TEXT NOT NULL,
  game_id TEXT NOT NULL,
  predicted_winner TEXT NOT NULL,
  ai_predicted_winner TEXT,
  ai_confidence REAL,
  is_upset_pick BOOLEAN DEFAULT FALSE,
  locked_at TEXT,
  actual_winner TEXT,
  user_correct BOOLEAN,
  ai_correct BOOLEAN,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
