
-- Insert dummy user predictions to show stats functionality
INSERT INTO user_predictions (user_id, game_id, predicted_winner, ai_predicted_winner, ai_confidence, is_upset_pick, actual_winner, user_correct, ai_correct) VALUES
('user_demo123', 'completed_1', 'TB', 'TB', 0.68, false, 'TB', true, true),
('user_demo123', 'completed_2', 'CIN', 'PIT', 0.59, true, 'CIN', true, false),
('user_demo123', 'game_1', 'BUF', 'KC', 0.72, true, null, null, null),
('user_demo123', 'game_2', 'SF', 'DAL', 0.58, true, null, null, null),
('user_demo123', 'game_3', 'PHI', 'PHI', 0.85, false, null, null, null);
