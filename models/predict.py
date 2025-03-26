import pandas as pd
import joblib
import sqlite3

class Predictor:
    """Class for making predictions with the trained model."""

    def __init__(self, model_path="models/latest_model.pkl", db_file="nba_stats.db"):
        """Load trained model, scaler, and connect to the database."""
        model_data = joblib.load(model_path)
        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        self.db_file = db_file

    def get_player_stats(self, player_name):
        """Fetch the latest rolling averages for the player."""
        conn = sqlite3.connect(self.db_file)
        query = f"""
        SELECT PTS_rolling_avg, AST_rolling_avg, REB_rolling_avg, FG_PCT_rolling_avg,
               STL_rolling_avg, BLK_rolling_avg
        FROM engineered_stats
        WHERE PLAYER_NAME = ?
        ORDER BY GAME_DATE DESC
        LIMIT 1
        """
        df = pd.read_sql(query, conn, params=(player_name,))
        conn.close()
        return df

    def get_opponent_defensive_stats(self, opponent_team, home_or_away):
        """Fetch the opponent’s average defensive stats allowed."""
        conn = sqlite3.connect(self.db_file)
        query = f"""
        SELECT PTS_allowed, AST_allowed, REB_allowed, FG_PCT_allowed, STL_allowed, BLK_allowed
        FROM engineered_stats
        WHERE OPPONENT_TEAM = ? AND "HOME/AWAY" = ?
        LIMIT 1
        """
        df = pd.read_sql(query, conn, params=(opponent_team, home_or_away,))
        conn.close()
        return df

    def predict(self, player_name, opponent_team, home_or_away):
        """Predict player stat line for the given opponent."""
        # Convert home_or_away into valid form: 0/1
        if home_or_away.lower() == "home":
            home_or_away = "0"
        elif home_or_away.lower() == "away":
            home_or_away = "1"
        
        # Fetch the player's latest rolling averages
        player_stats = self.get_player_stats(player_name)
        if player_stats.empty:
            print(f"⚠️ No recent stats found for {player_name}.")
            return None

        # Fetch the opponent's defensive stats
        opponent_stats = self.get_opponent_defensive_stats(opponent_team, home_or_away)
        if opponent_stats.empty:
            print(f"⚠️ No defensive stats found for {opponent_team}.")
            return None

        # Combine both data sources into one input row
        input_data = pd.concat([player_stats, opponent_stats], axis=1)

        # Scale input data
        X_scaled = self.scaler.transform(input_data)

        # Make prediction
        predictions = self.model.predict(X_scaled)

        # Return predictions as a DataFrame
        return pd.DataFrame(predictions, columns=["PTS", "AST", "TRB", "FG_PCT", "STL", "BLK"])
