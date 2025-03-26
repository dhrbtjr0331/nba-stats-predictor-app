import pandas as pd
import sqlite3  # Use PostgreSQL in production

class FeatureEngineer:
    """Class for feature engineering on raw NBA data."""

    def __init__(self, df):
        """Initialize with raw NBA data."""
        self.df = df.copy()

    def add_home_away_column(self):
        """Adds home/away column"""
        df = self.df
        df.columns = df.columns.str.strip()

        # Create the 'Home/Away' column based on the presence of '@' in the 'MATCHUP' column
        df["HOME/AWAY"] = df["MATCHUP"].apply(lambda x: "1" if "@" in x else "0")

        self.df = df
        return self 
    
    def add_opponent_team_column(self):
        """Adds opponent team column"""
        df = self.df
        df.columns = df.columns.str.strip()

        # Extract the last three characters (opponent team)
        df["OPPONENT_TEAM"] = df["MATCHUP"].apply(lambda x: x[-3:])

        self.df = df
        return self

    def get_rolling_average(self, window_size=5):
        """Computes rolling averages for key stats over a selected window size."""
        df = self.df
        df.columns = df.columns.str.strip()

        print("🔍 Available columns before sorting:", df.columns.tolist())  # Debugging

        if "PLAYER_NAME" not in df.columns:
            raise KeyError("🚨 'PLAYER_NAME' is missing! Check fetch_nba_stats().")
        
        # Convert date to datetime and sort
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], format="%b %d, %Y")
        df = df.sort_values(by=["PLAYER_NAME", "GAME_DATE"])

        # Define stat columns
        stat_columns = ["PTS", "AST", "REB", "FG_PCT", "STL", "BLK"]

        # Compute rolling averages per player
        for col in stat_columns:
            df[f"{col}_rolling_avg"] = (
                df.groupby("PLAYER_NAME")[col]
                .rolling(window=window_size, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )

        self.df = df
        return self

    def get_opponent_defensive_stats(self):
        """Computes opponent defensive stats (average stats allowed per game)."""
        df = self.df

        # Ensure correct column names
        df.columns = df.columns.str.strip()
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

        # Define stat columns
        stat_columns = ["PTS", "AST", "REB", "STL", "BLK"]

        # Compute opponent stats allowed
        opponent_stats = df.groupby("MATCHUP")[stat_columns].mean()

        # Compute FG% allowed
        fg_stats = df.groupby("MATCHUP")[["FGM", "FGA"]].sum()
        fg_stats["FG_PCT_allowed"] = fg_stats["FGM"] / fg_stats["FGA"]

        # Merge FG% into opponent stats
        opponent_stats = opponent_stats.merge(fg_stats[["FG_PCT_allowed"]], on="MATCHUP")

        # Rename columns to indicate they represent opponent stats allowed
        opponent_stats = opponent_stats.rename(columns=lambda x: f"{x}_allowed" if x != "FG_PCT_allowed" else x).reset_index()

        # Merge back into the dataset
        self.df = self.df.merge(opponent_stats, on="MATCHUP", how="left")
        return self

    def save_to_database(self, db_file="nba_stats.db", table_name="engineered_stats"):
        """Saves the processed data into a SQLite database."""
        conn = sqlite3.connect(db_file)
        self.df.to_sql(table_name, conn, if_exists="append", index=False)
        conn.close()
        print(f"✅ Engineered data saved to {table_name} in {db_file}")

    def get_engineered_data(self):
        """Returns the final dataset with engineered features."""
        return self.df
