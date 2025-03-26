from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players
import pandas as pd
import time
from database import Database

class DataFetcher:
    """Class to fetch NBA player stats and player IDs."""

    def __init__(self, season="2024-25"):
        """Initialize the DataFetcher class with a default season."""
        self.season = season

    def fetch_nba_stats(self, initial_fetch=False):
        """
        Fetches NBA game logs.
        
        - If `initial_fetch=True`, fetches the full season for all players.
        - Otherwise, fetches only new stats since the last recorded game.
        """
        all_players = players.get_active_players()
        all_stats = []

        for player in all_players:
            player_id = player["id"]
            player_name = player["full_name"]
            print(f"Fetching data for {player_name} ({player_id})...")

            if initial_fetch:
                # Fetch all season data if it's an initial fetch
                gamelog = playergamelog.PlayerGameLog(player_id=player_id, season=self.season)
            else:
                # Fetch only new stats after the last recorded game
                last_game_date = Database.get_last_recorded_game_date(player_id)
                if last_game_date:
                    gamelog = playergamelog.PlayerGameLog(
                        player_id=player_id, season=self.season, date_from_nullable=last_game_date
                    )
                else:
                    gamelog = playergamelog.PlayerGameLog(player_id=player_id, season=self.season)

            time.sleep(0.6)  # Prevent API rate limits
            df = gamelog.get_data_frames()[0] if gamelog.get_data_frames() else pd.DataFrame()

            if not df.empty:
                df["PLAYER_NAME"] = player_name
                print(f"✅ Data fetched for {player_name}: {df.shape}")
                all_stats.append(df)

        return pd.concat(all_stats, ignore_index=True) if all_stats else pd.DataFrame()
