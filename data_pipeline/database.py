import sqlite3

class Database:
    """Class to handle SQLite database operations for NBA stats."""

    def __init__(self, db_file="nba_stats.db"):
        """Initialize the database connection."""
        self.db_file = db_file

    def connect(self):
        """Create a connection to the database."""
        return sqlite3.connect(self.db_file)

    def create_tables(self):
        """Create necessary tables in the database if they do not exist."""
        conn = self.connect()
        cursor = conn.cursor()

        cursor.execute("DROP TABLE IF EXISTS game_logs")  # ⚠️ Deletes existing data
        print("🚀 Table game_logs has been reset!")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS game_logs (
                SEASON_ID TEXT,
                PLAYER_ID INTEGER,
                PLAYER_NAME TEXT,
                GAME_ID INTEGER,
                GAME_DATE TEXT,
                MATCHUP TEXT,
                WL TEXT,
                MIN INTEGER,
                FGM INTEGER,
                FGA INTEGER,
                FG_PCT REAL,
                FG3M INTEGER,
                FG3A INTEGER,
                FG3_PCT REAL,
                FTM INTEGER,
                FTA INTEGER,
                FT_PCT REAL,
                OREB INTEGER,
                DREB INTEGER,
                REB INTEGER,
                AST INTEGER,
                STL INTEGER,
                BLK INTEGER,
                TOV INTEGER,
                PF INTEGER,
                PTS INTEGER,
                PLUS_MINUS INTEGER,
                VIDEO_AVAILABLE INTEGER,
                PRIMARY KEY (PLAYER_ID, GAME_ID)
            )
        """)

        conn.commit()
        conn.close()


    def is_table_empty(self, table_name):
        """Check if a table is empty."""
        conn = self.connect()
        cursor = conn.cursor()
        
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        
        conn.close()
        return count == 0

    def insert_game_logs(self, df):
        """Insert new game logs into the database."""
        if df.empty:
            print("⚠️ No new data to insert.")
            return

        conn = self.connect()
        df.to_sql("game_logs", conn, if_exists="append", index=False)
        conn.close()
        print("✅ New game logs inserted into the database.")

    def get_last_recorded_game_date(self):
        """Fetch the last recorded game date for a specific player."""
        conn = self.connect()
        query = "SELECT MAX(GAME_DATE) FROM game_logs"
        last_date = conn.execute(query).fetchone()[0]
        conn.close()

        return last_date if last_date else "2024-10-01"
