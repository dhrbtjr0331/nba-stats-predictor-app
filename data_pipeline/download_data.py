import time
from database import Database
from fetch_data import DataFetcher
from preprocess_data import FeatureEngineer

def main():
    # Initialize Database and DataFetcher
    db = Database()
    db.create_tables()
    
    fetcher = DataFetcher()


    while True:
        print("🔄 Fetching latest NBA stats...")
        if db.is_table_empty("game_logs"):
            new_data = fetcher.fetch_nba_stats(initial_fetch=True)
        else:
            new_data = fetcher.fetch_nba_stats(initial_fetch=False)

        if new_data.empty:
            print("⚠️ No new game logs available. Sleeping for 24 hours...")
        else:
            print("✅ New game logs fetched. Processing data...")
            
            # Feature Engineering
            processor = FeatureEngineer(new_data)
            processor.get_rolling_average().get_opponent_defensive_stats()
            processor.add_home_away_column().add_opponent_team_column()
            
            # Save raw data to database
            db.insert_game_logs(new_data)

            # Save processed data to the database
            processor.save_to_database()

            print("✅ Database updated successfully.")

        # Run the script daily
        time.sleep(86400)  # Sleep for 24 hours before running again

if __name__ == "__main__":
    main()
