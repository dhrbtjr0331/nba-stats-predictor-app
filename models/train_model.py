import pandas as pd
import sqlite3
from model_trainer import ModelTrainer

def load_data():
    """Load engineered data from database."""
    conn = sqlite3.connect("nba_stats.db")
    query = "SELECT * FROM engineered_stats"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def main():
    """Train the model"""
    print("📊 Loading data for training...")
    df = load_data()

    trainer = ModelTrainer(df)
    trainer.train_regression_model(model_type="xgboost")  # Change model type if needed
    trainer.save_model()

if __name__ == "__main__":
    main()
