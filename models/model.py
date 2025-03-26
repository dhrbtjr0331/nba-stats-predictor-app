import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib  # For saving and loading models


class Model:
    """Class for training regression models to predict player stat lines"""

    def __init__(self, df):
        """Initialize"""
        self.df = df.copy()
        self.model = None
        self.scaler = StandardScaler()

    def train_regression_model(self, model_type="linear"):
        """
        Trains a regression model to predict PTS, AST, TRB, FG%, STL, BLK
        Available model types: "linear", "random_forest", "xgboost"
        """
        # Define target (Y) and features (X)
        target_columns = ["PTS", "AST", "TRB", "FG%", "STL", "BLK"]
        feature_columns = [
            "PTS_rolling_avg", "AST_rolling_avg", "TRB_rolling_avg", "FG%_rolling_avg",
            "STL_rolling_avg", "BLK_rolling_avg", "PTS_allowed", "AST_allowed",
            "TRB_allowed", "FG_PCT_allowed_allowed", "STL_allowed", "BLK_allowed", "HOME/AWAY"
        ]

        # Split into features (X) and target (Y)
        X = self.df[feature_columns]
        y = self.df[target_columns]

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42)

        # Chosse model type
        if model_type == "linear":
            model = MultiOutputRegressor(LinearRegression())
        elif model_type == "random_forest":
            model = MultiOutputRegressor(
                RandomForestRegressor(n_estimators=100, random_state=42))
        elif model_type == "xgboost":
            model = MultiOutputRegressor(XGBRegressor(
                objective="reg:squarederror", n_estimators=100))
        else:
            raise ValueError(
                "Invalid model type. Choose from 'linear', 'random_forest', 'xgboost'.")

        # Train model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Evaluate model
        mae = mean_absolute_error(y_test, y_pred, multioutput='raw_values')
        r2 = r2_score(y_test, y_pred, multioutput='raw_values')

        # Store trained model
        self.model = model

        # Print evaluation results
        results_df = pd.DataFrame(
            {"Stat": target_columns, "MAE": mae, "R^2 Score": r2})
        print(results_df)

        return results_df

    def save_model(self, filename="trained_model.pkl"):
        """Save trained model and scaler for later use."""
        if self.model:
            joblib.dump({"model": self.model, "scaler": self.scaler}, filename)
            print(f"Model saved as {filename}.")
        else:
            print("No trained model to save!")

    def load_model(self, filename="trained_model.pkl"):
        """Load a previously saved model."""
        model_data = joblib.load(filename)
        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        print(f"Model loaded from {filename}.")