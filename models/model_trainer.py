import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score


class ModelTrainer:
    """Class for training regression models to predict NBA player stat lines."""

    def __init__(self, df):
        """Initialize ModelTrainer with data."""
        self.df = df.copy()
        self.model = None
        self.scaler = StandardScaler()

    def train_regression_model(self, model_type="linear"):
        """
        Trains a regression model to predict PTS, AST, TRB, FG%, STL, BLK
        Available models: "linear", "random_forest", "xgboost"
        """
        target_columns = ["PTS", "AST", "REB", "FG_PCT", "STL", "BLK"]
        feature_columns = [
            "PTS_rolling_avg", "AST_rolling_avg", "REB_rolling_avg", "FG_PCT_rolling_avg",
            "STL_rolling_avg", "BLK_rolling_avg", "PTS_allowed", "AST_allowed",
            "REB_allowed", "FG_PCT_allowed", "STL_allowed", "BLK_allowed"
        ]

        # Check if required columns exist
        missing_features = [col for col in feature_columns if col not in self.df.columns]
        missing_targets = [col for col in target_columns if col not in self.df.columns]
        if missing_features or missing_targets:
            raise ValueError(f"Missing required columns: {missing_features + missing_targets}")

        # Prepare feature and target data
        X = self.df[feature_columns]
        y = self.df[target_columns]

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        # Choose model type
        models = {
            "linear": MultiOutputRegressor(LinearRegression()),
            "random_forest": MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42)),
            "xgboost": MultiOutputRegressor(XGBRegressor(objective="reg:squarederror", n_estimators=100))
        }

        if model_type not in models:
            raise ValueError("Invalid model type. Choose from 'linear', 'random_forest', 'xgboost'.")

        self.model = models[model_type]

        # Train model
        self.model.fit(X_train, y_train)

        # Evaluate model
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred, multioutput="raw_values")
        r2 = r2_score(y_test, y_pred, multioutput="raw_values")

        # Display evaluation results
        results_df = pd.DataFrame({"Stat": target_columns, "MAE": mae, "R^2 Score": r2})
        print("📊 Model Evaluation Results:")
        print(results_df)

        return results_df

    def save_model(self, filename="models/latest_model.pkl"):
        """Save trained model and scaler."""
        if self.model:
            joblib.dump({"model": self.model, "scaler": self.scaler}, filename)
            print(f"✅ Model saved as {filename}.")
        else:
            print("⚠️ No trained model to save!")

    def load_model(self, filename="models/latest_model.pkl"):
        """Load a previously saved model."""
        model_data = joblib.load(filename)
        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        print(f"✅ Model loaded from {filename}.")
