from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "synthetic_solar_daily.csv"
MODEL_PATH = ROOT / "models" / "solar_energy_model.joblib"
FEATURES = ["month", "irradiance_kwh_m2", "temperature_c", "cloud_cover_pct"]

def train():
    data = pd.read_csv(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        data[FEATURES], data["energy_index"], test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    print(f"MAE: {mean_absolute_error(y_test, predictions):.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2: {r2_score(y_test, predictions):.4f}")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved model to: {MODEL_PATH}")
    return model

if __name__ == "__main__":
    train()
