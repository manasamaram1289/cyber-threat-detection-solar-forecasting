from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "solar_energy_model.joblib"
FEATURES = ["month", "irradiance_kwh_m2", "temperature_c", "cloud_cover_pct"]

def predict_energy(month: int, irradiance_kwh_m2: float,
                   temperature_c: float, cloud_cover_pct: float) -> float:
    if not 1 <= month <= 12:
        raise ValueError("month must be between 1 and 12")
    if irradiance_kwh_m2 < 0 or not 0 <= cloud_cover_pct <= 100:
        raise ValueError("irradiance must be non-negative and cloud cover must be 0-100")
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Train the model first: python -m solar_forecasting.train_forecast")
    model = joblib.load(MODEL_PATH)
    sample = pd.DataFrame([{
        "month": month,
        "irradiance_kwh_m2": irradiance_kwh_m2,
        "temperature_c": temperature_c,
        "cloud_cover_pct": cloud_cover_pct
    }], columns=FEATURES)
    return round(float(model.predict(sample)[0]), 3)

if __name__ == "__main__":
    print(predict_energy(5, 5.8, 31.0, 20.0))
