import pandas as pd
import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_cyber_dataset_schema_and_labels():
    data = pd.read_csv(ROOT / "data" / "synthetic_cyber_events.csv")
    assert {"event_id", "event_text", "label"}.issubset(data.columns)
    assert len(data) >= 100
    assert set(data["label"].unique()) == {0, 1}

def test_solar_dataset_schema_and_size():
    data = pd.read_csv(ROOT / "data" / "synthetic_solar_daily.csv")
    assert {"month", "irradiance_kwh_m2", "temperature_c", "cloud_cover_pct", "energy_index"}.issubset(data.columns)
    assert len(data) == 365
    assert data["cloud_cover_pct"].between(0, 100).all()

def test_solar_prediction_rejects_invalid_month():
    from solar_forecasting.predict import predict_energy
    with pytest.raises(ValueError):
        predict_energy(13, 5.0, 25.0, 20.0)
