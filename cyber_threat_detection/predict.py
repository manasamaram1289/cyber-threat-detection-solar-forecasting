from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "cyber_text_classifier.joblib"

def predict_event(event_text: str) -> dict:
    if not isinstance(event_text, str) or not event_text.strip():
        raise ValueError("event_text must be a non-empty string")
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Train the model first: python -m cyber_threat_detection.train_model")
    model = joblib.load(MODEL_PATH)
    prediction = int(model.predict([event_text])[0])
    probabilities = model.predict_proba([event_text])[0]
    return {
        "classification": "malicious" if prediction == 1 else "benign",
        "malicious_probability": round(float(probabilities[1]), 4)
    }

if __name__ == "__main__":
    print(predict_event("unknown executable detected and suspicious login attempts"))
