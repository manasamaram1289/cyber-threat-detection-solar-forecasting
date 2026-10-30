from pathlib import Path
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "synthetic_cyber_events.csv"
MODEL_PATH = ROOT / "models" / "cyber_text_classifier.joblib"

def train():
    data = pd.read_csv(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        data["event_text"], data["label"], test_size=0.25,
        random_state=42, stratify=data["label"]
    )
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), stop_words="english")),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions, target_names=["benign", "malicious"], zero_division=0))
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved model to: {MODEL_PATH}")
    return model

if __name__ == "__main__":
    train()
