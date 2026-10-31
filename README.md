# Cyber Threat Detection & Solar Energy Forecasting

A beginner-friendly Python machine learning project featuring two independent experiments.

1. **Cyber threat text classification** — TF-IDF features and Logistic Regression classify short synthetic security-event descriptions as benign or potentially malicious.
2. **Solar energy index estimation** — a Random Forest Regressor estimates a synthetic energy index from month, irradiance, temperature and cloud-cover inputs.

> **Important data note:** Both CSV datasets are generated synthetic teaching data. They are not real security logs or measured solar-generation records. Metrics from these datasets are demonstrations only and must not be presented as validated real-world results.

## Tech stack

Python, Pandas, scikit-learn, TF-IDF, Logistic Regression, Random Forest, Joblib and Pytest.

## Folder structure

```text
cyber-threat-detection-solar-forecasting/
├── cyber_threat_detection/
│   ├── train_model.py
│   └── predict.py
├── solar_forecasting/
│   ├── train_forecast.py
│   └── predict.py
├── data/
│   ├── synthetic_cyber_events.csv
│   └── synthetic_solar_daily.csv
├── tests/
│   └── test_project.py
├── requirements.txt
└── README.md
```

## 1. Set up

Create and activate a virtual environment.

**Windows**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux**
```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 2. Cyber threat classification

Train and evaluate the text classifier:

```bash
python -m cyber_threat_detection.train_model
```

Run a sample prediction:

```bash
python -m cyber_threat_detection.predict
```

The training command prints accuracy, a confusion matrix, precision, recall and F1-score. This is a baseline model, not a production intrusion-detection system.

## 3. Solar forecasting

Train and evaluate the regression model:

```bash
python -m solar_forecasting.train_forecast
```

Run a sample prediction:

```bash
python -m solar_forecasting.predict
```

The training command prints MAE, RMSE and R-squared. The target is a synthetic energy index, not actual kWh production.

## 4. Run tests

```bash
pytest -q
```

## Limitations and next steps

- Replace synthetic data with appropriately licensed, representative datasets before making real-world claims.
- Validate on a time-based holdout for forecasting and an independently sourced test set for threat detection.
- Review false positives and false negatives before any security use.
- ANN, CNN and LSTM models mentioned in some project descriptions are not implemented in this baseline.
- The reported resume metrics (such as 92% cyber accuracy or 90% solar accuracy) are not reproduced or claimed here.

## Portfolio integrity

This repository is a fresh educational implementation based on a short project summary. It should be described as a recreated/demo implementation unless it is verified against the original work. Only claim techniques and results you can explain and demonstrate.
