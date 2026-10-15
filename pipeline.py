"""Unsupervised telemetry anomaly detection on deterministic synthetic data."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
FEATURES = ["cpu_percent", "ram_percent", "api_latency_ms", "requests_per_minute"]


def _dataset(seed=SEED, normal_rows=1800, anomaly_rows=200):
    rng = np.random.default_rng(seed)
    normal = np.column_stack([
        np.clip(rng.normal(38, 12, normal_rows), 1, 95),
        np.clip(rng.normal(58, 9, normal_rows), 5, 95),
        np.clip(rng.normal(140, 35, normal_rows), 30, 300),
        np.clip(rng.normal(115, 28, normal_rows), 10, 240),
    ])
    anomalies = np.empty((anomaly_rows, 4))
    for i in range(anomaly_rows):
        mode = i % 4
        if mode == 0:  # CPU saturation
            anomalies[i] = [rng.uniform(90, 100), rng.normal(65, 8), rng.uniform(400, 900), rng.normal(125, 20)]
        elif mode == 1:  # memory pressure
            anomalies[i] = [rng.normal(48, 10), rng.uniform(94, 100), rng.uniform(250, 600), rng.normal(110, 25)]
        elif mode == 2:  # latency spike
            anomalies[i] = [rng.normal(45, 12), rng.normal(60, 9), rng.uniform(700, 1600), rng.normal(100, 25)]
        else:  # traffic burst
            anomalies[i] = [rng.uniform(72, 95), rng.normal(72, 10), rng.uniform(250, 650), rng.uniform(260, 500)]
    X = pd.DataFrame(np.vstack([normal, anomalies]), columns=FEATURES)
    y = np.r_[np.zeros(normal_rows, dtype=int), np.ones(anomaly_rows, dtype=int)]
    order = rng.permutation(len(y))
    return X.iloc[order].reset_index(drop=True), y[order]


def _model(seed=SEED):
    return make_pipeline(StandardScaler(), IsolationForest(n_estimators=250, contamination=0.05, random_state=seed, n_jobs=1))


def sample_input():
    return {"cpu_percent": 98, "ram_percent": 96, "api_latency_ms": 820, "requests_per_minute": 315}


def train_model(seed=SEED):
    X, y = _dataset(seed)
    return _model(seed).fit(X.loc[y == 0])


def predict(model, payload):
    row = pd.DataFrame([payload], columns=FEATURES)
    flagged = int(model.predict(row)[0] == -1)
    return {"anomaly": bool(flagged), "label": "anomaly" if flagged else "normal",
            "decision_score": round(float(model.decision_function(row)[0]), 4)}


def evaluate(seed=SEED):
    X, y = _dataset(seed)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)
    model = _model(seed).fit(x_train.loc[y_train == 0])
    pred = (model.predict(x_test) == -1).astype(int)
    return {
        "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, pred, zero_division=0)), 4),
        "normal_training_rows": int((y_train == 0).sum()),
        "test_rows": int(len(y_test)),
        "known_anomaly_test_rows": int(y_test.sum()),
    }
