# Telemetry Anomaly Detection

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)  [![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)  [![MIT License](https://img.shields.io/badge/License-MIT-2ea44f?style=for-the-badge)](LICENSE)  ![Synthetic Data](https://img.shields.io/badge/Data-Synthetic-6b7280?style=for-the-badge)

Flags unusual CPU, memory, API latency, and traffic measurements with an Isolation Forest trained on normal observations.

## Data

Deterministic synthetic data is generated locally in `pipeline.py`. No private data or external download is required.

## Install and run

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
python cli.py evaluate
python cli.py train --model model.joblib
python cli.py predict --model model.joblib
```

## Evaluation results

With seed 42, the Isolation Forest (contamination 0.05) was fit on 1,350 normal training rows. On a 500-row holdout containing 50 known anomalies, precision was 0.6338, recall was 0.9000, and F1 was 0.7438. The synthetic generator defines four anomaly patterns; these results do not measure performance on real telemetry.

## Limitations

The generated patterns are simplified and may not reflect real services. Validate thresholds and anomaly behavior on representative, appropriately licensed telemetry before relying on alerts.
