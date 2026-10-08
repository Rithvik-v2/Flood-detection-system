"""Generate an explicitly simulated dataset and train an educational classifier."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent
FEATURES = ["rainfall", "water_level", "temperature", "humidity"]
LABELS = ["LOW", "MEDIUM", "HIGH"]
SCENARIOS = {
    "Normal Conditions": [8.0, 1.0, 28.0, 60.0],
    "Moderate Rainfall": [70.0, 3.5, 26.0, 80.0],
    "Heavy Rainfall": [180.0, 7.5, 24.0, 95.0],
}


def generate_data():
    rng = np.random.default_rng(42)
    size = 4500
    data = pd.DataFrame({
        "rainfall": rng.uniform(0, 300, size).round(2),
        "water_level": rng.uniform(0, 10, size).round(2),
        "temperature": rng.uniform(10, 45, size).round(2),
        "humidity": rng.uniform(20, 100, size).round(2),
    })
    # Teaching assumptions only: rainfall is an hourly depth and water level
    # has an imaginary local reference. These are NOT validated flood thresholds.
    score = (data.rainfall / 100 + data.water_level / 3
             + (data.humidity >= 85) * 0.25 + (data.temperature <= 25) * 0.1)
    data["risk"] = np.select([score >= 3.2, score >= 1.5], ["HIGH", "MEDIUM"], default="LOW")
    data["data_source"] = "SIMULATED DATA"
    return data


def train():
    (ROOT / "data").mkdir(exist_ok=True)
    data = generate_data()
    data.to_csv(ROOT / "data/flood_data.csv", index=False)
    x_train, x_test, y_train, y_test = train_test_split(
        data[FEATURES], data.risk, test_size=0.25, random_state=42, stratify=data.risk)
    # Trees use raw numeric features; the same named columns are used in the app.
    model = RandomForestClassifier(n_estimators=160, random_state=42, n_jobs=2,
                                   class_weight="balanced")
    model.fit(x_train, y_train)
    predicted = model.predict(x_test)
    metrics = {
        "data_source": "SIMULATED DATA; educational rule-generated labels",
        "accuracy": accuracy_score(y_test, predicted),
        "train_rows": len(x_train), "test_rows": len(x_test),
        "labels": LABELS,
        "classification_report": classification_report(y_test, predicted, labels=LABELS, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, predicted, labels=LABELS).tolist(),
    }
    joblib.dump(model, ROOT / "flood_model.pkl")
    (ROOT / "data/model_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Simulated holdout accuracy: {metrics['accuracy']:.4f}")
    print(classification_report(y_test, predicted, labels=LABELS))
    print("Confusion matrix (LOW, MEDIUM, HIGH):", metrics["confusion_matrix"])
    for name, values in SCENARIOS.items():
        print(name, model.predict(pd.DataFrame([values], columns=FEATURES))[0])


if __name__ == "__main__":
    train()
