
#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
)

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", default=None)
args = parser.parse_args()

df = pd.read_csv(args.input)

required = ["true_label", "interaction_probability", "predicted_label"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

if df[required].isna().any().any():
    raise ValueError("Missing labels or probabilities detected")

y_true = df["true_label"].astype(int).to_numpy()
y_pred = df["predicted_label"].astype(int).to_numpy()
y_prob = df["interaction_probability"].astype(float).to_numpy()

if not set(np.unique(y_true)).issubset({0, 1}):
    raise ValueError("true_label must contain only 0 and 1")
if not set(np.unique(y_pred)).issubset({0, 1}):
    raise ValueError("predicted_label must contain only 0 and 1")
if len(np.unique(y_true)) != 2:
    raise ValueError("Both classes are required to calculate ROC-AUC")
if not np.isfinite(y_prob).all() or ((y_prob < 0) | (y_prob > 1)).any():
    raise ValueError("Invalid interaction probabilities")

tn, fp, fn, tp = confusion_matrix(
    y_true, y_pred, labels=[0, 1]
).ravel()

results = {
    "n_samples": len(df),
    "positive": int(sum(y_true == 1)),
    "negative": int(sum(y_true == 0)),
    "acc": float(accuracy_score(y_true, y_pred)),
    "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    "auc": float(roc_auc_score(y_true, y_prob)),
    "aupr": float(average_precision_score(y_true, y_prob)),
    "precision": float(precision_score(y_true, y_pred, zero_division=0)),
    "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    "tp": int(tp),
    "tn": int(tn),
    "fp": int(fp),
    "fn": int(fn),
}

output = Path(args.output) if args.output else Path(args.input).with_suffix(".metrics.json")
output.write_text(json.dumps(results, indent=2))

print(json.dumps(results, indent=2))
print(f"Saved: {output}")

# python evaluate_prediction_PepLM_GNN.py \
#  --input Test3664_predictions_PepLM_GNN.csv