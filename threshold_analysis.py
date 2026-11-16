
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score


ROOT = Path(__file__).resolve().parent

model = joblib.load(
    ROOT / "models" / "lfcc_rbf_svm.joblib"
)

data = np.load(
    ROOT / "artifacts" / "dev_features.npz"
)

X = data["X"]
y = data["y"]

# SVM decision scores
scores = model.decision_function(X)

# Higher score means more likely spoof
# Search the full score range so the minimum cannot fall outside the grid
thresholds = np.arange(
    np.floor(scores.min()),
    np.ceil(scores.max()) + 0.001,
    0.001,
)
results = []

for threshold in thresholds:

    # Classify spoof when score >= threshold
    predictions = (scores >= threshold).astype(int)

    # Confusion matrix components
    tn = np.sum((y == 0) & (predictions == 0))
    fp = np.sum((y == 0) & (predictions == 1))
    fn = np.sum((y == 1) & (predictions == 0))
    tp = np.sum((y == 1) & (predictions == 1))

    apcer = fn / (fn + tp)
    bpcer = fp / (fp + tn)

    accuracy = accuracy_score(y, predictions)

    results.append({
        "threshold": threshold,
        "accuracy": accuracy,
        "APCER": apcer,
        "BPCER": bpcer,
        "ACER": (apcer + bpcer) / 2,
    })


df = pd.DataFrame(results)

# Save results
output_path = ROOT / "artifacts" / "threshold_results.csv"

df.to_csv(output_path, index=False)

print("\nTHRESHOLD ANALYSIS")
print("Thresholds tested:", len(df))

# Plot error rates
plt.figure(figsize=(9, 6))

plt.plot(
    df["threshold"],
    df["APCER"] * 100,
    label="APCER",
)

plt.plot(
    df["threshold"],
    df["BPCER"] * 100,
    label="BPCER",
)

plt.plot(
    df["threshold"],
    df["ACER"] * 100,
    label="ACER",
)

plt.xlabel("SVM Decision Threshold")
plt.ylabel("Error Rate (%)")

plt.title("Threshold Analysis - LFCC + RBF-SVM")

plt.legend()
plt.grid(True)

plt.savefig(
    ROOT / "artifacts" / "threshold_analysis.png",
    dpi=300,
    bbox_inches="tight",
)

# Find minimum ACER on development data
best = df.loc[df["ACER"].idxmin()]

print("\nMinimum development ACER:")
print(best)

print("\nResults saved:", output_path)

# Show the plot last so the results above print without waiting on the window
plt.show()
