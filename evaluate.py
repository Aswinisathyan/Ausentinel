
from pathlib import Path

import joblib
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import brentq

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "models" / "lfcc_rbf_svm.joblib"

DEV_FEATURES_PATH = (
    ROOT / "artifacts" / "dev_features.npz"
)

ARTIFACT_DIR = ROOT / "artifacts"

ARTIFACT_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# 2. LOAD MODEL AND DEVELOPMENT FEATURES
# --------------------------------------------------

print("\nLoading trained model...")

model = joblib.load(MODEL_PATH)

print("Model loaded successfully.")

print("\nLoading development features...")

data = np.load(DEV_FEATURES_PATH)

X_dev = data["X"]
y_dev = data["y"]

print("Development feature shape:", X_dev.shape)
print("Development label shape:", y_dev.shape)

# Verify class labels
print("\nClass distribution:")

unique, counts = np.unique(y_dev, return_counts=True)

for label, count in zip(unique, counts):
    class_name = "bonafide" if label == 0 else "spoof"
    print(f"{class_name}: {count}")


# --------------------------------------------------
# 3. GENERATE PREDICTIONS AND SCORES
# --------------------------------------------------

print("\nGenerating predictions...")

# Predicted labels using the model's default threshold
y_pred = model.predict(X_dev)

# Continuous decision scores
# Higher scores indicate stronger spoof predictions
y_scores = model.decision_function(X_dev)


# --------------------------------------------------
# 4. CLASSIFICATION METRICS
# --------------------------------------------------

accuracy = accuracy_score(y_dev, y_pred)

balanced_acc = balanced_accuracy_score(
    y_dev, y_pred
)

precision = precision_score(
    y_dev, y_pred, pos_label=1
)

recall = recall_score(
    y_dev, y_pred, pos_label=1
)

f1 = f1_score(
    y_dev, y_pred, pos_label=1
)

print("\n========== CLASSIFICATION METRICS ==========")

print(f"Accuracy          : {accuracy:.4f}")
print(f"Balanced Accuracy : {balanced_acc:.4f}")
print(f"Spoof Precision   : {precision:.4f}")
print(f"Spoof Recall      : {recall:.4f}")
print(f"Spoof F1-score    : {f1:.4f}")


# --------------------------------------------------
# 5. CONFUSION MATRIX
# --------------------------------------------------

cm = confusion_matrix(
    y_dev,
    y_pred,
    labels=[0, 1],
)

tn, fp, fn, tp = cm.ravel()

# With spoof = positive class:
# BPCER = genuine speech classified as spoof
# APCER = spoof speech classified as genuine

bpcer = fp / (tn + fp)

apcer = fn / (fn + tp)

acer = (apcer + bpcer) / 2

print("\n========== ANTI-SPOOFING ERROR RATES ==========")

print(f"BPCER : {bpcer:.4f} ({bpcer * 100:.2f}%)")
print(f"APCER : {apcer:.4f} ({apcer * 100:.2f}%)")
print(f"ACER  : {acer:.4f} ({acer * 100:.2f}%)")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")

print(
    classification_report(
        y_dev,
        y_pred,
        labels=[0, 1],
        target_names=["bonafide", "spoof"],
        zero_division=0,
    )
)


# --------------------------------------------------
# 6. ROC CURVE AND ROC-AUC
# --------------------------------------------------

fpr, tpr, thresholds = roc_curve(
    y_dev,
    y_scores,
    pos_label=1,
)

roc_auc = roc_auc_score(
    y_dev,
    y_scores,
)

print("\n========== ROC-AUC ==========")

print(f"ROC-AUC: {roc_auc:.4f}")


# --------------------------------------------------
# 7. EQUAL ERROR RATE (EER)
# --------------------------------------------------

fnr = 1 - tpr

# Find the point where FPR and FNR are equal
eer = brentq(
    lambda x: 1.0 - x - np.interp(x, fpr, tpr),
    0.0,
    1.0,
)

# Approximate threshold corresponding to the EER point
eer_index = np.argmin(np.abs(fpr - fnr))

eer_threshold = thresholds[eer_index]

print("\n========== EQUAL ERROR RATE ==========")

print(f"EER           : {eer:.4f}")
print(f"EER percentage: {eer * 100:.2f}%")
print(f"EER threshold : {eer_threshold:.6f}")


# --------------------------------------------------
# 8. PLOT ROC CURVE
# --------------------------------------------------

plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    label=f"RBF-SVM (AUC = {roc_auc:.4f})",
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random classifier",
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve - LFCC + RBF-SVM")

plt.legend()
plt.grid(True)

roc_path = ARTIFACT_DIR / "roc_curve.png"

plt.savefig(
    roc_path,
    dpi=300,
    bbox_inches="tight",
)

print(f"\nROC curve saved to: {roc_path}")


# --------------------------------------------------
# 9. SAVE EVALUATION RESULTS
# --------------------------------------------------

results_path = ARTIFACT_DIR / "advanced_evaluation.txt"

with open(results_path, "w") as file:

    file.write("LFCC + RBF-SVM MODEL EVALUATION\n")
    file.write("=" * 45 + "\n\n")

    file.write(f"Accuracy: {accuracy:.4f}\n")
    file.write(f"Balanced Accuracy: {balanced_acc:.4f}\n")
    file.write(f"Spoof Precision: {precision:.4f}\n")
    file.write(f"Spoof Recall: {recall:.4f}\n")
    file.write(f"Spoof F1-score: {f1:.4f}\n")

    file.write(f"\nROC-AUC: {roc_auc:.4f}\n")
    file.write(f"EER: {eer:.4f}\n")
    file.write(f"EER Percentage: {eer * 100:.2f}%\n")

    file.write(f"\nAPCER: {apcer:.4f}\n")
    file.write(f"BPCER: {bpcer:.4f}\n")
    file.write(f"ACER: {acer:.4f}\n")

    file.write("\nConfusion Matrix:\n")
    file.write(str(cm))

    file.write("\n\nClassification Report:\n")
    file.write(
        classification_report(
            y_dev,
            y_pred,
            labels=[0, 1],
            target_names=["bonafide", "spoof"],
            zero_division=0,
        )
    )

print("\nEvaluation results saved to:")
print(results_path)

print("\nAdvanced evaluation completed successfully!")

# Show the plot last so closing the window cannot interrupt saving
plt.show()
