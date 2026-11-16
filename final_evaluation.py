
import csv
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import brentq
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from features import extract_lfcc_from_file


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

DATA_ROOT = ROOT / "data" / "ASVspoof2019_LA" / "LA"

AUDIO_DIR = (
    DATA_ROOT
    / "ASVspoof2019_LA_eval"
    / "flac"
)

PROTOCOL_FILE = (
    DATA_ROOT
    / "ASVspoof2019_LA_cm_protocols"
    / "ASVspoof2019.LA.cm.eval.trl.txt"
)

MODEL_FILE = ROOT / "models" / "lfcc_rbf_svm.joblib"

OUTPUT_DIR = ROOT / "artifacts" / "final_evaluation"

FEATURE_FILE = OUTPUT_DIR / "evaluation_features.npy"
STATUS_FILE = OUTPUT_DIR / "feature_status.npy"
PROTOCOL_HASH_FILE = OUTPUT_DIR / "protocol_hash.txt"

THRESHOLD = 0.455310

EXPECTED_FEATURES = 80
EXPECTED_PROTOCOL_FILES = 71237

CHECKPOINT_INTERVAL = 500

LABEL_MAP = {
    "bonafide": 0,
    "spoof": 1,
}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD AND VALIDATE PROTOCOL
# ============================================================

print("\n" + "=" * 65)
print("VOICEGUARD AI - FINAL EVALUATION")
print("=" * 65)

print("\n[1/6] Loading evaluation protocol...")

if not PROTOCOL_FILE.exists():
    raise FileNotFoundError(f"Protocol not found: {PROTOCOL_FILE}")

protocol = pd.read_csv(
    PROTOCOL_FILE,
    sep=r"\s+",
    header=None,
    names=["speaker", "file", "unused", "attack_id", "key"],
    dtype=str,
)

if len(protocol.columns) != 5:
    raise ValueError("Protocol must contain exactly five columns.")

if len(protocol) != EXPECTED_PROTOCOL_FILES:
    raise ValueError(
        f"Expected {EXPECTED_PROTOCOL_FILES} protocol rows, "
        f"but found {len(protocol)}."
    )

if not (protocol["unused"] == "-").all():
    raise ValueError("Unexpected values found in protocol column 3.")

if not protocol["key"].isin(LABEL_MAP).all():
    invalid = protocol.loc[
        ~protocol["key"].isin(LABEL_MAP), "key"
    ].unique()

    raise ValueError(f"Unexpected protocol labels: {invalid}")

if protocol["file"].duplicated().any():
    raise ValueError("Duplicate file IDs found in protocol.")

protocol["true_label"] = protocol["key"].map(LABEL_MAP)

protocol["audio_path"] = protocol["file"].apply(
    lambda name: AUDIO_DIR / f"{name}.flac"
)

missing_files = protocol.loc[
    ~protocol["audio_path"].apply(Path.exists)
]

if not missing_files.empty:
    raise FileNotFoundError(
        f"{len(missing_files)} protocol audio files are missing. "
        "Evaluation stopped."
    )

protocol_hash = hashlib.sha256(
    PROTOCOL_FILE.read_bytes()
).hexdigest()

print(f"Protocol rows: {len(protocol):,}")
print(f"Bonafide samples: {(protocol['true_label'] == 0).sum():,}")
print(f"Spoof samples: {(protocol['true_label'] == 1).sum():,}")
print("All protocol audio files exist.")

audio_count = len(list(AUDIO_DIR.glob("*.flac")))
extra_audio_count = audio_count - len(protocol)

print(f"Audio files in directory: {audio_count:,}")
print(f"Unlisted extra audio files: {extra_audio_count:,}")
print("Only protocol-listed files will be evaluated.")


# ============================================================
# 3. LOAD MODEL
# ============================================================

print("\n[2/6] Loading trained model...")

if not MODEL_FILE.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_FILE}")

model = joblib.load(MODEL_FILE)

if not hasattr(model, "decision_function"):
    raise TypeError("Loaded model has no decision_function method.")

print("Model loaded successfully.")
print(f"Model type: {type(model).__name__}")
print(f"Frozen threshold: {THRESHOLD}")


# ============================================================
# 4. FEATURE EXTRACTION WITH CHECKPOINTING
# ============================================================

print("\n[3/6] Extracting LFCC features...")

n_samples = len(protocol)

if FEATURE_FILE.exists() and STATUS_FILE.exists():
    X_map = np.load(FEATURE_FILE, mmap_mode="r+")
    status = np.load(STATUS_FILE, mmap_mode="r+")

    if X_map.shape != (n_samples, EXPECTED_FEATURES):
        raise ValueError("Existing feature cache has an invalid shape.")

    if status.shape != (n_samples,):
        raise ValueError("Existing feature status cache has an invalid shape.")

    if not PROTOCOL_HASH_FILE.exists():
        raise ValueError("Protocol hash missing. Cannot safely resume.")

    saved_hash = PROTOCOL_HASH_FILE.read_text().strip()

    if saved_hash != protocol_hash:
        raise ValueError(
            "Protocol has changed since the cache was created. "
            "Move the old cache files before starting a new evaluation."
        )

    print("Existing feature cache found. Resuming where possible.")

else:
    if FEATURE_FILE.exists() or STATUS_FILE.exists():
        raise ValueError(
            "Incomplete cache files detected. Inspect the output folder "
            "before starting a fresh extraction."
        )

    X_map = np.lib.format.open_memmap(
        FEATURE_FILE,
        mode="w+",
        dtype=np.float32,
        shape=(n_samples, EXPECTED_FEATURES),
    )

    status = np.lib.format.open_memmap(
        STATUS_FILE,
        mode="w+",
        dtype=np.uint8,
        shape=(n_samples,),
    )

    status[:] = 0
    X_map.flush()
    status.flush()

    PROTOCOL_HASH_FILE.write_text(protocol_hash)

# Status values:
# 0 = not processed
# 1 = successful
# 2 = failed

start_time = time.time()
success_count = int(np.sum(status == 1))
failure_count = int(np.sum(status == 2))

for index, row in protocol.iterrows():

    if status[index] == 1:
        continue

    audio_path = row["audio_path"]

    try:
        feature = extract_lfcc_from_file(str(audio_path))

        feature = np.asarray(feature, dtype=np.float32).reshape(-1)

        if feature.shape != (EXPECTED_FEATURES,):
            raise ValueError(
                f"Expected {EXPECTED_FEATURES} features, "
                f"received {feature.shape}"
            )

        if not np.isfinite(feature).all():
            raise ValueError("Feature vector contains NaN or infinity.")

        X_map[index] = feature
        status[index] = 1
        success_count += 1

    except Exception as exc:
        status[index] = 2
        failure_count += 1

        failure_file = OUTPUT_DIR / "feature_failures.csv"

        write_header = not failure_file.exists()

        with open(
            failure_file,
            "a",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.writer(file)

            if write_header:
                writer.writerow(["file", "error"])

            writer.writerow([row["file"], repr(exc)])

    processed = index + 1

    if processed % CHECKPOINT_INTERVAL == 0:
        X_map.flush()
        status.flush()

        elapsed = time.time() - start_time

        print(
            f"Progress: {processed:,}/{n_samples:,} | "
            f"Successful: {success_count:,} | "
            f"Failed: {failure_count:,} | "
            f"Elapsed: {elapsed / 60:.1f} min"
        )

X_map.flush()
status.flush()

print("\nFeature extraction completed.")
print(f"Successful features: {success_count:,}")
print(f"Failed features: {failure_count:,}")

if failure_count > 0:
    raise RuntimeError(
        "Feature extraction failures detected. "
        "Review feature_failures.csv before continuing."
    )

if success_count != n_samples:
    raise RuntimeError("Feature extraction is incomplete.")

# Load features into memory for model inference.
X = np.asarray(X_map, dtype=np.float32)

del X_map
del status


# ============================================================
# 5. MODEL INFERENCE AND PERFORMANCE METRICS
# ============================================================

print("\n[4/6] Running model predictions...")

y_true = protocol["true_label"].to_numpy(dtype=np.int32)

# Higher decision score means more likely to be spoof.
scores = model.decision_function(X)

# Apply the frozen development threshold.
y_pred = (scores >= THRESHOLD).astype(np.int32)

accuracy = accuracy_score(y_true, y_pred)
balanced_accuracy = balanced_accuracy_score(y_true, y_pred)

precision = precision_score(
    y_true, y_pred, zero_division=0
)

recall = recall_score(
    y_true, y_pred, zero_division=0
)

f1 = f1_score(
    y_true, y_pred, zero_division=0
)

tn, fp, fn, tp = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1],
).ravel()

apcer = fn / (fn + tp) if (fn + tp) else 0.0
bpcer = fp / (fp + tn) if (fp + tn) else 0.0

acer = (apcer + bpcer) / 2

auc = roc_auc_score(y_true, scores)

fpr, tpr, roc_thresholds = roc_curve(y_true, scores)

eer = brentq(
    lambda x: 1.0 - x - np.interp(x, fpr, tpr),
    0.0,
    1.0,
)

# This EER is descriptive only. It is not used to change
# the frozen threshold used for final predictions.

print("Inference completed.")


# ============================================================
# 6. SAVE RESULTS, PLOTS AND ATTACK METRICS
# ============================================================

print("\n[5/6] Saving evaluation results...")

predictions = protocol[
    ["speaker", "file", "attack_id", "key"]
].copy()

predictions["true_label"] = y_true
predictions["score"] = scores
predictions["predicted_label"] = y_pred
predictions["predicted_class"] = np.where(
    y_pred == 1, "spoof", "bonafide"
)
predictions["correct"] = y_true == y_pred

predictions.to_csv(
    OUTPUT_DIR / "predictions.csv",
    index=False,
)

# Attack-specific APCER
attack_rows = []

spoof_protocol = protocol.loc[
    protocol["true_label"] == 1
].copy()

for attack_id, group in spoof_protocol.groupby("attack_id"):

    indices = group.index.to_numpy()

    attack_true = y_true[indices]
    attack_pred = y_pred[indices]

    attack_fn = int(
        np.sum((attack_true == 1) & (attack_pred == 0))
    )

    attack_total = len(indices)

    attack_apcer = (
        attack_fn / attack_total
        if attack_total else 0.0
    )

    attack_rows.append({
        "attack_id": attack_id,
        "spoof_samples": attack_total,
        "false_accepts": attack_fn,
        "APCER": attack_apcer,
        "APCER_percent": attack_apcer * 100,
    })

attack_df = pd.DataFrame(attack_rows)

attack_df.to_csv(
    OUTPUT_DIR / "attack_metrics.csv",
    index=False,
)

# Confusion matrix plot
cm = np.array([
    [tn, fp],
    [fn, tp],
])

plt.figure(figsize=(6, 5))
plt.imshow(cm)
plt.title("VoiceGuard AI - Confusion Matrix")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.xticks([0, 1], ["Bonafide", "Spoof"])
plt.yticks([0, 1], ["Bonafide", "Spoof"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "confusion_matrix.png",
    dpi=300,
)
plt.close()

# ROC curve
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("VoiceGuard AI - ROC Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "roc_curve.png",
    dpi=300,
)
plt.close()

# Score distribution
plt.figure(figsize=(8, 5))

plt.hist(
    scores[y_true == 0],
    bins=60,
    alpha=0.6,
    label="Bonafide",
    density=True,
)

plt.hist(
    scores[y_true == 1],
    bins=60,
    alpha=0.6,
    label="Spoof",
    density=True,
)

plt.axvline(
    THRESHOLD,
    linestyle="--",
    label=f"Frozen threshold = {THRESHOLD}",
)

plt.xlabel("SVM decision score")
plt.ylabel("Density")
plt.title("Decision Score Distribution")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "score_distribution.png",
    dpi=300,
)
plt.close()


# ============================================================
# FINAL REPORT
# ============================================================

report = f"""
VOICEGUARD AI
ASVspoof 2019 LA - FINAL EVALUATION REPORT
============================================================

MODEL
------------------------------------------------------------
Classifier: LFCC + RBF-SVM
Feature dimension: {EXPECTED_FEATURES}
Threshold: {THRESHOLD:.6f}
Threshold source: Frozen development-set selection

DATASET
------------------------------------------------------------
Protocol samples: {len(protocol)}
Bonafide samples: {int(np.sum(y_true == 0))}
Spoof samples: {int(np.sum(y_true == 1))}
Audio files in directory: {audio_count}
Unlisted audio files excluded: {extra_audio_count}

PERFORMANCE
------------------------------------------------------------
Accuracy:           {accuracy:.6f} ({accuracy * 100:.2f}%)
Balanced Accuracy:  {balanced_accuracy:.6f} ({balanced_accuracy * 100:.2f}%)
Precision:          {precision:.6f}
Recall:             {recall:.6f}
F1-score:           {f1:.6f}
ROC-AUC:            {auc:.6f}
Interpolated EER:   {eer:.6f} ({eer * 100:.2f}%)

CONFUSION MATRIX
------------------------------------------------------------
True Negative (TN): {tn}
False Positive (FP): {fp}
False Negative (FN): {fn}
True Positive (TP): {tp}

ASVspoof ERROR METRICS
------------------------------------------------------------
APCER: {apcer:.6f} ({apcer * 100:.2f}%)
BPCER: {bpcer:.6f} ({bpcer * 100:.2f}%)
ACER:  {acer:.6f} ({acer * 100:.2f}%)

INTERPRETATION
------------------------------------------------------------
APCER: Proportion of spoof samples classified as bonafide.
BPCER: Proportion of bonafide samples classified as spoof.
ACER: Average of APCER and BPCER.

EER is calculated from the ROC curve and is descriptive.
The reported predictions use the frozen development threshold.
The evaluation labels were not used to select that threshold.

OUTPUT FILES
------------------------------------------------------------
predictions.csv
attack_metrics.csv
confusion_matrix.png
roc_curve.png
score_distribution.png
evaluation_features.npy
feature_status.npy
protocol_hash.txt

============================================================
"""

(OUTPUT_DIR / "evaluation_results.txt").write_text(
    report,
    encoding="utf-8",
)

print("\n[6/6] FINAL EVALUATION COMPLETED")

print(report)

print(f"\nAll outputs saved in:\n{OUTPUT_DIR}")
