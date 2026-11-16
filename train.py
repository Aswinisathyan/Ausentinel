
from pathlib import Path
import time
import joblib
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from features import extract_lfcc_from_file


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data" / "ASVspoof2019_LA" / "LA"

TRAIN_AUDIO_DIR = DATA_DIR / "ASVspoof2019_LA_train" / "flac"
DEV_AUDIO_DIR = DATA_DIR / "ASVspoof2019_LA_dev" / "flac"

PROTOCOL_DIR = DATA_DIR / "ASVspoof2019_LA_cm_protocols"

TRAIN_PROTOCOL = (
    PROTOCOL_DIR / "ASVspoof2019.LA.cm.train.trn.txt"
)

DEV_PROTOCOL = (
    PROTOCOL_DIR / "ASVspoof2019.LA.cm.dev.trl.txt"
)

ARTIFACT_DIR = ROOT / "artifacts"
MODEL_DIR = ROOT / "models"

ARTIFACT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# 2. LABEL MAPPING
# --------------------------------------------------

LABEL_MAP = {
    "bonafide": 0,
    "spoof": 1,
}


# --------------------------------------------------
# 3. LOAD PROTOCOL
# --------------------------------------------------

def load_protocol(protocol_path, audio_dir):

    samples = []

    with open(protocol_path, "r") as file:

        for line_number, line in enumerate(file, start=1):

            parts = line.strip().split()

            if len(parts) != 5:
                raise ValueError(
                    f"Invalid protocol row at line {line_number}: {line}"
                )

            speaker_id, file_id, _, system_id, label = parts

            if label not in LABEL_MAP:
                raise ValueError(
                    f"Unknown label '{label}' at line {line_number}"
                )

            audio_path = audio_dir / f"{file_id}.flac"

            samples.append({
                "file_id": file_id,
                "path": audio_path,
                "label": LABEL_MAP[label],
            })

    return samples


# --------------------------------------------------
# 4. EXTRACT LFCC FEATURES
# --------------------------------------------------

def extract_dataset_features(samples, cache_path):

    if cache_path.exists():

        print(f"\nLoading cached features: {cache_path}")

        cached = np.load(cache_path)

        return cached["X"], cached["y"]

    features = []
    labels = []

    failed_files = []

    total = len(samples)

    print(f"\nExtracting LFCC features from {total} files...")

    start_time = time.time()

    for index, sample in enumerate(samples, start=1):

        audio_path = sample["path"]

        if not audio_path.exists():
            failed_files.append(str(audio_path))
            continue

        try:

            feature = extract_lfcc_from_file(audio_path)

            if feature.shape != (80,):
                raise ValueError(
                    f"Expected 80 features, received {feature.shape}"
                )

            if not np.all(np.isfinite(feature)):
                raise ValueError("Non-finite feature values")

            features.append(feature)
            labels.append(sample["label"])

        except Exception as error:

            failed_files.append(
                f"{audio_path} | {error}"
            )

        if index % 500 == 0 or index == total:

            elapsed = time.time() - start_time

            print(
                f"Processed {index}/{total} | "
                f"Successful: {len(features)} | "
                f"Failed: {len(failed_files)} | "
                f"Time: {elapsed:.1f}s"
            )

    if failed_files:

        error_path = ARTIFACT_DIR / (
            cache_path.stem + "_errors.txt"
        )

        with open(error_path, "w") as file:
            file.write("\n".join(failed_files))

        print(f"\nFailed file list saved to: {error_path}")

    if not features:
        raise RuntimeError("No LFCC features were extracted.")

    X = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)

    np.savez_compressed(
        cache_path,
        X=X,
        y=y,
    )

    print("\nFeature extraction completed.")
    print("Feature matrix shape:", X.shape)
    print("Label array shape:", y.shape)

    return X, y


# --------------------------------------------------
# 5. TRAIN MODEL
# --------------------------------------------------

def train_model(X_train, y_train):

    print("\nTraining RBF-SVM classifier...")

    model = Pipeline([
        ("scaler", StandardScaler()),

        ("svm", SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced",
            cache_size=2000,
        )),
    ])

    start_time = time.time()

    model.fit(X_train, y_train)

    print(
        f"Training completed in "
        f"{time.time() - start_time:.2f} seconds."
    )

    return model


# --------------------------------------------------
# 6. EVALUATION
# --------------------------------------------------

def evaluate_model(model, X_dev, y_dev):

    print("\nEvaluating model on development dataset...")

    predictions = model.predict(X_dev)

    accuracy = accuracy_score(y_dev, predictions)

    precision = precision_score(
        y_dev, predictions, zero_division=0
    )

    recall = recall_score(
        y_dev, predictions, zero_division=0
    )

    f1 = f1_score(
        y_dev, predictions, zero_division=0
    )

    cm = confusion_matrix(
        y_dev,
        predictions,
        labels=[0, 1],
    )

    print("\n========== MODEL RESULTS ==========")

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")

    print(
        classification_report(
            y_dev,
            predictions,
            labels=[0, 1],
            target_names=["bonafide", "spoof"],
            zero_division=0,
        )
    )

    results_path = ARTIFACT_DIR / "evaluation_results.txt"

    with open(results_path, "w") as file:

        file.write(f"Accuracy: {accuracy:.4f}\n")
        file.write(f"Precision: {precision:.4f}\n")
        file.write(f"Recall: {recall:.4f}\n")
        file.write(f"F1-score: {f1:.4f}\n")
        file.write(f"Confusion Matrix:\n{cm}\n")

        file.write(
            classification_report(
                y_dev,
                predictions,
                labels=[0, 1],
                target_names=["bonafide", "spoof"],
                zero_division=0,
            )
        )

    print(f"\nResults saved to: {results_path}")


# --------------------------------------------------
# 7. MAIN PIPELINE
# --------------------------------------------------

def main():

    print("====================================")
    print(" AUDIO SPOOF DETECTION")
    print(" LFCC + RBF-SVM")
    print("====================================")

    # Verify directories and protocol files

    required_paths = [
        TRAIN_AUDIO_DIR,
        DEV_AUDIO_DIR,
        TRAIN_PROTOCOL,
        DEV_PROTOCOL,
    ]

    for path in required_paths:

        if not path.exists():
            raise FileNotFoundError(
                f"Required path not found: {path}"
            )

    # Load protocols

    train_samples = load_protocol(
        TRAIN_PROTOCOL,
        TRAIN_AUDIO_DIR,
    )

    dev_samples = load_protocol(
        DEV_PROTOCOL,
        DEV_AUDIO_DIR,
    )

    print("\nTraining samples:", len(train_samples))
    print("Development samples:", len(dev_samples))

    # Extract training features

    X_train, y_train = extract_dataset_features(
        train_samples,
        ARTIFACT_DIR / "train_features.npz",
    )

    # Extract development features

    X_dev, y_dev = extract_dataset_features(
        dev_samples,
        ARTIFACT_DIR / "dev_features.npz",
    )

    print("\nTraining feature shape:", X_train.shape)
    print("Development feature shape:", X_dev.shape)

    # Train classifier

    model = train_model(X_train, y_train)

    # Save trained model

    model_path = MODEL_DIR / "lfcc_rbf_svm.joblib"

    joblib.dump(model, model_path)

    print(f"\nModel saved to: {model_path}")

    # Evaluate

    evaluate_model(
        model,
        X_dev,
        y_dev,
    )

    print("\n====================================")
    print(" PIPELINE COMPLETED")
    print("====================================")


if __name__ == "__main__":
    main()
