
from pathlib import Path
import json

import joblib
import librosa
import numpy as np
import soundfile as sf

from features import MAX_DURATION, extract_lfcc


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "models" / "lfcc_rbf_svm.joblib"

THRESHOLD_PATH = (
    ROOT / "artifacts" / "threshold_study" / "selected_threshold.json"
)

MODEL_NAME = "LFCC + RBF-SVM"

SHORT_AUDIO_SECONDS = 1.0
SILENCE_PEAK = 1e-4


# --------------------------------------------------
# 2. ERRORS
# --------------------------------------------------

class AudioError(ValueError):
    """The uploaded file could not be used as audio."""


# --------------------------------------------------
# 3. DETECTOR
# --------------------------------------------------

class SpoofDetector:
    """
    Load the trained model and frozen threshold once, then score files.

    Decision rule, as in threshold_study.py:
        spoof when decision_function(x) >= threshold
    """

    def __init__(self, model_path=MODEL_PATH, threshold_path=THRESHOLD_PATH):

        self.model = joblib.load(model_path)

        if self.model.n_features_in_ != 80:
            raise RuntimeError(
                f"Model expects {self.model.n_features_in_} features, not 80"
            )

        selection = json.loads(Path(threshold_path).read_text())

        self.threshold = float(selection["threshold"])
        self.threshold_source = selection["selection_data"]

    def predict_file(self, path):

        try:
            info = sf.info(str(path))
        except Exception as error:
            raise AudioError(
                "The file could not be read as WAV or FLAC audio."
            ) from error

        if info.frames == 0:
            raise AudioError("The audio file contains no samples.")

        # Same loading call as features.extract_lfcc_from_file
        audio, sample_rate = librosa.load(
            str(path),
            sr=None,
            mono=True,
        )

        if audio.size == 0:
            raise AudioError("The audio file contains no samples.")

        duration = len(audio) / sample_rate

        features = extract_lfcc(audio, sample_rate)

        score = float(
            self.model.decision_function(features.reshape(1, -1))[0]
        )

        is_spoof = score >= self.threshold

        warnings = []

        if duration > MAX_DURATION:
            warnings.append(
                f"Only the first {MAX_DURATION:g} seconds were analysed."
            )

        if duration < SHORT_AUDIO_SECONDS:
            warnings.append(
                "The clip is shorter than 1 second, so the result is less reliable."
            )

        if np.max(np.abs(audio)) < SILENCE_PEAK:
            warnings.append("The clip appears to be silent.")

        return {
            "label": "spoof" if is_spoof else "bonafide",
            "is_spoof": bool(is_spoof),
            "score": score,
            "threshold": self.threshold,
            "margin": score - self.threshold,
            "duration_seconds": round(duration, 3),
            "analysed_seconds": round(min(duration, MAX_DURATION), 3),
            "sample_rate": int(sample_rate),
            "channels": int(info.channels),
            "format": info.format,
            "warnings": warnings,
            "model": MODEL_NAME,
        }
