
from pathlib import Path
import json
import os
import subprocess
import tempfile

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

# Read directly by soundfile
NATIVE_EXTENSIONS = {".wav", ".flac"}

# Converted to WAV with ffmpeg first
FFMPEG_EXTENSIONS = {".mp3", ".m4a"}

# Decodable, but the model gives misleading results on them.
# On 60 dev files, MP3 encoding turned all 30 bona fide files into
# "spoof" (it removes most 7-8 kHz energy), while AAC/M4A changed
# no labels. Remove MP3 from here only after retraining on MP3 audio.
UNRELIABLE_EXTENSIONS = {
    ".mp3": (
        "MP3 files are not supported. Testing showed that MP3 compression "
        "makes this model label genuine speech as spoof. Please upload "
        "WAV, FLAC or M4A audio instead."
    ),
}

SUPPORTED_EXTENSIONS = (
    NATIVE_EXTENSIONS | FFMPEG_EXTENSIONS
) - set(UNRELIABLE_EXTENSIONS)

FFMPEG_TIMEOUT_SECONDS = 30


# --------------------------------------------------
# 2. ERRORS
# --------------------------------------------------

class AudioError(ValueError):
    """The uploaded file could not be used as audio."""


# --------------------------------------------------
# 3. FFMPEG CONVERSION
# --------------------------------------------------

def ffmpeg_executable():
    """Path to the ffmpeg binary bundled with the imageio-ffmpeg package."""

    try:
        import imageio_ffmpeg
    except ImportError as error:
        raise AudioError(
            "MP3 and M4A support needs the imageio-ffmpeg package."
        ) from error

    return imageio_ffmpeg.get_ffmpeg_exe()


def convert_to_wav(source, destination):
    """
    Decode the first audio stream to 32-bit float WAV.

    Sample rate and channels are kept, so features.extract_lfcc does
    the same mono conversion and resampling as for WAV and FLAC files.
    """

    command = [
        ffmpeg_executable(),
        "-nostdin",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-i", str(source),
        "-map", "0:a:0",
        "-vn",
        "-acodec", "pcm_f32le",
        "-f", "wav",
        str(destination),
    ]

    try:
        subprocess.run(
            command,
            capture_output=True,
            timeout=FFMPEG_TIMEOUT_SECONDS,
            check=True,
        )
    except subprocess.TimeoutExpired as error:
        raise AudioError("The audio file took too long to decode.") from error
    except subprocess.CalledProcessError as error:
        raise AudioError("The file could not be read as audio.") from error


# --------------------------------------------------
# 4. DETECTOR
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

        extension = Path(path).suffix.lower()

        if extension in UNRELIABLE_EXTENSIONS:
            raise AudioError(UNRELIABLE_EXTENSIONS[extension])

        if extension not in SUPPORTED_EXTENSIONS:
            raise AudioError("Unsupported audio format.")

        if extension in NATIVE_EXTENSIONS:
            return self._predict_readable(path, extension.lstrip(".").upper())

        handle, wav_path = tempfile.mkstemp(
            suffix=".wav",
            dir=Path(path).parent,
        )
        os.close(handle)

        try:
            convert_to_wav(path, wav_path)

            result = self._predict_readable(
                wav_path,
                extension.lstrip(".").upper(),
            )

        finally:
            os.remove(wav_path)

        result["warnings"].append(
            f"{result['format']} is a compressed format. The model was "
            "trained on lossless FLAC audio, so this result is less reliable."
        )

        return result

    def _predict_readable(self, path, file_format):

        try:
            info = sf.info(str(path))
        except Exception as error:
            raise AudioError("The file could not be read as audio.") from error

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
            "format": file_format,
            "warnings": warnings,
            "model": MODEL_NAME,
        }
