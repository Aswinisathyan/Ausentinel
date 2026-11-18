
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from sklearn.metrics import roc_auc_score, roc_curve


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent

STUDY_DIR = ROOT / "artifacts" / "threshold_study"
FINAL_DIR = ROOT / "artifacts" / "final_evaluation"

# Only these files can be served by the /charts route
CHARTS = {
    "dev-acer": {
        "path": STUDY_DIR / "acer.png",
        "title": "Error rate (ACER) at every threshold, development set",
        "alt": "Line chart of ACER against threshold with a shaded 95% interval. "
               "The curve is flat between about 0.3 and 0.6.",
    },
    "dev-det": {
        "path": STUDY_DIR / "det.png",
        "title": "DET curve, development set",
        "alt": "DET curve of APCER against BPCER with markers for the candidate thresholds.",
    },
    "eval-scores": {
        "path": FINAL_DIR / "score_distribution.png",
        "title": "Score distribution, evaluation set",
        "alt": "Histograms of bona fide and spoof scores on the evaluation set, "
               "with many spoof scores below the threshold.",
    },
    "eval-confusion": {
        "path": FINAL_DIR / "confusion_matrix.png",
        "title": "Confusion matrix, evaluation set",
        "alt": "Confusion matrix: 7,070 bona fide correct, 285 flagged, "
               "21,955 spoofs missed, 41,927 spoofs detected.",
    },
}

# From the ASVspoof 2019 LA README
ATTACK_TYPES = {
    "A01": "TTS, neural waveform",
    "A02": "TTS, vocoder",
    "A03": "TTS, vocoder",
    "A04": "TTS, waveform concatenation",
    "A05": "VC, vocoder",
    "A06": "VC, spectral filtering",
    "A07": "TTS, vocoder + GAN",
    "A08": "TTS, neural waveform",
    "A09": "TTS, vocoder",
    "A10": "TTS, neural waveform",
    "A11": "TTS, Griffin-Lim",
    "A12": "TTS, neural waveform",
    "A13": "TTS + VC, waveform concatenation + filtering",
    "A14": "TTS + VC, vocoder",
    "A15": "TTS + VC, neural waveform",
    "A16": "TTS, waveform concatenation",
    "A17": "VC, waveform filtering",
    "A18": "VC, vocoder",
    "A19": "VC, spectral filtering",
}


# --------------------------------------------------
# 2. LOADERS
# --------------------------------------------------

def load_dev():
    """Development-set results at the selected (EER) threshold."""

    comparison = pd.read_csv(STUDY_DIR / "threshold_comparison.csv")
    intervals = pd.read_csv(STUDY_DIR / "bootstrap_metric_ci.csv")
    attacks = pd.read_csv(STUDY_DIR / "apcer_by_attack.csv")

    row = comparison[comparison["candidate"] == "EER"].iloc[0]

    def free(metric):
        return intervals[
            (intervals["operating_point"] == "threshold-free")
            & (intervals["metric"] == metric)
        ].iloc[0]

    acer = intervals[
        (intervals["operating_point"] == "EER")
        & (intervals["metric"] == "ACER")
    ].iloc[0]

    selected = attacks[attacks["candidate"] == "EER"]

    # Prefer the interpolated EER from evaluate.py, the same method used
    # for the eval set. The study's value picks the closest real threshold
    # instead, so it can differ in the second decimal place.
    eer = float(free("EER")["value"])

    report = ROOT / "artifacts" / "advanced_evaluation.txt"

    if report.exists():
        for line in report.read_text().splitlines():
            if line.startswith("EER:"):
                eer = float(line.split(":", 1)[1])

    return {
        "threshold": float(row["threshold"]),
        "files": None,
        "accuracy": float(row["accuracy"]),
        "apcer": float(row["APCER"]),
        "bpcer": float(row["BPCER"]),
        "acer": float(row["ACER"]),
        "acer_low": float(acer["speaker_ci_low"]),
        "acer_high": float(acer["speaker_ci_high"]),
        "eer": eer,
        "auc": float(free("AUC")["value"]),
        "attacks": [
            {
                "attack": r.attack,
                "type": ATTACK_TYPES.get(r.attack, ""),
                "files": int(r.n_files),
                "missed": int(r.errors),
                "apcer": float(r.APCER),
            }
            for r in selected.itertuples()
        ],
    }


def load_eval():
    """Evaluation-set results recomputed from the saved predictions."""

    predictions = pd.read_csv(FINAL_DIR / "predictions.csv")
    attacks = pd.read_csv(FINAL_DIR / "attack_metrics.csv")

    y = predictions["true_label"].to_numpy()
    scores = predictions["score"].to_numpy()
    predicted = predictions["predicted_label"].to_numpy()

    apcer = float(np.mean(predicted[y == 1] == 0))
    bpcer = float(np.mean(predicted[y == 0] == 1))

    fpr, tpr, _ = roc_curve(y, scores)

    eer = brentq(lambda x: 1.0 - x - np.interp(x, fpr, tpr), 0.0, 1.0)

    return {
        "files": len(y),
        "accuracy": float(np.mean(predicted == y)),
        "apcer": apcer,
        "bpcer": bpcer,
        "acer": (apcer + bpcer) / 2,
        "eer": float(eer),
        "auc": float(roc_auc_score(y, scores)),
        "attacks": [
            {
                "attack": r.attack_id,
                "type": ATTACK_TYPES.get(r.attack_id, ""),
                "files": int(r.spoof_samples),
                "missed": int(r.false_accepts),
                "apcer": float(r.APCER),
            }
            for r in attacks.itertuples()
        ],
    }


def load_results():
    """
    Load everything the results page shows.

    A section is None when its files are missing, so the page can
    say so instead of failing.
    """

    results = {}

    for name, loader in (("dev", load_dev), ("eval", load_eval)):
        try:
            results[name] = loader()
        except (FileNotFoundError, KeyError, IndexError, ValueError):
            results[name] = None

    results["charts"] = {
        name: chart
        for name, chart in CHARTS.items()
        if chart["path"].exists()
    }

    return results
