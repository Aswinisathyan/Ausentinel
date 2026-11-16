
import json
from pathlib import Path

import joblib
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "models" / "lfcc_rbf_svm.joblib"

THRESHOLD_PATH = (
    ROOT / "artifacts" / "threshold_study" / "selected_threshold.json"
)

# Generated locally and not stored in git
DEV_FEATURES_PATH = ROOT / "artifacts" / "dev_features.npz"
DEV_SCORES_PATH = ROOT / "artifacts" / "threshold_study" / "dev_scores.npz"


@pytest.fixture(scope="module")
def model():

    return joblib.load(MODEL_PATH)


# --------------------------------------------------
# 1. SAVED MODEL
# --------------------------------------------------

def test_model_expects_80_features(model):

    assert model.n_features_in_ == 80


def test_model_classes_are_bonafide_then_spoof(model):

    # decision_function is positive for class 1 (spoof)
    np.testing.assert_array_equal(model.classes_, [0, 1])


def test_model_scores_one_sample(model):

    score = model.decision_function(np.zeros((1, 80), dtype=np.float32))

    assert score.shape == (1,)
    assert np.isfinite(score).all()


# --------------------------------------------------
# 2. FROZEN THRESHOLD
# --------------------------------------------------

def test_threshold_file_matches_study_selection():

    selection = json.loads(THRESHOLD_PATH.read_text())

    assert selection["selected_candidate"] == "EER"
    assert selection["threshold"] == pytest.approx(0.45531, abs=1e-5)
    assert "development set only" in selection["selection_data"]


# --------------------------------------------------
# 3. REGRESSION AGAINST SAVED DEV SCORES
# --------------------------------------------------

@pytest.mark.skipif(
    not (DEV_FEATURES_PATH.exists() and DEV_SCORES_PATH.exists()),
    reason="Cached dev features or scores not present on this machine",
)
def test_model_reproduces_saved_dev_scores(model):

    features = np.load(DEV_FEATURES_PATH)["X"]
    saved = np.load(DEV_SCORES_PATH)["score"]

    sample = np.linspace(0, len(features) - 1, 200).astype(int)

    np.testing.assert_allclose(
        model.decision_function(features[sample]),
        saved[sample],
        rtol=1e-9,
        atol=1e-9,
    )
