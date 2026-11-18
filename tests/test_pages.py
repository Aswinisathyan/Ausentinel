
import pytest

from app import (
    BORDERLINE_MARGIN,
    SCALE_MAX,
    SCALE_MIN,
    create_app,
    present,
    scale_position,
)
from inference import SpoofDetector
from results_data import load_results


@pytest.fixture(scope="module")
def detector():

    return SpoofDetector()


@pytest.fixture(scope="module")
def results():

    return load_results()


@pytest.fixture
def client(detector, results, tmp_path):

    app = create_app(detector=detector, upload_dir=str(tmp_path), results=results)
    app.config["TESTING"] = True

    return app.test_client()


def fake_result(score, threshold=0.45531):

    return {
        "score": score,
        "threshold": threshold,
        "margin": score - threshold,
        "is_spoof": score >= threshold,
        "borderline": abs(score - threshold) < BORDERLINE_MARGIN,
    }


# --------------------------------------------------
# 1. PAGES AND NAVIGATION
# --------------------------------------------------

@pytest.mark.parametrize("path, active", [
    ("/", "Analyse"),
    ("/results", "Results"),
    ("/about", "About"),
])
def test_pages_load_with_current_nav_item(client, path, active):

    response = client.get(path)
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert f'aria-current="page">{active}</a>' in html
    assert 'name="viewport"' in html


def test_stylesheet_and_script_are_served(client):

    assert client.get("/static/style.css").status_code == 200
    assert client.get("/static/app.js").status_code == 200


def test_about_page_explains_threshold_and_mp3(client, detector):

    html = client.get("/about").get_data(as_text=True)

    assert f"{detector.threshold:.4f}" in html
    assert "MP3 is not accepted" in html


# --------------------------------------------------
# 2. RESULTS PAGE
# --------------------------------------------------

def test_results_page_shows_dev_and_eval_figures(client, results):

    if results["dev"] is None or results["eval"] is None:
        pytest.skip("Saved results not present on this machine")

    html = client.get("/results").get_data(as_text=True)

    assert "7.20%" in html      # dev APCER at the frozen threshold
    assert "34.37%" in html     # eval APCER
    assert "A17" in html


def test_results_loader_matches_final_evaluation_report(results):

    if results["eval"] is None:
        pytest.skip("Saved eval predictions not present on this machine")

    ev = results["eval"]

    assert ev["files"] == 71237
    assert ev["apcer"] == pytest.approx(0.343681, abs=1e-6)
    assert ev["bpcer"] == pytest.approx(0.038749, abs=1e-6)
    assert ev["eer"] == pytest.approx(0.151869, abs=1e-5)
    assert ev["auc"] == pytest.approx(0.922848, abs=1e-5)


def test_results_page_without_saved_results(detector, tmp_path):

    empty = {"dev": None, "eval": None, "charts": {}}

    app = create_app(detector=detector, upload_dir=str(tmp_path), results=empty)

    response = app.test_client().get("/results")

    assert response.status_code == 200
    assert b"not available" in response.data


# --------------------------------------------------
# 3. CHART ROUTE
# --------------------------------------------------

def test_known_chart_is_served_as_png(client, results):

    if not results["charts"]:
        pytest.skip("Chart images not present on this machine")

    name = next(iter(results["charts"]))

    response = client.get(f"/charts/{name}")

    assert response.status_code == 200
    assert response.mimetype == "image/png"
    assert response.data[:8] == b"\x89PNG\r\n\x1a\n"


@pytest.mark.parametrize("name", [
    "unknown",
    "..%2Fapp.py",
    "..%2F..%2Fmodels%2Flfcc_rbf_svm.joblib",
])
def test_other_paths_are_refused(client, name):

    assert client.get(f"/charts/{name}").status_code == 404


# --------------------------------------------------
# 4. RESULT PRESENTATION
# --------------------------------------------------

def test_scale_position_is_clipped():

    assert scale_position(SCALE_MIN) == 0
    assert scale_position(SCALE_MAX) == 100
    assert scale_position(SCALE_MIN - 10) == 0
    assert scale_position(SCALE_MAX + 10) == 100


@pytest.mark.parametrize("score, verdict, tone", [
    (2.40, "Likely spoof", "critical"),
    (-1.50, "Likely bona fide", "good"),
    (0.60, "Borderline, leaning spoof", "warning"),
    (0.28, "Borderline, leaning bona fide", "warning"),
])
def test_verdict_wording(score, verdict, tone):

    view = present(fake_result(score))

    assert view["verdict"] == verdict
    assert view["tone"] == tone


def test_off_scale_scores_are_flagged():

    assert present(fake_result(9.0))["off_scale"]
    assert not present(fake_result(1.0))["off_scale"]


def test_result_page_shows_scale_and_borderline(client):

    import io
    import numpy as np
    import soundfile as sf

    time = np.arange(32000) / 16000
    buffer = io.BytesIO()
    sf.write(buffer, 0.3 * np.sin(2 * np.pi * 220 * time), 16000, format="WAV")
    buffer.seek(0)

    response = client.post(
        "/predict",
        data={"audio": (buffer, "tone.wav")},
        content_type="multipart/form-data",
    )
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert 'class="scale"' in html
    assert "Result: " in html
    assert "Borderline zone" in html


def test_api_reports_borderline_flag(client):

    import io
    import numpy as np
    import soundfile as sf

    time = np.arange(32000) / 16000
    buffer = io.BytesIO()
    sf.write(buffer, 0.3 * np.sin(2 * np.pi * 220 * time), 16000, format="WAV")
    buffer.seek(0)

    result = client.post(
        "/api/predict",
        data={"audio": (buffer, "tone.wav")},
        content_type="multipart/form-data",
    ).json

    assert result["borderline"] == (abs(result["margin"]) < BORDERLINE_MARGIN)
