
import io
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app import create_app
from inference import SpoofDetector, ffmpeg_executable


ROOT = Path(__file__).resolve().parent.parent

DEV_AUDIO_DIR = (
    ROOT / "data" / "ASVspoof2019_LA" / "LA" / "ASVspoof2019_LA_dev" / "flac"
)

# Generated locally and not stored in git
DEV_SCORES_PATH = ROOT / "artifacts" / "threshold_study" / "dev_scores.npz"


@pytest.fixture(scope="module")
def detector():

    return SpoofDetector()


@pytest.fixture
def upload_dir(tmp_path):

    return tmp_path


@pytest.fixture
def client(detector, upload_dir):

    app = create_app(detector=detector, upload_dir=str(upload_dir))
    app.config["TESTING"] = True

    return app.test_client()


def audio_bytes(duration=2.0, sample_rate=16000, file_format="WAV", channels=1):
    """Encode a test tone as an in-memory audio file."""

    time = np.arange(int(sample_rate * duration)) / sample_rate
    tone = 0.3 * np.sin(2 * np.pi * 220 * time)

    if channels == 2:
        tone = np.stack([tone, tone], axis=1)

    buffer = io.BytesIO()
    sf.write(buffer, tone, sample_rate, format=file_format)
    buffer.seek(0)

    return buffer


def compressed_bytes(tmp_path, extension):
    """Encode a test tone as MP3 or M4A with the bundled ffmpeg."""

    source = tmp_path / "source.wav"
    target = tmp_path / f"encoded{extension}"

    source.write_bytes(audio_bytes().read())

    subprocess.run(
        [ffmpeg_executable(), "-nostdin", "-loglevel", "error", "-y",
         "-i", str(source), str(target)],
        check=True,
        capture_output=True,
    )

    data = io.BytesIO(target.read_bytes())

    source.unlink()
    target.unlink()

    return data


def upload(client, data, filename, path="/api/predict"):

    return client.post(
        path,
        data={"audio": (data, filename)},
        content_type="multipart/form-data",
    )


# --------------------------------------------------
# 1. SUCCESSFUL PREDICTIONS
# --------------------------------------------------

def test_health_reports_threshold(client, detector):

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json["threshold"] == detector.threshold


def test_index_page_loads(client):

    response = client.get("/")

    assert response.status_code == 200
    assert b'name="audio"' in response.data


def test_api_scores_wav(client):

    response = upload(client, audio_bytes(), "tone.wav")
    result = response.json

    assert response.status_code == 200
    assert result["label"] in {"spoof", "bonafide"}
    assert result["is_spoof"] == (result["score"] >= result["threshold"])
    assert result["margin"] == pytest.approx(result["score"] - result["threshold"])
    assert result["duration_seconds"] == pytest.approx(2.0)
    assert result["filename"] == "tone.wav"


def test_api_scores_stereo_flac_at_other_rate(client):

    response = upload(
        client,
        audio_bytes(sample_rate=44100, file_format="FLAC", channels=2),
        "tone.flac",
    )

    assert response.status_code == 200
    assert response.json["sample_rate"] == 44100
    assert response.json["channels"] == 2


def test_long_audio_reports_truncation(client):

    response = upload(client, audio_bytes(duration=6.0), "long.wav")

    assert response.status_code == 200
    assert response.json["analysed_seconds"] == pytest.approx(4.0)
    assert any("first 4 seconds" in w for w in response.json["warnings"])


def test_short_audio_is_flagged(client):

    response = upload(client, audio_bytes(duration=0.5), "short.wav")

    assert response.status_code == 200
    assert any("shorter than 1 second" in w for w in response.json["warnings"])


def test_html_form_shows_result(client):

    response = upload(client, audio_bytes(), "tone.wav", path="/predict")

    assert response.status_code == 200
    assert b"Result:" in response.data
    assert b"experimental prediction" in response.data


def test_api_scores_m4a(client, tmp_path_factory):

    data = compressed_bytes(tmp_path_factory.mktemp("encode"), ".m4a")

    response = upload(client, data, "tone.m4a")
    result = response.json

    assert response.status_code == 200
    assert result["format"] == "M4A"
    assert result["duration_seconds"] == pytest.approx(2.0, abs=0.1)
    assert any("compressed format" in w for w in result["warnings"])


def test_corrupt_m4a(client):

    response = upload(client, io.BytesIO(b"this is not audio"), "fake.m4a")

    assert response.status_code == 422
    assert "could not be read" in response.json["error"]


def test_mp3_is_rejected_with_reason(client, tmp_path_factory):

    data = compressed_bytes(tmp_path_factory.mktemp("encode"), ".mp3")

    response = upload(client, data, "tone.mp3")

    assert response.status_code == 415
    assert "MP3 compression" in response.json["error"]


def test_uploads_are_deleted(client, upload_dir, tmp_path_factory):

    m4a = compressed_bytes(tmp_path_factory.mktemp("encode"), ".m4a")

    upload(client, audio_bytes(), "tone.wav")
    upload(client, m4a, "tone.m4a")
    upload(client, io.BytesIO(b"not audio"), "broken.wav")
    upload(client, io.BytesIO(b"not audio"), "broken.m4a")

    # Covers both the upload and the WAV converted from it
    assert list(upload_dir.iterdir()) == []


@pytest.mark.skipif(
    not (DEV_AUDIO_DIR.exists() and DEV_SCORES_PATH.exists()),
    reason="ASVspoof dev audio or saved dev scores not present on this machine",
)
def test_api_matches_threshold_study_scores(client):

    saved = np.load(DEV_SCORES_PATH)

    for index in (0, len(saved["score"]) // 2, len(saved["score"]) - 1):

        file_id = str(saved["file_id"][index])

        with open(DEV_AUDIO_DIR / f"{file_id}.flac", "rb") as file:
            response = upload(client, io.BytesIO(file.read()), f"{file_id}.flac")

        assert response.status_code == 200
        assert response.json["score"] == pytest.approx(saved["score"][index], abs=1e-9)
        assert response.json["is_spoof"] == (saved["score"][index] >= response.json["threshold"])


# --------------------------------------------------
# 2. REJECTED UPLOADS
# --------------------------------------------------

def test_missing_file_field(client):

    response = client.post("/api/predict", data={}, content_type="multipart/form-data")

    assert response.status_code == 400


def test_empty_filename(client):

    response = upload(client, io.BytesIO(b""), "")

    assert response.status_code == 400


def test_unsupported_extension(client):

    response = upload(client, io.BytesIO(b"hello"), "notes.txt")

    assert response.status_code == 415


def test_empty_file(client):

    response = upload(client, io.BytesIO(b""), "empty.wav")

    assert response.status_code == 400


def test_corrupt_audio(client):

    response = upload(client, io.BytesIO(b"this is not audio"), "fake.wav")

    assert response.status_code == 422
    assert "could not be read" in response.json["error"]


def test_html_form_shows_error(client):

    response = upload(client, io.BytesIO(b"hello"), "notes.txt", path="/predict")

    assert response.status_code == 415
    assert b"Unsupported file type" in response.data


def test_oversized_upload(detector, upload_dir):

    app = create_app(detector=detector, upload_dir=str(upload_dir))
    app.config["MAX_CONTENT_LENGTH"] = 1024

    response = upload(app.test_client(), audio_bytes(), "big.wav")

    assert response.status_code == 413
    assert "limit" in response.json["error"]
