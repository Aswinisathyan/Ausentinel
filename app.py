
from pathlib import Path
import os
import tempfile

from flask import Flask, abort, jsonify, render_template, request, send_file
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from features import MAX_DURATION
from inference import (
    AudioError,
    SpoofDetector,
    SUPPORTED_EXTENSIONS,
    UNRELIABLE_EXTENSIONS,
)
from results_data import load_results


# --------------------------------------------------
# 1. SETTINGS
# --------------------------------------------------

ALLOWED_EXTENSIONS = SUPPORTED_EXTENSIONS

ALLOWED_NAMES = ", ".join(
    sorted(extension.lstrip(".").upper() for extension in ALLOWED_EXTENSIONS)
)

MAX_UPLOAD_MB = 20

RESULT_NOTE = (
    "The threshold was selected on the ASVspoof 2019 LA development set. "
    "On unseen attacks the model's error rate is much higher, and accuracy "
    "on real-world recordings has not been measured. Treat this result as "
    "an experimental prediction, not a verdict."
)

# Scores this close to the threshold are often wrong. Within 0.5 of the
# threshold, balanced accuracy was 68.8% on dev (19.7% of files) and
# 65.0% on eval, against 97.9% and 83.8% further away.
BORDERLINE_MARGIN = 0.5

# Range drawn on the score scale; covers the 1st-99th percentile of
# dev and eval scores. Scores outside it are pinned to the ends.
SCALE_MIN = -4.0
SCALE_MAX = 5.0


class UploadError(Exception):
    """A request problem reported to the user with an HTTP status."""

    def __init__(self, message, status):
        super().__init__(message)
        self.message = message
        self.status = status


# --------------------------------------------------
# 2. UPLOAD HANDLING
# --------------------------------------------------

def score_upload(detector, upload_dir):
    """Validate the uploaded file, score it, and always delete it."""

    upload = request.files.get("audio")

    if upload is None:
        raise UploadError("No file was uploaded. Use the field name 'audio'.", 400)

    filename = secure_filename(upload.filename or "")

    if not filename:
        raise UploadError("Please choose an audio file.", 400)

    extension = Path(filename).suffix.lower()

    if extension in UNRELIABLE_EXTENSIONS:
        raise UploadError(UNRELIABLE_EXTENSIONS[extension], 415)

    if extension not in ALLOWED_EXTENSIONS:
        raise UploadError(
            f"Unsupported file type. Please upload one of: {ALLOWED_NAMES}.",
            415,
        )

    handle, temp_path = tempfile.mkstemp(suffix=extension, dir=upload_dir)
    os.close(handle)

    try:
        upload.save(temp_path)

        if os.path.getsize(temp_path) == 0:
            raise UploadError("The uploaded file is empty.", 400)

        result = detector.predict_file(temp_path)

    except AudioError as error:
        raise UploadError(str(error), 422) from error

    finally:
        os.remove(temp_path)

    result["filename"] = filename
    result["borderline"] = abs(result["margin"]) < BORDERLINE_MARGIN
    result["note"] = RESULT_NOTE

    return result


def scale_position(score):
    """Percentage position of a score on the drawn scale, kept within 0-100."""

    share = (score - SCALE_MIN) / (SCALE_MAX - SCALE_MIN)

    return round(min(max(share, 0.0), 1.0) * 100, 2)


def present(result):
    """Wording and positions the result card needs."""

    leaning = "spoof" if result["is_spoof"] else "bona fide"

    if result["borderline"]:
        verdict = f"Borderline, leaning {leaning}"
        tone = "warning"
    elif result["is_spoof"]:
        verdict = "Likely spoof"
        tone = "critical"
    else:
        verdict = "Likely bona fide"
        tone = "good"

    direction = "above" if result["margin"] >= 0 else "below"

    return {
        "verdict": verdict,
        "tone": tone,
        "direction": direction,
        "distance": abs(result["margin"]),
        "score_position": scale_position(result["score"]),
        "threshold_position": scale_position(result["threshold"]),
        "band_low": scale_position(result["threshold"] - BORDERLINE_MARGIN),
        "band_high": scale_position(result["threshold"] + BORDERLINE_MARGIN),
        "off_scale": not SCALE_MIN <= result["score"] <= SCALE_MAX,
    }


# --------------------------------------------------
# 3. APPLICATION
# --------------------------------------------------

def create_app(detector=None, upload_dir=None, results=None):

    app = Flask(__name__)

    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
    app.config["UPLOAD_DIR"] = upload_dir

    # Load the model and saved results once, when the app starts
    detector = detector or SpoofDetector()
    results = results if results is not None else load_results()

    @app.context_processor
    def shared():
        return {
            "threshold": detector.threshold,
            "borderline_margin": BORDERLINE_MARGIN,
            "max_duration": MAX_DURATION,
            "max_upload_mb": MAX_UPLOAD_MB,
            "allowed": ALLOWED_NAMES,
        }

    def page(status=200, **context):
        return render_template(
            "index.html",
            active="analyse",
            accept=",".join(sorted(ALLOWED_EXTENSIONS)),
            **context,
        ), status

    @app.get("/")
    def index():
        return page()

    @app.post("/predict")
    def predict_page():

        try:
            result = score_upload(detector, app.config["UPLOAD_DIR"])
        except UploadError as error:
            return page(error.status, error=error.message)

        return page(result=result, view=present(result))

    @app.get("/results")
    def results_page():
        return render_template("results.html", active="results", results=results)

    @app.get("/about")
    def about_page():
        return render_template("about.html", active="about", results=results)

    @app.get("/charts/<name>")
    def chart(name):

        # Only named charts can be served, never arbitrary paths
        chart_info = results["charts"].get(name)

        if chart_info is None:
            abort(404)

        return send_file(chart_info["path"], mimetype="image/png", max_age=3600)

    @app.post("/api/predict")
    def predict_api():

        try:
            result = score_upload(detector, app.config["UPLOAD_DIR"])
        except UploadError as error:
            return jsonify(error=error.message), error.status

        return jsonify(result)

    @app.get("/health")
    def health():
        return jsonify(
            status="ok",
            model_loaded=True,
            threshold=detector.threshold,
            threshold_source=detector.threshold_source,
        )

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(error):

        message = f"The file is larger than the {MAX_UPLOAD_MB} MB limit."

        if request.path.startswith("/api/"):
            return jsonify(error=message), 413

        return page(413, error=message)

    return app


if __name__ == "__main__":

    # Flask's development server, for local testing only.
    # Use serve.py (Waitress) to run the app for other people.
    # Debug mode is off unless FLASK_DEBUG=1 is set
    create_app().run(
        host="127.0.0.1",
        port=5000,
        debug=os.environ.get("FLASK_DEBUG") == "1",
    )
