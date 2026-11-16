
from pathlib import Path
import os
import tempfile

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from inference import AudioError, SpoofDetector


# --------------------------------------------------
# 1. SETTINGS
# --------------------------------------------------

ALLOWED_EXTENSIONS = {".wav", ".flac"}

MAX_UPLOAD_MB = 20

RESULT_NOTE = (
    "The threshold was selected on the ASVspoof 2019 LA development set. "
    "On unseen attacks the model's error rate is much higher, and accuracy "
    "on real-world recordings has not been measured. Treat this result as "
    "an experimental prediction, not a verdict."
)


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

    if extension not in ALLOWED_EXTENSIONS:
        raise UploadError(
            "Unsupported file type. Please upload a WAV or FLAC file.",
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
    result["note"] = RESULT_NOTE

    return result


# --------------------------------------------------
# 3. APPLICATION
# --------------------------------------------------

def create_app(detector=None, upload_dir=None):

    app = Flask(__name__)

    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
    app.config["UPLOAD_DIR"] = upload_dir

    # Load the model once, when the app starts
    detector = detector or SpoofDetector()

    def page(status=200, **context):
        return render_template(
            "index.html",
            max_upload_mb=MAX_UPLOAD_MB,
            allowed=", ".join(sorted(e.lstrip(".").upper() for e in ALLOWED_EXTENSIONS)),
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

        return page(result=result)

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

    # Debug mode is off unless FLASK_DEBUG=1 is set
    create_app().run(
        host="127.0.0.1",
        port=5000,
        debug=os.environ.get("FLASK_DEBUG") == "1",
    )
