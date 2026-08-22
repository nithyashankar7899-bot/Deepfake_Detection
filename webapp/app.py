"""
app.py
-------
A small Flask website for the Deepfake Video Detection project.

Lets the user upload a video in the browser and see REAL/FAKE + confidence,
using the exact same model and prediction logic as scripts/predict.py.

Run:
    python app.py
Then open http://127.0.0.1:5000 in your browser.
"""

import os
import sys
import uuid

from flask import Flask, render_template, request

# reuse the existing prediction logic from scripts/predict.py without
# duplicating any code
SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts")
sys.path.insert(0, SCRIPTS_DIR)
from predict import predict  # noqa: E402  (import after sys.path tweak, on purpose)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
CHECKPOINT_PATH = os.path.join(BASE_DIR, "..", "models", "best_model.pth")

os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB upload limit


def allowed_file(filename):
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", result=None, error=None)


@app.route("/predict", methods=["POST"])
def run_prediction():
    if not os.path.exists(CHECKPOINT_PATH):
        return render_template(
            "index.html",
            result=None,
            error=(
                "No trained model found at models/best_model.pth. "
                "Run scripts/train.py first to create one."
            ),
        )

    if "video" not in request.files or request.files["video"].filename == "":
        return render_template("index.html", result=None, error="Please choose a video file first.")

    file = request.files["video"]

    if not allowed_file(file.filename):
        return render_template(
            "index.html",
            result=None,
            error="Unsupported file type. Please upload an mp4, avi, mov, or mkv video.",
        )

    # save with a unique name to avoid collisions between users/uploads
    safe_name = f"{uuid.uuid4().hex}_{file.filename}"
    saved_path = os.path.join(UPLOAD_DIR, safe_name)
    file.save(saved_path)

    try:
        label, confidence = predict(saved_path, CHECKPOINT_PATH)
        result = {"label": label, "confidence": f"{confidence:.2f}", "filename": file.filename}
        error = None
    except Exception as e:
        result = None
        error = f"Could not process this video: {e}"
    finally:
        # clean up the uploaded file after prediction so uploads/ doesn't fill up
        if os.path.exists(saved_path):
            os.remove(saved_path)

    return render_template("index.html", result=result, error=error)


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
