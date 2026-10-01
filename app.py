from __future__ import annotations

from importlib import import_module
from pathlib import Path

from flask import Flask, render_template, request, send_from_directory, url_for
from werkzeug.utils import secure_filename

# Import the EfficientNet fused predictor, whose filename starts with a number.
predict_image = import_module("src.13_predict_efficientnet_image").predict_image

app = Flask(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent
UPLOAD_FOLDER = PROJECT_ROOT / "data/uploads"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    image_url = None

    if request.method == "POST":
        file = request.files.get("image")
        if file and file.filename:
            filename = secure_filename(file.filename)
            image_path = UPLOAD_FOLDER / filename
            file.save(image_path)
            try:
                prediction, confidence = predict_image(image_path)
                confidence = f"{confidence * 100:.2f}%"
                image_url = url_for("uploaded_file", filename=filename)
            except Exception as exc:
                prediction = "Error"
                confidence = str(exc)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        image_url=image_url,
    )


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
