from __future__ import annotations

from importlib import import_module
from pathlib import Path
from argparse import Namespace

from flask import Flask, render_template, request, send_from_directory, url_for
from werkzeug.utils import secure_filename

# Import the EfficientNet fused predictor, whose filename starts with a number.
predictor_module = import_module("src.13_predict_efficientnet_image")
predict_image = predictor_module.predict_image
load_predictor = predictor_module.load_predictor
predict_image_details = predictor_module.predict_image_details
robustness_module = import_module("src.05_robustness_testing")
TRANSFORM_ARGS = Namespace(crop_fraction=0.9, resize_scale=0.75, jpeg_quality=60)

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
    robustness_results = None

    if request.method == "POST":
        file = request.files.get("image")
        if file and file.filename:
            filename = secure_filename(file.filename)
            image_path = UPLOAD_FOLDER / filename
            file.save(image_path)
            try:
                predictor = load_predictor()
                original = predict_image_details(image_path, predictor=predictor)
                prediction = original["predicted_class"]
                confidence = f"{original['confidence'] * 100:.2f}%"
                if request.form.get("robustness_demo") == "on":
                    from PIL import Image
                    with Image.open(image_path) as source:
                        rgb = source.convert("RGB")
                    robustness_results = []
                    for key, display_name, operations in robustness_module.CONDITIONS:
                        transformed = robustness_module.transform_image(rgb, operations, TRANSFORM_ARGS)
                        result = predict_image_details(transformed, predictor=predictor)
                        robustness_results.append({
                            "condition": display_name,
                            "prediction": result["predicted_class"],
                            "confidence": f"{result['confidence'] * 100:.2f}%",
                        })
                image_url = url_for("uploaded_file", filename=filename)
            except Exception as exc:
                prediction = "Error"
                confidence = str(exc)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        image_url=image_url,
        robustness_results=robustness_results,
    )


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
