from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "results/best_model.pth"
MANIFEST_PATH = ROOT / "data/processed/dataset_manifest.csv"
RESULTS_PATH = ROOT / "results/robustness_results.json"
CHART_PATH = ROOT / "results/robustness_comparison.png"
LABELS = ["Real", "AI-Generated", "Manipulated"]
BATCH_SIZE = 32

PREPROCESS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(LABELS))
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint)
    model.eval()
    return model


def read_test_rows():
    with MANIFEST_PATH.open("r", newline="", encoding="utf-8-sig") as manifest:
        rows = [row for row in csv.DictReader(manifest) if row["split"].strip().lower() == "test"]
    if not rows:
        raise ValueError(f"No test rows found in {MANIFEST_PATH}")
    for row in rows:
        row["label"] = int(row["label"])
        if not 0 <= row["label"] < len(LABELS):
            raise ValueError(f"Invalid class index {row['label']} in manifest")
        image_path = Path(row["filepath"])
        if not image_path.is_absolute():
            image_path = ROOT / image_path
        row["image_path"] = image_path
    return rows


def transform_image(image: Image.Image, mode: str) -> Image.Image:
    if mode == "original":
        return image
    if mode == "cropped":
        image = ImageOps.fit(image, (180, 180), method=Image.Resampling.BICUBIC)
        return image.resize((224, 224), Image.Resampling.BICUBIC)
    if mode == "resized":
        return image.resize((224, 224), Image.Resampling.BICUBIC)
    if mode == "jpeg_compressed":
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=60)
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            return compressed.convert("RGB")
    raise ValueError(f"Unsupported mode: {mode}")


def evaluate_mode(model, rows, mode):
    correct = 0
    for start in range(0, len(rows), BATCH_SIZE):
        batch_rows = rows[start:start + BATCH_SIZE]
        tensors = []
        for row in batch_rows:
            with Image.open(row["image_path"]) as source:
                image = transform_image(source.convert("RGB"), mode)
                tensors.append(PREPROCESS(image))
        batch = torch.stack(tensors)
        labels = torch.tensor([row["label"] for row in batch_rows])
        with torch.inference_mode():
            predictions = model(batch).argmax(dim=1)
        correct += int((predictions == labels).sum().item())
        processed = start + len(batch_rows)
        if processed % 300 == 0 or processed == len(rows):
            print(f"  {mode}: {processed}/{len(rows)} images", flush=True)
    return {"correct": correct, "total": len(rows), "accuracy": correct / len(rows)}


def save_chart(results):
    names = list(results)
    accuracies = [results[name]["accuracy"] * 100 for name in names]
    labels = ["Original", "Cropped", "Resized", "JPEG compressed"]
    figure, axis = plt.subplots(figsize=(8, 5))
    bars = axis.bar(labels, accuracies, color=["#2563eb", "#0f766e", "#7c3aed", "#d97706"])
    axis.set_ylabel("Accuracy (%)")
    axis.set_ylim(0, 100)
    axis.set_title("Robustness on the test set")
    axis.bar_label(bars, fmt="%.2f%%", padding=4)
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    figure.tight_layout()
    figure.savefig(CHART_PATH, dpi=160)
    plt.close(figure)


def main():
    rows = read_test_rows()
    if len(rows) != 900:
        raise ValueError(f"Expected 900 test images; found {len(rows)}")
    missing = [str(row["image_path"]) for row in rows if not row["image_path"].is_file()]
    if missing:
        raise FileNotFoundError(f"Missing test images, including: {missing[0]} ({len(missing)} missing)")

    model = load_model()
    results = {}
    for mode in ["original", "cropped", "resized", "jpeg_compressed"]:
        print(f"Evaluating {mode} on {len(rows)} test images...", flush=True)
        results[mode] = evaluate_mode(model, rows, mode)
        print(f"  accuracy: {results[mode]['accuracy'] * 100:.2f}%", flush=True)

    output = {
        "model": str(MODEL_PATH.relative_to(ROOT)),
        "dataset_manifest": str(MANIFEST_PATH.relative_to(ROOT)),
        "test_images": len(rows),
        "jpeg_quality": 60,
        "results": results,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    save_chart(results)
    print("\nAccuracy summary:")
    for mode, result in results.items():
        print(f"{mode.replace('_', ' ').title():<18} {result['accuracy'] * 100:.2f}% ({result['correct']}/{result['total']})")
    print(f"Saved results: {RESULTS_PATH}")
    print(f"Saved chart:   {CHART_PATH}")


if __name__ == "__main__":
    main()
