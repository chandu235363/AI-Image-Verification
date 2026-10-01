from __future__ import annotations

"""Evaluate the fused checkpoint on manifest test rows excluded from training."""

import hashlib
import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torchvision import transforms

from features.fused_features import (
    FEATURE_COUNT,
    FEATURE_IMAGE_SIZE,
    extract_handcrafted_features,
    preprocess_feature_image,
)
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))
_predictor_spec = importlib.util.spec_from_file_location(
    "fused_predictor", ROOT / "src/10_predict_fused_image.py"
)
_predictor = importlib.util.module_from_spec(_predictor_spec)
_predictor_spec.loader.exec_module(_predictor)
FusedResNet18 = _predictor.FusedResNet18
MANIFEST = ROOT / "data/processed/dataset_manifest.csv"
MODEL_PATH = ROOT / "results/fused_best_model.pth"
METRICS_PATH = ROOT / "results/unseen_evaluation_metrics.json"
MATRIX_PATH = ROOT / "results/unseen_confusion_matrix.png"
LABELS = ["Real", "AI-Generated", "Manipulated"]
CLASS_TO_IDX = {name: i for i, name in enumerate(LABELS)}
EXPECTED_FEATURES = 512 + FEATURE_COUNT


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_evaluation_rows() -> tuple[pd.DataFrame, dict[str, int]]:
    if not MANIFEST.is_file():
        raise FileNotFoundError(f"Dataset manifest not found: {MANIFEST}")
    manifest = pd.read_csv(MANIFEST)
    required = {"filepath", "label", "class_name", "split"}
    if required - set(manifest.columns):
        raise ValueError(f"Manifest missing columns: {sorted(required - set(manifest.columns))}")
    train = manifest[manifest["split"] == "train"]
    test = manifest[manifest["split"] == "test"].copy()
    if test.empty:
        raise ValueError("Manifest has no test rows; refusing to substitute training data.")
    if set(test["class_name"]) != set(LABELS):
        raise ValueError("Test split must contain all three known classes.")
    for row in manifest.itertuples(index=False):
        if row.class_name not in CLASS_TO_IDX or int(row.label) != CLASS_TO_IDX[row.class_name]:
            raise ValueError(f"Invalid class/label mapping in manifest: {row}")

    train_paths = [Path(p) for p in train["filepath"]]
    test_paths = [Path(p) for p in test["filepath"]]
    missing = [p for p in train_paths + test_paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Manifest image is missing: {missing[0]}")
    train_paths_resolved = {p.resolve() for p in train_paths}
    if train_paths_resolved.intersection(p.resolve() for p in test_paths):
        raise ValueError("A test image path also appears in the training split.")

    # Path separation alone cannot catch copied images, so compare file contents too.
    train_hashes = {sha256(p) for p in train_paths}
    overlaps = [p for p in test_paths if sha256(p) in train_hashes]
    if overlaps:
        raise ValueError(
            "Test images contain byte-identical files from training; refusing evaluation. "
            f"Example: {overlaps[0]}"
        )
    counts = test["class_name"].value_counts().reindex(LABELS, fill_value=0).astype(int).to_dict()
    if any(count == 0 for count in counts.values()):
        raise ValueError(f"Every class needs test images; found counts: {counts}")
    return test.reset_index(drop=True), counts


def main() -> None:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Fused checkpoint not found: {MODEL_PATH}")
    rows, counts = load_evaluation_rows()
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    if checkpoint.get("feature_count") != FEATURE_COUNT:
        raise ValueError("Checkpoint feature count does not match this extractor.")
    if checkpoint.get("feature_image_size") != FEATURE_IMAGE_SIZE:
        raise ValueError("Checkpoint handcrafted preprocessing size does not match.")
    if checkpoint.get("class_to_idx") != CLASS_TO_IDX:
        raise ValueError("Checkpoint class mapping does not match the evaluation labels.")
    cnn_size = int(checkpoint.get("image_size", 0))
    mean = torch.as_tensor(checkpoint.get("feature_scaler_mean"), dtype=torch.float32)
    scale = torch.as_tensor(checkpoint.get("feature_scaler_scale"), dtype=torch.float32)
    if mean.shape != (FEATURE_COUNT,) or scale.shape != (FEATURE_COUNT,):
        raise ValueError("Checkpoint lacks valid training-fitted scaler values.")
    if cnn_size <= 0 or not torch.isfinite(mean).all() or not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("Checkpoint preprocessing/scaler values are invalid.")

    model = FusedResNet18()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((cnn_size, cnn_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    truth, predicted, records = [], [], []
    with torch.no_grad():
        for row in rows.itertuples(index=False):
            path = Path(row.filepath)
            with Image.open(path) as source:
                rgb = source.convert("RGB")
                feature_image = preprocess_feature_image(rgb)
                raw_features = extract_handcrafted_features(np.asarray(feature_image))
                handcrafted = (torch.from_numpy(raw_features) - mean) / scale
                image_tensor = transform(rgb).unsqueeze(0)
            if handcrafted.numel() + 512 != EXPECTED_FEATURES:
                raise ValueError("Fused feature vector is not 529 values.")
            probs = torch.softmax(model(image_tensor, handcrafted.unsqueeze(0)), dim=1)[0]
            pred_idx = int(probs.argmax())
            actual_idx = CLASS_TO_IDX[row.class_name]
            confidence = float(probs[pred_idx])
            truth.append(actual_idx)
            predicted.append(pred_idx)
            record = {
                "image": str(path.resolve()),
                "sha256": sha256(path),
                "actual_class": row.class_name,
                "predicted_class": LABELS[pred_idx],
                "confidence": confidence,
            }
            records.append(record)
            print(
                f"{record['image']} | actual={row.class_name} | "
                f"predicted={LABELS[pred_idx]} | confidence={confidence:.4f}"
            )

    metrics = {
        "accuracy": float(accuracy_score(truth, predicted)),
        "precision_weighted": float(precision_score(truth, predicted, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(truth, predicted, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(truth, predicted, average="weighted", zero_division=0)),
    }
    matrix = confusion_matrix(truth, predicted, labels=range(len(LABELS)))
    results = {
        "evaluation": "manifest test split; test rows verified disjoint from train by resolved path and SHA-256",
        "training_images": int((pd.read_csv(MANIFEST)["split"] == "train").sum()),
        "evaluation_images": len(rows),
        "class_counts": counts,
        "classes": LABELS,
        "checkpoint": str(MODEL_PATH.resolve()),
        "preprocessing": {
            "resnet_resize": [cnn_size, cnn_size],
            "handcrafted_resize": [FEATURE_IMAGE_SIZE, FEATURE_IMAGE_SIZE],
            "handcrafted_features": {"fft": 5, "haar_wavelet": 8, "noise": 4},
            "handcrafted_scaler": "checkpoint training-derived mean and scale",
            "resnet_features": 512,
            "fused_features": EXPECTED_FEATURES,
        },
        "metrics": metrics,
        "confusion_matrix": {"labels": LABELS, "rows_actual_columns_predicted": matrix.tolist()},
        "predictions": records,
    }
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set(
        xticks=range(3), yticks=range(3), xticklabels=LABELS, yticklabels=LABELS,
        xlabel="Predicted label", ylabel="Actual label",
        title="Unseen Test Split Confusion Matrix",
    )
    for r in range(3):
        for c in range(3):
            ax.text(c, r, matrix[r, c], ha="center", va="center")
    fig.tight_layout()
    fig.savefig(MATRIX_PATH, dpi=150)
    plt.close(fig)

    print("\nUnseen evaluation metrics (weighted precision/recall/F1)")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    print(f"Images per class: {counts}")
    print(f"Saved metrics and per-image predictions: {METRICS_PATH}")
    print(f"Saved confusion matrix: {MATRIX_PATH}")


if __name__ == "__main__":
    main()
