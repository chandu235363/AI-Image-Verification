from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from features.fused_features import (
    FEATURE_COUNT,
    FEATURE_IMAGE_SIZE,
    extract_handcrafted_features,
    preprocess_feature_image,
)

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data/processed/dataset_manifest.csv"
BASELINE = ROOT / "results/best_model.pth"
MODEL_PATH = ROOT / "results/fused_best_model.pth"
METRICS_PATH = ROOT / "results/fused_metrics.json"
MATRIX_PATH = ROOT / "results/fused_confusion_matrix.png"
LABELS = ["Real", "AI-Generated", "Manipulated"]
CLASS_TO_IDX = {name: index for index, name in enumerate(LABELS)}
IMAGE_SIZE = 112
BATCH_SIZE = 32
EPOCHS = 2


class FusedDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, transform, handcrafted_features, scaler_mean, scaler_scale):
        self.rows = rows.reset_index(drop=True)
        self.transform = transform
        self.handcrafted_features = torch.from_numpy(handcrafted_features)
        self.scaler_mean = torch.from_numpy(scaler_mean)
        self.scaler_scale = torch.from_numpy(scaler_scale)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        with Image.open(row.filepath) as source:
            image = source.convert("RGB")
        handcrafted = (self.handcrafted_features[index] - self.scaler_mean) / self.scaler_scale
        return (self.transform(image), handcrafted, int(row.label))


def extract_split_features(rows: pd.DataFrame) -> np.ndarray:
    """Extract features after the shared 224x224 preprocessing, once per image."""
    values = []
    for path in rows["filepath"]:
        with Image.open(path) as source:
            image = preprocess_feature_image(source)
        values.append(extract_handcrafted_features(np.asarray(image)))
    return np.stack(values).astype(np.float32)


class FusedResNet18(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Linear(512 + FEATURE_COUNT, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.25), nn.Linear(128, len(LABELS)),
        )

    def forward(self, images, handcrafted):
        image_features = self.backbone(images)  # 512-dimensional pre-classifier ResNet vector
        fused_features = torch.cat((image_features, handcrafted), dim=1)
        return self.classifier(fused_features)


def build_model(device):
    backbone = models.resnet18(weights=None)
    initialized_from_baseline = False
    if BASELINE.exists():
        checkpoint = torch.load(BASELINE, map_location="cpu", weights_only=False)
        state = checkpoint.get("model_state_dict", checkpoint)
        compatible = {key: value for key, value in state.items() if not key.startswith("fc.")}
        result = backbone.load_state_dict(compatible, strict=False)
        initialized_from_baseline = not result.missing_keys or set(result.missing_keys) == {"fc.weight", "fc.bias"}
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    return FusedResNet18(backbone).to(device), initialized_from_baseline


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = total_correct = total_items = 0
    with (torch.enable_grad() if training else torch.no_grad()):
        for images, handcrafted, labels in loader:
            images, handcrafted, labels = images.to(device), handcrafted.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images, handcrafted)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(labels)
            total_correct += (logits.argmax(1) == labels).sum().item()
            total_items += len(labels)
    return total_loss / total_items, total_correct / total_items


def main():
    if not MANIFEST.exists():
        raise FileNotFoundError(f"Prepared dataset manifest not found: {MANIFEST}")
    manifest = pd.read_csv(MANIFEST)
    required = {"filepath", "label", "class_name", "split"}
    if required - set(manifest.columns):
        raise ValueError(f"Manifest missing columns: {sorted(required - set(manifest.columns))}")
    if set(manifest.label.unique()) != {0, 1, 2}:
        raise ValueError("The prepared dataset must use labels 0, 1, and 2.")
    splits = {name: manifest[manifest.split == name] for name in ("train", "val", "test")}
    if any(frame.empty for frame in splits.values()):
        raise ValueError("Train, validation, and test splits must all contain images.")
    print("Extracting FFT, Haar wavelet, and noise features from 224x224 RGB images...", flush=True)
    raw_features = {name: extract_split_features(frame) for name, frame in splits.items()}
    # Fit the scaler strictly on training rows. Validation and test use these same statistics.
    scaler_mean = raw_features["train"].mean(axis=0, dtype=np.float64).astype(np.float32)
    scaler_scale = raw_features["train"].std(axis=0, dtype=np.float64).astype(np.float32)
    scaler_scale[scaler_scale < 1e-6] = 1.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training fused ResNet18 on {device}; rows: " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    loaders = {
        name: DataLoader(
            FusedDataset(rows, transform, raw_features[name], scaler_mean, scaler_scale),
            batch_size=BATCH_SIZE, shuffle=(name == "train"), num_workers=0,
        )
        for name, rows in splits.items()
    }
    model, baseline_initialized = build_model(device)
    print(f"ResNet18 initialized from existing baseline checkpoint: {baseline_initialized}")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    best_val = -1.0
    for epoch in range(EPOCHS):
        train_loss, train_acc = run_epoch(model, loaders["train"], criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, loaders["val"], criterion, device)
        print(f"Epoch {epoch + 1}/{EPOCHS} | training loss={train_loss:.4f} accuracy={train_acc:.4f} | validation loss={val_loss:.4f} accuracy={val_acc:.4f}", flush=True)
        if val_acc > best_val:
            best_val = val_acc
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(), "class_to_idx": CLASS_TO_IDX,
                "image_size": IMAGE_SIZE, "feature_count": FEATURE_COUNT,
                "feature_image_size": FEATURE_IMAGE_SIZE,
                "feature_scaler_mean": scaler_mean,
                "feature_scaler_scale": scaler_scale,
                "feature_names": ["fft_mean", "fft_std", "fft_low_energy", "fft_high_energy", "fft_high_low_ratio",
                                  "haar_approx_mean", "haar_approx_std", "haar_horizontal_mean", "haar_horizontal_std",
                                  "haar_vertical_mean", "haar_vertical_std", "haar_diagonal_mean", "haar_diagonal_std",
                                  "noise_mean", "noise_std", "noise_abs_mean", "noise_energy"],
                "epochs": EPOCHS, "baseline_initialized": baseline_initialized,
            }, MODEL_PATH)
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    truth, predictions = [], []
    with torch.no_grad():
        for images, handcrafted, labels in loaders["test"]:
            predicted = model(images.to(device), handcrafted.to(device)).argmax(1).cpu().tolist()
            truth.extend(labels.tolist())
            predictions.extend(predicted)
    metrics = {
        "accuracy": accuracy_score(truth, predictions),
        "precision": precision_score(truth, predictions, average="weighted", zero_division=0),
        "recall": recall_score(truth, predictions, average="weighted", zero_division=0),
        "f1_score": f1_score(truth, predictions, average="weighted", zero_division=0),
    }
    matrix = confusion_matrix(truth, predictions, labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set(xticks=range(3), yticks=range(3), xticklabels=LABELS, yticklabels=LABELS,
           xlabel="Predicted label", ylabel="True label", title="Fused ResNet18 Confusion Matrix")
    for r in range(3):
        for c in range(3):
            ax.text(c, r, matrix[r, c], ha="center", va="center")
    fig.tight_layout()
    fig.savefig(MATRIX_PATH, dpi=150)
    plt.close(fig)
    with METRICS_PATH.open("w", encoding="utf-8") as stream:
        json.dump({"classes": CLASS_TO_IDX, "epochs": EPOCHS, "best_validation_accuracy": best_val,
                   "baseline_initialized": baseline_initialized, "handcrafted_feature_count": FEATURE_COUNT,
                   "test_metrics": metrics}, stream, indent=2)
    print("Final test metrics")
    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-score: {metrics['f1_score']:.4f}")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    print(f"Saved confusion matrix: {MATRIX_PATH}")


if __name__ == "__main__":
    main()
