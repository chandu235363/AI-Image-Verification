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
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights

from features.fused_features import (
    FEATURE_COUNT,
    FEATURE_IMAGE_SIZE,
    extract_handcrafted_features,
    preprocess_feature_image,
)

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data/processed/dataset_manifest.csv"
MODEL_PATH = ROOT / "results/efficientnet_fused_best_model.pth"
METRICS_PATH = ROOT / "results/efficientnet_fused_metrics.json"
MATRIX_PATH = ROOT / "results/efficientnet_fused_confusion_matrix.png"
LABELS = ["Real", "AI-Generated", "Manipulated"]
CLASS_TO_IDX = {name: index for index, name in enumerate(LABELS)}
BATCH_SIZE = 16
EPOCHS = 2
NUM_WORKERS = 0
WEIGHTS = EfficientNet_B0_Weights.DEFAULT


class EfficientNetFused(nn.Module):
    def __init__(self, backbone: nn.Module, deep_feature_dim: int):
        super().__init__()
        self.backbone = backbone
        self.deep_feature_dim = deep_feature_dim
        self.classifier = nn.Sequential(
            nn.Linear(deep_feature_dim + FEATURE_COUNT, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, len(LABELS)),
        )

    def extract_deep_features(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone.features(images)
        pooled = self.backbone.avgpool(features)
        return torch.flatten(pooled, 1)

    def forward(self, images: torch.Tensor, handcrafted: torch.Tensor) -> torch.Tensor:
        deep = self.extract_deep_features(images)
        fused = torch.cat((deep, handcrafted), dim=1)
        return self.classifier(fused)


class ImageRows(Dataset):
    def __init__(self, rows: pd.DataFrame, transform, raw_features: np.ndarray, mean: np.ndarray, scale: np.ndarray):
        self.rows = rows.reset_index(drop=True)
        self.transform = transform
        self.raw_features = torch.from_numpy(raw_features)
        self.mean = torch.from_numpy(mean)
        self.scale = torch.from_numpy(scale)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        with Image.open(row.filepath) as source:
            image = source.convert("RGB")
        handcrafted = (self.raw_features[index] - self.mean) / self.scale
        return self.transform(image), handcrafted, int(row.label)


def load_splits() -> dict[str, pd.DataFrame]:
    if not MANIFEST.is_file():
        raise FileNotFoundError(f"Dataset manifest not found: {MANIFEST}")
    manifest = pd.read_csv(MANIFEST)
    required = {"filepath", "label", "class_name", "split"}
    if required - set(manifest.columns):
        raise ValueError(f"Manifest missing columns: {sorted(required - set(manifest.columns))}")
    if set(manifest["split"].unique()) != {"train", "val", "test"}:
        raise ValueError("Expected the existing train, val, and test splits in the manifest.")
    if set(manifest["class_name"]) != set(LABELS):
        raise ValueError("Manifest must contain exactly the three expected classes.")
    for row in manifest.itertuples(index=False):
        if row.class_name not in CLASS_TO_IDX or int(row.label) != CLASS_TO_IDX[row.class_name]:
            raise ValueError(f"Unexpected label mapping: {row}")
        if not Path(row.filepath).is_file():
            raise FileNotFoundError(f"Manifest image is missing: {row.filepath}")
    splits = {name: manifest[manifest.split == name].copy().reset_index(drop=True) for name in ("train", "val", "test")}
    expected = {"train": 4200, "val": 900, "test": 900}
    for name, count in expected.items():
        if len(splits[name]) != count:
            raise ValueError(f"Expected {count} {name} images, found {len(splits[name])}; refusing altered split.")
    return splits


def extract_handcrafted(rows: pd.DataFrame) -> np.ndarray:
    results = []
    for path in rows["filepath"]:
        with Image.open(path) as source:
            image = preprocess_feature_image(source)
        results.append(extract_handcrafted_features(np.asarray(image)))
    return np.stack(results).astype(np.float32)


def extract_deep(model, loader, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    deep_values, handcrafted_values, labels = [], [], []
    model.eval()
    with torch.inference_mode():
        for images, handcrafted, target in loader:
            deep_values.append(model.extract_deep_features(images.to(device)).cpu())
            handcrafted_values.append(handcrafted)
            labels.append(target)
    return torch.cat(deep_values), torch.cat(handcrafted_values), torch.cat(labels)


def evaluate_classifier(classifier, deep, handcrafted, labels, device):
    classifier.eval()
    predictions, targets = [], []
    with torch.inference_mode():
        for start in range(0, len(labels), 256):
            end = start + 256
            logits = classifier(torch.cat((deep[start:end], handcrafted[start:end]), dim=1).to(device))
            predictions.extend(logits.argmax(1).cpu().tolist())
            targets.extend(labels[start:end].tolist())
    return targets, predictions


def save_confusion_matrix(matrix: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    shown = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(shown, ax=ax)
    ax.set(xticks=range(3), yticks=range(3), xticklabels=LABELS, yticklabels=LABELS,
           xlabel="Predicted label", ylabel="Actual label", title="EfficientNet-B0 Fused Confusion Matrix")
    for r in range(3):
        for c in range(3):
            ax.text(c, r, matrix[r, c], ha="center", va="center")
    fig.tight_layout()
    fig.savefig(MATRIX_PATH, dpi=150)
    plt.close(fig)


def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)
    splits = load_splits()
    print("Loading torchvision EfficientNet-B0 ImageNet weights. A one-time download may be required.", flush=True)
    device = torch.device("cpu")
    transform = WEIGHTS.transforms()
    print(f"Torchvision weight preprocessing: {transform}", flush=True)
    raw = {name: extract_handcrafted(rows) for name, rows in splits.items()}
    mean = raw["train"].mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = raw["train"].std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-6] = 1.0

    loaders = {
        name: DataLoader(
            ImageRows(splits[name], transform, raw[name], mean, scale),
            batch_size=BATCH_SIZE, shuffle=(name == "train"), num_workers=NUM_WORKERS,
        )
        for name in ("train", "val")
    }

    b0 = models.efficientnet_b0(weights=WEIGHTS)
    deep_dim = int(b0.classifier[-1].in_features)
    if deep_dim != 1280:
        print(f"Detected EfficientNet-B0 deep feature size: {deep_dim}", flush=True)
    for parameter in b0.parameters():
        parameter.requires_grad = False
    model = EfficientNetFused(b0, deep_dim).to(device)
    model.eval()
    total_dim = deep_dim + FEATURE_COUNT
    print(f"Architecture reports {deep_dim} deep features + {FEATURE_COUNT} handcrafted = {total_dim} fused.", flush=True)

    print("Extracting frozen EfficientNet features for train and validation splits...", flush=True)
    train_deep, train_hand, train_labels = extract_deep(model, loaders["train"], device)
    val_deep, val_hand, val_labels = extract_deep(model, loaders["val"], device)
    if train_deep.shape[1] != deep_dim or train_hand.shape[1] != FEATURE_COUNT:
        raise RuntimeError("Extracted feature dimensions disagree with the architecture.")

    classifier = model.classifier
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    best_val = -1.0
    for epoch in range(EPOCHS):
        classifier.train()
        order = torch.randperm(len(train_labels))
        total_loss, correct = 0.0, 0
        for start in range(0, len(order), 128):
            idx = order[start:start + 128]
            fused = torch.cat((train_deep[idx], train_hand[idx]), dim=1).to(device)
            target = train_labels[idx].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = classifier(fused)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)
            correct += (logits.argmax(1) == target).sum().item()
        y_val, p_val = evaluate_classifier(classifier, val_deep, val_hand, val_labels, device)
        val_acc = accuracy_score(y_val, p_val)
        print(f"Epoch {epoch + 1}/{EPOCHS} | train loss={total_loss / len(train_labels):.4f} "
              f"train accuracy={correct / len(train_labels):.4f} | val accuracy={val_acc:.4f}", flush=True)
        if val_acc > best_val:
            best_val = float(val_acc)
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "class_to_idx": CLASS_TO_IDX,
                "architecture": "torchvision.efficientnet_b0",
                "weights": "EfficientNet_B0_Weights.DEFAULT",
                "deep_feature_dim": deep_dim,
                "feature_count": FEATURE_COUNT,
                "fused_feature_dim": total_dim,
                "feature_image_size": FEATURE_IMAGE_SIZE,
                "feature_scaler_mean": mean,
                "feature_scaler_scale": scale,
                "epochs": EPOCHS,
                "best_validation_accuracy": best_val,
                "train_count": len(splits["train"]),
                "validation_count": len(splits["val"]),
                "test_count": len(splits["test"]),
                "preprocessing": "torchvision EfficientNet-B0 DEFAULT transforms; shared 224x224 RGB handcrafted preprocessing",
            }
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, MODEL_PATH)

    # The held-out test split is first used after model selection and checkpoint saving.
    best = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    test_loader = DataLoader(
        ImageRows(splits["test"], transform, raw["test"], mean, scale),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
    )
    test_deep, test_hand, test_labels = extract_deep(model, test_loader, device)
    y_true, y_pred = evaluate_classifier(model.classifier, test_deep, test_hand, test_labels, device)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    matrix = confusion_matrix(y_true, y_pred, labels=range(len(LABELS)))
    save_confusion_matrix(matrix)
    metrics_data = {
        "architecture": "EfficientNet-B0",
        "weights": "torchvision EfficientNet_B0_Weights.DEFAULT (ImageNet pretrained)",
        "deep_feature_dim": deep_dim,
        "handcrafted_feature_dim": FEATURE_COUNT,
        "fused_feature_dim": total_dim,
        "split_counts": {name: len(frame) for name, frame in splits.items()},
        "best_validation_accuracy": best_val,
        "scaler_fit_split": "train only",
        "test_used_for_training_or_model_selection": False,
        "preprocessing": str(transform),
        "test_metrics": metrics,
        "confusion_matrix": {"labels": LABELS, "rows_actual_columns_predicted": matrix.tolist()},
        "comparison_resnet18_fused_accuracy": 0.8066666666666666,
        "accuracy_difference_from_resnet18": metrics["accuracy"] - 0.8066666666666666,
    }
    METRICS_PATH.write_text(json.dumps(metrics_data, indent=2), encoding="utf-8")
    print("\nEfficientNet-B0 fused held-out test results (weighted precision/recall/F1)")
    print(f"Training images: {len(splits['train'])}")
    print(f"Validation images: {len(splits['val'])}")
    print(f"Held-out test images: {len(splits['test'])}")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    print(f"Existing ResNet18 fused accuracy: {0.8066666666666666:.4f}")
    print(f"Accuracy difference (EfficientNet - ResNet18): {metrics['accuracy'] - 0.8066666666666666:+.4f}")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    print(f"Saved confusion matrix: {MATRIX_PATH}")


if __name__ == "__main__":
    main()
