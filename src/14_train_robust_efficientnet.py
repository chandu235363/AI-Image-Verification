"""Fine-tune the existing EfficientNet-B0 fused head with train-only image variants."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import EfficientNet_B0_Weights

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.fused_features import (
    FEATURE_COUNT,
    FEATURE_IMAGE_SIZE,
    extract_handcrafted_features,
    preprocess_feature_image,
)

predictor_module = import_module("src.13_predict_efficientnet_image")
EfficientNetFused = predictor_module.EfficientNetFused

MANIFEST_PATH = ROOT / "data/processed/dataset_manifest.csv"
BASELINE_PATH = ROOT / "results/efficientnet_fused_best_model.pth"
MODEL_PATH = ROOT / "results/efficientnet_fused_robust_best_model.pth"
OUTPUT_DIR = ROOT / "results/robustness_improved"
BASELINE_BACKUP = OUTPUT_DIR / "efficientnet_fused_baseline_backup.pth"
LABELS = ["Real", "AI-Generated", "Manipulated"]
CLASS_TO_IDX = {name: index for index, name in enumerate(LABELS)}
IMAGE_TRANSFORM = EfficientNet_B0_Weights.DEFAULT.transforms()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_splits_and_check_leakage(manifest_path: Path) -> dict[str, pd.DataFrame]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Dataset manifest not found: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    required = {"filepath", "label", "class_name", "split"}
    if required - set(manifest.columns):
        raise ValueError(f"Manifest missing columns: {sorted(required - set(manifest.columns))}")
    if set(manifest["class_name"]) != set(LABELS):
        raise ValueError(f"Expected classes {LABELS}; found {sorted(manifest['class_name'].unique())}")
    if set(manifest["split"]) != {"train", "val", "test"}:
        raise ValueError("Manifest must contain the existing train, val, and test splits.")
    for row in manifest.itertuples(index=False):
        if row.class_name not in CLASS_TO_IDX or int(row.label) != CLASS_TO_IDX[row.class_name]:
            raise ValueError(f"Invalid manifest class/label mapping: {row.class_name}, {row.label}")
        if not Path(row.filepath).is_file():
            raise FileNotFoundError(f"Manifest image is missing: {row.filepath}")

    splits = {
        name: manifest[manifest["split"] == name].copy().reset_index(drop=True)
        for name in ("train", "val", "test")
    }
    for name, rows in splits.items():
        if rows.empty:
            raise ValueError(f"Split {name} is empty.")
    for name, rows in splits.items():
        counts = rows["class_name"].value_counts().reindex(LABELS, fill_value=0)
        if (counts == 0).any():
            raise ValueError(f"Every class must occur in {name}; counts={counts.to_dict()}")

    split_paths = {name: {Path(value).resolve() for value in rows["filepath"]} for name, rows in splits.items()}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if split_paths[left] & split_paths[right]:
            raise ValueError(f"Manifest path overlap between {left} and {right}.")

    # Audit source and prepared paths and hashes when the provenance manifest provides them.
    owner: dict[tuple[str, str], str] = {}
    total = sum(len(frame) for frame in splits.values())
    processed = 0
    for split_name in ("train", "val", "test"):
        for row in splits[split_name].itertuples(index=False):
            path = Path(row.filepath)
            values = [("prepared_sha256", sha256_file(path))]
            if hasattr(row, "source_sha256") and pd.notna(row.source_sha256):
                values.append(("source_sha256", str(row.source_sha256)))
            if hasattr(row, "source_path") and pd.notna(row.source_path):
                source_path = Path(row.source_path)
                values.append(("source_path", str(source_path.resolve()).casefold()))
            for kind, digest in values:
                key = (kind, digest)
                previous = owner.get(key)
                if previous is not None and previous != split_name:
                    raise ValueError(f"Cross-split {kind} leakage across {previous}/{split_name}: {path}")
                owner[key] = split_name
            processed += 1
            if processed % 1000 == 0 or processed == total:
                print(f"  Leakage audit: hashed {processed}/{total} split images", flush=True)
    return splits


def augment_training_image(image: Image.Image, config: argparse.Namespace) -> tuple[Image.Image, list[str]]:
    """Apply independent, mild train-only operations before all model preprocessing."""
    result = image.convert("RGB")
    width, height = result.size
    operations = []

    if random.random() < config.crop_probability:
        retained_w = random.uniform(config.crop_scale_min, 1.0)
        retained_h = random.uniform(config.crop_scale_min, 1.0)
        crop_w = max(1, round(width * retained_w))
        crop_h = max(1, round(height * retained_h))
        left = random.randint(0, width - crop_w)
        top = random.randint(0, height - crop_h)
        result = result.crop((left, top, left + crop_w, top + crop_h))
        result = result.resize((width, height), Image.Resampling.BICUBIC)
        operations.append("crop")

    if random.random() < config.resize_probability:
        scale = random.uniform(config.resize_scale_min, config.resize_scale_max)
        intermediate = (max(1, round(width * scale)), max(1, round(height * scale)))
        result = result.resize(intermediate, Image.Resampling.BICUBIC)
        result = result.resize((width, height), Image.Resampling.BICUBIC)
        operations.append("resize")

    if random.random() < config.jpeg_probability:
        import io

        buffer = io.BytesIO()
        quality = random.choice(config.jpeg_qualities)
        result.save(buffer, format="JPEG", quality=quality, optimize=False, progressive=False)
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            result = decoded.convert("RGB")
        operations.append(f"jpeg_q{quality}")

    return result, operations


class ImageFeatureRows(Dataset):
    def __init__(self, rows: pd.DataFrame, feature_mean: torch.Tensor, feature_scale: torch.Tensor,
                 augmentation: argparse.Namespace | None = None):
        self.rows = rows.reset_index(drop=True)
        self.feature_mean = feature_mean
        self.feature_scale = feature_scale
        self.augmentation = augmentation
        self.operations: Counter[str] = Counter()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        with Image.open(row.filepath) as source:
            image = source.convert("RGB")
        if self.augmentation is not None:
            image, operations = augment_training_image(image, self.augmentation)
            self.operations["augmented_views"] += 1
            if not operations:
                self.operations["unmodified_aug_view"] += 1
            for operation in operations:
                self.operations[operation] += 1
            if len(operations) >= 2:
                self.operations["multi_operation_views"] += 1

        feature_image = preprocess_feature_image(image)
        raw_features = torch.from_numpy(extract_handcrafted_features(np.asarray(feature_image)))
        scaled_features = (raw_features - self.feature_mean) / self.feature_scale
        return IMAGE_TRANSFORM(image), scaled_features, int(row.label)


def extract_features(model: nn.Module, loader: DataLoader, device: torch.device):
    deep_values, hand_values, targets = [], [], []
    model.eval()
    with torch.inference_mode():
        for images, handcrafted, labels in loader:
            deep_values.append(model.extract_deep_features(images.to(device)).cpu())
            hand_values.append(handcrafted.cpu())
            targets.append(labels.cpu())
    return torch.cat(deep_values), torch.cat(hand_values), torch.cat(targets)


def classifier_metrics(classifier: nn.Module, deep: torch.Tensor, handcrafted: torch.Tensor,
                       labels: torch.Tensor, batch_size: int, device: torch.device) -> dict[str, float]:
    classifier.eval()
    total_loss = 0.0
    correct = 0
    criterion = nn.CrossEntropyLoss()
    with torch.inference_mode():
        for start in range(0, len(labels), batch_size):
            end = start + batch_size
            fused = torch.cat((deep[start:end], handcrafted[start:end]), dim=1).to(device)
            target = labels[start:end].to(device)
            logits = classifier(fused)
            total_loss += criterion(logits, target).item() * len(target)
            correct += int((logits.argmax(1) == target).sum())
    return {"loss": total_loss / len(labels), "accuracy": correct / len(labels)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-path", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--initial-checkpoint", type=Path, default=BASELINE_PATH)
    parser.add_argument("--output-checkpoint", type=Path, default=MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=2, help="Keep the current training schedule by default.")
    parser.add_argument("--patience", type=int, default=2, help="Stop after this many epochs without val improvement.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--feature-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--crop-probability", type=float, default=0.45)
    parser.add_argument("--resize-probability", type=float, default=0.45)
    parser.add_argument("--jpeg-probability", type=float, default=0.45)
    parser.add_argument("--crop-scale-min", type=float, default=0.85,
                        help="Random crop retains at least this fraction of each dimension.")
    parser.add_argument("--resize-scale-min", type=float, default=0.70)
    parser.add_argument("--resize-scale-max", type=float, default=0.95)
    parser.add_argument("--jpeg-qualities", type=int, nargs="+", default=[60, 70, 80, 90])
    args = parser.parse_args()
    manifest_path = args.manifest_path if args.manifest_path.is_absolute() else ROOT / args.manifest_path
    initial_path = args.initial_checkpoint if args.initial_checkpoint.is_absolute() else ROOT / args.initial_checkpoint
    model_path = args.output_checkpoint if args.output_checkpoint.is_absolute() else ROOT / args.output_checkpoint
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    if not initial_path.is_file():
        raise FileNotFoundError(f"Initial checkpoint not found: {initial_path}")
    if model_path.exists():
        raise FileExistsError(f"Refusing to overwrite candidate checkpoint: {model_path}")
    if model_path.resolve() == initial_path.resolve():
        raise RuntimeError("Candidate output must be different from the initialization checkpoint.")
    for name in ("crop_probability", "resize_probability", "jpeg_probability"):
        if not 0 <= getattr(args, name) <= 1:
            parser.error(f"--{name.replace('_', '-')} must be in [0, 1].")
    if not 0 < args.crop_scale_min <= 1:
        parser.error("--crop-scale-min must be in (0, 1].")
    if not 0 < args.resize_scale_min <= args.resize_scale_max <= 1:
        parser.error("Resize scales must satisfy 0 < min <= max <= 1.")
    if not args.jpeg_qualities or any(q < 40 or q > 95 for q in args.jpeg_qualities):
        parser.error("JPEG quality choices must be between 40 and 95.")
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1 or args.feature_batch_size < 1:
        parser.error("Epoch, patience, and batch-size values must be positive.")
    if initial_path.resolve() == BASELINE_PATH.resolve():
        if not BASELINE_BACKUP.is_file() or sha256_file(BASELINE_BACKUP) != sha256_file(BASELINE_PATH):
            raise ValueError("Baseline backup is missing or differs from the original checkpoint.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = load_splits_and_check_leakage(manifest_path)
    counts = {name: splits[name]["class_name"].value_counts().reindex(LABELS).astype(int).to_dict()
              for name in ("train", "val", "test")}
    print(f"Train class counts: {counts['train']}")
    print(f"Validation class counts: {counts['val']}")
    print(f"Test class counts (kept unseen): {counts['test']}")

    # Start from the exact existing checkpoint, including its training-fitted feature scaler.
    predictor = predictor_module.load_predictor(initial_path)
    baseline_sha = sha256_file(initial_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = predictor["model"].to(device)
    mean, scale = predictor["mean"], predictor["scale"]
    if mean.shape != (FEATURE_COUNT,) or scale.shape != (FEATURE_COUNT,):
        raise ValueError("Baseline scaler feature dimensions do not match the feature extractor.")
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False
    model.eval()
    print(f"Initialized from baseline SHA-256 {baseline_sha}; device={device}")
    print(f"Architecture: EfficientNet-B0; {model.deep_feature_dim} deep + {FEATURE_COUNT} handcrafted features.")

    feature_loaders = {}
    for split in ("train", "val"):
        dataset = ImageFeatureRows(splits[split], mean, scale)
        feature_loaders[split] = DataLoader(dataset, batch_size=args.feature_batch_size,
                                            shuffle=False, num_workers=0)
    augmented_dataset = ImageFeatureRows(splits["train"], mean, scale, augmentation=args)
    augmented_loader = DataLoader(augmented_dataset, batch_size=args.feature_batch_size,
                                 shuffle=False, num_workers=0)

    print("Extracting original train and untouched validation features...", flush=True)
    original_deep, original_hand, train_labels = extract_features(model, feature_loaders["train"], device)
    val_deep, val_hand, val_labels = extract_features(model, feature_loaders["val"], device)
    print("Generating one random augmented view per TRAIN image and extracting its existing features...", flush=True)
    aug_deep, aug_hand, aug_labels = extract_features(model, augmented_loader, device)
    if not torch.equal(train_labels, aug_labels):
        raise RuntimeError("Augmented-view labels or row order differ from the original training rows.")
    original_train_val = classifier_metrics(model.classifier, val_deep, val_hand, val_labels,
                                           args.batch_size, device)
    train_deep = torch.cat((original_deep, aug_deep))
    train_hand = torch.cat((original_hand, aug_hand))
    combined_labels = torch.cat((train_labels, aug_labels))

    class_counts = torch.bincount(combined_labels, minlength=len(LABELS)).float()
    class_weights = class_counts.sum() / (len(LABELS) * class_counts)
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    best_val_accuracy = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.classifier.train()
        order = torch.randperm(len(combined_labels))
        total_loss = 0.0
        total_correct = 0
        for start in range(0, len(order), args.batch_size):
            indices = order[start:start + args.batch_size]
            fused = torch.cat((train_deep[indices], train_hand[indices]), dim=1).to(device)
            target = combined_labels[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model.classifier(fused)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(indices)
            total_correct += int((logits.argmax(1) == target).sum())

        val_metrics = classifier_metrics(model.classifier, val_deep, val_hand, val_labels,
                                          args.batch_size, device)
        train_metrics = {"loss": total_loss / len(combined_labels),
                         "accuracy": total_correct / len(combined_labels)}
        record = {"epoch": epoch, "train": train_metrics, "validation": val_metrics}
        history.append(record)
        print(f"Epoch {epoch}/{args.epochs} | train loss={train_metrics['loss']:.4f} "
              f"accuracy={train_metrics['accuracy']:.4f} | validation loss={val_metrics['loss']:.4f} "
              f"accuracy={val_metrics['accuracy']:.4f}", flush=True)

        improved = (val_metrics["accuracy"] > best_val_accuracy or
                    (val_metrics["accuracy"] == best_val_accuracy and val_metrics["loss"] < best_val_loss))
        if improved:
            best_val_accuracy = val_metrics["accuracy"]
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            stale_epochs = 0
            checkpoint = {
                "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "class_to_idx": CLASS_TO_IDX,
                "architecture": "torchvision.efficientnet_b0",
                "weights": "initialized from existing EfficientNet_B0_Weights.DEFAULT checkpoint",
                "deep_feature_dim": model.deep_feature_dim,
                "feature_count": FEATURE_COUNT,
                "fused_feature_dim": model.deep_feature_dim + FEATURE_COUNT,
                "feature_image_size": FEATURE_IMAGE_SIZE,
                "feature_scaler_mean": mean.detach().cpu(),
                "feature_scaler_scale": scale.detach().cpu(),
                "epochs": epoch,
                "best_validation_accuracy": best_val_accuracy,
                "best_validation_loss": best_val_loss,
                "train_count": len(splits["train"]),
                "augmented_train_views": len(splits["train"]),
                "validation_count": len(splits["val"]),
                "test_count": len(splits["test"]),
                "test_used_for_training_or_model_selection": False,
                "scaler_source": "reused unchanged from baseline checkpoint, fit on original train split",
                "initial_baseline_validation": original_train_val,
                "baseline_checkpoint_sha256": baseline_sha,
                "augmentation": {
                    "seed": args.seed,
                    "independent_probabilities": {
                        "crop": args.crop_probability, "resize": args.resize_probability,
                        "jpeg": args.jpeg_probability,
                    },
                    "crop_scale_min": args.crop_scale_min,
                    "resize_scale_range": [args.resize_scale_min, args.resize_scale_max],
                    "jpeg_quality_choices": args.jpeg_qualities,
                    "train_augmentation_operation_counts": dict(augmented_dataset.operations),
                    "original_views_retained": True,
                },
                "preprocessing": "baseline EfficientNet-B0 DEFAULT image transform; shared 224x224 RGB FFT/Haar/noise preprocessing",
            }
            torch.save(checkpoint, model_path)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping after {stale_epochs} epochs without validation improvement.", flush=True)
                break

    if best_epoch == 0 or not model_path.is_file():
        raise RuntimeError("Training did not save a validation-selected robustness checkpoint.")
    best = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    model.eval()
    training_record = {
        "started_or_completed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "initial_checkpoint": str(initial_path.relative_to(ROOT)),
        "baseline_checkpoint_sha256": baseline_sha,
        "new_checkpoint": str(model_path.relative_to(ROOT)),
        "architecture": best["architecture"],
        "class_counts_by_split": counts,
        "data_leakage_check": "manifest paths, source hashes when supplied, and prepared image hashes checked across train/val/test",
        "validation_original_baseline": original_train_val,
        "selected_epoch": best_epoch,
        "selected_validation_accuracy": best_val_accuracy,
        "test_rows_used_for_training_or_selection": 0,
        "training_views": {"original": len(splits["train"]), "augmented": len(splits["train"])},
        "augmentation_config": best["augmentation"],
        "history": history,
        "training_command_parameters": {key: (str(value) if isinstance(value, Path) else value)
                                        for key, value in vars(args).items()},
    }
    (output_dir / "training_metrics.json").write_text(json.dumps(training_record, indent=2) + "\n", encoding="utf-8")
    print(f"Saved robustness checkpoint: {model_path}")
    print(f"Best validation accuracy: {best_val_accuracy:.4f} at epoch {best_epoch}")
    print(f"Saved training record: {output_dir / 'training_metrics.json'}")
    print("The test split was not passed through the model during training or model selection.")


if __name__ == "__main__":
    main()
