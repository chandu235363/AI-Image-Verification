from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_MANIFEST = PROJECT_ROOT / "data/processed/dataset_manifest.csv"
MODEL_PATH = PROJECT_ROOT / "results/best_model.pth"
METRICS_PATH = PROJECT_ROOT / "results/metrics.json"
CONFUSION_MATRIX_PATH = PROJECT_ROOT / "results/confusion_matrix.png"
LABELS = ["Real", "AI-Generated", "Manipulated"]
CLASS_TO_IDX = {label: index for index, label in enumerate(LABELS)}
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5


class ImageDataset(Dataset):
    def __init__(self, manifest_df: pd.DataFrame, transform):
        self.manifest = manifest_df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, index):
        row = self.manifest.iloc[index]
        image = Image.open(row["filepath"]).convert("RGB")
        return self.transform(image), int(row["label"])


def build_model() -> tuple[nn.Module, bool]:
    """Build ResNet18 with ImageNet weights when available."""
    pretrained = True
    try:
        model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    except Exception as error:
        print(f"ImageNet weights unavailable ({error}). Using uninitialized weights.")
        model = models.resnet18(weights=None)
        pretrained = False

    for parameter in model.parameters():
        parameter.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, len(LABELS))
    return model, pretrained


def make_transforms():
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    evaluation_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    return train_transform, evaluation_transform


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_items = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            if training:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total_items += labels.size(0)

    return total_loss / total_items, total_correct / total_items


def evaluate_test_set(model, loader, device):
    model.eval()
    true_labels = []
    predicted_labels = []
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(device))
            true_labels.extend(labels.tolist())
            predicted_labels.extend(outputs.argmax(dim=1).cpu().tolist())

    metrics = {
        "accuracy": accuracy_score(true_labels, predicted_labels),
        "precision": precision_score(true_labels, predicted_labels, average="weighted", zero_division=0),
        "recall": recall_score(true_labels, predicted_labels, average="weighted", zero_division=0),
        "f1_score": f1_score(true_labels, predicted_labels, average="weighted", zero_division=0),
    }
    matrix = confusion_matrix(true_labels, predicted_labels, labels=list(range(len(LABELS))))
    return metrics, matrix


def save_confusion_matrix(matrix):
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=range(len(LABELS)),
        yticks=range(len(LABELS)),
        xticklabels=LABELS,
        yticklabels=LABELS,
        xlabel="Predicted label",
        ylabel="True label",
        title="Baseline ResNet18 Confusion Matrix",
    )
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            axis.text(column_index, row_index, matrix[row_index, column_index], ha="center", va="center")
    figure.tight_layout()
    CONFUSION_MATRIX_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(CONFUSION_MATRIX_PATH, dpi=150)
    plt.close(figure)


def main():
    if not DATASET_MANIFEST.exists():
        raise FileNotFoundError(f"Dataset manifest not found: {DATASET_MANIFEST}")

    manifest = pd.read_csv(DATASET_MANIFEST)
    required_columns = {"filepath", "label", "class_name", "split"}
    missing_columns = required_columns - set(manifest.columns)
    if missing_columns:
        raise ValueError(f"Manifest is missing columns: {sorted(missing_columns)}")

    train_df = manifest[manifest["split"] == "train"].reset_index(drop=True)
    validation_df = manifest[manifest["split"] == "val"].reset_index(drop=True)
    test_df = manifest[manifest["split"] == "test"].reset_index(drop=True)
    if train_df.empty or validation_df.empty or test_df.empty:
        raise ValueError("Train, validation, and test splits must all contain images.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training baseline ResNet18 classifier for 3 classes.")
    print(f"Using device: {device}")
    print(f"Images: train={len(train_df)}, val={len(validation_df)}, test={len(test_df)}")

    train_transform, evaluation_transform = make_transforms()
    train_loader = DataLoader(ImageDataset(train_df, train_transform), batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(ImageDataset(validation_df, evaluation_transform), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(ImageDataset(test_df, evaluation_transform), batch_size=BATCH_SIZE, shuffle=False)

    model, pretrained = build_model()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.fc.parameters(), lr=1e-3, weight_decay=1e-4)
    best_validation_accuracy = -1.0

    for epoch in range(EPOCHS):
        train_loss, train_accuracy = run_epoch(model, train_loader, criterion, device, optimizer)
        validation_loss, validation_accuracy = run_epoch(model, validation_loader, criterion, device)
        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train loss: {train_loss:.4f} | Train acc: {train_accuracy:.4f} | "
            f"Val loss: {validation_loss:.4f} | Val acc: {validation_accuracy:.4f}"
        )

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_to_idx": CLASS_TO_IDX,
                "image_size": IMAGE_SIZE,
                "pretrained": pretrained,
            }, MODEL_PATH)

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics, matrix = evaluate_test_set(model, test_loader, device)
    save_confusion_matrix(matrix)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with METRICS_PATH.open("w", encoding="utf-8") as metrics_file:
        json.dump({
            "classes": CLASS_TO_IDX,
            "epochs": EPOCHS,
            "pretrained_imagenet_weights": pretrained,
            "best_validation_accuracy": best_validation_accuracy,
            "test_metrics": test_metrics,
        }, metrics_file, indent=2)

    print("\nFinal test metrics (test set evaluated once after training)")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    print(f"F1-score: {test_metrics['f1_score']:.4f}")
    print(f"Best model: {MODEL_PATH.resolve()}")
    print(f"Metrics: {METRICS_PATH.resolve()}")
    print(f"Confusion matrix: {CONFUSION_MATRIX_PATH.resolve()}")


if __name__ == "__main__":
    main()
