from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from torchvision import models, transforms

MODEL_PATH = Path("results/models/baseline_resnet18.pth")
CLASS_INDEX_PATH = Path("results/models/class_to_idx.json")
DATASET_MANIFEST = Path("data/processed/dataset_manifest.csv")
LABELS = ["Real", "AI-Generated", "Manipulated"]


def build_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(LABELS))
    return model


def eval_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def main():
    if not MODEL_PATH.exists() or not CLASS_INDEX_PATH.exists() or not DATASET_MANIFEST.exists():
        print("Model or dataset files are missing.")
        print("Train the model first with:")
        print("python src/03_train_baseline.py")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with CLASS_INDEX_PATH.open("r", encoding="utf-8") as file:
        class_to_idx = json.load(file)
    idx_to_class = {value: key for key, value in class_to_idx.items()}

    test_df = pd.read_csv(DATASET_MANIFEST)
    test_df = test_df[test_df["split"] == "test"].reset_index(drop=True)

    if test_df.empty:
        print("No test data found. Prepare the dataset and split it before evaluation.")
        return

    all_true = []
    all_pred = []
    transform = eval_transform()

    with torch.no_grad():
        for _, row in test_df.iterrows():
            image_path = Path(row["image_path"])
            image = Image.open(image_path).convert("RGB")
            input_tensor = transform(image).unsqueeze(0).to(device)
            output = model(input_tensor)
            _, predicted_idx = torch.max(output, 1)
            predicted_label = idx_to_class[int(predicted_idx.item())]
            true_label = row["class_name"]
            all_true.append(true_label)
            all_pred.append(predicted_label)

    acc = accuracy_score(all_true, all_pred)
    precision = precision_score(all_true, all_pred, average="weighted", zero_division=0)
    recall = recall_score(all_true, all_pred, average="weighted", zero_division=0)
    f1 = f1_score(all_true, all_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(all_true, all_pred, labels=LABELS)

    print("Evaluation report")
    print("=" * 80)
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("\nConfusion matrix:")
    print(cm)
    print("\nPer-class metrics:")
    print(classification_report(all_true, all_pred, labels=LABELS, zero_division=0))


if __name__ == "__main__":
    main()
