from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

MODEL_PATH = Path("results/best_model.pth")

LABELS = ["Real", "AI-Generated", "Manipulated"]


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 3)

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def predict_image(image_path: str | Path):
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    model = load_model()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted_index = torch.max(
            probabilities, dim=1
        )

    predicted_label = LABELS[int(predicted_index.item())]

    return predicted_label, float(confidence.item())


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python src/06_predict_one_image.py <image_path>")
        raise SystemExit(1)

    prediction, confidence = predict_image(sys.argv[1])

    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence:.4f}")
    print(f"Confidence: {confidence * 100:.2f}%")