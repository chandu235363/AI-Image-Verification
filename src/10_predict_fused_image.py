from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

try:
    from src.features.fused_features import (
        FEATURE_COUNT,
        FEATURE_IMAGE_SIZE,
        extract_handcrafted_features,
        preprocess_feature_image,
    )
except ModuleNotFoundError:
    # Direct execution adds src/ to sys.path instead of the project root.
    from features.fused_features import (
        FEATURE_COUNT,
        FEATURE_IMAGE_SIZE,
        extract_handcrafted_features,
        preprocess_feature_image,
    )

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "results/fused_best_model.pth"
LABELS = ["Real", "AI-Generated", "Manipulated"]


class FusedResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(512 + FEATURE_COUNT, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.25), nn.Linear(128, 3),
        )

    def forward(self, image, handcrafted):
        image_features = self.backbone(image)
        fused = torch.cat((image_features, handcrafted), dim=1)
        return self.classifier(fused)


def predict_image(image_path: str | Path):
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Fused model not found: {MODEL_PATH}. Run src/09_train_fused_model.py first.")
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    if checkpoint.get("feature_count") != FEATURE_COUNT:
        raise ValueError("Checkpoint feature count does not match the fused feature extractor.")
    scaler_mean = torch.as_tensor(checkpoint.get("feature_scaler_mean"), dtype=torch.float32)
    scaler_scale = torch.as_tensor(checkpoint.get("feature_scaler_scale"), dtype=torch.float32)
    if scaler_mean.shape != (FEATURE_COUNT,) or scaler_scale.shape != (FEATURE_COUNT,):
        raise ValueError("Checkpoint is missing valid training-derived handcrafted feature scaler statistics.")
    if checkpoint.get("feature_image_size") != FEATURE_IMAGE_SIZE:
        raise ValueError("Checkpoint feature preprocessing size does not match the extractor.")
    model = FusedResNet18()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((int(checkpoint.get("image_size", 112)), int(checkpoint.get("image_size", 112)))), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    with Image.open(path) as source:
        image = source.convert("RGB")
        feature_image = preprocess_feature_image(image)
        handcrafted = torch.from_numpy(extract_handcrafted_features(np.asarray(feature_image)))
        handcrafted = ((handcrafted - scaler_mean) / scaler_scale).unsqueeze(0)
        tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor, handcrafted), dim=1)[0]
    index = int(probabilities.argmax())
    return LABELS[index], float(probabilities[index])


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Usage: python src/10_predict_fused_image.py "<image_path>"')
        raise SystemExit(1)
    prediction, confidence = predict_image(sys.argv[1])
    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence:.4f}")
