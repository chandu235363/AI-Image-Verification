from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights

try:
    from src.features.fused_features import (
        FEATURE_COUNT,
        FEATURE_IMAGE_SIZE,
        extract_handcrafted_features,
        preprocess_feature_image,
    )
except ModuleNotFoundError:
    from features.fused_features import (
        FEATURE_COUNT,
        FEATURE_IMAGE_SIZE,
        extract_handcrafted_features,
        preprocess_feature_image,
    )

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "results/efficientnet_fused_best_model.pth"
LABELS = ["Real", "AI-Generated", "Manipulated"]


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
        pooled = self.backbone.avgpool(self.backbone.features(images))
        return torch.flatten(pooled, 1)

    def forward(self, images: torch.Tensor, handcrafted: torch.Tensor) -> torch.Tensor:
        return self.classifier(torch.cat((self.extract_deep_features(images), handcrafted), dim=1))


def predict_image(image_path: str | Path) -> tuple[str, float]:
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"EfficientNet fused model not found: {MODEL_PATH}. Run src/12_train_efficientnet_fused.py first.")
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture") != "torchvision.efficientnet_b0":
        raise ValueError("Checkpoint architecture is not EfficientNet-B0.")
    deep_dim = int(checkpoint.get("deep_feature_dim", 0))
    if deep_dim <= 0 or checkpoint.get("fused_feature_dim") != deep_dim + FEATURE_COUNT:
        raise ValueError("Checkpoint fused feature dimensions are inconsistent.")
    if checkpoint.get("feature_count") != FEATURE_COUNT or checkpoint.get("feature_image_size") != FEATURE_IMAGE_SIZE:
        raise ValueError("Checkpoint handcrafted feature configuration does not match this extractor.")
    if checkpoint.get("class_to_idx") != {name: idx for idx, name in enumerate(LABELS)}:
        raise ValueError("Checkpoint class mapping does not match the predictor.")
    mean = torch.as_tensor(checkpoint.get("feature_scaler_mean"), dtype=torch.float32)
    scale = torch.as_tensor(checkpoint.get("feature_scaler_scale"), dtype=torch.float32)
    if mean.shape != (FEATURE_COUNT,) or scale.shape != (FEATURE_COUNT,) or (scale <= 0).any():
        raise ValueError("Checkpoint has invalid training-fitted scaler statistics.")

    backbone = models.efficientnet_b0(weights=None)
    if int(backbone.classifier[-1].in_features) != deep_dim:
        raise ValueError("Checkpoint feature dimension disagrees with installed EfficientNet-B0.")
    model = EfficientNetFused(backbone, deep_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    transform = EfficientNet_B0_Weights.DEFAULT.transforms()
    with Image.open(path) as source:
        rgb = source.convert("RGB")
        feature_image = preprocess_feature_image(rgb)
        handcrafted = torch.from_numpy(extract_handcrafted_features(np.asarray(feature_image)))
        handcrafted = ((handcrafted - mean) / scale).unsqueeze(0)
        tensor = transform(rgb).unsqueeze(0)
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor, handcrafted), dim=1)[0]
    index = int(probabilities.argmax())
    return LABELS[index], float(probabilities[index])


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Usage: python src/13_predict_efficientnet_image.py "<image_path>"')
        raise SystemExit(1)
    label, confidence = predict_image(sys.argv[1])
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.4f}")
