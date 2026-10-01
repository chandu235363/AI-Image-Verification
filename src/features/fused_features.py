from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pywt
from PIL import Image


FEATURE_COUNT = 17
FEATURE_IMAGE_SIZE = 224


def preprocess_feature_image(image: Image.Image) -> Image.Image:
    """Apply the shared RGB 224x224 preprocessing used before handcrafted features."""
    image = image.convert("RGB")
    if image.size != (FEATURE_IMAGE_SIZE, FEATURE_IMAGE_SIZE):
        image = image.resize((FEATURE_IMAGE_SIZE, FEATURE_IMAGE_SIZE), Image.Resampling.BILINEAR)
    return image


def extract_handcrafted_features(image: np.ndarray) -> np.ndarray:
    """Return 5 FFT, 8 Haar-wavelet, and 4 residual statistics for a gray image."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    gray = gray.astype(np.float32)

    spectrum = np.fft.fftshift(np.fft.fft2(gray))
    log_magnitude = np.log1p(np.abs(spectrum)).astype(np.float32)
    height, width = gray.shape
    yy, xx = np.ogrid[:height, :width]
    radius = np.sqrt((yy - height // 2) ** 2 + (xx - width // 2) ** 2)
    low_mask = radius <= min(height, width) * 0.10
    high_mask = radius >= min(height, width) * 0.25
    low_energy = float(np.mean(np.square(log_magnitude[low_mask])))
    high_energy = float(np.mean(np.square(log_magnitude[high_mask])))
    fft_features = [
        float(log_magnitude.mean()), float(log_magnitude.std()), low_energy,
        high_energy, high_energy / (low_energy + 1e-8),
    ]

    approximation, (horizontal, vertical, diagonal) = pywt.dwt2(gray, "haar")
    wavelet_features = []
    for coefficients in (approximation, horizontal, vertical, diagonal):
        wavelet_features.extend((float(coefficients.mean()), float(coefficients.std())))

    residual = gray - cv2.GaussianBlur(gray, (0, 0), sigmaX=1.5)
    noise_features = [
        float(residual.mean()), float(residual.std()),
        float(np.abs(residual).mean()), float(np.mean(np.square(residual))),
    ]
    features = np.asarray(fft_features + wavelet_features + noise_features, dtype=np.float32)
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def extract_features_from_path(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return extract_handcrafted_features(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
