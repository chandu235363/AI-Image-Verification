from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pywt


def load_image_grayscale(image_path: str | Path):
    """Load an image as grayscale and return a NumPy array."""
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    return image


def compute_fft_features(image_path: str | Path):
    """Compute a simple FFT magnitude feature vector for an image."""
    image = load_image_grayscale(image_path)
    fft = np.fft.fft2(image)
    fft_shifted = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shifted)
    magnitude = np.log1p(magnitude)
    return magnitude.astype(np.float32)


def compute_haar_features(image_path: str | Path):
    """Compute Haar wavelet approximation and detail features."""
    image = load_image_grayscale(image_path)
    coeffs = pywt.dwt2(image, "haar")
    approx, details = coeffs
    horizontal, vertical, diagonal = details
    feature_map = np.concatenate([
        approx.ravel(),
        horizontal.ravel(),
        vertical.ravel(),
        diagonal.ravel(),
    ])
    return feature_map.astype(np.float32)
