
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt

# Create a sample grayscale image
image = np.zeros((256, 256), dtype=np.uint8)

cv2.rectangle(image, (50, 50), (200, 200), 255, -1)
cv2.circle(image, (128, 128), 40, 100, -1)

# FFT Analysis
fft = np.fft.fft2(image)
fft_shifted = np.fft.fftshift(fft)

magnitude_spectrum = 20 * np.log(
    np.abs(fft_shifted) + 1
)

# Wavelet Analysis
approximation, (horizontal, vertical, diagonal) = pywt.dwt2(
    image, "haar"
)

# Display results
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(magnitude_spectrum, cmap="gray")
plt.title("FFT Spectrum")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(approximation, cmap="gray")
plt.title("Wavelet Approximation")
plt.axis("off")


plt.subplot(2, 3, 4)
plt.imshow(horizontal, cmap="gray")
plt.title("Horizontal Details")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(vertical, cmap="gray")
plt.title("Vertical Details")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.imshow(diagonal, cmap="gray")
plt.title("Diagonal Details")
plt.axis("off")

plt.tight_layout()

plt.savefig(
    "results/frequency_wavelet_test.png",
    dpi=150
)

plt.show()

print("FFT and Wavelet analysis completed!")
print("Result saved successfully!")