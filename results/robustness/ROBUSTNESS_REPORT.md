# Robustness Evaluation Report

## 1. Objective

Measure how the existing image classifier's predictions change after deterministic cropping, resizing, JPEG compression, and combinations of those operations. Ground-truth labels are held fixed because these operations change image encoding or framing, not the source label.

## 2. Experimental setup

- Run time: 2026-10-02T08:51:07+00:00
- Checkpoint: `results\efficientnet_fused_best_model.pth` (SHA-256 `b09f53d3dd5f347ce93f0a3bdd19df5f79e631b650020d76fd55064a17e0e724`)
- Inference code: `src/13_predict_efficientnet_image.py`; the robustness runner calls its shared `predict_image_details` function with one cached predictor.
- Dataset manifest and split: `data\processed\dataset_manifest.csv`, existing `test` split only.
- Available test images: 900; evaluated sample: 300 (seed 42, class-stratified).
- Evaluated sample counts: {"Real": 100, "AI-Generated": 100, "Manipulated": 100}.
- Original test files were read only. Derived transformed images were saved under `results/robustness/transformed/` as PNGs, including the JPEG-decoded pixels for compression conditions.
- Class metrics: macro precision/recall/F1 across all three classifier classes; class-specific precision/recall/F1 are also listed below.

## 3. Transformation definitions

All operations use RGB images and are deterministic. Operations in combined conditions run in this order: crop, resize degradation, JPEG encode/decode.

- **Crop:** centered crop retaining 90% of each dimension, then bicubic resize back to the input dimensions.
- **Resize:** bicubic downsampling to 75% of each dimension, then bicubic upsampling back to input dimensions.
- **JPEG compression:** encode and decode at quality 60.
- Combined conditions apply the corresponding operations in the order above.

The model's original torchvision resize/center-crop/tensor normalization and shared 224x224 FFT, Haar wavelet, and residual feature extraction run after each condition's transformation, exactly as they do in the Flask predictor.

## 4. Dataset and test split

The project manifest identifies the sources as Real from COCO val2017, AI-Generated from Tiny-GenImage, and Manipulated from CASIA Tp. This run uses only rows marked `test`; it does not substitute uploads, training images, or another dataset. The prepared split has 900 images ({"Real": 300, "AI-Generated": 300, "Manipulated": 300}). The run sampled 300 images reproducibly to limit duplicate output files.

## 5. Model

The evaluated model is the existing three-class EfficientNet-B0 fused classifier: 1,280 deep features plus 17 handcrafted FFT/wavelet/noise features (1,297 fused inputs), with checkpoint training-fitted feature scaling. The classes are Real, AI-Generated, and Manipulated. This is not a binary Authentic/Manipulated classifier; Real is the dataset's authentic class.

## 6. Evaluation metrics

Accuracy is overall correct predictions divided by sample count. Precision, recall, and F1 in the summary are macro averages across all three classes. Prediction consistency is the fraction whose transformed predicted class equals that same image's original prediction. Confidence change is the mean signed difference in top-class confidence (transformed minus original), shown as percentage points; mean absolute confidence change is also shown. Accuracy drop is original accuracy minus condition accuracy, in percentage points; a negative value means accuracy increased.

## 7. Results

| Transformation | Accuracy | Precision | Recall | F1 | Prediction Consistency | Mean Confidence Change (pp) | Mean Absolute Confidence Change (pp) | Accuracy Drop (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Original | 0.8933 | 0.8960 | 0.8933 | 0.8935 | 1.0000 | +0.00 pp | 0.00 pp | +0.00 pp |
| Crop | 0.8700 | 0.8690 | 0.8700 | 0.8686 | 0.9100 | -0.35 pp | 5.97 pp | +2.33 pp |
| Resize | 0.8100 | 0.8197 | 0.8100 | 0.8027 | 0.8500 | -1.09 pp | 10.67 pp | +8.33 pp |
| JPEG Compression | 0.8200 | 0.8435 | 0.8200 | 0.8108 | 0.8333 | -0.93 pp | 9.76 pp | +7.33 pp |
| Crop + Resize | 0.7867 | 0.8023 | 0.7867 | 0.7738 | 0.8200 | -1.82 pp | 12.26 pp | +10.67 pp |
| Crop + Compression | 0.7900 | 0.8195 | 0.7900 | 0.7773 | 0.8033 | -1.33 pp | 11.61 pp | +10.33 pp |
| Resize + Compression | 0.7500 | 0.7991 | 0.7500 | 0.7209 | 0.7500 | -0.66 pp | 13.34 pp | +14.33 pp |
| Crop + Resize + Compression | 0.7467 | 0.7916 | 0.7467 | 0.7180 | 0.7533 | -1.64 pp | 13.98 pp | +14.67 pp |

### Per-class results

### Original
| Ground-truth class | Precision | Recall | F1 | Support | Correct |
|---|---:|---:|---:|---:|---:|
| Real | 0.8273 | 0.9100 | 0.8667 | 100 | 91 |
| AI-Generated | 0.9694 | 0.9500 | 0.9596 | 100 | 95 |
| Manipulated | 0.8913 | 0.8200 | 0.8542 | 100 | 82 |

### Crop
| Ground-truth class | Precision | Recall | F1 | Support | Correct |
|---|---:|---:|---:|---:|---:|
| Real | 0.8526 | 0.8100 | 0.8308 | 100 | 81 |
| AI-Generated | 0.8899 | 0.9700 | 0.9282 | 100 | 97 |
| Manipulated | 0.8646 | 0.8300 | 0.8469 | 100 | 83 |

### Resize
| Ground-truth class | Precision | Recall | F1 | Support | Correct |
|---|---:|---:|---:|---:|---:|
| Real | 0.8824 | 0.6000 | 0.7143 | 100 | 60 |
| AI-Generated | 0.8376 | 0.9800 | 0.9032 | 100 | 98 |
| Manipulated | 0.7391 | 0.8500 | 0.7907 | 100 | 85 |

### JPEG Compression
| Ground-truth class | Precision | Recall | F1 | Support | Correct |
|---|---:|---:|---:|---:|---:|
| Real | 0.9333 | 0.5600 | 0.7000 | 100 | 56 |
| AI-Generated | 0.8818 | 0.9700 | 0.9238 | 100 | 97 |
| Manipulated | 0.7154 | 0.9300 | 0.8087 | 100 | 93 |

### Crop + Resize
| Ground-truth class | Precision | Recall | F1 | Support | Correct |
|---|---:|---:|---:|---:|---:|
| Real | 0.8814 | 0.5200 | 0.6541 | 100 | 52 |
| AI-Generated | 0.7778 | 0.9800 | 0.8673 | 100 | 98 |
| Manipulated | 0.7478 | 0.8600 | 0.8000 | 100 | 86 |

### Crop + Compression
| Ground-truth class | Precision | Recall | F1 | Support | Correct |
|---|---:|---:|---:|---:|---:|
| Real | 0.9259 | 0.5000 | 0.6494 | 100 | 50 |
| AI-Generated | 0.8509 | 0.9700 | 0.9065 | 100 | 97 |
| Manipulated | 0.6818 | 0.9000 | 0.7759 | 100 | 90 |

### Resize + Compression
| Ground-truth class | Precision | Recall | F1 | Support | Correct |
|---|---:|---:|---:|---:|---:|
| Real | 0.9211 | 0.3500 | 0.5072 | 100 | 35 |
| AI-Generated | 0.8435 | 0.9700 | 0.9023 | 100 | 97 |
| Manipulated | 0.6327 | 0.9300 | 0.7530 | 100 | 93 |

### Crop + Resize + Compression
| Ground-truth class | Precision | Recall | F1 | Support | Correct |
|---|---:|---:|---:|---:|---:|
| Real | 0.9231 | 0.3600 | 0.5180 | 100 | 36 |
| AI-Generated | 0.8017 | 0.9700 | 0.8778 | 100 | 97 |
| Manipulated | 0.6500 | 0.9100 | 0.7583 | 100 | 91 |


## 8. Interpretation

The measured consistency and accuracy values describe this checkpoint on this test sample and the stated transformation strengths. They do not establish a guarantee for other image sources, crop locations, resize kernels, compression qualities, or manipulation methods. The strongest combined condition by observed accuracy was **Crop + Resize + Compression** (0.7467 accuracy).

## 9. Limitations

- The test sample is 300 of 900 available test images; the estimate has sampling uncertainty.
- The prepared dataset consists of three source-specific classes, and results may reflect dataset/source characteristics.
- Each transform uses one fixed deterministic strength and centered crop; real-world transformations vary.
- The labels and class mapping come from the existing project manifest. No independent relabeling or forensic review was performed.
- Metrics measure classification behavior, not proof that a transformed image itself is authentic or manipulated.

## 10. Conclusion

This experiment evaluates the existing EfficientNet-B0 fused model under the defined transformations. Use the reported measurements to describe observed performance on this split and sample; do not claim general robustness beyond this evaluation.
