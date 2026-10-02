# Image Authenticity Verification

This project evaluates whether an image is **Real**, **AI-Generated**, or **Manipulated**. It is a three-class research and demonstration system; predictions are model estimates and can be wrong.

## Final production model

Flask uses the validated candidate checkpoint:

```text
results/dataset_coverage_improved/efficientnet_fused_robust_balanced_best_model.pth
```

The previous checkpoint is retained in this project workspace at `results/efficientnet_fused_robust_best_model.pth` as a backup; it is not needed to run the final model. The production model is an EfficientNet-B0 fused classifier. It combines deep image features with 17 handcrafted features: five FFT statistics, eight Haar wavelet statistics, and four noise/residual statistics. The feature vector is standardized with the training-fitted scaler stored inside the checkpoint and concatenated with EfficientNet features before the three-class classifier.

Raw inference uses RGB conversion, square 224 by 224 LANCZOS resizing, and the same JPEG quality 95 round trip used during dataset preparation. The image transform, FFT, wavelet and residual feature extraction, scaler, and feature concatenation order are shared between training and inference.

## Measured evaluation

On the held-out evaluation set (901 images), the final candidate achieved **91.56% accuracy**, **91.59% macro F1**, and **89.04% Manipulated-class recall**. The preceding robustness-improved model achieved 90.90% accuracy, 90.86% macro F1, and 81.73% Manipulated recall on the same set.

| Transformation | Final accuracy |
|---|---:|
| Original | 91.56% |
| Crop | 92.34% |
| Resize | 91.56% |
| JPEG Compression | 91.12% |
| Crop + Resize | 91.68% |
| Crop + Compression | 91.12% |
| Resize + Compression | 90.57% |
| Crop + Resize + Compression | 91.23% |

The model was evaluated for robustness against cropping, resizing, and JPEG compression at the settings documented in `results/dataset_coverage_improved/DATASET_COVERAGE_IMPROVEMENT_REPORT.md`. These measurements describe the held-out set and tested transformations; they do not guarantee results on other images or transformation strengths.

The representative CASIA Tp preparation sampled 2,000 manipulated images from 5,123 readable files across 16 parsed filename groups. Sampling was deterministic and group-aware; train/validation/test splits were made within the groups so represented groups also occur in the test set. The three classes were kept balanced where practical. The manifest records source paths, source and prepared hashes, split, and CASIA provenance. The completed leakage audit found no cross-split duplicate paths or hashes.

The training pipeline used transformation-aware training views for the training split. Validation selected the candidate; the held-out test split was not used for training or checkpoint selection. Preparation and training scripts are retained for reproducibility, but training is not required to run Flask.

## Setup and run Flask

The final checkpoint is approximately 22 MB. It is explicitly exempted from the repository's `*.pth` ignore rule so it can be included in a repository commit. Once the final checkpoint is included in the repository version being shared, a teammate can clone and run it as follows:

```powershell
git clone https://github.com/chandu235363/AI-Image-Verification.git
cd AI-Image-Verification
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Confirm that this file exists at the exact path below:

```text
results/dataset_coverage_improved/efficientnet_fused_robust_balanced_best_model.pth
```

The checkpoint contains the model weights, class mapping, and scaler statistics; no separate scaler file is needed. If the checkpoint is absent from the repository copy, obtain this exact checkpoint from the project owner and put it at that path before starting Flask. Do not substitute the old checkpoint if the goal is to run the final model.

Start the app and open the local URL:

```powershell
python app.py
```

Open `http://127.0.0.1:5000/`, choose an image, and select **Upload Image**. The result displays the predicted class and confidence. The page also states: “Tested against: Crop • Resize • JPEG Compression”. An internal robustness request option remains available to the route, but the interface stays focused on the single-image result.

## Unknown-image and difficult-example checks

A new unknown image was uploaded through Flask. Its SHA-256 was not present in the candidate manifest. The candidate predicted Real at 66.76% confidence; its tested crop, resize, JPEG, and combined transformation variants also predicted Real. This is one inference example, not an independent performance estimate.

The candidate classified `Tp_S_NNN_S_N_nat20004_nat20004_01508.tif` as Manipulated at 60.32%, and `Tp_S_NRN_S_N_cha00037_cha00037_00357.tif` as Manipulated at 84.52%. It still classified `Tp_S_NRN_S_N_cha10207_cha10207_12340.jpg` as Real at 71.22%. Some manipulated images can still be misclassified; the model does not detect every manipulation.

## Project files

- `app.py` — Flask upload route and production checkpoint selection.
- `src/13_predict_efficientnet_image.py` — model loading and shared inference preprocessing.
- `src/features/fused_features.py` — FFT, Haar wavelet, and noise/residual feature extraction.
- `src/02_prepare_dataset.py` — deterministic dataset preparation and CASIA group sampling.
- `src/14_train_robust_efficientnet.py` — transformation-aware candidate training procedure.
- `src/18_evaluate_dataset_coverage_candidate.py` — held-out, subgroup, and robustness evaluation.
- `results/FINAL_MODEL_REPORT.md` — baseline/candidate results, audit outcome, unknown check, and limitations.

Raw training datasets are not bundled. Training requires the source datasets and is not needed for inference. To run the scripts, install the pinned dependencies from `requirements.txt` in a compatible Python environment.
