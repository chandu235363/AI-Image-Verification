# AI-Based Detection and Verification of Authentic, AI-Generated, and Manipulated Images

This project classifies images into three classes:
- Real
- AI-generated
- Manipulated

It includes:
- dataset validation and manifest creation
- an EfficientNet-B0 fused classifier used by the Flask app
- older ResNet18 baseline and fused training/evaluation scripts for comparison
- evaluation with accuracy, precision, recall, F1-score, and confusion matrix
- transformation robustness evaluation for the active EfficientNet-B0 classifier
- single-image prediction
- a simple Flask web demo
- FFT and Haar wavelet feature exploration as a separate module

## Folder structure

- data/raw/ - raw dataset folders expected for GenImage, CASIA, and COCO
- data/processed/ - prepared dataset manifest and intermediate files
- src/ - project scripts
- src/features/ - FFT and Haar wavelet feature utilities
- results/ - saved models and output plots
- templates/ - Flask web UI templates

## Required dataset folders

The project expects these folders to exist before training:

- data/raw/genimage
- data/raw/casia
- data/raw/coco

If they are missing, the scripts will report the missing paths instead of inventing data.

## Setup

From the project root:

```powershell
cd "C:\Users\user\OneDrive\Desktop\AI_Image_Verification"
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Data preparation

```powershell
python src/02_prepare_dataset.py
```

This checks the dataset folders, validates image files, creates labels, and splits them into train/validation/test sets.

## Training

```powershell
python src/03_train_baseline.py
```

## Evaluation

```powershell
python src/04_evaluate_model.py
```

## Robustness Evaluation

```powershell
python src/05_robustness_testing.py --sample-size 300 --seed 42
```

The evaluation reuses the Flask application's trained EfficientNet-B0 fused predictor and its full preprocessing path: torchvision ImageNet image transforms plus 17 FFT, Haar wavelet, and noise residual features scaled with training-fitted checkpoint statistics. It uses only the existing `test` split in `data/processed/dataset_manifest.csv`; it does not train a model or alter source test images. The default run uses a reproducible class-stratified sample of 300 images to limit saved duplicates. Use `--sample-size 0` to evaluate all available test rows, or choose another positive sample size. `--seed` controls sampling.

Conditions and deterministic operation settings:

- Original image
- Center crop retaining 90% of image width and height, resized back to the original dimensions with bicubic interpolation
- Resize degradation: bicubic downsample to 75% of each dimension, then bicubic upsample to the original dimensions
- JPEG encode/decode at quality 60
- Each two-operation and three-operation combination; combined order is crop, resize, JPEG

Ground-truth labels remain unchanged for every transformed version. The project manifest uses three classes: Real (the dataset's authentic class), AI-Generated, and Manipulated. Therefore, metrics cover the three-class classifier, not a binary authentic/manipulated classifier. Summary precision, recall, and F1 are macro averages; class-specific scores are also provided. The output records accuracy, confusion matrices, predicted class and all class probabilities per image, prediction consistency against the same image's original prediction, signed confidence change, and accuracy change in percentage points. A negative accuracy drop indicates an increase relative to original.

Transformed sample images, JSON metrics, prediction CSV, plots, and the generated report are saved under `results/robustness/`. The report states the exact sample and per-class counts used. Cropping can remove localized evidence or change image framing; resizing interpolates pixels and can alter high-frequency/noise patterns; JPEG compression quantizes frequency information and may remove forensic traces. This evaluation measures only the stated centered crop, resize scale, and JPEG quality, so results do not guarantee performance for other transformation strengths, sources, or manipulations.

The measured summary from `results/robustness/robustness_results.json` is:

<!-- ROBUSTNESS_RESULTS_START -->
Sample: 300 of 900 existing test images; 100 per class (seed 42). The following measured metrics are macro averages across the three classes.

| Transformation | Accuracy | Precision | Recall | F1 | Consistency | Mean confidence change (pp) | Accuracy drop (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original | 0.8933 | 0.8960 | 0.8933 | 0.8935 | 1.0000 | +0.00 pp | +0.00 pp |
| Crop | 0.8700 | 0.8690 | 0.8700 | 0.8686 | 0.9100 | -0.35 pp | +2.33 pp |
| Resize | 0.8100 | 0.8197 | 0.8100 | 0.8027 | 0.8500 | -1.09 pp | +8.33 pp |
| JPEG Compression | 0.8200 | 0.8435 | 0.8200 | 0.8108 | 0.8333 | -0.93 pp | +7.33 pp |
| Crop + Resize | 0.7867 | 0.8023 | 0.7867 | 0.7738 | 0.8200 | -1.82 pp | +10.67 pp |
| Crop + Compression | 0.7900 | 0.8195 | 0.7900 | 0.7773 | 0.8033 | -1.33 pp | +10.33 pp |
| Resize + Compression | 0.7500 | 0.7991 | 0.7500 | 0.7209 | 0.7500 | -0.66 pp | +14.33 pp |
| Crop + Resize + Compression | 0.7467 | 0.7916 | 0.7467 | 0.7180 | 0.7533 | -1.64 pp | +14.67 pp |
<!-- ROBUSTNESS_RESULTS_END -->

The Flask page also has an optional checkbox to show predictions for the uploaded image and all seven transformed versions. Those displayed confidence values are generated by the same active model at request time.

## Single-image prediction

```powershell
python src/06_predict_one_image.py "path\to\image.jpg"
```

## Flask web app

```powershell
python app.py
```

Then open:

```text
http://127.0.0.1:5000/
```

## Important notes

- Do not download large datasets automatically.
- Do not claim successful training or accuracy results without real dataset files.
- The active fused EfficientNet uses FFT, Haar wavelet, and noise residual features; separate FFT/wavelet exploration utilities are also included.
- The project is structured for a college-level baseline pipeline, not for production deployment.
