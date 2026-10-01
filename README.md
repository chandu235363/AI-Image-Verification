# AI-Based Detection and Verification of Authentic, AI-Generated, and Manipulated Images

This project is a simple college-level baseline for classifying images into three classes:
- Real
- AI-generated
- Manipulated

It includes:
- dataset validation and manifest creation
- a baseline ResNet18 transfer-learning pipeline
- evaluation with accuracy, precision, recall, F1-score, and confusion matrix
- basic robustness checks
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

## Robustness testing

```powershell
python src/05_robustness_testing.py
```

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
- The FFT and wavelet analysis remains separate so it can be compared with the CNN baseline later.
- The project is structured for a college-level baseline pipeline, not for production deployment.
