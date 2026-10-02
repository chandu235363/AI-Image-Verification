# Dataset Coverage Improvement Report

## Objective and changes

The previous preparation selected the first 2,000 sorted CASIA Tp paths, leaving most Tp_S subgroups out. The new fixed-seed preparation samples CASIA proportionally by the parsed filename group, with up to a three-image floor per available group, and randomly samples the other classes. The three supplied diagnostic images were passed to the preparation command as exclusions and were not used in any split. No source image was modified. A separate candidate was fine-tuned from the preserved robustness-improved EfficientNet-B0 checkpoint.

## Dataset and leakage audit

- Manifest: `results\dataset_coverage_improved\new_manifest.csv`; seed 20261002; 2000 images per class.
- Counts by split: `{"train": 4198, "val": 901, "test": 901}`.
- Class counts: `{"train": {"AI-Generated": 1400, "Manipulated": 1398, "Real": 1400}, "val": {"AI-Generated": 300, "Manipulated": 301, "Real": 300}, "test": {"AI-Generated": 300, "Manipulated": 301, "Real": 300}}`.
- Exact source paths, source SHA-256, and prepared-image SHA-256 were checked across splits: **PASS**.
- CASIA group counts by split: `{"test": {"Tp_D_CND": 1, "Tp_D_CNN": 4, "Tp_D_CRD": 2, "Tp_D_CRN": 15, "Tp_D_NND": 2, "Tp_D_NNN": 22, "Tp_D_NRD": 5, "Tp_D_NRN": 59, "Tp_S_CND": 2, "Tp_S_CNN": 30, "Tp_S_CRD": 3, "Tp_S_CRN": 16, "Tp_S_NND": 10, "Tp_S_NNN": 85, "Tp_S_NRD": 8, "Tp_S_NRN": 37}, "train": {"Tp_D_CND": 2, "Tp_D_CNN": 17, "Tp_D_CRD": 9, "Tp_D_CRN": 68, "Tp_D_NND": 9, "Tp_D_NNN": 104, "Tp_D_NRD": 22, "Tp_D_NRN": 275, "Tp_S_CND": 12, "Tp_S_CNN": 138, "Tp_S_CRD": 17, "Tp_S_CRN": 78, "Tp_S_NND": 46, "Tp_S_NNN": 394, "Tp_S_NRD": 37, "Tp_S_NRN": 170}, "val": {"Tp_D_CND": 1, "Tp_D_CNN": 4, "Tp_D_CRD": 2, "Tp_D_CRN": 15, "Tp_D_NND": 2, "Tp_D_NNN": 22, "Tp_D_NRD": 5, "Tp_D_NRN": 59, "Tp_S_CND": 2, "Tp_S_CNN": 30, "Tp_S_CRD": 3, "Tp_S_CRN": 16, "Tp_S_NND": 10, "Tp_S_NNN": 85, "Tp_S_NRD": 8, "Tp_S_NRN": 37}}`.
- Groups present only in train: `[]`. All selected canonical groups are represented in test.

## Models and protocol

- Current checkpoint: `results\efficientnet_fused_robust_best_model.pth` (SHA-256 `041ecdf6b8ac9e585cf7f712147f5e89c5e908f9593097f9efc806215c4a20db`).
- Candidate checkpoint: `results\dataset_coverage_improved\efficientnet_fused_robust_balanced_best_model.pth` (SHA-256 `45944a9f29e5a8f78b4ac79b1f966441f822fb909f0451c6ed63fd5274d78b5a`).
- Both preserve EfficientNet-B0, FFT, Haar wavelet, residual/noise features, feature scaling, fusion order, and the three-class output.
- Training used original train rows plus one probabilistically augmented view per training image. Validation was untouched and selected the checkpoint; the test split was not used to update weights or select the epoch.
- Candidate checkpoint selected at epoch 1 with validation accuracy 0.9512; train views: 4198 original + 4198 augmented.
- Training augmentation settings: independent crop/resize/JPEG probabilities {'crop': 0.45, 'resize': 0.45, 'jpeg': 0.45}; crop retains at least 85% of each dimension; resize scale 70%–95%; JPEG qualities [60, 70, 80, 90].
- Raw inference and transformed raw inference use the same RGB square 224 LANCZOS resize and JPEG quality-95 decode as dataset preparation, followed by the unchanged torchvision resize/center-crop/normalization and 17 handcrafted features.
- Test images were processed from source paths through that shared preprocessing. Eight conditions use the project's existing deterministic centered 90% crop, 75% down/up resize, JPEG quality 60, and combinations in crop/resize/JPEG order. The ground-truth label is unchanged.

Training command:

```text
python src/14_train_robust_efficientnet.py --manifest-path results/dataset_coverage_improved/new_manifest.csv --initial-checkpoint results/efficientnet_fused_robust_best_model.pth --output-checkpoint results/dataset_coverage_improved/efficientnet_fused_robust_balanced_best_model.pth --output-dir results/dataset_coverage_improved --epochs 2 --patience 2 --seed 20261002
```

## Original-condition comparison

| Metric | Current | Candidate |
|---|---:|---:|
| Accuracy | 0.9090 | 0.9156 |
| Macro precision | 0.9125 | 0.9162 |
| Macro recall | 0.9091 | 0.9157 |
| Macro F1 | 0.9086 | 0.9159 |
| Manipulated recall | 0.8173 | 0.8904 |
| Manipulated → Real | 48 | 28 |
| Manipulated → AI-Generated | 7 | 5 |
| Real → Manipulated | 16 | 29 |

Confusion matrices (rows = actual; columns = predicted; order Real, AI-Generated, Manipulated):

- Current: `[[281, 3, 16], [6, 292, 2], [48, 7, 246]]`
- Candidate: `[[269, 2, 29], [8, 288, 4], [28, 5, 268]]`

## Eight-condition comparison

Accuracy, manipulated recall, macro F1, consistency versus each model's own original prediction, and accuracy drop are measured on the complete held-out test split.

| Condition | Current accuracy | Candidate accuracy | Difference | Current manipulated recall | Candidate manipulated recall | Current macro F1 | Candidate macro F1 | Current/candidate consistency | Current/candidate accuracy drop (pp) | Current M→Real/M→AI | Candidate M→Real/M→AI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original | 0.9090 | 0.9156 | +0.67 pp | 0.8173 | 0.8904 | 0.9086 | 0.9159 | 1.0000 | 1.0000 | +0.00 | +0.00 | 48/7 | 28/5 |
| Crop | 0.9023 | 0.9234 | +2.11 pp | 0.8073 | 0.9037 | 0.9020 | 0.9235 | 0.9700 | 0.9700 | +0.67 | -0.78 | 49/9 | 24/5 |
| Resize | 0.9034 | 0.9156 | +1.22 pp | 0.8007 | 0.8870 | 0.9026 | 0.9157 | 0.9878 | 0.9922 | +0.55 | +0.00 | 49/11 | 28/6 |
| JPEG Compression | 0.9046 | 0.9112 | +0.67 pp | 0.8306 | 0.9070 | 0.9041 | 0.9113 | 0.9756 | 0.9745 | +0.44 | +0.44 | 42/9 | 24/4 |
| Crop + Resize | 0.9012 | 0.9168 | +1.55 pp | 0.7907 | 0.8904 | 0.9003 | 0.9168 | 0.9656 | 0.9623 | +0.78 | -0.11 | 51/12 | 27/6 |
| Crop + Compression | 0.9001 | 0.9112 | +1.11 pp | 0.8206 | 0.9136 | 0.8996 | 0.9113 | 0.9545 | 0.9589 | +0.89 | +0.44 | 43/11 | 21/5 |
| Resize + Compression | 0.9046 | 0.9057 | +0.11 pp | 0.8206 | 0.9037 | 0.9040 | 0.9058 | 0.9745 | 0.9656 | +0.44 | +1.00 | 45/9 | 24/5 |
| Crop + Resize + Compression | 0.8990 | 0.9123 | +1.33 pp | 0.8106 | 0.9070 | 0.8982 | 0.9123 | 0.9545 | 0.9567 | +1.00 | +0.33 | 43/14 | 22/6 |

## CASIA subgroup results

All subgroups below existed in the test split. Family aggregates (`Tp_D`, `Tp_S`) and parsed fine groups are calculated from original-condition inference.

| CASIA subgroup | Support | Current manipulated recall | Candidate manipulated recall | Current M→Real / M→AI | Candidate M→Real / M→AI |
|---|---:|---:|---:|---:|---:|
| Tp_D | 110 | 0.9364 | 0.9182 | 6 / 1 | 9 / 0 |
| Tp_S | 191 | 0.7487 | 0.8743 | 42 / 6 | 19 / 5 |
| Tp_D_CND | 1 | 1.0000 | 1.0000 | 0 / 0 | 0 / 0 |
| Tp_D_CNN | 4 | 0.7500 | 0.7500 | 1 / 0 | 1 / 0 |
| Tp_D_CRD | 2 | 1.0000 | 1.0000 | 0 / 0 | 0 / 0 |
| Tp_D_CRN | 15 | 1.0000 | 0.8667 | 0 / 0 | 2 / 0 |
| Tp_D_NND | 2 | 1.0000 | 1.0000 | 0 / 0 | 0 / 0 |
| Tp_D_NNN | 22 | 0.9091 | 0.9091 | 2 / 0 | 2 / 0 |
| Tp_D_NRD | 5 | 1.0000 | 1.0000 | 0 / 0 | 0 / 0 |
| Tp_D_NRN | 59 | 0.9322 | 0.9322 | 3 / 1 | 4 / 0 |
| Tp_S_CND | 2 | 1.0000 | 1.0000 | 0 / 0 | 0 / 0 |
| Tp_S_CNN | 30 | 0.8000 | 0.8667 | 5 / 1 | 3 / 1 |
| Tp_S_CRD | 3 | 0.6667 | 1.0000 | 1 / 0 | 0 / 0 |
| Tp_S_CRN | 16 | 0.5625 | 0.6875 | 6 / 1 | 4 / 1 |
| Tp_S_NND | 10 | 0.9000 | 0.9000 | 1 / 0 | 1 / 0 |
| Tp_S_NNN | 85 | 0.7765 | 0.9294 | 17 / 2 | 4 / 2 |
| Tp_S_NRD | 8 | 0.8750 | 1.0000 | 1 / 0 | 0 / 0 |
| Tp_S_NRN | 37 | 0.6486 | 0.7838 | 11 / 2 | 7 / 1 |

## Previous false-negative diagnostic images

These externally specified files were verified against the manifest by path and SHA-256 and excluded from dataset creation. They are diagnostics only and did not affect training or selection.

| Image | Condition | Current prediction | Candidate prediction | Candidate Real / AI / Manipulated probabilities |
|---|---|---|---|---|
| Tp_S_NNN_S_N_nat20004_nat20004_01508.tif | Original | Real (0.6263) | Manipulated (0.6032) | 0.3924 / 0.0044 / 0.6032 |
| Tp_S_NRN_S_N_cha10207_cha10207_12340.jpg | Original | Real (0.9963) | Real (0.7122) | 0.7122 / 0.0000 / 0.2878 |
| Tp_S_NRN_S_N_cha00037_cha00037_00357.tif | Original | Real (0.9605) | Manipulated (0.8452) | 0.1232 / 0.0316 / 0.8452 |
| Tp_S_NNN_S_N_nat20004_nat20004_01508.tif | Crop | Real (0.7598) | Manipulated (0.6575) | 0.3408 / 0.0017 / 0.6575 |
| Tp_S_NRN_S_N_cha10207_cha10207_12340.jpg | Crop | Real (0.9971) | Real (0.7875) | 0.7875 / 0.0000 / 0.2125 |
| Tp_S_NRN_S_N_cha00037_cha00037_00357.tif | Crop | Real (0.9574) | Manipulated (0.7478) | 0.1326 / 0.1196 / 0.7478 |
| Tp_S_NNN_S_N_nat20004_nat20004_01508.tif | Resize | Real (0.6209) | Manipulated (0.6122) | 0.3837 / 0.0041 / 0.6122 |
| Tp_S_NRN_S_N_cha10207_cha10207_12340.jpg | Resize | Real (0.9954) | Real (0.6942) | 0.6942 / 0.0000 / 0.3058 |
| Tp_S_NRN_S_N_cha00037_cha00037_00357.tif | Resize | Real (0.9699) | Manipulated (0.7430) | 0.2096 / 0.0474 / 0.7430 |
| Tp_S_NNN_S_N_nat20004_nat20004_01508.tif | JPEG Compression | Real (0.5709) | Manipulated (0.6184) | 0.3761 / 0.0055 / 0.6184 |
| Tp_S_NRN_S_N_cha10207_cha10207_12340.jpg | JPEG Compression | Real (0.9947) | Real (0.6413) | 0.6413 / 0.0000 / 0.3587 |
| Tp_S_NRN_S_N_cha00037_cha00037_00357.tif | JPEG Compression | Real (0.9367) | Manipulated (0.8721) | 0.0809 / 0.0470 / 0.8721 |
| Tp_S_NNN_S_N_nat20004_nat20004_01508.tif | Crop + Resize | Real (0.7809) | Manipulated (0.6468) | 0.3515 / 0.0017 / 0.6468 |
| Tp_S_NRN_S_N_cha10207_cha10207_12340.jpg | Crop + Resize | Real (0.9964) | Real (0.7776) | 0.7776 / 0.0000 / 0.2224 |
| Tp_S_NRN_S_N_cha00037_cha00037_00357.tif | Crop + Resize | Real (0.9760) | Manipulated (0.5862) | 0.2634 / 0.1504 / 0.5862 |
| Tp_S_NNN_S_N_nat20004_nat20004_01508.tif | Crop + Compression | Real (0.7241) | Manipulated (0.6731) | 0.3250 / 0.0019 / 0.6731 |
| Tp_S_NRN_S_N_cha10207_cha10207_12340.jpg | Crop + Compression | Real (0.9973) | Real (0.7757) | 0.7757 / 0.0000 / 0.2243 |
| Tp_S_NRN_S_N_cha00037_cha00037_00357.tif | Crop + Compression | Real (0.9223) | Manipulated (0.7785) | 0.0750 / 0.1465 / 0.7785 |
| Tp_S_NNN_S_N_nat20004_nat20004_01508.tif | Resize + Compression | Real (0.5523) | Manipulated (0.6307) | 0.3646 / 0.0048 / 0.6307 |
| Tp_S_NRN_S_N_cha10207_cha10207_12340.jpg | Resize + Compression | Real (0.9956) | Real (0.6663) | 0.6663 / 0.0000 / 0.3337 |
| Tp_S_NRN_S_N_cha00037_cha00037_00357.tif | Resize + Compression | Real (0.9512) | Manipulated (0.7689) | 0.1533 / 0.0778 / 0.7689 |
| Tp_S_NNN_S_N_nat20004_nat20004_01508.tif | Crop + Resize + Compression | Real (0.7363) | Manipulated (0.6713) | 0.3269 / 0.0018 / 0.6713 |
| Tp_S_NRN_S_N_cha10207_cha10207_12340.jpg | Crop + Resize + Compression | Real (0.9973) | Real (0.7648) | 0.7648 / 0.0000 / 0.2352 |
| Tp_S_NRN_S_N_cha00037_cha00037_00357.tif | Crop + Resize + Compression | Real (0.9603) | Manipulated (0.6833) | 0.1749 / 0.1418 / 0.6833 |

## Interpretation

The candidate increased Original accuracy by +0.67 pp, Manipulated recall by +7.31 pp, and mean accuracy across the seven transformed conditions by +1.16 pp; its accuracy was higher in all seven conditions. The trade-off is lower Real recall (-4.00 pp), lower Manipulated precision (-4.15 pp), and +13 additional Real→Manipulated errors on 300 Real test images (16 to 29). It classified 2/3 of the supplied diagnostic examples as Manipulated on their original versions. The prediction-consistency values also vary by condition; they are listed in the comparison table. These measurements show a candidate trade-off, not a guarantee for other data or transformations.

## Limitations

- Dataset labels remain source-level class labels; subgroup results do not independently validate the exact manipulation operation.
- Filename parsing supplies CASIA group provenance and may not describe every image's manipulation precisely.
- The per-group three-image floor is a sampling policy, not equalized subgroup representation; subgroup estimates with low support are uncertain.
- Transformations use fixed deterministic strengths and a centered crop; real transformations vary.
- The supplied three diagnostic cases are not a representative evaluation sample.
- Confidence values are softmax scores and were not calibrated.
- The candidate is not active in Flask; no promotion decision was applied automatically.

## Conclusion

The candidate should replace the current model only if the measured overall, manipulated, subgroup, and transformed-condition trade-offs are acceptable. This report makes no robustness guarantee outside the evaluated held-out split and transformations.
