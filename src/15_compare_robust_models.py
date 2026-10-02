"""Compare baseline and augmentation-trained models on the preserved robustness sample."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

predictor_module = import_module("src.13_predict_efficientnet_image")
robustness_module = import_module("src.05_robustness_testing")
LABELS = predictor_module.LABELS
BASELINE_MODEL = ROOT / "results/efficientnet_fused_best_model.pth"
NEW_MODEL = ROOT / "results/efficientnet_fused_robust_best_model.pth"
BASELINE_RESULTS = ROOT / "results/robustness/robustness_results.json"
BASELINE_PREDICTIONS = ROOT / "results/robustness/predictions.csv"
OUTPUT_DIR = ROOT / "results/robustness_improved"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_baseline_sample() -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, list[dict[str, str]]]]:
    if not BASELINE_RESULTS.is_file() or not BASELINE_PREDICTIONS.is_file():
        raise FileNotFoundError("Preserved baseline robustness JSON/CSV is missing; refusing to substitute a new sample.")
    baseline = json.loads(BASELINE_RESULTS.read_text(encoding="utf-8"))
    with BASELINE_PREDICTIONS.open("r", newline="", encoding="utf-8-sig") as stream:
        prediction_rows = list(csv.DictReader(stream))
    conditions: dict[str, list[dict[str, str]]] = {}
    for row in prediction_rows:
        conditions.setdefault(row["condition"], []).append(row)
    original = conditions.get("original", [])
    if len(original) != int(baseline["sample_size"]):
        raise ValueError("Baseline prediction CSV does not match its recorded sample size.")
    selected = []
    for row in original:
        path = (ROOT / row["image"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Baseline test sample image is missing: {path}")
        if row["ground_truth"] not in LABELS:
            raise ValueError(f"Unexpected test label: {row['ground_truth']}")
        selected.append({"filepath": path, "ground_truth": row["ground_truth"],
                         "label": LABELS.index(row["ground_truth"])})
    if len(selected) != baseline["sample_size"]:
        raise ValueError("Could not reconstruct exact baseline test sample.")
    for condition, _, _ in robustness_module.CONDITIONS:
        if condition not in conditions or len(conditions[condition]) != len(selected):
            raise ValueError(f"Baseline predictions incomplete for condition {condition}.")
    return selected, baseline, conditions


def plot_comparison(old: dict[str, Any], new: dict[str, Any], strongest: str) -> None:
    order = [key for key, _, _ in robustness_module.CONDITIONS]
    labels = [name for _, name, _ in robustness_module.CONDITIONS]
    x = np.arange(len(order))
    width = 0.36

    def paired_plot(old_key: str, new_key: str, ylabel: str, title: str, filename: str,
                    percent: bool = True, include_original: bool = True):
        indices = list(range(len(order))) if include_original else list(range(1, len(order)))
        display_labels = [labels[index] for index in indices]
        old_values = [old[order[index]][old_key] * (100 if percent else 1) for index in indices]
        new_values = [new[order[index]][new_key] * (100 if percent else 1) for index in indices]
        positions = np.arange(len(indices))
        fig, ax = plt.subplots(figsize=(13, 6))
        ax.bar(positions - width / 2, old_values, width, label="Baseline", color="#64748b")
        ax.bar(positions + width / 2, new_values, width, label="Augmentation-trained", color="#2563eb")
        ax.set_xticks(positions, display_labels, rotation=23, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        if percent:
            ax.set_ylim(0, 100)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / filename, dpi=180)
        plt.close(fig)

    paired_plot("accuracy", "accuracy", "Accuracy (%)", "Baseline vs New Accuracy by Condition",
                "robustness_accuracy_comparison.png")
    paired_plot("f1_macro", "f1_macro", "Macro F1 (%)", "Baseline vs New Macro F1",
                "f1_comparison.png")
    paired_plot("prediction_consistency", "prediction_consistency", "Consistency (%)",
                "Baseline vs New Prediction Consistency", "prediction_consistency_comparison.png",
                include_original=False)

    fig, ax = plt.subplots(figsize=(13, 6))
    old_drops = [old[key]["accuracy_drop_percentage_points"] for key in order]
    new_drops = [new[key]["accuracy_drop_percentage_points"] for key in order]
    ax.bar(x - width / 2, old_drops, width, label="Baseline", color="#64748b")
    ax.bar(x + width / 2, new_drops, width, label="Augmentation-trained", color="#d97706")
    ax.axhline(0, color="#111827", linewidth=0.8)
    ax.set_xticks(x, labels, rotation=23, ha="right")
    ax.set_ylabel("Accuracy drop (percentage points)")
    ax.set_title("Accuracy Drop from Each Model's Original Condition")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "accuracy_drop_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    old_original = old["original"]["accuracy"] * 100
    new_original = new["original"]["accuracy"] * 100
    bars = ax.bar(["Baseline", "Augmentation-trained"], [old_original, new_original],
                  color=["#64748b", "#2563eb"])
    ax.bar_label(bars, fmt="%.2f%%", padding=4)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Original Test-Sample Accuracy")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "original_accuracy_comparison.png", dpi=180)
    plt.close(fig)

    for condition, filename, title in (
        ("original", "new_confusion_matrix_original.png", "New Model: Original Test Sample"),
        (strongest, "new_confusion_matrix_strongest_transformation.png",
         f"New Model: Strongest Combined Condition ({dict((k, n) for k, n, _ in robustness_module.CONDITIONS)[strongest]})"),
    ):
        matrix = np.asarray(new[condition]["metrics"]["confusion_matrix"]["rows_actual_columns_predicted"])
        fig, ax = plt.subplots(figsize=(7, 6))
        image = ax.imshow(matrix, cmap="Blues")
        fig.colorbar(image, ax=ax, label="Image count")
        ax.set(xticks=range(len(LABELS)), yticks=range(len(LABELS)), xticklabels=LABELS,
               yticklabels=LABELS, xlabel="Predicted class", ylabel="Ground-truth class", title=title)
        for row in range(len(LABELS)):
            for col in range(len(LABELS)):
                ax.text(col, row, str(matrix[row, col]), ha="center", va="center")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / filename, dpi=180)
        plt.close(fig)


def run_unknown_demo(image_path: Path, selected_paths: set[Path], all_manifest_paths: set[Path],
                     predictor: dict[str, Any], config: SimpleNamespace) -> dict[str, Any]:
    path = image_path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Unknown-image demonstration file not found: {path}")
    if path in all_manifest_paths or path in selected_paths:
        raise ValueError("Demonstration image is part of the prepared dataset; choose a new image.")
    test_manifest = pd.read_csv(ROOT / "data/processed/dataset_manifest.csv")
    for name in test_manifest["filepath"]:
        if file_hash(Path(name)) == file_hash(path):
            raise ValueError("Demonstration image content matches a dataset image; refusing to present it as unseen.")

    with Image.open(path) as source:
        original = source.convert("RGB")
    records = []
    for key, display, operations in robustness_module.CONDITIONS:
        if key != "original" and key not in {
            "crop", "resize", "jpeg_compression", "crop_resize_compression",
        }:
            continue
        transformed = robustness_module.transform_image(original, operations, config) if operations else original
        result = predictor_module.predict_image_details(transformed, predictor=predictor)
        records.append({"condition": display, **result})
        print(f"Unknown image | {display}: {result['predicted_class']} ({result['confidence'] * 100:.2f}%)")
    return {"image": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else path.name,
            "sha256": file_hash(path), "used_for_training_or_selection": False,
            "predictions": records}


def write_report(data: dict[str, Any]) -> None:
    condition_names = {key: name for key, name, _ in robustness_module.CONDITIONS}
    old, new = data["baseline_results"], data["new_results"]
    overall_rows, robustness_rows = [], []
    for key, _, _ in robustness_module.CONDITIONS:
        baseline_item, new_item = old[key], new[key]
        base, robust = baseline_item["metrics"], new_item["metrics"]
        overall_rows.append(
            f"| {condition_names[key]} | {base['accuracy']:.4f} | {base['precision_macro']:.4f} | "
            f"{base['recall_macro']:.4f} | {base['f1_macro']:.4f} | "
            f"{robust['accuracy']:.4f} | {robust['precision_macro']:.4f} | "
            f"{robust['recall_macro']:.4f} | {robust['f1_macro']:.4f} |"
        )
        robustness_rows.append(
            f"| {condition_names[key]} | {base['accuracy']:.4f} | {robust['accuracy']:.4f} | "
            f"{robust['accuracy'] - base['accuracy']:+.4f} | "
            f"{base['precision_macro']:.4f} / {robust['precision_macro']:.4f} | "
            f"{base['recall_macro']:.4f} / {robust['recall_macro']:.4f} | "
            f"{base['f1_macro']:.4f} / {robust['f1_macro']:.4f} | "
            f"{baseline_item['prediction_consistency']:.4f} / {new_item['prediction_consistency']:.4f} | "
            f"{baseline_item['accuracy_drop_percentage_points']:+.2f} / "
            f"{new_item['accuracy_drop_percentage_points']:+.2f} pp |"
        )
    decision = data["deployment_decision"]
    confidence_rows = []
    per_class_sections = []
    for key, _, _ in robustness_module.CONDITIONS:
        confidence_rows.append(
            f"| {condition_names[key]} | {old[key]['mean_confidence_change'] * 100:+.2f} / "
            f"{old[key]['mean_absolute_confidence_change'] * 100:.2f} | "
            f"{new[key]['mean_confidence_change'] * 100:+.2f} / "
            f"{new[key]['mean_absolute_confidence_change'] * 100:.2f} |"
        )
        class_rows = [
            f"| {name} | {row['precision']:.4f} | {row['recall']:.4f} | "
            f"{row['f1']:.4f} | {row['support']} |"
            for name, row in new[key]["metrics"]["per_class"].items()
        ]
        per_class_sections.append(
            f"#### {condition_names[key]}\n\n| Class | Precision | Recall | F1 | Support |\n"
            f"|---|---:|---:|---:|---:|\n" + "\n".join(class_rows)
        )
    if data.get("unknown_image_demo"):
        demo_summary = (
            f"An image outside all manifest paths and with a non-matching SHA-256 was used only for inference: "
            f"`{data['unknown_image_demo']['image']}`. It was not used for training or model selection. Predictions: "
            + "; ".join(f"{row['condition']} = {row['predicted_class']} ({row['confidence'] * 100:.2f}%)"
                         for row in data["unknown_image_demo"]["predictions"])
        )
    else:
        demo_summary = "No external demonstration image was evaluated."
    report = f'''# Robustness Improvement Report

## Objective

Train a transformation-aware version of the existing EfficientNet-B0 fused classifier and compare its actual predictions with the preserved baseline under crop, resize, JPEG compression, and combinations.

## Baseline model and problem

- Baseline checkpoint: `{data['baseline_checkpoint']}`
- Baseline checkpoint SHA-256: `{data['baseline_checkpoint_sha256']}`
- A byte-for-byte baseline checkpoint copy is preserved at `results/robustness_improved/efficientnet_fused_baseline_backup.pth`.
- Existing baseline metrics and the previous robustness JSON are copied into this directory; their original files remain untouched.
- The baseline experiment observed Real predictions changing under combinations. On its fixed 300-image test sample the strongest combined condition had accuracy {old['crop_resize_compression']['metrics']['accuracy']:.4f}, versus {old['original']['metrics']['accuracy']:.4f} for Original.

## Training augmentation and preserved architecture

The new checkpoint keeps torchvision EfficientNet-B0, its 1,280 deep features, 17 handcrafted FFT/Haar/noise features, scaler, feature fusion head, and Real / AI-Generated / Manipulated output. The backbone remained frozen as in the current training pipeline; the existing fused classifier was fine-tuned. Each original training row was retained and paired with one random in-memory augmented view. Random crop, resize down/up degradation, and JPEG encoding were independently sampled using the training parameters recorded in `training_metrics.json`. Augmentation ran before the unchanged image and forensic-feature preprocessing. No augmented files were added to the dataset.

## Dataset split and anti-leakage

The existing manifest was used unchanged: train {data['split_counts']['train']}, validation {data['split_counts']['val']}, and test {data['split_counts']['test']} rows. Each of the three classes was balanced in these splits. The training script checked manifest path separation and SHA-256 content overlap across train, validation, and test before feature extraction. Only train rows generated augmented views or received gradients. Validation images were untransformed and selected the checkpoint by validation accuracy. Test predictions were run afterward and did not choose an epoch or train weights. The same existing 300-image, 100-per-class held-out sample and condition definitions as the archived baseline were used for old/new comparison.

## New model and experimental setup

- New checkpoint: `{data['new_checkpoint']}` (separate from baseline)
- Initialization: baseline EfficientNet-B0 checkpoint, including its train-fitted feature scaler
- Selected epoch: {data['training']['selected_epoch']}; validation accuracy: {data['training']['selected_validation_accuracy']:.4f}
- Evaluated test sample: {data['sample_size']} images; per-class counts `{json.dumps(data['sample_class_counts'])}`
- Crop: centered 90% of each dimension, restored with bicubic resampling
- Resize: bicubic downsample to 75%, then upsample to original dimensions
- JPEG: quality 60
- Combined transformations use crop, resize, then JPEG
- Precision, recall, and F1 are macro averages across the three classes. Prediction consistency compares each model's transformed prediction with that model's own Original prediction. Confidence deltas are recorded in the JSON and per-image CSV; they are not calibration measurements.

## Accuracy, precision, recall, and F1 by condition

| Condition | Baseline accuracy | Baseline precision | Baseline recall | Baseline F1 | New accuracy | New precision | New recall | New F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(overall_rows)}

## Robustness comparison

| Transformation | Old accuracy | New accuracy | Improvement | Old / new precision | Old / new recall | Old / new F1 | Old / new consistency | Old / new accuracy drop |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(robustness_rows)}

### Confidence stability

Signed mean confidence change (percentage points) and mean absolute confidence change (percentage points), transformed minus that model's Original condition:

| Condition | Baseline signed / absolute | New signed / absolute |
|---|---:|---:|
{chr(10).join(confidence_rows)}

### New-model per-class metrics

{chr(10).join(per_class_sections)}

## Unknown-image demonstration

{demo_summary}

## Interpretation and limitations

- Mean transformed accuracy change (excluding Original): {decision['mean_transformed_accuracy_change_pp']:+.2f} percentage points.
- Mean combined-condition accuracy change: {decision['mean_combined_accuracy_change_pp']:+.2f} percentage points.
- Original-condition accuracy change: {decision['original_accuracy_change_pp']:+.2f} percentage points.
- Release check (original loss no worse than {decision['max_original_loss_pp']:.2f} pp, with positive mean transformed and combined accuracy changes): **{'PASS' if decision['passed'] else 'NOT PASSED'}**.
- The test sample has 100 images per class and has sampling uncertainty. Transform strengths and kernels are fixed; performance may differ for other edits, sources, crop locations, or JPEG settings.
- This dataset's Real, AI-Generated, and Manipulated classes come from distinct source datasets, so source-specific cues may affect classification.
- Confidence is a softmax score, not a calibrated probability. Calibration was not evaluated.
- Any observed gains apply only to this checkpoint, this test sample, and the stated transformations. A passing comparison does not guarantee general robustness.

## Conclusion

{decision['conclusion']}
'''
    (OUTPUT_DIR / "ROBUSTNESS_IMPROVEMENT_REPORT.md").write_text(report, encoding="utf-8")


def finalize_artifacts(data: dict[str, Any]) -> None:
    old_results = data["baseline_results"]
    new_results = data["new_results"]
    strongest = data["strongest_combined_condition"]
    demo_rows = data.get("unknown_image_demo", {}).get("predictions", [])
    with (OUTPUT_DIR / "unknown_image_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["condition", "predicted_class", "confidence"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in demo_rows)
    plot_comparison(old_results, new_results, strongest)
    write_report(data)
    print("\nBASELINE vs NEW MODEL (macro metrics; same 300 held-out images)")
    for key, display, _ in robustness_module.CONDITIONS:
        old_item, new_item = old_results[key], new_results[key]
        print(f"{display}: accuracy {old_item['accuracy']:.4f} -> {new_item['accuracy']:.4f} "
              f"({(new_item['accuracy'] - old_item['accuracy']) * 100:+.2f} pp); "
              f"P {old_item['precision_macro']:.4f}->{new_item['precision_macro']:.4f}; "
              f"R {old_item['recall_macro']:.4f}->{new_item['recall_macro']:.4f}; "
              f"F1 {old_item['f1_macro']:.4f}->{new_item['f1_macro']:.4f}; "
              f"consistency {old_item['prediction_consistency']:.4f}->{new_item['prediction_consistency']:.4f}; "
              f"accuracy drop {old_item['accuracy_drop_percentage_points']:+.2f}->"
              f"{new_item['accuracy_drop_percentage_points']:+.2f} pp")
        for label in LABELS:
            old_class = old_item["metrics"]["per_class"][label]
            new_class = new_item["metrics"]["per_class"][label]
            print(f"  {label} recall: {old_class['recall']:.4f}->{new_class['recall']:.4f}; "
                  f"F1: {old_class['f1']:.4f}->{new_class['f1']:.4f}")
    print(f"\nDecision: {'PASS' if data['deployment_decision']['passed'] else 'NOT PASSED'} | "
          f"{data['deployment_decision']['conclusion']}")
    if demo_rows:
        print(f"Unknown-image demo: {data['unknown_image_demo']['image']}")
        for row in demo_rows:
            print(f"  {row['condition']}: {row['predicted_class']} ({row['confidence'] * 100:.2f}%)")
    print(f"Saved comparison artifacts to {OUTPUT_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-image", type=Path,
                        default=ROOT / "data/uploads/WIN_20260921_13_21_51_Pro.jpg",
                        help="Image outside the manifest used only for a final inference demonstration.")
    parser.add_argument("--max-original-loss-pp", type=float, default=2.0,
                        help="Maximum acceptable original-condition accuracy loss for Flask promotion.")
    parser.add_argument("--finalize-existing", action="store_true",
                        help="Regenerate plots/report/terminal table from the existing measured comparison JSON.")
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.finalize_existing:
        existing = OUTPUT_DIR / "comparison_results.json"
        if not existing.is_file():
            raise FileNotFoundError(f"No saved comparison to finalize: {existing}")
        finalize_artifacts(json.loads(existing.read_text(encoding="utf-8")))
        return
    if not BASELINE_MODEL.is_file() or not NEW_MODEL.is_file():
        raise FileNotFoundError("Both baseline and separately trained robustness checkpoint must exist.")
    selected, baseline, baseline_rows = load_baseline_sample()
    baseline_hash = file_hash(BASELINE_MODEL)
    if baseline_hash != baseline["checkpoint_sha256"]:
        raise ValueError("Current baseline checkpoint hash differs from the checkpoint used for preserved baseline results.")
    backup = OUTPUT_DIR / "efficientnet_fused_baseline_backup.pth"
    if not backup.is_file() or file_hash(backup) != baseline_hash:
        raise ValueError("Preserved baseline model backup is missing or does not match the original checkpoint.")

    old_results = {}
    for condition, _, _ in robustness_module.CONDITIONS:
        summary = baseline["results"][condition]
        old_results[condition] = {
            "accuracy": summary["metrics"]["accuracy"],
            "precision_macro": summary["metrics"]["precision_macro"],
            "recall_macro": summary["metrics"]["recall_macro"],
            "f1_macro": summary["metrics"]["f1_macro"],
            "prediction_consistency": summary["prediction_consistency"],
            "accuracy_drop_percentage_points": summary["accuracy_drop_percentage_points"],
            "mean_confidence_change": summary["mean_confidence_change"],
            "mean_absolute_confidence_change": summary["mean_absolute_confidence_change"],
            "metrics": summary["metrics"],
        }

    print(f"Loading new checkpoint: {NEW_MODEL}", flush=True)
    predictor = predictor_module.load_predictor(NEW_MODEL)
    transform_config = SimpleNamespace(crop_fraction=0.9, resize_scale=0.75, jpeg_quality=60)
    truth = [row["label"] for row in selected]
    condition_predictions: dict[str, list[int]] = {}
    condition_confidence: dict[str, list[float]] = {}
    detailed_predictions: dict[str, list[dict[str, Any]]] = {}
    prediction_rows = []

    for key, display, operations in robustness_module.CONDITIONS:
        print(f"Evaluating new model: {display} ({len(selected)} shared held-out images)", flush=True)
        images = []
        for row in selected:
            with Image.open(row["filepath"]) as source:
                original = source.convert("RGB")
            images.append(robustness_module.transform_image(original, operations, transform_config)
                          if operations else original)
        results = []
        for start in range(0, len(images), 32):
            # These images come from the prepared manifest, so do not repeat its JPEG/resize step.
            results.extend(predictor_module.predict_images_details(images[start:start + 32], predictor=predictor,
                                                                   already_prepared=True))
            print(f"  {min(start + 32, len(images))}/{len(images)}", flush=True)
        detailed_predictions[key] = results
        condition_predictions[key] = [LABELS.index(result["predicted_class"]) for result in results]
        condition_confidence[key] = [result["confidence"] for result in results]
        for row, result in zip(selected, results):
            prediction_rows.append({
                "condition": key, "display_name": display,
                "image": str(row["filepath"].relative_to(ROOT)),
                "ground_truth": row["ground_truth"], "predicted_class": result["predicted_class"],
                "confidence": result["confidence"],
                **{f"probability_{name}": result["probabilities"][name] for name in LABELS},
            })

    new_results = {}
    original_predictions = condition_predictions["original"]
    original_confidence = condition_confidence["original"]
    for key, display, _ in robustness_module.CONDITIONS:
        prediction = condition_predictions[key]
        confidence = condition_confidence[key]
        metrics = robustness_module.metric_bundle(truth, prediction)
        consistency = sum(left == right for left, right in zip(original_predictions, prediction)) / len(truth)
        delta = [right - left for left, right in zip(original_confidence, confidence)]
        new_results[key] = {
            "display_name": display, "accuracy": metrics["accuracy"],
            "precision_macro": metrics["precision_macro"], "recall_macro": metrics["recall_macro"],
            "f1_macro": metrics["f1_macro"], "prediction_consistency": consistency,
            "mean_confidence_change": float(np.mean(delta)),
            "mean_absolute_confidence_change": float(np.mean(np.abs(delta))),
            "accuracy_drop_percentage_points": (new_results["original"]["accuracy"] - metrics["accuracy"]) * 100
                if key != "original" else 0.0,
            "metrics": metrics,
            "predictions": [result["predicted_class"] for result in detailed_predictions[key]],
        }

    # The strongest condition is chosen from combined transforms by lowest measured new accuracy.
    combined = [key for key, _, operations in robustness_module.CONDITIONS if len(operations) >= 2]
    strongest = min(combined, key=lambda key: new_results[key]["accuracy"])
    transformed = [key for key, _, operations in robustness_module.CONDITIONS if operations]
    mean_transformed_delta = float(np.mean([new_results[key]["accuracy"] - old_results[key]["accuracy"]
                                            for key in transformed]) * 100)
    mean_combined_delta = float(np.mean([new_results[key]["accuracy"] - old_results[key]["accuracy"]
                                         for key in combined]) * 100)
    original_delta = float((new_results["original"]["accuracy"] - old_results["original"]["accuracy"]) * 100)
    passed = (original_delta >= -args.max_original_loss_pp and mean_transformed_delta > 0
              and mean_combined_delta > 0)
    decision = {
        "passed": passed, "max_original_loss_pp": args.max_original_loss_pp,
        "original_accuracy_change_pp": original_delta,
        "mean_transformed_accuracy_change_pp": mean_transformed_delta,
        "mean_combined_accuracy_change_pp": mean_combined_delta,
        "flask_should_use_new_checkpoint": passed,
        "conclusion": (
            "The candidate meets the predeclared comparison check on this sample; promotion is limited to this measured evaluation."
            if passed else
            "The candidate did not meet the predeclared comparison check. Keep Flask on the baseline and use the observed trade-off to guide the next experiment."
        ),
    }

    manifest = pd.read_csv(ROOT / "data/processed/dataset_manifest.csv")
    all_manifest_paths = {Path(path).resolve() for path in manifest["filepath"]}
    selected_paths = {row["filepath"] for row in selected}
    demo = run_unknown_demo(args.demo_image, selected_paths, all_manifest_paths, predictor, transform_config)
    training_record = json.loads((OUTPUT_DIR / "training_metrics.json").read_text(encoding="utf-8"))
    data = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "baseline_checkpoint": str(BASELINE_MODEL.relative_to(ROOT)),
        "baseline_checkpoint_sha256": baseline_hash,
        "new_checkpoint": str(NEW_MODEL.relative_to(ROOT)),
        "new_checkpoint_sha256": file_hash(NEW_MODEL),
        "dataset_manifest": "data/processed/dataset_manifest.csv",
        "split_counts": {name: int(count) for name, count in manifest["split"].value_counts().items()},
        "sample_size": len(selected), "sample_class_counts": baseline["sample_class_counts"],
        "conditions": [key for key, _, _ in robustness_module.CONDITIONS],
        "strongest_combined_condition": strongest,
        "baseline_results": old_results, "new_results": new_results,
        "training": {"selected_epoch": training_record["selected_epoch"],
                     "selected_validation_accuracy": training_record["selected_validation_accuracy"]},
        "training_record": "results/robustness_improved/training_metrics.json",
        "deployment_decision": decision, "unknown_image_demo": demo,
    }
    (OUTPUT_DIR / "comparison_results.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    with (OUTPUT_DIR / "new_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)
    with (OUTPUT_DIR / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Condition", "Old Accuracy", "New Accuracy", "Improvement (pp)",
                         "Old Precision", "New Precision", "Old Recall", "New Recall",
                         "Old F1", "New F1", "Old Consistency", "New Consistency",
                         "Old Accuracy Drop (pp)", "New Accuracy Drop (pp)"])
        for key, display, _ in robustness_module.CONDITIONS:
            writer.writerow([display, old_results[key]["accuracy"], new_results[key]["accuracy"],
                             (new_results[key]["accuracy"] - old_results[key]["accuracy"]) * 100,
                             old_results[key]["precision_macro"], new_results[key]["precision_macro"],
                             old_results[key]["recall_macro"], new_results[key]["recall_macro"],
                             old_results[key]["f1_macro"], new_results[key]["f1_macro"],
                             old_results[key]["prediction_consistency"], new_results[key]["prediction_consistency"],
                             old_results[key]["accuracy_drop_percentage_points"],
                             new_results[key]["accuracy_drop_percentage_points"]])
    with (OUTPUT_DIR / "unknown_image_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["condition", "predicted_class", "confidence"])
        writer.writeheader()
        writer.writerows({field: row[field] for field in ("condition", "predicted_class", "confidence")}
                         for row in demo["predictions"])
    finalize_artifacts(data)


if __name__ == "__main__":
    main()
