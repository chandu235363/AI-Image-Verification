"""Evaluate the active EfficientNet-B0 fused predictor under image transformations."""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import sys
from importlib import import_module
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_predictor_module = import_module("src.13_predict_efficientnet_image")
LABELS = _predictor_module.LABELS
load_predictor = _predictor_module.load_predictor
predict_image_details = _predictor_module.predict_image_details
predict_images_details = _predictor_module.predict_images_details

MANIFEST_PATH = ROOT / "data/processed/dataset_manifest.csv"
OUTPUT_DIR = ROOT / "results/robustness"
CONDITIONS = [
    ("original", "Original", ()),
    ("crop", "Crop", ("crop",)),
    ("resize", "Resize", ("resize",)),
    ("jpeg_compression", "JPEG Compression", ("jpeg",)),
    ("crop_resize", "Crop + Resize", ("crop", "resize")),
    ("crop_compression", "Crop + Compression", ("crop", "jpeg")),
    ("resize_compression", "Resize + Compression", ("resize", "jpeg")),
    ("crop_resize_compression", "Crop + Resize + Compression", ("crop", "resize", "jpeg")),
]


def read_test_rows() -> list[dict[str, Any]]:
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Dataset manifest not found: {MANIFEST_PATH}")
    with MANIFEST_PATH.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"filepath", "label", "class_name", "split"}
        if required - set(reader.fieldnames or []):
            raise ValueError(f"Manifest missing columns: {sorted(required - set(reader.fieldnames or []))}")
        rows = []
        for row in reader:
            if row["split"].strip().lower() != "test":
                continue
            class_name = row["class_name"].strip()
            label = int(row["label"])
            if class_name not in LABELS or LABELS[label] != class_name:
                raise ValueError(f"Invalid class/label pair in manifest: {class_name!r}, {label}")
            path = Path(row["filepath"])
            if not path.is_absolute():
                path = ROOT / path
            if not path.is_file():
                raise FileNotFoundError(f"Test image missing: {path}")
            rows.append({"filepath": path, "label": label, "class_name": class_name})
    if not rows:
        raise ValueError("Manifest has no test split; refusing to evaluate another split.")
    present = {row["class_name"] for row in rows}
    if present != set(LABELS):
        raise ValueError(f"Test split must contain all classes; found {sorted(present)}")
    return rows


def stratified_sample(rows: list[dict[str, Any]], sample_size: int, seed: int) -> list[dict[str, Any]]:
    if sample_size < 1 or sample_size >= len(rows):
        return rows
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["class_name"]].append(row)
    rng = random.Random(seed)
    for items in groups.values():
        rng.shuffle(items)
    quotient, remainder = divmod(sample_size, len(LABELS))
    selected = []
    for index, label in enumerate(LABELS):
        count = quotient + (1 if index < remainder else 0)
        if count > len(groups[label]):
            raise ValueError(f"Sample size requests more {label} images than exist.")
        selected.extend(groups[label][:count])
    rng.shuffle(selected)
    return selected


def transform_image(image: Image.Image, operations: tuple[str, ...], args) -> Image.Image:
    result = image.convert("RGB")
    width, height = result.size
    for operation in operations:
        if operation == "crop":
            crop_width = max(1, round(width * args.crop_fraction))
            crop_height = max(1, round(height * args.crop_fraction))
            left, top = (width - crop_width) // 2, (height - crop_height) // 2
            result = result.crop((left, top, left + crop_width, top + crop_height))
            result = result.resize((width, height), Image.Resampling.BICUBIC)
        elif operation == "resize":
            reduced_size = (max(1, round(width * args.resize_scale)), max(1, round(height * args.resize_scale)))
            result = result.resize(reduced_size, Image.Resampling.BICUBIC)
            result = result.resize((width, height), Image.Resampling.BICUBIC)
        elif operation == "jpeg":
            buffer = io.BytesIO()
            result.save(buffer, format="JPEG", quality=args.jpeg_quality, optimize=False, progressive=False)
            buffer.seek(0)
            with Image.open(buffer) as decoded:
                result = decoded.convert("RGB")
        else:
            raise ValueError(f"Unsupported transformation: {operation}")
    return result


def metric_bundle(truth: list[int], predicted: list[int]) -> dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=range(len(LABELS)), average=None, zero_division=0,
    )
    matrix = confusion_matrix(truth, predicted, labels=range(len(LABELS)))
    total = len(truth)
    correct = int(np.trace(matrix))
    per_class = {}
    for index, label in enumerate(LABELS):
        per_class[label] = {
            "precision": float(precision[index]), "recall": float(recall[index]),
            "f1": float(f1[index]), "support": int(support[index]),
            "correct": int(matrix[index, index]),
        }
    return {
        "total": total, "correct": correct, "accuracy": correct / total,
        "precision_macro": float(np.mean(precision)),
        "recall_macro": float(np.mean(recall)), "f1_macro": float(np.mean(f1)),
        "per_class": per_class,
        "confusion_matrix": {"labels": LABELS, "rows_actual_columns_predicted": matrix.tolist()},
    }


def save_plots(results: dict[str, Any], strongest_combined: str) -> None:
    names = [key for key, _, _ in CONDITIONS]
    labels = [display for _, display, _ in CONDITIONS]
    x = np.arange(len(names))

    def bar_plot(values, ylabel: str, title: str, filename: str, ylim=(0, 100), color="#2563eb"):
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(x, values, color=color)
        ax.set_xticks(x, labels, rotation=24, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(*ylim)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=8)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / filename, dpi=180)
        plt.close(fig)

    bar_plot([results[n]["metrics"]["accuracy"] * 100 for n in names], "Accuracy (%)",
             "Accuracy by Transformation Condition", "accuracy_comparison.png")

    fig, ax = plt.subplots(figsize=(12, 6))
    width = 0.25
    for offset, metric, color in [(-width, "precision_macro", "#2563eb"), (0, "recall_macro", "#0f766e"), (width, "f1_macro", "#7c3aed")]:
        values = [results[n]["metrics"][metric] * 100 for n in names]
        ax.bar(x + offset, values, width, label=metric.replace("_macro", "").title(), color=color)
    ax.set_xticks(x, labels, rotation=24, ha="right")
    ax.set_ylabel("Macro score (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Macro Precision, Recall, and F1 by Transformation")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "precision_recall_f1.png", dpi=180)
    plt.close(fig)

    bar_plot([results[n]["prediction_consistency"] * 100 for n in names], "Prediction consistency (%)",
             "Prediction Consistency with Original Images", "prediction_consistency.png", color="#0f766e")
    drops = [results[n]["accuracy_drop_percentage_points"] for n in names]
    low = min(-2, min(drops) - 1)
    high = max(2, max(drops) + 1)
    bar_plot(drops, "Accuracy drop (percentage points)", "Accuracy Change Relative to Original",
             "accuracy_drop.png", ylim=(low, high), color="#d97706")

    for condition, filename, title in [
        ("original", "confusion_matrix_original.png", "Confusion Matrix: Original"),
        (strongest_combined, "confusion_matrix_strongest_combined.png", f"Confusion Matrix: {dict((key, name) for key, name, _ in CONDITIONS)[strongest_combined]}"),
    ]:
        matrix = np.asarray(results[condition]["metrics"]["confusion_matrix"]["rows_actual_columns_predicted"])
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


def write_report(data: dict[str, Any]) -> None:
    display = {key: name for key, name, _ in CONDITIONS}
    rows = []
    for key, _, _ in CONDITIONS:
        item = data["results"][key]
        metrics = item["metrics"]
        rows.append(
            f"| {display[key]} | {metrics['accuracy']:.4f} | {metrics['precision_macro']:.4f} | "
            f"{metrics['recall_macro']:.4f} | {metrics['f1_macro']:.4f} | "
            f"{item['prediction_consistency']:.4f} | {item['mean_confidence_change'] * 100:+.2f} pp | "
            f"{item['mean_absolute_confidence_change'] * 100:.2f} pp | "
            f"{item['accuracy_drop_percentage_points']:+.2f} pp |"
        )
    per_class_lines = []
    for condition, _, _ in CONDITIONS:
        classes = data["results"][condition]["metrics"]["per_class"]
        per_class_lines.append(f"### {display[condition]}")
        per_class_lines.append("| Ground-truth class | Precision | Recall | F1 | Support | Correct |")
        per_class_lines.append("|---|---:|---:|---:|---:|---:|")
        for class_name in LABELS:
            item = classes[class_name]
            per_class_lines.append(f"| {class_name} | {item['precision']:.4f} | {item['recall']:.4f} | {item['f1']:.4f} | {item['support']} | {item['correct']} |")
        per_class_lines.append("")
    counts = data["sample_class_counts"]
    report = f'''# Robustness Evaluation Report

## 1. Objective

Measure how the existing image classifier's predictions change after deterministic cropping, resizing, JPEG compression, and combinations of those operations. Ground-truth labels are held fixed because these operations change image encoding or framing, not the source label.

## 2. Experimental setup

- Run time: {data['run_timestamp_utc']}
- Checkpoint: `{data['model_checkpoint']}` (SHA-256 `{data['checkpoint_sha256']}`)
- Inference code: `src/13_predict_efficientnet_image.py`; the robustness runner calls its shared `predict_image_details` function with one cached predictor.
- Dataset manifest and split: `{data['dataset_manifest']}`, existing `test` split only.
- Available test images: {data['available_test_images']}; evaluated sample: {data['sample_size']} (seed {data['seed']}, class-stratified).
- Evaluated sample counts: {json.dumps(counts)}.
- Original test files were read only. Derived transformed images were saved under `results/robustness/transformed/` as PNGs, including the JPEG-decoded pixels for compression conditions.
- Class metrics: macro precision/recall/F1 across all three classifier classes; class-specific precision/recall/F1 are also listed below.

## 3. Transformation definitions

All operations use RGB images and are deterministic. Operations in combined conditions run in this order: crop, resize degradation, JPEG encode/decode.

- **Crop:** centered crop retaining {data['transformations']['crop_retained_fraction']:.0%} of each dimension, then bicubic resize back to the input dimensions.
- **Resize:** bicubic downsampling to {data['transformations']['resize_scale']:.0%} of each dimension, then bicubic upsampling back to input dimensions.
- **JPEG compression:** encode and decode at quality {data['transformations']['jpeg_quality']}.
- Combined conditions apply the corresponding operations in the order above.

The model's original torchvision resize/center-crop/tensor normalization and shared 224x224 FFT, Haar wavelet, and residual feature extraction run after each condition's transformation, exactly as they do in the Flask predictor.

## 4. Dataset and test split

The project manifest identifies the sources as Real from COCO val2017, AI-Generated from Tiny-GenImage, and Manipulated from CASIA Tp. This run uses only rows marked `test`; it does not substitute uploads, training images, or another dataset. The prepared split has {data['available_test_images']} images ({json.dumps(data['available_class_counts'])}). The run sampled {data['sample_size']} images reproducibly to limit duplicate output files.

## 5. Model

The evaluated model is the existing three-class EfficientNet-B0 fused classifier: 1,280 deep features plus 17 handcrafted FFT/wavelet/noise features (1,297 fused inputs), with checkpoint training-fitted feature scaling. The classes are Real, AI-Generated, and Manipulated. This is not a binary Authentic/Manipulated classifier; Real is the dataset's authentic class.

## 6. Evaluation metrics

Accuracy is overall correct predictions divided by sample count. Precision, recall, and F1 in the summary are macro averages across all three classes. Prediction consistency is the fraction whose transformed predicted class equals that same image's original prediction. Confidence change is the mean signed difference in top-class confidence (transformed minus original), shown as percentage points; mean absolute confidence change is also shown. Accuracy drop is original accuracy minus condition accuracy, in percentage points; a negative value means accuracy increased.

## 7. Results

| Transformation | Accuracy | Precision | Recall | F1 | Prediction Consistency | Mean Confidence Change (pp) | Mean Absolute Confidence Change (pp) | Accuracy Drop (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

### Per-class results

{chr(10).join(per_class_lines)}

## 8. Interpretation

The measured consistency and accuracy values describe this checkpoint on this test sample and the stated transformation strengths. They do not establish a guarantee for other image sources, crop locations, resize kernels, compression qualities, or manipulation methods. The strongest combined condition by observed accuracy was **{display[data['strongest_combined_condition']]}** ({data['results'][data['strongest_combined_condition']]['metrics']['accuracy']:.4f} accuracy).

## 9. Limitations

- The test sample is {data['sample_size']} of {data['available_test_images']} available test images; the estimate has sampling uncertainty.
- The prepared dataset consists of three source-specific classes, and results may reflect dataset/source characteristics.
- Each transform uses one fixed deterministic strength and centered crop; real-world transformations vary.
- The labels and class mapping come from the existing project manifest. No independent relabeling or forensic review was performed.
- Metrics measure classification behavior, not proof that a transformed image itself is authentic or manipulated.

## 10. Conclusion

This experiment evaluates the existing EfficientNet-B0 fused model under the defined transformations. Use the reported measurements to describe observed performance on this split and sample; do not claim general robustness beyond this evaluation.
'''
    (OUTPUT_DIR / "ROBUSTNESS_REPORT.md").write_text(report, encoding="utf-8")


def evaluate(args) -> dict[str, Any]:
    rows = read_test_rows()
    selected = stratified_sample(rows, args.sample_size, args.seed)
    checkpoint = ROOT / "results/efficientnet_fused_best_model.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint}")
    import hashlib
    digest = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    predictor = load_predictor()
    ground_truth = [row["label"] for row in selected]
    condition_records: dict[str, list[dict[str, Any]]] = {}
    original_predictions: list[int] = []
    original_confidences: list[float] = []
    prediction_rows: list[dict[str, Any]] = []

    for condition_key, display_name, operations in CONDITIONS:
        print(f"Evaluating {display_name}: {len(selected)} images", flush=True)
        condition_images = []
        for row in selected:
            with Image.open(row["filepath"]) as source:
                source_rgb = source.convert("RGB")
            transformed = transform_image(source_rgb, operations, args) if operations else source_rgb
            if operations:
                image_dir = OUTPUT_DIR / "transformed" / condition_key / row["class_name"]
                image_dir.mkdir(parents=True, exist_ok=True)
                transformed.save(image_dir / f"{row['filepath'].stem}.png", format="PNG", optimize=False)
            condition_images.append(transformed)
        results_for_condition = []
        for start in range(0, len(condition_images), 32):
            batch_results = predict_images_details(condition_images[start:start + 32], predictor=predictor)
            results_for_condition.extend(batch_results)
            processed = min(start + 32, len(condition_images))
            if processed % 50 < 32 or processed == len(condition_images):
                print(f"  {processed}/{len(selected)}", flush=True)
        predictions, confidences, records = [], [], []
        for row, result in zip(selected, results_for_condition):
            pred_index = LABELS.index(result["predicted_class"])
            predictions.append(pred_index)
            confidences.append(result["confidence"])
            records.append(result)
            if condition_key == "original":
                original_predictions.append(pred_index)
                original_confidences.append(result["confidence"])
            prediction_rows.append({
                "condition": condition_key, "display_name": display_name,
                "image": str(row["filepath"].relative_to(ROOT)), "ground_truth": row["class_name"],
                "predicted_class": result["predicted_class"], "confidence": result["confidence"],
                **{f"probability_{name}": result["probabilities"][name] for name in LABELS},
            })
        condition_records[condition_key] = [{"pred": p, "confidence": c, **r} for p, c, r in zip(predictions, confidences, records)]

    results = {}
    baseline_accuracy = None
    for condition_key, display_name, _ in CONDITIONS:
        records = condition_records[condition_key]
        pred = [record["pred"] for record in records]
        conf = [record["confidence"] for record in records]
        metrics = metric_bundle(ground_truth, pred)
        if condition_key == "original":
            baseline_accuracy = metrics["accuracy"]
            consistency = 1.0
            confidence_delta = 0.0
            confidence_abs_delta = 0.0
        else:
            consistency = sum(a == b for a, b in zip(original_predictions, pred)) / len(pred)
            deltas = [new - old for new, old in zip(conf, original_confidences)]
            confidence_delta = float(np.mean(deltas))
            confidence_abs_delta = float(np.mean(np.abs(deltas)))
        results[condition_key] = {
            "display_name": display_name, "metrics": metrics,
            "prediction_consistency": consistency,
            "mean_confidence_change": confidence_delta,
            "mean_absolute_confidence_change": confidence_abs_delta,
            "accuracy_drop_percentage_points": 0.0 if condition_key == "original" else (baseline_accuracy - metrics["accuracy"]) * 100,
        }

    combined_keys = [key for key, _, operations in CONDITIONS if len(operations) >= 2]
    strongest_combined = min(combined_keys, key=lambda key: results[key]["metrics"]["accuracy"])
    class_counts = {name: sum(row["class_name"] == name for row in selected) for name in LABELS}
    available_counts = {name: sum(row["class_name"] == name for row in rows) for name in LABELS}
    from datetime import datetime, timezone
    data = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_checkpoint": str(checkpoint.relative_to(ROOT)), "checkpoint_sha256": digest.hexdigest(),
        "dataset_manifest": str(MANIFEST_PATH.relative_to(ROOT)), "test_split": "test",
        "available_test_images": len(rows), "available_class_counts": available_counts,
        "sample_size": len(selected), "sample_class_counts": class_counts, "seed": args.seed,
        "classes": LABELS, "metric_averaging": "macro",
        "transformations": {"crop_retained_fraction": args.crop_fraction, "resize_scale": args.resize_scale,
                            "jpeg_quality": args.jpeg_quality, "combined_operation_order": ["crop", "resize", "jpeg"]},
        "strongest_combined_condition": strongest_combined,
        "results": results,
    }
    (OUTPUT_DIR / "robustness_results.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    with (OUTPUT_DIR / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)
    save_plots(results, strongest_combined)
    write_report(data)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=300,
                        help="Stratified test images to evaluate (default: 300; use 0 for all available test rows).")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed (default: 42).")
    parser.add_argument("--crop-fraction", type=float, default=0.9,
                        help="Fraction retained along each dimension for centered crop (default: 0.9).")
    parser.add_argument("--resize-scale", type=float, default=0.75,
                        help="Intermediate resize fraction before restoring original dimensions (default: 0.75).")
    parser.add_argument("--jpeg-quality", type=int, default=60,
                        help="JPEG quality for compression condition (default: 60).")
    args = parser.parse_args()
    if not 0 < args.crop_fraction < 1:
        parser.error("--crop-fraction must be between 0 and 1.")
    if not 0 < args.resize_scale < 1:
        parser.error("--resize-scale must be between 0 and 1.")
    if not 1 <= args.jpeg_quality <= 95:
        parser.error("--jpeg-quality must be between 1 and 95.")
    if args.sample_size < 0:
        parser.error("--sample-size must be 0 (all rows) or positive.")
    try:
        data = evaluate(args)
    except Exception as exc:
        print(f"Robustness evaluation failed: {exc}", file=sys.stderr)
        raise
    print("\nComplete robustness results (macro precision/recall/F1):")
    for key, display_name, _ in CONDITIONS:
        item = data["results"][key]
        metric = item["metrics"]
        print(f"{display_name} | accuracy={metric['accuracy']:.4f} ({metric['correct']}/{metric['total']}) | "
              f"P={metric['precision_macro']:.4f}, R={metric['recall_macro']:.4f}, F1={metric['f1_macro']:.4f} | "
              f"consistency={item['prediction_consistency']:.4f} | "
              f"confidence Δ={item['mean_confidence_change'] * 100:+.2f} pp, "
              f"mean |Δ|={item['mean_absolute_confidence_change'] * 100:.2f} pp | "
              f"accuracy drop={item['accuracy_drop_percentage_points']:+.2f} pp")
        for class_name in LABELS:
            class_result = metric["per_class"][class_name]
            print(f"  {class_name}: precision={class_result['precision']:.4f}, "
                  f"recall={class_result['recall']:.4f}, F1={class_result['f1']:.4f}, "
                  f"support={class_result['support']}, correct={class_result['correct']}")
        print("  confusion matrix (actual rows, predicted columns):")
        for matrix_row in metric["confusion_matrix"]["rows_actual_columns_predicted"]:
            print(f"    {matrix_row}")
    print(f"\nSaved evaluation artifacts under: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
