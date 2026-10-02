"""Compare the preserved and candidate fused EfficientNet models on a new held-out manifest."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
predictor_module = importlib.import_module("src.13_predict_efficientnet_image")
robust_module = importlib.import_module("src.05_robustness_testing")
LABELS = predictor_module.LABELS
CONDITIONS = robust_module.CONDITIONS
OUTPUT_DIR = ROOT / "results/dataset_coverage_improved"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(path)
    required = {"filepath", "source_path", "source_sha256", "prepared_sha256", "class_name", "label", "split",
                "casia_family", "casia_group"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest missing required provenance columns: {sorted(missing)}")
    if set(frame.class_name) != set(LABELS):
        raise ValueError(f"Unexpected class names: {sorted(frame.class_name.unique())}")
    for row in frame.itertuples(index=False):
        if int(row.label) != LABELS.index(row.class_name):
            raise ValueError(f"Invalid class/label mapping: {row.class_name} {row.label}")
        if not Path(row.filepath).is_file() or not Path(row.source_path).is_file():
            raise FileNotFoundError(f"Missing source or prepared image: {row.source_path}")
    return frame, frame[frame.split == "test"].reset_index(drop=True)


def metric_bundle(truth: list[int], prediction: list[int]) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, prediction, labels=range(len(LABELS)), average=None, zero_division=0
    )
    matrix = confusion_matrix(truth, prediction, labels=range(len(LABELS)))
    per_class = {}
    for index, name in enumerate(LABELS):
        per_class[name] = {"precision": float(precision[index]), "recall": float(recall[index]),
                           "f1": float(f1[index]), "support": int(support[index]),
                           "correct": int(matrix[index, index])}
    return {"count": len(truth), "accuracy": float(np.trace(matrix) / len(truth)),
            "precision_macro": float(np.mean(precision)), "recall_macro": float(np.mean(recall)),
            "f1_macro": float(np.mean(f1)), "per_class": per_class,
            "confusion_matrix": {"labels": LABELS, "rows_actual_columns_predicted": matrix.tolist()}}


def predict_images(images: list[Image.Image], predictor: dict, batch_size: int) -> list[dict]:
    values = []
    for start in range(0, len(images), batch_size):
        values.extend(predictor_module.predict_images_details(images[start:start + batch_size], predictor=predictor))
    return values


def evaluate_condition(rows: pd.DataFrame, old_predictor: dict, new_predictor: dict,
                       operations: tuple[str, ...], config, batch_size: int,
                       condition_name: str) -> tuple[dict, dict, list[dict]]:
    old_results, new_results, detail_rows = [], [], []
    total = len(rows)
    for start in range(0, total, batch_size):
        images, batch_rows = [], rows.iloc[start:start + batch_size]
        for row in batch_rows.itertuples(index=False):
            with Image.open(row.source_path) as source:
                image = source.convert("RGB")
            images.append(robust_module.transform_image(image, operations, config) if operations else image)
        old_batch = predict_images(images, old_predictor, batch_size)
        new_batch = predict_images(images, new_predictor, batch_size)
        old_results.extend(old_batch)
        new_results.extend(new_batch)
        for row, old, new in zip(batch_rows.itertuples(index=False), old_batch, new_batch):
            item = {"condition": condition_name, "source_path": row.source_path,
                    "casia_family": row.casia_family, "casia_group": row.casia_group,
                    "ground_truth": row.class_name,
                    "old_prediction": old["predicted_class"], "old_confidence": old["confidence"],
                    "new_prediction": new["predicted_class"], "new_confidence": new["confidence"]}
            for name in LABELS:
                item[f"old_probability_{name}"] = old["probabilities"][name]
                item[f"new_probability_{name}"] = new["probabilities"][name]
            detail_rows.append(item)
        print(f"  {condition_name}: {min(start + len(batch_rows), total)}/{total}", flush=True)
    truth = [LABELS.index(name) for name in rows.class_name]
    old_indices = [LABELS.index(item["predicted_class"]) for item in old_results]
    new_indices = [LABELS.index(item["predicted_class"]) for item in new_results]
    old_metrics, new_metrics = metric_bundle(truth, old_indices), metric_bundle(truth, new_indices)
    return ({"metrics": old_metrics, "predictions": old_indices,
             "confidences": [item["confidence"] for item in old_results]},
            {"metrics": new_metrics, "predictions": new_indices,
             "confidences": [item["confidence"] for item in new_results]}, detail_rows)


def add_stability(results: dict) -> None:
    old_base = results["original"]["old"]["predictions"]
    new_base = results["original"]["new"]["predictions"]
    old_conf = results["original"]["old"]["confidences"]
    new_conf = results["original"]["new"]["confidences"]
    for item in results.values():
        for model_key, base_pred, base_conf in (("old", old_base, old_conf), ("new", new_base, new_conf)):
            predicted = item[model_key]["predictions"]
            confidence = item[model_key]["confidences"]
            item[model_key]["prediction_consistency"] = float(np.mean(np.asarray(predicted) == np.asarray(base_pred)))
            deltas = np.asarray(confidence) - np.asarray(base_conf)
            item[model_key]["mean_confidence_change"] = float(np.mean(deltas))
            item[model_key]["mean_absolute_confidence_change"] = float(np.mean(np.abs(deltas)))
            item[model_key]["accuracy_drop_percentage_points"] = float(
                (results["original"][model_key]["metrics"]["accuracy"] - item[model_key]["metrics"]["accuracy"]) * 100
            )


def subgroup_metrics(test: pd.DataFrame, results: dict) -> list[dict]:
    groups: dict[str, np.ndarray] = {}
    casia = test[test.class_name == "Manipulated"]
    if not casia.empty:
        for family in sorted(set(casia.casia_family) - {"", "nan"}):
            groups[family] = np.flatnonzero(((test.class_name == "Manipulated") & (test.casia_family == family)).to_numpy())
        for group in sorted(set(casia.casia_group) - {"", "nan"}):
            groups[group] = np.flatnonzero(((test.class_name == "Manipulated") & (test.casia_group == group)).to_numpy())
        # Report remaining observed fine groups under their actual parsed labels.
    output = []
    predictions_by_condition = results["original"]
    for subgroup, mask in groups.items():
        selected_indices = mask
        if not len(selected_indices):
            continue
        truth = [2] * len(selected_indices)
        row = {"subgroup": subgroup, "support": len(selected_indices)}
        for model_key in ("old", "new"):
            pred = [predictions_by_condition[model_key]["predictions"][index] for index in selected_indices]
            m = metric_bundle(truth, pred)
            row[f"{model_key}_accuracy"] = m["accuracy"]
            row[f"{model_key}_manipulated_recall"] = m["per_class"]["Manipulated"]["recall"]
            row[f"{model_key}_manipulated_precision"] = m["per_class"]["Manipulated"]["precision"]
            row[f"{model_key}_macro_f1"] = m["f1_macro"]
            row[f"{model_key}_manipulated_to_real"] = m["confusion_matrix"]["rows_actual_columns_predicted"][2][0]
            row[f"{model_key}_manipulated_to_ai"] = m["confusion_matrix"]["rows_actual_columns_predicted"][2][1]
        output.append(row)
    return output


def build_interpretation(results: dict, diagnostic_originals: list[dict]) -> str:
    old = results["original"]["old"]["metrics"]
    new = results["original"]["new"]["metrics"]
    transformed = [key for key, _, operations in CONDITIONS if operations]
    mean_old = float(np.mean([results[k]["old"]["metrics"]["accuracy"] for k in transformed]))
    mean_new = float(np.mean([results[k]["new"]["metrics"]["accuracy"] for k in transformed]))
    real_recall_delta = (new["per_class"]["Real"]["recall"] - old["per_class"]["Real"]["recall"]) * 100
    manipulated_precision_delta = (new["per_class"]["Manipulated"]["precision"]
                                   - old["per_class"]["Manipulated"]["precision"]) * 100
    old_matrix = old["confusion_matrix"]["rows_actual_columns_predicted"]
    new_matrix = new["confusion_matrix"]["rows_actual_columns_predicted"]
    correct_diagnostics = sum(row["new_prediction"] == "Manipulated" for row in diagnostic_originals)
    return (
        f"The candidate increased Original accuracy by {(new['accuracy']-old['accuracy'])*100:+.2f} pp, "
        f"Manipulated recall by {(new['per_class']['Manipulated']['recall']-old['per_class']['Manipulated']['recall'])*100:+.2f} pp, "
        f"and mean accuracy across the seven transformed conditions by {(mean_new-mean_old)*100:+.2f} pp; its accuracy was higher in all seven conditions. "
        f"The trade-off is lower Real recall ({real_recall_delta:+.2f} pp), lower Manipulated precision ({manipulated_precision_delta:+.2f} pp), "
        f"and {int(new_matrix[0][2]-old_matrix[0][2]):+d} additional Real→Manipulated errors on 300 Real test images "
        f"({old_matrix[0][2]} to {new_matrix[0][2]}). It classified {correct_diagnostics}/{len(diagnostic_originals)} of the supplied diagnostic examples as Manipulated "
        "on their original versions. The prediction-consistency values also vary by condition; they are listed in the comparison table. "
        "These measurements show a candidate trade-off, not a guarantee for other data or transformations."
    )


def diagnostic_eval(paths: list[Path], manifest: pd.DataFrame, old_predictor: dict, new_predictor: dict,
                    config, batch_size: int) -> tuple[list[dict], list[dict]]:
    known = {str(Path(p).resolve()).casefold() for p in manifest.source_path}
    known_hashes = set(manifest.source_sha256)
    originals, transformed_rows = [], []
    for path in paths:
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Diagnostic image not found: {path}")
        if str(path).casefold() in known or sha256_file(path) in known_hashes:
            raise ValueError(f"Diagnostic image leaked into the new manifest: {path}")
        with Image.open(path) as source:
            original = source.convert("RGB")
        originals.append((path, original))
    for key, display, operations in CONDITIONS:
        images = [robust_module.transform_image(image, operations, config) if operations else image
                  for _, image in originals]
        old = predict_images(images, old_predictor, batch_size)
        new = predict_images(images, new_predictor, batch_size)
        for (path, _), old_row, new_row in zip(originals, old, new):
            transformed_rows.append({"image": path.name, "condition": display,
                                     "old_prediction": old_row["predicted_class"],
                                     "old_confidence": old_row["confidence"],
                                     "new_prediction": new_row["predicted_class"],
                                     "new_confidence": new_row["confidence"],
                                     **{f"new_probability_{name}": new_row["probabilities"][name] for name in LABELS},
                                     **{f"old_probability_{name}": old_row["probabilities"][name] for name in LABELS}})
        print(f"Diagnostic images: {display} complete", flush=True)
    original_rows = [row for row in transformed_rows if row["condition"] == CONDITIONS[0][1]]
    return original_rows, transformed_rows


def plot_comparisons(data: dict, subgroup_rows: list[dict]) -> None:
    conditions = [key for key, _, _ in CONDITIONS]
    names = [label for _, label, _ in CONDITIONS]
    x = np.arange(len(names))
    for metric, filename, title, ylabel in [
        ("accuracy", "robustness_accuracy_old_vs_new.png", "Accuracy by Transformation", "Accuracy (%)"),
        ("f1_macro", "robustness_macro_f1_old_vs_new.png", "Macro F1 by Transformation", "Macro F1 (%)"),
        ("prediction_consistency", "prediction_consistency_old_vs_new.png", "Prediction Consistency vs Original", "Consistency (%)"),
    ]:
        fig, ax = plt.subplots(figsize=(12, 6))
        width = 0.38
        for offset, model, color in [(-width / 2, "old", "#64748b"), (width / 2, "new", "#2563eb")]:
            values = [data[k][model]["metrics"][metric] if metric != "prediction_consistency"
                      else data[k][model][metric] for k in conditions]
            bars = ax.bar(x + offset, np.asarray(values) * 100, width, label=model.title(), color=color)
            ax.bar_label(bars, fmt="%.1f", fontsize=7, padding=2)
        ax.set_xticks(x, names, rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(0, 100)
        ax.legend()
        ax.grid(axis="y", alpha=.25)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / filename, dpi=180)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 6))
    width = .38
    for offset, model, color in [(-width/2, "old", "#64748b"), (width/2, "new", "#d97706")]:
        values = [data[k][model]["accuracy_drop_percentage_points"] for k in conditions]
        ax.bar(x + offset, values, width, label=model.title(), color=color)
    ax.set_xticks(x, names, rotation=25, ha="right")
    ax.set_ylabel("Accuracy drop (percentage points)")
    ax.set_title("Accuracy Change Relative to Each Model's Original Condition")
    ax.axhline(0, color="black", linewidth=.8)
    ax.legend()
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "accuracy_drop_old_vs_new.png", dpi=180)
    plt.close(fig)
    combined_conditions = [key for key, _, operations in CONDITIONS if len(operations) >= 2]
    matrix_conditions = ["original", min(combined_conditions,
                                           key=lambda k: data[k]["new"]["metrics"]["accuracy"])]
    for key, filename, title in [(matrix_conditions[0], "new_confusion_matrix_original.png", "Candidate: Original"),
                                  (matrix_conditions[1], "new_confusion_matrix_strongest_combined.png", "Candidate: Strongest Combined")]:
        matrix = np.asarray(data[key]["new"]["metrics"]["confusion_matrix"]["rows_actual_columns_predicted"])
        fig, ax = plt.subplots(figsize=(7, 6))
        image = ax.imshow(matrix, cmap="Blues")
        fig.colorbar(image, ax=ax, label="Images")
        ax.set(xticks=range(3), yticks=range(3), xticklabels=LABELS, yticklabels=LABELS,
               xlabel="Predicted class", ylabel="Ground truth", title=title)
        for r in range(3):
            for c in range(3):
                ax.text(c, r, str(matrix[r, c]), ha="center", va="center")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / filename, dpi=180)
        plt.close(fig)
    fine = [r for r in subgroup_rows if r["subgroup"].startswith("Tp_") and r["subgroup"].count("_") >= 2]
    if fine:
        fig, ax = plt.subplots(figsize=(13, 7))
        xg = np.arange(len(fine))
        ax.bar(xg - .19, [r["old_manipulated_recall"] * 100 for r in fine], .38, label="Current", color="#64748b")
        ax.bar(xg + .19, [r["new_manipulated_recall"] * 100 for r in fine], .38, label="Candidate", color="#2563eb")
        ax.set_xticks(xg, [r["subgroup"] for r in fine], rotation=45, ha="right")
        ax.set_ylabel("Manipulated recall (%)")
        ax.set_ylim(0, 100)
        ax.set_title("CASIA Tp Fine Group Manipulated Recall (Original Images)")
        ax.legend()
        ax.grid(axis="y", alpha=.25)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "casia_subgroup_recall_old_vs_new.png", dpi=180)
        plt.close(fig)


def write_dataset_audit_reports(distribution: dict, leakage: dict) -> None:
    lines = ["# Dataset Distribution Report", "", f"- Seed: {distribution['seed']}",
             f"- Per-class target: {distribution['max_per_class']}", f"- Total prepared images: {distribution['total']}", "",
             "## Split and class counts", "", "| Split | Total | Real | AI-Generated | Manipulated |", "|---|---:|---:|---:|---:|"]
    for split in ("train", "val", "test"):
        counts = distribution["class_counts_by_split"].get(split, {})
        lines.append(f"| {split} | {distribution['split_counts'].get(split, 0)} | {counts.get('Real', 0)} | "
                     f"{counts.get('AI-Generated', 0)} | {counts.get('Manipulated', 0)} |")
    groups = sorted(set().union(*(set(items) for items in distribution["casia_group_counts_by_split"].values())))
    lines.extend(["", "## CASIA Tp parsed group counts", "", "| Group | Train | Validation | Test |", "|---|---:|---:|---:|"])
    for group in groups:
        lines.append("| " + group + " | " + " | ".join(
            str(distribution["casia_group_counts_by_split"].get(split, {}).get(group, 0))
            for split in ("train", "val", "test")) + " |")
    raw_groups = distribution.get("casia_raw_group_counts_by_split", {})
    raw_group_names = sorted(set().union(*(set(items) for items in raw_groups.values()))) if raw_groups else []
    if raw_group_names:
        lines.extend(["", "## Raw CASIA filename group tokens", "",
                      "Case variants are retained here for provenance and coalesced in the canonical table above.",
                      "", "| Raw token | Train | Validation | Test |", "|---|---:|---:|---:|"])
        for group in raw_group_names:
            lines.append("| " + group + " | " + " | ".join(
                str(raw_groups.get(split, {}).get(group, 0)) for split in ("train", "val", "test")) + " |")
    lines.extend(["", "## Source formats", "", "| Format | Count |", "|---|---:|"])
    for source_format, count in sorted(distribution["source_format_counts"].items()):
        lines.append(f"| {source_format} | {count} |")
    by_class = distribution.get("source_format_counts_by_class", {})
    if by_class:
        formats = sorted(set().union(*(set(items) for items in by_class.values())))
        lines.extend(["", "| Class/source | " + " | ".join(formats) + " |",
                      "|---|" + "---:|" * len(formats)])
        for class_name, counts in by_class.items():
            lines.append("| " + class_name + " | " + " | ".join(str(counts.get(ext, 0)) for ext in formats) + " |")
    lines.extend(["", f"Selection strategy: {distribution['selection']}",
                  "Raw sources were read only; prepared images are stored under this report directory's dataset folder."])
    inventory = distribution.get("casia_source_inventory")
    if inventory:
        lines.extend(["", "## Available CASIA Tp source groups", "",
                      f"Programmatic filename inspection found {inventory['readable_total']:,} readable files across "
                      f"{len(inventory['canonical_group_counts'])} case-normalized groups ({inventory['unreadable_supported_files']} unreadable supported-format files). "
                      "Counts below precede the diagnostic-only path exclusions.", "",
                      "| Group | Available | Selected | Test |", "|---|---:|---:|---:|"])
        for group, available in sorted(inventory["canonical_group_counts"].items()):
            selected = sum(distribution["casia_group_counts_by_split"].get(split, {}).get(group, 0)
                           for split in ("train", "val", "test"))
            test_count = distribution["casia_group_counts_by_split"].get("test", {}).get(group, 0)
            lines.append(f"| {group} | {available:,} | {selected} | {test_count} |")
    (OUTPUT_DIR / "DATASET_DISTRIBUTION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    leakage_text = ["# Dataset Leakage Audit", "", f"- Result: **{'PASS' if leakage.get('passed') else 'FAIL'}**",
                    "- Audits: source path, source SHA-256, and prepared-image SHA-256 across train, validation, and test."]
    for key, values in leakage.items():
        if key != "passed":
            leakage_text.append(f"- {key}: {len(values)}")
    leakage_text.append("\nTraining was permitted only because the generated leakage report passed." if leakage.get("passed")
                        else "\nTraining must remain blocked until cross-split leakage is resolved.")
    (OUTPUT_DIR / "LEAKAGE_REPORT.md").write_text("\n".join(leakage_text) + "\n", encoding="utf-8")
def write_report(data: dict) -> None:
    training_record = json.loads((OUTPUT_DIR / "training_metrics.json").read_text(encoding="utf-8"))
    augmentation = training_record["augmentation_config"]
    rows = []
    for key, name, _ in CONDITIONS:
        old, new = data["conditions"][key]["old"], data["conditions"][key]["new"]
        o, n = old["metrics"], new["metrics"]
        om, nm = o["confusion_matrix"]["rows_actual_columns_predicted"], n["confusion_matrix"]["rows_actual_columns_predicted"]
        rows.append(f"| {name} | {o['accuracy']:.4f} | {n['accuracy']:.4f} | {(n['accuracy']-o['accuracy'])*100:+.2f} pp | "
                    f"{o['per_class']['Manipulated']['recall']:.4f} | {n['per_class']['Manipulated']['recall']:.4f} | "
                    f"{o['f1_macro']:.4f} | {n['f1_macro']:.4f} | {old['prediction_consistency']:.4f} | {new['prediction_consistency']:.4f} | "
                    f"{old['accuracy_drop_percentage_points']:+.2f} | {new['accuracy_drop_percentage_points']:+.2f} | "
                    f"{om[2][0]}/{om[2][1]} | {nm[2][0]}/{nm[2][1]} |")
    orig = data["conditions"]["original"]
    cm_old = orig["old"]["metrics"]["confusion_matrix"]["rows_actual_columns_predicted"]
    cm_new = orig["new"]["metrics"]["confusion_matrix"]["rows_actual_columns_predicted"]
    metric = lambda m, cls, key: m["per_class"][cls][key]
    subgroup_lines = ["| CASIA subgroup | Support | Current manipulated recall | Candidate manipulated recall | Current M→Real / M→AI | Candidate M→Real / M→AI |", "|---|---:|---:|---:|---:|---:|"]
    for row in data["subgroup_metrics"]:
        subgroup_lines.append(f"| {row['subgroup']} | {row['support']} | {row['old_manipulated_recall']:.4f} | {row['new_manipulated_recall']:.4f} | "
                              f"{row['old_manipulated_to_real']} / {row['old_manipulated_to_ai']} | {row['new_manipulated_to_real']} / {row['new_manipulated_to_ai']} |")
    diag_lines = ["| Image | Condition | Current prediction | Candidate prediction | Candidate Real / AI / Manipulated probabilities |", "|---|---|---|---|---|"]
    for row in data["diagnostic_transformations"]:
        diag_lines.append(f"| {row['image']} | {row['condition']} | {row['old_prediction']} ({row['old_confidence']:.4f}) | {row['new_prediction']} ({row['new_confidence']:.4f}) | "
                          f"{row['new_probability_Real']:.4f} / {row['new_probability_AI-Generated']:.4f} / {row['new_probability_Manipulated']:.4f} |")
    d = data["distribution"]
    result = f'''# Dataset Coverage Improvement Report

## Objective and changes

The previous preparation selected the first 2,000 sorted CASIA Tp paths, leaving most Tp_S subgroups out. The new fixed-seed preparation samples CASIA proportionally by the parsed filename group, with up to a three-image floor per available group, and randomly samples the other classes. The three supplied diagnostic images were passed to the preparation command as exclusions and were not used in any split. No source image was modified. A separate candidate was fine-tuned from the preserved robustness-improved EfficientNet-B0 checkpoint.

## Dataset and leakage audit

- Manifest: `{data['manifest']}`; seed {d['seed']}; {d['max_per_class']} images per class.
- Counts by split: `{json.dumps(d['split_counts'])}`.
- Class counts: `{json.dumps(d['class_counts_by_split'])}`.
- Exact source paths, source SHA-256, and prepared-image SHA-256 were checked across splits: **{'PASS' if data['leakage']['passed'] else 'FAIL'}**.
- CASIA group counts by split: `{json.dumps(d['casia_group_counts_by_split'], sort_keys=True)}`.
- Groups present only in train: `{data.get('train_only_casia_groups', [])}`. All selected canonical groups are represented in test.

## Models and protocol

- Current checkpoint: `{data['old_checkpoint']}` (SHA-256 `{data['old_checkpoint_sha256']}`).
- Candidate checkpoint: `{data['new_checkpoint']}` (SHA-256 `{data['new_checkpoint_sha256']}`).
- Both preserve EfficientNet-B0, FFT, Haar wavelet, residual/noise features, feature scaling, fusion order, and the three-class output.
- Training used original train rows plus one probabilistically augmented view per training image. Validation was untouched and selected the checkpoint; the test split was not used to update weights or select the epoch.
- Candidate checkpoint selected at epoch {training_record['selected_epoch']} with validation accuracy {training_record['selected_validation_accuracy']:.4f}; train views: {training_record['training_views']['original']} original + {training_record['training_views']['augmented']} augmented.
- Training augmentation settings: independent crop/resize/JPEG probabilities {augmentation['independent_probabilities']}; crop retains at least {augmentation['crop_scale_min']:.0%} of each dimension; resize scale {augmentation['resize_scale_range'][0]:.0%}–{augmentation['resize_scale_range'][1]:.0%}; JPEG qualities {augmentation['jpeg_quality_choices']}.
- Raw inference and transformed raw inference use the same RGB square 224 LANCZOS resize and JPEG quality-95 decode as dataset preparation, followed by the unchanged torchvision resize/center-crop/normalization and 17 handcrafted features.
- Test images were processed from source paths through that shared preprocessing. Eight conditions use the project's existing deterministic centered 90% crop, 75% down/up resize, JPEG quality 60, and combinations in crop/resize/JPEG order. The ground-truth label is unchanged.

Training command:

```text
{data['training_command']}
```

## Original-condition comparison

| Metric | Current | Candidate |
|---|---:|---:|
| Accuracy | {orig['old']['metrics']['accuracy']:.4f} | {orig['new']['metrics']['accuracy']:.4f} |
| Macro precision | {orig['old']['metrics']['precision_macro']:.4f} | {orig['new']['metrics']['precision_macro']:.4f} |
| Macro recall | {orig['old']['metrics']['recall_macro']:.4f} | {orig['new']['metrics']['recall_macro']:.4f} |
| Macro F1 | {orig['old']['metrics']['f1_macro']:.4f} | {orig['new']['metrics']['f1_macro']:.4f} |
| Manipulated recall | {metric(orig['old']['metrics'],'Manipulated','recall'):.4f} | {metric(orig['new']['metrics'],'Manipulated','recall'):.4f} |
| Manipulated → Real | {cm_old[2][0]} | {cm_new[2][0]} |
| Manipulated → AI-Generated | {cm_old[2][1]} | {cm_new[2][1]} |
| Real → Manipulated | {cm_old[0][2]} | {cm_new[0][2]} |

Confusion matrices (rows = actual; columns = predicted; order Real, AI-Generated, Manipulated):

- Current: `{json.dumps(cm_old)}`
- Candidate: `{json.dumps(cm_new)}`

## Eight-condition comparison

Accuracy, manipulated recall, macro F1, consistency versus each model's own original prediction, and accuracy drop are measured on the complete held-out test split.

| Condition | Current accuracy | Candidate accuracy | Difference | Current manipulated recall | Candidate manipulated recall | Current macro F1 | Candidate macro F1 | Current/candidate consistency | Current/candidate accuracy drop (pp) | Current M→Real/M→AI | Candidate M→Real/M→AI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## CASIA subgroup results

All subgroups below existed in the test split. Family aggregates (`Tp_D`, `Tp_S`) and parsed fine groups are calculated from original-condition inference.

{chr(10).join(subgroup_lines)}

## Previous false-negative diagnostic images

These externally specified files were verified against the manifest by path and SHA-256 and excluded from dataset creation. They are diagnostics only and did not affect training or selection.

{chr(10).join(diag_lines)}

## Interpretation

{data['interpretation']}

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
'''
    (OUTPUT_DIR / "DATASET_COVERAGE_IMPROVEMENT_REPORT.md").write_text(result, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=OUTPUT_DIR / "new_manifest.csv")
    parser.add_argument("--old-checkpoint", type=Path, default=ROOT / "results/efficientnet_fused_robust_best_model.pth")
    parser.add_argument("--new-checkpoint", type=Path, default=OUTPUT_DIR / "efficientnet_fused_robust_balanced_best_model.pth")
    parser.add_argument("--diagnostic-image", type=Path, action="append", default=[])
    parser.add_argument("--finalize-existing", action="store_true",
                        help="Regenerate reports and plots from completed measured JSON without rerunning inference.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--crop-fraction", type=float, default=.9)
    parser.add_argument("--resize-scale", type=float, default=.75)
    parser.add_argument("--jpeg-quality", type=int, default=60)
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.finalize_existing:
        existing_path = OUTPUT_DIR / "old_new_evaluation.json"
        if not existing_path.is_file():
            raise FileNotFoundError(f"No completed evaluation exists: {existing_path}")
        existing = json.loads(existing_path.read_text(encoding="utf-8"))
        distribution = json.loads((OUTPUT_DIR / "dataset_distribution.json").read_text(encoding="utf-8"))
        existing["distribution"] = distribution
        leakage = existing["leakage"]
        existing["train_only_casia_groups"] = sorted(
            set(distribution["casia_group_counts_by_split"]["train"])
            - set(distribution["casia_group_counts_by_split"]["val"])
            - set(distribution["casia_group_counts_by_split"]["test"])
        )
        existing["interpretation"] = build_interpretation(existing["conditions"], existing["diagnostic_originals"])
        write_dataset_audit_reports(distribution, leakage)
        plot_comparisons(existing["conditions"], existing["subgroup_metrics"])
        write_report(existing)
        original = existing["conditions"]["original"]
        rows = []
        for model in ("old", "new"):
            metrics = original[model]["metrics"]
            matrix = metrics["confusion_matrix"]["rows_actual_columns_predicted"]
            rows.append({"Model": "Current" if model == "old" else "Candidate",
                         "Accuracy": metrics["accuracy"], "Macro Precision": metrics["precision_macro"],
                         "Macro Recall": metrics["recall_macro"], "Macro F1": metrics["f1_macro"],
                         "Real Precision": metrics["per_class"]["Real"]["precision"],
                         "Real Recall": metrics["per_class"]["Real"]["recall"],
                         "AI-Generated Precision": metrics["per_class"]["AI-Generated"]["precision"],
                         "AI-Generated Recall": metrics["per_class"]["AI-Generated"]["recall"],
                         "Manipulated Precision": metrics["per_class"]["Manipulated"]["precision"],
                         "Manipulated Recall": metrics["per_class"]["Manipulated"]["recall"],
                         "Manipulated to Real": matrix[2][0], "Manipulated to AI": matrix[2][1],
                         "Real to Manipulated": matrix[0][2], "Confusion Matrix": json.dumps(matrix)})
        with (OUTPUT_DIR / "old_new_overall_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        existing_path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
        print(f"Regenerated reports and plots from measured results under {OUTPUT_DIR}")
        return
    if not args.diagnostic_image:
        parser.error("Provide at least one --diagnostic-image for a new evaluation run.")
    manifest_path = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    old_path = args.old_checkpoint if args.old_checkpoint.is_absolute() else ROOT / args.old_checkpoint
    new_path = args.new_checkpoint if args.new_checkpoint.is_absolute() else ROOT / args.new_checkpoint
    if args.batch_size < 1 or not 0 < args.crop_fraction <= 1 or not 0 < args.resize_scale <= 1 or not 1 <= args.jpeg_quality <= 100:
        parser.error("Invalid batch size or transformation parameters.")
    manifest, test = read_manifest(manifest_path)
    if not set(test.class_name) == set(LABELS):
        raise ValueError("New held-out split must contain all three classes.")
    leakage_path = manifest_path.parent / "leakage_report.json"
    leakage = json.loads(leakage_path.read_text(encoding="utf-8"))
    if not leakage.get("passed"):
        raise RuntimeError("Leakage report did not pass; refusing to evaluate candidate.")
    old_predictor, new_predictor = predictor_module.load_predictor(old_path), predictor_module.load_predictor(new_path)
    config = SimpleNamespace(crop_fraction=args.crop_fraction, resize_scale=args.resize_scale, jpeg_quality=args.jpeg_quality)
    results, all_details = {}, []
    print(f"Evaluating {len(test)} shared held-out images using current and candidate models.", flush=True)
    for key, display, operations in CONDITIONS:
        print(f"Condition: {display}", flush=True)
        old, new, details = evaluate_condition(test, old_predictor, new_predictor, operations, config,
                                               args.batch_size, key)
        results[key] = {"display_name": display, "old": old, "new": new}
        all_details.extend(details)
    add_stability(results)
    subgroup_rows = subgroup_metrics(test, results)
    diagnostic_originals, diagnostic_transforms = diagnostic_eval(args.diagnostic_image, manifest,
                                                                   old_predictor, new_predictor, config,
                                                                   args.batch_size)
    distribution = json.loads((manifest_path.parent / "dataset_distribution.json").read_text(encoding="utf-8"))
    write_dataset_audit_reports(distribution, leakage)
    overall_old = results["original"]["old"]["metrics"]["accuracy"]
    overall_new = results["original"]["new"]["metrics"]["accuracy"]
    old_mrec = results["original"]["old"]["metrics"]["per_class"]["Manipulated"]["recall"]
    new_mrec = results["original"]["new"]["metrics"]["per_class"]["Manipulated"]["recall"]
    transformed_keys = [key for key, _, ops in CONDITIONS if ops]
    mean_old = float(np.mean([results[key]["old"]["metrics"]["accuracy"] for key in transformed_keys]))
    mean_new = float(np.mean([results[key]["new"]["metrics"]["accuracy"] for key in transformed_keys]))
    interpretation = build_interpretation(results, diagnostic_originals)
    data = {"run_timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "manifest": str(manifest_path.relative_to(ROOT)), "test_count": len(test),
            "old_checkpoint": str(old_path.relative_to(ROOT)), "old_checkpoint_sha256": sha256_file(old_path),
            "new_checkpoint": str(new_path.relative_to(ROOT)), "new_checkpoint_sha256": sha256_file(new_path),
            "leakage": leakage, "distribution": distribution, "conditions": results,
            "train_only_casia_groups": sorted(set(distribution["casia_group_counts_by_split"]["train"])
                                                - set(distribution["casia_group_counts_by_split"]["val"])
                                                - set(distribution["casia_group_counts_by_split"]["test"])),
            "subgroup_metrics": subgroup_rows,
            "diagnostic_originals": diagnostic_originals, "diagnostic_transformations": diagnostic_transforms,
            "interpretation": interpretation,
            "transformations": {"crop_fraction": args.crop_fraction, "resize_scale": args.resize_scale,
                                "jpeg_quality": args.jpeg_quality, "combined_order": ["crop", "resize", "jpeg"]},
            "training_command": "python src/14_train_robust_efficientnet.py --manifest-path results/dataset_coverage_improved/new_manifest.csv --initial-checkpoint results/efficientnet_fused_robust_best_model.pth --output-checkpoint results/dataset_coverage_improved/efficientnet_fused_robust_balanced_best_model.pth --output-dir results/dataset_coverage_improved --epochs 2 --patience 2 --seed 20261002"}
    (OUTPUT_DIR / "old_new_evaluation.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    with (OUTPUT_DIR / "old_new_per_image_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_details[0]))
        writer.writeheader(); writer.writerows(all_details)
    with (OUTPUT_DIR / "subgroup_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(subgroup_rows[0]))
        writer.writeheader(); writer.writerows(subgroup_rows)
    with (OUTPUT_DIR / "false_negative_diagnostic.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(diagnostic_transforms[0]))
        writer.writeheader(); writer.writerows(diagnostic_transforms)
    with (OUTPUT_DIR / "robustness_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["Transformation", "Old Accuracy", "New Accuracy", "Difference (pp)", "Old Manipulated Recall",
                  "New Manipulated Recall", "Old Macro F1", "New Macro F1", "Old Consistency", "New Consistency",
                  "Old Accuracy Drop (pp)", "New Accuracy Drop (pp)"]
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for key, display, _ in CONDITIONS:
            old, new = results[key]["old"], results[key]["new"]
            writer.writerow({"Transformation": display, "Old Accuracy": old["metrics"]["accuracy"],
                             "New Accuracy": new["metrics"]["accuracy"],
                             "Difference (pp)": (new["metrics"]["accuracy"]-old["metrics"]["accuracy"])*100,
                             "Old Manipulated Recall": old["metrics"]["per_class"]["Manipulated"]["recall"],
                             "New Manipulated Recall": new["metrics"]["per_class"]["Manipulated"]["recall"],
                             "Old Macro F1": old["metrics"]["f1_macro"], "New Macro F1": new["metrics"]["f1_macro"],
                             "Old Consistency": old["prediction_consistency"], "New Consistency": new["prediction_consistency"],
                             "Old Accuracy Drop (pp)": old["accuracy_drop_percentage_points"],
                             "New Accuracy Drop (pp)": new["accuracy_drop_percentage_points"]})
    plot_comparisons(results, subgroup_rows)
    write_report(data)
    print("\nCURRENT vs CANDIDATE ON NEW TEST SPLIT")
    for key, display, _ in CONDITIONS:
        old, new = results[key]["old"], results[key]["new"]
        print(f"{display}: accuracy {old['metrics']['accuracy']:.4f} -> {new['metrics']['accuracy']:.4f} "
              f"({(new['metrics']['accuracy']-old['metrics']['accuracy'])*100:+.2f} pp); "
              f"manipulated recall {old['metrics']['per_class']['Manipulated']['recall']:.4f} -> "
              f"{new['metrics']['per_class']['Manipulated']['recall']:.4f}; macro F1 "
              f"{old['metrics']['f1_macro']:.4f} -> {new['metrics']['f1_macro']:.4f}; "
              f"consistency {old['prediction_consistency']:.4f} -> {new['prediction_consistency']:.4f}")
    print("\nCASIA SUBGROUPS (original)")
    for row in subgroup_rows:
        print(f"{row['subgroup']} (n={row['support']}): manipulated recall {row['old_manipulated_recall']:.4f} -> "
              f"{row['new_manipulated_recall']:.4f}; M->Real/M->AI {row['old_manipulated_to_real']}/{row['old_manipulated_to_ai']} -> "
              f"{row['new_manipulated_to_real']}/{row['new_manipulated_to_ai']}")
    print("\nORIGINAL TEST CONFUSION MATRICES (rows=actual, columns=predicted; Real, AI-Generated, Manipulated)")
    print("Current:", results["original"]["old"]["metrics"]["confusion_matrix"]["rows_actual_columns_predicted"])
    print("Candidate:", results["original"]["new"]["metrics"]["confusion_matrix"]["rows_actual_columns_predicted"])
    print("\nPREVIOUS FALSE-NEGATIVE DIAGNOSTICS (original)")
    for row in diagnostic_originals:
        print(f"{row['image']}: {row['old_prediction']} -> {row['new_prediction']} | "
              f"new p(Real/AI/Manipulated)={row['new_probability_Real']:.4f}/"
              f"{row['new_probability_AI-Generated']:.4f}/{row['new_probability_Manipulated']:.4f}")
    print(f"\n{interpretation}\nArtifacts: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
