from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_SIZE = (224, 224)
DEFAULT_MAX_PER_CLASS = 2000
DEFAULT_SEED = 20261002
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SOURCES = {
    "Real": {"label": 0, "path": PROJECT_ROOT / "data/raw/coco/val2017"},
    "AI-Generated": {"label": 1, "path": PROJECT_ROOT / "data/raw/genimage/ai"},
    "Manipulated": {"label": 2, "path": PROJECT_ROOT / "data/raw/casia/Tp"},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def readable(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def filename_group(path: Path) -> tuple[str, str]:
    parts = path.stem.split("_")
    # CASIA Tp files carry source family and tampering subtype in the first
    # three underscore-separated filename tokens (e.g. Tp_S_NNN). Normalize
    # subtype token case so capitalization variants do not form false groups.
    family = f"Tp_{parts[1].upper()}" if len(parts) >= 2 else "unparsed"
    group = f"{family}_{parts[2].upper()}" if len(parts) >= 3 else "unparsed"
    return family, group


def inspect_sources(exclusions: set[Path]) -> tuple[dict[str, list[dict]], dict[str, int]]:
    records: dict[str, list[dict]] = {}
    duplicate_hash_owners: dict[str, str] = {}
    duplicate_hashes = []
    for class_name, source in SOURCES.items():
        root = source["path"]
        if not root.is_dir():
            raise FileNotFoundError(f"Dataset source directory not found: {root}")
        found = []
        for path in sorted(p.resolve() for p in root.rglob("*")
                           if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS):
            if path in exclusions or not readable(path):
                continue
            digest = sha256_file(path)
            prior_class = duplicate_hash_owners.get(digest)
            if prior_class is not None:
                if prior_class != class_name:
                    raise RuntimeError(f"Identical source content has conflicting labels: {path} ({class_name}, {prior_class})")
                duplicate_hashes.append(str(path))
                continue
            duplicate_hash_owners[digest] = class_name
            with Image.open(path) as image:
                width, height = image.size
                mode = image.mode
            family, group = filename_group(path) if class_name == "Manipulated" else ("", "")
            group_raw = "_".join(path.stem.split("_")[:3]) if class_name == "Manipulated" else ""
            found.append({"source_path": path, "source_sha256": digest, "source_format": path.suffix.lower().lstrip("."),
                          "source_width": width, "source_height": height, "source_mode": mode,
                          "casia_family": family, "casia_group": group, "casia_group_raw": group_raw})
        records[class_name] = found
        print(f"{class_name}: {len(found)} readable unique files; source={root}", flush=True)
    if duplicate_hashes:
        print(f"Removed {len(duplicate_hashes)} duplicate file(s) by SHA-256 before sampling.", flush=True)
    return records, {"identical_source_files_skipped": len(duplicate_hashes)}


def largest_remainder(counts: dict[str, int], total: int, minimums: dict[str, int] | None = None) -> dict[str, int]:
    minimums = minimums or {}
    if sum(minimums.values()) > total or sum(counts.values()) < total:
        raise ValueError("Cannot allocate requested sample under the available group counts.")
    result = dict(minimums)
    remaining = total - sum(result.values())
    capacities = {key: counts[key] - result.get(key, 0) for key in counts}
    while remaining:
        capacity_total = sum(capacities.values())
        if capacity_total <= 0:
            raise ValueError("Group allocation ran out of capacity.")
        exact = {key: remaining * capacities[key] / capacity_total for key in counts}
        adds = {key: min(capacities[key], int(exact[key])) for key in counts}
        added = sum(adds.values())
        for key, value in adds.items():
            result[key] = result.get(key, 0) + value
            capacities[key] -= value
        remaining -= added
        if remaining:
            order = sorted(counts, key=lambda key: (-(exact[key] - int(exact[key])), key))
            progressed = False
            for key in order:
                if capacities[key] > 0:
                    result[key] = result.get(key, 0) + 1
                    capacities[key] -= 1
                    remaining -= 1
                    progressed = True
                    if remaining == 0:
                        break
            if not progressed:
                raise ValueError("Could not complete proportional allocation.")
    return result


def select_records(records: dict[str, list[dict]], max_per_class: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    selected = []
    for class_name, all_rows in records.items():
        if len(all_rows) < max_per_class:
            raise ValueError(f"{class_name} has {len(all_rows)} unique images; need {max_per_class}.")
        if class_name != "Manipulated":
            candidates = list(all_rows)
            rng.shuffle(candidates)
            picked = candidates[:max_per_class]
        else:
            groups: dict[str, list[dict]] = defaultdict(list)
            for row in all_rows:
                groups[row["casia_group"]].append(row)
            # Give each available subtype group a small floor, then allocate
            # the rest proportionally to its readable unique source count.
            counts = {group: len(items) for group, items in groups.items()}
            floors = {group: min(count, 3) for group, count in counts.items()}
            quotas = largest_remainder(counts, max_per_class, floors)
            picked = []
            for group in sorted(groups):
                candidates = list(groups[group])
                rng.shuffle(candidates)
                picked.extend(candidates[:quotas[group]])
            print("CASIA selected quota by group:", json.dumps(quotas, sort_keys=True), flush=True)
        for row in picked:
            selected.append({**row, "class_name": class_name, "label": SOURCES[class_name]["label"]})
    return selected


def split_selected(selected: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed)
    output = []
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in selected:
        key = (row["class_name"], row["casia_group"] if row["class_name"] == "Manipulated" else "all")
        grouped[key].append(row)
    for key, rows in sorted(grouped.items()):
        rng.shuffle(rows)
        n = len(rows)
        n_val = max(1, round(n * 0.15)) if n >= 3 else 0
        n_test = max(1, round(n * 0.15)) if n >= 2 else 0
        if key[0] == "Manipulated" and n >= 2:
            n_test = max(1, n_test)
        n_train = n - n_val - n_test
        if n_train < 0:
            n_train, n_val, n_test = 0, max(0, n - 1), min(1, n)
        for index, row in enumerate(rows):
            split = "train" if index < n_train else "val" if index < n_train + n_val else "test"
            output.append({**row, "split": split, "selection_seed": seed})
    return output


def prepare_image(source_path: Path, destination: Path) -> None:
    with Image.open(source_path) as image:
        result = image.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
        result.save(destination, format="JPEG", quality=95)
    with Image.open(destination) as image:
        image.verify()


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a reproducible, group-balanced prepared dataset in a new directory.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results/dataset_coverage_improved",
                        help="Must be a new isolated dataset output directory (defaults to results/dataset_coverage_improved).")
    parser.add_argument("--max-per-class", type=int, default=DEFAULT_MAX_PER_CLASS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--exclude-path", type=Path, action="append", default=[],
                        help="Optional source image exclusion, repeat for diagnostic-only images.")
    args = parser.parse_args()
    if args.max_per_class < 30:
        parser.error("--max-per-class must be at least 30 to preserve all three classes in each split.")
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty dataset output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = output_dir / "dataset"
    manifest_path = output_dir / "new_manifest.csv"
    exclusions = {path.resolve() for path in args.exclude_path}
    print(f"Output: {output_dir}\nSeed: {args.seed}\nPer-class target: {args.max_per_class}", flush=True)
    sources, source_audit = inspect_sources(exclusions)
    selected = select_records(sources, args.max_per_class, args.seed)
    rows = split_selected(selected, args.seed)

    # Prepare output paths only after the selected paths/splits are fixed.
    for class_name in SOURCES:
        class_index = 0
        for row in (item for item in rows if item["class_name"] == class_name):
            class_index += 1
            destination = data_dir / row["split"] / class_name / f"{class_name.lower().replace('-', '_')}_{class_index:05d}.jpg"
            destination.parent.mkdir(parents=True, exist_ok=True)
            prepare_image(row["source_path"], destination)
            row["filepath"] = str(destination.resolve())
            row["prepared_sha256"] = sha256_file(destination)
    fields = ["filepath", "source_path", "source_sha256", "prepared_sha256", "label", "class_name", "split",
              "casia_family", "casia_group", "casia_group_raw", "source_format", "source_width", "source_height", "source_mode", "selection_seed"]
    manifest_rows = [{key: row.get(key, "") for key in fields} for row in rows]
    write_csv(manifest_path, manifest_rows, fields)

    path_owners: dict[str, str] = {}
    hash_owners: dict[str, str] = {}
    leakage = {"path_cross_split_duplicates": [], "source_hash_cross_split_duplicates": [],
               "prepared_hash_cross_split_duplicates": []}
    for row in manifest_rows:
        split = row["split"]
        for field, findings in (("source_path", leakage["path_cross_split_duplicates"]),):
            key = str(Path(row[field]).resolve()).casefold()
            if key in path_owners and path_owners[key] != split:
                findings.append({"path": key, "splits": [path_owners[key], split]})
            path_owners[key] = split
        for field, findings in (("source_sha256", leakage["source_hash_cross_split_duplicates"]),
                                ("prepared_sha256", leakage["prepared_hash_cross_split_duplicates"])):
            key = row[field]
            owner_key = f"{field}:{key}"
            if owner_key in hash_owners and hash_owners[owner_key] != split:
                findings.append({"sha256": key, "splits": [hash_owners[owner_key], split]})
            hash_owners[owner_key] = split
    leakage["passed"] = not any(leakage[key] for key in leakage if key.endswith("duplicates"))

    split_counts = Counter(row["split"] for row in rows)
    class_counts = {split: dict(Counter(row["class_name"] for row in rows if row["split"] == split))
                    for split in ("train", "val", "test")}
    casia_counts = {split: dict(sorted(Counter(row["casia_group"] for row in rows
                                                if row["split"] == split and row["class_name"] == "Manipulated").items()))
                    for split in ("train", "val", "test")}
    casia_raw_counts = {split: dict(sorted(Counter(row["casia_group_raw"] for row in rows
                                                    if row["split"] == split and row["class_name"] == "Manipulated").items()))
                        for split in ("train", "val", "test")}
    format_counts_by_class = {class_name: dict(sorted(Counter(row["source_format"] for row in selected
                                                              if row["class_name"] == class_name).items()))
                              for class_name in SOURCES}
    available_casia_counts = dict(sorted(Counter(row["casia_group"] for row in sources["Manipulated"]).items()))
    selected_casia_counts = dict(sorted(Counter(row["casia_group"] for row in selected
                                                 if row["class_name"] == "Manipulated").items()))
    distribution = {"seed": args.seed, "max_per_class": args.max_per_class, "total": len(rows),
                    "split_counts": dict(split_counts), "class_counts_by_split": class_counts,
                    "casia_group_counts_by_split": casia_counts,
                    "casia_raw_group_counts_by_split": casia_raw_counts,
                    "casia_available_readable_unique_group_counts": available_casia_counts,
                    "casia_selected_group_counts": selected_casia_counts,
                    "available_readable_unique_counts_by_class": {name: len(items) for name, items in sources.items()},
                    "source_format_counts": dict(Counter(row["source_format"] for row in selected)),
                    "source_format_counts_by_class": format_counts_by_class,
                    "source_audit": source_audit, "excluded_paths": [str(p) for p in sorted(exclusions)],
                    "selection": "seeded random Real/AI sample; CASIA Tp proportional allocation by parsed filename group with up to 3-item floor; deterministic within-group sample; per-group 70/15/15 split"}
    (output_dir / "dataset_distribution.json").write_text(json.dumps(distribution, indent=2) + "\n", encoding="utf-8")
    (output_dir / "leakage_report.json").write_text(json.dumps(leakage, indent=2) + "\n", encoding="utf-8")
    if not leakage["passed"]:
        raise RuntimeError(f"Cross-split leakage detected; training is blocked. See {output_dir / 'leakage_report.json'}")
    print("Split counts:", json.dumps(dict(split_counts), sort_keys=True))
    print("Class counts:", json.dumps(class_counts, sort_keys=True))
    print("CASIA groups by split:", json.dumps(casia_counts, sort_keys=True))
    print(f"Leakage audit: PASS (source paths, source SHA-256, prepared SHA-256); manifest={manifest_path}", flush=True)


if __name__ == "__main__":
    main()
