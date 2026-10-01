from pathlib import Path

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DATASET_PATHS = [
    Path("data/raw/genimage"),
    Path("data/raw/casia"),
    Path("data/raw/coco"),
]


def count_images(folder: Path) -> int:
    if not folder.exists() or not folder.is_dir():
        return 0

    image_count = 0
    for file_path in folder.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in VALID_EXTENSIONS:
            image_count += 1
    return image_count


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent

    print("Dataset check for AI image verification project")
    print(f"Project root: {project_root}")
    print("=" * 80)

    for dataset_path in DATASET_PATHS:
        full_path = project_root / dataset_path
        count = count_images(full_path)

        if full_path.exists():
            status = "FOUND" if full_path.is_dir() else "NOT A DIRECTORY"
        else:
            status = "MISSING"

        print(f"Dataset folder: {full_path}")
        print(f"Image count: {count}")
        print(f"Status: {status}")
        print("-" * 80)

    print("Dataset scan complete.")


if __name__ == "__main__":
    main()
