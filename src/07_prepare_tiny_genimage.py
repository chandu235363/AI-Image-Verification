from __future__ import annotations

import shutil
from pathlib import Path

from datasets import load_dataset
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data/raw/genimage/ai"
DATASET_NAME = "TheKernel01/Tiny-GenImage"
TARGET_LABEL = 1
TARGET_COUNT = 2000
IMAGE_COLUMN = "image"
LABEL_COLUMN = "label"


def save_image(image, output_path: Path) -> bool:
    """Convert one dataset image to RGB JPEG and verify the saved file."""
    try:
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert("RGB")
        image.save(output_path, format="JPEG", quality=95)
        with Image.open(output_path) as saved_image:
            saved_image.verify()
        return True
    except Exception:
        if output_path.exists():
            output_path.unlink()
        return False


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    existing_files = sorted(OUTPUT_DIR.glob("tiny_genimage_ai_*.jpg"))
    if existing_files:
        print(f"Found {len(existing_files)} existing Tiny-GenImage files in {OUTPUT_DIR}.")
        print("Remove only these previously generated files before rerunning if a fresh 2,000-image subset is required.")
        return

    print(f"Loading {DATASET_NAME} in streaming mode.")
    print("Only label 1 (fake / AI-generated) examples will be saved.")
    dataset = load_dataset(DATASET_NAME, split="train", streaming=True)

    saved_count = 0
    failed_count = 0
    scanned_count = 0

    for example in dataset:
        scanned_count += 1
        if int(example[LABEL_COLUMN]) != TARGET_LABEL:
            continue

        output_path = OUTPUT_DIR / f"tiny_genimage_ai_{saved_count + 1:05d}.jpg"
        if save_image(example[IMAGE_COLUMN], output_path):
            saved_count += 1
            if saved_count % 100 == 0 or saved_count == TARGET_COUNT:
                print(f"Saved {saved_count}/{TARGET_COUNT} AI-generated images.")
        else:
            failed_count += 1

        if saved_count >= TARGET_COUNT:
            break

    disk_usage = sum(path.stat().st_size for path in OUTPUT_DIR.glob("tiny_genimage_ai_*.jpg"))
    disk_usage_mb = disk_usage / (1024 * 1024)

    print("Tiny-GenImage preparation complete.")
    print(f"Images saved: {saved_count}")
    print(f"Failed images: {failed_count}")
    print(f"Examples scanned: {scanned_count}")
    print(f"Final folder: {OUTPUT_DIR.resolve()}")
    print(f"Disk space used: {disk_usage_mb:.2f} MB ({disk_usage} bytes)")

    if saved_count != TARGET_COUNT:
        raise RuntimeError(
            f"Expected exactly {TARGET_COUNT} valid label-1 images, but saved {saved_count}."
        )


if __name__ == "__main__":
    main()
