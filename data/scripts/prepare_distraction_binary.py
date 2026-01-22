from __future__ import annotations

import csv
import hashlib
import os
import random
import shutil
from pathlib import Path
from collections import Counter, defaultdict

# ====== RAW INPUTS (StateFarm) ======
RAW_ROOT = Path("data/raw/statefarm")
RAW_TRAIN = RAW_ROOT / "imgs" / "train"
DRIVER_CSV = RAW_ROOT / "driver_imgs_list.csv"

# ====== OUTPUT (binary dataset) ======
OUT_ROOT = Path("data/processed/distraction_binary")

# ====== CLASS MAPPING ======
SAFE_CLASS = "c0"
SAFE_LABEL = "not_distracted"
DISTRACTED_LABEL = "distracted"

# ====== SPLIT SETTINGS ======
SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
RANDOM_SEED = 42

# If True: copy images (safe but uses disk). If False: symlink (faster, but can be annoying on Windows).
COPY_FILES = True


def ensure_dirs() -> None:
    for split in ("train", "val", "test"):
        (OUT_ROOT / split / SAFE_LABEL).mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / split / DISTRACTED_LABEL).mkdir(parents=True, exist_ok=True)


def read_driver_map(csv_path: Path) -> dict[str, list[str]]:
    """
    Returns: driver_id -> list of relative image paths like 'c3/img_123.jpg'
    CSV columns (StateFarm): subject, classname, img
    """
    driver_to_images: dict[str, list[str]] = defaultdict(list)

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        expected = {"subject", "classname", "img"}
        if not expected.issubset(reader.fieldnames or set()):
            raise ValueError(f"driver_imgs_list.csv missing columns. Found: {reader.fieldnames}")

        for row in reader:
            driver = row["subject"].strip()
            cls = row["classname"].strip()
            img = row["img"].strip()
            rel = f"{cls}/{img}"
            driver_to_images[driver].append(rel)

    return driver_to_images


def assign_driver_splits(drivers: list[str], seed: int = 42) -> dict[str, str]:
    """
    Returns: driver_id -> split ('train'|'val'|'test') using ratios.
    """
    rng = random.Random(seed)
    drivers = drivers.copy()
    rng.shuffle(drivers)

    n = len(drivers)
    n_train = int(n * SPLIT_RATIOS["train"])
    n_val = int(n * SPLIT_RATIOS["val"])
    # remainder goes to test
    n_test = n - n_train - n_val

    train_dr = set(drivers[:n_train])
    val_dr = set(drivers[n_train:n_train + n_val])
    test_dr = set(drivers[n_train + n_val:])

    assert len(train_dr) + len(val_dr) + len(test_dr) == n

    out: dict[str, str] = {}
    for d in train_dr:
        out[d] = "train"
    for d in val_dr:
        out[d] = "val"
    for d in test_dr:
        out[d] = "test"

    return out


def map_binary_label(classname: str) -> str:
    return SAFE_LABEL if classname == SAFE_CLASS else DISTRACTED_LABEL


def copy_or_link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if COPY_FILES:
        shutil.copy2(src, dst)
    else:
        # best-effort symlink
        os.symlink(src.resolve(), dst)


def main() -> None:
    # Sanity checks
    if not DRIVER_CSV.exists():
        raise FileNotFoundError(f"Missing {DRIVER_CSV}. Did you finish Step 1?")
    if not RAW_TRAIN.exists():
        raise FileNotFoundError(f"Missing {RAW_TRAIN}. Did you finish Step 1?")

    ensure_dirs()

    driver_to_images = read_driver_map(DRIVER_CSV)
    drivers = sorted(driver_to_images.keys())

    driver_split = assign_driver_splits(drivers, seed=RANDOM_SEED)

    # Stats counters
    counts = Counter()
    split_driver_counts = Counter(driver_split.values())

    print("Drivers per split:", dict(split_driver_counts))
    print("Building binary dataset... (this can take a while)")

    missing_files = 0

    for driver, rel_images in driver_to_images.items():
        split = driver_split[driver]

        for rel in rel_images:
            # rel is "cX/img_YYY.jpg"
            cls = rel.split("/", 1)[0]
            label = map_binary_label(cls)

            src = RAW_TRAIN / rel
            if not src.exists():
                missing_files += 1
                continue

            # Keep filenames unique to avoid collisions across classes/drivers
            # Format: <driver>__<original_filename>
            orig_name = Path(rel).name
            dst_name = f"{driver}__{orig_name}"
            dst = OUT_ROOT / split / label / dst_name

            copy_or_link(src, dst)
            counts[(split, label)] += 1

    if missing_files:
        print(f"WARNING: {missing_files} files referenced in CSV were missing on disk.")

    # Print class counts
    print("\nImage counts:")
    for split in ("train", "val", "test"):
        for label in (SAFE_LABEL, DISTRACTED_LABEL):
            print(f"{split:5s} {label:14s}: {counts[(split, label)]}")

    print("\nDone. Output at:", OUT_ROOT.resolve())


if __name__ == "__main__":
    main()
