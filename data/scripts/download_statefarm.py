from __future__ import annotations

import subprocess
from pathlib import Path
import zipfile

COMPETITION = "state-farm-distracted-driver-detection"
OUT_DIR = Path("data/raw/statefarm")


def run(cmd: list[str]) -> None:
    print(">", " ".join(cmd))
    subprocess.run(cmd, check=True)


def unzip(zip_path: Path, dest: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    run([
        "kaggle",
        "competitions",
        "download",
        "-c",
        COMPETITION,
        "-p",
        str(OUT_DIR)
    ])

    zips = list(OUT_DIR.glob("*.zip"))
    if not zips:
        raise FileNotFoundError("No zip file found")

    zip_path = max(zips, key=lambda p: p.stat().st_mtime)
    print(f"Unzipping {zip_path.name}")

    unzip(zip_path, OUT_DIR)

    train_dir = OUT_DIR / "train"
    if train_dir.exists():
        classes = sorted(p.name for p in train_dir.iterdir() if p.is_dir())
        print("Found classes:", classes)
    else:
        print("WARNING: train folder not found")


if __name__ == "__main__":
    main()
