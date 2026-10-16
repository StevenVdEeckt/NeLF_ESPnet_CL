#!/usr/bin/env python3
"""
Prepare CHOREC-style data directories with additional text variants.

For each dataset (train, dev, or custom):
  - If verbatim data:
      • copy text → text.lc.verbatim
      • create text.lc.subtitle with "<unk>" for all utterances
      • create utt2category with "verbatim"
  - If subtitle data:
      • copy text → text.lc.subtitle
      • create text.lc.verbatim with "<unk>" for all utterances
      • create utt2category with "subtitle"
Existing files are never modified, only new ones are created.
"""

import argparse
from pathlib import Path
import shutil


def copy_text_file(src: Path, dst: Path):
    """Copy text file safely."""
    if not src.exists():
        raise FileNotFoundError(f"Missing required file: {src}")
    shutil.copy2(src, dst)
    print(f"[INFO] Copied {src.name} → {dst.name}")


def make_placeholder_text(src: Path, dst: Path):
    """Create a <unk> version of text."""
    if not src.exists():
        raise FileNotFoundError(f"Missing required file: {src}")

    lines_out = []
    with src.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            utt_id = line.split(maxsplit=1)[0]
            lines_out.append(f"{utt_id} <unk>")

    with dst.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines_out) + "\n")

    print(f"[INFO] Generated placeholder file {dst.name} ({len(lines_out)} entries)")


def make_utt2category(src: Path, dst: Path, category: str):
    """Generate utt2category file: 'utt_id category' for each utterance."""
    if not src.exists():
        raise FileNotFoundError(f"Missing required file: {src}")

    lines_out = []
    with src.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            utt_id = line.split(maxsplit=1)[0]
            lines_out.append(f"{utt_id} {category}")

    with dst.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines_out) + "\n")

    print(f"[INFO] Created {dst.name} with category '{category}' ({len(lines_out)} entries)")


def process_dataset(dataset_dir: Path, is_verbatim: bool):
    text_file = dataset_dir / "text"

    if not text_file.exists():
        print(f"[WARN] Skipping {dataset_dir}, no 'text' file found.")
        return

    if is_verbatim:
        # Verbatim → copy to text.lc.verbatim, make subtitle <unk>
        copy_text_file(text_file, dataset_dir / "text.lc.verbatim")
        make_placeholder_text(text_file, dataset_dir / "text.lc.subtitle")
        make_utt2category(text_file, dataset_dir / "utt2category", "verbatim")
    else:
        # Subtitle → copy to text.lc.subtitle, make verbatim <unk>
        copy_text_file(text_file, dataset_dir / "text.lc.subtitle")
        make_placeholder_text(text_file, dataset_dir / "text.lc.verbatim")
        make_utt2category(text_file, dataset_dir / "utt2category", "subtitle")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare text.lc.* and utt2category files for CHOREC/CGN data."
    )
    parser.add_argument(
        "data_dir",
        type=str,
        help="Main data directory containing subdirectories (train, dev, ...)",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="train,dev",
        help="Comma-separated list of dataset subdirectories (default: train,dev)",
    )
    parser.add_argument(
        "--is_verbatim",
        action="store_true",
        default=True,
        help="If set, treats data as verbatim (default: False = subtitle).",
    )
    parser.add_argument(
        "--not_verbatim",
        dest="is_verbatim",
        action="store_false",
        default=False,
        help="Explicitly set to subtitle data."
    )

    args = parser.parse_args()

    base_dir = Path(args.data_dir)
    dataset_names = [d.strip() for d in args.datasets.split(",") if d.strip()]
    is_verbatim = args.is_verbatim

    print(f"[INFO] Base directory: {base_dir}")
    print(f"[INFO] Datasets: {dataset_names}")
    print(f"[INFO] Mode: {'verbatim' if is_verbatim else 'subtitle'}")

    for name in dataset_names:
        dataset_path = base_dir / name
        if not dataset_path.exists():
            print(f"[WARN] Dataset directory not found: {dataset_path}")
            continue
        process_dataset(dataset_path, is_verbatim)


if __name__ == "__main__":
    main()

