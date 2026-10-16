#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


REGIONS = [
    "vlaams_brabant",
    "oost_vlaanderen",
    "west_vlaanderen",
    "antwerpen",
    "limburg",
    "brussel",
    "andere",
]


def postal_code_to_region(postal_code: str) -> str:
    """
    Map Belgian postal code to one of the requested regions.

    Brussels:
        1000-1299

    Flemish Brabant:
        1500-1999 and 3000-3499

    Antwerp:
        2000-2999

    Limburg:
        3500-3999

    West Flanders:
        8000-8999

    East Flanders:
        9000-9999

    Everything else:
        andere
    """
    try:
        pc = int(postal_code.strip())
    except ValueError:
        return "andere"

    if 1000 <= pc <= 1299:
        return "brussel"
    if 1500 <= pc <= 1999 or 3000 <= pc <= 3499:
        return "vlaams_brabant"
    if 2000 <= pc <= 2999:
        return "antwerpen"
    if 3500 <= pc <= 3999:
        return "limburg"
    if 8000 <= pc <= 8999:
        return "west_vlaanderen"
    if 9000 <= pc <= 9999:
        return "oost_vlaanderen"

    return "andere"


def read_wav_scp(path: Path) -> dict[str, Path]:
    utt2wav: dict[str, Path] = {}

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line.strip():
                continue

            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(f"Malformed wav.scp line {line_no}: {line!r}")

            utt_id, wav_path = parts
            utt2wav[utt_id] = Path(wav_path)

    return utt2wav


def read_text(path: Path) -> dict[str, str]:
    utt2text: dict[str, str] = {}

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line.strip():
                continue

            parts = line.split(maxsplit=1)
            if len(parts) == 1:
                utt_id, transcription = parts[0], ""
            else:
                utt_id, transcription = parts

            utt2text[utt_id] = transcription

    return utt2text


def read_spk2birthplace(path: Path) -> dict[str, str]:
    spk2birthplace: dict[str, str] = {}

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(f"Malformed spk2birthplace line {line_no}: {line!r}")

            spk_id, postal_code = parts
            spk2birthplace[spk_id] = postal_code

    return spk2birthplace

def speaker_from_utterance_id(utt_id: str) -> str:
    """
    UTTERANCE_ID = f"{SPEAKER_ID}-{utterance_number}"

    For example:
      007e10b0-ca32-4d9a-87c8-cfe258cc0896-0005

    Speaker ID:
      007e10b0-ca32-4d9a-87c8-cfe258cc0896
    """
    if "-" not in utt_id:
        raise ValueError(f"Cannot infer speaker ID from utterance ID without hyphen: {utt_id!r}")

    return utt_id.rsplit("-", maxsplit=1)[0]


def resolve_wav_path(dataset_dir: Path, wav_path: Path) -> Path:
    """
    wav.scp paths may be:
      1. absolute,
      2. relative to the current working directory,
      3. relative to the dataset directory.

    Try them in that order.
    """
    if wav_path.is_absolute():
        return wav_path

    # Case 1: wav.scp path is relative to current working directory.
    if wav_path.exists():
        return wav_path

    # Case 2: wav.scp path is relative to dataset directory.
    candidate = dataset_dir / wav_path
    if candidate.exists():
        return candidate

    # Return the most likely path for error/debug reporting.
    return wav_path

def prepare_output_dirs(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for region in REGIONS:
        (output_dir / region).mkdir(parents=True, exist_ok=True)


def convert_dataset(
    dataset_dir: Path,
    output_dir: Path,
    dry_run: bool = False,
    overwrite: bool = False,
) -> None:
    wav_scp = dataset_dir / "wav.scp"
    text_file = dataset_dir / "text"
    spk2birthplace_file = dataset_dir / "spk2birthplace"

    for required in [wav_scp, text_file, spk2birthplace_file]:
        if not required.exists():
            raise FileNotFoundError(f"Missing required file: {required}")

    utt2wav = read_wav_scp(wav_scp)
    utt2text = read_text(text_file)
    spk2birthplace = read_spk2birthplace(spk2birthplace_file)

    prepare_output_dirs(output_dir)

    copied = 0
    skipped_missing_text = 0
    skipped_missing_birthplace = 0
    skipped_missing_audio = 0

    region_counts = {region: 0 for region in REGIONS}

    for utt_id, raw_wav_path in sorted(utt2wav.items()):
        if utt_id not in utt2text:
            skipped_missing_text += 1
            continue

        try:
            spk_id = speaker_from_utterance_id(utt_id)
        except ValueError:
            skipped_missing_birthplace += 1
            continue

        postal_code = spk2birthplace.get(spk_id)
        if postal_code is None:
            skipped_missing_birthplace += 1
            continue

        region = postal_code_to_region(postal_code)

        src_wav = resolve_wav_path(dataset_dir, raw_wav_path)
        if not src_wav.exists():
            skipped_missing_audio += 1
            continue

        dest_flac = output_dir / region / f"{utt_id}.flac"
        dest_txt = output_dir / region / f"{utt_id}.txt"

        if not overwrite and (dest_flac.exists() or dest_txt.exists()):
            raise FileExistsError(
                f"Output already exists for {utt_id}. "
                f"Use --overwrite to replace existing files."
            )

        if not dry_run:
            shutil.copy2(src_wav, dest_flac)
            dest_txt.write_text(utt2text[utt_id] + "\n", encoding="utf-8")

        copied += 1
        region_counts[region] += 1

    print("Done.")
    print(f"Dataset directory: {dataset_dir}")
    print(f"Output directory:  {output_dir}")
    print(f"Dry run:           {dry_run}")
    print()
    print("Utterances written by region:")
    for region in REGIONS:
        print(f"  {region:16s} {region_counts[region]}")

    print()
    print("Skipped:")
    print(f"  missing text:        {skipped_missing_text}")
    print(f"  missing birthplace:  {skipped_missing_birthplace}")
    print(f"  missing audio:       {skipped_missing_audio}")
    print()
    print(f"Total written: {copied}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert an ESPnet-style dataset into regional folders containing "
            "UTTERANCE_ID.flac and UTTERANCE_ID.txt files."
        )
    )

    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="Path to the ESPnet-style dataset directory, e.g. Maarallee_10s",
    )

    parser.add_argument(
        "output_dir",
        type=Path,
        help="Path to the output directory to create",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and count files without copying or writing anything",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )

    args = parser.parse_args()

    convert_dataset(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
