"""Few-shot family builder refactored for VoxCeleb-style metadata."""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence

DEFAULT_META_DIR = "/data/voxceleb/meta"
DEFAULT_AUDIO_DIR = "/data/voxceleb/audio"
DEFAULT_OUTPUT_BASE = "/data/voxceleb/5shot3way"

csv_files = [
    "messages_data_20250710_195241_part_001.csv",
    "messages_data_20250710_195241_part_002.csv",
    "messages_data_20250710_195241_part_003.csv",
]
meta_dir = DEFAULT_META_DIR
audio_dir = DEFAULT_AUDIO_DIR
output_base = DEFAULT_OUTPUT_BASE
speakers_per_family = 5
shot_num = 3

KNOWN_SPEAKER_COLUMNS = ["speaker_id", "chat_id", "user_id", "speaker"]
KNOWN_WAV_COLUMNS = [
    "wav_id",
    "voice_id",
    "voice_embedding_id",
    "wav_filename",
    "wav_file",
]


def detect_columns(fieldnames: Sequence[str]) -> tuple[str, str]:
    normalized = {name.lower(): name for name in fieldnames}
    for speaker_col in KNOWN_SPEAKER_COLUMNS:
        for wav_col in KNOWN_WAV_COLUMNS:
            if speaker_col in normalized and wav_col in normalized:
                return normalized[speaker_col], normalized[wav_col]
    speaker = next(
        (name for name in fieldnames if "speaker" in name.lower()), None
    )
    wav = next(
        (name for name in fieldnames if "wav" in name.lower() or "voice" in name.lower()),
        None,
    )
    if speaker and wav:
        return speaker, wav
    if len(fieldnames) >= 2:
        return fieldnames[0], fieldnames[1]
    raise ValueError("Cannot deduce speaker/wav columns from metadata header.")


def read_meta(meta_dir: str, csv_files: Iterable[str]) -> List[Mapping[str, str]]:
    base = Path(meta_dir)
    if not base.exists():
        raise FileNotFoundError(f"Metadata directory not found: {meta_dir}")

    chosen = list(csv_files)
    if not chosen:
        chosen = [str(p.name) for p in sorted(base.glob("*.csv"))]

    rows: list[Mapping[str, str]] = []
    for csv_file in chosen:
        path = base / csv_file
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            if not reader.fieldnames:
                continue
            speaker_col, wav_col = detect_columns(reader.fieldnames)
            for row in reader:
                speaker = row.get(speaker_col, "").strip()
                wav_file = row.get(wav_col, "").strip()
                if not speaker or not wav_file:
                    continue
                rows.append({"speaker": speaker, "wav": wav_file})
    return rows


def collect_speaker_wavs(
    audio_dir: str, rows: Iterable[Mapping[str, str]]
) -> dict[str, List[Path]]:
    audio_root = Path(audio_dir)
    data: dict[str, List[Path]] = defaultdict(list)
    for row in rows:
        speaker = row["speaker"]
        wav_file = Path(row["wav"])
        candidate = wav_file if wav_file.is_absolute() else audio_root / wav_file
        if candidate.exists():
            data[speaker].append(candidate)
    return data


def filter_speakers(
    speaker_map: dict[str, List[Path]], min_samples: int
) -> dict[str, List[Path]]:
    valid: dict[str, List[Path]] = {}
    for speaker, paths in speaker_map.items():
        unique_paths = sorted(set(paths))
        if len(unique_paths) >= min_samples:
            valid[speaker] = unique_paths
    return valid


def copy_speaker_files(
    speaker_id: str,
    paths: List[Path],
    family_dir: Path,
    shot_num: int,
) -> None:
    train_dir = family_dir / "train" / speaker_id
    test_dir = family_dir / "test" / speaker_id
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    for i, src in enumerate(paths):
        dest = train_dir if i < shot_num else test_dir
        shutil.copy2(src, dest / src.name)


def build_families(
    speaker_index: dict[str, List[Path]],
    output_base: str,
    speakers_per_family: int,
    shot_num: int,
) -> int:
    output_root = Path(output_base)
    output_root.mkdir(parents=True, exist_ok=True)
    counter = 1
    group: list[str] = []

    def flush_group() -> None:
        nonlocal counter
        if not group:
            return
        family_dir = output_root / f"family{counter:05d}"
        family_dir.mkdir(parents=True, exist_ok=True)
        for speaker in group:
            copy_speaker_files(speaker, speaker_index[speaker], family_dir, shot_num)
        counter += 1
        group.clear()

    for speaker in sorted(speaker_index):
        group.append(speaker)
        if len(group) == speakers_per_family:
            flush_group()
    flush_group()
    return counter - 1


def generate_families(
    meta_dir: str,
    audio_dir: str,
    output_base: str,
    csv_files: Iterable[str],
    speakers_per_family: int,
    shot_num: int,
) -> None:
    metadata = read_meta(meta_dir, csv_files)
    if not metadata:
        raise RuntimeError("No metadata rows found; check the CSV inputs.")
    speaker_map = collect_speaker_wavs(audio_dir, metadata)
    filtered = filter_speakers(speaker_map, shot_num + 1)
    if not filtered:
        raise RuntimeError("No speaker has enough samples for the requested shot configuration.")
    families_created = build_families(filtered, output_base, speakers_per_family, shot_num)
    print(
        f"Generated {families_created} families from {len(filtered)} valid speakers "
        f"(shot={shot_num}, {speakers_per_family} per family) into {output_base}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build few-shot families from VoxCeleb-style metadata."
    )
    parser.add_argument("--meta-dir", help="Directory containing metadata CSV files.")
    parser.add_argument("--audio-dir", help="Directory containing WAV files.")
    parser.add_argument("--output-base", help="Output base directory for families.")
    parser.add_argument(
        "--csv-files",
        nargs="*",
        help="List of CSV files (relative to meta dir) to read.",
    )
    parser.add_argument(
        "--speakers-per-family",
        type=int,
        help="Number of distinct speakers per family.",
    )
    parser.add_argument("--shot-num", type=int, help="Number of shots per speaker.")
    parser.add_argument("--dataset-name", default="voxceleb", help="Dataset label for logging.")
    return parser.parse_args()


def main(
    meta_dir: str,
    audio_dir: str,
    output_base: str,
    csv_files: Sequence[str],
    speakers_per_family: int,
    shot_num: int,
    dataset_name: str = "voxceleb",
) -> None:
    csv_files_list = list(csv_files) if csv_files else []
    print(f"Building families ({dataset_name})")
    print(f"meta dir: {meta_dir}, audio dir: {audio_dir}, output: {output_base}")
    if not csv_files_list:
        print("No CSV files configured; will scan the metadata directory for *.csv files.")
    generate_families(
        meta_dir=meta_dir,
        audio_dir=audio_dir,
        output_base=output_base,
        csv_files=csv_files_list,
        speakers_per_family=speakers_per_family,
        shot_num=shot_num,
    )


if __name__ == "__main__":
    args = parse_args()
    main(
        meta_dir=args.meta_dir or meta_dir,
        audio_dir=args.audio_dir or audio_dir,
        output_base=args.output_base or output_base,
        csv_files=args.csv_files or csv_files,
        speakers_per_family=args.speakers_per_family or speakers_per_family,
        shot_num=args.shot_num or shot_num,
        dataset_name=args.dataset_name,
    )
