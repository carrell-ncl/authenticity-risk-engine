#!/usr/bin/env python3
"""
prepare_asvspoof2021_df.py

Prepare ASVspoof 2021 DF into a canonical folder layout:

  output_dir/
    real/
    fake/
    manifests/
      files.csv

This script:
- extracts the DF keys/metadata tar.gz (optional)
- parses protocol/key files to map utterance_id -> label (bonafide/spoof)
- finds matching audio files under the dataset root (e.g., flac/ or wav/)
- copies/moves/symlinks audio into output_dir/real and output_dir/fake
- writes a CSV manifest for traceability

Google-style docstrings are included on all functions.

Notes:
- ASVspoof protocols typically contain an utterance ID and a label token
  such as "bonafide" or "spoof". This script searches each line for those.
- If multiple protocol files exist (train/dev/eval), the script can use:
    --split train|dev|eval|all

Usage:
python src/data/prepare_asvspoof2021_df.py \
  --dataset_root data/audio/raw/asvspoof_2021_df \
  --keys_root data/audio/raw/asvspoof_2021_df/protocols \
  --output_dir data/audio/processed/asvspoof_2021_df \
  --audio_subdir flac \
  --mode copy \
  --split all

"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


LABEL_REAL = "bonafide"
LABEL_FAKE = "spoof"


@dataclass(frozen=True)
class LabeledUtterance:
    """Container for a labeled utterance."""
    utt_id: str
    label: str
    protocol_path: str
    raw_line: str


def ensure_dir(path: Path) -> None:
    """Create a directory (including parents) if it does not exist.

    Args:
        path: Directory path to create.
    """
    path.mkdir(parents=True, exist_ok=True)


def extract_tar_gz(tar_path: Path, dest_dir: Path) -> None:
    """Extract a .tar.gz archive to a destination directory.

    Args:
        tar_path: Path to the .tar.gz file.
        dest_dir: Destination directory to extract into.

    Raises:
        FileNotFoundError: If tar_path does not exist.
        tarfile.TarError: If extraction fails.
    """
    if not tar_path.exists():
        raise FileNotFoundError(f"Tar file not found: {tar_path}")
    ensure_dir(dest_dir)
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(dest_dir)


def find_protocol_files(keys_root: Path) -> List[Path]:
    """Find protocol/key files under the extracted keys/metadata directory.

    This searches for text files that likely contain protocol lines.
    It is intentionally permissive: it will include .txt and files with "trial"/"protocol"/"keys" in name.

    Args:
        keys_root: Root directory where keys/metadata have been extracted.

    Returns:
        List of protocol file paths (sorted).
    """
    candidates: List[Path] = []
    for p in keys_root.rglob("*"):
        if not p.is_file():
            continue
        name = p.name.lower()
        if p.suffix.lower() in {".txt", ".tsv"}:
            candidates.append(p)
        elif any(x in name for x in ("protocol", "trial", "trials", "keys", "key")):
            # Some protocol files may not have .txt extension.
            candidates.append(p)

    # De-dup and sort
    uniq = sorted({c.resolve() for c in candidates})
    return uniq


def infer_split_from_path(p: Path) -> Optional[str]:
    """Infer split name (train/dev/eval) from a protocol filename or its path.

    Args:
        p: Protocol file path.

    Returns:
        "train", "dev", "eval" if found; otherwise None.
    """
    s = str(p).lower()
    # common variants: train, trn, dev, val, eval, test
    if re.search(r"\b(train|trn)\b", s):
        return "train"
    if re.search(r"\b(dev|devel|val|valid)\b", s):
        return "dev"
    if re.search(r"\b(eval|test)\b", s):
        return "eval"
    return None


def parse_protocol_file(protocol_path: Path) -> List[LabeledUtterance]:
    """Parse a single ASVspoof-style protocol/metadata file to extract utterance IDs and labels.

    This implementation is designed to work with ASVspoof 2021 DF `trial_metadata.txt`
    lines of the form:

        <speaker_or_source_id> <utt_id> <codec> <corpus> <attack_id> <label> ...

    Example:
        LA_0023 DF_E_2000011 nocodec asvspoof A14 spoof notrim ...

    In this format, the audio filename stem is always the SECOND token (tokens[1]),
    and the label token is either "bonafide" (real) or "spoof" (fake).

    Args:
        protocol_path: Path to the protocol/metadata file.

    Returns:
        List of labeled utterances found in the file.
    """
    items: List[LabeledUtterance] = []

    with protocol_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            raw = line.rstrip("\n")
            line_stripped = raw.strip()
            if not line_stripped:
                continue
            if line_stripped.startswith("#"):
                continue

            tokens = line_stripped.split()
            if len(tokens) < 2:
                # Need at least: <source_id> <utt_id>
                continue

            tokens_lower = [t.lower() for t in tokens]

            # Extract label
            label: Optional[str] = None
            if LABEL_REAL in tokens_lower:
                label = LABEL_REAL
            elif LABEL_FAKE in tokens_lower:
                label = LABEL_FAKE
            else:
                # Not a protocol line we understand (no bonafide/spoof token)
                continue

            # ASVspoof2021 DF trial_metadata: tokens[1] is the utterance id / audio stem
            utt_id = tokens[1]

            items.append(
                LabeledUtterance(
                    utt_id=utt_id,
                    label=label,
                    protocol_path=str(protocol_path),
                    raw_line=raw,
                )
            )

    return items



def select_by_split(protocol_files: Sequence[Path], split: str) -> List[Path]:
    """Select protocol files matching a requested split.

    Args:
        protocol_files: Candidate protocol files.
        split: One of "train", "dev", "eval", "all".

    Returns:
        Filtered list of protocol file paths.

    Raises:
        ValueError: If split is invalid.
    """
    split = split.lower()
    if split not in {"train", "dev", "eval", "all"}:
        raise ValueError("split must be one of: train, dev, eval, all")

    if split == "all":
        return list(protocol_files)

    chosen: List[Path] = []
    for p in protocol_files:
        s = infer_split_from_path(p)
        if s == split:
            chosen.append(p)

    return chosen


def build_label_map(protocol_files: Sequence[Path]) -> Dict[str, LabeledUtterance]:
    """Build a mapping from utterance_id to label using one or more protocol files.

    If the same utt_id appears multiple times, the first occurrence is kept and later
    duplicates are ignored.

    Args:
        protocol_files: List of protocol files to parse.

    Returns:
        Dictionary mapping utt_id -> LabeledUtterance.
    """
    label_map: Dict[str, LabeledUtterance] = {}
    for pf in protocol_files:
        for item in parse_protocol_file(pf):
            if item.utt_id not in label_map:
                label_map[item.utt_id] = item
    return label_map


def index_audio_files(audio_root: Path, exts: Tuple[str, ...]) -> Dict[str, Path]:
    """Index audio files under a directory by stem (filename without extension).

    Args:
        audio_root: Directory containing audio files (possibly nested).
        exts: Audio extensions to include (lowercase, include dot).

    Returns:
        Mapping from stem -> absolute path to the file.
    """
    idx: Dict[str, Path] = {}
    for p in audio_root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue
        stem = p.stem
        # If duplicates exist, keep the first. (Manifest will help detect this.)
        idx.setdefault(stem, p.resolve())
    return idx


def sha256_of_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA-256 hash of a file.

    Args:
        path: Path to file.
        chunk_size: Bytes per read chunk.

    Returns:
        Hex string hash.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def transfer_file(src: Path, dst: Path, mode: str) -> None:
    """Transfer a file using copy/move/symlink mode.

    Args:
        src: Source file path.
        dst: Destination file path.
        mode: One of "copy", "move", "symlink".

    Raises:
        ValueError: If mode is invalid.
    """
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "move":
        shutil.move(str(src), str(dst))
    elif mode == "symlink":
        # Use relative symlink where possible
        try:
            rel = os.path.relpath(src, start=dst.parent)
            dst.symlink_to(rel)
        except Exception:
            dst.symlink_to(src)
    else:
        raise ValueError("mode must be one of: copy, move, symlink")


def prepare_dataset(
    dataset_root: Path,
    keys_root: Path,
    output_dir: Path,
    split: str,
    audio_subdir: Optional[str],
    mode: str,
    exts: Tuple[str, ...],
    compute_hashes: bool,
) -> Path:
    """Prepare ASVspoof 2021 DF data into real/fake folders.

    Args:
        dataset_root: Root folder of the ASVspoof 2021 DF dataset (audio lives somewhere under here).
        keys_root: Root folder of extracted keys/metadata (contains protocol files).
        output_dir: Destination folder that will contain real/ and fake/.
        split: "train", "dev", "eval", or "all" to select which protocol files to use.
        audio_subdir: Optional subdirectory (relative to dataset_root) to search for audio files.
        mode: "copy", "move", or "symlink".
        exts: Audio extensions to search for.
        compute_hashes: If True, compute sha256 hashes for manifest rows.

    Returns:
        Path to the generated CSV manifest.
    """
    ensure_dir(output_dir)
    out_real = output_dir / "real"
    out_fake = output_dir / "fake"
    ensure_dir(out_real)
    ensure_dir(out_fake)
    ensure_dir(output_dir / "manifests")

    protocol_files = find_protocol_files(keys_root)
    selected_protocols = select_by_split(protocol_files, split)
    if not selected_protocols:
        # Fall back: if split filtering yields none, use all protocols.
        selected_protocols = protocol_files

    label_map = build_label_map(selected_protocols)
    if not label_map:
        raise RuntimeError(
            f"No labeled utterances found. Check keys_root={keys_root} and split={split}."
        )

    audio_root = (dataset_root / audio_subdir) if audio_subdir else dataset_root
    audio_index = index_audio_files(audio_root, exts=exts)
    if not audio_index:
        raise RuntimeError(
            f"No audio files found under {audio_root} with extensions {exts}."
        )

    manifest_path = output_dir / "manifests" / f"files_{split}.csv"
    rows_written = 0
    missing_audio = 0
    duplicate_out = 0

    with manifest_path.open("w", newline="", encoding="utf-8") as mf:
        writer = csv.DictWriter(
            mf,
            fieldnames=[
                "utt_id",
                "label",
                "src_path",
                "dst_path",
                "protocol_path",
                "sha256",
                "raw_protocol_line",
            ],
        )
        writer.writeheader()

        for utt_id, item in sorted(label_map.items(), key=lambda kv: kv[0]):
            src = audio_index.get(utt_id)
            if src is None:
                # Sometimes the protocol ID matches filename with suffixes; try best-effort lookup.
                # Example: utt_id might include extension or extra tokens.
                # Attempt to normalise:
                norm = Path(utt_id).stem
                src = audio_index.get(norm)

            if src is None:
                missing_audio += 1
                continue

            label_dir = out_real if item.label == LABEL_REAL else out_fake
            dst = label_dir / src.name

            # Avoid overwriting if name collides (rare but possible). Append a counter if needed.
            if dst.exists():
                duplicate_out += 1
                base = src.stem
                suf = src.suffix
                k = 1
                while dst.exists():
                    dst = label_dir / f"{base}__dup{k}{suf}"
                    k += 1

            transfer_file(src, dst, mode=mode)

            sha = sha256_of_file(dst) if compute_hashes else ""
            writer.writerow(
                {
                    "utt_id": utt_id,
                    "label": item.label,
                    "src_path": str(src),
                    "dst_path": str(dst),
                    "protocol_path": item.protocol_path,
                    "sha256": sha,
                    "raw_protocol_line": item.raw_line,
                }
            )
            rows_written += 1

    print(f"Prepared dataset -> {output_dir}")
    print(f"Protocol files used ({split}): {len(selected_protocols)}")
    print(f"Labeled utterances: {len(label_map)}")
    print(f"Files written: {rows_written}")
    print(f"Missing audio for protocol IDs: {missing_audio}")
    print(f"Output name collisions handled: {duplicate_out}")
    print(f"Manifest written: {manifest_path}")

    return manifest_path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        Parsed argparse namespace.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True, help="Root of ASVspoof2021 DF audio data")
    ap.add_argument(
        "--keys_tar_gz",
        default=None,
        help="Optional path to DF-keys-full.tar.gz (if provided, it will be extracted).",
    )
    ap.add_argument(
        "--keys_root",
        default=None,
        help="Path to extracted keys/metadata directory. If --keys_tar_gz is given, this can be omitted.",
    )
    ap.add_argument("--output_dir", required=True, help="Output directory containing real/ and fake/")
    ap.add_argument("--split", default="all", choices=["train", "dev", "eval", "all"])
    ap.add_argument(
        "--audio_subdir",
        default=None,
        help="Optional subdir under dataset_root to search for audio (e.g., 'flac').",
    )
    ap.add_argument("--mode", default="copy", choices=["copy", "move", "symlink"])
    ap.add_argument(
        "--exts",
        default=".flac,.wav",
        help="Comma-separated list of audio extensions to include (e.g., .flac,.wav).",
    )
    ap.add_argument(
        "--compute_hashes",
        action="store_true",
        help="Compute SHA-256 hashes for output files in the manifest (slower).",
    )
    return ap.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    exts = tuple(e.strip().lower() for e in args.exts.split(",") if e.strip())
    if not exts:
        raise ValueError("No extensions provided via --exts")

    keys_root: Optional[Path] = Path(args.keys_root).expanduser().resolve() if args.keys_root else None

    if args.keys_tar_gz:
        tar_path = Path(args.keys_tar_gz).expanduser().resolve()
        # If keys_root not provided, extract into output_dir/_keys_extracted
        if keys_root is None:
            keys_root = output_dir / "_keys_extracted"
        extract_tar_gz(tar_path, keys_root)

    if keys_root is None or not keys_root.exists():
        raise ValueError(
            "Either --keys_root must be provided (and exist), or --keys_tar_gz must be provided."
        )

    prepare_dataset(
        dataset_root=dataset_root,
        keys_root=keys_root,
        output_dir=output_dir,
        split=args.split,
        audio_subdir=args.audio_subdir,
        mode=args.mode,
        exts=exts,
        compute_hashes=args.compute_hashes,
    )


if __name__ == "__main__":
    main()
