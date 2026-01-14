#!/usr/bin/env python3
"""
Usage:
    python src/image_det/processing/extract_hugging_face_images.py \
  --root data/image_det/raw/hugging_face/ImageData \
  --out data/image_det/processed/hugging_face/imageData

Extract Hugging Face image dataset archives that are split into shards like:
  something.tar.gz.00, something.tar.gz.01, ...

Expected input layout (example):
  <root>/
    train/
      stylegan3-80K/
        stylegan3-80K.tar.gz.00
        stylegan3-80K.tar.gz.01
        ...
    val/   (or validation/ or valid/)
      ...

Output layout (safe, avoids collisions):
  <out>/
    train/
      <label>/
        <archive_name>/
          ...extracted files...
    val/
      <label>/
        <archive_name>/
          ...extracted files...

Notes:
- Uses streaming tar extraction from concatenated shard parts.
- Handles split folder aliases: val/valid/validation/dev.
- If your archives include non-images and you only want images, set --images-only.
- Uses a safe extraction check to prevent path traversal.
"""

from __future__ import annotations

import argparse
import re
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Matches e.g. stylegan3-80K.tar.gz.00
PART_RE = re.compile(r"^(?P<base>.+\.tar\.gz)\.(?P<part>\d+)$")

SPLIT_ALIASES = {
    "train": {"train", "training"},
    "val": {"val", "valid", "validation", "dev"},
    "test": {"test"},
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif"}


def find_split_dirs(root: Path) -> Dict[str, Path]:
    """
    Find split directories under root, allowing one-level nesting and common alias names.
    """
    candidates = [root]
    if root.exists():
        candidates += [p for p in root.iterdir() if p.is_dir()]

    found: Dict[str, Path] = {}
    for base in candidates:
        try:
            subdirs = {p.name.lower(): p for p in base.iterdir() if p.is_dir()}
        except FileNotFoundError:
            continue

        for canon, aliases in SPLIT_ALIASES.items():
            if canon in found:
                continue
            for a in aliases:
                if a in subdirs:
                    found[canon] = subdirs[a]
                    break
    return found


def group_sharded_archives(label_dir: Path) -> Dict[str, List[Path]]:
    """
    Group files like x.tar.gz.00, x.tar.gz.01 into:
      { 'x.tar.gz': [Path(...00), Path(...01), ...] }
    Searches recursively under label_dir.
    """
    groups: Dict[str, List[Path]] = defaultdict(list)

    for p in label_dir.rglob("*.tar.gz.*"):
        m = PART_RE.match(p.name)
        if not m:
            continue
        base = m.group("base")
        groups[base].append(p)

    # Sort parts numerically
    for base, parts in groups.items():
        groups[base] = sorted(parts, key=lambda x: int(PART_RE.match(x.name).group("part")))  # type: ignore

    return dict(groups)


class ConcatReader:
    """
    File-like object that sequentially reads from multiple shard files.
    Used with tarfile.open(fileobj=..., mode='r|gz') for streaming extraction.
    """

    def __init__(self, files: List[Path]):
        if not files:
            raise ValueError("ConcatReader received empty file list.")
        self.files = files
        self.idx = 0
        self.f = open(self.files[self.idx], "rb")

    def read(self, n: int = -1) -> bytes:
        if self.f is None:
            return b""
        chunk = self.f.read(n)
        if chunk:
            return chunk

        # Move to next file
        self.f.close()
        self.idx += 1
        if self.idx >= len(self.files):
            self.f = None
            return b""

        self.f = open(self.files[self.idx], "rb")
        return self.read(n)

    def close(self) -> None:
        if self.f is not None:
            self.f.close()
            self.f = None


def _is_within_directory(base: Path, target: Path) -> bool:
    """
    Prevent path traversal by ensuring target is within base.
    """
    try:
        target.resolve().relative_to(base.resolve())
        return True
    except Exception:
        return False


def _safe_members(
    tf: tarfile.TarFile,
    out_dir: Path,
    images_only: bool,
) -> List[tarfile.TarInfo]:
    """
    Filter tar members to prevent path traversal and optionally keep only images.
    """
    safe: List[tarfile.TarInfo] = []
    for m in tf:
        if not m.isfile():
            continue

        if images_only:
            suffix = Path(m.name).suffix.lower()
            if suffix not in IMAGE_EXTS:
                continue

        # Build intended extraction path (tarfile will join out_dir + m.name)
        dest = out_dir / m.name
        if not _is_within_directory(out_dir, dest):
            continue

        safe.append(m)
    return safe


def extract_sharded_tar_gz(
    parts: List[Path],
    out_dir: Path,
    images_only: bool = False,
) -> Tuple[int, int]:
    """
    Extract concatenated tar.gz shards using streaming tarfile mode.
    Returns (extracted_files, skipped_files_estimate).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    extracted = 0
    skipped = 0

    reader = ConcatReader(parts)
    try:
        # Streaming mode reads sequentially from fileobj (ideal for concatenated shards)
        with tarfile.open(fileobj=reader, mode="r|gz") as tf:
            members = _safe_members(tf, out_dir, images_only=images_only)
            # We already iterated tf in _safe_members; need to reopen to extract members in streaming mode.
            # Streaming tarfile cannot "rewind", so reopen the stream.
    finally:
        reader.close()

    # Reopen reader to actually extract (stream mode cannot seek)
    reader = ConcatReader(parts)
    try:
        with tarfile.open(fileobj=reader, mode="r|gz") as tf:
            # Iterate again and extract safely
            for m in tf:
                if not m.isfile():
                    continue

                if images_only:
                    suffix = Path(m.name).suffix.lower()
                    if suffix not in IMAGE_EXTS:
                        skipped += 1
                        continue

                dest = out_dir / m.name
                if not _is_within_directory(out_dir, dest):
                    skipped += 1
                    continue

                tf.extract(m, path=out_dir)
                extracted += 1
    finally:
        reader.close()

    return extracted, skipped


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract sharded Hugging Face tar.gz.00 archives into a training-friendly folder layout."
    )
    ap.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Dataset root containing train/ and val/ (or valid/validation/).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory for extracted files.",
    )
    ap.add_argument(
        "--images-only",
        action="store_true",
        help="Extract only image files (jpg/png/webp/...).",
    )
    ap.add_argument(
        "--flat",
        action="store_true",
        help="Flatten archive extraction into <out>/<split>/<label>/ (not recommended; may cause filename collisions).",
    )
    args = ap.parse_args()

    root: Path = args.root
    out: Path = args.out
    images_only: bool = args.images_only
    flat: bool = args.flat

    if not root.exists():
        raise SystemExit(f"[ERROR] Root path does not exist: {root}")

    split_dirs = find_split_dirs(root)
    if not split_dirs:
        raise SystemExit(
            f"[ERROR] No split folders found under {root}. Expected train and val/valid/validation."
        )

    # Only warn if missing; some datasets may not have val
    if "train" not in split_dirs:
        print(f"[WARN] No train split found under {root}")
    if "val" not in split_dirs:
        print(f"[WARN] No val/validation split found under {root}")

    for split in ["train", "val", "test"]:
        if split not in split_dirs:
            continue

        split_dir = split_dirs[split]
        label_dirs = [p for p in split_dir.iterdir() if p.is_dir()]

        print(f"\n=== Split: {split} ({split_dir}) | Labels: {len(label_dirs)} ===")

        for label_dir in sorted(label_dirs):
            label = label_dir.name

            groups = group_sharded_archives(label_dir)
            if not groups:
                print(f"[WARN] No *.tar.gz.XX shards found under {label_dir}")
                continue

            label_out = out / split / label
            label_out.mkdir(parents=True, exist_ok=True)

            print(f"  - {label}: {len(groups)} archive(s)")

            for base, parts in sorted(groups.items(), key=lambda kv: kv[0]):
                archive_name = base.replace(".tar.gz", "")

                if flat:
                    dest = label_out
                else:
                    dest = label_out / archive_name

                print(f"    Extracting {base} ({len(parts)} parts) -> {dest}")

                extracted, skipped = extract_sharded_tar_gz(
                    parts,
                    dest,
                    images_only=images_only,
                )
                print(f"      Done. Extracted: {extracted} | Skipped: {skipped}")

    print("\nDone.")


if __name__ == "__main__":
    main()
