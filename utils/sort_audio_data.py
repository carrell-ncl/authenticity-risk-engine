from pathlib import Path
import shutil


def organise_audio_by_protocol(
    source_audio_dir: str,
    protocol_file: str,
    target_root_dir: str,
    move: bool = True,
    audio_exts=(".wav", ".flac", ".mp3")
):
    """
    Organise mixed audio files into real/ and fake/ folders using an ASVspoof protocol file.

    Args:
        source_audio_dir (str): Directory containing mixed audio files.
        protocol_file (str): Path to ASVspoof protocol file.
        target_root_dir (str): Output directory; will create real/ and fake/ subdirs.
        move (bool): If True, move files. If False, copy files.
        audio_exts (tuple): Allowed audio extensions.

    Returns:
        dict: Summary with counts and missing files.
    """

    source_audio_dir = Path(source_audio_dir)
    protocol_file = Path(protocol_file)
    target_root_dir = Path(target_root_dir)

    real_dir = target_root_dir / "real"
    fake_dir = target_root_dir / "fake"

    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    # --- Parse protocol ---
    utt_to_label = {}

    with protocol_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            utt_id = parts[1]
            label = parts[-1].lower()

            if label not in {"bonafide", "spoof"}:
                continue

            utt_to_label[utt_id] = label

    moved_real = 0
    moved_fake = 0
    missing = []

    # --- Process files ---
    for utt_id, label in utt_to_label.items():
        src = None
        for ext in audio_exts:
            candidate = source_audio_dir / f"{utt_id}{ext}"
            if candidate.exists():
                src = candidate
                break

        if src is None:
            missing.append(utt_id)
            continue

        dst_dir = real_dir if label == "bonafide" else fake_dir
        dst = dst_dir / src.name

        if move:
            shutil.move(str(src), str(dst))
        else:
            shutil.copy2(str(src), str(dst))

        if label == "bonafide":
            moved_real += 1
        else:
            moved_fake += 1

    return {
        "real": moved_real,
        "fake": moved_fake,
        "missing": missing,
        "total_protocol_entries": len(utt_to_label),
    }
