from __future__ import annotations

from pathlib import Path


def select_audio_ids(
    available_ids: list[str],
    *,
    limit: int | None = None,
    audio_ids_file: Path | None = None,
) -> list[str]:
    """Select an auditable subset without silently changing its order."""
    if limit is not None and audio_ids_file is not None:
        raise ValueError("--limit and --audio-ids-file are mutually exclusive")
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive")
        return available_ids[:limit]
    if audio_ids_file is None:
        return available_ids

    path = audio_ids_file.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    selected = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not selected:
        raise ValueError(f"Audio ID file is empty: {path}")
    duplicates = sorted({audio_id for audio_id in selected if selected.count(audio_id) > 1})
    if duplicates:
        raise ValueError(f"Duplicate audio IDs in {path}: {duplicates}")
    unknown = sorted(set(selected) - set(available_ids))
    if unknown:
        raise ValueError(f"Unknown audio IDs in {path}: {unknown}")
    return selected
