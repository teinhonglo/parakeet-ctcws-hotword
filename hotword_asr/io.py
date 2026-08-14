from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .text_normalization import to_taiwan_traditional


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_transcription(candidate_root: str | Path, audio_id: str, text: str) -> None:
    # Matches hotword_benchmark/evaluate.py exactly.
    write_json(
        Path(candidate_root) / audio_id / "transcription.json",
        {"text": to_taiwan_traditional(text)},
    )
