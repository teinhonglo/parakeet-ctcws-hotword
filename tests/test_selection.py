from pathlib import Path

import pytest

from hotword_asr.selection import select_audio_ids


def test_select_audio_ids_preserves_explicit_file_order(tmp_path: Path) -> None:
    ids = tmp_path / "dev_ids.txt"
    ids.write_text("# development set\n517\n503\n", encoding="utf-8")

    assert select_audio_ids(["503", "517", "538"], audio_ids_file=ids) == [
        "517",
        "503",
    ]


def test_select_audio_ids_rejects_unknown_or_duplicate_ids(tmp_path: Path) -> None:
    ids = tmp_path / "dev_ids.txt"
    ids.write_text("503\n999\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown audio IDs"):
        select_audio_ids(["503", "517"], audio_ids_file=ids)

    ids.write_text("503\n503\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate audio IDs"):
        select_audio_ids(["503", "517"], audio_ids_file=ids)
