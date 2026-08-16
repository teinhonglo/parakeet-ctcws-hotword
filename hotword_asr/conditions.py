from __future__ import annotations

from pathlib import Path

CONDITIONS = ("vanilla", "all_hotwords", "oracle_hotwords")


def normalize_condition(value: str) -> list[str]:
    value = value.replace("-", "_")
    if value == "all":
        return list(CONDITIONS)
    if value not in CONDITIONS:
        raise ValueError(f"Unsupported condition: {value}")
    return [value]


def select_hotwords(
    condition: str,
    audio_id: str,
    hotword_map: dict[str, list[str]],
    all_hotwords: list[str],
) -> list[str]:
    if condition == "vanilla":
        return []
    if condition == "all_hotwords":
        return list(all_hotwords)
    if condition == "oracle_hotwords":
        return list(hotword_map[audio_id])
    raise ValueError(f"Unsupported condition: {condition}")


def build_hotwords_used(
    condition: str,
    audio_ids: list[str],
    hotword_map: dict[str, list[str]],
    all_hotwords: list[str],
) -> dict[str, list[str]]:
    used = {
        audio_id: select_hotwords(condition, audio_id, hotword_map, all_hotwords)
        for audio_id in audio_ids
    }
    validate_hotwords_used(condition, used, audio_ids, hotword_map, all_hotwords)
    return used


def validate_hotwords_used(
    condition: str,
    used: dict[str, list[str]],
    audio_ids: list[str],
    hotword_map: dict[str, list[str]],
    all_hotwords: list[str],
) -> None:
    """Fail fast unless every audio received the condition's exact vocabulary."""
    for audio_id in audio_ids:
        expected = select_hotwords(condition, audio_id, hotword_map, all_hotwords)
        if audio_id not in used:
            raise AssertionError(
                f"{condition} hotword audit is missing audio {audio_id}"
            )
        if used[audio_id] != expected:
            raise AssertionError(
                f"{condition} hotword audit mismatch for audio {audio_id}: "
                f"{used[audio_id]!r} != {expected!r}"
            )


def write_hotwords_used(path: Path, used: dict[str, list[str]]) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(used, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
