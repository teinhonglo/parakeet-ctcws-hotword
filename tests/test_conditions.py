from __future__ import annotations

import json

import pytest

from hotword_asr.conditions import (
    build_hotwords_used,
    select_hotwords,
    validate_hotwords_used,
    write_hotwords_used,
)


HOTWORD_MAP = {"1": ["A", "B"], "2": ["C"]}
ALL_HOTWORDS = ["A", "B", "C", "D"]


def test_condition_selection_is_per_audio() -> None:
    assert select_hotwords("vanilla", "1", HOTWORD_MAP, ALL_HOTWORDS) == []
    assert select_hotwords("vanilla", "2", HOTWORD_MAP, ALL_HOTWORDS) == []
    assert select_hotwords("all_hotwords", "1", HOTWORD_MAP, ALL_HOTWORDS) == ALL_HOTWORDS
    assert select_hotwords("all_hotwords", "2", HOTWORD_MAP, ALL_HOTWORDS) == ALL_HOTWORDS
    oracle_1 = select_hotwords("oracle_hotwords", "1", HOTWORD_MAP, ALL_HOTWORDS)
    oracle_2 = select_hotwords("oracle_hotwords", "2", HOTWORD_MAP, ALL_HOTWORDS)
    assert oracle_1 == ["A", "B"]
    assert oracle_2 == ["C"]
    assert oracle_1 != ["A", "B", "C"]
    assert oracle_2 != ["A", "B", "C"]


def test_hotwords_used_audit_json(tmp_path) -> None:
    ids = ["1", "2"]
    expected = {
        "vanilla": {"1": [], "2": []},
        "all_hotwords": {"1": ALL_HOTWORDS, "2": ALL_HOTWORDS},
        "oracle_hotwords": HOTWORD_MAP,
    }
    for condition, value in expected.items():
        used = build_hotwords_used(condition, ids, HOTWORD_MAP, ALL_HOTWORDS)
        path = tmp_path / condition / "hotwords_used.json"
        write_hotwords_used(path, used)
        assert json.loads(path.read_text(encoding="utf-8")) == value


def test_validation_rejects_global_union_disguised_as_oracle() -> None:
    incorrect = {"1": ["A", "B", "C"], "2": ["A", "B", "C"]}
    with pytest.raises(AssertionError, match="oracle_hotwords.*audio 1"):
        validate_hotwords_used(
            "oracle_hotwords", incorrect, ["1", "2"], HOTWORD_MAP, ALL_HOTWORDS
        )
