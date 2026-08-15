from __future__ import annotations

import contextlib
import sys
import types
from pathlib import Path

from hotword_asr.nemotron_benchmark import (
    _transcription_text,
    configure_greedy_decoding,
    prepare_gpu_pb_hotwords,
    select_vocabulary,
)


class Converter:
    def convert(self, text: str) -> str:
        return text.replace("洗腎", "洗肾").replace("苑裡", "苑里")


class Node:
    pass


class Model:
    def __init__(self) -> None:
        self.cfg = Node()
        self.cfg.decoding = Node()
        self.cfg.decoding.strategy = "original"
        self.cfg.decoding.greedy = Node()
        self.cfg.decoding.greedy.boosting_tree_alpha = None
        self.cfg.decoding.greedy.boosting_tree = Node()
        tree = self.cfg.decoding.greedy.boosting_tree
        tree.key_phrases_file = None
        tree.context_score = None
        tree.depth_scaling = None
        tree.bpe_mode = None
        self.changed = False

    def change_decoding_strategy(self, decoding) -> None:
        assert decoding is self.cfg.decoding
        self.changed = True


def install_fake_omegaconf(monkeypatch) -> None:
    module = types.ModuleType("omegaconf")
    module.open_dict = lambda _: contextlib.nullcontext()
    monkeypatch.setitem(sys.modules, "omegaconf", module)


def test_select_vocabulary_matches_existing_benchmark_policy() -> None:
    hotword_map = {"1": ["洗腎", "elbew"], "2": ["洗腎", "X-RAY"]}
    all_hotwords = ["洗腎", "elbow", "X-RAY"]
    assert select_vocabulary(
        hotword_map, all_hotwords, "ground-truth-union"
    ) == ["elbew", "X-RAY", "洗腎"]
    assert select_vocabulary(
        hotword_map, all_hotwords, "all-hotwords"
    ) == all_hotwords


def test_prepare_gpu_pb_hotwords_preserves_source_entries(tmp_path: Path) -> None:
    phrase_file, simplified, _ = prepare_gpu_pb_hotwords(
        ["洗腎", "苑裡", "X-RAY"], tmp_path, Converter()
    )
    assert simplified == ["洗肾", "苑里", "X-RAY"]
    assert phrase_file.read_text(encoding="utf-8").splitlines() == simplified


def test_baseline_and_gpu_pb_use_same_greedy_decoder(
    tmp_path: Path, monkeypatch
) -> None:
    install_fake_omegaconf(monkeypatch)
    model = Model()
    configure_greedy_decoding(
        model,
        None,
        boosting_tree_alpha=0.75,
        context_score=1.0,
        depth_scaling=2.0,
        bpe_mode="case_insensitive",
    )
    assert model.changed
    assert model.cfg.decoding.strategy == "greedy_batch"
    assert model.cfg.decoding.greedy.boosting_tree_alpha == 0.0

    phrase_file = tmp_path / "phrases.txt"
    phrase_file.write_text("洗肾\n", encoding="utf-8")
    configure_greedy_decoding(
        model,
        phrase_file,
        boosting_tree_alpha=0.75,
        context_score=1.0,
        depth_scaling=2.0,
        bpe_mode="case_insensitive",
    )
    tree = model.cfg.decoding.greedy.boosting_tree
    assert model.cfg.decoding.greedy.boosting_tree_alpha == 0.75
    assert tree.key_phrases_file == str(phrase_file)
    assert tree.context_score == 1.0
    assert tree.depth_scaling == 2.0
    assert tree.bpe_mode == "case_insensitive"


def test_transcription_text_accepts_string_and_hypothesis() -> None:
    hypothesis = Node()
    hypothesis.text = "洗肾"
    assert _transcription_text("苑里") == "苑里"
    assert _transcription_text(hypothesis) == "洗肾"
