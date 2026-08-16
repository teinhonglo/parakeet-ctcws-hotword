from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path
from unittest.mock import Mock

import pytest

from hotword_asr.funasr_benchmark import run_benchmark


def _benchmark(root: Path) -> Path:
    benchmark = root / "benchmark"
    (benchmark / "audio").mkdir(parents=True)
    (benchmark / "hotwords.json").write_text(
        json.dumps({"1": ["A", "B"], "2": ["C"]}), encoding="utf-8"
    )
    (benchmark / "all_hotwords.json").write_text(
        json.dumps(["A", "B", "C", "D"]), encoding="utf-8"
    )
    for audio_id in ("1", "2"):
        with wave.open(str(benchmark / "audio" / f"{audio_id}.wav"), "wb") as wav:
            wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(16000)
            wav.writeframes(b"\0\0" * 160)
    return benchmark


def _args(benchmark: Path, output: Path, **updates) -> argparse.Namespace:
    values = dict(
        benchmark_dir=benchmark, output_dir=output, model="test-model",
        device="cpu", condition="all", language="中文", itn=False,
        vad_model="fsmn-vad", max_single_segment_time=30000, hub="hf",
        batch_size_s=30.0,
        limit=None, overwrite=False,
    )
    values.update(updates)
    return argparse.Namespace(**values)


def test_all_conditions_pass_exact_hotwords_and_load_model_once(
    tmp_path: Path, monkeypatch
) -> None:
    benchmark, output = _benchmark(tmp_path), tmp_path / "output"
    # Exercise the shared write_transcription path without requiring OpenCC in
    # the model-free unit-test environment.
    monkeypatch.setattr(
        "hotword_asr.io.to_taiwan_traditional",
        lambda text: text.replace("肾", "腎"),
    )
    monkeypatch.setattr(
        "hotword_asr.funasr_benchmark.to_taiwan_traditional",
        lambda text: text.replace("肾", "腎"),
    )
    model = Mock()
    model.generate.side_effect = [
        [{"text": "洗肾"}], [{"text": "洗肾"}],
        [{"text": "洗肾"}], [{"text": "洗肾"}],
        [{"text": "洗肾"}], [{"text": "洗肾"}],
    ]
    loader = Mock(return_value=model)

    run_benchmark(_args(benchmark, output), model_loader=loader)

    assert loader.call_count == 1
    actual = [call.kwargs["hotwords"] for call in model.generate.call_args_list]
    assert actual == [
        [], [],
        ["A", "B", "C", "D"], ["A", "B", "C", "D"],
        ["A", "B"], ["C"],
    ]
    assert all(call.kwargs["batch_size_s"] == 30.0 for call in model.generate.call_args_list)
    for condition in ("vanilla", "all_hotwords", "oracle_hotwords"):
        for audio_id in ("1", "2"):
            assert (output / condition / "asr" / audio_id / "transcription.json").is_file()
    candidate = json.loads((output / "vanilla/asr/1/transcription.json").read_text())
    assert candidate == {"text": "洗腎"}
    config = json.loads((output / "run_config.json").read_text())
    assert config["model_load_count"] == 1


def test_cache_rejects_global_vocabulary_disguised_as_oracle(tmp_path: Path) -> None:
    benchmark, output = _benchmark(tmp_path), tmp_path / "output"
    details = output / "oracle_hotwords/details/1.json"
    details.parent.mkdir(parents=True)
    details.write_text(json.dumps({
        "condition": "oracle_hotwords",
        "hotwords_used": ["A", "B", "C", "D"],
        "model_hotwords": ["A", "B", "C", "D"],
    }), encoding="utf-8")
    model = Mock()
    loader = Mock(return_value=model)

    with pytest.raises(AssertionError, match="oracle_hotwords.*audio 1"):
        run_benchmark(
            _args(benchmark, output, condition="oracle-hotwords", limit=1),
            model_loader=loader,
        )
    model.generate.assert_not_called()
