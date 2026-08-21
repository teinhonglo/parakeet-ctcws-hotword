from __future__ import annotations

import numpy as np
import soundfile as sf

from hotword_asr.engine import CTCWSConfig, _write_chunks


def test_accuracy_baseline_uses_bounded_chunks_and_exact_hotwords() -> None:
    config = CTCWSConfig()
    assert config.chunk_seconds == 30.0
    assert config.auto_variants is False


def test_zero_chunk_seconds_writes_one_complete_utterance(tmp_path) -> None:
    sample_rate = 16_000
    audio = np.arange(sample_rate * 65, dtype=np.float32) / (sample_rate * 65)

    chunks = _write_chunks(audio, sample_rate, 0.0, tmp_path)

    assert len(chunks) == 1
    chunk_audio, chunk_rate = sf.read(chunks[0][0], dtype="float32")
    assert chunk_rate == sample_rate
    assert len(chunk_audio) == len(audio)
    assert chunks[0][1:] == (0.0, 65.0)


def test_positive_chunk_seconds_remains_available_for_memory_fallback(tmp_path) -> None:
    sample_rate = 10
    audio = np.zeros(25, dtype=np.float32)

    chunks = _write_chunks(audio, sample_rate, 1.0, tmp_path)

    assert [(start, end) for _, start, end in chunks] == [
        (0.0, 1.0),
        (1.0, 2.0),
        (2.0, 2.5),
    ]
