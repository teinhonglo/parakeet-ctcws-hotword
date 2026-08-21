import json
from pathlib import Path

from hotword_asr.subset_score import score_subset


def test_score_subset_uses_only_explicit_ids_and_benchmark_evaluator(
    tmp_path: Path,
) -> None:
    benchmark = tmp_path / "benchmark"
    candidate = tmp_path / "candidate"
    benchmark.mkdir()
    (benchmark / "hotwords.json").write_text(
        json.dumps({"1": ["HOT"], "2": ["SKIP"]}), encoding="utf-8"
    )
    (benchmark / "pseudo_transcripts.json").write_text(
        json.dumps({"1": "hot text", "2": "skip text"}), encoding="utf-8"
    )
    (benchmark / "evaluate.py").write_text(
        """
EVALUATOR_VERSION = "test-v1"
def validate_number_itn():
    return None
def preprocess_text(text):
    return ''.join(text.lower().split())
def compute_mer(reference, hypothesis):
    ref = reference.lower().split()
    hyp = hypothesis.lower().split()
    errors = sum(a != b for a, b in zip(ref, hyp)) + abs(len(ref) - len(hyp))
    return errors, len(ref), ref, hyp
""".strip(),
        encoding="utf-8",
    )
    ids = tmp_path / "dev_ids.txt"
    ids.write_text("1\n", encoding="utf-8")
    transcript = candidate / "1" / "transcription.json"
    transcript.parent.mkdir(parents=True)
    transcript.write_text(json.dumps({"text": "HOT text"}), encoding="utf-8")

    result = score_subset(
        benchmark_dir=benchmark,
        candidate_dir=candidate,
        audio_ids_file=ids,
    )

    assert result["audio_ids"] == ["1"]
    assert result["evaluator_version"] == "test-v1"
    assert result["hotword_recall"] == 1.0
    assert result["mer"] == 0.0
