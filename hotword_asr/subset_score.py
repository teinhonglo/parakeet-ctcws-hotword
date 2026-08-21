from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

from .io import write_json
from .selection import select_audio_ids


def _load_evaluator(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("hotword_benchmark_evaluator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("compute_mer", "preprocess_text"):
        if not callable(getattr(module, name, None)):
            raise TypeError(f"Evaluator does not expose {name}(): {path}")
    return module


def _candidate_text(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        text = value.get("text", "")
        if not isinstance(text, str):
            raise TypeError(f"Candidate text must be a string: {path}")
        return text
    if isinstance(value, list):
        parts = []
        for segment in value:
            if not isinstance(segment, dict) or not isinstance(segment.get("text", ""), str):
                raise TypeError(f"Invalid candidate segment: {path}")
            parts.append(segment.get("text", ""))
        return " ".join(parts)
    raise TypeError(f"Candidate must be a dictionary or list: {path}")


def score_subset(
    *,
    benchmark_dir: Path,
    candidate_dir: Path,
    audio_ids_file: Path,
) -> dict[str, Any]:
    benchmark_dir = benchmark_dir.resolve()
    candidate_dir = candidate_dir.resolve()
    hotwords = json.loads((benchmark_dir / "hotwords.json").read_text(encoding="utf-8"))
    references = json.loads(
        (benchmark_dir / "pseudo_transcripts.json").read_text(encoding="utf-8")
    )
    if not isinstance(hotwords, dict) or not isinstance(references, dict):
        raise TypeError("Benchmark hotwords and pseudo transcripts must be dictionaries")
    available = sorted(hotwords, key=int)
    selected = select_audio_ids(available, audio_ids_file=audio_ids_file)
    missing_references = sorted(set(selected) - set(references))
    if missing_references:
        raise ValueError(f"Missing pseudo transcripts: {missing_references}")

    evaluator_path = benchmark_dir / "evaluate.py"
    evaluator = _load_evaluator(evaluator_path)
    if callable(getattr(evaluator, "validate_number_itn", None)):
        evaluator.validate_number_itn()

    target_count = hit_count = total_errors = total_ref_tokens = 0
    per_audio: dict[str, Any] = {}
    for audio_id in selected:
        text = _candidate_text(candidate_dir / audio_id / "transcription.json")
        normalized_text = evaluator.preprocess_text(text)
        targets = hotwords[audio_id]
        if not isinstance(targets, list) or not all(isinstance(word, str) for word in targets):
            raise TypeError(f"Hotwords for audio {audio_id} must be a list of strings")
        hits = []
        for word in targets:
            normalized_word = evaluator.preprocess_text(word)
            if not normalized_word:
                raise ValueError(f"Hotword for audio {audio_id} normalizes to empty: {word!r}")
            if normalized_word in normalized_text:
                hits.append(word)
        errors, ref_count, *_ = evaluator.compute_mer(references[audio_id], text)
        target_count += len(targets)
        hit_count += len(hits)
        total_errors += int(errors)
        total_ref_tokens += int(ref_count)
        per_audio[audio_id] = {
            "target_count": len(targets),
            "hit_count": len(hits),
            "errors": int(errors),
            "ref_tokens": int(ref_count),
        }

    return {
        "evaluator": str(evaluator_path),
        "evaluator_version": getattr(evaluator, "EVALUATOR_VERSION", None),
        "audio_ids_file": str(audio_ids_file.resolve()),
        "audio_ids": selected,
        "audio_count": len(selected),
        "hotword_hits": hit_count,
        "hotword_targets": target_count,
        "hotword_recall": hit_count / target_count if target_count else None,
        "edit_errors": total_errors,
        "reference_tokens": total_ref_tokens,
        "mer": total_errors / total_ref_tokens if total_ref_tokens else None,
        "per_audio": per_audio,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score only an explicit development subset with benchmark evaluate.py"
    )
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--audio-ids-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = score_subset(
        benchmark_dir=args.benchmark_dir,
        candidate_dir=args.candidate_dir,
        audio_ids_file=args.audio_ids_file,
    )
    write_json(args.output, result)
    print(
        f"audio={result['audio_count']} recall={result['hotword_recall']:.2%} "
        f"MER={result['mer']:.2%}"
    )


if __name__ == "__main__":
    main()
