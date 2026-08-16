from __future__ import annotations

import argparse
import time
import wave
from pathlib import Path
from typing import Any, Callable

from .conditions import (
    build_hotwords_used,
    normalize_condition,
    validate_hotwords_used,
    write_hotwords_used,
)
from .hotwords import compare_vocabularies, load_hotword_list, load_hotword_map
from .io import write_json, write_transcription
from .metrics import RuntimeMeter
from .text_normalization import to_taiwan_traditional


DEFAULT_MODEL = "FunAudioLLM/Fun-ASR-Nano-2512"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the three Fun-ASR-Nano conditions")
    parser.add_argument("--benchmark-dir", required=True, type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=Path("exp/funasr_nano"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--condition", choices=("all", "vanilla", "all-hotwords", "oracle-hotwords"), default="all")
    parser.add_argument("--language", default="中文")
    parser.add_argument("--itn", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--vad-model", default="fsmn-vad")
    parser.add_argument("--max-single-segment-time", type=int, default=30000)
    parser.add_argument("--hub", default="hf")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_model(
    model_name: str, device: str, vad_model: str, max_segment_ms: int, hub: str
) -> Any:
    from funasr import AutoModel

    return AutoModel(
        model=model_name,
        trust_remote_code=True,
        vad_model=vad_model,
        vad_kwargs={"max_single_segment_time": max_segment_ms},
        device=device,
        hub=hub,
    )


def _wav_duration(audio_path: Path) -> float:
    with wave.open(str(audio_path), "rb") as audio:
        return audio.getnframes() / float(audio.getframerate())


def transcribe_file(
    model: Any,
    audio_path: Path,
    *,
    hotwords: list[str],
    language: str,
    itn: bool,
) -> dict[str, Any]:
    duration = _wav_duration(audio_path)
    started = time.perf_counter()
    result = model.generate(
        input=[str(audio_path)],
        cache={},
        language=language,
        hotwords=hotwords,
        itn=itn,
    )
    elapsed = time.perf_counter() - started
    if not isinstance(result, list) or len(result) != 1 or not isinstance(result[0], dict):
        raise TypeError("FunASR generate() must return one dictionary for one input")
    text = result[0].get("text")
    if not isinstance(text, str):
        raise TypeError("FunASR result dictionary does not contain string field 'text'")
    return {
        "audio_path": str(audio_path.resolve()),
        "duration_sec": round(duration, 4),
        "raw_text": text,
        "timing": {
            "inference_seconds": round(elapsed, 4),
            "rtf": round(elapsed / duration, 6) if duration else None,
        },
    }


def run_benchmark(
    args: argparse.Namespace,
    *,
    model_loader: Callable[..., Any] = load_model,
) -> None:
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.max_single_segment_time <= 0:
        raise ValueError("--max-single-segment-time must be positive")

    benchmark_dir = args.benchmark_dir.resolve()
    output_dir = args.output_dir.resolve()
    hotword_map = load_hotword_map(benchmark_dir / "hotwords.json")
    all_hotwords = load_hotword_list(benchmark_dir / "all_hotwords.json")
    vocabulary_check = compare_vocabularies(hotword_map, all_hotwords)
    write_json(output_dir / "benchmark_vocabulary_check.json", vocabulary_check)
    audio_ids = sorted(hotword_map, key=int)
    if args.limit is not None:
        audio_ids = audio_ids[: args.limit]

    print(f"Loading Fun-ASR model once: {args.model}")
    model = model_loader(
        args.model, args.device, args.vad_model,
        args.max_single_segment_time, args.hub,
    )
    runtimes: dict[str, Any] = {}
    labels = {
        "vanilla": "Vanilla",
        "all_hotwords": "Hotword + All Hotwords",
        "oracle_hotwords": "Hotword + Oracle Hotwords",
    }
    for condition in normalize_condition(args.condition):
        print("=" * 40); print(f"Fun-ASR-Nano: {labels[condition]}"); print("=" * 40)
        condition_dir = output_dir / condition
        used = build_hotwords_used(condition, audio_ids, hotword_map, all_hotwords)
        validate_hotwords_used(condition, used, audio_ids, hotword_map, all_hotwords)
        write_hotwords_used(condition_dir / "hotwords_used.json", used)
        per_audio: dict[str, Any] = {}
        inferred_seconds = dataset_seconds = 0.0
        inferred_count = 0
        meter = RuntimeMeter(args.device)
        meter.start()
        for index, audio_id in enumerate(audio_ids, 1):
            audio_path = benchmark_dir / "audio" / f"{audio_id}.wav"
            if not audio_path.exists():
                raise FileNotFoundError(audio_path)
            details_path = condition_dir / "details" / f"{audio_id}.json"
            expected = used[audio_id]
            if details_path.exists() and not args.overwrite:
                from .hotwords import load_json
                result = load_json(details_path)
                validate_hotwords_used(condition, {audio_id: result.get("hotwords_used")}, [audio_id], hotword_map, all_hotwords)
                if result.get("condition") != condition:
                    raise AssertionError(f"Cached details condition mismatch for {audio_id}")
                if result.get("model_hotwords") != expected:
                    raise AssertionError(f"Cached model hotword mismatch for {condition} audio {audio_id}")
                print(f"[{condition} {index:02d}/{len(audio_ids)}] {audio_id}: skip existing")
            else:
                # This exact list is both sent to the model and retained in both audits.
                model_hotwords = list(expected)
                result = transcribe_file(model, audio_path, hotwords=model_hotwords, language=args.language, itn=args.itn)
                evaluation_text = to_taiwan_traditional(result["raw_text"])
                result.update(
                    audio_id=audio_id, condition=condition,
                    hotwords_used=list(expected), model_hotwords=model_hotwords,
                    language=args.language, itn=args.itn,
                    evaluation_text=evaluation_text,
                )
                write_json(details_path, result)
                inferred_seconds += float(result["duration_sec"])
                inferred_count += 1
            write_transcription(condition_dir / "asr", audio_id, result["raw_text"])
            per_audio[audio_id] = result["timing"]
            dataset_seconds += float(result["duration_sec"])
        runtime = meter.stop(inferred_seconds)
        runtime.update(condition=condition, dataset_audio_seconds=round(dataset_seconds, 4),
                       audio_count=len(audio_ids), inferred_audio_count_latest_run=inferred_count,
                       reused_audio_count_latest_run=len(audio_ids) - inferred_count,
                       per_audio=per_audio)
        runtimes[condition] = runtime

    write_json(output_dir / "runtime_metrics.json", runtimes)
    write_json(output_dir / "run_config.json", {
        "model": args.model, "device": args.device, "model_load_count": 1,
        "language": args.language, "itn": args.itn, "hub": args.hub,
        "vad_model": args.vad_model,
        "vad_kwargs": {"max_single_segment_time": args.max_single_segment_time},
        "conditions": {
            "vanilla": {"hotword_source": None},
            "all_hotwords": {"hotword_source": "all_hotwords.json", "scope": "global"},
            "oracle_hotwords": {"hotword_source": "hotwords.json[audio_id]", "scope": "per_audio_ground_truth"},
        },
        "selected_audio_ids": audio_ids,
    })


def main() -> None:
    run_benchmark(parse_args())


if __name__ == "__main__":
    main()
