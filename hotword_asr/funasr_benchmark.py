from __future__ import annotations

import argparse
import gc

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
from .provenance import require_matching_signature, run_signature
from .selection import select_audio_ids
from .text_normalization import to_taiwan_traditional


DEFAULT_MODEL = "FunAudioLLM/Fun-ASR-Nano-2512"
DEFAULT_MAX_SINGLE_SEGMENT_TIME = 15000
DEFAULT_MAX_LENGTH = 512


def truncate_repetition(
    text: str, min_repeat_len: int = 3, max_repeats: int = 3
) -> str:
    """Truncate runaway repeated spans using FunASR's production guard.

    This intentionally mirrors the guard in FunASR's official Nano vLLM
    service.  It keeps one copy of the first span observed three times in a
    row.  Short strings are left alone so ordinary conversational repetition
    is not touched.
    """
    if min_repeat_len <= 0:
        raise ValueError("min_repeat_len must be positive")
    if max_repeats < 2:
        raise ValueError("max_repeats must be at least 2")
    if not text or len(text) < 20:
        return text

    length_limit = min(len(text) // max_repeats, 30)
    for length in range(min_repeat_len, length_limit):
        for start in range(len(text) - length * max_repeats):
            chunk = text[start : start + length]
            if text[start : start + length * max_repeats] == chunk * max_repeats:
                return text[: start + length]
    return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the three Fun-ASR-Nano conditions")
    parser.add_argument("--benchmark-dir", required=True, type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=Path("exp/funasr_nano"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--condition", choices=("all", "vanilla", "all-hotwords", "oracle-hotwords"), default="all")
    parser.add_argument("--language", default="中文")
    parser.add_argument("--itn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--vad-model",
        default="funasr/fsmn-vad",
        help="Use the hub-specific model ID; funasr/fsmn-vad is the HF ID",
    )
    parser.add_argument(
        "--max-single-segment-time",
        type=int,
        default=DEFAULT_MAX_SINGLE_SEGMENT_TIME,
        help=(
            "Maximum VAD segment length in milliseconds. FunASR documents "
            "15-second chunks as the stable long-audio setting for Nano."
        ),
    )
    parser.add_argument(
        "--batch-size-s", type=float, default=30.0,
        help="Maximum accumulated VAD audio seconds per inference batch",
    )
    parser.add_argument("--hub", default="hf")
    parser.add_argument(
        "--max-length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help="Maximum new LLM tokens generated for each inference segment",
    )
    parser.add_argument(
        "--truncate-repetition",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply the repetition guard used by FunASR's official Nano service",
    )
    parser.add_argument("--repetition-min-length", type=int, default=3)
    parser.add_argument("--repetition-max-repeats", type=int, default=3)
    subset = parser.add_mutually_exclusive_group()
    subset.add_argument("--limit", type=int)
    subset.add_argument(
        "--audio-ids-file",
        type=Path,
        help="One audio ID per line; intended for a held-out tuning subset",
    )
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
    batch_size_s: float,
    max_length: int,
    truncate_repetitions: bool,
    repetition_min_length: int,
    repetition_max_repeats: int,
) -> dict[str, Any]:
    import torch

    duration = _wav_duration(audio_path)
    started = time.perf_counter()
    # inference_mode includes no_grad semantics and additionally disables
    # autograd bookkeeping/version tracking for this inference-only runner.
    with torch.inference_mode():
        result = model.generate(
            input=[str(audio_path)],
            cache={},
            language=language,
            hotwords=hotwords,
            itn=itn,
            batch_size_s=batch_size_s,
            max_length=max_length,
            llm_kwargs={"do_sample": False},
        )
    elapsed = time.perf_counter() - started
    if not isinstance(result, list) or len(result) != 1 or not isinstance(result[0], dict):
        raise TypeError("FunASR generate() must return one dictionary for one input")
    text = result[0].get("text")
    if not isinstance(text, str):
        raise TypeError("FunASR result dictionary does not contain string field 'text'")

    decoded_text = (
        truncate_repetition(
            text,
            min_repeat_len=repetition_min_length,
            max_repeats=repetition_max_repeats,
        )
        if truncate_repetitions
        else text
    )
        
    # FunASR's VAD path can leave large temporary generation buffers in the
    # CUDA caching allocator. Release them before the next audio/condition;
    # this does not unload the shared model.
    del result
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "audio_path": str(audio_path.resolve()),
        "duration_sec": round(duration, 4),
        "raw_text": text,
        "decoded_text": decoded_text,
        "repetition_guard": {
            "enabled": truncate_repetitions,
            "changed": decoded_text != text,
            "min_repeat_length": repetition_min_length,
            "max_repeats": repetition_max_repeats,
            "raw_characters": len(text),
            "decoded_characters": len(decoded_text),
        },
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
    if args.max_single_segment_time <= 0:
        raise ValueError("--max-single-segment-time must be positive")
    if args.batch_size_s <= 0:
        raise ValueError("--batch-size-s must be positive")
    if args.max_length <= 0:
        raise ValueError("--max-length must be positive")
    if args.repetition_min_length <= 0:
        raise ValueError("--repetition-min-length must be positive")
    if args.repetition_max_repeats < 2:
        raise ValueError("--repetition-max-repeats must be at least 2")

    benchmark_dir = args.benchmark_dir.resolve()
    output_dir = args.output_dir.resolve()
    hotword_map = load_hotword_map(benchmark_dir / "hotwords.json")
    all_hotwords = load_hotword_list(benchmark_dir / "all_hotwords.json")
    vocabulary_check = compare_vocabularies(hotword_map, all_hotwords)
    write_json(output_dir / "benchmark_vocabulary_check.json", vocabulary_check)
    audio_ids = select_audio_ids(
        sorted(hotword_map, key=int),
        limit=args.limit,
        audio_ids_file=args.audio_ids_file,
    )

    print(f"Loading Fun-ASR model once: {args.model}")
    model = model_loader(
        args.model, args.device, args.vad_model,
        args.max_single_segment_time, args.hub,
    )
    signature_base = {
        "backend": "funasr_nano",
        "model": args.model,
        "device": args.device,
        "language": args.language,
        "itn": args.itn,
        "hub": args.hub,
        "vad_model": args.vad_model,
        "max_single_segment_time": args.max_single_segment_time,
        "batch_size_s": args.batch_size_s,
        "max_length": args.max_length,
        "llm_kwargs": {"do_sample": False},
        "repetition_guard": {
            "enabled": args.truncate_repetition,
            "min_repeat_length": args.repetition_min_length,
            "max_repeats": args.repetition_max_repeats,
            "source": "FunASR official Nano service",
        },
        "output_normalization": "repetition-guard+OpenCC-s2tw",
    }
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
            signature = run_signature(
                {
                    **signature_base,
                    "condition": condition,
                    "audio_id": audio_id,
                    "hotwords": expected,
                }
            )
            if details_path.exists() and not args.overwrite:
                from .hotwords import load_json
                result = load_json(details_path)
                validate_hotwords_used(condition, {audio_id: result.get("hotwords_used")}, [audio_id], hotword_map, all_hotwords)
                if result.get("condition") != condition:
                    raise AssertionError(f"Cached details condition mismatch for {audio_id}")
                if result.get("model_hotwords") != expected:
                    raise AssertionError(f"Cached model hotword mismatch for {condition} audio {audio_id}")
                require_matching_signature(
                    result,
                    signature,
                    context=f"Fun-ASR {condition} audio {audio_id}",
                )
                print(f"[{condition} {index:02d}/{len(audio_ids)}] {audio_id}: skip existing")
            else:
                # This exact list is both sent to the model and retained in both audits.
                model_hotwords = list(expected)
                
                result = transcribe_file(
                    model, audio_path, hotwords=model_hotwords,
                    language=args.language, itn=args.itn,
                    batch_size_s=args.batch_size_s,
                    max_length=args.max_length,
                    truncate_repetitions=args.truncate_repetition,
                    repetition_min_length=args.repetition_min_length,
                    repetition_max_repeats=args.repetition_max_repeats,
                )
                evaluation_text = to_taiwan_traditional(result["decoded_text"])
                result.update(
                    audio_id=audio_id, condition=condition,
                    hotwords_used=list(expected), model_hotwords=model_hotwords,
                    language=args.language, itn=args.itn,
                    batch_size_s=args.batch_size_s,
                    max_length=args.max_length,
                    llm_kwargs={"do_sample": False},
                    evaluation_text=evaluation_text,
                    run_signature=signature,
                )
                write_json(details_path, result)
                inferred_seconds += float(result["duration_sec"])
                inferred_count += 1
            write_transcription(
                condition_dir / "asr", audio_id, result["evaluation_text"]
            )
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
        "batch_size_s": args.batch_size_s,
        "max_length": args.max_length,
        "llm_kwargs": {"do_sample": False},
        "repetition_guard": {
            "enabled": args.truncate_repetition,
            "min_repeat_length": args.repetition_min_length,
            "max_repeats": args.repetition_max_repeats,
            "source": (
                "https://github.com/modelscope/FunASR/blob/main/examples/"
                "industrial_data_pretraining/fun_asr_nano/serve_vllm.py"
            ),
        },
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
