from __future__ import annotations

import argparse
import hashlib
import tempfile
import time
from pathlib import Path
from typing import Any

from .hotwords import (
    compare_vocabularies,
    load_hotword_list,
    load_hotword_map,
)
from .io import write_json, write_transcription
from .metrics import RuntimeMeter
from .text_normalization import to_simplified_chinese, to_taiwan_traditional


DEFAULT_MODEL = "nvidia/nemotron-3.5-asr-streaming-0.6b"
MODEL_CARD = "https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b"
GPU_PB_DOCUMENTATION = (
    "https://docs.nvidia.com/nemo/speech/nightly/asr/"
    "asr_customization/word_boosting.html"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Nemotron 3.5 baseline and NeMo GPU Phrase Boosting on the "
            "hospital hotword benchmark"
        )
    )
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("exp/nemotron_gpu_pb")
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--chunk-seconds", type=float, default=30.0)
    parser.add_argument(
        "--target-lang",
        choices=("zh-CN",),
        default="zh-CN",
        help=(
            "Official Nemotron Mandarin locale. zh-TW is not listed as a "
            "supported transcription locale."
        ),
    )
    parser.add_argument(
        "--vocabulary-source",
        choices=("ground-truth-union", "all-hotwords"),
        default="ground-truth-union",
        help=(
            "Use the same global vocabulary selection rule as the Parakeet "
            "CTC-WS benchmark."
        ),
    )
    parser.add_argument(
        "--boosting-tree-alpha",
        type=float,
        default=1.0,
        help="GPU-PB shallow-fusion weight. NVIDIA recommends tuning it on the data.",
    )
    parser.add_argument(
        "--boosting-context-score",
        type=float,
        default=1.0,
        help="GPU-PB context graph arc score (NVIDIA recommendation: 1.0)",
    )
    parser.add_argument(
        "--boosting-depth-scaling",
        type=float,
        default=2.0,
        help="GPU-PB depth scaling for RNNT (NVIDIA recommendation: 2.0)",
    )
    parser.add_argument(
        "--boosting-bpe-mode",
        choices=("default", "case_insensitive", "bpe_dropout", "var_bpe"),
        default="case_insensitive",
    )
    parser.add_argument(
        "--condition",
        choices=("both", "baseline", "gpu-pb"),
        default="both",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Run only the first N files"
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def select_vocabulary(
    hotword_map: dict[str, list[str]],
    all_hotwords: list[str],
    source: str,
) -> list[str]:
    if source == "ground-truth-union":
        return sorted(
            {word for words in hotword_map.values() for word in words},
            key=str.casefold,
        )
    if source == "all-hotwords":
        return list(all_hotwords)
    raise ValueError(f"Unsupported vocabulary source: {source}")


def load_model(model_name: str, device: str) -> Any:
    import torch
    from nemo.collections.asr.models import ASRModel

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")

    model = ASRModel.from_pretrained(model_name=model_name)
    model = model.to(device)
    model.eval()
    model.freeze()
    if not hasattr(model, "cfg") or not hasattr(model.cfg, "decoding"):
        raise TypeError("Nemotron model does not expose an RNNT decoding config")
    if not hasattr(model, "change_decoding_strategy"):
        raise TypeError("Nemotron model cannot change its decoding strategy")
    return model


def prepare_gpu_pb_hotwords(
    vocabulary: list[str],
    output_dir: Path,
) -> tuple[Path, list[str], str]:
    simplified = [to_simplified_chinese(word) for word in vocabulary]
    if not all(word.strip() for word in simplified):
        raise ValueError("Traditional-to-Simplified conversion produced an empty hotword")

    conversion_map = [
        {"benchmark_term": source, "gpu_pb_zh_cn": converted}
        for source, converted in zip(vocabulary, simplified)
    ]
    write_json(output_dir / "hotword_conversion_zh_tw_to_zh_cn.json", conversion_map)

    phrase_text = "\n".join(simplified) + "\n"
    phrase_file = output_dir / "gpu_pb_hotwords.zh_cn.txt"
    phrase_file.parent.mkdir(parents=True, exist_ok=True)
    phrase_file.write_text(phrase_text, encoding="utf-8")

    duplicate_count = len(simplified) - len(set(simplified))
    if duplicate_count:
        print(
            "WARNING: tw2s conversion created "
            f"{duplicate_count} duplicate phrase entries. The source list and "
            "conversion map are preserved unchanged."
        )

    return (
        phrase_file,
        simplified,
        hashlib.sha256(phrase_text.encode("utf-8")).hexdigest(),
    )


def configure_greedy_decoding(
    model: Any,
    phrase_file: Path | None,
    *,
    boosting_tree_alpha: float,
    context_score: float,
    depth_scaling: float,
    bpe_mode: str,
) -> None:
    from omegaconf import open_dict

    decoding = model.cfg.decoding
    try:
        with open_dict(decoding):
            decoding.strategy = "greedy_batch"
            # NeMo places alpha on greedy, while phrase construction options are
            # nested under greedy.boosting_tree.
            decoding.greedy.boosting_tree_alpha = 0.0
            if phrase_file is not None:
                decoding.greedy.boosting_tree.key_phrases_file = str(phrase_file)
                decoding.greedy.boosting_tree.context_score = context_score
                decoding.greedy.boosting_tree.depth_scaling = depth_scaling
                decoding.greedy.boosting_tree.bpe_mode = bpe_mode
                decoding.greedy.boosting_tree_alpha = boosting_tree_alpha
        model.change_decoding_strategy(decoding)
    except (AttributeError, KeyError, TypeError) as error:
        raise RuntimeError(
            "The installed NeMo build does not expose the documented RNNT GPU-PB "
            "configuration. Re-run scripts/install.sh to install the required "
            "NeMo main-branch version."
        ) from error


def _transcription_text(item: Any) -> str:
    if isinstance(item, str):
        return item
    text = getattr(item, "text", None)
    if isinstance(text, str):
        return text
    raise TypeError(
        "Unexpected NeMo transcription type. Expected str or an object with a "
        f"string .text attribute, got {type(item).__name__}."
    )


def _transcribe_chunks(
    model: Any,
    chunk_paths: list[str],
    batch_size: int,
    target_lang: str,
) -> list[Any]:
    if not hasattr(model, "get_transcribe_config"):
        raise TypeError(
            "Nemotron prompt model does not expose get_transcribe_config()"
        )

    transcribe_config = model.get_transcribe_config()
    transcribe_config.batch_size = batch_size
    transcribe_config.num_workers = 0
    transcribe_config.verbose = False
    transcribe_config.return_hypotheses = False
    transcribe_config.target_lang = target_lang

    # Current NeMo's prompt-aware Lhotse path creates path-only manifest rows.
    # Their supervision language is therefore None, even when target_lang was
    # supplied to transcribe(), and prompt lookup fails with
    # "Unknown prompt key: 'None'". The non-Lhotse path yields a four-item
    # batch, so Nemotron's _transcribe_forward() constructs prompt indices
    # directly from transcribe_config.target_lang.
    transcribe_config.use_lhotse = False
    result = model.transcribe(
        chunk_paths,
        override_config=transcribe_config,
    )

    if isinstance(result, tuple):
        result = result[0]
    return list(result)


def transcribe_file(
    model: Any,
    audio_path: Path,
    *,
    batch_size: int,
    chunk_seconds: float,
    target_lang: str,
) -> dict[str, Any]:
    import torch

    from .engine import _load_audio, _write_chunks

    audio, sample_rate = _load_audio(audio_path)
    duration = len(audio) / sample_rate
    started = time.perf_counter()

    with tempfile.TemporaryDirectory(prefix="nemotron_chunks_") as tmp:
        chunk_meta = _write_chunks(
            audio, sample_rate, chunk_seconds, Path(tmp)
        )
        chunk_paths = [str(path) for path, _, _ in chunk_meta]
        with torch.inference_mode():
            hypotheses = _transcribe_chunks(
                model, chunk_paths, batch_size, target_lang
            )
        if next(model.parameters()).device.type == "cuda":
            torch.cuda.synchronize()

    if len(hypotheses) != len(chunk_meta):
        raise RuntimeError(
            f"NeMo returned {len(hypotheses)} hypotheses for "
            f"{len(chunk_meta)} chunks"
        )

    raw_zh_cn = " ".join(
        text.strip()
        for text in (_transcription_text(item) for item in hypotheses)
        if text.strip()
    )
    traditional = to_taiwan_traditional(raw_zh_cn)
    elapsed = time.perf_counter() - started
    return {
        "audio_path": str(audio_path.resolve()),
        "duration_sec": round(duration, 4),
        "raw_zh_cn_text": raw_zh_cn,
        "text": traditional,
        "timing": {
            "inference_seconds": round(elapsed, 4),
            "rtf": round(elapsed / duration, 6) if duration else None,
        },
    }


def run_condition(
    *,
    condition_name: str,
    model: Any,
    audio_ids: list[str],
    benchmark_dir: Path,
    output_dir: Path,
    device: str,
    batch_size: int,
    chunk_seconds: float,
    target_lang: str,
    overwrite: bool,
) -> dict[str, Any]:
    candidate_root = output_dir / (
        "raw_asr" if condition_name == "baseline" else "gpu_pb_asr"
    )
    raw_root = output_dir / "raw_zh_cn" / condition_name
    details_root = output_dir / "details" / condition_name
    per_audio: dict[str, dict[str, Any]] = {}
    inferred_audio_seconds = 0.0
    dataset_audio_seconds = 0.0
    inferred_count = 0

    meter = RuntimeMeter(device)
    meter.start()
    for index, audio_id in enumerate(audio_ids, start=1):
        audio_path = benchmark_dir / "audio" / f"{audio_id}.wav"
        if not audio_path.exists():
            raise FileNotFoundError(audio_path)
        details_path = details_root / f"{audio_id}.json"

        if details_path.exists() and not overwrite:
            from .hotwords import load_json

            result = load_json(details_path)
            print(
                f"[{condition_name} {index:02d}/{len(audio_ids)}] "
                f"{audio_id}: skip existing"
            )
        else:
            print(
                f"[{condition_name} {index:02d}/{len(audio_ids)}] "
                f"{audio_id}: {audio_path.name}"
            )
            result = transcribe_file(
                model,
                audio_path,
                batch_size=batch_size,
                chunk_seconds=chunk_seconds,
                target_lang=target_lang,
            )
            write_json(details_path, result)
            inferred_audio_seconds += float(result["duration_sec"])
            inferred_count += 1

        # Keep the model's zh-CN output unchanged for auditing. The shared
        # write_transcription() helper intentionally normalizes evaluator
        # candidates to Taiwan Traditional, so raw output uses write_json().
        write_json(
            raw_root / audio_id / "transcription.json",
            {"text": result["raw_zh_cn_text"]},
        )
        write_transcription(candidate_root, audio_id, result["text"])
        per_audio[audio_id] = result["timing"]
        dataset_audio_seconds += float(result["duration_sec"])

    runtime = meter.stop(inferred_audio_seconds)
    runtime.update(
        {
            "condition": condition_name,
            "dataset_audio_seconds": round(dataset_audio_seconds, 4),
            "audio_count": len(audio_ids),
            "inferred_audio_count_latest_run": inferred_count,
            "reused_audio_count_latest_run": len(audio_ids) - inferred_count,
            "per_audio": per_audio,
        }
    )
    return runtime


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.chunk_seconds <= 0:
        raise ValueError("--chunk-seconds must be positive")
    if args.boosting_tree_alpha < 0:
        raise ValueError("--boosting-tree-alpha cannot be negative")
    if args.boosting_context_score <= 0:
        raise ValueError("--boosting-context-score must be positive")
    if args.boosting_depth_scaling <= 0:
        raise ValueError("--boosting-depth-scaling must be positive")

    benchmark_dir = args.benchmark_dir.resolve()
    output_dir = args.output_dir.resolve()
    hotword_map = load_hotword_map(benchmark_dir / "hotwords.json")
    all_hotwords = load_hotword_list(benchmark_dir / "all_hotwords.json")
    vocabulary_check = compare_vocabularies(hotword_map, all_hotwords)
    vocabulary = select_vocabulary(
        hotword_map, all_hotwords, args.vocabulary_source
    )
    write_json(output_dir / "benchmark_vocabulary_check.json", vocabulary_check)

    phrase_file, simplified_vocabulary, phrase_sha256 = prepare_gpu_pb_hotwords(
        vocabulary, output_dir
    )

    audio_ids = sorted(hotword_map, key=int)
    if args.limit is not None:
        audio_ids = audio_ids[: args.limit]

    print(f"Loading model: {args.model}")
    model = load_model(args.model, args.device)
    runtime_metrics_path = output_dir / "runtime_metrics.json"
    runtime_metrics: dict[str, Any] = {}
    if runtime_metrics_path.exists():
        from .hotwords import load_json

        existing_metrics = load_json(runtime_metrics_path)
        if not isinstance(existing_metrics, dict):
            raise ValueError(f"Invalid runtime metrics: {runtime_metrics_path}")
        runtime_metrics.update(existing_metrics)

    if args.condition in {"both", "baseline"}:
        configure_greedy_decoding(
            model,
            None,
            boosting_tree_alpha=args.boosting_tree_alpha,
            context_score=args.boosting_context_score,
            depth_scaling=args.boosting_depth_scaling,
            bpe_mode=args.boosting_bpe_mode,
        )
        runtime_metrics["baseline"] = run_condition(
            condition_name="baseline",
            model=model,
            audio_ids=audio_ids,
            benchmark_dir=benchmark_dir,
            output_dir=output_dir,
            device=args.device,
            batch_size=args.batch_size,
            chunk_seconds=args.chunk_seconds,
            target_lang=args.target_lang,
            overwrite=args.overwrite,
        )

    if args.condition in {"both", "gpu-pb"}:
        configure_greedy_decoding(
            model,
            phrase_file,
            boosting_tree_alpha=args.boosting_tree_alpha,
            context_score=args.boosting_context_score,
            depth_scaling=args.boosting_depth_scaling,
            bpe_mode=args.boosting_bpe_mode,
        )
        runtime_metrics["gpu_pb"] = run_condition(
            condition_name="gpu_pb",
            model=model,
            audio_ids=audio_ids,
            benchmark_dir=benchmark_dir,
            output_dir=output_dir,
            device=args.device,
            batch_size=args.batch_size,
            chunk_seconds=args.chunk_seconds,
            target_lang=args.target_lang,
            overwrite=args.overwrite,
        )

    run_config = {
        "model": args.model,
        "model_card": MODEL_CARD,
        "device": args.device,
        "target_lang": args.target_lang,
        "language_note": (
            "The official model card lists Mandarin as zh-CN, not zh-TW. "
            "Hotwords are converted with OpenCC tw2s before GPU-PB and model "
            "output is converted with OpenCC s2tw before benchmark evaluation."
        ),
        "vocabulary_source": args.vocabulary_source,
        "source_vocabulary_size": len(vocabulary),
        "gpu_pb_vocabulary_size": len(simplified_vocabulary),
        "gpu_pb_phrase_file": str(phrase_file),
        "gpu_pb_phrase_sha256": phrase_sha256,
        "gpu_pb_documentation": GPU_PB_DOCUMENTATION,
        "decoding": {
            "strategy": "greedy_batch",
            "boosting_tree_alpha": args.boosting_tree_alpha,
            "context_score": args.boosting_context_score,
            "depth_scaling": args.boosting_depth_scaling,
            "bpe_mode": args.boosting_bpe_mode,
        },
        "transcription": {
            "use_lhotse": False,
            "reason": "preserve target_lang in Nemotron prompt conditioning",
        },
        "batch_size": args.batch_size,
        "chunk_seconds": args.chunk_seconds,
        "candidate_schema": {"text": "OpenCC s2tw transcription"},
        "evaluator": str(benchmark_dir / "evaluate.py"),
    }
    write_json(output_dir / "run_config.json", run_config)
    write_json(runtime_metrics_path, runtime_metrics)
    print(f"Done. Outputs: {output_dir}")


if __name__ == "__main__":
    main()
