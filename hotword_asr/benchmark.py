from __future__ import annotations

import argparse
import time
from pathlib import Path

from .engine import CTCWSConfig, CTCWordSpotterASR
from .hotwords import (
    compare_vocabularies,
    load_aliases,
    load_hotword_list,
    load_hotword_map,
)
from .io import write_json, write_transcription
from .metrics import RuntimeMeter
from .model import load_ctc_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Parakeet CTC + NeMo CTC-WS on the hospital hotword benchmark"
    )
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument(
        "--model",
        required=True,
        help=".nemo file/directory from NGC, or a NeMo pretrained model id",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("exp/parakeet_ctcws"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--chunk-seconds", type=float, default=30.0)
    parser.add_argument("--beam-threshold", type=float, default=7.0)
    parser.add_argument("--context-score", type=float, default=3.0)
    parser.add_argument("--ctc-ali-token-weight", type=float, default=0.5)
    parser.add_argument("--aliases", type=Path)
    parser.add_argument(
        "--vocabulary-source",
        choices=("ground-truth-union", "all-hotwords"),
        default="ground-truth-union",
        help=(
            "Global spotting vocabulary. ground-truth-union uses the union across all 71 "
            "files, never a per-audio list. This also keeps the current elbew label consistent."
        ),
    )
    parser.add_argument(
        "--no-auto-variants",
        action="store_true",
        help="Disable English case/hyphen/acronym alternative CTC paths",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Run only the first N files for a smoke test"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Re-run audio ids that already have details JSON"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmark_dir = args.benchmark_dir.resolve()
    output_dir = args.output_dir.resolve()
    hotword_map = load_hotword_map(benchmark_dir / "hotwords.json")
    all_hotwords = load_hotword_list(benchmark_dir / "all_hotwords.json")
    vocabulary_check = compare_vocabularies(hotword_map, all_hotwords)

    if vocabulary_check["missing_from_vocabulary"] or vocabulary_check["extra_in_vocabulary"]:
        print("WARNING: hotwords.json and all_hotwords.json do not define the same vocabulary:")
        print(f"  missing from all_hotwords: {vocabulary_check['missing_from_vocabulary']}")
        print(f"  extra in all_hotwords:     {vocabulary_check['extra_in_vocabulary']}")

    if args.vocabulary_source == "ground-truth-union":
        vocabulary = sorted(
            {word for words in hotword_map.values() for word in words}, key=str.casefold
        )
    else:
        vocabulary = all_hotwords

    aliases = load_aliases(args.aliases)
    config = CTCWSConfig(
        beam_threshold=args.beam_threshold,
        context_score=args.context_score,
        ctc_ali_token_weight=args.ctc_ali_token_weight,
        chunk_seconds=args.chunk_seconds,
        batch_size=args.batch_size,
        auto_variants=not args.no_auto_variants,
    )

    print(f"Loading model: {args.model}")
    model = load_ctc_model(args.model, args.device)
    print(f"Building global CTC-WS graph with {len(vocabulary)} canonical hotwords")
    engine = CTCWordSpotterASR(model, vocabulary, config=config, aliases=aliases)
    write_json(output_dir / "ctcws_text_variants.json", engine.used_variants)
    write_json(output_dir / "benchmark_vocabulary_check.json", vocabulary_check)

    audio_ids = sorted(hotword_map, key=int)
    if args.limit is not None:
        audio_ids = audio_ids[: args.limit]

    raw_root = output_dir / "raw_asr"
    merged_root = output_dir / "ctcws_asr"
    details_root = output_dir / "details"
    predicted_map: dict[str, list[str]] = {}
    per_audio_timing: dict[str, dict] = {}
    total_audio_seconds = 0.0

    meter = RuntimeMeter(args.device)
    meter.start()
    run_started = time.time()

    for index, audio_id in enumerate(audio_ids, start=1):
        audio_path = benchmark_dir / "audio" / f"{audio_id}.wav"
        if not audio_path.exists():
            raise FileNotFoundError(audio_path)
        details_path = details_root / f"{audio_id}.json"

        if details_path.exists() and not args.overwrite:
            from .hotwords import load_json

            result = load_json(details_path)
            print(f"[{index:02d}/{len(audio_ids)}] {audio_id}: skip existing")
        else:
            print(f"[{index:02d}/{len(audio_ids)}] {audio_id}: {audio_path.name}")
            result = engine.transcribe_file(audio_path)
            write_json(details_path, result)

        write_transcription(raw_root, audio_id, result["raw_text"])
        write_transcription(merged_root, audio_id, result["merged_text"])
        predicted_map[audio_id] = result["predicted_hotwords"]
        per_audio_timing[audio_id] = result["timing"]
        total_audio_seconds += float(result["duration_sec"])

    write_json(output_dir / "predicted_keywords.json", predicted_map)
    runtime = meter.stop(total_audio_seconds)
    runtime.update(
        {
            "processed_audio_count": len(audio_ids),
            "model": args.model,
            "device": args.device,
            "vocabulary_source": args.vocabulary_source,
            "vocabulary_size": len(vocabulary),
            "unix_started_at": run_started,
            "per_audio": per_audio_timing,
        }
    )
    write_json(output_dir / "runtime_metrics.json", runtime)

    print(f"Done. Outputs: {output_dir}")
    print(
        f"RTF={runtime['rtf']}  speed={runtime['throughput_x_realtime']}x  "
        f"peak_gpu_allocated={runtime['peak_gpu_allocated_mb']} MB"
    )


if __name__ == "__main__":
    main()

