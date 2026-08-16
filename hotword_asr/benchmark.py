from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from .conditions import (
    build_hotwords_used,
    normalize_condition,
    validate_hotwords_used,
    write_hotwords_used,
)
from .engine import CTCWSConfig, CTCWordSpotterASR
from .hotwords import compare_vocabularies, load_aliases, load_hotword_list, load_hotword_map
from .io import write_json, write_transcription
from .metrics import RuntimeMeter
from .model import load_ctc_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the three Parakeet benchmark conditions")
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("exp/parakeet_ctcws"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--chunk-seconds", type=float, default=30.0)
    parser.add_argument("--beam-threshold", type=float, default=7.0)
    parser.add_argument("--context-score", type=float, default=3.0)
    parser.add_argument("--ctc-ali-token-weight", type=float, default=0.5)
    parser.add_argument("--aliases", type=Path)
    parser.add_argument("--condition", choices=("all", "vanilla", "all-hotwords", "oracle-hotwords"), default="all")
    parser.add_argument("--no-auto-variants", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _heading(condition: str) -> None:
    labels = {"vanilla": "Vanilla", "all_hotwords": "CTC-WS + All Hotwords", "oracle_hotwords": "CTC-WS + Oracle Hotwords"}
    print("=" * 40); print(f"Parakeet: {labels[condition]}"); print("=" * 40)


def run_benchmark(args: argparse.Namespace) -> None:
    benchmark_dir, output_dir = args.benchmark_dir.resolve(), args.output_dir.resolve()
    hotword_map = load_hotword_map(benchmark_dir / "hotwords.json")
    all_hotwords = load_hotword_list(benchmark_dir / "all_hotwords.json")
    check = compare_vocabularies(hotword_map, all_hotwords)
    write_json(output_dir / "benchmark_vocabulary_check.json", check)
    if check["missing_from_vocabulary"] or check["extra_in_vocabulary"]:
        print("WARNING: hotwords.json and all_hotwords.json differ:", check)
    audio_ids = sorted(hotword_map, key=int)
    if args.limit is not None: audio_ids = audio_ids[:args.limit]
    conditions = normalize_condition(args.condition)
    aliases = load_aliases(args.aliases)
    config = CTCWSConfig(args.beam_threshold, args.context_score, args.ctc_ali_token_weight, args.chunk_seconds, args.batch_size, not args.no_auto_variants)

    print(f"Loading model once: {args.model}")
    model = load_ctc_model(args.model, args.device)
    runtimes: dict[str, Any] = {}
    all_engine = None
    for condition in conditions:
        _heading(condition)
        condition_dir = output_dir / condition
        used = build_hotwords_used(condition, audio_ids, hotword_map, all_hotwords)
        write_hotwords_used(condition_dir / "hotwords_used.json", used)
        if condition == "all_hotwords":
            all_engine = CTCWordSpotterASR(model, all_hotwords, config=config, aliases=aliases)
            write_json(condition_dir / "ctcws_text_variants.json", all_engine.used_variants)
        predicted: dict[str, list[str]] = {}
        per_audio: dict[str, dict[str, Any]] = {}
        audio_seconds = 0.0
        meter = RuntimeMeter(args.device); meter.start(); started = time.time()
        for index, audio_id in enumerate(audio_ids, 1):
            details_path = condition_dir / "details" / f"{audio_id}.json"
            if details_path.exists() and not args.overwrite:
                from .hotwords import load_json
                result = load_json(details_path); print(f"[{index:02d}/{len(audio_ids)}] {audio_id}: skip existing")
                validate_hotwords_used(
                    condition,
                    {audio_id: result.get("hotwords_used")},
                    [audio_id],
                    hotword_map,
                    all_hotwords,
                )
                if result.get("condition") != condition:
                    raise AssertionError(
                        f"Cached details condition mismatch for {audio_id}: "
                        f"{result.get('condition')!r} != {condition!r}"
                    )
            else:
                words = used[audio_id]
                engine = all_engine if condition == "all_hotwords" else CTCWordSpotterASR(model, words, config=config, aliases=aliases)
                result = engine.transcribe_file(benchmark_dir / "audio" / f"{audio_id}.wav", enable_ctcws=condition != "vanilla")
                result.update(audio_id=audio_id, condition=condition, hotwords_used=words)
                write_json(details_path, result)
            text = result["raw_text"] if condition == "vanilla" else result["merged_text"]
            write_transcription(condition_dir / "asr", audio_id, text)
            predicted[audio_id] = result.get("predicted_hotwords", [])
            per_audio[audio_id] = result["timing"]
            audio_seconds += float(result["duration_sec"])
        if condition != "vanilla": write_json(condition_dir / "predicted_keywords.json", predicted)
        runtime = meter.stop(audio_seconds)
        runtime.update(processed_audio_count=len(audio_ids), unix_started_at=started, per_audio=per_audio)
        runtimes[condition] = runtime
    write_json(output_dir / "runtime_metrics.json", runtimes)
    write_json(output_dir / "run_config.json", {
        "model": args.model, "model_load_count": 1, "conditions": {
            "vanilla": {"hotword_source": None, "description": "No contextual biasing"},
            "all_hotwords": {"hotword_source": "all_hotwords.json", "scope": "global"},
            "oracle_hotwords": {"hotword_source": "hotwords.json[audio_id]", "scope": "per_audio_ground_truth"}},
        "ctcws": vars(config), "selected_audio_ids": audio_ids})

def main() -> None:
    run_benchmark(parse_args())

if __name__ == "__main__": main()
