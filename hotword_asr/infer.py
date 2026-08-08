from __future__ import annotations

import argparse
from pathlib import Path

from .engine import CTCWSConfig, CTCWordSpotterASR
from .hotwords import load_aliases, load_hotword_list
from .io import write_json
from .metrics import RuntimeMeter
from .model import load_ctc_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ASR + CTC word spotting for one wav file")
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--hotwords", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, default=Path("result.json"))
    parser.add_argument("--aliases", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--chunk-seconds", type=float, default=30.0)
    parser.add_argument("--beam-threshold", type=float, default=7.0)
    parser.add_argument("--context-score", type=float, default=3.0)
    parser.add_argument("--ctc-ali-token-weight", type=float, default=0.5)
    parser.add_argument("--no-auto-variants", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hotwords = load_hotword_list(args.hotwords)
    aliases = load_aliases(args.aliases)
    config = CTCWSConfig(
        beam_threshold=args.beam_threshold,
        context_score=args.context_score,
        ctc_ali_token_weight=args.ctc_ali_token_weight,
        chunk_seconds=args.chunk_seconds,
        batch_size=args.batch_size,
        auto_variants=not args.no_auto_variants,
    )
    model = load_ctc_model(args.model, args.device)
    engine = CTCWordSpotterASR(model, hotwords, config, aliases)
    meter = RuntimeMeter(args.device)
    meter.start()
    result = engine.transcribe_file(args.audio)
    result["runtime"] = meter.stop(float(result["duration_sec"]))
    write_json(args.output, result)
    print(result["merged_text"])
    print(f"Spotted: {result['predicted_hotwords']}")
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()

