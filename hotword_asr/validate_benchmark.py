from __future__ import annotations

import argparse
from pathlib import Path

from .hotwords import compare_vocabularies, load_hotword_list, load_hotword_map


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark_dir", type=Path)
    args = parser.parse_args()
    root = args.benchmark_dir.resolve()
    hotword_map = load_hotword_map(root / "hotwords.json")
    all_hotwords = load_hotword_list(root / "all_hotwords.json")
    report = compare_vocabularies(hotword_map, all_hotwords)
    audio_ids = set(hotword_map)
    wav_ids = {p.stem for p in (root / "audio").glob("*.wav")}

    print(f"audio ids in hotwords.json : {len(audio_ids)}")
    print(f"wav files                 : {len(wav_ids)}")
    print(f"ground-truth unique words : {report['ground_truth_unique']}")
    print(f"all_hotwords unique words : {report['vocabulary_unique']}")
    print(f"keyword instances         : {report['ground_truth_instances']}")
    print(f"missing wav ids           : {sorted(audio_ids - wav_ids, key=int)}")
    print(f"extra wav ids             : {sorted(wav_ids - audio_ids, key=int)}")
    print(f"missing from all_hotwords : {report['missing_from_vocabulary']}")
    print(f"extra in all_hotwords     : {report['extra_in_vocabulary']}")


if __name__ == "__main__":
    main()

