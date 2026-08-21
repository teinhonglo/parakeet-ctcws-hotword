from __future__ import annotations

import argparse
from pathlib import Path


REPORTS = (
    ("Parakeet", "Vanilla", "parakeet", "report_vanilla_asr.xlsx"),
    ("Parakeet", "All Hotwords", "parakeet", "report_ctcws_all_hotwords_asr.xlsx"),
    ("Parakeet", "Oracle Hotwords", "parakeet", "report_ctcws_oracle_hotwords_asr.xlsx"),
    ("Nemotron", "Vanilla", "nemotron", "report_vanilla_asr.xlsx"),
    ("Nemotron", "All Hotwords", "nemotron", "report_gpu_pb_all_hotwords_asr.xlsx"),
    ("Nemotron", "Oracle Hotwords", "nemotron", "report_gpu_pb_oracle_hotwords_asr.xlsx"),
    ("Fun-ASR", "Vanilla", "funasr", "report_vanilla_asr.xlsx"),
    ("Fun-ASR", "All Hotwords", "funasr", "report_hotword_all_hotwords_asr.xlsx"),
    ("Fun-ASR", "Oracle Hotwords", "funasr", "report_hotword_oracle_hotwords_asr.xlsx"),
)


def _metrics(path: Path) -> tuple[float, float]:
    from openpyxl import load_workbook

    workbook = load_workbook(path, read_only=True, data_only=True)
    rows = list(workbook["Summary"].iter_rows(values_only=True))
    recall = next(value for key, value, *_ in rows if key and "Recall" in str(key))
    mer = next(value for key, value, *_ in rows if key and "MER（" in str(key))
    def as_ratio(value: object) -> float:
        if isinstance(value, (int, float)):
            numeric = float(value)
            return numeric / 100.0 if numeric > 1.0 else numeric
        rendered = str(value).strip()
        return float(rendered.rstrip("%")) / (100.0 if rendered.endswith("%") else 1.0)

    return as_ratio(recall), as_ratio(mer)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize all nine benchmark reports")
    parser.add_argument("--parakeet-dir", type=Path, required=True)
    parser.add_argument("--nemotron-dir", type=Path, required=True)
    parser.add_argument("--funasr-dir", type=Path, required=True)
    parser.add_argument("--target-mer", type=float, default=0.15)
    args = parser.parse_args()

    print("Model       Condition         Recall     MER       <= target")
    print("----------- ----------------- ---------- --------- ---------")
    missing = []
    roots = {
        "parakeet": args.parakeet_dir,
        "nemotron": args.nemotron_dir,
        "funasr": args.funasr_dir,
    }
    for model, condition, root_name, filename in REPORTS:
        path = roots[root_name] / filename
        if not path.exists():
            missing.append(str(path))
            continue
        recall, mer = _metrics(path)
        print(
            f"{model:11s} {condition:17s} {recall:9.2%} {mer:9.2%} "
            f"{'YES' if mer <= args.target_mer else 'NO'}"
        )
    if missing:
        raise FileNotFoundError("Missing canonical reports:\n  " + "\n  ".join(missing))


if __name__ == "__main__":
    main()
