from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .io import write_json


def rank_scores(root: Path) -> dict[str, Any]:
    root = root.resolve()
    rows = []
    for path in sorted(root.rglob("dev_score.json")):
        score = json.loads(path.read_text(encoding="utf-8"))
        config_path = path.parent / "run_config.json"
        config = (
            json.loads(config_path.read_text(encoding="utf-8"))
            if config_path.is_file()
            else None
        )
        rows.append(
            {
                "experiment_dir": str(path.parent),
                "mer": score["mer"],
                "hotword_recall": score["hotword_recall"],
                "audio_count": score["audio_count"],
                "audio_ids": score["audio_ids"],
                "run_config": config,
            }
        )
    if not rows:
        raise FileNotFoundError(f"No dev_score.json files found under {root}")

    expected_ids = rows[0]["audio_ids"]
    inconsistent = [row["experiment_dir"] for row in rows if row["audio_ids"] != expected_ids]
    if inconsistent:
        raise ValueError(
            "Tuning results do not use the same audio IDs: " + ", ".join(inconsistent)
        )
    rows.sort(key=lambda row: (row["mer"], -row["hotword_recall"]))
    return {
        "selection_rule": "lowest development MER; highest hotword recall breaks ties",
        "warning": (
            "Do not report these development scores as final benchmark results. "
            "Rerun the selected parameters once on an untouched evaluation set."
        ),
        "best": rows[0],
        "ranked": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank decoder tuning runs")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = rank_scores(args.root)
    write_json(args.output, result)
    print("rank  MER       recall    experiment")
    for index, row in enumerate(result["ranked"], start=1):
        print(
            f"{index:4d}  {row['mer']:8.2%}  {row['hotword_recall']:8.2%}  "
            f"{row['experiment_dir']}"
        )


if __name__ == "__main__":
    main()
