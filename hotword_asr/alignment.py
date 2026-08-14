from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ASCII_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)*")
TEXT_KEYS = ("text", "transcription", "transcript", "pseudo_transcript")
ID_KEYS = ("audio_id", "id", "utt_id", "utterance_id")
EPSILON = "<eps>"


@dataclass(frozen=True)
class AlignmentItem:
    operation: str
    reference: str
    hypothesis: str


def _extract_text(value: Any, context: str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in TEXT_KEYS:
            text = value.get(key)
            if isinstance(text, str):
                return text
    raise ValueError(f"Cannot find transcript text in {context}")


def load_references(path: str | Path) -> dict[str, str]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        for container_key in (
            "pseudo_transcripts",
            "references",
            "data",
            "items",
            "utterances",
        ):
            if isinstance(data.get(container_key), (dict, list)):
                data = data[container_key]
                break
        else:
            return {
                str(audio_id): _extract_text(value, f"{path}:{audio_id}")
                for audio_id, value in data.items()
            }

    if isinstance(data, list):
        references: dict[str, str] = {}
        for index, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(f"Expected an object at {path}[{index}]")
            audio_id = next((item.get(key) for key in ID_KEYS if item.get(key) is not None), None)
            if audio_id is None:
                raise ValueError(f"Cannot find an audio id at {path}[{index}]")
            references[str(audio_id)] = _extract_text(item, f"{path}[{index}]")
        return references

    raise ValueError(f"Unsupported reference JSON root in {path}: {type(data)!r}")


def load_candidates(candidate_root: str | Path) -> dict[str, str]:
    candidate_root = Path(candidate_root)
    candidates: dict[str, str] = {}
    for path in candidate_root.glob("*/transcription.json"):
        with path.open("r", encoding="utf-8") as f:
            candidates[path.parent.name] = _extract_text(json.load(f), str(path))
    if not candidates:
        raise FileNotFoundError(f"No */transcription.json files found under {candidate_root}")
    return candidates


def tokenize_mixed_text(text: str) -> list[str]:
    """Tokenize Mandarin characters individually and Latin words as word units."""
    text = unicodedata.normalize("NFKC", text)
    tokens: list[str] = []
    index = 0
    while index < len(text):
        match = ASCII_WORD_RE.match(text, index)
        if match:
            tokens.append(match.group(0).upper())
            index = match.end()
            continue

        char = text[index]
        index += 1
        if char.isspace() or unicodedata.category(char).startswith(("P", "S")):
            continue
        if char.isalnum():
            tokens.append(char.upper())
    return tokens


def align_tokens(reference: list[str], hypothesis: list[str]) -> list[AlignmentItem]:
    rows = len(reference) + 1
    cols = len(hypothesis) + 1
    costs = [[0] * cols for _ in range(rows)]
    matches = [[0] * cols for _ in range(rows)]
    steps = [[""] * cols for _ in range(rows)]

    for i in range(1, rows):
        costs[i][0] = i
        steps[i][0] = "D"
    for j in range(1, cols):
        costs[0][j] = j
        steps[0][j] = "I"

    for i in range(1, rows):
        for j in range(1, cols):
            if reference[i - 1] == hypothesis[j - 1]:
                costs[i][j] = costs[i - 1][j - 1]
                matches[i][j] = matches[i - 1][j - 1] + 1
                steps[i][j] = "="
                continue

            choices = (
                (costs[i - 1][j - 1] + 1, matches[i - 1][j - 1], "S", 0),
                (costs[i - 1][j] + 1, matches[i - 1][j], "D", 1),
                (costs[i][j - 1] + 1, matches[i][j - 1], "I", 2),
            )
            cost, match_count, operation, _ = min(
                choices, key=lambda choice: (choice[0], -choice[1], choice[3])
            )
            costs[i][j] = cost
            matches[i][j] = match_count
            steps[i][j] = operation

    aligned: list[AlignmentItem] = []
    i, j = len(reference), len(hypothesis)
    while i or j:
        operation = steps[i][j]
        if operation in ("=", "S"):
            aligned.append(AlignmentItem(operation, reference[i - 1], hypothesis[j - 1]))
            i -= 1
            j -= 1
        elif operation == "D":
            aligned.append(AlignmentItem(operation, reference[i - 1], EPSILON))
            i -= 1
        elif operation == "I":
            aligned.append(AlignmentItem(operation, EPSILON, hypothesis[j - 1]))
            j -= 1
        else:
            raise RuntimeError(f"Invalid alignment step at ({i}, {j}): {operation!r}")
    aligned.reverse()
    return aligned


def _sort_audio_ids(audio_ids: Iterable[str]) -> list[str]:
    return sorted(audio_ids, key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value))


def _display_width(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(char) in ("W", "F") else 1 for char in text)


def _pad(text: str, width: int) -> str:
    return text + " " * (width - _display_width(text))


def _format_blocks(items: list[AlignmentItem], tokens_per_line: int) -> list[str]:
    lines: list[str] = []
    for start in range(0, len(items), tokens_per_line):
        block = items[start : start + tokens_per_line]
        widths = [
            max(_display_width(item.reference), _display_width(item.hypothesis), 1)
            for item in block
        ]
        lines.append("REF: " + " | ".join(_pad(item.reference, width) for item, width in zip(block, widths)))
        lines.append("HYP: " + " | ".join(_pad(item.hypothesis, width) for item, width in zip(block, widths)))
        lines.append("OPS: " + " | ".join(_pad(item.operation, width) for item, width in zip(block, widths)))
        lines.append("")
    return lines


def write_alignment_reports(
    references: dict[str, str],
    candidates: dict[str, str],
    text_output: str | Path,
    tsv_output: str | Path,
    tokens_per_line: int = 20,
) -> int:
    if tokens_per_line <= 0:
        raise ValueError("tokens_per_line must be greater than zero")

    unknown_ids = set(candidates) - set(references)
    if unknown_ids:
        raise KeyError(f"Candidate ids missing from references: {_sort_audio_ids(unknown_ids)}")

    audio_ids = _sort_audio_ids(candidates)
    text_output = Path(text_output)
    tsv_output = Path(tsv_output)
    text_output.parent.mkdir(parents=True, exist_ok=True)
    tsv_output.parent.mkdir(parents=True, exist_ok=True)

    text_lines = ["OPS: = correct, S substitution, I insertion, D deletion", ""]
    tsv_rows: list[list[str | int]] = []

    for audio_id in audio_ids:
        reference_text = references[audio_id]
        hypothesis_text = candidates[audio_id]
        aligned = align_tokens(
            tokenize_mixed_text(reference_text), tokenize_mixed_text(hypothesis_text)
        )
        text_lines.extend(
            [
                f"ID: {audio_id}",
                f"REFERENCE_TEXT: {reference_text}",
                f"HYPOTHESIS_TEXT: {hypothesis_text}",
            ]
        )
        text_lines.extend(_format_blocks(aligned, tokens_per_line))
        text_lines.append("=" * 80)
        text_lines.append("")

        for position, item in enumerate(aligned, start=1):
            tsv_rows.append(
                [audio_id, position, item.operation, item.reference, item.hypothesis]
            )

    text_output.write_text("\n".join(text_lines) + "\n", encoding="utf-8")
    with tsv_output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(("audio_id", "position", "operation", "reference", "hypothesis"))
        writer.writerows(tsv_rows)
    return len(audio_ids)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write token-aligned REF/HYP reports")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tsv-output", type=Path, required=True)
    parser.add_argument("--tokens-per-line", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = write_alignment_reports(
        load_references(args.reference),
        load_candidates(args.candidate),
        args.output,
        args.tsv_output,
        args.tokens_per_line,
    )
    print(f"Wrote {count} aligned REF/HYP pairs to {args.output} and {args.tsv_output}")


if __name__ == "__main__":
    main()
