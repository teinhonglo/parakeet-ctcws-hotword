from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


ASCII_ALPHA_RE = re.compile(r"[A-Za-z]")


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        item = item.strip()
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def load_hotword_list(path: str | Path) -> list[str]:
    """Load either all_hotwords.json (list) or hotwords.json (id -> list)."""
    data = load_json(path)
    if isinstance(data, list):
        return _dedupe([str(x) for x in data])
    if isinstance(data, dict):
        values: list[str] = []
        for words in data.values():
            if not isinstance(words, list):
                raise ValueError(f"Expected a list of hotwords, got: {type(words)!r}")
            values.extend(str(x) for x in words)
        return sorted(_dedupe(values), key=str.casefold)
    raise ValueError(f"Unsupported hotword JSON root: {type(data)!r}")


def load_hotword_map(path: str | Path) -> dict[str, list[str]]:
    data = load_json(path)
    if not isinstance(data, dict):
        raise ValueError("Expected hotwords.json to be an object: audio_id -> hotword list")
    result: dict[str, list[str]] = {}
    for audio_id, words in data.items():
        if not isinstance(words, list):
            raise ValueError(f"hotwords[{audio_id!r}] is not a list")
        result[str(audio_id)] = _dedupe([str(x) for x in words])
    return result


def load_aliases(path: str | Path | None) -> dict[str, list[str]]:
    if path is None:
        return {}
    data = load_json(path)
    if not isinstance(data, dict):
        raise ValueError("Alias file must be a JSON object: canonical -> [alternative spellings]")
    aliases: dict[str, list[str]] = {}
    for canonical, variants in data.items():
        if isinstance(variants, str):
            variants = [variants]
        if not isinstance(variants, list):
            raise ValueError(f"Aliases for {canonical!r} must be a string or list")
        aliases[str(canonical)] = _dedupe([str(x) for x in variants])
    return aliases


def compare_vocabularies(
    hotword_map: dict[str, list[str]], all_hotwords: list[str]
) -> dict[str, Any]:
    target_set = {word for words in hotword_map.values() for word in words}
    vocabulary_set = set(all_hotwords)
    return {
        "ground_truth_unique": len(target_set),
        "vocabulary_unique": len(vocabulary_set),
        "ground_truth_instances": sum(len(set(words)) for words in hotword_map.values()),
        "missing_from_vocabulary": sorted(target_set - vocabulary_set, key=str.casefold),
        "extra_in_vocabulary": sorted(vocabulary_set - target_set, key=str.casefold),
    }


def automatic_variants(word: str) -> list[str]:
    """Generate conservative spelling/case variants for English medical hotwords.

    The canonical word is always first.  These are only alternative paths in the
    CTC context graph.  A successful path is still emitted as the canonical word.
    """
    variants = [word]
    if not ASCII_ALPHA_RE.search(word):
        return variants

    variants.extend([word.lower(), word.upper()])

    # Common benchmark forms such as X-RAY, L-SPINE, E-coli and I/O can be
    # acoustically/tokenizer-equivalent to a space-separated spelling.
    for sep in ("-", "/", "_"):
        if sep in word:
            variants.extend(
                [
                    word.replace(sep, " "),
                    word.lower().replace(sep, " "),
                    word.upper().replace(sep, " "),
                ]
            )

    compact = re.sub(r"[^A-Za-z]", "", word)
    # Acronyms are often decoded as "A F" rather than "AF".
    if word.upper() == word and 2 <= len(compact) <= 6:
        variants.append(" ".join(compact))
        variants.append(" ".join(compact.lower()))

    return _dedupe(variants)


def context_transcripts(
    model: Any,
    hotwords: list[str],
    aliases: dict[str, list[str]] | None = None,
    add_automatic_variants: bool = True,
) -> tuple[list[list[Any]], dict[str, list[str]]]:
    """Build NeMo ContextGraphCTC input and retain the textual variants used."""
    if not hasattr(model, "tokenizer"):
        raise TypeError(
            "The selected model has no tokenizer. The zh-CN Parakeet model is expected "
            "to be a BPE CTC model with model.tokenizer.text_to_ids()."
        )

    aliases = aliases or {}
    graph_entries: list[list[Any]] = []
    used_variants: dict[str, list[str]] = {}

    for canonical in hotwords:
        variants = automatic_variants(canonical) if add_automatic_variants else [canonical]
        variants.extend(aliases.get(canonical, []))
        variants = _dedupe(variants)

        id_sequences: list[list[int]] = []
        seen_ids: set[tuple[int, ...]] = set()
        kept_text: list[str] = []
        for text in variants:
            ids = [int(x) for x in model.tokenizer.text_to_ids(text)]
            key = tuple(ids)
            if ids and key not in seen_ids:
                seen_ids.add(key)
                id_sequences.append(ids)
                kept_text.append(text)

        if not id_sequences:
            raise ValueError(f"Tokenizer produced no ids for hotword {canonical!r}")

        # ContextGraphCTC expects [canonical_word, [token-id sequence, ...]].
        graph_entries.append([canonical, id_sequences])
        used_variants[canonical] = kept_text

    return graph_entries, used_variants

