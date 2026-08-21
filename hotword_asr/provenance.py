from __future__ import annotations

import hashlib
import json
from typing import Any


def run_signature(payload: dict[str, Any]) -> str:
    """Return a stable cache key for all accuracy-affecting settings."""
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def require_matching_signature(
    result: dict[str, Any], expected: str, *, context: str
) -> None:
    actual = result.get("run_signature")
    if actual != expected:
        raise RuntimeError(
            f"Stale cached inference for {context}: run_signature "
            f"{actual!r} != {expected!r}. Re-run with --overwrite."
        )
