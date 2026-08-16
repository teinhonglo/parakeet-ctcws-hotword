from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=2)
def _converter(config: str):
    # Keep OpenCC optional at module-import time so model-independent condition
    # selection/tests do not require the inference environment.
    from opencc import OpenCC

    return OpenCC(config)


def to_simplified_chinese(text: str) -> str:
    """Convert Taiwan Traditional Chinese to Simplified Chinese for CTC paths."""
    return _converter("tw2s.json").convert(text)


def to_taiwan_traditional(text: str) -> str:
    """Normalize ASR output to Taiwan Traditional Chinese for output/scoring."""
    return _converter("s2tw.json").convert(text)
