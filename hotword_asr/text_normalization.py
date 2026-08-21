from __future__ import annotations

from functools import lru_cache
import re


MODEL_CONTROL_TAG_RE = re.compile(
    r"<(?:(?:[a-z]{2,3})(?:-[A-Za-z]{2,4})?|unk|blank)>"
)


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


def strip_model_control_tags(text: str) -> str:
    """Remove model metadata tokens that are not spoken transcript content."""
    return " ".join(MODEL_CONTROL_TAG_RE.sub(" ", text).split())
