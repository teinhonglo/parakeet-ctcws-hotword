from __future__ import annotations

from opencc import OpenCC


# The benchmark labels use Taiwan Traditional Chinese, while the zh-CN model
# produces Simplified Chinese token posteriors and transcription text.
_TO_SIMPLIFIED = OpenCC("tw2s.json")
_TO_TAIWAN_TRADITIONAL = OpenCC("s2tw.json")


def to_simplified_chinese(text: str) -> str:
    """Convert Taiwan Traditional Chinese to Simplified Chinese for CTC paths."""
    return _TO_SIMPLIFIED.convert(text)


def to_taiwan_traditional(text: str) -> str:
    """Normalize ASR output to Taiwan Traditional Chinese for output/scoring."""
    return _TO_TAIWAN_TRADITIONAL.convert(text)
