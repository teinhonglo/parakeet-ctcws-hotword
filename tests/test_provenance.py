from __future__ import annotations

import pytest

from hotword_asr.provenance import require_matching_signature, run_signature


def test_run_signature_is_order_independent_but_setting_sensitive() -> None:
    assert run_signature({"model": "m", "itn": True}) == run_signature(
        {"itn": True, "model": "m"}
    )
    assert run_signature({"model": "m", "itn": True}) != run_signature(
        {"model": "m", "itn": False}
    )


def test_stale_cache_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="--overwrite"):
        require_matching_signature(
            {"run_signature": "old"}, "new", context="backend condition audio"
        )
