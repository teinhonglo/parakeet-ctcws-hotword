import json
from pathlib import Path

import pytest

from hotword_asr.rank_tuning import rank_scores


def _score(path: Path, mer: float, recall: float, ids: list[str]) -> None:
    path.mkdir(parents=True)
    (path / "dev_score.json").write_text(
        json.dumps(
            {
                "mer": mer,
                "hotword_recall": recall,
                "audio_count": len(ids),
                "audio_ids": ids,
            }
        ),
        encoding="utf-8",
    )


def test_rank_scores_uses_mer_then_recall(tmp_path: Path) -> None:
    _score(tmp_path / "a", 0.25, 0.60, ["1", "2"])
    _score(tmp_path / "b", 0.20, 0.40, ["1", "2"])
    _score(tmp_path / "c", 0.20, 0.70, ["1", "2"])

    result = rank_scores(tmp_path)

    assert Path(result["best"]["experiment_dir"]).name == "c"


def test_rank_scores_rejects_mixed_development_sets(tmp_path: Path) -> None:
    _score(tmp_path / "a", 0.20, 0.60, ["1"])
    _score(tmp_path / "b", 0.19, 0.60, ["2"])
    with pytest.raises(ValueError, match="same audio IDs"):
        rank_scores(tmp_path)
