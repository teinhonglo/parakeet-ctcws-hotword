from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from hotword_asr.io import write_transcription


class IOTests(unittest.TestCase):
    def test_transcription_matches_benchmark_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            write_transcription(tmp_path, "534", "Mixtard")
            data = json.loads((tmp_path / "534" / "transcription.json").read_text())
            self.assertEqual(data, {"text": "Mixtard"})
