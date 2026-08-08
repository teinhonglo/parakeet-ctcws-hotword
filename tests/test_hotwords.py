from __future__ import annotations

import unittest

from hotword_asr.hotwords import (
    automatic_variants,
    compare_vocabularies,
    context_transcripts,
)


class FakeTokenizer:
    def text_to_ids(self, text: str) -> list[int]:
        return [ord(ch) for ch in text]


class FakeModel:
    tokenizer = FakeTokenizer()


class HotwordTests(unittest.TestCase):
    def test_automatic_variants_for_acronym_and_hyphen(self) -> None:
        xray = automatic_variants("X-RAY")
        self.assertIn("X-RAY", xray)
        self.assertIn("x-ray", xray)
        self.assertIn("X RAY", xray)
        self.assertIn("X R A Y", xray)

    def test_chinese_hotword_is_not_case_expanded(self) -> None:
        self.assertEqual(automatic_variants("洗腎"), ["洗腎"])

    def test_context_graph_entries_keep_canonical_label(self) -> None:
        entries, variants = context_transcripts(
            FakeModel(), ["Mixtard"], aliases={"Mixtard": ["mix tard"]}
        )
        self.assertEqual(entries[0][0], "Mixtard")
        self.assertGreaterEqual(len(entries[0][1]), 2)
        self.assertIn("mix tard", variants["Mixtard"])

    def test_vocabulary_mismatch_is_reported(self) -> None:
        report = compare_vocabularies({"743": ["elbew"]}, ["elbow"])
        self.assertEqual(report["missing_from_vocabulary"], ["elbew"])
        self.assertEqual(report["extra_in_vocabulary"], ["elbow"])
