from __future__ import annotations

import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

from .hotwords import context_transcripts
from .text_normalization import to_taiwan_traditional


@dataclass
class CTCWSConfig:
    beam_threshold: float = 7.0
    context_score: float = 3.0
    ctc_ali_token_weight: float = 0.5
    chunk_seconds: float = 30.0
    batch_size: int = 8
    auto_variants: bool = True


def _hypothesis_dict(hyp: Any) -> dict[str, Any]:
    """Serialize NeMo WSHyp without depending on one NeMo release's exact type."""
    if hasattr(hyp, "_asdict"):
        raw = dict(hyp._asdict())
    elif hasattr(hyp, "__dict__"):
        raw = dict(vars(hyp))
    else:
        raw = {}
        for name in ("word", "text", "phrase", "score", "start_frame", "end_frame"):
            if hasattr(hyp, name):
                raw[name] = getattr(hyp, name)

    out: dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(value, np.generic):
            value = value.item()
        elif isinstance(value, torch.Tensor):
            value = value.detach().cpu().tolist()
        if isinstance(value, (str, int, float, bool)) or value is None:
            out[str(key)] = value
        elif isinstance(value, (list, tuple)):
            out[str(key)] = list(value)
    if not out:
        out["repr"] = repr(hyp)
    return out


def _spotted_word(hyp: Any, serialized: dict[str, Any]) -> str:
    for name in ("word", "text", "phrase"):
        value = getattr(hyp, name, None)
        if isinstance(value, str) and value:
            return value
        value = serialized.get(name)
        if isinstance(value, str) and value:
            return value
    raise RuntimeError(
        "Could not identify the canonical word field in NeMo's CTC-WS hypothesis: "
        f"{serialized or repr(hyp)}"
    )


def _load_audio(path: Path, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sample_rate != target_sr:
        import torchaudio.functional as AF

        tensor = torch.from_numpy(np.asarray(audio))
        audio = AF.resample(tensor, sample_rate, target_sr).numpy()
        sample_rate = target_sr
    return np.asarray(audio, dtype=np.float32), sample_rate


def _write_chunks(
    audio: np.ndarray, sample_rate: int, chunk_seconds: float, directory: Path
) -> list[tuple[Path, float, float]]:
    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be > 0")
    chunk_samples = max(1, int(round(chunk_seconds * sample_rate)))
    chunks: list[tuple[Path, float, float]] = []
    for index, start in enumerate(range(0, len(audio), chunk_samples)):
        end = min(start + chunk_samples, len(audio))
        path = directory / f"chunk_{index:05d}.wav"
        sf.write(path, audio[start:end], sample_rate, subtype="PCM_16")
        chunks.append((path, start / sample_rate, end / sample_rate))
    return chunks


class CTCWordSpotterASR:
    def __init__(
        self,
        model: Any,
        hotwords: list[str],
        config: CTCWSConfig | None = None,
        aliases: dict[str, list[str]] | None = None,
    ) -> None:
        from nemo.collections.asr.parts import context_biasing

        self.model = model
        self.config = config or CTCWSConfig()
        self.context_biasing = context_biasing
        self.blank_idx = int(model.decoding.blank_id)
        self.hotwords = list(dict.fromkeys(hotwords))

        entries, used_variants = context_transcripts(
            model,
            self.hotwords,
            aliases=aliases,
            add_automatic_variants=self.config.auto_variants,
        )
        self.used_variants = used_variants
        self.graph = context_biasing.ContextGraphCTC(blank_id=self.blank_idx)
        self.graph.add_to_graph(entries)

    def _transcribe_chunks(self, chunk_paths: list[str]) -> list[Any]:
        kwargs = {
            "batch_size": self.config.batch_size,
            "return_hypotheses": True,
        }
        try:
            return self.model.transcribe(chunk_paths, verbose=False, **kwargs)
        except TypeError:
            # Older NeMo releases do not expose the verbose argument.
            return self.model.transcribe(chunk_paths, **kwargs)

    def transcribe_file(
        self, audio_path: str | Path, *, enable_ctcws: bool = True
    ) -> dict[str, Any]:
        audio_path = Path(audio_path).resolve()
        audio, sample_rate = _load_audio(audio_path)
        duration = len(audio) / sample_rate

        raw_parts: list[str] = []
        merged_parts: list[str] = []
        alignment_raw_parts: list[str] = []
        spotted: list[dict[str, Any]] = []
        predicted: list[str] = []
        spotting_started = time.perf_counter()

        for chunk_index, (logprobs, raw_text, start_sec, end_sec) in enumerate(chunks):
            ws_hyps = self.context_biasing.run_word_spotter(
                logprobs,
                self.graph,
                self.model,
                blank_idx=self.blank_idx,
                beam_threshold=self.config.beam_threshold,
                cb_weight=self.config.context_score,
                ctc_ali_token_weight=self.config.ctc_ali_token_weight,
            )
            greedy = np.argmax(logprobs, axis=1)
            if ws_hyps:
                merged_text, alignment_raw_text = (
                    self.context_biasing.merge_alignment_with_ws_hyps(
                        greedy,
                        self.model,
                        ws_hyps,
                        decoder_type="ctc",
                        blank_idx=self.blank_idx,
                        print_stats=False,
                    )
                )
            else:
                merged_text = raw_text
                alignment_raw_text = raw_text

            alignment_raw_parts.append(str(alignment_raw_text or ""))
            for ws_hyp in ws_hyps:
                item = _hypothesis_dict(ws_hyp)
                canonical = _spotted_word(ws_hyp, item)
                item.update(
                    {
                        "canonical": canonical,
                        "chunk_index": chunk_index,
                        "chunk_start_sec": round(start_sec, 3),
                        "chunk_end_sec": round(end_sec, 3),
                    }
                )
                spotted.append(item)
                predicted.append(canonical)
            merged_parts.append(str(merged_text or ""))

        merged_text_model = " ".join(x.strip() for x in merged_parts if x.strip())
        alignment_raw_text_model = " ".join(
            x.strip() for x in alignment_raw_parts if x.strip()
        )
        return {
            "merged_text": to_taiwan_traditional(merged_text_model),
            "merged_text_model": merged_text_model,
            "alignment_raw_text_model": alignment_raw_text_model,
            "predicted_hotwords": sorted(set(predicted), key=str.casefold),
            "spotted": spotted,
            "hotword_count": len(self.hotwords),
            "ctcws_seconds": time.perf_counter() - spotting_started,
            "ctcws_config": asdict(self.config),
        }

    def transcribe_file_conditions(
        self,
        audio_path: str | Path,
        spotters: dict[str, "CTCWordSpotterASR"],
    ) -> dict[str, Any]:
        """Run acoustic inference once, then apply multiple CTC-WS graphs."""
        if not spotters:
            raise ValueError("At least one CTC-WS condition is required")
        for name, spotter in spotters.items():
            if spotter.model is not self.model:
                raise ValueError(f"Condition {name!r} does not share the acoustic model")
            if (
                spotter.config.batch_size != self.config.batch_size
                or spotter.config.chunk_seconds != self.config.chunk_seconds
            ):
                raise ValueError(
                    f"Condition {name!r} must share batch/chunk inference settings"
                )

        audio_path = Path(audio_path).resolve()
        audio, sample_rate = _load_audio(audio_path)
        duration = len(audio) / sample_rate

        with tempfile.TemporaryDirectory(prefix="ctcws_chunks_") as tmp:
            chunk_meta = _write_chunks(
                audio, sample_rate, self.config.chunk_seconds, Path(tmp)
            )
            chunk_paths = [str(path) for path, _, _ in chunk_meta]

            t0 = time.perf_counter()
            with torch.inference_mode():
                hypotheses = self._transcribe_chunks(chunk_paths)
            if next(self.model.parameters()).device.type == "cuda":
                torch.cuda.synchronize()
            asr_seconds = time.perf_counter() - t0

            if len(hypotheses) != len(chunk_meta):
                raise RuntimeError(
                    f"NeMo returned {len(hypotheses)} hypotheses for {len(chunk_meta)} chunks"
                )

            prepared_chunks: list[tuple[np.ndarray, str, float, float]] = []
            for hyp, (_, start_sec, end_sec) in zip(hypotheses, chunk_meta):
                logprobs = hyp.alignments.detach().float().cpu().numpy()
                if logprobs.ndim != 2:
                    raise RuntimeError(
                        f"Expected CTC alignment matrix [T,V], got {logprobs.shape}"
                    )
                raw_text = str(getattr(hyp, "text", "") or "")
                raw_parts.append(raw_text)

                t1 = time.perf_counter()
                ws_hyps = self.context_biasing.run_word_spotter(
                    logprobs,
                    self.graph,
                    self.model,
                    blank_idx=self.blank_idx,
                    beam_threshold=self.config.beam_threshold,
                    cb_weight=self.config.context_score,
                    ctc_ali_token_weight=self.config.ctc_ali_token_weight,
                ) if enable_ctcws else []
                greedy = np.argmax(logprobs, axis=1)
                if ws_hyps:
                    merged_text, merge_raw_text = (
                        self.context_biasing.merge_alignment_with_ws_hyps(
                            greedy,
                            self.model,
                            ws_hyps,
                            decoder_type="ctc",
                            blank_idx=self.blank_idx,
                            print_stats=False,
                        )
                    )
                    # merge_raw_text is useful for detecting tokenizer/NeMo drift.
                    if merge_raw_text:
                        raw_parts[-1] = str(merge_raw_text)
                else:
                    merged_text = raw_text

                for ws_hyp in ws_hyps:
                    item = _hypothesis_dict(ws_hyp)
                    canonical = _spotted_word(ws_hyp, item)
                    item.update(
                        {
                            "canonical": canonical,
                            "chunk_index": chunk_index,
                            "chunk_start_sec": round(start_sec, 3),
                            "chunk_end_sec": round(end_sec, 3),
                        }
                    )
                    spotted.append(item)
                    predicted.append(canonical)
                merged_parts.append(str(merged_text or ""))
                spotting_seconds += time.perf_counter() - t1

        #NOTE: Vanilla ASR always comes directly from model hypotheses. It must
        # not depend on whether either hotword graph found a candidate.
        raw_text_model = " ".join(
            raw_text.strip()
            for _, raw_text, _, _ in prepared_chunks
            if raw_text.strip()
        )
        condition_results: dict[str, dict[str, Any]] = {}
        condition_seconds: dict[str, float] = {}
        for name, spotter in spotters.items():
            condition = spotter._spot_chunks(prepared_chunks)
            ctcws_seconds = float(condition.pop("ctcws_seconds"))
            condition_seconds[name] = round(ctcws_seconds, 4)
            condition["timing"] = {
                "asr_seconds_shared": round(asr_seconds, 4),
                "ctcws_and_merge_seconds": round(ctcws_seconds, 4),
                "total_condition_seconds": round(asr_seconds + ctcws_seconds, 4),
                "rtf_condition": round((asr_seconds + ctcws_seconds) / duration, 6)
                if duration
                else None,
            }
            condition_results[name] = condition

        return {
            "audio_path": str(audio_path),
            "duration_sec": round(duration, 4),
            "raw_text": to_taiwan_traditional(raw_text_model),
            "raw_text_model": raw_text_model,
            "conditions": condition_results,
            "timing": {
                "asr_seconds": round(asr_seconds, 4),
                "ctcws_and_merge_seconds_by_condition": condition_seconds,
                "total_work_seconds": round(
                    asr_seconds + sum(condition_seconds.values()), 4
                ),
                "rtf_total_work": round(
                    (asr_seconds + sum(condition_seconds.values())) / duration, 6
                )
                if duration
                else None,
            },
            "text_normalization": {
                "ctc_graph": "tw2s",
                "output": "s2tw",
            },
        }

    def transcribe_file(self, audio_path: str | Path) -> dict[str, Any]:
        """Backward-compatible single-hotword-condition transcription."""
        multi = self.transcribe_file_conditions(audio_path, {"default": self})
        condition = multi["conditions"]["default"]
        timing = condition["timing"]
        return {
            "audio_path": multi["audio_path"],
            "duration_sec": multi["duration_sec"],
            "raw_text": multi["raw_text"],
            "merged_text": condition["merged_text"],
            "raw_text_model": multi["raw_text_model"],
            "merged_text_model": condition["merged_text_model"],
            "alignment_raw_text_model": condition["alignment_raw_text_model"],
            "predicted_hotwords": condition["predicted_hotwords"],
            "spotted": condition["spotted"],
            "timing": {
                "asr_seconds": timing["asr_seconds_shared"],
                "ctcws_and_merge_seconds": timing["ctcws_and_merge_seconds"],
                "total_pipeline_seconds": timing["total_condition_seconds"],
                "rtf_pipeline": timing["rtf_condition"],
            },
            "ctcws_config": condition["ctcws_config"],
            "text_normalization": multi["text_normalization"],
        }
