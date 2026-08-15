# Parakeet CTC-WS + Nemotron GPU-PB Hotword Benchmark

This project runs a Mandarin-English CTC ASR model and NVIDIA NeMo's CTC-based
Word Spotter (CTC-WS) on the 71-file hospital hotword benchmark.

It also provides a directly comparable Nemotron 3.5 pipeline with baseline
RNNT decoding and NVIDIA NeMo GPU Phrase Boosting (GPU-PB).

Default target model:

- `nvidia/riva/parakeet-ctc-riva-0-6b-unified-zh-cn:trainable_v3.0`
- FastConformer-CTC, about 600M parameters
- Mandarin + English code-switching
- 7000-subword vocabulary

The important detail is that CTC-WS needs the acoustic model's **frame-level CTC
log probabilities**. The hosted/NIM transcription API returns transcription
results, not the internal `[time, vocabulary]` matrix needed by `run_word_spotter`.
For this reason the project downloads NVIDIA's **trainable `.nemo` acoustic
checkpoint** from NGC and runs it locally with NeMo.

## Pipeline

```text
wav
  -> Parakeet FastConformer-CTC
  -> CTC log probabilities
       |-> greedy CTC ASR ---------------------> raw_asr
       `-> ContextGraphCTC -> run_word_spotter
                              -> merge_alignment_with_ws_hyps
                              -> ctcws_asr + predicted_keywords.json
```

The context graph is built **once from the global 139-hotword vocabulary** and
reused for every recording. It never receives the ground-truth hotwords for the
current audio file.

Nemotron uses the same global vocabulary policy and benchmark evaluator:

```text
wav -> Nemotron 3.5 zh-CN RNNT
       |-> greedy_batch, alpha=0 -----------------> raw_asr
       `-> greedy_batch + GPU-PB boosting tree ---> gpu_pb_asr
```

NVIDIA lists Mandarin support for this checkpoint as `zh-CN`, not `zh-TW`.
Before GPU-PB, Chinese hotwords are converted from Traditional to Simplified
with OpenCC `t2s`. Raw model output is retained for auditing, then converted
with OpenCC `s2t` before it is written to the evaluator candidate directory.

## 1. System requirements

- Ubuntu/Linux
- NVIDIA GPU recommended (a 24 GB RTX 3090 is suitable for this 0.6B model)
- Miniconda or Anaconda (`conda` available on `PATH`)
- Python >= 3.12
- a recent NVIDIA driver compatible with the PyTorch CUDA wheel you install
- an NVIDIA NGC account for the trainable zh-CN model artifact

Current NeMo Speech documentation requires Python >= 3.12 and PyTorch >= 2.7
for the bring-your-own PyTorch/CUDA installation route.

## 2. Install Python environment

```bash
cd parakeet_ctcws_hotword
bash scripts/install.sh
source path.sh
```

Stage 0 creates the Conda environment `parakeet_ctcws` with Python 3.12.
`path.sh` is the single place that activates this environment. To use another
environment name, set `CONDA_ENV_NAME` consistently before install/run:

```bash
export CONDA_ENV_NAME=my_parakeet_env
bash scripts/install.sh
source path.sh
```

If you need a particular CUDA PyTorch wheel, set its index explicitly before
running the installer. For example, use the index appropriate for the NVIDIA
driver installed on your machine:

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu129 bash scripts/install.sh
```

If PyTorch + torchaudio are already installed in the chosen environment, the
installer keeps them and only installs NeMo and this project.

## 3. Install/configure NGC CLI

The included installer pins NGC CLI 4.34.10 and verifies NVIDIA's published
SHA256 before installing it under this project:

```bash
bash scripts/install_ngc_cli.sh
export PATH="$PWD/.tools/ngc-cli:$PATH"
ngc config set
```

`ngc config set` asks for your NGC API key. Do not put the key in this repository.

## 4. Download the Mandarin-English Parakeet checkpoint

```bash
bash scripts/download_model.sh
```

Equivalent NGC command:

```bash
ngc registry model download-version \
  "nvidia/riva/parakeet-ctc-riva-0-6b-unified-zh-cn:trainable_v3.0" \
  --dest models
```

The inference loader searches the downloaded directory for the `.nemo`
checkpoint and restores its original NeMo model class.

## 5. Prepare the hospital benchmark

Unzip the supplied `hotword_benchmark.zip` so this project can see:

```text
hotword_benchmark/
  audio/<id>.wav
  hotwords.json
  all_hotwords.json
  pseudo_transcripts.json
  evaluate.py
```

Validate it first:

```bash
python -m hotword_asr.validate_benchmark hotword_benchmark
```

For the zip supplied on 2026-08-08, the expected counts are:

- 71 WAV files
- 139 unique target hotwords
- 198 `(audio, hotword)` target instances

There is currently one vocabulary inconsistency in the supplied package:
`hotwords.json` contains `elbew`, while `all_hotwords.json` contains `elbow`.
The benchmark runner reports this instead of silently changing the data.

By default the runner builds the global vocabulary from the **union of all
`hotwords.json` entries**. This is still one global 139-word list for all 71
recordings, not a per-file oracle, and keeps the spotting labels consistent with
the current ground truth. After the benchmark is corrected, you can instead use:

```bash
--vocabulary-source all-hotwords
```

## 6. Smoke test

Run two recordings first:

```bash
source path.sh
CUDA_VISIBLE_DEVICES=0 python -m hotword_asr.benchmark \
  --benchmark-dir hotword_benchmark \
  --model models \
  --output-dir exp/smoke \
  --limit 2
```

## 7. Full benchmark

The convenient staged runner follows the usual `stage/stop_stage` pattern:

```bash
bash run.sh \
  --stage 2 \
  --stop-stage 3 \
  --gpuid 0 \
  --benchmark-dir /path/to/hotword_benchmark
```

Stages:

| stage | action |
|---:|---|
| 0 | `conda create` the Python 3.12 environment and install dependencies |
| 1 | download the trainable zh-CN Parakeet `.nemo` model from NGC |
| 2 | run Parakeet raw ASR and CTC-WS with both vocabulary policies |
| 3 | evaluate raw ASR and both Parakeet CTC-WS results |
| 4 | run Nemotron baseline and GPU-PB with both vocabulary policies |
| 5 | evaluate baseline and both Nemotron GPU-PB results |

Already completed per-audio inference is skipped. Add `--overwrite` when calling
`python -m hotword_asr.benchmark` directly if you intentionally want to recompute it.

Run only the Nemotron comparison and evaluation with:

```bash
bash run.sh \
  --stage 4 \
  --stop-stage 5 \
  --gpuid 0 \
  --benchmark-dir /path/to/hotword_benchmark
```

The default runs through stage 5. Set `--stop-stage 3` to omit the Nemotron
experiment.

`run.sh` sources `path.sh` after stage 0, so stages 1-5 always run inside the
same Conda environment. When starting directly from `--stage 1`, `--stage 2`,
`--stage 3`, `--stage 4`, or `--stage 5`, that environment must already have
been created once by stage 0.

## 8. Run one audio file

```bash
python -m hotword_asr.infer \
  --audio hotword_benchmark/audio/534.wav \
  --hotwords hotword_benchmark/all_hotwords.json \
  --model models \
  --output exp/534.json
```

The JSON contains:

- `raw_text`: greedy Parakeet CTC transcription
- `merged_text`: ASR after CTC-WS merge
- `predicted_hotwords`: canonical hotwords detected by CTC-WS
- `spotted`: individual CTC-WS hypotheses and chunk metadata
- `timing`: ASR time, spotting/merge time and RTF
- `runtime`: process/GPU peak-memory measurements for single-file inference

## 9. Benchmark outputs

`exp/parakeet_ctcws/` contains the baseline report plus separate CTC-WS runs
for the ground-truth union and phrase-boosting (`all_hotwords.json`)
vocabularies:

```text
ground_truth_union/raw_asr/<id>/transcription.json
ground_truth_union/ctcws_asr/<id>/transcription.json
ground_truth_union/details/<id>.json
ground_truth_union/predicted_keywords.json
phrase_boosting_vocabulary/ctcws_asr/<id>/transcription.json
phrase_boosting_vocabulary/details/<id>.json
phrase_boosting_vocabulary/predicted_keywords.json
report_raw_asr.xlsx
report_ctcws_ground_truth_union_asr.xlsx
report_ctcws_phrase_boosting_vocabulary_asr.xlsx
```

The original benchmark evaluator therefore computes:

- raw ASR: MER vs. pseudo transcript + hotword recall
- merged ASR: MER vs. pseudo transcript + hotword recall
- CTC-WS predictions: hotword precision / recall / F1

`runtime_metrics.json` additionally records:

- wall-clock time
- total audio duration
- real-time factor (RTF)
- x-real-time throughput
- peak process RSS
- peak CUDA allocated memory
- peak CUDA reserved memory

Model loading is outside the RTF timer. This makes the number represent steady
inference cost rather than download/initialization cost.

Nemotron results are written to `exp/nemotron_gpu_pb/` using the same two
vocabulary policies. The baseline is produced only in the ground-truth-union
run because it does not use phrase boosting:

```text
ground_truth_union/raw_asr/<id>/transcription.json
ground_truth_union/gpu_pb_asr/<id>/transcription.json
ground_truth_union/details/baseline/<id>.json
ground_truth_union/details/gpu_pb/<id>.json
phrase_boosting_vocabulary/gpu_pb_asr/<id>/transcription.json
phrase_boosting_vocabulary/details/gpu_pb/<id>.json
report_raw_asr.xlsx
report_gpu_pb_ground_truth_union_asr.xlsx
report_gpu_pb_phrase_boosting_vocabulary_asr.xlsx
```

## 10. CTC-WS parameters

Defaults follow NVIDIA's NeMo CTC-WS tutorial:

```text
beam_threshold       = 7.0
context_score        = 3.0
ctc_ali_token_weight = 0.5
```

Tune on a development subset rather than on the final 71-file test set. Example:

```bash
python -m hotword_asr.benchmark \
  --benchmark-dir hotword_benchmark \
  --model models \
  --beam-threshold 8.0 \
  --context-score 4.0 \
  --ctc-ali-token-weight 0.6 \
  --output-dir exp/tune_b8_c4_a06
```

For English medical terms the project automatically adds conservative
alternative graph paths for case, hyphen/space variants and spelled acronyms.
The CTC-WS output always maps these paths back to the canonical benchmark word.
Extra domain-specific pronunciations can be supplied through an aliases JSON:

```json
{
  "Mixtard": ["mix tard"],
  "X-RAY": ["x ray"]
}
```

Then pass `--aliases my_aliases.json`. The included
`config/hotword_aliases.json` is intentionally empty so no benchmark label is
silently corrected.

## 11. Nemotron GPU-PB parameters

The Nemotron condition uses NeMo's token-level GPU Phrase Boosting during
RNNT shallow-fusion decoding. Defaults follow NVIDIA's documented RNNT
recommendations where a fixed recommendation is available:

```text
strategy                     = greedy_batch
boosting_context_score       = 1.0
boosting_depth_scaling       = 2.0
boosting_bpe_mode            = case_insensitive
boosting_tree_alpha          = 1.0  # starting point; tune on development data
```

For example:

```bash
bash run.sh \
  --stage 4 \
  --stop-stage 5 \
  --nemotron-boosting-tree-alpha 0.5 \
  --gpuid 0
```

The staged runner evaluates both vocabulary policies for Parakeet CTC-WS and
Nemotron GPU-PB: `ground-truth-union` uses the union of benchmark labels, while
the phrase-boosting-vocabulary condition uses `all-hotwords` to consume
`all_hotwords.json` verbatim. When invoking either Python module directly,
select one policy with `--vocabulary-source`.

The checkpoint's serialized OmegaConf decoding configuration may omit the
optional `greedy.boosting_tree` node. The Nemotron runner creates that node when
GPU-PB is selected; its absence does not by itself mean that NeMo must be
reinstalled.

## 12. Long recordings

Some hospital recordings are several minutes long. Feeding a whole long file to
a 0.6B FastConformer at once can cause unnecessary GPU-memory growth. The runner
therefore uses non-overlapping 30-second chunks by default and batches chunks for
the acoustic model:

```bash
--chunk-seconds 30 --batch-size 8
```

The same CTC-WS graph is reused for every chunk. Non-overlapping chunks keep the
final transcript free of overlap duplicates. A keyword that lands exactly on a
chunk boundary can theoretically be missed, so chunk length is configurable.
The Nemotron runner uses the same chunking policy for a directly comparable
memory profile.

## References

- NVIDIA Mandarin-English Parakeet collection: <https://catalog.ngc.nvidia.com/orgs/nvidia/collections/parakeet-ctc-0.6b-zh-cn>
- NVIDIA NeMo CTC-WS tutorial: <https://github.com/NVIDIA-NeMo/Speech/blob/main/tutorials/asr/ASR_Context_Biasing.ipynb>
- NVIDIA Nemotron 3.5 ASR model card: <https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b>
- NVIDIA NeMo GPU-PB documentation: <https://docs.nvidia.com/nemo/speech/nightly/asr/asr_customization/word_boosting.html>
- NeMo Speech installation: <https://github.com/NVIDIA-NeMo/Speech>
- NGC CLI documentation: <https://docs.ngc.nvidia.com/cli/cmd.html>
