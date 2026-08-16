# Parakeet, Nemotron, and Fun-ASR-Nano Hotword Benchmark

This repository evaluates the existing Parakeet FastConformer-CTC/NeMo CTC
Word Spotter, Nemotron 3.5 RNNT/NeMo GPU Phrase Boosting, and the official
Fun-ASR-Nano-2512 runtime hotword interface.
It preserves Taiwan Traditional evaluator output, Simplified-Chinese contextual
decoder input, runtime measurement, and the benchmark's own `evaluate.py`.

## Primary experiment matrix

| Model | Condition | Vocabulary |
|---|---|---|
| Parakeet | Vanilla | none |
| Parakeet | CTC-WS + All Hotwords | complete `all_hotwords.json` for every audio |
| Parakeet | CTC-WS + Oracle Hotwords | only `hotwords.json[audio_id]` |
| Nemotron | Vanilla | none |
| Nemotron | GPU-PB + All Hotwords | complete `all_hotwords.json` for every audio |
| Nemotron | GPU-PB + Oracle Hotwords | only `hotwords.json[audio_id]` |
| Fun-ASR-Nano | Vanilla | none |
| Fun-ASR-Nano | Hotword + All Hotwords | complete `all_hotwords.json` for every audio |
| Fun-ASR-Nano | Hotword + Oracle Hotwords | only `hotwords.json[audio_id]` |

> **ground-truth-union is NOT Oracle.** It is the global union of ground-truth
> hotwords across the benchmark and is no longer used as a primary experiment.
> Oracle is strictly the current recording's `hotword_map[audio_id]` list.

The supplied benchmark discrepancy (`elbew` in `hotwords.json`, `elbow` in
`all_hotwords.json`) is deliberately not repaired. The runner reports the
vocabulary difference and passes each source verbatim to its defined condition.

## Installation and data

Requirements are Linux, Python 3.12+, and a CUDA-capable NVIDIA environment for
full inference. Install the existing environment and configure NGC for the
trainable Parakeet checkpoint:

```bash
bash scripts/install.sh
source path.sh
bash scripts/install_ngc_cli.sh
ngc config set
bash scripts/download_model.sh
```

Install FunASR separately so its dependencies cannot alter the working NeMo
environment. Its complete dependency list is kept in
`requirements-funasr.txt`; the shared evaluator requirements are included from
the existing `requirements.txt`:

```bash
bash scripts/install_funasr.sh
```

The installer explicitly installs `torch` and `torchaudio` before importing
FunASR. It defaults to the official PyTorch CUDA 12.6 wheel index; set
`FUNASR_TORCH_INDEX_URL` to the index matching the server's driver/toolkit when
needed. It also disables Python user-site packages so an old `~/.local` FunASR
cannot mask missing packages in `funasr_hotword`.

One invocation installs the model runtime, audio loading, Hugging Face/ModelScope
support, Traditional-Chinese output normalization, benchmark spreadsheet/ITN
dependencies, and this local package. It finishes with `pip check`, dependency
imports, and the local FunASR benchmark CLI check; no second requirements or
project-install command is needed.

The installer resolves the target environment's Python with `conda run -n
funasr_hotword` and uses that exact executable for every pip/install/check
command. `run_funasr.sh` also refuses to start inference when `AutoModel` is not
importable and prints the selected environment, Python path, and repair command.

For a fresh Conda environment, the installer creates the Python `site-packages`
directory when it is not present and verifies writability with an actual probe
file before installing packages.

It is safe to run this after `source path.sh`. That command activates
`parakeet_ctcws`, but it does not determine which Conda channels are queried;
channel configuration comes from Conda's user/system configuration. The

installer creates (or reuses) `funasr_hotword` and runs every install through
that environment's absolute Python, so FunASR packages are not installed into
`parakeet_ctcws`.

The installer creates the environment with `--override-channels` and uses only
`conda-forge` by default, so unrelated channels in a user or system `.condarc`
(such as an unavailable Intel package channel) cannot break this environment.
Override it with `FUNASR_CONDA_CHANNEL=<channel>` if required. To diagnose a
certificate error, first inspect `conda config --show-sources`. If all HTTPS
channels fail behind an institutional proxy, configure the supplied CA bundle
with `conda config --set ssl_verify /path/to/organization-ca-bundle.pem` rather
than disabling SSL verification globally.

The benchmark must retain its original layout:

```text
hotword_benchmark/
  audio/<audio_id>.wav
  hotwords.json
  all_hotwords.json
  pseudo_transcripts.json
  evaluate.py
```

## One-command benchmark

One invocation runs and evaluates all nine experiments:

```bash
bash run.sh \
  --stage 2 \
  --stop-stage 7 \
  --gpuid 0 \
  --benchmark-dir /path/to/hotword_benchmark
```

Stages are: (1) Parakeet download, (2) all Parakeet inference, (3) all
Parakeet evaluation, (4) all Nemotron inference, (5) all Nemotron evaluation,
(6) all Fun-ASR-Nano inference, and (7) all Fun-ASR-Nano evaluation.
`--limit N` selects the same first N IDs for every condition;
`--overwrite` recomputes each selected condition's own cached result.

`path.sh` selects an environment from `BACKEND`: `default`, `parakeet`, and
`nemotron` activate `parakeet_ctcws`, while `funasr` activates
`funasr_hotword`. `run.sh` delegates commands to `run_parakeet.sh`,
`run_nemotron.sh`, or `run_funasr.sh`; each launcher activates its own backend
environment and then replaces itself with the requested command. Consequently,
Stages 1--5 cannot leak their NeMo environment into Stages 6--7. Optional
`PARAKEET_CUDA_DIR` and `FUNASR_CUDA_DIR` select backend-specific CUDA toolkits;
when unset, the existing system CUDA paths are preserved.
Conda shell integration defaults to
`/share/homes/teinhonglo/anaconda3/bin/conda shell.bash hook` for this server;
an explicit `CONDA_EXE` or a `conda` executable on `PATH` remains a fallback for
other installations and automated tests.

The launchers can also be used directly, for example:

```bash
bash run_parakeet.sh python -m hotword_asr.benchmark --help
bash run_nemotron.sh python -m hotword_asr.nemotron_benchmark --help
bash run_funasr.sh python -m hotword_asr.funasr_benchmark --help
```

Stage 2 invokes one Python process and loads Parakeet once. Its shared model is
used by Vanilla, one reusable global All-Hotwords graph, and per-audio Oracle
graphs. Stage 4 likewise loads Nemotron once; GPU-PB is disabled for Vanilla,
configured once for All Hotwords, and reconfigured from an auditable per-audio
phrase file for Oracle. Model loading remains outside condition RTF meters.
Stage 6 similarly makes one `AutoModel` and passes each condition's exact
audited list through `generate(..., hotwords=...)`; its VAD is `fsmn-vad`, its
default fixed language is `中文`, and internal ITN is disabled.

For debugging, all three runners accept `--condition all`, `vanilla`,
`all-hotwords`, or `oracle-hotwords`:

```bash
python -m hotword_asr.benchmark --benchmark-dir hotword_benchmark \
  --model models --output-dir exp/parakeet_ctcws --condition all
python -m hotword_asr.nemotron_benchmark --benchmark-dir hotword_benchmark \
  --model nvidia/nemotron-3.5-asr-streaming-0.6b \
  --output-dir exp/nemotron_gpu_pb --condition all
python -m hotword_asr.funasr_benchmark --benchmark-dir hotword_benchmark \
  --model FunAudioLLM/Fun-ASR-Nano-2512 \
  --output-dir exp/funasr_nano --condition all --language 中文
```

## Outputs and auditing

Each model root has `vanilla/`, `all_hotwords/`, and `oracle_hotwords/`.
Each condition contains `asr/<id>/transcription.json`, `details/<id>.json`, and
the mandatory `hotwords_used.json`. Parakeet contextual conditions also contain
actual CTC-WS detections in `predicted_keywords.json`; Nemotron Oracle stores
converted GPU-PB phrases in `oracle_hotwords/phrase_files/<id>.txt`.
Fun-ASR creates no synthetic keyword detections: it preserves raw model text
and records both `hotwords_used` and the identical `model_hotwords` sent to the
official inference call in each details file.

Reports are:

```text
exp/parakeet_ctcws/report_vanilla_asr.xlsx
exp/parakeet_ctcws/report_ctcws_all_hotwords_asr.xlsx
exp/parakeet_ctcws/report_ctcws_oracle_hotwords_asr.xlsx
exp/nemotron_gpu_pb/report_vanilla_asr.xlsx
exp/nemotron_gpu_pb/report_gpu_pb_all_hotwords_asr.xlsx
exp/nemotron_gpu_pb/report_gpu_pb_oracle_hotwords_asr.xlsx
exp/funasr_nano/report_vanilla_asr.xlsx
exp/funasr_nano/report_hotword_all_hotwords_asr.xlsx
exp/funasr_nano/report_hotword_oracle_hotwords_asr.xlsx
```

`run_config.json` records the unambiguous condition sources and shared decoding
hyperparameters. `runtime_metrics.json` records each condition separately,
including audio duration/count, wall time, RTF, throughput, process/GPU peaks,
and per-audio timing.
