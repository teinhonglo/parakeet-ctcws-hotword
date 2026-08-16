from __future__ import annotations

import subprocess
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_SH = PROJECT_ROOT / "run.sh"


def test_run_sh_is_independent_of_calling_directory(tmp_path: Path) -> None:
    missing_benchmark = tmp_path / "does-not-exist"
    result = subprocess.run(
        [
            "bash",
            str(RUN_SH),
            "--stage",
            "2",
            "--stop-stage",
            "5",
            "--benchmark-dir",
            str(missing_benchmark),
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "missing benchmark file" in result.stderr
    # Reaching validation proves parse_options.sh was found relative to run.sh,
    # rather than relative to the caller's working directory.
    assert "local/parse_options.sh" not in result.stderr


def test_run_sh_rejects_inverted_stage_range(tmp_path: Path) -> None:
    result = subprocess.run(
        ["bash", str(RUN_SH), "--stage", "5", "--stop-stage", "2"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "cannot exceed" in result.stderr


def test_run_sh_routes_all_conditions_and_evaluations(tmp_path: Path) -> None:
    benchmark = tmp_path / "benchmark"
    (benchmark / "audio").mkdir(parents=True)
    for name in ("hotwords.json", "all_hotwords.json", "evaluate.py"):
        (benchmark / name).write_text("{}\n", encoding="utf-8")

    fake_bin = tmp_path / "bin"
    conda_base = tmp_path / "conda"
    conda_profile = conda_base / "etc" / "profile.d" / "conda.sh"
    fake_bin.mkdir()
    conda_profile.parent.mkdir(parents=True)
    conda_profile.write_text("conda() { return 0; }\n", encoding="utf-8")
    conda = fake_bin / "conda"
    conda.write_text(
        f"#!/usr/bin/env bash\n[[ $1 == info ]] && echo {conda_base!s}\n",
        encoding="utf-8",
    )
    conda.chmod(0o755)

    calls = tmp_path / "python-calls.txt"
    python = fake_bin / "python"
    python.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$*\" >> \"$RUN_SH_CALLS\"\n",
        encoding="utf-8",
    )
    python.chmod(0o755)
    environment = os.environ.copy()
    environment.update(
        PATH=f"{fake_bin}:{environment['PATH']}", RUN_SH_CALLS=str(calls)
    )
    parakeet_exp = tmp_path / "parakeet-exp"
    nemotron_exp = tmp_path / "nemotron-exp"
    parakeet_exp.mkdir()
    nemotron_exp.mkdir()

    result = subprocess.run(
        [
            "bash",
            str(RUN_SH),
            "--stage",
            "2",
            "--stop-stage",
            "5",
            "--benchmark-dir",
            str(benchmark),
            "--exp-dir",
            str(parakeet_exp),
            "--nemotron-exp-dir",
            str(nemotron_exp),
            "--limit",
            "2",
            "--overwrite",
        ],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    logged = calls.read_text(encoding="utf-8").splitlines()
    assert sum("-m hotword_asr.benchmark" in call for call in logged) == 1
    assert sum("-m hotword_asr.nemotron_benchmark" in call for call in logged) == 1
    inference_calls = [call for call in logged if " -m hotword_asr." in f" {call}"]
    assert all("--condition all" in call for call in inference_calls)
    assert sum("evaluate.py" in call for call in logged) == 6
    assert "Completed experiments:" in result.stdout
