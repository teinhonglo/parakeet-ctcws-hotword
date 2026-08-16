from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_funasr_requirements_and_installer_include_pytorch() -> None:
    requirements = (ROOT / "requirements-funasr.txt").read_text(encoding="utf-8")
    assert "torch\n" in requirements
    assert "torchaudio\n" in requirements
    
    for package in ("transformers", "sentencepiece", "safetensors", "soundfile", "librosa"):
        assert f"{package}\n" in requirements

    installer = (ROOT / "scripts/install_funasr.sh").read_text(encoding="utf-8")
    assert "FUNASR_TORCH_INDEX_URL" in installer
    assert "python -c 'import torch, torchaudio'" in installer
    assert "PYTHONNOUSERSITE=1" in installer
    assert "PIP_USER=false" in installer
    assert "python -m pip check" in installer
    assert "import openpyxl" in installer
    assert 'version("wetext")' in installer
    assert "python -m hotword_asr.funasr_benchmark --help" in installer
