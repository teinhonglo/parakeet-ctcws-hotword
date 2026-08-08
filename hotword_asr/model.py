from __future__ import annotations

from pathlib import Path
from typing import Any


def resolve_nemo_file(model_spec: str | Path) -> Path | None:
    path = Path(model_spec).expanduser()
    if not path.exists():
        return None
    if path.is_file():
        if path.suffix != ".nemo":
            raise ValueError(f"Expected a .nemo model, got {path}")
        return path.resolve()

    candidates = sorted(path.rglob("*.nemo"))
    if not candidates:
        raise FileNotFoundError(f"No .nemo checkpoint found under {path}")
    if len(candidates) > 1:
        names = "\n  ".join(str(x) for x in candidates)
        raise RuntimeError(
            f"Multiple .nemo checkpoints found under {path}. Pass one explicitly:\n  {names}"
        )
    return candidates[0].resolve()


def load_ctc_model(model_spec: str | Path, device: str = "cuda") -> Any:
    """Load an NGC-downloaded .nemo checkpoint or a NeMo pretrained model id."""
    import torch
    from nemo.collections.asr.models import ASRModel
    from nemo.utils import model_utils

    nemo_file = resolve_nemo_file(model_spec)
    if nemo_file is not None:
        cfg = ASRModel.restore_from(restore_path=str(nemo_file), return_config=True)
        imported_class = model_utils.import_class_by_path(cfg.target)
        model = imported_class.restore_from(restore_path=str(nemo_file))
    else:
        model = ASRModel.from_pretrained(model_name=str(model_spec))

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    model = model.to(device)
    model.eval()
    model.freeze()

    if not hasattr(model, "decoding") or not hasattr(model.decoding, "blank_id"):
        raise TypeError("Selected NeMo model does not expose a CTC decoding.blank_id")
    return model

