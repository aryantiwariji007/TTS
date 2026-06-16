"""Dia TTS backend — minimal, following nari-labs/Dia documentation exactly."""

from __future__ import annotations

from pathlib import Path
import numpy as np

_PROJECT_ROOT = Path(__file__).parent.parent.parent

DIA_VOICES: dict[str, dict] = {
    "british_female": {"speaker": "[S1]", "label": "GPU Female", "backend": "dia", "model_dir": "dia_british_female"},
    "british_male":   {"speaker": "[S2]", "label": "GPU Male",   "backend": "dia", "model_dir": "dia_british_male"},
}

SAMPLE_RATE = 44100

_models: dict = {}

_BASE_REPO = "nari-labs/Dia-1.6B-0626"


def _resolve_ckpt(voice: str) -> Path:
    if voice not in DIA_VOICES:
        raise ValueError(f"Unknown GPU voice: {voice!r}.")
    model_dir = _PROJECT_ROOT / "models" / DIA_VOICES[voice]["model_dir"]
    sentinel = model_dir / "ckpt_epoch4.pth"
    if not sentinel.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found for {voice!r} at {model_dir}.\n"
            f"Run: python scripts/download_dia_models.py"
        )
    return model_dir


def _ensure_config(model_dir: Path) -> Path:
    """Download config.json from the base Dia repo if not already present."""
    config_path = model_dir / "config.json"
    if not config_path.exists():
        print("Downloading config.json from base Dia repo (~1 KB)…", flush=True)
        from huggingface_hub import hf_hub_download
        hf_hub_download(
            repo_id=_BASE_REPO,
            filename="config.json",
            local_dir=str(model_dir),
        )
    return config_path


def _load_model(voice: str):
    if voice in _models:
        return _models[voice]

    import torch
    from dia.model import Dia

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. GPU voices require an NVIDIA GPU.")

    model_dir  = _resolve_ckpt(voice)
    ckpt_path  = model_dir / "ckpt_epoch4.pth"
    config_path = _ensure_config(model_dir)

    print(f"Loading GPU voice model for {voice} onto CPU first (saves VRAM)…", flush=True)
    # Load on CPU to avoid VRAM OOM: float32 checkpoint is 6.4 GB, RTX 3050 has only 4 GB.
    # float16 halves the model to ~3.2 GB; loading to CPU first prevents double-allocation.
    model = Dia.from_local(
        config_path=str(config_path),
        checkpoint_path=str(ckpt_path),
        compute_dtype="float16",
        device=torch.device("cpu"),
    )
    print("Moving model to CUDA…", flush=True)
    model.model.to(torch.device("cuda"))
    model.device = torch.device("cuda")
    if hasattr(model, "dac_model") and model.dac_model is not None:
        model.dac_model.to(torch.device("cuda"))

    _models[voice] = model
    print(f"GPU voice model for {voice} ready.", flush=True)
    return model


def synthesise(text: str, voice: str, **kwargs) -> np.ndarray:
    if voice not in DIA_VOICES:
        raise ValueError(f"Unknown GPU voice: {voice!r}.")

    speaker = DIA_VOICES[voice]["speaker"]
    model   = _load_model(voice)

    tagged     = f"{speaker} {text}"
    word_count = len(text.split())
    # Budget: ~86 codec frames per second, Dia speaks ~1.4 wps.
    # Allow 1.25× expected duration — enough for natural pacing variance
    # without letting loops run long before the hard stop.
    max_tokens = max(256, min(3072, int(word_count / 1.4 * 86 * 1.25)))
    print(f"  GPU generating ({word_count}w, {max_tokens}tok): {tagged[:70]}", flush=True)

    audio = model.generate(tagged, max_tokens=max_tokens, use_torch_compile=False)

    if not isinstance(audio, np.ndarray):
        audio = np.array(audio)
    audio = audio.astype(np.float32)
    if audio.ndim > 1:
        audio = audio.squeeze()

    print(f"  GPU done: {len(audio)/SAMPLE_RATE:.1f}s", flush=True)
    return audio
