"""Dia TTS backend — minimal, following nari-labs/Dia documentation exactly."""

from __future__ import annotations

from pathlib import Path
import numpy as np

_PROJECT_ROOT = Path(__file__).parent.parent.parent

DIA_VOICES: dict[str, dict] = {
    "british_female": {"speaker": "[S1]", "label": "GPU Female", "backend": "dia"},
    "british_male":   {"speaker": "[S2]", "label": "GPU Male",   "backend": "dia"},
}

SAMPLE_RATE = 44100

_model = None


def _load_model():
    global _model
    if _model is not None:
        return _model

    import torch
    from dia.model import Dia

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. GPU voices require an NVIDIA GPU.")

    print("Loading GPU voice model…", flush=True)
    base_path = str(_PROJECT_ROOT / "models" / "dia_base")
    _model = Dia.from_pretrained(base_path, compute_dtype="float32")
    print("GPU voice model ready.", flush=True)
    return _model


def synthesise(text: str, voice: str, **kwargs) -> np.ndarray:
    if voice not in DIA_VOICES:
        raise ValueError(f"Unknown GPU voice: {voice!r}.")

    speaker = DIA_VOICES[voice]["speaker"]
    model   = _load_model()

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
