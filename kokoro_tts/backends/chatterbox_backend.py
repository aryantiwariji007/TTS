"""Chatterbox TTS GPU backend — voice cloning with reference audio.

Uses Resemble AI's Chatterbox TTS (2025) to synthesise speech that matches
a reference audio clip. Place the reference files before use:
  reference_audio/british_female.mp3
  reference_audio/british_male.mp3
"""

from __future__ import annotations
from pathlib import Path
import numpy as np

_PROJECT_ROOT = Path(__file__).parent.parent.parent

GPU_VOICES: dict[str, dict] = {
    "british_female": {
        "label": "GPU Female",
        "backend": "chatterbox",
        "reference_audio": "reference_audio/british_female.mp3",
    },
    "british_male": {
        "label": "GPU Male",
        "backend": "chatterbox",
        "reference_audio": "reference_audio/british_male.mp3",
    },
}

SAMPLE_RATE = 24000

_model = None


def _load_model():
    global _model
    if _model is not None:
        return _model
    import torch
    from chatterbox.tts import ChatterboxTTS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tag = "CUDA" if device == "cuda" else "CPU"
    print(f"Loading Chatterbox TTS ({tag})…", flush=True)
    _model = ChatterboxTTS.from_pretrained(device=device)
    print("Chatterbox TTS ready.", flush=True)
    return _model


def synthesise(
    text: str,
    voice: str,
    *,
    speed: float = 1.0,
    **_,
) -> tuple[np.ndarray, int]:
    if voice not in GPU_VOICES:
        raise ValueError(f"Unknown GPU voice: {voice!r}")

    cfg = GPU_VOICES[voice]
    ref_path = _PROJECT_ROOT / cfg["reference_audio"]
    if not ref_path.exists():
        raise FileNotFoundError(
            f"Reference audio not found: {ref_path}\n"
            f"Place the MP3 file there before using GPU voices."
        )

    model = _load_model()
    sr = getattr(model, 'sr', SAMPLE_RATE)

    print(f"  Chatterbox ({cfg['label']}): {text[:70]}", flush=True)

    wav = model.generate(
        text,
        audio_prompt_path=str(ref_path),
        exaggeration=0.5,
        cfg_weight=0.5,
    )

    import torch
    if isinstance(wav, torch.Tensor):
        wav = wav.squeeze().cpu().numpy().astype(np.float32)
    else:
        wav = np.array(wav, dtype=np.float32).squeeze()

    print(f"  Chatterbox done: {len(wav)/sr:.1f}s", flush=True)
    return wav, sr
