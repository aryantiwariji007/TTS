"""Dia TTS backend — lazy-loaded, GPU-only (CUDA required)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np

# Project root is three levels up: backends/ -> kokoro_tts/ -> kokoro-tts/
_PROJECT_ROOT = Path(__file__).parent.parent.parent

# ---------------------------------------------------------------------------
# Voice registry
# ---------------------------------------------------------------------------

DIA_VOICES: dict[str, dict] = {
    "british_female": {
        "ckpt": str(_PROJECT_ROOT / "models" / "dia_british_female" / "ckpt_epoch4.pth"),
        "label": "British Female (Dia)",
        "backend": "dia",
    },
    "british_male": {
        "ckpt": str(_PROJECT_ROOT / "models" / "dia_british_male" / "ckpt_epoch4.pth"),
        "label": "British Male (Dia)",
        "backend": "dia",
    },
}

SAMPLE_RATE = 44100

# ---------------------------------------------------------------------------
# Module-level lazy state
# ---------------------------------------------------------------------------

_loaded_models: dict[str, object] = {}
_warned_loading = False

_SPEAKER_TAG_RE = re.compile(r"\[S\d+\]")

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _has_speaker_tags(text: str) -> bool:
    return bool(_SPEAKER_TAG_RE.search(text))


def _wrap_speaker_tags(text: str) -> str:
    """Wrap plain text in a [S1] opening tag if no speaker tags are present.
    No closing tag — a trailing [S1] makes the model treat it as a new turn
    start and loop back into generation instead of producing EOS tokens."""
    if _has_speaker_tags(text):
        return text
    return f"[S1] {text}"


def _estimate_max_tokens(text: str) -> int:
    """Token budget: 1.5× expected speech duration.
    DAC runs at 86 fps; average English TTS ≈ 2.5 words/second → ~34 tok/word.
    1.5× headroom covers slower speech; duration trim handles the rest. Clamp 512–2048."""
    words = len(text.split())
    return max(512, min(2048, int(words * 34 * 1.5)))


def _trim_to_duration(audio: np.ndarray, sr: int, word_count: int) -> np.ndarray:
    """Hard-trim audio to expected speech duration + 20% buffer.
    More reliable than amplitude-based silence detection, which false-triggers
    on quiet fricatives and inter-word gaps in Dia output."""
    # 2.3 words/second is a conservative (slightly slow) speech rate so we
    # never cut off the last word even for slower synthesis.
    expected_secs = word_count / 2.3
    max_samples = int(expected_secs * 1.2 * sr)   # 20% buffer
    trimmed = audio[:max_samples]
    trim_secs = len(trimmed) / sr
    print(f"  Dia trim: {len(audio)/sr:.1f}s → {trim_secs:.1f}s  ({word_count} words)", flush=True)
    return trimmed


def _resolve_ckpt(voice: str) -> str:
    if voice not in DIA_VOICES:
        raise ValueError(
            f"Unknown Dia voice: {voice!r}. "
            f"Valid choices: {list(DIA_VOICES)}"
        )
    ckpt_path = DIA_VOICES[voice]["ckpt"]
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "Run: python scripts/download_dia_models.py"
        )
    return ckpt_path


def _load_model(voice: str, compile_model: bool = True):
    """Load (or return cached) Dia model for the given voice."""
    global _warned_loading

    if voice in _loaded_models:
        return _loaded_models[voice]

    # Deferred imports — only executed when a Dia voice is actually requested.
    try:
        import torch
        from dia.model import Dia
    except ImportError as exc:
        raise ImportError(
            "Dia dependencies are not installed.\n"
            "Install them with: pip install -r requirements-dia.txt"
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Dia voices require a CUDA-capable GPU.\n"
            "Use a Kokoro voice for CPU-only synthesis."
        )

    if not _warned_loading:
        print(
            "Loading Dia model (~4.4 GB VRAM). This may take 30-60s on first run.",
            flush=True,
        )
        _warned_loading = True

    ckpt_path = _resolve_ckpt(voice)

    # The fine-tuned checkpoint is a complete state dict (all 343 keys match).
    # from_local loads it directly in float32 — bfloat16 loses enough precision
    # to cause generation drift and static noise within a few seconds.
    _config_path = str(_PROJECT_ROOT / "models" / "dia_base" / "config.json")
    model = Dia.from_local(_config_path, ckpt_path, compute_dtype="float32")

    _loaded_models[voice] = model
    return model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def synthesise(
    text: str,
    voice: str,
    *,
    temperature: float = 1.3,
    top_p: float = 0.90,
    top_k: int = 45,
    guidance_scale: float = 3.0,
    max_new_tokens: int = 3072,
    seed: Optional[int] = None,
    compile_model: bool = True,
) -> np.ndarray:
    """Synthesise text using a Dia voice. Returns float32 audio at 44100 Hz."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "Dia dependencies are not installed.\n"
            "Install them with: pip install -r requirements-dia.txt"
        ) from exc

    model = _load_model(voice, compile_model=compile_model)

    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    word_count = len(text.split())   # count before [S1] tag is added
    text = _wrap_speaker_tags(text)
    token_budget = _estimate_max_tokens(text) if max_new_tokens == 3072 else max_new_tokens

    audio = model.generate(
        text,
        temperature=temperature,
        top_p=top_p,
        cfg_filter_top_k=top_k,
        cfg_scale=guidance_scale,
        max_tokens=token_budget,
        use_torch_compile=False,
    )

    if not isinstance(audio, np.ndarray):
        audio = np.array(audio)
    audio = audio.astype(np.float32)
    if audio.ndim > 1:
        audio = audio.squeeze()

    return _trim_to_duration(audio, SAMPLE_RATE, word_count)
