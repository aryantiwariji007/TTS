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

    print(f"Loading GPU voice model for {voice}…", flush=True)
    model = Dia.from_local(
        config_path=str(config_path),
        checkpoint_path=str(ckpt_path),
        compute_dtype="float32",
        device=torch.device("cuda"),
    )

    _models[voice] = model
    print(f"GPU voice model for {voice} ready.", flush=True)
    return model


def _trim_trailing_loop(audio: np.ndarray, sr: int,
                        silence_ms: int = 400,
                        threshold: float = 0.008) -> np.ndarray:
    """Cut audio at the first silence ≥ silence_ms after speech starts.

    The fine-tuned Dia models never emit a natural EOS, so they loop at the
    end of each chunk.  A gap ≥ 400 ms marks the end of real speech; everything
    after that is the loop restart and should be discarded.
    """
    if len(audio) == 0:
        return audio
    win = int(sr * 0.025)              # 25 ms analysis window
    min_sil = max(1, silence_ms // 25) # frames needed to declare silence
    n_frames = len(audio) // win
    if n_frames == 0:
        return audio

    rms = np.array([
        float(np.sqrt(np.mean(audio[i * win:(i + 1) * win].astype(np.float64) ** 2)))
        for i in range(n_frames)
    ])

    speech_started = False
    sil_run = 0
    for i, r in enumerate(rms):
        if not speech_started:
            if r > threshold:
                speech_started = True
        else:
            if r < threshold:
                sil_run += 1
                if sil_run >= min_sil:
                    cut = (i - sil_run + 1) * win
                    return audio[:cut]
            else:
                sil_run = 0
    return audio


def synthesise(
    text: str,
    voice: str,
    *,
    temperature: float = 1.2,
    top_p: float = 0.95,
    top_k: int = 45,
    guidance_scale: float = 3.0,
    seed: int | None = None,
    compile_model: bool = False,
    **_,
) -> np.ndarray:
    if voice not in DIA_VOICES:
        raise ValueError(f"Unknown GPU voice: {voice!r}.")

    import re
    speaker = DIA_VOICES[voice]["speaker"]
    model   = _load_model(voice)

    # Normalise text: expand ellipsis, collapse whitespace
    text = re.sub(r'\.{2,}', '. ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    tagged     = f"{speaker} {text}"
    word_count = len(text.split())
    # Budget: ~86 codec frames/s at 44100 Hz. Keep budget ≥860 tokens (~10 s);
    # values below ~800 cause the model to produce silence on many inputs.
    max_tokens = max(860, min(3072, int(word_count / 1.8 * 86 * 1.3)))
    print(f"  GPU generating ({word_count}w, {max_tokens}tok, T={temperature}): {tagged[:70]}", flush=True)

    import torch
    if seed is not None:
        torch.manual_seed(seed)

    audio = model.generate(
        tagged,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        cfg_scale=guidance_scale,
        cfg_filter_top_k=top_k,
        use_torch_compile=compile_model,
    )

    if audio is None or (hasattr(audio, 'size') and audio.size == 0):
        print("  GPU returned empty audio — skipping chunk", flush=True)
        return np.zeros(0, dtype=np.float32)

    if not isinstance(audio, np.ndarray):
        audio = np.array(audio, dtype=np.float32)
    audio = audio.astype(np.float32)
    if audio.ndim > 1:
        audio = audio.squeeze()

    raw_dur = len(audio) / SAMPLE_RATE
    audio = _trim_trailing_loop(audio, SAMPLE_RATE)
    peak = float(np.max(np.abs(audio))) if audio.size > 0 else 0.0
    print(f"  GPU done: {raw_dur:.1f}s → {len(audio)/SAMPLE_RATE:.1f}s trimmed  peak={peak:.4f}", flush=True)
    return audio
