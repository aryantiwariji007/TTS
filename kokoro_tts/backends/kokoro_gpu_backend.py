"""GPU Kokoro backend — runs the PyTorch Kokoro model on CUDA.

Replaces the Dia backend for GPU voices.  Uses the same built-in British
voices as the CPU Kokoro ONNX path, so no reference audio is needed.
"""

from __future__ import annotations
import numpy as np

SAMPLE_RATE = 24000

GPU_VOICES: dict[str, dict] = {
    "british_female": {
        "voice_id": "bf_isabella",
        "label": "GPU Female",
        "backend": "kokoro_gpu",
        "lang_code": "b",
    },
    "british_male": {
        "voice_id": "bm_daniel",
        "label": "GPU Male",
        "backend": "kokoro_gpu",
        "lang_code": "b",
    },
}

_pipelines: dict = {}


def _load_pipeline(lang_code: str = "b"):
    if lang_code in _pipelines:
        return _pipelines[lang_code]

    import torch
    from kokoro import KPipeline

    device = "cuda" if torch.cuda.is_available() else None
    tag = "CUDA" if device == "cuda" else "CPU"
    print(f"Loading Kokoro GPU pipeline (lang={lang_code!r}, device={tag})…", flush=True)
    pipeline = KPipeline(lang_code=lang_code, device=device)
    _pipelines[lang_code] = pipeline
    print("Kokoro GPU pipeline ready.", flush=True)
    return pipeline


def synthesise(
    text: str,
    voice: str,
    *,
    speed: float = 1.0,
    voice_b: str | None = None,
    blend: float = 0.0,
    **_,
) -> np.ndarray:
    if voice not in GPU_VOICES:
        raise ValueError(f"Unknown GPU voice: {voice!r}")

    cfg = GPU_VOICES[voice]
    pipeline = _load_pipeline(cfg["lang_code"])

    # Build voice pack — blend two style tensors if requested
    if voice_b and voice_b in GPU_VOICES and blend >= 0.99:
        voice_pack = GPU_VOICES[voice_b]["voice_id"]
        label = GPU_VOICES[voice_b]["voice_id"]
    elif voice_b and voice_b in GPU_VOICES and blend > 0.01:
        import torch
        pack_a = pipeline.load_voice(GPU_VOICES[voice]["voice_id"])
        pack_b = pipeline.load_voice(GPU_VOICES[voice_b]["voice_id"])
        voice_pack = pack_a * (1.0 - blend) + pack_b * blend
        label = f"{GPU_VOICES[voice]['voice_id']}+{GPU_VOICES[voice_b]['voice_id']} {blend:.0%}"
    else:
        voice_pack = GPU_VOICES[voice]["voice_id"]
        label = GPU_VOICES[voice]["voice_id"]

    print(f"  GPU Kokoro ({label}): {text[:70]}", flush=True)

    chunks: list[np.ndarray] = []
    for result in pipeline(text, voice=voice_pack, speed=speed):
        audio = result.audio
        if audio is not None and audio.numel() > 0:
            chunks.append(audio.numpy().astype(np.float32))

    if not chunks:
        return np.zeros(0, dtype=np.float32)

    result_audio = np.concatenate(chunks)
    print(f"  GPU Kokoro done: {len(result_audio)/SAMPLE_RATE:.1f}s", flush=True)
    return result_audio
