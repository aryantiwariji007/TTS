#!/usr/bin/env python3
import io
import os
import sys
import math
import numpy as np
import soundfile as sf
from flask import Flask, request, send_file, jsonify

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from kokoro_onnx import Kokoro

# Kokoro GPU backend — PyTorch Kokoro on CUDA, same voices as CPU path.
# Commented out while experimenting with Chatterbox TTS voice cloning.
# To switch back: comment out the Chatterbox block below and uncomment this one.
# try:
#     import importlib.util as _ilu
#     _spec = _ilu.spec_from_file_location(
#         'kokoro_gpu_backend',
#         os.path.join(SCRIPT_DIR, 'kokoro_tts', 'backends', 'kokoro_gpu_backend.py'),
#     )
#     _gpu_mod = _ilu.module_from_spec(_spec)
#     _spec.loader.exec_module(_gpu_mod)
#     GPU_VOICES     = _gpu_mod.GPU_VOICES
#     _gpu_synth     = _gpu_mod.synthesise
#     _GPU_AVAILABLE = True
# except Exception as _e:
#     GPU_VOICES     = {}
#     _gpu_synth     = None
#     _GPU_AVAILABLE = False
#     print(f"  Kokoro GPU backend unavailable: {_e}")

# Chatterbox TTS GPU backend — voice cloning with British reference audio.
try:
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location(
        'chatterbox_backend',
        os.path.join(SCRIPT_DIR, 'kokoro_tts', 'backends', 'chatterbox_backend.py'),
    )
    _gpu_mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_gpu_mod)
    GPU_VOICES     = _gpu_mod.GPU_VOICES
    _gpu_synth     = _gpu_mod.synthesise
    _GPU_AVAILABLE = True
except Exception as _e:
    GPU_VOICES     = {}
    _gpu_synth     = None
    _GPU_AVAILABLE = False
    print(f"  Chatterbox backend unavailable: {_e}")

_GPU_IDS = set(GPU_VOICES.keys())

MODEL_PATH  = os.path.join(SCRIPT_DIR, 'kokoro-v1.0.onnx')
VOICES_PATH = os.path.join(SCRIPT_DIR, 'voices-v1.0.bin')

app = Flask(__name__, static_folder=SCRIPT_DIR, static_url_path='')

_kokoro = None

def get_kokoro():
    global _kokoro
    if _kokoro is None:
        print("  Loading Kokoro model...")
        _kokoro = Kokoro(MODEL_PATH, VOICES_PATH)
        print("  Model ready.\n")
    return _kokoro

BRITISH_VOICES = [
    ('bf_isabella', 'Isabella'),
    ('bf_lily',     'Lily'),
    ('bm_fable',    'Fable'),
    ('bm_lewis',    'Lewis'),
    ('bf_alice',    'Alice'),
    ('bm_daniel',   'Daniel'),
]

def split_text(text, size=500):
    sentences = [s.strip() + '.' for s in text.replace('\n', ' ').split('.') if s.strip()]
    chunks, cur, cur_len = [], [], 0
    for s in sentences:
        if cur_len + len(s) > size and cur:
            chunks.append(' '.join(cur))
            cur, cur_len = [], 0
        cur.append(s)
        cur_len += len(s)
    if cur:
        chunks.append(' '.join(cur))
    return chunks or [text]

# ─── Effects ────────────────────────────────────────────────────────────────

def pitch_shift(samples, sr, semitones):
    """Shift pitch without changing duration using FFT resampling (O(n log n))."""
    if abs(semitones) < 0.05:
        return samples.astype(np.float32)
    from scipy.signal import resample as fft_resample
    ratio = 2.0 ** (semitones / 12.0)
    n = len(samples)
    n2 = max(1, int(round(n / ratio)))
    # Step 1: change pitch+speed via FFT resample (fast for any length)
    step1 = fft_resample(samples.astype(np.float64), n2)
    # Step 2: restore original duration (keeps pitch shift, corrects speed)
    step2 = fft_resample(step1, n)
    return step2.astype(np.float32)

def apply_eq(samples, sr, bass_db=0.0, treble_db=0.0):
    """Low-shelf (500 Hz) and high-shelf (2 kHz) EQ — audible range for speech."""
    if abs(bass_db) < 0.1 and abs(treble_db) < 0.1:
        return samples.astype(np.float32)
    from scipy.signal import butter, sosfilt
    out = samples.astype(np.float64)
    nyq = sr / 2.0

    if abs(bass_db) >= 0.1:
        # 500 Hz captures voice body / warmth
        sos  = butter(2, 500.0 / nyq, btype='low', output='sos')
        lows = sosfilt(sos, out)
        out  = lows * (10 ** (bass_db / 20.0)) + (out - lows)

    if abs(treble_db) >= 0.1:
        # 2 kHz captures presence and sibilance
        sos   = butter(2, 2000.0 / nyq, btype='high', output='sos')
        highs = sosfilt(sos, out)
        out   = (out - highs) + highs * (10 ** (treble_db / 20.0))

    return out.astype(np.float32)

def apply_reverb(samples, sr, wet=0.0, room=0.5):
    """Synthetic reverb with early reflections + exponential tail."""
    if wet < 0.01:
        return samples.astype(np.float32)
    from scipy.signal import fftconvolve

    # Build impulse response: early reflections + decay tail
    tail_len = int(sr * (0.15 + 0.65 * room))
    ir = np.zeros(tail_len)

    # Early reflections
    reflections = [0.018, 0.030, 0.048, 0.072, 0.100, 0.135]
    for i, t in enumerate(reflections):
        idx = int(sr * t)
        if idx < tail_len:
            ir[idx] = (0.90 ** i) * 0.55

    # Exponential decay tail from ~50ms
    tail_start = int(sr * 0.05)
    if tail_start < tail_len:
        t_arr = np.arange(tail_len - tail_start)
        decay_tc = sr * (0.08 + 0.25 * room)
        ir[tail_start:] += np.exp(-t_arr / decay_tc) * 0.25 * room

    wet_sig = fftconvolve(samples.astype(np.float64), ir)[:len(samples)]
    out = samples.astype(np.float64) * (1.0 - wet) + wet_sig * wet
    return np.clip(out, -1.0, 1.0).astype(np.float32)

def time_stretch(samples, sr, rate):
    """Change tempo without changing pitch. rate > 1 = slower, rate < 1 = faster."""
    if abs(rate - 1.0) < 0.01:
        return samples.astype(np.float32)
    from scipy.signal import resample as fft_resample
    n = len(samples)
    n2 = max(1, int(round(n * rate)))
    # Step 1: FFT resample changes duration AND lowers/raises pitch by factor rate
    stretched = fft_resample(samples.astype(np.float64), n2)
    # Step 2: correct pitch back — pitch was divided by rate, so shift up by log2(rate) octaves
    semitones = 12.0 * math.log2(rate)
    return pitch_shift(stretched, sr, semitones).astype(np.float32)

def apply_vibrato(samples, sr, depth=0.0, rate_hz=5.5):
    """Sinusoidal pitch wobble via variable delay line. depth 0-1 → 0-4 ms swing."""
    if depth < 0.01:
        return samples.astype(np.float32)
    n = len(samples)
    max_delay = sr * (depth * 4.0) / 1000.0          # 0–4 ms in samples
    t = np.arange(n) / float(sr)
    delay = max_delay * np.sin(2.0 * np.pi * rate_hz * t)
    read_pos = np.clip(np.arange(n, dtype=np.float64) - delay, 0, n - 1)
    i0   = np.floor(read_pos).astype(np.intp)
    frac = read_pos - i0
    i1   = np.minimum(i0 + 1, n - 1)
    s    = samples.astype(np.float64)
    return (s[i0] * (1.0 - frac) + s[i1] * frac).astype(np.float32)

def apply_warmth(samples, amount=0.0):
    """Soft-saturation (tanh waveshaping) for analogue warmth."""
    if amount < 0.01:
        return samples.astype(np.float32)
    drive = 1.0 + amount * 5.0
    warmed = np.tanh(samples.astype(np.float64) * drive) / math.tanh(drive)
    return warmed.astype(np.float32)

def normalise(samples, target_peak=0.92):
    peak = np.max(np.abs(samples))
    if peak > 1e-6:
        samples = samples * (target_peak / peak)
    return samples.astype(np.float32)

def apply_effects(samples, sr, pitch=0.0, tempo=1.0, bass=0.0, treble=0.0,
                  reverb=0.0, room=0.5, warmth=0.0, vibrato=0.0):
    print(f"  FX: pitch={pitch:+.1f}st  tempo={tempo:.2f}x  bass={bass:+.0f}dB  "
          f"treble={treble:+.0f}dB  warmth={warmth:.0%}  vibrato={vibrato:.0%}  "
          f"reverb={reverb:.2f}  room={room:.0%}")
    s = samples.astype(np.float32)
    if abs(pitch) > 0.05:
        try:    s = pitch_shift(s, sr, pitch)
        except Exception as e: print(f"  pitch error: {e}")
    if abs(tempo - 1.0) > 0.01:
        try:    s = time_stretch(s, sr, tempo)
        except Exception as e: print(f"  tempo error: {e}")
    if abs(bass) > 0.1 or abs(treble) > 0.1:
        try:    s = apply_eq(s, sr, bass, treble)
        except Exception as e: print(f"  eq error: {e}")
    if warmth > 0.01:
        try:    s = apply_warmth(s, warmth)
        except Exception as e: print(f"  warmth error: {e}")
    if vibrato > 0.01:
        try:    s = apply_vibrato(s, sr, vibrato)
        except Exception as e: print(f"  vibrato error: {e}")
    if reverb > 0.01:
        try:    s = apply_reverb(s, sr, reverb, room)
        except Exception as e: print(f"  reverb error: {e}")
    return normalise(s)

# ─── Routes ─────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_file(os.path.join(SCRIPT_DIR, 'ui.html'))

@app.route('/generate', methods=['POST'])
def generate():
    data  = request.get_json(force=True)
    text  = (data.get('text') or '').strip()
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    voice_a = data.get('voice_a', 'bf_emma')
    voice_b = data.get('voice_b', 'bf_alice')
    blend   = max(0.0, min(1.0, float(data.get('blend',   0.0))))
    speed   = max(0.5, min(2.0, float(data.get('speed',   0.95))))
    pitch   =          float(data.get('pitch',   0.0))
    bass    = max(-12, min(12, float(data.get('bass',    0.0))))
    treble  = max(-12, min(12, float(data.get('treble',  0.0))))
    reverb  = max(0.0, min(0.8, float(data.get('reverb', 0.0))))
    room    = max(0.1, min(1.0, float(data.get('room',   0.5))))
    warmth  = max(0.0, min(1.0, float(data.get('warmth', 0.0))))
    tempo   = max(0.4, min(2.0, float(data.get('tempo',  1.0))))
    vibrato = max(0.0, min(1.0, float(data.get('vibrato', 0.0))))

    # GPU voice wins when voice_a is GPU, or voice_b is GPU and blend is fully at B.
    gpu_mode = voice_a in _GPU_IDS or (voice_b in _GPU_IDS and blend >= 0.99)

    all_samples = []

    if gpu_mode:
        if not _GPU_AVAILABLE:
            return jsonify({'error': 'GPU backend not available — run: pip install chatterbox-tts'}), 500
        sr = 24000
        _primary  = voice_a if voice_a in _GPU_IDS else voice_b
        _both_gpu = voice_a in _GPU_IDS and voice_b in _GPU_IDS
        print(f"  GPU voice: {_primary}", flush=True)
        gpu_error = None
        for chunk in split_text(text):
            try:
                result = _gpu_synth(chunk, _primary, speed=speed,
                                    voice_b=voice_b if _both_gpu else None,
                                    blend=blend if _both_gpu else 0.0)
                if isinstance(result, tuple):
                    s, sr = result   # chatterbox returns (audio, sample_rate)
                else:
                    s = result       # kokoro_gpu_backend returns ndarray only
                if s is not None and len(s) > 0:
                    all_samples.extend(s.tolist() if hasattr(s, 'tolist') else list(s))
            except Exception as e:
                print(f"  GPU chunk error: {e}")
                gpu_error = str(e)
                break
        if not all_samples:
            return jsonify({'error': gpu_error or 'GPU synthesis failed'}), 500
    else:
        k = get_kokoro()
        if 0.01 < blend < 0.99 and voice_b != voice_a:
            sa    = k.get_voice_style(voice_a)
            sb    = k.get_voice_style(voice_b)
            voice = np.add(sa * (1.0 - blend), sb * blend)
        elif blend >= 0.99:
            voice = voice_b
        else:
            voice = voice_a
        sr = 24000
        print(f"  Kokoro voice: {voice}  speed={speed}", flush=True)
        for chunk in split_text(text):
            try:
                samples, sr = k.create(chunk, voice=voice, speed=speed, lang='en-gb')
                # 20 ms fade-in to smooth any cold-start onset from the decoder.
                samples = samples.copy()
                fi = min(int(0.020 * sr), len(samples))
                samples[:fi] = samples[:fi] * np.linspace(0.0, 1.0, fi)
                all_samples.extend(samples.tolist() if hasattr(samples, 'tolist') else list(samples))
            except Exception as e:
                print(f"  Chunk error: {e}")

    if not all_samples:
        return jsonify({'error': 'Audio generation failed'}), 500

    audio = np.array(all_samples, dtype=np.float32)
    audio = apply_effects(audio, sr,
                          pitch=pitch, tempo=tempo, bass=bass, treble=treble,
                          reverb=reverb, room=room, warmth=warmth, vibrato=vibrato)

    buf = io.BytesIO()
    sf.write(buf, audio, sr, format='WAV')
    buf.seek(0)
    return send_file(buf, mimetype='audio/wav', download_name='scot-ai-tts.wav')

if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    print("\n  ScotAi TTS Studio v1.0")
    print("  -------------------------")
    print("  http://localhost:5000\n")
    get_kokoro()
    app.run(host='0.0.0.0', port=5000, debug=False)
