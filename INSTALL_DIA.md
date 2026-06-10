# Installing the Dia TTS Backend

The Dia backend uses two fine-tuned Dia 1.6B checkpoints for British voices.
It requires **CUDA** (NVIDIA GPU). Kokoro voices continue to work on CPU without
any of these steps.

---

## 1. RTX 5000 series (Blackwell) — torch nightly required

RTX 5090 / 5080 / 5070 / 5060 require torch 2.8 nightly built against CUDA 12.8.
See [nari-labs/dia issue #26](https://github.com/nari-labs/dia/issues/26) for details.

```bash
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
```

RTX 30/40 series (Ampere/Ada) can use the stable release instead:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## 2. Install Dia and its dependencies

```bash
pip install -r requirements-dia.txt
```

## 3. Download the checkpoints

The fine-tuned checkpoints are hosted on HuggingFace and must be downloaded
separately (they are not included in this repo).

```bash
python scripts/download_dia_models.py
```

This places the checkpoints at:

```
models/
  dia_british_female/ckpt_epoch4.pth
  dia_british_male/ckpt_epoch4.pth
```

If you already have the files, the script will skip the download.

## 4. Verify

```bash
kokoro-tts --help-voices          # British Female (Dia) and British Male (Dia) should appear
kokoro-tts input.txt --voice british_female
```

The first run will print:

> Loading Dia model (~4.4 GB VRAM). This may take 30–60s on first run.

Subsequent runs in the same process reuse the loaded model.

---

## Generation parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--dia-temperature` | 1.3 | Expressiveness / variation |
| `--dia-top-p` | 0.90 | Nucleus sampling |
| `--dia-top-k` | 45 | Top-k sampling |
| `--dia-guidance` | 3.0 | CFG adherence |
| `--dia-seed` | — | Integer seed for reproducible output |
| `--dia-no-compile` | — | Disable `torch.compile()` (~40% slower but faster startup) |

Example with reproducible output:

```bash
kokoro-tts input.txt output.wav --voice british_female --dia-seed 42 --dia-temperature 1.6
```
