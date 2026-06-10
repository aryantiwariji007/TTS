#!/usr/bin/env python3
"""Download Dia TTS fine-tuned checkpoints from HuggingFace."""

from pathlib import Path
import sys

MODELS_DIR = Path(__file__).parent.parent / "models"

CHECKPOINTS = [
    {
        "repo_id": "vaidaryan13/dia_tts_british_female_4",
        "local_dir": MODELS_DIR / "dia_british_female",
        "sentinel": "ckpt_epoch4.pth",
    },
    {
        "repo_id": "vaidaryan13/dia_tts_british_male_4",
        "local_dir": MODELS_DIR / "dia_british_male",
        "sentinel": "ckpt_epoch4.pth",
    },
]


def main():
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Error: huggingface_hub is not installed.")
        print("Install Dia dependencies first: pip install -r requirements-dia.txt")
        sys.exit(1)

    all_ready = True
    for spec in CHECKPOINTS:
        sentinel = spec["local_dir"] / spec["sentinel"]
        if sentinel.exists():
            print(f"Already present, skipping: {spec['local_dir']}")
            continue

        all_ready = False
        print(f"Downloading {spec['repo_id']} -> {spec['local_dir']} ...")
        spec["local_dir"].mkdir(parents=True, exist_ok=True)
        try:
            snapshot_download(
                repo_id=spec["repo_id"],
                local_dir=str(spec["local_dir"]),
            )
            print(f"Done: {spec['local_dir']}")
        except Exception as e:
            print(f"Error downloading {spec['repo_id']}: {e}")
            sys.exit(1)

    if all_ready:
        print("All Dia checkpoints already present.")
    else:
        print("\nAll Dia checkpoints downloaded successfully.")


if __name__ == "__main__":
    main()
