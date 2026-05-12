"""Pre-download a faster-whisper model so the first transcribe call is instant.

Usage:
    python -m smartcut.prefetch                     # downloads default (large-v3)
    python -m smartcut.prefetch base                # downloads 'base'
    python -m smartcut.prefetch large-v3 cuda       # downloads for CUDA
"""

import sys

VALID_MODELS = {"tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3"}
APPROX_SIZE = {
    "tiny": "75 MB",
    "base": "145 MB",
    "small": "485 MB",
    "medium": "1.5 GB",
    "large-v1": "3 GB",
    "large-v2": "3 GB",
    "large-v3": "3 GB",
}


def main(argv: list[str]) -> int:
    model_size = argv[1] if len(argv) > 1 else "large-v3"
    device = argv[2] if len(argv) > 2 else "cpu"

    if model_size not in VALID_MODELS:
        print(f"Error: '{model_size}' is not a valid model size.", file=sys.stderr)
        print(f"Choose from: {', '.join(sorted(VALID_MODELS))}", file=sys.stderr)
        return 2

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print(
            "Error: faster-whisper is not installed.\n"
            "Install it first:  pip install -e '.[local]'",
            file=sys.stderr,
        )
        return 1

    compute_type = "int8" if device == "cpu" else "float16"
    size = APPROX_SIZE.get(model_size, "?")

    print(f"Pre-downloading Whisper model '{model_size}' ({size}, {device}/{compute_type})...")
    print("Cache: ~/.cache/huggingface/hub/")
    print("Subsequent transcribe_project calls will read from cache (no network).\n")

    try:
        WhisperModel(model_size, device=device, compute_type=compute_type)
    except Exception as e:
        print(f"\nDownload failed: {e}", file=sys.stderr)
        return 1

    print(f"\nDone. Model '{model_size}' is ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
