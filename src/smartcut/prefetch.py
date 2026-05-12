"""Pre-download a faster-whisper model so the first transcribe call is instant.

Usage:
    python -m smartcut.prefetch                     # downloads default (large-v3)
    python -m smartcut.prefetch base                # downloads 'base'
    python -m smartcut.prefetch large-v3 cuda       # downloads for CUDA
"""

import sys

VALID_MODELS = {
    "tiny", "tiny.en",
    "base", "base.en",
    "small", "small.en",
    "medium", "medium.en",
    "large-v1", "large-v2", "large-v3", "large",
    "large-v3-turbo", "turbo",
    "distil-large-v2", "distil-large-v3",
    "distil-medium.en", "distil-small.en",
}


def main(argv: list[str]) -> int:
    model_size = argv[1] if len(argv) > 1 else "large-v3"
    device = argv[2] if len(argv) > 2 else "cpu"

    if model_size not in VALID_MODELS and "/" not in model_size:
        print(f"Error: '{model_size}' is not a valid model size.", file=sys.stderr)
        print(f"Choose from: {', '.join(sorted(VALID_MODELS))}", file=sys.stderr)
        return 2

    try:
        from faster_whisper import WhisperModel  # noqa: F401
    except ImportError:
        print(
            "Error: faster-whisper is not installed.\n"
            "Install it first:  pip install -e '.[local]'",
            file=sys.stderr,
        )
        return 1

    try:
        from smartcut.core.model_download import ensure_model_downloaded
        path = ensure_model_downloaded(model_size, progress=True)
    except Exception as e:
        print(f"\nDownload failed: {e}", file=sys.stderr)
        return 1

    compute_type = "int8" if device == "cpu" else "float16"
    print(
        f"\nWarming up runtime ({device}/{compute_type})...",
        file=sys.stderr,
        flush=True,
    )
    try:
        from faster_whisper import WhisperModel
        WhisperModel(path, device=device, compute_type=compute_type)
    except Exception as e:
        print(
            f"\nModel downloaded to {path} but runtime init failed: {e}\n"
            f"(The cache is fine — this often means the device/compute_type isn't supported.)",
            file=sys.stderr,
        )
        return 1

    print(f"\nDone. Model '{model_size}' is ready at:\n  {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
