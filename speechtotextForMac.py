import argparse
import platform
import shutil
import subprocess
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel

# Sakın alttaki 2 yorum satırını silme
# source .venv/bin/activate && python speechtotextForMac.py --backend openai-whisper --device mps --model large-v3 --use-ffmpeg --language tr 82.mp3
# source .venv/bin/activate && python speechtotextForMac.py --backend openai-whisper --device cpu --model large-v3 --use-ffmpeg --language tr 82.mp3

def load_audio_with_ffmpeg(path: str | Path, sample_rate: int = 16000) -> np.ndarray:
    """Decode an audio file to mono float32 PCM using ffmpeg.

    This avoids relying on PyAV (the `av` package). It requires `ffmpeg` to be installed
    and available on PATH.
    """

    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg was not found on PATH. Install it (macOS: `brew install ffmpeg`) or add it to PATH."
        )

    # Output raw 16-bit PCM mono at the desired sample rate.
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-f",
        "s16le",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "pipe:1",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(
            "ffmpeg failed to decode audio.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stderr: {proc.stderr.decode(errors='replace')}"
        )

    audio_i16 = np.frombuffer(proc.stdout, dtype=np.int16)
    if audio_i16.size == 0:
        raise RuntimeError("ffmpeg returned empty audio.")

    # Normalize to [-1, 1]
    return audio_i16.astype(np.float32) / 32768.0


def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm (easier to read than raw seconds)."""

    if seconds < 0:
        seconds = 0.0

    total_ms = int(round(seconds * 1000.0))
    ms = total_ms % 1000
    total_s = total_ms // 1000

    s = total_s % 60
    total_m = total_s // 60

    m = total_m % 60
    h = total_m // 60

    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Transcribe audio on macOS (Apple Silicon) with GPU when possible. "
            "Prefers OpenAI Whisper on MPS (Metal) if installed; otherwise falls back to faster-whisper CPU."
        )
    )
    parser.add_argument("audio", help="Path to an audio file (mp3/wav/m4a/etc)")
    parser.add_argument("--model", default="large-v3", help="Model size/name (default: large-v3)")
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "openai-whisper", "faster-whisper"],
        help="Transcription backend (default: auto)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "mps"],
        help="Compute device for openai-whisper (default: auto -> mps if available)",
    )
    parser.add_argument(
        "--compute-type",
        default="int8",
        help="Compute type for faster-whisper CPU (default: int8)",
    )
    parser.add_argument("--beam-size", type=int, default=5, help="Beam size")
    parser.add_argument(
        "--language",
        default=None,
        help="Force language code (e.g. en, tr). If omitted, auto-detect.",
    )
    parser.add_argument(
        "--use-ffmpeg",
        action="store_true",
        help="Decode audio via ffmpeg and pass raw PCM to the model (avoids PyAV).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help=(
            "Output .txt path to save the transcript. "
            "Default: <audio>.transcript.txt next to the audio file."
        ),
    )
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    def mps_available() -> bool:
        try:
            import torch

            return bool(torch.backends.mps.is_available()) and bool(torch.backends.mps.is_built())
        except Exception:
            return False

    def resolve_device() -> str:
        if args.device != "auto":
            return args.device
        return "mps" if (platform.system() == "Darwin" and mps_available()) else "cpu"

    def run_openai_whisper() -> tuple[str | None, list[dict]]:
        try:
            import whisper
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "openai-whisper is not installed. Install GPU-enabled deps on macOS with:\n"
                "  pip install -U openai-whisper torch\n"
                "Also ensure ffmpeg is installed: `brew install ffmpeg`."
            ) from e

        device = resolve_device()
        if device == "mps" and not mps_available():
            print("MPS (Metal) is not available; falling back to CPU.")
            device = "cpu"

        model = whisper.load_model(args.model, device=device)

        # Whisper uses fp16 on GPU-like devices.
        fp16 = device != "cpu"
        result = model.transcribe(
            str(audio_path),
            language=args.language,
            fp16=fp16,
            beam_size=args.beam_size,
            verbose=False,
        )

        language = result.get("language")
        segments = result.get("segments") or []
        return language, segments

    def run_faster_whisper_cpu() -> tuple[str | None, float | None, list[object]]:
        # Note: pip ctranslate2 on macOS is typically CPU-only (no Metal backend).
        model = WhisperModel(args.model, device="cpu", compute_type=args.compute_type)
        if args.use_ffmpeg:
            audio = load_audio_with_ffmpeg(audio_path)
            segments, info = model.transcribe(audio, beam_size=args.beam_size, language=args.language)
        else:
            segments, info = model.transcribe(
                str(audio_path),
                beam_size=args.beam_size,
                language=args.language,
            )
        return info.language, getattr(info, "language_probability", None), list(segments)

    backend = args.backend
    if backend == "auto":
        backend = "openai-whisper" if mps_available() else "faster-whisper"

    lines: list[str] = []
    if backend == "openai-whisper":
        language, segments = run_openai_whisper()
        header = f"Detected language '{language}'" if language else "Detected language: (unknown)"
        print(header)
        lines.append(header)
        for segment in segments:
            start = format_timestamp(float(segment.get("start", 0.0)))
            end = format_timestamp(float(segment.get("end", 0.0)))
            text = (segment.get("text") or "").strip()
            line = f"[{start} -> {end}] {text}"
            print(line)
            lines.append(line)
    elif backend == "faster-whisper":
        language, language_probability, segments = run_faster_whisper_cpu()
        if language_probability is None:
            header = f"Detected language '{language}'"
        else:
            header = "Detected language '%s' with probability %f" % (language, language_probability)
        print(header)
        lines.append(header)
        for segment in segments:
            start = format_timestamp(segment.start)
            end = format_timestamp(segment.end)
            text = (segment.text or "").strip()
            line = f"[{start} -> {end}] {text}"
            print(line)
            lines.append(line)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    out_path = Path(args.out) if args.out else audio_path.with_suffix(audio_path.suffix + ".transcript.txt")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSaved transcript to: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())