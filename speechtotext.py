import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel

# Sakın alttaki 2 yorum satırını silme
# .\.venv\Scripts\python.exe speechtotext.py --device cuda --model large-v3 --use-ffmpeg --language tr 8.mp3
# .\.venv\Scripts\python.exe speechtotext.py --device cpu --compute-type int8 --model large-v3 --use-ffmpeg --language tr 8.mp3

def load_audio_with_ffmpeg(path: str | Path, sample_rate: int = 16000) -> np.ndarray:
    """Decode an audio file to mono float32 PCM using ffmpeg.

    This avoids relying on PyAV (the `av` package). It requires `ffmpeg` to be installed
    and available on PATH.
    """

    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg was not found on PATH. Install it (e.g., via winget/choco) or add it to PATH."
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
    parser = argparse.ArgumentParser(description="Transcribe audio with faster-whisper")
    parser.add_argument("audio", help="Path to an audio file (mp3/wav/m4a/etc)")
    parser.add_argument("--model", default="large-v3", help="Model size/name (default: large-v3)")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Compute device")
    parser.add_argument(
        "--compute-type",
        default="float16",
        help="Compute type (e.g., float16, int8_float16 for CUDA; int8 for CPU)",
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

    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)

    if args.use_ffmpeg:
        audio = load_audio_with_ffmpeg(audio_path)
        segments, info = model.transcribe(audio, beam_size=args.beam_size, language=args.language)
    else:
        segments, info = model.transcribe(str(audio_path), beam_size=args.beam_size, language=args.language)

    header = "Detected language '%s' with probability %f" % (
        info.language,
        info.language_probability,
    )
    print(header)

    lines: list[str] = [header]
    for segment in segments:
        start = format_timestamp(segment.start)
        end = format_timestamp(segment.end)
        # Keep text intact; strip only outer whitespace to avoid weird leading spaces.
        text = (segment.text or "").strip()
        line = f"[{start} -> {end}] {text}"
        print(line)
        lines.append(line)

    out_path = Path(args.out) if args.out else audio_path.with_suffix(audio_path.suffix + ".transcript.txt")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSaved transcript to: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())