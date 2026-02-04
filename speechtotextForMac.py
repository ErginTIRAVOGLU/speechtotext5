from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import queue
from datetime import datetime, timedelta
from pathlib import Path

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

# Sakın alttaki 2 yorum satırını silme
# source .venv/bin/activate && python speechtotextForMac.py --backend openai-whisper --device mps --model large-v3 --use-ffmpeg --language tr 82.mp3
# source .venv/bin/activate && python speechtotextForMac.py --backend openai-whisper --device cpu --model large-v3 --use-ffmpeg --language tr 82.mp3


def _is_frozen_executable() -> bool:
    # PyInstaller sets sys.frozen = True
    return bool(getattr(sys, "frozen", False))


def _maybe_add_ffmpeg_to_path() -> None:
    """Best-effort helper to make ffmpeg discoverable.

    If the user has placed an ffmpeg path in `FFMPEG_PATH.txt` (either a folder or
    a full path to ffmpeg), we prepend it to PATH for the current process.
    """

    try:
        base_dir = Path(sys.executable).resolve().parent if _is_frozen_executable() else Path(__file__).resolve().parent
        cfg = base_dir / "FFMPEG_PATH.txt"
        if not cfg.exists():
            return

        raw = cfg.read_text(encoding="utf-8", errors="ignore").strip().strip('"')
        if not raw:
            return

        p = Path(raw)
        if p.is_file():
            ffmpeg_dir = p.parent
        else:
            ffmpeg_dir = p

        if not ffmpeg_dir.exists() or not ffmpeg_dir.is_dir():
            return

        ffmpeg_name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
        if not (ffmpeg_dir / ffmpeg_name).exists():
            return

        current = os.environ.get("PATH", "")
        parts = current.split(os.pathsep) if current else []
        ffmpeg_dir_str = str(ffmpeg_dir)
        if ffmpeg_dir_str not in parts:
            os.environ["PATH"] = ffmpeg_dir_str + os.pathsep + current
    except Exception:
        return


def _configure_stdio_for_windows() -> None:
    """Avoid crashing on Windows consoles with legacy encodings.

    No-op on macOS, but keeping this makes the script portable.
    """

    try:
        if os.name != "nt":
            return
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.reconfigure(errors="replace")
            except Exception:
                pass
    except Exception:
        pass


def _gui_config_path() -> Path:
    # Prefer per-user config, not next to the script (which might be read-only in packaged apps).
    base = os.getenv("APPDATA")
    if not base:
        base = os.getenv("XDG_CONFIG_HOME")
    if not base:
        base = str(Path.home() / ".config")
    folder = Path(base) / "speechtotext5"
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except Exception:
        return Path(__file__).resolve().parent / ".speechtotext_gui_config.json"
    return folder / "gui_config.json"


def _should_launch_gui(argv: list[str]) -> bool:
    # Explicit flags always win.
    if any(a == "--nogui" for a in argv):
        return False
    if any(a == "--gui" for a in argv):
        return True

    # No args => GUI.
    if len(argv) == 0:
        return True

    # Any other args => CLI.
    return False

def load_audio_with_ffmpeg(path: str | Path, sample_rate: int = 16000) -> "np.ndarray":
    """Decode an audio file to mono float32 PCM using ffmpeg.

    This avoids relying on PyAV (the `av` package). It requires `ffmpeg` to be installed
    and available on PATH.
    """

    _maybe_add_ffmpeg_to_path()

    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg was not found on PATH. Install it (macOS: `brew install ffmpeg`) or add it to PATH. "
            "Tip: you can also put the ffmpeg folder path (or ffmpeg binary path) into FFMPEG_PATH.txt next to this script."
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

    import numpy as np

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Transcribe audio on macOS (Apple Silicon) with GPU when possible. "
            "Prefers OpenAI Whisper on MPS (Metal) if installed; otherwise falls back to faster-whisper CPU."
        )
    )
    parser.add_argument(
        "audio",
        nargs="?",
        help="Path to an audio file (mp3/wav/m4a/etc)",
    )
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
    parser.add_argument(
        "--no-timeline",
        dest="timeline",
        action="store_false",
        help="Disable [start -> end] timestamps in transcript lines (plain text only).",
    )
    parser.add_argument(
        "--save-both",
        action="store_true",
        help="Also save the other transcript format (timeline/plain) from the same run.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print extra debug logs (useful when startup seems to hang).",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Optional path to also write console logs to a UTF-8 file.",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch a simple GUI to pick files and options.",
    )
    parser.add_argument(
        "--nogui",
        action="store_true",
        help="Run in CLI mode (do not launch GUI).",
    )
    return parser


def gui_main(
    parser: argparse.ArgumentParser,
    initial_args: argparse.Namespace | None = None,
    explicit_argv: list[str] | None = None,
) -> int:
    try:
        import tkinter as tk
        from tkinter import ttk, filedialog, messagebox
    except Exception as e:
        raise RuntimeError(
            "Tkinter is not available in this Python environment. "
            "On macOS, install a Python build that includes Tk (e.g. python.org installer), "
            "or run the script from CLI without --gui."
        ) from e

    root = tk.Tk()
    root.title("SpeechToText")
    root.geometry("980x720")

    config_path = _gui_config_path()

    def load_cfg() -> dict:
        try:
            if not config_path.exists():
                return {}
            return json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def save_cfg(data: dict) -> None:
        try:
            config_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except Exception:
            # Best-effort only; GUI should still function.
            pass

    # ---- Layout: options (scrollable) + output (Transcript + Log) ----
    outer = ttk.Frame(root, padding=10)
    outer.pack(fill="both", expand=True)

    paned = ttk.PanedWindow(outer, orient="vertical")
    paned.pack(fill="both", expand=True)

    options_container = ttk.Frame(paned)
    log_container = ttk.Frame(paned)
    paned.add(options_container, weight=3)
    paned.add(log_container, weight=2)

    # Scrollable options area
    canvas = tk.Canvas(options_container, highlightthickness=0)
    vsb = ttk.Scrollbar(options_container, orient="vertical", command=canvas.yview)
    options_frame = ttk.Frame(canvas, padding=(0, 0, 6, 0))

    options_frame_id = canvas.create_window((0, 0), window=options_frame, anchor="nw")
    canvas.configure(yscrollcommand=vsb.set)

    canvas.pack(side="left", fill="both", expand=True)
    vsb.pack(side="right", fill="y")

    def _on_frame_configure(_event: object = None) -> None:
        canvas.configure(scrollregion=canvas.bbox("all"))

    def _on_canvas_configure(event: object) -> None:
        canvas.itemconfigure(options_frame_id, width=event.width)

    options_frame.bind("<Configure>", _on_frame_configure)
    canvas.bind("<Configure>", _on_canvas_configure)

    output_paned = ttk.PanedWindow(log_container, orient="vertical")
    output_paned.pack(fill="both", expand=True)

    transcript_frame = ttk.Frame(output_paned)
    log_frame = ttk.Frame(output_paned)
    output_paned.add(transcript_frame, weight=3)
    output_paned.add(log_frame, weight=1)

    ttk.Label(transcript_frame, text="Transcript").pack(anchor="w")
    ttk.Label(log_frame, text="Log").pack(anchor="w")

    transcript_text = tk.Text(transcript_frame, height=12, wrap="word")
    transcript_scroll = ttk.Scrollbar(transcript_frame, orient="vertical", command=transcript_text.yview)
    transcript_text.configure(yscrollcommand=transcript_scroll.set)
    transcript_text.pack(side="left", fill="both", expand=True)
    transcript_scroll.pack(side="right", fill="y")

    log_text = tk.Text(log_frame, height=8, wrap="word")
    log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=log_text.yview)
    log_text.configure(yscrollcommand=log_scroll.set)
    log_text.pack(side="left", fill="both", expand=True)
    log_scroll.pack(side="right", fill="y")

    def _append(widget: tk.Text, text: str) -> None:
        widget.insert("end", text)
        widget.see("end")

    def append_log(text: str) -> None:
        _append(log_text, text)

    def append_transcript(text: str) -> None:
        _append(transcript_text, text)

    def _looks_like_log_timestamp_prefix(s: str) -> bool:
        # Verbose log lines look like: [23:52:21] Message...
        try:
            return (
                len(s) >= 10
                and s[0] == "["
                and s[3] == ":"
                and s[6] == ":"
                and s[9] == "]"
                and s[1:3].isdigit()
                and s[4:6].isdigit()
                and s[7:9].isdigit()
            )
        except Exception:
            return False

    def is_transcript_line(line: str) -> bool:
        # Timeline-on transcript segment lines look like:
        # [00:00:00.000 -> 00:00:27.420] ...
        raw = (line or "")
        s = raw.strip("\r\n")
        s_l = s.lstrip()

        if not s_l:
            return False

        # Timestamped transcript segment line
        if s_l.startswith("[") and (" -> " in s_l) and ("]" in s_l):
            return True

        # Timestamped verbose log line
        if s_l.startswith("[") and _looks_like_log_timestamp_prefix(s_l):
            return False

        # Plain-text transcript mode: keep known headers/status lines in the log box.
        lower = s_l.lower()
        if lower.startswith("[gui]"):
            return False
        if s_l.startswith("Başlangıç Zamanı:"):
            return False
        if s_l.startswith("Bitiş Zamanı:") or s_l.startswith("Bitiş hesaplanıyor"):
            return False
        if s_l.startswith("Tahmini Bitiş Zamanı:"):
            return False
        if s_l.startswith("Detected language"):
            return False
        if lower.startswith("saved transcript to"):
            return False
        if lower.startswith("saved other format to"):
            return False
        if lower.startswith("usage:") or lower.startswith("transcribe audio"):
            return False
        if lower.startswith("mps"):
            return False

        return True

    # ---- Build controls from argparse actions ----
    vars_by_dest: dict[str, object] = {}
    widgets_by_dest: dict[str, object] = {}

    # We’ll show positional audio first, then the rest.
    actions = [a for a in parser._actions if a.dest != "help"]
    actions.sort(key=lambda a: (0 if not a.option_strings else 1, a.dest))

    row = 0
    model_presets = ["tiny", "base", "small", "medium", "large-v3"]

    for action in actions:
        if action.dest in ("gui", "nogui"):
            continue

        label_text = action.dest.replace("_", " ")
        if action.option_strings:
            label_text = f"{action.option_strings[-1]}  ({label_text})"

        ttk.Label(options_frame, text=label_text).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)

        if getattr(action, "choices", None):
            var = tk.StringVar(value=str(action.default) if action.default is not None else "")
            widget = ttk.OptionMenu(options_frame, var, var.get(), *[str(c) for c in action.choices])
            widget.grid(row=row, column=1, sticky="ew", pady=4)
        elif isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
            var = tk.BooleanVar(value=bool(action.default))
            widget = ttk.Checkbutton(options_frame, variable=var)
            widget.grid(row=row, column=1, sticky="w", pady=4)
        else:
            default_value = "" if action.default is None else str(action.default)
            var = tk.StringVar(value=default_value)
            if action.dest == "model":
                widget = ttk.Combobox(options_frame, textvariable=var, values=model_presets, state="normal")
            else:
                widget = ttk.Entry(options_frame, textvariable=var)
            widget.grid(row=row, column=1, sticky="ew", pady=4)

        vars_by_dest[action.dest] = var
        widgets_by_dest[action.dest] = widget

        # Add browse buttons for audio/out
        if action.dest == "audio":
            def _pick_audio() -> None:
                path = filedialog.askopenfilename(
                    title="Select audio file",
                    filetypes=[
                        ("Audio", "*.mp3 *.wav *.m4a *.aac *.flac *.ogg *.opus *.wma *.mp4 *.mkv *.mov"),
                        ("All files", "*.*"),
                    ],
                )
                if path:
                    vars_by_dest["audio"].set(path)

            ttk.Button(options_frame, text="Browse…", command=_pick_audio).grid(
                row=row, column=2, sticky="e", padx=(8, 0), pady=4
            )

        if action.dest == "out":
            def _pick_out() -> None:
                path = filedialog.asksaveasfilename(
                    title="Save transcript as",
                    defaultextension=".txt",
                    filetypes=[("Text", "*.txt"), ("All files", "*.*")],
                )
                if path:
                    vars_by_dest["out"].set(path)

            ttk.Button(options_frame, text="Save As…", command=_pick_out).grid(
                row=row, column=2, sticky="e", padx=(8, 0), pady=4
            )

        row += 1

    options_frame.columnconfigure(1, weight=1)

    # Restore previous GUI settings
    cfg = load_cfg()
    if isinstance(cfg, dict):
        geom = cfg.get("geometry")
        if isinstance(geom, str) and geom.strip():
            try:
                root.geometry(geom)
            except Exception:
                pass

        args_cfg = cfg.get("args")
        if isinstance(args_cfg, dict):
            for dest, value in args_cfg.items():
                var_obj = vars_by_dest.get(dest)
                if var_obj is None:
                    continue
                try:
                    if isinstance(var_obj, tk.BooleanVar):
                        var_obj.set(bool(value))
                    else:
                        var_obj.set("" if value is None else str(value))
                except Exception:
                    pass

    # Apply CLI-provided initial values on top, but ONLY for arguments that were
    # explicitly provided on the command line.
    if initial_args is not None:
        try:
            explicit_set = set(explicit_argv or [])

            # Positional audio can be provided when launching GUI via CLI.
            if getattr(initial_args, "audio", None):
                var_obj = vars_by_dest.get("audio")
                if var_obj is not None:
                    var_obj.set(str(getattr(initial_args, "audio")))

            # Only override option values when their flag is present in argv.
            for action in parser._actions:
                dest = getattr(action, "dest", None)
                if not dest or dest in ("help", "audio", "gui", "nogui"):
                    continue
                if not hasattr(initial_args, dest):
                    continue

                opt_strings = set(getattr(action, "option_strings", []) or [])
                if not opt_strings.intersection(explicit_set):
                    continue

                var_obj = vars_by_dest.get(dest)
                if var_obj is None:
                    continue

                value = getattr(initial_args, dest)
                if value is None:
                    continue
                try:
                    if isinstance(var_obj, tk.BooleanVar):
                        var_obj.set(bool(value))
                    else:
                        var_obj.set(str(value))
                except Exception:
                    pass
        except Exception:
            pass

    # ---- Run/Stop controls ----
    controls = ttk.Frame(outer, padding=(0, 8, 0, 0))
    controls.pack(fill="x")

    status_var = tk.StringVar(value="Idle")
    ttk.Label(controls, textvariable=status_var).pack(side="left")

    run_btn = ttk.Button(controls, text="Run")
    stop_btn = ttk.Button(controls, text="Stop", state="disabled")
    run_btn.pack(side="right")
    stop_btn.pack(side="right", padx=(0, 8))

    proc: subprocess.Popen[str] | None = None
    q: queue.Queue[str] = queue.Queue()

    def _set_controls_enabled(enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for w in widgets_by_dest.values():
            try:
                w.configure(state=state)
            except Exception:
                pass
        run_btn.configure(state=state)
        stop_btn.configure(state=("disabled" if enabled else "normal"))

    def _build_argv() -> list[str] | None:
        audio = str(vars_by_dest.get("audio").get()).strip() if "audio" in vars_by_dest else ""
        if not audio:
            messagebox.showerror("Missing input", "Please select an audio file.")
            return None

        exe = sys.executable
        if not exe or not Path(exe).exists():
            messagebox.showerror("Launch error", f"Cannot locate current executable:\n{exe}")
            return None

        if _is_frozen_executable():
            argv: list[str] = [exe, "--nogui", audio]
        else:
            argv = [exe, str(Path(__file__).resolve()), "--nogui", audio]

        for action in parser._actions:
            if action.dest in ("help", "audio", "gui", "nogui"):
                continue
            if not action.option_strings:
                continue

            opt = action.option_strings[-1]
            val_obj = vars_by_dest.get(action.dest)
            if val_obj is None:
                continue

            if isinstance(action, argparse._StoreTrueAction):
                if bool(val_obj.get()):
                    argv.append(opt)
                continue

            if isinstance(action, argparse._StoreFalseAction):
                # Include the flag (e.g. --no-timeline) when unchecked.
                if not bool(val_obj.get()):
                    argv.append(opt)
                continue

            raw = str(val_obj.get()).strip()
            if raw == "":
                continue

            argv.extend([opt, raw])

        return argv

    def _reader_thread(p: subprocess.Popen[str]) -> None:
        try:
            assert p.stdout is not None
            for line in p.stdout:
                q.put(line)
        except Exception as e:
            q.put(f"\n[GUI] Output read error: {e}\n")

    def _poll_queue() -> None:
        nonlocal proc
        try:
            while True:
                line = q.get_nowait()
                if is_transcript_line(line):
                    append_transcript(line)
                else:
                    append_log(line)
        except queue.Empty:
            pass

        if proc is not None:
            rc = proc.poll()
            if rc is not None:
                status_var.set(f"Finished (exit code {rc})")
                _set_controls_enabled(True)
                proc = None
        root.after(100, _poll_queue)

    def on_run() -> None:
        nonlocal proc
        argv = _build_argv()
        if argv is None:
            return

        # Persist current settings
        try:
            args_state: dict[str, object] = {}
            for dest, var_obj in vars_by_dest.items():
                try:
                    if isinstance(var_obj, tk.BooleanVar):
                        args_state[dest] = bool(var_obj.get())
                    else:
                        args_state[dest] = str(var_obj.get())
                except Exception:
                    pass
            save_cfg({"geometry": root.geometry(), "args": args_state})
        except Exception:
            pass

        transcript_text.delete("1.0", "end")
        log_text.delete("1.0", "end")
        append_log("[GUI] Running:\n  " + " ".join(argv) + "\n\n")
        status_var.set("Running…")
        _set_controls_enabled(False)

        proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        threading.Thread(target=_reader_thread, args=(proc,), daemon=True).start()

    def on_stop() -> None:
        nonlocal proc
        if proc is None:
            return
        status_var.set("Stopping…")
        try:
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()
        except Exception as e:
            append_log(f"\n[GUI] Failed to stop process: {e}\n")

    run_btn.configure(command=on_run)
    stop_btn.configure(command=on_stop)

    def on_close() -> None:
        # Save on exit too
        try:
            args_state: dict[str, object] = {}
            for dest, var_obj in vars_by_dest.items():
                try:
                    if isinstance(var_obj, tk.BooleanVar):
                        args_state[dest] = bool(var_obj.get())
                    else:
                        args_state[dest] = str(var_obj.get())
                except Exception:
                    pass
            save_cfg({"geometry": root.geometry(), "args": args_state})
        except Exception:
            pass
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)

    # Start polling log queue
    root.after(100, _poll_queue)

    # If script was called like `python speechtotextForMac.py --gui somefile.mp3`, prefill
    try:
        ns = parser.parse_args([])
        _ = ns
    except Exception:
        pass

    root.mainloop()
    return 0


def main() -> int:
    parser = build_parser()

    # Accept plain `nogui` (without dashes) as an alias for convenience.
    if len(sys.argv) >= 2 and sys.argv[1].lower() == "nogui":
        sys.argv[1] = "--nogui"

    argv_tail = sys.argv[1:]
    args = parser.parse_args(argv_tail)

    if _should_launch_gui(argv_tail):
        return gui_main(parser, initial_args=args, explicit_argv=argv_tail)

    return cli_main(args)


def cli_main(args: argparse.Namespace) -> int:
    _configure_stdio_for_windows()
    if not getattr(args, "audio", None):
        raise ValueError("audio is required")

    log_fp: Path | None = None
    if getattr(args, "log_file", None):
        try:
            log_fp = Path(args.log_file)
            log_fp.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            log_fp = None

    def log(msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        try:
            print(line, flush=True)
        except Exception:
            try:
                print(line.encode("utf-8", errors="replace").decode("utf-8", errors="replace"), flush=True)
            except Exception:
                pass
        if log_fp is not None:
            try:
                with log_fp.open("a", encoding="utf-8", errors="replace") as f:
                    f.write(line + "\n")
            except Exception:
                pass

    audio_path = Path(args.audio)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    def unique_output_path(p: Path) -> Path:
        """Return a non-existing path by appending a timestamp if needed."""

        try:
            if not p.exists():
                return p
        except Exception:
            return p

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        candidate = p.with_name(f"{p.stem}_{ts}{p.suffix}")
        if not candidate.exists():
            return candidate

        i = 2
        while True:
            candidate_i = p.with_name(f"{p.stem}_{ts}_{i}{p.suffix}")
            if not candidate_i.exists():
                return candidate_i
            i += 1

    def mps_available() -> bool:
        try:
            import torch

            return bool(torch.backends.mps.is_available()) and bool(torch.backends.mps.is_built())
        except Exception:
            return False

    def resolve_device() -> str:
        if getattr(args, "device", "auto") != "auto":
            return str(args.device)
        return "mps" if (platform.system() == "Darwin" and mps_available()) else "cpu"

    def probe_duration_seconds(path: Path) -> float | None:
        # Best-effort duration probe for ETA (works when ffprobe is available).
        try:
            _maybe_add_ffmpeg_to_path()
            if shutil.which("ffprobe") is None:
                if getattr(args, "verbose", False):
                    log("ffprobe not found; cannot estimate duration from file metadata.")
                return None
            cmd = [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if proc.returncode != 0:
                if getattr(args, "verbose", False):
                    log("ffprobe failed; cannot estimate duration.")
                return None
            value = (proc.stdout or "").strip()
            if not value:
                return None
            dur = float(value)
            return dur if dur > 0 else None
        except Exception:
            return None

    start_dt = datetime.now()
    start_line = f"Başlangıç Zamanı: {start_dt.strftime('%Y-%m-%d %H:%M:%S')}"
    print(start_line, flush=True)

    backend = getattr(args, "backend", "auto")
    if backend == "auto":
        backend = "openai-whisper" if mps_available() else "faster-whisper"

    if getattr(args, "verbose", False):
        log(f"Input: {audio_path}")
        log(f"Backend: {backend}")
        if backend == "openai-whisper":
            log(f"Device: {resolve_device()} | Model: {args.model} | Beam: {args.beam_size}")
        else:
            log(f"Device: cpu | Model: {args.model} | Compute: {args.compute_type} | Beam: {args.beam_size}")
        if getattr(args, "use_ffmpeg", False):
            log("Audio decode: ffmpeg (raw PCM)")
        else:
            log("Audio decode: direct file (backend may use ffprobe for duration)")

    audio_duration_s: float | None = None
    transcription_start_dt: datetime | None = None

    # Collect normalized segments as tuples: (start_s, end_s, text)
    normalized_segments: list[tuple[float, float, str]] = []
    header: str = "Detected language: (unknown)"

    if backend == "openai-whisper":
        try:
            import whisper
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "openai-whisper is not installed. Install deps on macOS with:\n"
                "  pip install -U openai-whisper torch\n"
                "Also ensure ffmpeg is installed: `brew install ffmpeg`."
            ) from e

        device = resolve_device()
        if device == "mps" and not mps_available():
            print("MPS (Metal) is not available; falling back to CPU.")
            device = "cpu"

        if getattr(args, "verbose", False):
            log("Loading model (this can take a while on first run: download + init)…")
        model = whisper.load_model(args.model, device=device)
        if getattr(args, "verbose", False):
            log("Model loaded.")

        if getattr(args, "use_ffmpeg", False):
            if getattr(args, "verbose", False):
                log("Decoding audio with ffmpeg…")
            audio = load_audio_with_ffmpeg(audio_path)
            try:
                audio_duration_s = float(len(audio)) / 16000.0
            except Exception:
                audio_duration_s = None
            if getattr(args, "verbose", False) and audio_duration_s is not None:
                log(f"Audio duration (from decoded PCM): {audio_duration_s:.1f}s")
            if getattr(args, "verbose", False):
                log("Starting transcription…")
            transcription_start_dt = datetime.now()
            result = model.transcribe(
                audio,
                language=args.language,
                fp16=(device != "cpu"),
                beam_size=args.beam_size,
                verbose=False,
            )
        else:
            if getattr(args, "verbose", False):
                log("Probing duration with ffprobe (for ETA)…")
            audio_duration_s = probe_duration_seconds(audio_path)
            if getattr(args, "verbose", False) and audio_duration_s is not None:
                log(f"Audio duration (from ffprobe): {audio_duration_s:.1f}s")
            if getattr(args, "verbose", False):
                log("Starting transcription…")
            transcription_start_dt = datetime.now()
            result = model.transcribe(
                str(audio_path),
                language=args.language,
                fp16=(device != "cpu"),
                beam_size=args.beam_size,
                verbose=False,
            )

        language = result.get("language") if isinstance(result, dict) else None
        header = f"Detected language '{language}'" if language else "Detected language: (unknown)"
        segments = (result.get("segments") if isinstance(result, dict) else None) or []
        for seg in segments:
            try:
                s = float(seg.get("start", 0.0))
                e = float(seg.get("end", 0.0))
                t = str(seg.get("text") or "")
            except Exception:
                continue
            normalized_segments.append((s, e, t))
    elif backend == "faster-whisper":
        from faster_whisper import WhisperModel

        if getattr(args, "verbose", False):
            log("Loading model (this can take a while on first run: download + init)…")
        model = WhisperModel(args.model, device="cpu", compute_type=args.compute_type)
        if getattr(args, "verbose", False):
            log("Model loaded.")

        if getattr(args, "use_ffmpeg", False):
            if getattr(args, "verbose", False):
                log("Decoding audio with ffmpeg…")
            audio = load_audio_with_ffmpeg(audio_path)
            try:
                audio_duration_s = float(len(audio)) / 16000.0
            except Exception:
                audio_duration_s = None
            if getattr(args, "verbose", False) and audio_duration_s is not None:
                log(f"Audio duration (from decoded PCM): {audio_duration_s:.1f}s")
            if getattr(args, "verbose", False):
                log("Starting transcription…")
            transcription_start_dt = datetime.now()
            segments_iter, info = model.transcribe(audio, beam_size=args.beam_size, language=args.language)
        else:
            if getattr(args, "verbose", False):
                log("Probing duration with ffprobe (for ETA)…")
            audio_duration_s = probe_duration_seconds(audio_path)
            if getattr(args, "verbose", False) and audio_duration_s is not None:
                log(f"Audio duration (from ffprobe): {audio_duration_s:.1f}s")
            if getattr(args, "verbose", False):
                log("Starting transcription…")
            transcription_start_dt = datetime.now()
            segments_iter, info = model.transcribe(str(audio_path), beam_size=args.beam_size, language=args.language)

        # Some faster-whisper versions expose duration on info; use it if we don't have one yet.
        if audio_duration_s is None:
            try:
                dur = getattr(info, "duration", None)
                if dur is not None and float(dur) > 0:
                    audio_duration_s = float(dur)
            except Exception:
                pass

        language = getattr(info, "language", None)
        language_probability = getattr(info, "language_probability", None)
        if language_probability is None:
            header = f"Detected language '{language}'"
        else:
            header = "Detected language '%s' with probability %f" % (language, language_probability)

        for seg in list(segments_iter):
            try:
                normalized_segments.append((float(seg.start), float(seg.end), str(seg.text or "")))
            except Exception:
                continue
    else:
        raise ValueError(f"Unknown backend: {backend}")

    print(header, flush=True)
    print("Bitiş hesaplanıyor...", flush=True)

    # Build two transcript variants from the SAME segments to avoid content drift.
    timeline_lines: list[str] = [start_line, header]
    plain_lines: list[str] = [start_line, header]

    first_segment_logged = False
    eta_printed = False
    buffered_segment_lines: list[str] = []

    if audio_duration_s is None:
        eta_line = "Tahmini Bitiş Zamanı: (bilinmiyor)"
        print(eta_line, flush=True)
        timeline_lines.insert(1, eta_line)
        plain_lines.insert(1, eta_line)
        eta_printed = True

    for (seg_start_s, seg_end_s, seg_text) in normalized_segments:
        start = format_timestamp(seg_start_s)
        end = format_timestamp(seg_end_s)
        text = (seg_text or "").strip()
        text_one_line = " ".join(text.split()) if text else ""

        timeline_line = f"[{start} -> {end}] {text_one_line}"
        plain_line = text_one_line
        timeline_lines.append(timeline_line)
        plain_lines.append(plain_line)

        line = timeline_line if getattr(args, "timeline", True) else plain_line

        if not eta_printed:
            buffered_segment_lines.append(line)
        else:
            if getattr(args, "verbose", False) and not first_segment_logged:
                log("First segments are coming in (startup phase is over).")
                first_segment_logged = True
            print(line, flush=True)

        if (not eta_printed) and audio_duration_s is not None:
            try:
                processed_audio_s = float(seg_end_s)
                if processed_audio_s > 0:
                    threshold = min(10.0, audio_duration_s * 0.05)
                    if processed_audio_s >= threshold:
                        eta_base_dt = transcription_start_dt or start_dt
                        elapsed_wall_s = (datetime.now() - eta_base_dt).total_seconds()
                        if elapsed_wall_s > 0.5:
                            total_wall_s_est = elapsed_wall_s * (audio_duration_s / processed_audio_s)
                            est_end_dt = eta_base_dt + timedelta(seconds=total_wall_s_est)
                            eta_line = f"Tahmini Bitiş Zamanı: {est_end_dt.strftime('%Y-%m-%d %H:%M:%S')}"
                            print(eta_line, flush=True)
                            timeline_lines.insert(1, eta_line)
                            plain_lines.insert(1, eta_line)
                            eta_printed = True

                            if getattr(args, "verbose", False) and not first_segment_logged:
                                log("First segments are coming in (startup phase is over).")
                                first_segment_logged = True

                            for buffered in buffered_segment_lines:
                                print(buffered, flush=True)
                            buffered_segment_lines.clear()
            except Exception:
                pass

    if not eta_printed and buffered_segment_lines:
        eta_line = "Tahmini Bitiş Zamanı: (hesaplanamadı)"
        print(eta_line, flush=True)
        timeline_lines.insert(1, eta_line)
        plain_lines.insert(1, eta_line)
        if getattr(args, "verbose", False) and not first_segment_logged:
            log("First segments are coming in (startup phase is over).")
            first_segment_logged = True
        for buffered in buffered_segment_lines:
            print(buffered, flush=True)
        buffered_segment_lines.clear()

    end_dt = datetime.now()
    end_line = f"Bitiş Zamanı: {end_dt.strftime('%Y-%m-%d %H:%M:%S')}"
    timeline_lines.append(end_line)
    plain_lines.append(end_line)

    out_path = Path(args.out) if getattr(args, "out", None) else audio_path.with_suffix(audio_path.suffix + ".transcript.txt")
    out_path = unique_output_path(out_path)

    primary_lines = timeline_lines if getattr(args, "timeline", True) else plain_lines
    out_path.write_text("\n".join(primary_lines) + "\n", encoding="utf-8")
    print(f"\nSaved transcript to: {out_path}", flush=True)

    if bool(getattr(args, "save_both", False)):
        other_lines = plain_lines if getattr(args, "timeline", True) else timeline_lines
        other_suffix = ".plain" if getattr(args, "timeline", True) else ".timeline"
        other_path = out_path.with_name(out_path.stem + other_suffix + out_path.suffix)
        other_path = unique_output_path(other_path)
        try:
            other_path.write_text("\n".join(other_lines) + "\n", encoding="utf-8")
            print(f"Saved other format to: {other_path}", flush=True)
        except Exception:
            pass

    print(end_line, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())