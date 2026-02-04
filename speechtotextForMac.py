from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import threading
import queue
from pathlib import Path

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

# Sakın alttaki 2 yorum satırını silme
# source .venv/bin/activate && python speechtotextForMac.py --backend openai-whisper --device mps --model large-v3 --use-ffmpeg --language tr 82.mp3
# source .venv/bin/activate && python speechtotextForMac.py --backend openai-whisper --device cpu --model large-v3 --use-ffmpeg --language tr 82.mp3

def load_audio_with_ffmpeg(path: str | Path, sample_rate: int = 16000) -> "np.ndarray":
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


def gui_main(parser: argparse.ArgumentParser, initial_args: argparse.Namespace | None = None) -> int:
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
    root.title("SpeechToText (macOS)")
    root.geometry("980x720")

    config_path = Path(__file__).resolve().parent / ".speechtotext_gui_config.json"

    def load_gui_config() -> dict:
        try:
            if not config_path.exists():
                return {}
            return json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def save_gui_config(data: dict) -> None:
        try:
            config_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except Exception:
            # Best-effort only; GUI should still function.
            pass

    def default_python_executable() -> str:
        # Prefer workspace venv if present.
        script_dir = Path(__file__).resolve().parent
        venv_py = script_dir / ".venv" / "bin" / "python"
        if venv_py.exists():
            return str(venv_py)
        return sys.executable

    # ---- Layout: options (scrollable) + log output ----
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

    # Log area
    log_text = tk.Text(log_container, height=12, wrap="word")
    log_scroll = ttk.Scrollbar(log_container, orient="vertical", command=log_text.yview)
    log_text.configure(yscrollcommand=log_scroll.set)
    log_text.pack(side="left", fill="both", expand=True)
    log_scroll.pack(side="right", fill="y")

    def append_log(text: str) -> None:
        log_text.insert("end", text)
        log_text.see("end")

    # ---- Build controls from argparse actions ----
    vars_by_dest: dict[str, object] = {}
    widgets_by_dest: dict[str, object] = {}

    # GUI-only: choose interpreter (so subprocess can run inside venv)
    ttk.Label(options_frame, text="Python interpreter").grid(
        row=0, column=0, sticky="w", padx=(0, 8), pady=4
    )
    python_var = tk.StringVar(value=default_python_executable())
    python_entry = ttk.Entry(options_frame, textvariable=python_var)
    python_entry.grid(row=0, column=1, sticky="ew", pady=4)

    def _pick_python() -> None:
        path = filedialog.askopenfilename(title="Select Python", filetypes=[("Python", "python"), ("All files", "*.*")])
        if path:
            python_var.set(path)

    ttk.Button(options_frame, text="Browse…", command=_pick_python).grid(
        row=0, column=2, sticky="e", padx=(8, 0), pady=4
    )

    widgets_by_dest["__python__"] = python_entry

    # We’ll show positional audio first, then the rest.
    actions = [a for a in parser._actions if a.dest != "help"]
    actions.sort(key=lambda a: (0 if not a.option_strings else 1, a.dest))

    row = 1
    model_presets = ["tiny", "base", "small", "medium", "large-v3"]

    for action in actions:
        if action.dest == "gui":
            continue

        label_text = action.dest.replace("_", " ")
        if action.option_strings:
            label_text = f"{action.option_strings[-1]}  ({label_text})"

        ttk.Label(options_frame, text=label_text).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)

        if getattr(action, "choices", None):
            var = tk.StringVar(value=str(action.default) if action.default is not None else "")
            widget = ttk.OptionMenu(options_frame, var, var.get(), *[str(c) for c in action.choices])
            widget.grid(row=row, column=1, sticky="ew", pady=4)
        elif isinstance(action, argparse._StoreTrueAction):
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
                        ("Audio", "*.mp3 *.wav *.m4a *.aac *.flac *.ogg *.opus *.wma"),
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

    # Restore previous GUI settings (best-effort)
    cfg = load_gui_config()
    if isinstance(cfg, dict):
        geom = cfg.get("geometry")
        if isinstance(geom, str) and geom.strip():
            try:
                root.geometry(geom)
            except Exception:
                pass

        py_cfg = cfg.get("python")
        if isinstance(py_cfg, str) and py_cfg.strip():
            python_var.set(py_cfg)

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

    # Apply CLI-provided initial values on top (so `python ... file.mp3` prefills audio)
    if initial_args is not None:
        try:
            # positional
            if getattr(initial_args, "audio", None):
                var_obj = vars_by_dest.get("audio")
                if var_obj is not None:
                    var_obj.set(str(getattr(initial_args, "audio")))

            for dest, var_obj in vars_by_dest.items():
                if not hasattr(initial_args, dest):
                    continue
                if dest in ("gui", "nogui"):
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

        py = python_var.get().strip() or sys.executable
        if not Path(py).exists():
            messagebox.showerror("Invalid Python", f"Python executable not found:\n{py}")
            return None

        argv: list[str] = [py, str(Path(__file__).resolve()), "--nogui"]
        argv.append(audio)

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

            raw = str(val_obj.get()).strip()
            if raw == "":
                # If default is None, omit. Otherwise, keep empty as omission.
                if action.default is None:
                    continue
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
            save_gui_config(
                {
                    "geometry": root.geometry(),
                    "python": python_var.get(),
                    "args": args_state,
                }
            )
        except Exception:
            pass

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
            save_gui_config(
                {
                    "geometry": root.geometry(),
                    "python": python_var.get(),
                    "args": args_state,
                }
            )
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

    args = parser.parse_args()

    # Default behavior is GUI unless user explicitly requests CLI.
    # (Arg --gui is kept for backwards compatibility, but is effectively the default.)
    if not args.nogui:
        return gui_main(parser, initial_args=args)

    if not args.audio:
        parser.error("audio is required (or omit --nogui to use GUI)")

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
        from faster_whisper import WhisperModel

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