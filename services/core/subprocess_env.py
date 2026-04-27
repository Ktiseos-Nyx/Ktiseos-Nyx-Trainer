"""Shared subprocess environment helpers."""
import os


def python_subprocess_env() -> dict[str, str]:
    """Return a copy of os.environ with UTF-8 I/O and unbuffered output forced.

    Apply to every Python child process so Unicode-rich output (box-drawing
    chars, emoji from accelerator.print()) doesn't crash on Windows cp1252, and
    so stdout/stderr stream in real-time rather than buffering until exit.
    """
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env
