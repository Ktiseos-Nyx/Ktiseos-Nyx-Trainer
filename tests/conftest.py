"""
Stub heavy production deps before pytest collects any test module.

Why this exists
---------------
services/__init__.py eagerly imports the full service layer on first import.
Any test that touches `services.*` (even a subpackage like services.core)
triggers that chain, which pulls in tomlkit, aiohttp, aiofiles, and
python-multipart — all real packages that aren't needed for GPU-free unit tests.

This file is loaded by pytest before collection begins (conftest.py semantics),
so the stubs are in place before a single test module is imported.

If a package is already installed (e.g. the user ran `pip install -r
requirements_dev.txt`), _stub_if_absent() leaves sys.modules untouched so the
real implementation is used instead of the mock.
"""
import sys
from unittest.mock import MagicMock


def _stub_if_absent(name: str) -> MagicMock:
    """Insert a MagicMock into sys.modules only when the real package is absent."""
    if name not in sys.modules:
        sys.modules[name] = MagicMock()
    return sys.modules[name]


# tomlkit — kohya_toml.py uses it for TOML serialisation; no test triggers generation
_stub_if_absent("tomlkit")

# aiohttp — civitai.py uses it for HTTP proxying; no test calls Civitai endpoints
_stub_if_absent("aiohttp")

# aiofiles — files.py uses it for async file I/O; no test downloads files
_stub_if_absent("aiofiles")

# python-multipart — FastAPI evaluates __version__ > "0.0.12" at route-registration
# time (string comparison).  Supply a real string so the assertion passes and
# FastAPI can register file-upload routes without the real package installed.
_pm = _stub_if_absent("python_multipart")
if not isinstance(getattr(_pm, "__version__", None), str):
    _pm.__version__ = "0.0.99"

_mp = _stub_if_absent("multipart")
if not isinstance(getattr(_mp, "__version__", None), str):
    _mp.__version__ = "0.0.99"

_mp_mp = _stub_if_absent("multipart.multipart")
if not callable(getattr(_mp_mp, "parse_options_header", None)):
    _mp_mp.parse_options_header = MagicMock()
