#!/usr/bin/env python3
"""Arc En Ciel API smoke probe — Phase 0 of the additive download system.

Unauthenticated, read-only probe that pins Arc's public *pull* contract so the
ArcEnCielAdapter (Phase 2) and engine hardening (Phase 1) build on verified
shapes instead of assumptions. **No API key is required** — Arc's public
search/detail/versions/download endpoints work anonymously (verified with the
Arc dev, 2026-07-04). Safe to re-run whenever Arc changes its deploy so a broken
contract is caught before users hit failed downloads.

Verifies (per docs/specs/2026-07-03-additive-download-system.md §5):
  1. GET /models/classes                         -> base-model / class taxonomy
  2. GET /models/search?...&compact=true         -> open access + summary shape
  3. GET /models/{id}                            -> detail shape
  4. GET /models/{id}/versions                   -> authoritative file metadata
  5. HEAD /models/{id}/versions/{vid}/download   -> 302 -> uploads.arcenciel.io

It never downloads a model body (HEAD only) and makes a handful of requests well
under Arc's advertised 1200 req/min limit.

Run:  python scripts/arc_api_probe.py
Exit: 0 if the core contract held, 1 if a probe step failed.
"""

from __future__ import annotations

import sys
from typing import Any

try:
    import requests
except ImportError:  # pragma: no cover - requests is a backend dependency
    sys.exit("arc_api_probe: `requests` is required (pip install requests)")

# Windows consoles default to cp1252 and die on any non-ASCII byte; force UTF-8
# so the report can't crash on a stray character (project cross-platform rule).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # non-reconfigurable stream
        pass

BASE = "https://arcenciel.io/api"
UPLOADS_HOST = "uploads.arcenciel.io"
TIMEOUT = 20
HEADERS = {
    # Spec §6.11: identify KNX clearly in the User-Agent.
    "User-Agent": "Ecosystem/arc-probe (+https://github.com/UselessToys/Ecosystem_WebUI)",
    "Accept": "application/json",
}

# Version fields KNX's adapter will read (spec §6.6). We report which are present
# so a silent Arc rename is caught here rather than at download time.
EXPECTED_VERSION_FIELDS = [
    "originalName", "fileName", "filePath", "externalDownloadUrl",
    "fileSizeKb", "sha256", "sha256webui", "baseModel", "status", "fileScanStatus",
]

RATELIMIT_PREFIXES = ("ratelimit", "x-ratelimit", "retry-after")


def _hr(title: str) -> None:
    print(f"\n{'=' * 4} {title} {'=' * (72 - len(title))}")


def _report_ratelimit(resp: requests.Response) -> None:
    """Surface any rate-limit headers so we can tune backoff (spec §6.9)."""
    found = {k: v for k, v in resp.headers.items()
             if k.lower().startswith(RATELIMIT_PREFIXES)}
    if found:
        print("  rate-limit headers:", found)


def get_json(path: str, params: dict[str, Any] | None = None) -> Any:
    """GET an Arc JSON endpoint, printing status + rate-limit headers.

    Raises requests.HTTPError on a non-2xx response so the caller can decide
    whether the failed step is fatal to the probe.
    """
    url = f"{BASE}{path}"
    resp = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)
    print(f"  GET {resp.url} -> {resp.status_code}")
    _report_ratelimit(resp)
    resp.raise_for_status()
    return resp.json()


def probe_classes() -> None:
    """Step 1 — base-model/class taxonomy that drives search filters."""
    _hr("1. GET /models/classes")
    data = get_json("/models/classes")
    classes = data.get("classes") if isinstance(data, dict) else data
    names = [c.get("name") for c in classes] if isinstance(classes, list) else classes
    print(f"  classes ({len(names) if names else 0}):", names)


def probe_search() -> str | None:
    """Step 2 — confirm anonymous search works; return a probe-able model id.

    Uses compact=true (grid mode). Compact responses omit file metadata, so we
    only use it to *find* a model, never to resolve a download (spec §4.2).
    """
    _hr("2. GET /models/search (compact, status=available)")
    params = {
        "sort": "newest",
        "page": 1,
        "limit": 5,
        "status": "available",
        "compact": "true",
        "versionLimit": 1,
    }
    data = get_json("/models/search", params)
    items = data.get("models") or data.get("items") or data.get("data") \
        if isinstance(data, dict) else data
    if not isinstance(items, list) or not items:
        print("  !! no models returned — cannot continue version/download probe")
        return None
    first = items[0]
    print(f"  {len(items)} models; first-model summary keys:", sorted(first.keys()))
    model_id = first.get("id") or first.get("modelId")
    print("  probing model id:", model_id)
    return str(model_id) if model_id is not None else None


def probe_detail(model_id: str) -> None:
    """Step 3 — model detail shape."""
    _hr(f"3. GET /models/{model_id}")
    data = get_json(f"/models/{model_id}")
    if isinstance(data, dict):
        print("  detail keys:", sorted(data.keys()))


def probe_versions(model_id: str) -> str | None:
    """Step 4 — authoritative download metadata; return a probe-able version id."""
    _hr(f"4. GET /models/{model_id}/versions")
    data = get_json(f"/models/{model_id}/versions")
    versions = data.get("versions") if isinstance(data, dict) else data
    if not isinstance(versions, list) or not versions:
        print("  !! no versions returned")
        return None
    v = versions[0]
    present = [f for f in EXPECTED_VERSION_FIELDS if f in v]
    missing = [f for f in EXPECTED_VERSION_FIELDS if f not in v]
    print(f"  {len(versions)} versions; first-version keys:", sorted(v.keys()))
    print("  expected fields PRESENT:", present)
    if missing:
        print("  expected fields MISSING (contract drift?):", missing)
    print("  status:", v.get("status"), "| fileScanStatus:", v.get("fileScanStatus"))
    print("  externalDownloadUrl:", v.get("externalDownloadUrl"))
    print("  filePath:", v.get("filePath"), "| sha256:", v.get("sha256"))
    return str(v.get("id")) if v.get("id") is not None else None


def probe_download(model_id: str, version_id: str) -> None:
    """Step 5 — canonical download URL: expect 302 -> uploads.arcenciel.io, 200.

    HEAD only — we never pull the model body. Redirects are inspected manually so
    we can confirm the exact allowlist hosts the engine must permit (spec §4.3).
    """
    _hr(f"5. HEAD /models/{model_id}/versions/{version_id}/download")
    url = f"{BASE}/models/{model_id}/versions/{version_id}/download"

    # First hop, no auto-follow: capture the redirect target host.
    r1 = requests.head(url, headers=HEADERS, timeout=TIMEOUT, allow_redirects=False)
    print(f"  HEAD {url} -> {r1.status_code}")
    _report_ratelimit(r1)
    location = r1.headers.get("Location")
    print("  Location:", location)
    if location and UPLOADS_HOST in location:
        print(f"  OK: redirects to expected host ({UPLOADS_HOST})")
    elif r1.is_redirect:
        print("  !! redirects to an UNEXPECTED host - update the allowlist")

    # Follow to the final resource and report the content headers the engine
    # relies on for resume + filename. Some CDNs reject HEAD (405) — that's fine
    # to note; the adapter would GET with a Range probe instead.
    try:
        r2 = requests.head(url, headers=HEADERS, timeout=TIMEOUT, allow_redirects=True)
        print("  final URL:", r2.url, "->", r2.status_code)
        for h in ("Content-Length", "Accept-Ranges", "Content-Disposition"):
            print(f"    {h}:", r2.headers.get(h))
    except requests.RequestException as exc:
        print("  (following redirect failed / HEAD unsupported):", exc)


def main() -> int:
    print("Arc En Ciel API smoke probe —", BASE)
    try:
        probe_classes()
        model_id = probe_search()
        if not model_id:
            return 1
        probe_detail(model_id)
        version_id = probe_versions(model_id)
        if not version_id:
            print("\nNo version id to probe download with — stopping after versions.")
            return 1
        probe_download(model_id, version_id)
    except requests.HTTPError as exc:
        print("\n!! HTTP error — Arc returned a non-2xx:", exc)
        return 1
    except requests.RequestException as exc:
        print("\n!! Network error reaching Arc (run from a box with outbound access):", exc)
        return 1

    _hr("DONE")
    print("Core contract held. Pin these shapes into ArcEnCielAdapter (Phase 2).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
