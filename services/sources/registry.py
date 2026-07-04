"""Source-adapter registry.

Routes and UI iterate the registry so adding a new source is one new adapter
class plus one ``register()`` call. Adapters register themselves on import
(see ``services/sources/__init__.py``).
"""

from __future__ import annotations

from typing import Optional

from services.sources.base import SourceAdapter

_ADAPTERS: dict[str, SourceAdapter] = {}


def register(adapter: SourceAdapter) -> None:
    """Register an adapter under its ``name`` (later registration wins)."""
    _ADAPTERS[adapter.name] = adapter


def get_adapter(name: str) -> Optional[SourceAdapter]:
    """Return the adapter registered under ``name``, or None."""
    return _ADAPTERS.get(name)


def list_adapters() -> list[SourceAdapter]:
    """Return all registered adapters (order-stable by registration)."""
    return list(_ADAPTERS.values())
