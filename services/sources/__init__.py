"""Source-adapter package for the additive download system.

Adapters register themselves here on import, so importing this package makes the
registry ready. Adding a source = one new adapter module + one ``register()``.
"""

from services.sources.registry import get_adapter, list_adapters, register
from services.sources.arcenciel import ArcEnCielAdapter

register(ArcEnCielAdapter())

__all__ = ["register", "get_adapter", "list_adapters", "ArcEnCielAdapter"]
