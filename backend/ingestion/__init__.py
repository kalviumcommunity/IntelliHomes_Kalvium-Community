"""Document ingestion: turn mixed-format files into plain text for the RAG pipeline.

Public symbols are re-exported lazily (PEP 562) so ``python -m ingestion.loader``
runs without a duplicate-import warning.
"""

from typing import Any

_LAZY_EXPORTS = (
    "Document",
    "LoadResult",
    "SkippedFile",
    "load_file",
    "load_folder",
    "print_intake_report",
    "main",
)


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        from . import loader  # noqa: PLC0415 - import on first use

        return getattr(loader, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals()) + list(_LAZY_EXPORTS))
