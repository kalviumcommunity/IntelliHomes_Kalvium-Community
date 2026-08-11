"""Query-to-answer RAG pipeline: embed -> retrieve -> assemble -> generate.

Public symbols are re-exported lazily (PEP 562) so ``python -m pipeline.runner``
and ``python scripts/run_pipeline.py`` run without duplicate-import warnings.
"""

from typing import Any

# symbol -> module (within this package) that defines it
_LAZY_EXPORTS = {
    "embed_stage": "embed",
    "retrieve_stage": "retrieve",
    "assemble_stage": "assemble",
    "generate_stage": "generate",
    "make_llm": "generate",
    "run_pipeline": "runner",
}


def __getattr__(name: str) -> Any:
    module = _LAZY_EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(f"{__name__}.{module}"), name)


def __dir__() -> list[str]:
    return sorted(list(globals()) + list(_LAZY_EXPORTS))
