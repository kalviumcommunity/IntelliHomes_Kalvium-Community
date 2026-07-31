"""
loader.py — Corpus ingestion: load mixed-format documents into plain text.

IntelliHomes RAG pipeline — Document Loading stage.

Responsibilities
----------------
* Read PDF, HTML, Markdown, and plain-text files into a common plain-text form.
* Tag every loaded document with its source filename/path so it can be cited
  by the retrieval layer later.
* Survive bad input: missing, corrupt, or unsupported files are skipped with a
  clear reason instead of crashing the whole run.
* Print an intake report (character count + short sample) for each document
  so ingestion results can be eyeballed.

Usage
-----
Import the functions::

    from ingestion.loader import load_folder, print_intake_report

    result = load_folder("data/raw/sample")
    print_intake_report(result)

Or run directly as a script::

    python backend/ingestion/loader.py data/raw/sample

The script accepts files and folders; folders are scanned recursively.
"""

from __future__ import annotations

import logging
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "Document",
    "SkippedFile",
    "LoadResult",
    "UnsupportedFormatError",
    "DocumentLoadError",
    "load_file",
    "load_folder",
    "print_intake_report",
    "main",
]


# ── Errors ────────────────────────────────────────────────────────────────


class UnsupportedFormatError(Exception):
    """Raised when a file's extension is not a supported document format."""


class DocumentLoadError(Exception):
    """Raised when a supported file could not be turned into plain text."""


# ── Data model ────────────────────────────────────────────────────────────


@dataclass
class Document:
    """A single loaded document in a common plain-text representation."""

    source: str  # original filename or identifier (for citations)
    path: str  # filesystem path the text came from
    format: str  # "pdf" | "html" | "markdown" | "text"
    text: str
    char_count: int = field(default=0)

    def __post_init__(self) -> None:
        self.char_count = len(self.text)


@dataclass
class SkippedFile:
    """A file that was intentionally skipped, with the reason why."""

    path: str
    reason: str


@dataclass
class LoadResult:
    """Outcome of loading one folder: everything that loaded and everything skipped."""

    documents: list[Document] = field(default_factory=list)
    skipped: list[SkippedFile] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.documents)


# ── Format dispatch ───────────────────────────────────────────────────────


def _read_text_robust(path: Path) -> str:
    """Read a file's bytes and decode them without ever crashing on encoding."""
    data = path.read_bytes()
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _extract_pdf(path: Path) -> str:
    """Extract plain text from every page of a PDF using pypdf."""
    try:
        from pypdf import PdfReader
        from pypdf.errors import PdfReadWarning
    except ImportError as exc:  # pragma: no cover - environment check
        raise DocumentLoadError(f"pypdf is not installed ({exc})") from exc

    # A corrupt PDF makes pypdf warn (via logging) *and* raise; we report the
    # skip ourselves, so quiet pypdf down to keep intake output readable.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PdfReadWarning)
        pypdf_logger = logging.getLogger("pypdf")
        previous_level = pypdf_logger.level
        pypdf_logger.setLevel(logging.ERROR)
        try:
            reader = PdfReader(str(path))
        finally:
            pypdf_logger.setLevel(previous_level)

    pages = []
    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # noqa: BLE001 - one bad page shouldn't sink the file
            raise DocumentLoadError(f"page extraction failed: {exc}") from exc
        if text.strip():
            pages.append(text.strip())
    return "\n\n".join(pages)


def _extract_html(path: Path) -> str:
    """Parse an HTML file and return its visible text (scripts/style removed)."""
    raw = _read_text_robust(path)
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:  # pragma: no cover - environment check
        raise DocumentLoadError(f"beautifulsoup4 is not installed ({exc})") from exc

    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "header", "footer"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines)


def _extract_markdown(path: Path) -> str:
    """Read a Markdown file, stripping any YAML front matter at the top."""
    text = _read_text_robust(path)
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            text = text[end + 4 :].lstrip("\n")
    return text.strip()


def _extract_text(path: Path) -> str:
    """Read a plain-text file as-is."""
    return _read_text_robust(path).strip()


# extension -> (display format, extractor)
_SUPPORTED_FORMATS: dict[str, tuple[str, object]] = {
    ".pdf": ("pdf", _extract_pdf),
    ".html": ("html", _extract_html),
    ".htm": ("html", _extract_html),
    ".md": ("markdown", _extract_markdown),
    ".markdown": ("markdown", _extract_markdown),
    ".txt": ("text", _extract_text),
    ".text": ("text", _extract_text),
}


# ── Public API ────────────────────────────────────────────────────────────


def load_file(path: str | Path) -> Document:
    """Load a single document file into plain text.

    Raises
    ------
    FileNotFoundError
        If the path does not exist.
    UnsupportedFormatError
        If the extension is not a supported document format.
    DocumentLoadError
        If the file exists but cannot be read or yields no extractable text.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"file does not exist: {p}")
    if p.is_dir():
        raise DocumentLoadError(f"{p.name} is a directory, not a file")

    ext = p.suffix.lower()
    if ext not in _SUPPORTED_FORMATS:
        raise UnsupportedFormatError(f"unsupported format {ext!r}")

    fmt, extractor = _SUPPORTED_FORMATS[ext]
    try:
        text = extractor(p)  # type: ignore[operator]
    except (UnsupportedFormatError, DocumentLoadError):
        raise
    except Exception as exc:  # noqa: BLE001 - corrupt file should not crash the run
        raise DocumentLoadError(f"could not read {p.name}: {exc}") from exc

    if not text.strip():
        raise DocumentLoadError(f"{p.name} yielded no extractable text")

    return Document(source=p.name, path=str(p), format=fmt, text=text)


def load_folder(folder: str | Path, recursive: bool = True) -> LoadResult:
    """Load every supported document under *folder*; skip the rest gracefully.

    Missing folders, corrupt files, unsupported extensions, and files with no
    extractable text are recorded in ``LoadResult.skipped`` instead of raising.
    """
    root = Path(folder)
    result = LoadResult()

    if not root.exists():
        result.skipped.append(SkippedFile(str(root), "folder does not exist"))
        return result

    iterator = root.rglob("*") if recursive else root.glob("*")
    for candidate in sorted(iterator):
        if not candidate.is_file():
            continue
        try:
            result.documents.append(load_file(candidate))
        except (FileNotFoundError, UnsupportedFormatError, DocumentLoadError) as exc:
            result.skipped.append(SkippedFile(str(candidate), str(exc)))

    return result


def print_intake_report(result: LoadResult, sample_chars: int = 120) -> None:
    """Print an intake confirmation for every document: length + short sample."""
    print(f"Loaded {len(result.documents)} document(s), "
          f"skipped {len(result.skipped)} file(s).")

    for doc in result.documents:
        sample = " ".join(doc.text.split())
        if len(sample) > sample_chars:
            sample = sample[:sample_chars] + "…"
        print(f"\n  [ok] {doc.source}  ({doc.format}, {doc.char_count} chars)")
        print(f"       sample: “{sample}”")

    for skip in result.skipped:
        print(f"  [skip] {skip.path} — {skip.reason}")


# ── CLI ───────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point: ``python loader.py <file-or-folder>...``."""
    args = sys.argv[1:] if argv is None else argv
    if not args:
        print("usage: python loader.py <file-or-folder> [<file-or-folder> ...]")
        return 2

    result = LoadResult()
    for arg in args:
        p = Path(arg)
        if p.is_dir():
            sub = load_folder(p)
            result.documents.extend(sub.documents)
            result.skipped.extend(sub.skipped)
        else:
            try:
                result.documents.append(load_file(p))
            except (FileNotFoundError, UnsupportedFormatError, DocumentLoadError) as exc:
                result.skipped.append(SkippedFile(str(p), str(exc)))

    print_intake_report(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
