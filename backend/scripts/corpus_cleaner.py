import re
import unicodedata
from pathlib import Path
from typing import Iterable


BOILERPLATE_PATTERNS = [
    re.compile(r"page\s+\d+\s+of\s+\d+", re.IGNORECASE),
    re.compile(r"table\s+of\s+contents", re.IGNORECASE),
    re.compile(r"navigation", re.IGNORECASE),
    re.compile(r"copyright\s+\d{4}", re.IGNORECASE),
]


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def strip_bom(text: str) -> str:
    return text.lstrip("\ufeff")


def remove_boilerplate(text: str) -> str:
    cleaned = text
    for pattern in BOILERPLATE_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    return cleaned


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", "\n", text)
    text = text.replace("\u00a0", " ")
    return text.strip()


def clean_text(text: str) -> str:
    text = normalize_unicode(text)
    text = strip_bom(text)
    text = remove_boilerplate(text)
    text = normalize_whitespace(text)
    return text


def clean_documents(source_dir: str | Path, output_dir: str | Path) -> list[Path]:
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cleaned_files: list[Path] = []
    for file_path in sorted(source_path.glob("*.md")):
        original_text = file_path.read_text(encoding="utf-8")
        cleaned_text = clean_text(original_text)
        output_file = output_path / file_path.name
        output_file.write_text(cleaned_text, encoding="utf-8")
        cleaned_files.append(output_file)

    return cleaned_files


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    docs_dir = repo_root.parent / "docs"
    cleaned_dir = repo_root / "cleaned_corpus"
    cleaned_files = clean_documents(docs_dir, cleaned_dir)
    print(f"Cleaned {len(cleaned_files)} files into {cleaned_dir}")


if __name__ == "__main__":
    main()
