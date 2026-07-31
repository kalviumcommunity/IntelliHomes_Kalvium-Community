import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.corpus_cleaner import clean_text, clean_documents


def test_clean_text_removes_boilerplate_and_normalizes_whitespace():
    raw_text = "\uFEFFPage 1 of 3\n\n\t\tWelcome   to   IntelliHomes\n\n\n\n\u00A0\u00A0\n\nTable of Contents\n\nThe  property  documents  are  listed  below.\n"

    cleaned = clean_text(raw_text)

    assert "Page 1 of 3" not in cleaned
    assert "Table of Contents" not in cleaned
    assert "Welcome   to   IntelliHomes" not in cleaned
    assert "Welcome to IntelliHomes" in cleaned
    assert "The property documents are listed below." in cleaned
    assert "\n\n\n" not in cleaned


def test_clean_documents_writes_cleaned_files(tmp_path):
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "cleaned"
    source_dir.mkdir()
    output_dir.mkdir()

    (source_dir / "doc_one.md").write_text(
        "\uFEFFPage 1 of 2\n\n\tIntro   text\n\n",
        encoding="utf-8",
    )
    (source_dir / "doc_two.md").write_text(
        "Table of Contents\n\nSecond   section\n",
        encoding="utf-8",
    )

    cleaned_files = clean_documents(source_dir, output_dir)

    assert cleaned_files == [output_dir / "doc_one.md", output_dir / "doc_two.md"]
    assert (output_dir / "doc_one.md").exists()
    assert (output_dir / "doc_two.md").exists()
    assert "Page 1 of 2" not in (output_dir / "doc_one.md").read_text(encoding="utf-8")
    assert "Table of Contents" not in (output_dir / "doc_two.md").read_text(encoding="utf-8")
