import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ingestion.loader import DocumentLoadError, load_file


def test_scanned_pdf_is_ocr_searchable(tmp_path: Path) -> None:
    if shutil.which("tesseract") is None:
        pytest.skip("Tesseract is not installed")
    image = pytest.importorskip("PIL.Image")
    draw_module = pytest.importorskip("PIL.ImageDraw")
    font_module = pytest.importorskip("PIL.ImageFont")

    pdf_path = tmp_path / "scanned.pdf"
    canvas = image.new("RGB", (1200, 300), "white")
    font = font_module.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 64
    )
    draw_module.Draw(canvas).text(
        (40, 90), "LEGAL OWNERSHIP TITLE DEED", font=font, fill="black"
    )
    canvas.save(pdf_path, "PDF")

    document = load_file(pdf_path)

    assert "LEGAL OWNERSHIP TITLE DEED" in document.text


def test_scanned_pdf_reports_missing_ocr_engine(monkeypatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "scanned.pdf"
    pdf_path.write_bytes(b"not a pdf")
    monkeypatch.setenv("OCR_ENABLED", "1")
    monkeypatch.setattr(shutil, "which", lambda _command: None)

    with pytest.raises(DocumentLoadError):
        load_file(pdf_path)
