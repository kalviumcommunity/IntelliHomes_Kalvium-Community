"""
make_sample_corpus.py — Generate a small sample corpus for the document loader.

Writes these files under ``data/raw/sample/`` so reviewers can run the loader
without hunting for documents::

    sale-deed-sample.pdf    valid, extractable PDF (built by hand, correct xref)
    corrupt-sample.pdf      truncated PDF  -> must be skipped, not crash
    property-listing.html   HTML export (script/style stripped by the loader)
    locality-report.md      Markdown with YAML front matter
    tax-record.txt          plain text
    unsupported-sample.bin  binary blob with an unsupported extension -> skipped

Run from the repo root::

    python -m scripts.make_sample_corpus
    python backend/scripts/make_sample_corpus.py   # same thing

The script only uses the standard library, so it runs even before any
third-party dependencies are installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "data" / "raw" / "sample"


# ── Minimal PDF builder (stdlib only) ─────────────────────────────────────


def _pdf_escape(text: str) -> str:
    """Escape characters that are special inside a PDF string literal."""
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def build_simple_pdf(lines: list[str]) -> bytes:
    """Build a minimal one-page PDF whose text is the given lines.

    The xref table offsets are computed precisely, so pypdf can parse it.
    """
    content_lines = ["BT /F1 12 Tf 50 740 Td"]
    for i, line in enumerate(lines):
        if i:
            content_lines.append("0 -18 Td")
        content_lines.append(f"({_pdf_escape(line)}) Tj")
    content_lines.append("ET")

    stream = ("\n".join(content_lines) + "\n").encode("latin-1")

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
        ),
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"endstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for obj_num, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{obj_num} 0 obj\n".encode()
        out += body + b"\nendobj\n"

    xref_pos = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()

    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_pos}\n%%EOF\n"
    ).encode()
    return bytes(out)


# ── Corpus contents ───────────────────────────────────────────────────────


SALE_DEED_LINES = [
    "SALE DEED",
    "Property: Flat 402, Emerald Heights, Sector 62, Noida",
    "Seller: Ramesh Kumar Soni",
    "Buyer: Anita Sharma",
    "Executed on: 14 March 2025",
    "Registration No: 2451/2025",
    "Consideration: INR 48,50,000",
    "Stamp duty paid in full. Possession handed over on execution.",
]

PROPERTY_LISTING_HTML = """<!DOCTYPE html>
<html lang="en">
<head><title>Property Listing - 3 BHK in Whitefield</title></head>
<body>
<h1>3 BHK Apartment for Sale - Whitefield, Bangalore</h1>
<p>Price: INR 1,15,00,000 | Built-up area: 1,650 sq.ft.</p>
<ul>
  <li>3 bedrooms, 3 bathrooms</li>
  <li>24x7 security, gated community</li>
  <li>Clubhouse, gym, and swimming pool</li>
</ul>
<div class="notes">
  <p>Contact broker: +91 98765 43210</p>
</div>
<script>document.write("this text must never reach the RAG corpus");</script>
<style>.notes { color: gray; }</style>
</body>
</html>
"""

LOCALITY_REPORT_MD = """---
title: Locality Report - HSR Layout
author: IntelliHomes Research
date: 2026-06-01
---

# Locality Report: HSR Layout, Bangalore

## Overview

HSR Layout is a planned residential neighbourhood in South-East Bangalore,
popular with IT professionals due to its proximity to tech parks.

## Key Facts

- Average apartment price: INR 12,000 per sq.ft.
- Distance to nearest metro: 3.2 km
- Schools within 2 km: 14
- Hospitals within 2 km: 6

## Verdict

Good rental demand and steady capital appreciation over the past five years.
"""

TAX_RECORD_TXT = """PROPERTY TAX RECORD
------------------------------------
Property ID: NDA-402-EMERALD
Owner: Anita Sharma
Financial Year: 2025-26
Tax Assessed: INR 18,400
Tax Paid: INR 18,400
Status: PAID (no arrears)
"""


# ── Writer ────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    output_dir = Path(args[0]) if args else DEFAULT_OUTPUT
    output_dir.mkdir(parents=True, exist_ok=True)

    files: dict[str, bytes] = {
        "sale-deed-sample.pdf": build_simple_pdf(SALE_DEED_LINES),
        "corrupt-sample.pdf": build_simple_pdf(SALE_DEED_LINES)[:180],  # truncated!
        "property-listing.html": PROPERTY_LISTING_HTML.encode("utf-8"),
        "locality-report.md": LOCALITY_REPORT_MD.encode("utf-8"),
        "tax-record.txt": TAX_RECORD_TXT.encode("utf-8"),
        "unsupported-sample.bin": b"\x00\x01\x02PK\x03\x04 not a supported format",
    }

    for name, data in files.items():
        target = output_dir / name
        target.write_bytes(data)
        print(f"wrote {target} ({len(data)} bytes)")

    print("\nCorpus ready. Try:")
    print(f"  python backend/ingestion/loader.py {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
