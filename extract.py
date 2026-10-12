"""Extract plain text from PDF, Word (.docx), TXT, and image files."""

from pathlib import Path
from typing import NamedTuple

from docx import Document
from pypdf import PdfReader

import config
from config import MAX_OCR_PAGES, MIN_CHARS_PER_PAGE
from hidden_text import (
    HiddenTextReport,
    hidden_pdf_spans,
    remove_from_page,
    remove_invisible_characters,
    visible_docx_paragraph,
)
from ocr import IMAGE_EXTENSIONS, extract_text_ocr, ocr_pdf_pages, skipped_page_marker

# The extraction behind a stored paper's text. Bump it whenever the same file could now give a
# different text (OCR engine, model, DPI or page limit, hidden-text filter, the page markers):
# /upload and /summarize reuse a stored text for the same file bytes only when it was made by
# this version (rag.stored_paper_for_file).
EXTRACTOR_VERSION = ("2026-10-12 rapidocr PP-OCRv6 small, 200 DPI, 13 OCR pages, page markers, column reading order, "
                     "column lines by own coordinates")


class ExtractionResult(NamedTuple):
    text: str
    ocr_used: bool
    # Text of each page (PDFs and images); None for formats without pages.
    # `text` is these pages joined with "\n".
    pages: list[str] | None = None
    # With HIDDEN_TEXT_FILTER: {"count": ..., "preview": [...]} of the hidden text removed; None when off.
    hidden_removed: dict | None = None
    # Scanned PDF pages over MAX_OCR_PAGES that were not read (1-based); each holds a marker instead.
    ocr_skipped_pages: tuple[int, ...] = ()


def _extract_pdf(path: Path, report: HiddenTextReport | None) -> ExtractionResult:
    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    # Decide per page, so mixed PDFs keep their real text and only the scanned
    # pages go through the (slow) OCR engine. Decided before hidden text is
    # removed: a page that had a text layer is not a scan, and OCR would read
    # near-white text back in.
    scanned = [index for index, text in enumerate(pages) if len(text.strip()) < MIN_CHARS_PER_PAGE]
    if report is not None:
        for index, hidden in enumerate(hidden_pdf_spans(path)):
            if index < len(pages) and hidden:
                pages[index] = remove_from_page(pages[index], hidden, report)
    to_ocr, skipped = scanned[:MAX_OCR_PAGES], scanned[MAX_OCR_PAGES:]
    for index, text in ocr_pdf_pages(path, to_ocr).items():
        pages[index] = text
    for index in skipped:
        # The marker shows where the gap is; the little text layer the page had (a running header,
        # a page number) is kept after it.
        pages[index] = "\n".join(part for part in (skipped_page_marker(index + 1), pages[index].strip()) if part)

    return ExtractionResult("\n".join(pages), ocr_used=bool(to_ocr), pages=pages,
                            ocr_skipped_pages=tuple(index + 1 for index in skipped))


def _extract_docx(path: Path, report: HiddenTextReport | None) -> ExtractionResult:
    document = Document(path)
    if report is None:
        text = "\n".join(paragraph.text for paragraph in document.paragraphs)
    else:
        text = "\n".join(visible_docx_paragraph(paragraph, report) for paragraph in document.paragraphs)
    return ExtractionResult(text, ocr_used=False)


def _extract_txt(path: Path, report: HiddenTextReport | None) -> ExtractionResult:
    return ExtractionResult(path.read_text(encoding="utf-8"), ocr_used=False)


def _extract_image(path: Path, report: HiddenTextReport | None) -> ExtractionResult:
    text = extract_text_ocr(path)
    return ExtractionResult(text, ocr_used=True, pages=[text])


_EXTRACTORS = {
    ".pdf": _extract_pdf,
    ".docx": _extract_docx,
    ".txt": _extract_txt,
    **{extension: _extract_image for extension in IMAGE_EXTENSIONS},
}

SUPPORTED_EXTENSIONS = tuple(_EXTRACTORS)


def extract_text(file_path: str | Path) -> ExtractionResult:
    """Return the text of a supported file (chosen by extension) and whether OCR was used."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    extension = path.suffix.lower()
    extractor = _EXTRACTORS.get(extension)
    if extractor is None:
        supported = ", ".join(_EXTRACTORS)
        raise ValueError(f"Unsupported file type '{extension}'. Supported: {supported}")

    report = HiddenTextReport() if config.HIDDEN_TEXT_FILTER else None
    result = extractor(path, report)
    if report is None:
        return result

    # Invisible Unicode characters, in every format. For paged formats `text` is
    # the pages joined, so the same characters are removed from it without being
    # counted a second time.
    text, pages = result.text, result.pages
    if pages is not None:
        pages = [remove_invisible_characters(page, report) for page in pages]
        text = remove_invisible_characters(text, HiddenTextReport())
    else:
        text = remove_invisible_characters(text, report)
    return result._replace(text=text, pages=pages, hidden_removed=report.as_dict())


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python extract.py <file_path>")
        sys.exit(1)
    print(extract_text(sys.argv[1]).text)
