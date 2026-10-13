"""Extract plain text from PDF, Word (.docx), TXT, and image files."""

from pathlib import Path
from typing import NamedTuple

from docx import Document
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph
from pypdf import PdfReader

import config
from config import MAX_OCR_PAGES, MIN_CHARS_PER_PAGE
from hidden_text import (
    HiddenTextReport,
    docx_paragraph_text,
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
                     "column lines by own coordinates, hidden Arabic in visual order, DOCX links and fields, "
                     "DOCX tables")


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


_TABLE_EDGE = object()  # Where a blank line sets a table off from the text around it (see _extract_docx).


def _docx_paragraph(paragraph: Paragraph, report: HiddenTextReport | None) -> str:
    # Links, tracked insertions, content controls and simple fields too (see hidden_text.docx_runs).
    return docx_paragraph_text(paragraph) if report is None else visible_docx_paragraph(paragraph, report)


def _docx_blocks(parent, document, report: HiddenTextReport | None) -> list[str]:
    """The lines of a DOCX body (or a cell, or a content control) in document order: paragraphs,
    tables, and what block-level content controls (w:sdt) hold."""
    lines = []
    for child in parent.iterchildren():
        if child.tag == qn("w:p"):
            lines.append(_docx_paragraph(Paragraph(child, document), report))
        elif child.tag == qn("w:tbl"):
            lines += [_TABLE_EDGE, *_docx_table(child, document, report), _TABLE_EDGE]
        elif child.tag == qn("w:sdt"):
            content = child.find(qn("w:sdtContent"))
            if content is not None:
                lines += _docx_blocks(content, document, report)
    return lines


def _docx_table(table, document, report: HiddenTextReport | None) -> list[str]:
    """One line per row, cells separated by a tab: a quote of a row with its cells joined by spaces
    still matches (quote_in_source treats a tab as a space), and each number stays on its row.

    A cell merged across columns (gridSpan) is one cell, so its text comes once; a cell merged
    down rows (vMerge) has its text in its first row only. Empty cells at the end of a row are
    left out. A nested table's rows follow the row that holds it."""
    lines = []
    for row in table.tr_lst:
        cells, nested = [], []
        for cell in row.tc_lst:
            if cell.vMerge == "continue":
                cells.append("")
                continue
            parts = []
            for child in cell.iterchildren():
                if child.tag == qn("w:p"):
                    parts.append(_docx_paragraph(Paragraph(child, document), report))
                elif child.tag == qn("w:tbl"):
                    nested += _docx_table(child, document, report)
            cells.append(" ".join(" ".join(part.split()) for part in parts if part.strip()))
        while cells and not cells[-1]:
            cells.pop()
        if cells:
            lines.append("\t".join(cells))
        lines += nested
    return lines


def _extract_docx(path: Path, report: HiddenTextReport | None) -> ExtractionResult:
    document = Document(path)
    lines = _docx_blocks(document.element.body, document, report)
    # A table is set off by a blank line before and after it; none at the start or end of the text,
    # or where one is already there. The document's own empty paragraphs stay as they are.
    kept = []
    for index, line in enumerate(lines):
        if line is not _TABLE_EDGE:
            kept.append(line)
        elif kept and kept[-1] != "" and index < len(lines) - 1 and lines[index + 1] not in ("", _TABLE_EDGE):
            kept.append("")
    return ExtractionResult("\n".join(kept), ocr_used=False)


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
