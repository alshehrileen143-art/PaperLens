"""Find and remove text that a reader of the document cannot see.

Used by extract.py when config.HIDDEN_TEXT_FILTER is on. Removed text is
collected in a HiddenTextReport so the API can tell the user what was dropped.

Known limit: "near the background colour" assumes a white page; text hidden on
a coloured background is not detected.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

import config

# Zero-width characters and BOM, bidirectional controls, and Unicode tag characters.
_INVISIBLE = re.compile(
    "[​‌‍⁠﻿"
    "‎‏‪-‮⁦-⁩"
    "\U000e0000-\U000e007f]"
)
PREVIEW_ITEMS = 5
PREVIEW_CHARS = 80
# Hidden PDF text is removed by matching it in the extracted page text. Very
# short pieces ("1", "of") would match the wrong, visible occurrence, so they
# are left alone.
MIN_REMOVABLE_CHARS = 4


@dataclass
class HiddenTextReport:
    count: int = 0  # Hidden text pieces (spans / runs) plus invisible characters removed.
    preview: list[str] = field(default_factory=list)

    def add(self, text: str, count: int = 1) -> None:
        self.count += count
        text = " ".join(text.split())
        if text and len(self.preview) < PREVIEW_ITEMS:
            self.preview.append(text[:PREVIEW_CHARS])

    def as_dict(self) -> dict:
        return {"count": self.count, "preview": self.preview}


def _near_white(rgb) -> bool:
    return rgb is not None and len(rgb) >= 3 and all(channel >= config.HIDDEN_NEAR_WHITE for channel in rgb[:3])


def remove_invisible_characters(text: str, report: HiddenTextReport) -> str:
    found = _INVISIBLE.findall(text)
    if found:
        codes = ", ".join(sorted({f"U+{ord(char):04X}" for char in found}))
        report.add(f"{len(found)} invisible characters ({codes})", count=len(found))
        text = _INVISIBLE.sub("", text)
    return text


# --- PDF ---


def _span_text(span: dict) -> str:
    return "".join(chr(char[0]) for char in span.get("chars", []))


def _is_hidden_span(span: dict, page, backgrounds: list) -> bool:
    # Text-trace boxes are in unrotated page space; page.rect is rotated
    # (landscape table pages are often stored with rotation 90).
    raw = pymupdf.Rect(span["bbox"])
    rotated = raw * page.rotation_matrix
    if not rotated.intersects(page.rect):  # Entirely off the page.
        return True
    if span.get("size", 0) < config.HIDDEN_MIN_FONT_SIZE:
        return True
    invisible = span.get("type") == 3 or span.get("opacity", 1) == 0  # Render mode 3: neither filled nor stroked.
    if invisible or _near_white(span.get("color")):
        # Over an image or a coloured box this is normal: searchable scans put
        # their OCR text in render mode 3 on top of the page image, and table
        # headers often use white text on a dark fill. The reader sees it.
        return not any(raw.intersects(rect) or rotated.intersects(rect) for rect in backgrounds)
    return False


def _backgrounds(page) -> list:
    """Areas that are not white page: images and filled, non-white drawings."""
    rects = [pymupdf.Rect(info["bbox"]) for info in page.get_image_info()]
    for drawing in page.get_drawings():
        fill = drawing.get("fill")
        if fill is not None and not _near_white(fill):
            rects.append(pymupdf.Rect(drawing["rect"]))
    return rects


def hidden_pdf_spans(path: Path) -> list[list[str]]:
    """For each page, the texts of its hidden spans (those long enough to remove safely)."""
    pages = []
    with pymupdf.open(path) as document:
        for page in document:
            backgrounds = _backgrounds(page)
            hidden = []
            for span in page.get_texttrace():
                text = _span_text(span).strip()
                if len(text) >= MIN_REMOVABLE_CHARS and _is_hidden_span(span, page, backgrounds):
                    hidden.append(text)
            pages.append(hidden)
    return pages


def remove_from_page(page_text: str, hidden: list[str], report: HiddenTextReport) -> str:
    """Remove each hidden span's text from the page text extracted by pypdf."""
    for text in hidden:
        # pypdf may break lines or spacing differently than PyMuPDF.
        pattern = r"\s*".join(re.escape(word) for word in text.split())
        new_text, removed = re.subn(pattern, " ", page_text, count=1)
        if removed:
            page_text = new_text
            report.add(text)
    return page_text


# --- DOCX ---


def _docx_run_hidden(run, paragraph) -> bool:
    font = run.font
    hidden = font.hidden if font.hidden is not None else paragraph.style.font.hidden
    if hidden:
        return True
    if font.size is not None and font.size.pt < config.HIDDEN_MIN_FONT_SIZE:
        return True
    rgb = font.color.rgb if font.color is not None and font.color.type is not None else None
    if rgb is not None and _near_white(tuple(channel / 255 for channel in rgb)):
        return True
    return False


def visible_docx_paragraph(paragraph, report: HiddenTextReport) -> str:
    parts = []
    for run in paragraph.runs:
        if _docx_run_hidden(run, paragraph):
            report.add(run.text)
        else:
            parts.append(run.text)
    return "".join(parts)
