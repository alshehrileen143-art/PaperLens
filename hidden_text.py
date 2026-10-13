"""Find and remove text that a reader of the document cannot see.

Used by extract.py when config.HIDDEN_TEXT_FILTER is on. Removed text is
collected in a HiddenTextReport so the API can tell the user what was dropped.

Known limit: "near the background colour" assumes a white page; text hidden on
a coloured background is not detected.
"""

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf
from docx.oxml.ns import qn
from docx.text.run import Run

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


@dataclass
class PageHidden:
    """A page's hidden text, as PyMuPDF sees it, to remove from the text pypdf extracted."""
    # Hidden spans without Arabic letters, MIN_REMOVABLE_CHARS or longer, matched as they are.
    texts: list[str] = field(default_factory=list)
    # Each line that holds a hidden span with Arabic letters: all its hidden spans, in the PDF's order.
    arabic_lines: list[list[str]] = field(default_factory=list)
    # The page's visible spans with Arabic letters: hidden text that also reads as one of them stays.
    visible_arabic: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.texts or self.arabic_lines)


def hidden_pdf_spans(path: Path) -> list[PageHidden]:
    """For each page, its hidden text (see PageHidden)."""
    pages = []
    with pymupdf.open(path) as document:
        for page in document:
            backgrounds = _backgrounds(page)
            found, lines, line_middle, line_height = PageHidden(), [], None, 0.0
            for span in page.get_texttrace():
                text = _span_text(span).strip()
                if not text:
                    continue
                if not _is_hidden_span(span, page, backgrounds):
                    if _has_arabic(text):
                        found.visible_arabic.append(text)
                    continue
                if not _has_arabic(text) and len(text) >= MIN_REMOVABLE_CHARS:
                    found.texts.append(text)
                # Spans whose middles are within half a line of the line's first span share its line.
                top, bottom = span["bbox"][1], span["bbox"][3]
                middle, height = (top + bottom) / 2, max(bottom - top, 0.1)
                if line_middle is None or abs(middle - line_middle) > max(height, line_height) / 2:
                    lines.append([])
                    line_middle, line_height = middle, height
                lines[-1].append(text)
            found.arabic_lines = [line for line in lines if any(_has_arabic(text) for text in line)]
            pages.append(found)
    return pages


# --- Arabic hidden text ---
# PDF writers store Arabic in different orders. PyMuPDF gives the characters as the PDF stores them,
# while pypdf gives reading order: a Word export came back fully reversed ("زمرلا فضأو … لهاجت" for
# "تجاهل … الرمز"), and text written one word at a time came back with each word's letters reversed.
# Hidden Arabic is therefore matched in four forms, on a skeleton of the text (NFKC, which turns
# presentation forms into letters, without spaces, tatweel or harakat: pypdf also splits words,
# "السابق ة"). Digits and Latin text inside Arabic ("CANARY-7431") keep their own order in every form.

_ARABIC = re.compile(r"[؀-ۿﭐ-﷿ﹰ-﻿]")
_ARABIC_RUN = re.compile(r"[؀-ۿ]+")
# Only letters and digits are compared: whitespace, punctuation (it changes sides when a word is
# stored in visual order, ":ةقرولا" for "الورقة:"), tatweel and harakat are left out.
_NOT_IN_SKELETON = re.compile(r"[\W_ـً-ٰٟ]")
# A long hidden Arabic span needs this many letters (skeleton) and two words to be matched in a
# reversed form; shorter ones are only removed next to hidden text already removed (_REMOVED).
MIN_ARABIC_LETTERS = 8
# Between a short span and the removed text next to it, only this much may lie, with no letters
# (the "-7431" of a marker split from "CANARY"); it goes with the span.
MAX_GAP_CHARS = 12
_REMOVED = "\x00"  # Marks where hidden text was taken out; becomes a space at the end.


def _has_arabic(text: str) -> bool:
    return bool(_ARABIC.search(text))


_ALEF = "اأإآ"


def _skeleton(text: str) -> tuple[str, list[int]]:
    """The text as compared (NFKC, letters and digits only), and for each of its characters the
    index of the character of `text` it came from.

    Lam and alef next to each other are put in one order ("لا"): a lam-alef ligature is stored
    in reading order inside a word stored in visual order, so reversing the word turns "ملاحظة"
    ('ةظحلام' in a Word export) into "مالحظة", and a stored "لا" cannot be told from a reversed "ال"."""
    out, where = [], []
    for index, char in enumerate(text):
        for normal in unicodedata.normalize("NFKC", char):
            if not _NOT_IN_SKELETON.match(normal):
                out.append(normal)
                where.append(index)
    for i in range(len(out) - 1):
        if out[i] in _ALEF and out[i + 1] == "ل":
            out[i], out[i + 1] = out[i + 1], out[i]
            where[i], where[i + 1] = where[i + 1], where[i]
    return "".join(out), where


def _flip_arabic(text: str) -> str:
    """Each Arabic run reversed; digits and Latin text stay as they are."""
    return _ARABIC_RUN.sub(lambda match: match.group()[::-1], text)


def _arabic_forms(text: str) -> list[str]:
    """The span as it may read in pypdf's order: as stored, fully reversed, each word reversed,
    and with the word order reversed (Latin and digit runs never turned)."""
    text = unicodedata.normalize("NFKC", text)
    words = text.split()
    forms = [text, " ".join(_flip_arabic(word) for word in reversed(words)),
             " ".join(_flip_arabic(word) for word in words), " ".join(reversed(words))]
    return list(dict.fromkeys(forms))


def _arabic_letters(text: str) -> int:
    return sum(len(run) for run in _ARABIC_RUN.findall(_skeleton(text)[0]))


def _remove_arabic_line(page_text: str, line: list[str], visible: list[str], report: HiddenTextReport) -> str:
    # The visible Arabic text, as a skeleton in every form: what reads as visible text is not removed.
    seen = "\n".join(_skeleton(form)[0] for text in visible for form in _arabic_forms(text))
    pending = [text for text in line if _has_arabic(text)]
    for text in list(pending):  # Long spans: anywhere on the page, in any form.
        if _arabic_letters(text) < MIN_ARABIC_LETTERS or len(text.split()) < 2:
            continue
        for form in _arabic_forms(text):
            found = _find(page_text, form)
            if found and _skeleton(page_text[found[0]:found[1]])[0] not in seen:
                report.add(page_text[found[0]:found[1]])
                page_text = page_text[:found[0]] + _REMOVED + page_text[found[1]:]
                pending.remove(text)
                break
    # Short spans (and single words): only next to hidden text already removed from this page (by a
    # long span, or a Latin one such as the marker), so each removal can extend the next.
    changed = _REMOVED in page_text
    while changed and pending:
        changed = False
        for text in list(pending):
            for form in _arabic_forms(text):
                found = _find_next_to_removed(page_text, form)
                if found:
                    report.add(page_text[found[0]:found[1]])
                    page_text = page_text[:found[0]] + _REMOVED + page_text[found[1]:]
                    pending.remove(text)
                    changed = True
                    break
    return page_text


def _find(page_text: str, form: str) -> tuple[int, int] | None:
    """(start, end) in page_text of the first stretch whose skeleton is the form's, or None."""
    target = _skeleton(form)[0]
    skeleton, where = _skeleton(page_text)
    at = skeleton.find(target) if target else -1
    if at < 0:
        return None
    span = where[at:at + len(target)]  # Not always in order (lam and alef, see _skeleton).
    return min(span), max(span) + 1


_LETTER = re.compile(r"[^\W\d_]")


def _find_next_to_removed(page_text: str, form: str) -> tuple[int, int] | None:
    """Like _find, but only whole words right before or after removed text, with at most
    MAX_GAP_CHARS characters and no letters between them; the gap is taken out too."""
    target = _skeleton(form)[0]
    skeleton, where = _skeleton(page_text)
    start = skeleton.find(target) if target else -1
    while start >= 0:
        span = where[start:start + len(target)]
        first, last = min(span), max(span) + 1
        whole_word = (first == 0 or not _LETTER.match(page_text[first - 1])) and (
            last == len(page_text) or not _LETTER.match(page_text[last]))
        if whole_word:
            removed_before = page_text.rfind(_REMOVED, 0, first)
            if removed_before >= 0 and first - removed_before - 1 <= MAX_GAP_CHARS \
                    and not _LETTER.search(page_text[removed_before + 1:first]):
                return removed_before + 1, last
            removed_after = page_text.find(_REMOVED, last)
            if removed_after >= 0 and removed_after - last <= MAX_GAP_CHARS \
                    and not _LETTER.search(page_text[last:removed_after]):
                return first, removed_after
        start = skeleton.find(target, start + 1)
    return None


def remove_from_page(page_text: str, hidden: "PageHidden | list[str]", report: HiddenTextReport) -> str:
    """Remove the page's hidden text from the page text extracted by pypdf."""
    if not isinstance(hidden, PageHidden):
        hidden = PageHidden(texts=list(hidden))
    for text in hidden.texts:
        # pypdf may break lines or spacing differently than PyMuPDF.
        pattern = r"\s*".join(re.escape(word) for word in text.split())
        new_text, removed = re.subn(pattern, _REMOVED, page_text, count=1)
        if removed:
            page_text = new_text
            report.add(text)
    for line in hidden.arabic_lines:
        page_text = _remove_arabic_line(page_text, line, hidden.visible_arabic, report)
    # A few marks left between two removed pieces (the "-" of a marker split into "CANARY", "-",
    # "7431") go with them.
    page_text = re.sub(rf"{_REMOVED}[^\w{_REMOVED}]{{1,{MAX_GAP_CHARS}}}(?={_REMOVED})", _REMOVED, page_text)
    return re.sub(f"{_REMOVED}+", " ", page_text)


# --- DOCX ---


# Elements of a paragraph whose runs are read as its text, in order: links (DOIs, cross-references,
# the table of contents), tracked insertions and moves, content controls, simple fields (citations
# from reference managers), smart tags and custom XML. paragraph.runs and paragraph.text leave most of
# them out. Deleted and moved-away text (w:del, w:moveFrom) is not read.
_RUN_CONTAINERS = {qn(f"w:{tag}") for tag in ("hyperlink", "ins", "moveTo", "sdt", "sdtContent", "fldSimple",
                                               "smartTag", "customXml")}


def docx_runs(paragraph) -> list[Run]:
    """The runs of a DOCX paragraph in reading order, including those inside links, tracked
    insertions, content controls, simple fields, smart tags and custom XML."""
    def walk(element):
        for child in element.iterchildren():
            if child.tag == qn("w:r"):
                yield Run(child, paragraph)
            elif child.tag in _RUN_CONTAINERS:
                yield from walk(child)
    return list(walk(paragraph._p))


def docx_paragraph_text(paragraph) -> str:
    """All of a DOCX paragraph's text, as docx_runs reads it (the extraction without the filter)."""
    return "".join(run.text for run in docx_runs(paragraph))


def _docx_run_hidden(run, paragraph) -> bool:
    font = run.font
    # Direct formatting first, then the run's character style, then the paragraph's style.
    hidden = font.hidden
    if hidden is None and run.style is not None:
        hidden = run.style.font.hidden
    if hidden is None:
        hidden = paragraph.style.font.hidden
    if hidden:
        return True
    if font.size is not None and font.size.pt < config.HIDDEN_MIN_FONT_SIZE:
        return True
    rgb = font.color.rgb if font.color is not None and font.color.type is not None else None
    if rgb is not None and _near_white(tuple(channel / 255 for channel in rgb)):
        return True
    return False


def visible_docx_paragraph(paragraph, report: HiddenTextReport) -> str:
    """The paragraph's text without its hidden runs (docx_runs, so link text is kept).

    A hidden run of whitespace only (Word's PDF conversion writes 1 pt spaces between words) is
    a single space, not removed, so the words around it do not run together; it is not counted
    as hidden text either."""
    parts = []
    for run in docx_runs(paragraph):
        if not _docx_run_hidden(run, paragraph):
            parts.append(run.text)
        elif run.text and not run.text.strip():
            if parts and not parts[-1][-1:].isspace():
                parts.append(" ")
        else:
            report.add(run.text)
    return "".join(parts)
