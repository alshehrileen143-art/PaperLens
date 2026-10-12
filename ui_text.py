"""Interface strings (Arabic / English) and display-only text helpers for app.py.

Nothing here changes what the API returns; it only shapes how text is shown.
Every helper that returns HTML escapes its input, so paper text can never add
markup, links or scripts to the page.
"""

import html
import itertools
import re

from number_check import number_spans, numbers_in
from quote_match import quote_span

TEXTS = {
    "ar": {
        "title": "PaperLens",
        "tagline": "مساعد الأوراق البحثية بإجابات موثّقة",
        "language": "اللغة",
        "placeholder": "اكتب سؤالك عن الورقة، أو أرفق ورقة بزر 📎",
        "upload_label": "اسحب الورقة وأفلتها هنا، أو اخترها من جهازك",
        "upload_hint": "PDF وWord وTXT وصور PNG وJPG، حتى {mb} ميغابايت.",
        "card_upload_title": "ارفع ورقة",
        "card_upload_text": "PDF وWord وTXT وصور",
        "card_summary_title": "ملخص منظم",
        "card_summary_text": "الأهداف والنتائج والقيود ومهام الذكاء الاصطناعي",
        "card_ask_title": "اسأل بالكتابة أو الصوت",
        "card_ask_text": "إجابات مختصرة مع استشهادات متحقق منها",
        "server_down": "الخادم لا يعمل على {url}. شغّله من الطرفية، ثم حدّث الصفحة:",
        "timeout": "لم يستجب الخادم خلال {seconds} ثانية. حاول مرة أخرى.",
        "server_error": "خطأ من الخادم ({status}): {detail}",
        "uploading": "جارٍ رفع الورقة وفهرستها...",
        "uploaded": "تم رفع الورقة **{name}** (عدد الأجزاء: {chunks}). اسأل عنها، أو اطلب ملخصًا.",
        "uploaded_duplicate": "هذه الورقة مرفوعة مسبقًا، تم فتحها: **{name}** (عدد الأجزاء: {chunks}). اسأل عنها، أو اطلب ملخصًا.",
        "current_paper": "الورقة الحالية",
        "summarize": "لخّص الورقة",
        "summarizing": "جارٍ التلخيص...",
        "summary_title": "📋 ملخص الورقة",
        "searching": "جارٍ البحث عن الإجابة...",
        "transcribing": "جارٍ تفريغ الصوت...",
        "review_transcript": "🎤 راجع نص السؤال المفرَّغ في خانة الكتابة وعدّله إن لزم، ثم أرسله.",
        "cancel": "إلغاء",
        "need_paper": "أرفق ورقة بحثية أولًا بزر 📎، ثم اسأل عنها.",
        "one_file": "رُفع الملف الأول فقط ({name})، لأن الواجهة تعمل على ورقة واحدة في كل مرة.",
        "voice_caption": "🎤 مفرّغ من الصوت",
        "voice_question": "🎤 سؤال صوتي",
        "ocr_warning": "استُخرج النص بتقنية OCR. راجع الأرقام مع الملف الأصلي.",
        "hidden_removed": "أُزيل نص مخفي من الملف (عدد الأجزاء: {count}) لأنه غير مرئي للقارئ:",
        "more_details": "تفاصيل أكثر",
        "listen": "🔊 استمع إلى الإجابة",
        "speaking": "جارٍ تجهيز الصوت...",
        "speak_failed": "تعذّر تحويل الإجابة إلى صوت: {detail}",
        "sources": "المصادر",
        "page": "صفحة {page}",
        "pages": "صفحات {page}–{page_end}",
        "page_unknown": "صفحة غير معروفة",
        "no_title": "بدون عنوان",
        "excerpt_unverified": "أقرب جملة في القسم — لم يُتحقق من اقتباس النموذج",
        "number_not_found": "لم يُعثر على مقتطف يدعم الرقم — راجع النص الكامل",
        "full_text": "عرض النص الكامل",
        "rest_of_section": "عرض باقي القسم",
        "grounding_warning": "أرقام أو أسماء في الإجابة لم ترد في المصادر",
        "citation_mismatch": "يُرجَّح أن المصدر المذكور لا يحتوي على هذه المعلومة",
        "output_warning": "تحتوي الإجابة على رابط أو نص مريب",
        "summary_output_warning": "تحذير في الملخص",
        "warn_numbers": "الأرقام: {items}",
        "warn_names": "الأسماء: {items}",
        "warn_too_many": "تستشهد بأكثر من {n} مصادر",
        "warn_links": "روابط غير موجودة في الورقة: {items}",
        "warn_markers": "نص يطابق نمطًا مريبًا: {items}",
        "list_separator": "، ",
        "part_separator": "؛ ",
        "objective": "🎯 الهدف",
        "methodology": "🔬 المنهجية",
        "key_findings": "📊 أبرز النتائج",
        "limitations": "⚠️ القيود",
        "use_cases": "💡 حالات الاستخدام",
        "technical_use_cases": "🛠️ حالات الاستخدام التقنية",
        "ai_ml_tasks": "🤖 مهام الذكاء الاصطناعي",
        "data_and_tools": "🗂️ البيانات والأدوات",
        "task_type": "النوع",
        "task_input": "المدخلات",
        "task_output": "المخرجات",
        "datasets": "مجموعات البيانات",
        "variables": "المتغيرات",
        "databases": "قواعد البيانات",
        "tools": "الأدوات",
        "nothing": "لا شيء",
        "not_stated": "لم تذكرها الورقة",
        "suggested_limitations": "💭 قيود محتملة (استنتاج، ليست من الورقة)",
    },
    "en": {
        "title": "PaperLens",
        "tagline": "Research paper assistant with verified citations",
        "language": "Language",
        "placeholder": "Ask about the paper, or attach a paper with 📎",
        "upload_label": "Drag and drop the paper here, or choose it from your device",
        "upload_hint": "PDF, Word, TXT and PNG or JPG images, up to {mb} MB.",
        "card_upload_title": "Upload a paper",
        "card_upload_text": "PDF, Word, TXT and images",
        "card_summary_title": "Structured summary",
        "card_summary_text": "Objectives, findings, limitations and AI tasks",
        "card_ask_title": "Ask by typing or voice",
        "card_ask_text": "Short answers with verified citations",
        "server_down": "The server is not running at {url}. Start it from a terminal, then refresh the page:",
        "timeout": "The server did not respond within {seconds} seconds. Please try again.",
        "server_error": "Server error ({status}): {detail}",
        "uploading": "Uploading and indexing the paper...",
        "uploaded": "Uploaded **{name}** ({chunks} parts). Ask about it, or ask for a summary.",
        "uploaded_duplicate": "This paper was already uploaded; it has been opened: **{name}** ({chunks} parts). Ask about it, or ask for a summary.",
        "current_paper": "Current paper",
        "summarize": "Summarize the paper",
        "summarizing": "Summarizing...",
        "summary_title": "📋 Paper summary",
        "searching": "Looking for the answer...",
        "transcribing": "Transcribing...",
        "review_transcript": "🎤 Review the transcribed question in the message box, edit it if needed, then send it.",
        "cancel": "Cancel",
        "need_paper": "Attach a research paper with 📎 first, then ask about it.",
        "one_file": "Only the first file was uploaded ({name}). The app works on one paper at a time.",
        "voice_caption": "🎤 Transcribed from voice",
        "voice_question": "🎤 Voice question",
        "ocr_warning": "The text was read with OCR. Check numbers against the original file.",
        "hidden_removed": "Hidden text was removed from the file ({count} pieces) because a reader cannot see it:",
        "more_details": "More details",
        "listen": "🔊 Listen to the answer",
        "speaking": "Preparing the audio...",
        "speak_failed": "Could not read the answer aloud: {detail}",
        "sources": "Sources",
        "page": "Page {page}",
        "pages": "Pages {page}–{page_end}",
        "page_unknown": "Page unknown",
        "no_title": "Untitled",
        "excerpt_unverified": "Closest sentence in the section — the model's quote was not verified",
        "number_not_found": "No excerpt supporting the number was found — see the full text",
        "full_text": "Show full text",
        "rest_of_section": "Show the rest of the section",
        "grounding_warning": "Numbers or names in the answer were not found in the sources",
        "citation_mismatch": "The cited source is probably not the one that holds this information",
        "output_warning": "The answer contains a link or suspicious text",
        "summary_output_warning": "Warning in the summary",
        "warn_numbers": "Numbers: {items}",
        "warn_names": "Names: {items}",
        "warn_too_many": "Cites more than {n} sources",
        "warn_links": "Links not found in the paper: {items}",
        "warn_markers": "Text matching a marker pattern: {items}",
        "list_separator": ", ",
        "part_separator": "; ",
        "objective": "🎯 Objective",
        "methodology": "🔬 Methodology",
        "key_findings": "📊 Key findings",
        "limitations": "⚠️ Limitations",
        "use_cases": "💡 Use cases",
        "technical_use_cases": "🛠️ Technical use cases",
        "ai_ml_tasks": "🤖 AI tasks",
        "data_and_tools": "🗂️ Data and tools",
        "task_type": "Type",
        "task_input": "Input",
        "task_output": "Output",
        "datasets": "Datasets",
        "variables": "Variables",
        "databases": "Databases",
        "tools": "Tools",
        "nothing": "None",
        "not_stated": "Not stated in the paper",
        "suggested_limitations": "💭 Possible limitations (inferred, not from the paper)",
    },
}
LANGUAGES = {"ar": "العربية", "en": "English"}

# Colours of the interface's own elements, as CSS variables, for light and dark mode.
# "page-bg" and "page-text" are Streamlit's own background and text (.streamlit/config.toml,
# [theme.light] / [theme.dark]); they are listed here so their contrast is tested with the rest.
# Light mode is the original look; dark mode keeps every text at 4.5:1 or more (WCAG AA).
# The paper types the API extracts (extract.SUPPORTED_EXTENSIONS; a test keeps the two in step),
# for the upload area and 📎. Not imported from extract: the app talks to the API over HTTP only,
# without the extraction libraries.
PAPER_TYPES = ("pdf", "docx", "txt", "png", "jpg", "jpeg")

PALETTES = {
    "light": {
        "page-bg": "#FFFFFF", "page-text": "#1C2533",
        "text-secondary": "#5B6678", "muted-text": "#6B7585", "paper-text": "#3A4556",
        "accent": "#1F3A68", "on-accent": "#FFFFFF",
        "card-bg": "#FFFFFF", "data-bg": "#FFFFFF", "summary-bg": "#F7F9FC", "summary-border": "#C9D3E3",
        "border": "#DDE2EA", "chip-bg": "#EEF2F8", "chip-border": "#D5DDEA", "badge-bg": "#E3EAF4",
        "quote-bg": "#F4F6F9", "highlight-bg": "#E8EDF5", "highlight-text": "#3A4556",
        "table-head-bg": "#E8EDF5", "table-head-text": "#1F3A68",
        "unverified-text": "#4A5568", "unverified-border": "#A3ACBA",
        "warm-bg": "#FFF8EC", "warm-border": "#E2B860", "warm-heading": "#8A5A00", "warm-text": "#5C4A2A",
    },
    "dark": {
        "page-bg": "#0F172A", "page-text": "#E5E7EB",
        "text-secondary": "#9CA3AF", "muted-text": "#9CA3AF", "paper-text": "#E5E7EB",
        "accent": "#8AB4F8", "on-accent": "#0F172A",
        "card-bg": "#1E293B", "data-bg": "#0F172A", "summary-bg": "#1E293B", "summary-border": "#334155",
        "border": "#334155", "chip-bg": "#1E293B", "chip-border": "#334155", "badge-bg": "#1E3A5F",
        "quote-bg": "#0F172A", "highlight-bg": "#2B4C7E", "highlight-text": "#F8FAFC",
        "table-head-bg": "#1E3A5F", "table-head-text": "#E5E7EB",
        "unverified-text": "#CBD5E1", "unverified-border": "#64748B",
        "warm-bg": "#3A2E14", "warm-border": "#8A6A2A", "warm-heading": "#F5D08A", "warm-text": "#F5D08A",
    },
}

# (text, background) pairs that appear in the interface; each must reach CONTRAST_MINIMUM in both modes.
CONTRAST_PAIRS = [
    ("page-text", "page-bg"), ("page-text", "card-bg"), ("page-text", "quote-bg"), ("page-text", "data-bg"),
    ("text-secondary", "page-bg"), ("text-secondary", "card-bg"), ("text-secondary", "summary-bg"),
    ("text-secondary", "data-bg"), ("muted-text", "card-bg"), ("paper-text", "card-bg"),
    ("accent", "page-bg"), ("accent", "card-bg"), ("accent", "summary-bg"), ("accent", "chip-bg"),
    ("accent", "badge-bg"), ("on-accent", "accent"), ("highlight-text", "highlight-bg"),
    ("table-head-text", "table-head-bg"), ("unverified-text", "quote-bg"),
    ("warm-text", "warm-bg"), ("warm-heading", "warm-bg"),
]
CONTRAST_MINIMUM = 4.5


def contrast_ratio(foreground: str, background: str) -> float:
    """WCAG 2.1 contrast ratio of two "#RRGGBB" colours (1 to 21)."""
    def luminance(colour: str) -> float:
        channels = [int(colour.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4)]
        linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
        return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]
    lighter, darker = sorted((luminance(foreground), luminance(background)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


def palette_css() -> str:
    """CSS variables holding both modes' colours as light-dark(light, dark).

    The browser picks one by the element's color-scheme, which Streamlit sets on .stApp from the
    theme it is drawing ("light" when its background is light). So the app's own elements follow
    the mode chosen in Streamlit's menu at once, even when it differs from the device's setting.
    prefers-color-scheme follows the device, and st.context.theme lags behind a change in the menu.

    A browser without light-dark() gets the light colours from the @supports block. Declaring each
    variable twice would not do: a custom property's value is never rejected when it is declared,
    so the light-dark() one would win and fail only where it is used."""
    light, dark = PALETTES["light"], PALETTES["dark"]
    both = ":root { " + " ".join(f"--{key}: light-dark({light[key]}, {dark[key]});" for key in light) + " }"
    fallback = ":root { " + " ".join(f"--{key}: {value};" for key, value in light.items()) + " }"
    return both + "\n@supports not (color: light-dark(#000, #fff)) { " + fallback + " }"

_ARABIC = re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")


def has_arabic(text: str) -> bool:
    """True when the text has an Arabic letter. Arabic punctuation and digits ("؟", "،", "٢")
    do not count, so "what is IBS؟" is English text."""
    return any(char.isalpha() for char in _ARABIC.findall(text))


def shorten_filename(name: str, limit: int = 36) -> str:
    """Cut a long file name in the middle, keeping its start and its extension."""
    if len(name) <= limit:
        return name
    stem, dot, extension = name.rpartition(".")
    if not dot or len(extension) > 5:
        stem, extension = name, ""
    keep = limit - 1 - (len(extension) + 1 if extension else 0)  # Room for "…" and ".extension".
    head, tail = keep * 2 // 3, keep - keep * 2 // 3
    return f"{stem[:head]}…{stem[-tail:] if tail else ''}" + (f".{extension}" if extension else "")


# --- PDF lines -> paragraphs (display only) ---

_SENTENCE_END = re.compile(r"[.!?:؟]['\"”’)]?$")
_LIST_ITEM = re.compile(r"^(?:[•▪◦·\-–*]|\(?\d{1,2}[.)]|\[\d+\])\s")


def pdf_paragraphs(text: str) -> list[str]:
    """Join the hard-wrapped lines of PDF text into natural paragraphs.

    A paragraph ends at a blank line, before a list item, or after a line that
    ends a sentence and is clearly shorter than the text's usual line width.
    A word split by a line-end hyphen ("con-" + "trols") is joined back.
    """
    lines = [" ".join(line.split()) for line in text.splitlines()]
    widths = sorted(len(line) for line in lines if line)
    if not widths:
        return []
    full_width = widths[len(widths) * 3 // 4]  # A typical full line.
    paragraphs, current = [], ""
    for line in lines:
        if not line:
            if current:
                paragraphs.append(current)
            current = ""
            continue
        if current and _LIST_ITEM.match(line):
            paragraphs.append(current)
            current = ""
        if current.endswith("-") and current[-2:-1].isalnum() and line[0].islower():
            current = current[:-1] + line  # "con-" + "trols" -> "controls"
        elif current.endswith("-") and current[-2:-1].isalnum() and line[0].isalnum():
            current += line  # "GPT-" + "2 (February" -> "GPT-2 (February"
        else:
            current = f"{current} {line}" if current else line
        if _SENTENCE_END.search(line) and len(line) < full_width * 0.8:
            paragraphs.append(current)
            current = ""
    if current:
        paragraphs.append(current)
    return paragraphs


# --- Short excerpts for source cards ---

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?؟])\s+")
_WORD = re.compile(r"[^\W_]{3,}|\d+(?:[.,]\d+)*")


def clip(text: str, limit: int = 250) -> str:
    """At most `limit` characters: cut after the last full sentence that fits, else at a word, with "…"."""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    head = text[:limit]
    sentence_end = max(head.rfind(". "), head.rfind("? "), head.rfind("! "), head.rfind("؟ "))
    if sentence_end >= limit // 2:
        return head[:sentence_end + 1]
    return head[:head.rfind(" ")].rstrip(",;: ") + "…" if " " in head else head + "…"


def _words(text: str) -> set[str]:
    return {word.lower() for word in _WORD.findall(text)}


def nearest_sentence(source_text: str, answer: str, limit: int = 250) -> str:
    """The sentence of the source that shares the most words and numbers with the answer
    (the first sentence if none does), clipped to `limit` characters. Display only: unlike a
    quote, it is not checked by the server.

    Sentences that contain every number of the answer come first: if there is one, the
    choice is made among those only (numbers as number_check.numbers_in reads them, so
    "0 :001" is 0.001). Otherwise, all sentences compete as before.
    """
    best, _ = _nearest(source_text, answer)
    return clip(best, limit) if best else ""


def _nearest(source_text: str, answer: str) -> tuple[str, bool]:
    """(the sentence nearest_sentence picks, whether it really matches the answer).

    It matches when it holds every number of the answer, or shares at least one word or
    number with it; otherwise it is only the first sentence, picked by default.
    The answer's numbers are read without number words, as in the number check.
    """
    sentences = [s for paragraph in pdf_paragraphs(source_text) for s in _SENTENCE_SPLIT.split(paragraph) if s.strip()]
    needed = numbers_in(answer, words=False)
    with_numbers = [s for s in sentences if needed and needed <= numbers_in(s)]
    candidates = with_numbers or sentences
    if not candidates:
        return "", False
    wanted = _words(answer)
    _, best = max(enumerate(candidates), key=lambda item: (len(_words(item[1]) & wanted), -item[0]))
    return best, bool(with_numbers) or bool(_words(best) & wanted)


# --- Which sources to show ---

_MARKERS = re.compile(r"(\s*)\[(\d+(?:\s*,\s*\d+)*)\]")


def _renumber(text: str, numbers: dict[int, int]) -> str:
    """Rewrite citation markers with the new numbers; markers of hidden sources are removed."""
    def rewrite(match: re.Match) -> str:
        new = [numbers[int(n)] for n in re.split(r"\s*,\s*", match.group(2)) if int(n) in numbers]
        return match.group(1) + "".join(f"[{n}]" for n in dict.fromkeys(new)) if new else ""
    return _MARKERS.sub(rewrite, text)


def shown_citations(answer: str, details: str | None, sources: list[dict]) -> tuple[str, str | None, list[dict]]:
    """Display only: keep the sources that have a verified quote, numbered 1..n in their order,
    with the markers in the answer and details renumbered to match and those of hidden
    sources removed. If no source has a verified quote, the first one is kept, so at least
    one source always shows (as an unverified excerpt). If the answer is left with no marker
    (all of its markers were for hidden sources), "[1]..[n]" is added at its end, as the API
    does for an answer with no marker. The API response is not changed.
    """
    numbered = [(source.get("number", position), source) for position, source in enumerate(sources, start=1)]
    kept = [item for item in numbered if item[1].get("quote")] or numbered[:1]
    numbers = {old: new for new, (old, _) in enumerate(kept, start=1)}
    shown = [{**source, "number": numbers[old]} for old, source in kept]
    answer = _renumber(answer, numbers)
    if shown and not _MARKERS.search(answer):
        answer = answer.rstrip() + " " + "".join(f"[{n}]" for n in range(1, len(shown) + 1))
    return answer, _renumber(details, numbers) if details else details, shown


# --- Mixed Arabic / English text ---

# A run of Latin text inside Arabic text, with the brackets around it:
# "(relative risk)", "PubMed", "25(OH)D", "95% CI 1.26–1.59".
_LATIN_RUN = re.compile(r"[(\[]?[A-Za-z0-9](?:[^؀-ۿ\n]*[A-Za-z0-9%)\]])?")
_CITATION = re.compile(r"\[(\d+)\]")
_BOLD = re.compile(r"\*\*(.+?)\*\*")


def _escape_with_isolates(text: str) -> str:
    """Escaped text; in Arabic text, each Latin run is isolated (<bdi>) so its brackets
    and word order stay right inside a right-to-left line."""
    if not has_arabic(text):
        return html.escape(text)
    parts, position = [], 0
    for match in _LATIN_RUN.finditer(text):
        parts.append(html.escape(text[position:match.start()]))
        parts.append(f"<bdi>{html.escape(match.group())}</bdi>")
        position = match.end()
    parts.append(html.escape(text[position:]))
    return "".join(parts)


def _inline(text: str, citations: bool) -> str:
    """One line as HTML: citation markers as small numbered badges, **bold** kept."""
    if not citations:
        return _bold(_escape_with_isolates(text))
    pieces, position = [], 0
    for match in _CITATION.finditer(text):
        pieces.append(_bold(_escape_with_isolates(text[position:match.start()].rstrip())))
        pieces.append(f'<sup class="cite-ref">{match.group(1)}</sup>')
        position = match.end()
    pieces.append(_bold(_escape_with_isolates(text[position:])))
    return "".join(pieces)


def _bold(escaped: str) -> str:
    return _BOLD.sub(r"<strong>\1</strong>", escaped)


def rich_html(text: str, citations: bool = False, css_class: str = "rich") -> str:
    """Text as safe HTML paragraphs. Each paragraph takes its own direction from its
    first strong letter (dir="auto"), whatever the interface language is."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n|\n(?=\s*[-•*]\s)", text) if p.strip()]
    body = "".join(f'<p dir="auto">{_inline(p, citations)}</p>' for p in paragraphs)
    return f'<div class="{css_class}">{body}</div>'


def list_html(items: list[str], empty: str = "") -> str:
    """A bulleted list as safe HTML (each item with its own direction), or `empty` when there are no items."""
    if not items:
        return f'<div class="rich"><p dir="auto">{html.escape(empty)}</p></div>' if empty else ""
    return '<div class="rich"><ul>' + "".join(f'<li dir="auto">{_inline(item, False)}</li>' for item in items) + "</ul></div>"


def table_html(headers: list[str], rows: list[list[str]]) -> str:
    """A small table as safe HTML; every cell takes its own direction."""
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f'<td dir="auto">{_inline(cell, False)}</td>' for cell in row) + "</tr>"
                   for row in rows)
    return f'<table class="mini-table"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def _significant(value: str) -> bool:
    """Numbers worth highlighting: at least two digits ("1" and "2" alone appear everywhere)."""
    return sum(character.isdigit() for character in value) >= 2


# A long section is split: its key paragraph first, the rest folded (see source_html).
SPLIT_MIN_PARAGRAPHS = 3
SPLIT_MIN_CHARS = 1_200


def source_html(text: str, quote: str | None, answer: str, rest_label: str = "") -> str:
    """The full text of a source as safe HTML paragraphs, for "show full text".

    - The verified quote is highlighted where it is (found with quote_match.quote_span, the
      same rules as the server's check, so PDF artifacts such as "signi ﬁcantly" or
      "migrainepositive" do not stop it). If it cannot be located, nothing is highlighted.
    - With a highlighted quote, its paragraph keeps the normal colour and the others are muted.
    - Numbers of the answer (as number_check.numbers_in reads them, at least two digits) are bold.
    - A long section (SPLIT_MIN_PARAGRAPHS paragraphs or SPLIT_MIN_CHARS characters) is split when
      `rest_label` is given: the paragraph(s) holding the quote, or else the sentence
      nearest_sentence picks, come first, and the others follow, in their order, inside a
      <details> folded under `rest_label`. With no quote located and no sentence that really
      matches the answer, the whole section shows as before.
    Every piece of text is escaped before any tag is added around it.
    """
    paragraphs = pdf_paragraphs(text)
    joined = "\n\n".join(paragraphs)
    span = quote_span(quote, joined) if quote else None
    wanted = {value for value in numbers_in(answer, words=False) if _significant(value)}
    in_quote = [False] * len(joined)
    in_number = [False] * len(joined)
    if span:
        in_quote[span[0]:span[1]] = [True] * (span[1] - span[0])
    for start, end, value in number_spans(joined):
        if value in wanted:
            in_number[start:end] = [True] * (end - start)

    html_paragraphs, offset = [], 0
    for paragraph in paragraphs:
        start, end = offset, offset + len(paragraph)
        offset = end + 2
        pieces, run, state = [], "", None
        for position in range(start, end):
            flags = (in_quote[position], in_number[position])
            if flags != state and run:
                pieces.append((state, run))
                run = ""
            state = flags
            run += joined[position]
        if run:
            pieces.append((state, run))
        # One highlight per stretch of the quote, with the bold numbers inside it.
        body = ""
        for quoted, group in itertools.groupby(pieces, key=lambda piece: piece[0][0]):
            inner = "".join(f'<span class="answer-number">{html.escape(piece)}</span>' if bold else html.escape(piece)
                            for (_, bold), piece in group)
            body += f'<span class="quote-mark">{inner}</span>' if quoted else inner
        muted = span is not None and not any(in_quote[start:end])
        css = ' class="muted"' if muted else ""
        html_paragraphs.append((any(in_quote[start:end]), f'<p dir="auto"{css}>{body}</p>'))

    focus = {index for index, (quoted, _) in enumerate(html_paragraphs) if quoted}
    if not focus:
        sentence, matched = _nearest(text, answer)
        focus = {next(i for i, paragraph in enumerate(paragraphs) if sentence in paragraph)} if matched else set()
    long = len(paragraphs) >= SPLIT_MIN_PARAGRAPHS or len(joined) >= SPLIT_MIN_CHARS
    if not (rest_label and focus and long and len(focus) < len(paragraphs)):
        return '<div class="paper-text">' + "".join(markup for _, markup in html_paragraphs) + "</div>"
    first = "".join(markup for index, (_, markup) in enumerate(html_paragraphs) if index in focus)
    rest = "".join(markup for index, (_, markup) in enumerate(html_paragraphs) if index not in focus)
    return (f'<div class="paper-text">{first}<details class="rest-of-section">'
            f'<summary>{html.escape(rest_label)}</summary>{rest}</details></div>')


def paper_html(paragraphs: list[str]) -> str:
    """Paper text as safe HTML paragraphs, each in its own direction."""
    return '<div class="paper-text">' + "".join(f'<p dir="auto">{html.escape(p)}</p>' for p in paragraphs) + "</div>"


def interface_html(text: str) -> str:
    """An interface string as safe HTML: English inside Arabic is isolated, **bold** kept.
    Unlike rich_html it sets no direction, so the line follows the interface's direction
    even when it starts with an English word ("PDF وWord وTXT وصور")."""
    return _inline(text, False)


# Characters Streamlit's markdown would read as formatting ("*", "_", "$" for LaTeX, ...).
_MARKDOWN_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!|<>$~])")
# First-strong isolate ... pop directional isolate: <bdi> for plain text (st.warning shows no HTML).
_FSI, _PDI = "⁨", "⁩"


def warning_markdown(label: str, items: dict | None, texts: dict[str, str]) -> str:
    """A warning in the interface's language: "**label:** Numbers: 9.2; Names: X".

    items is the API's *_items field (numbers, names, links, markers found by the checks), so
    only the values come from the answer; every word is from texts. Each value is isolated and
    escaped, so a link or a number keeps its order and shape inside Arabic. With no items (an
    older server), the label alone, without the API's English sentence."""
    def values(key: str) -> str:
        return texts["list_separator"].join(_FSI + _MARKDOWN_SPECIAL.sub(r"\\\1", value) + _PDI
                                            for value in items[key])
    parts = []
    for key in ("numbers", "names", "links", "markers"):
        if (items or {}).get(key):
            parts.append(texts[f"warn_{key}"].format(items=values(key)))
    if (items or {}).get("too_many_citations"):
        parts.append(texts["warn_too_many"].format(n=items.get("max_cited_sources", "")))
    return f"**{label}**" + (": " + texts["part_separator"].join(parts) if parts else "")


def isolate(text: str) -> str:
    """A short piece of paper text (a title, a file name) isolated inside an interface line."""
    return f'<bdi dir="auto">{html.escape(text)}</bdi>'
