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
        "placeholder_no_paper": "أرفق ورقة بحثية بزر 📎 للبدء",
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
        "current_paper": "الورقة الحالية",
        "summarize": "لخّص الورقة",
        "summarizing": "جارٍ التلخيص...",
        "summary_title": "📋 ملخص الورقة",
        "searching": "جارٍ البحث عن الإجابة...",
        "transcribing": "جارٍ تفريغ الصوت والبحث عن الإجابة...",
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
        "page_unknown": "صفحة غير معروفة",
        "no_title": "بدون عنوان",
        "excerpt_unverified": "مقتطف (غير متحقق)",
        "number_not_found": "لم يُعثر على مقتطف يدعم الرقم — راجع النص الكامل",
        "full_text": "عرض النص الكامل",
        "grounding_warning": "أرقام أو أسماء في الإجابة لم ترد في المصادر",
        "citation_mismatch": "يُرجَّح أن المصدر المذكور لا يحتوي على هذه المعلومة",
        "output_warning": "تحتوي الإجابة على رابط أو نص مريب",
        "summary_output_warning": "تحذير في الملخص",
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
        "placeholder_no_paper": "Attach a research paper with 📎 to start",
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
        "current_paper": "Current paper",
        "summarize": "Summarize the paper",
        "summarizing": "Summarizing...",
        "summary_title": "📋 Paper summary",
        "searching": "Looking for the answer...",
        "transcribing": "Transcribing and looking for the answer...",
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
        "page_unknown": "Page unknown",
        "no_title": "Untitled",
        "excerpt_unverified": "Excerpt (not verified)",
        "number_not_found": "No excerpt supporting the number was found — see the full text",
        "full_text": "Show full text",
        "grounding_warning": "Numbers or names in the answer were not found in the sources",
        "citation_mismatch": "The cited source is probably not the one that holds this information",
        "output_warning": "The answer contains a link or suspicious text",
        "summary_output_warning": "Warning in the summary",
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

_ARABIC = re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")


def has_arabic(text: str) -> bool:
    return bool(_ARABIC.search(text))


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
    sentences = [s for paragraph in pdf_paragraphs(source_text) for s in _SENTENCE_SPLIT.split(paragraph) if s.strip()]
    needed = numbers_in(answer)
    with_numbers = [s for s in sentences if needed and needed <= numbers_in(s)]
    candidates = with_numbers or sentences
    if not candidates:
        return ""
    wanted = _words(answer)
    _, best = max(enumerate(candidates), key=lambda item: (len(_words(item[1]) & wanted), -item[0]))
    return clip(best, limit)


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


def source_html(text: str, quote: str | None, answer: str) -> str:
    """The full text of a source as safe HTML paragraphs, for "show full text".

    - The verified quote is highlighted where it is (found with quote_match.quote_span, the
      same rules as the server's check, so PDF artifacts such as "signi ﬁcantly" or
      "migrainepositive" do not stop it). If it cannot be located, nothing is highlighted.
    - With a highlighted quote, its paragraph keeps the normal colour and the others are muted.
    - Numbers of the answer (as number_check.numbers_in reads them, at least two digits) are bold.
    Every piece of text is escaped before any tag is added around it.
    """
    paragraphs = pdf_paragraphs(text)
    joined = "\n\n".join(paragraphs)
    span = quote_span(quote, joined) if quote else None
    wanted = {value for value in numbers_in(answer) if _significant(value)}
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
        html_paragraphs.append(f'<p dir="auto"{css}>{body}</p>')
    return '<div class="paper-text">' + "".join(html_paragraphs) + "</div>"


def paper_html(paragraphs: list[str]) -> str:
    """Paper text as safe HTML paragraphs, each in its own direction."""
    return '<div class="paper-text">' + "".join(f'<p dir="auto">{html.escape(p)}</p>' for p in paragraphs) + "</div>"


def interface_html(text: str) -> str:
    """An interface string as safe HTML: English inside Arabic is isolated, **bold** kept.
    Unlike rich_html it sets no direction, so the line follows the interface's direction
    even when it starts with an English word ("PDF وWord وTXT وصور")."""
    return _inline(text, False)


def isolate(text: str) -> str:
    """A short piece of paper text (a title, a file name) isolated inside an interface line."""
    return f'<bdi dir="auto">{html.escape(text)}</bdi>'
