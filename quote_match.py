"""Matching a quote against the paper's text: quote_in_source and quote_span.

Shared by the server (rag.py: verifying quotes, locating their page) and the interface
(ui_text.py: highlighting the quote in the full text), so both use exactly the same rules.
Light on purpose: no ChromaDB or OpenAI imports, so the interface can use it.
"""

import re
import unicodedata

from normalize import normalize_text
from number_check import COLON_DECIMAL, REFERENCE_NUMBER

_WHITESPACE = re.compile(r"\s+")
# Quotation marks the model sometimes wraps a quote in.
_QUOTE_MARKS = "\"'“”‘’«»"
# For comparison only: the model often writes "-" where the paper has "–" (e.g. "1.26-1.59").
_DASHES = str.maketrans({"‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "−": "-"})
# For comparison only: curly and low quotation marks as plain ' and ".
_QUOTE_FOLD = str.maketrans({"‘": "'", "’": "'", "‚": "'", "‛": "'", "“": '"', "”": '"', "„": '"'})
# For comparison only (PDF extraction artifacts): no space before , . ; : ) ] ! ? or after ( [,
# and a repeated quotation mark counts once.
_SPACE_BEFORE_PUNCTUATION = re.compile(r"\s+([,.;:)\]!?])")
_SPACE_AFTER_OPENING = re.compile(r"([(\[])\s+")
_REPEATED_QUOTES = re.compile(r"'{2,}|\"{2,}")
# For comparison only (more PDF artifacts, seen in a Hindawi paper): a space before a
# ligature's letters ("signi ﬁcantly"), a real hyphen joined at a line end
# ("migraine-positive" -> "migrainepositive"), a colon as decimal point ("0 :001"), and
# spaces around comparison signs ("p <0"). Applied to both sides, so they never forgive a word.
_LIGATURE_GAP = re.compile(r"(?<=[^\W\d_]) +(?=f[fil])")
_WORD_HYPHEN = re.compile(r"(?<=[^\W\d_])-(?=[^\W\d_])")
_COMPARATOR_GAP = re.compile(r" ?([<>=≤≥]) ?")


def _comparable(text: str) -> str:
    """Text as quote_in_source compares it. Only form differences are dropped, never words:
    normalize_text and NFKC (ligatures such as "ﬁ"), runs of whitespace (including PDF line
    breaks) as one space, one kind of dash, plain quotation marks, case (casefold), no space
    before closing punctuation or after an opening bracket, a repeated quotation mark once,
    no space before "fi"/"fl"/"ff", no hyphen between two letters, a colon between digits
    as a decimal point, no spaces around < > = ≤ ≥, and reference numbers stuck to "al."
    ("Turer et al.27" -> "Turer et al.")."""
    text = unicodedata.normalize("NFKC", normalize_text(text))
    text = _WHITESPACE.sub(" ", text).strip().translate(_DASHES).translate(_QUOTE_FOLD).casefold()
    # PDF extraction artifacts: "power , using" -> "power, using"; "‘‘meta’’" -> "'meta'".
    text = _SPACE_AFTER_OPENING.sub(r"\1", _SPACE_BEFORE_PUNCTUATION.sub(r"\1", text))
    text = _REPEATED_QUOTES.sub(lambda match: match.group()[0], text)
    text = _WORD_HYPHEN.sub("", _LIGATURE_GAP.sub("", text))
    text = _COMPARATOR_GAP.sub(r"\1", COLON_DECIMAL.sub(".", text))
    return REFERENCE_NUMBER.sub(r"\1", text)


def _quote_target(quote: str) -> str:
    return _comparable(quote.strip().strip(_QUOTE_MARKS)).removesuffix(".").rstrip()


def quote_in_source(quote: str, source_text: str) -> bool:
    """Whether the quote appears word for word in the source, after normalize_text on both.

    Also ignored: quotation marks around the quote, a full stop the model adds to
    close a part of a sentence, and the kind of dash. Nothing else is forgiven.
    """
    target = _quote_target(quote)
    return bool(target) and target in _comparable(source_text)


def quote_span(quote: str, text: str) -> tuple[int, int] | None:
    """(start, end) of the shortest stretch of the raw text that matches the quote (as in quote_in_source).

    Found by binary search on prefixes and suffixes, since normalization changes
    lengths (ligatures, line-end hyphens), so positions cannot be mapped back directly.
    """
    target = _quote_target(quote)
    if not target or target not in _comparable(text):
        return None
    low, high = 0, len(text)  # Shortest prefix that contains the quote.
    while low < high:
        middle = (low + high) // 2
        if target in _comparable(text[:middle]):
            high = middle
        else:
            low = middle + 1
    end = low
    low, high = 0, end  # Latest start that still contains it.
    while low < high:
        middle = (low + high + 1) // 2
        if target in _comparable(text[middle:end]):
            low = middle
        else:
            high = middle - 1
    return low, end
