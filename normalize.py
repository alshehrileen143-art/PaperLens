"""Text normalization shared by chunks and questions (see normalize_text).

Each rule is switched on or off in config.py. Rules read config at call time,
so the evaluation script can compare settings within one run.
"""

import re
import unicodedata
from collections import Counter

import config

# Three or more hyphens, optionally separated by spaces: "0- - -18", "1 --- 5", "3.5.1 ---- Vienna".
_BROKEN_DASH = re.compile(r"[ \t]*-(?:[ \t]*-){2,}[ \t]*")

# A word, a hyphen at the end of a line, and the word's continuation on the next line.
_LINE_END_HYPHEN = re.compile(r"(?P<left>[^\W\d_]+)-[ \t]*\r?\n[ \t]*(?P<right>[^\W\d_]+)")

# Words, including hyphenated compounds written on one line ("cross-sectional").
_WORD = re.compile(r"[^\W\d_]+(?:-[^\W\d_]+)*")

# Prefixes that usually form real compounds, used only when the document gives
# no other evidence either way.
_COMPOUND_PREFIXES = {"anti", "co", "cross", "meta", "multi", "non", "self", "semi", "well"}

RULES = ("nfkc", "dashes", "dehyphenate_joined", "dehyphenate_kept")


def build_vocabulary(text: str) -> set[str]:
    """Lower-cased words of a whole document, used to decide how to undo hyphenation."""
    if config.NORMALIZE_NFKC:
        text = unicodedata.normalize("NFKC", text)
    return {word.lower() for word in _WORD.findall(text)}


def _replace_dash(match: re.Match) -> str:
    text = match.string
    before = text[match.start() - 1] if match.start() > 0 else ""
    after = text[match.end()] if match.end() < len(text) else ""
    # Ranges ("0–18") stay tight; elsewhere the dash separates words.
    return "–" if before.isdigit() and after.isdigit() else " – "


def _dehyphenate(match: re.Match, vocabulary: set[str], stats: Counter) -> str:
    left, right = match["left"], match["right"]
    hyphenated, joined = f"{left}-{right}".lower(), f"{left}{right}".lower()
    keep_hyphen = (
        hyphenated in vocabulary  # The paper writes it hyphenated elsewhere ("cross-sectional").
        or right[0].isupper()  # Hyphenated names ("Elizondo-Montemayor").
        or (joined not in vocabulary and left.lower() in _COMPOUND_PREFIXES)
    )
    if keep_hyphen:
        stats["dehyphenate_kept"] += 1
        return f"{left}-{right}"
    stats["dehyphenate_joined"] += 1
    return left + right


def normalize_text(text: str, vocabulary: set[str] | None = None, stats: Counter | None = None) -> str:
    """Normalize text before embedding and keyword search.

    Applied to both chunks and questions. `vocabulary` (from build_vocabulary on
    the whole document) improves hyphenation decisions for chunks; without it
    the text's own words are used. If `stats` is given, it counts how many times
    each rule changed something.
    """
    stats = stats if stats is not None else Counter()

    if config.NORMALIZE_NFKC:
        stats["nfkc"] += sum(1 for char in text if unicodedata.normalize("NFKC", char) != char)
        text = unicodedata.normalize("NFKC", text)

    if config.NORMALIZE_DASHES:
        text, count = _BROKEN_DASH.subn(_replace_dash, text)
        stats["dashes"] += count

    if config.NORMALIZE_DEHYPHENATE:
        words = vocabulary if vocabulary is not None else build_vocabulary(text)
        text = _LINE_END_HYPHEN.sub(lambda match: _dehyphenate(match, words, stats), text)

    return text
