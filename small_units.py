"""Split a big chunk into small retrieval units for small-to-big search.

Units are one or two sentences, plus every table row on its own. Table rows in
PDF text often lose their label to the line above ("Khor" / "et al.22
Malaysia Cross-sectional 7-12 <20 ..."), so short label lines are reattached
to the row that follows them.
"""

import re

import config

# A sentence ends with . ! or ? before whitespace and a capital, digit or
# bracket, except after common abbreviations ("et al.", "Fig.", "e.g.").
_SENTENCE_BREAK = re.compile(
    r"(?<=[.!?])(?<!\bal\.)(?<!\bFig\.)(?<!\bfig\.)(?<!\be\.g\.)(?<!\bi\.e\.)(?<!\bvs\.)(?<!\bNo\.)"
    r"\s+(?=[A-Z0-9(\[\"“])"
)
MIN_NUMBERS_IN_ROW = 4  # A line with this many number-bearing tokens is treated as a table row.
MAX_LABEL_WORDS = 3  # Short lines before a row may be its split-off label.


def _numbers(line: str) -> int:
    return sum(1 for token in line.split() if any(char.isdigit() for char in token))


def table_rows(text: str) -> list[str]:
    rows, label = [], []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if _numbers(line) >= MIN_NUMBERS_IN_ROW:
            rows.append(" ".join(label + [line]))
            label = []
        elif len(line.split()) <= MAX_LABEL_WORDS:
            label = (label + [line])[-MAX_LABEL_WORDS:]
        else:
            label = []
    return rows


def _split_long(text: str) -> list[str]:
    """Cut text longer than SMALL_UNIT_MAX_CHARS at spaces."""
    pieces, current = [], ""
    for word in text.split():
        if current and len(current) + 1 + len(word) > config.SMALL_UNIT_MAX_CHARS:
            pieces.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        pieces.append(current)
    return pieces


def sentence_units(text: str) -> list[str]:
    sentences = [s for s in _SENTENCE_BREAK.split(" ".join(text.split())) if s.strip()]
    units, i = [], 0
    while i < len(sentences):
        unit = sentences[i]
        if len(unit) < config.SMALL_UNIT_MIN_CHARS and i + 1 < len(sentences):
            unit = f"{unit} {sentences[i + 1]}"  # One or two sentences per unit.
            i += 1
        units.extend(_split_long(unit))
        i += 1
    return units


def small_units(chunk_text: str) -> list[str]:
    """Distinct small units of a big chunk: table rows first, then sentences."""
    seen, units = set(), []
    for unit in table_rows(chunk_text) + sentence_units(chunk_text):
        key = " ".join(unit.split())
        if key and key not in seen:
            seen.add(key)
            units.append(key)
    return units
