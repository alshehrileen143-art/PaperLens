"""Number check on quotes: every number in a sentence that cites [n] must be in the quote of source n.

quote_in_source only checks that a quote is copied word for word from its source; it
does not check that the quote supports the sentence citing it. Example: the answer
"20 studies were included [1]" with the verified quote "A total of 24,600 children were
included." This check catches that case through the numbers. Pure functions on text,
so they can also be run offline on saved results.

Known gaps: numbers written as words ("twenty", "عشرين") and claims without numbers are
not checked, and digits inside other words still count ("I2" gives 2).
"""

import re

# Citation markers: "[1]", "[2,3]", "[2, 3]".
_MARKER = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
# Arabic-Indic and Persian digits -> 0-9; Arabic decimal and thousands separators.
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹٫٬·", "01234567890123456789.,.")
# A comma between digits is a thousands separator when exactly 3 digits follow it ("24,600");
# otherwise it is a decimal comma ("0,01").
_THOUSANDS_COMMA = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
_DECIMAL_COMMA = re.compile(r"(?<=\d),(?=\d)")
# Reference numbers stuck to "et al." ("Turer et al.27") are not data.
REFERENCE_NUMBER = re.compile(r"(\bal\.?)\d+")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
# A sentence ends after . ! ? ؟ and a space, but not after common abbreviations ("Turer et al. had").
_SENTENCE_BREAK = re.compile(r"(?<=[.!?؟])(?<!\bal\.)(?<!\be\.g\.)(?<!\bi\.e\.)(?<!\bFig\.)(?<!\bvs\.)\s+")
NUMBER_MISMATCH = "number_mismatch"


def numbers_in(text: str) -> set[str]:
    """The numbers in a text, normalized: "٢٠" -> "20", "24,600" -> "24600", "0,01" -> "0.01",
    "1·41" -> "1.41". Citation markers ("[1]") and reference numbers after "al." ("et al.27")
    are removed first. "1.41" and "141" stay different."""
    text = _MARKER.sub(" ", text.translate(_DIGITS))
    text = REFERENCE_NUMBER.sub(r"\1", text)
    text = _DECIMAL_COMMA.sub(".", _THOUSANDS_COMMA.sub("", text))
    return set(_NUMBER.findall(text))


# Adjacent markers ("[1][2]", "[1] [2]") cite the same words together.
_MARKER_GROUP = re.compile(r"\[\d+(?:\s*,\s*\d+)*\](?:\s*\[\d+(?:\s*,\s*\d+)*\])*")


def _markers(text: str) -> list[int]:
    return list(dict.fromkeys(int(n) for group in _MARKER.findall(text) for n in re.split(r"\s*,\s*", group)))


def _sentences(text: str) -> list[str]:
    """Sentences of the text; a marker standing alone after a sentence ("... included. [1]")
    is joined to the sentence before it."""
    sentences: list[str] = []
    for sentence in _SENTENCE_BREAK.split(text.strip()):
        if not _MARKER.sub("", sentence).strip(" .,;:،؛") and sentences:
            sentences[-1] += " " + sentence
        elif sentence.strip():
            sentences.append(sentence)
    return sentences


def claims(text: str, whole: bool = False) -> list[tuple[str, list[int]]]:
    """[(the words a group of markers cites, the citation numbers of that group)].

    A sentence citing one group is that group's claim, whole. A sentence with several
    groups ("1.41 [1] among 24,600 children [2]") is split after each group, so each group
    gets the words since the previous one; words after the last group go to it. With
    whole=True (markers added at the end of an answer that had none), the whole text is
    one claim of all its markers.
    """
    if whole:
        markers = _markers(text)
        return [(text, markers)] if markers else []
    result = []
    for sentence in _sentences(text):
        groups = list(_MARKER_GROUP.finditer(sentence))
        if len(groups) == 1:
            result.append((sentence, _markers(groups[0].group())))
            continue
        start = 0
        for index, group in enumerate(groups):
            end = len(sentence) if index == len(groups) - 1 else group.end()
            result.append((sentence[start:end], _markers(group.group())))
            start = group.end()
    return result


def cited_sentences(text: str, whole: bool = False) -> dict[int, list[str]]:
    """{citation number: the words citing it} (see claims)."""
    cited: dict[int, list[str]] = {}
    for words, markers in claims(text, whole):
        for n in markers:
            cited.setdefault(n, []).append(words)
    return cited


def missing_numbers(answer: str, details: str | None, quotes: dict[int, str | None],
                    markers_added: bool = False) -> dict[int, list[str]]:
    """{citation number: numbers in the words citing it that its quote does not have}.

    Claims without numbers never fail. With markers_added, the answer's markers were
    added at its end by the fallback, so the whole answer is one claim of all of them.

    - One source cited: every number of the claim must be in its quote.
    - A group of adjacent markers ("[1][2]", "[1][2][3]"): each quote must contain at
      least one of the claim's numbers, otherwise it fails (it supports none of them).
      And the group together must cover every number; if it does not, the quotes that
      contributed no number fail, which are the same quotes as above. So a group whose
      quotes each hold a different part of the numbers passes even if, together, they
      miss one (see clinic.md).
    - A source with no quote is not checked. In a group, its quote-less members could
      hold the numbers, so only the "at least one number" rule is applied to the others.
    """
    found = claims(answer, whole=markers_added) + claims(details or "")
    missing: dict[int, set[str]] = {}
    for words, markers in found:
        needed = numbers_in(words)
        quoted = [n for n in markers if quotes.get(n)]
        if not needed or not quoted:
            continue
        if len(markers) == 1:
            absent = needed - numbers_in(quotes[markers[0]])
            if absent:
                missing.setdefault(markers[0], set()).update(absent)
            continue
        for n in quoted:
            if not needed & numbers_in(quotes[n]):  # Contributes none of the claim's numbers.
                missing.setdefault(n, set()).update(needed)
    return {n: sorted(numbers, key=lambda number: (float(number), number)) for n, numbers in missing.items()}
