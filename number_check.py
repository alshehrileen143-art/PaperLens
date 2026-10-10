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
# A colon between digits, used by some PDFs as the decimal point ("p <0 :001", "2:49"). Read as
# "." everywhere it is compared (the answer and the paper alike), so a real time or ratio
# ("10:30", "1:2") still matches itself. A space may come before the colon ("0 :001") but not
# after it, so a label followed by a number ("Table 1: 20 patients") is not read as 1.20.
COLON_DECIMAL = re.compile(r"(?<=\d) ?:(?=\d)")
# Reference numbers stuck to "et al." ("Turer et al.27") are not data.
REFERENCE_NUMBER = re.compile(r"(\bal\.?)\d+")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
# A sentence ends after . ! ? ؟ and a space, but not after common abbreviations ("Turer et al. had").
_SENTENCE_BREAK = re.compile(r"(?<=[.!?؟])(?<!\bal\.)(?<!\be\.g\.)(?<!\bi\.e\.)(?<!\bFig\.)(?<!\bvs\.)\s+")
NUMBER_MISMATCH = "number_mismatch"


# English numbers written in words ("Two thousand eight hundred two" -> 2802).
_UNITS = {word: value for value, word in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen".split())}
_TENS = {word: value for word, value in zip("twenty thirty forty fifty sixty seventy eighty ninety".split(),
                                            range(20, 100, 10))}
_SCALES = {"hundred": 100, "thousand": 1_000, "million": 1_000_000}
_NUMBER_WORDS = "|".join(sorted([*_UNITS, *_TENS, *_SCALES], key=len, reverse=True))
# A run of number words joined by spaces or hyphens, with "and" allowed only after a scale
# ("one hundred and five").
_NUMBER_WORD_RUN = re.compile(
    rf"\b(?:{_NUMBER_WORDS})\b"
    rf"(?:(?:[\s-]+|(?:(?<=hundred)|(?<=thousand)|(?<=million))\s+and\s+)(?:{_NUMBER_WORDS})\b)*",
    re.IGNORECASE)
# A digit run between two letters of the same word ("R4DQ") is part of a name, not a number.
# Digits at the end of a word ("I2", "PP-OCRv6") still count.
_DIGITS_INSIDE_WORD = re.compile(r"(?<=[^\W\d_])\d+(?=[^\W\d_])")


def _words_to_number(words: str) -> str | None:
    """"seven hundred sixty-seven" -> "767". None for a lone "one" ("one of the studies")."""
    tokens = [token for token in re.split(r"[\s-]+", words.lower()) if token and token != "and"]
    if tokens == ["one"]:
        return None
    total = current = 0
    for token in tokens:
        if token in _UNITS:
            current += _UNITS[token]
        elif token in _TENS:
            current += _TENS[token]
        elif token == "hundred":
            current = (current or 1) * 100
        else:  # thousand, million
            total += (current or 1) * _SCALES[token]
            current = 0
    return str(total + current)


def _number_words_to_digits(text: str) -> str:
    def convert(match: re.Match) -> str:
        number = _words_to_number(match.group())
        return match.group() if number is None else number
    return _NUMBER_WORD_RUN.sub(convert, text)


def numbers_in(text: str, words: bool = True) -> set[str]:
    """The numbers in a text, normalized: "٢٠" -> "20", "24,600" -> "24600", "0,01" -> "0.01",
    "1·41" -> "1.41", "0 :001" -> "0.001", and with words=True "seven hundred sixty-seven" -> "767".
    Citation markers ("[1]"), reference numbers after "al." ("et al.27") and digits inside a word
    ("R4DQ") are removed first. "1.41" and "141" stay different.

    The number check reads number words in quotes only (words=False for the answer): an answer's
    "three months" is prose, not a figure the quote must hold, while a quote's "Two thousand eight
    hundred two" must still match an answer's 2802."""
    text = _MARKER.sub(" ", text.translate(_DIGITS))
    text = COLON_DECIMAL.sub(".", REFERENCE_NUMBER.sub(r"\1", text))
    text = _DIGITS_INSIDE_WORD.sub("", _number_words_to_digits(text) if words else text)
    text = _DECIMAL_COMMA.sub(".", _THOUSANDS_COMMA.sub("", text))
    return set(_NUMBER.findall(text))


# A number as written in the raw text, before normalization: "24,600", "0 :001", "1:6", "1·41".
_RAW_NUMBER = re.compile(r"\d+(?:(?: ?:|[.,])\d+)*")


def number_spans(text: str) -> list[tuple[int, int, str]]:
    """[(start, end, value)] of every number in the raw text, valued as numbers_in reads it, so
    "0 :001" is 0.001 and "Two thousand eight hundred two" is 2802. Display only (highlighting
    the answer's numbers in the full text). Digits inside a word ("R4DQ") and reference numbers
    after "al." are skipped, as in numbers_in."""
    translated = text.translate(_DIGITS)  # One character for one, so positions stay the same.
    spans = []
    for match in _NUMBER_WORD_RUN.finditer(translated):
        value = _words_to_number(match.group())
        if value is not None:
            spans.append((match.start(), match.end(), value))
    for match in _RAW_NUMBER.finditer(translated):
        start, end = match.span()
        before, after = translated[max(0, start - 3):start], translated[end:end + 1]
        if before[-1:].isalpha() and after.isalpha():
            continue  # "R4DQ"
        if re.search(r"\bal\.?$", before, re.IGNORECASE) or before.lower().endswith("al"):
            continue  # "et al.27"
        values = numbers_in(match.group())
        if len(values) == 1:
            spans.append((start, end, values.pop()))
    return sorted(spans)


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
        needed = numbers_in(words, words=False)  # Number words are read in quotes only.
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
