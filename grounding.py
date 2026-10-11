"""Check that an answer's numbers and names actually appear in the sources it cited.

The model can cite a section of the paper while taking a fact from somewhere
else (in tests: a funding source from another paper, cited as the vitamin D
paper). This check is deterministic and needs no API call: every number and
proper name in the answer must occur in the cited sources (both normalized
with normalize_text). Answers that cite an unusually large number of sources
are also flagged, since that pattern came with guessing in tests.
"""

import re
from dataclasses import dataclass, field

import config
from normalize import normalize_text
from number_check import COLON_DECIMAL

# Arabic-Indic and Persian digits, and the Arabic decimal/thousands separators.
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹٫٬", "01234567890123456789.,")
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
_THOUSANDS = re.compile(r"^\d{1,3}(?:,\d{3})+(?:\.\d+)?$")
# Study names such as "Turer et al." and capitalised Latin-script words (names,
# acronyms like PROSPERO, LILACS). Sentence-initial words are skipped below.
_ET_AL = re.compile(r"\b([A-Z][\w'’-]+)\s+et\s+al\b")
_CAPITALISED = re.compile(r"(?<![\w-])[A-Z][A-Za-z'’-]{2,}")
_SENTENCE_START = re.compile(r"(?:^|[.!?:;]\s+|\n\s*)$")
_WORD = re.compile(r"[A-Za-z][A-Za-z'’-]+")

# Capitalised words that are ordinary English rather than names.
_COMMON_WORDS = {
    "the", "this", "these", "that", "there", "however", "according", "based", "while", "although",
    "additionally", "furthermore", "moreover", "also", "both", "each", "all", "other", "overall",
    "sources", "source", "paper", "study", "studies", "table", "figure", "fig", "results", "methods",
    "discussion", "conclusion", "introduction", "limitations", "abstract", "et", "al",
}


@dataclass
class GroundingResult:
    # Not found where the model could have read them -> grounding warning.
    unsupported_numbers: list[str] = field(default_factory=list)
    unsupported_names: list[str] = field(default_factory=list)
    too_many_citations: bool = False
    # RAG mode only: in the context the model was given, but not in or next to
    # the sources it cited -> citation_mismatch (a lighter signal).
    mismatched_numbers: list[str] = field(default_factory=list)
    mismatched_names: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (self.unsupported_numbers or self.unsupported_names or self.too_many_citations)

    @property
    def citation_mismatch(self) -> bool:
        return bool(self.mismatched_numbers or self.mismatched_names)

    def describe(self) -> str:
        parts = []
        if self.unsupported_numbers:
            parts.append("numbers not found in the sources: " + ", ".join(self.unsupported_numbers))
        if self.unsupported_names:
            parts.append("names not found in the sources: " + ", ".join(self.unsupported_names))
        if self.too_many_citations:
            parts.append(f"cites more than {config.MAX_CITED_SOURCES} sources")
        return "; ".join(parts)

    def items(self) -> dict | None:
        """The grounding warning as data, so an interface can write it in its own language
        (describe() stays in English for the API and the evaluation scripts). None when ok."""
        if self.ok:
            return None
        return {"numbers": list(self.unsupported_numbers), "names": list(self.unsupported_names),
                "too_many_citations": self.too_many_citations, "max_cited_sources": config.MAX_CITED_SOURCES}

    def mismatch_items(self) -> dict | None:
        """The citation mismatch as data (see items). None when there is none."""
        if not self.citation_mismatch:
            return None
        return {"numbers": list(self.mismatched_numbers), "names": list(self.mismatched_names)}

    def describe_mismatch(self) -> str:
        parts = []
        if self.mismatched_numbers:
            parts.append("numbers taken from sources it did not cite: " + ", ".join(self.mismatched_numbers))
        if self.mismatched_names:
            parts.append("names taken from sources it did not cite: " + ", ".join(self.mismatched_names))
        return "; ".join(parts)


def _canonical_number(token: str) -> str:
    """'24,600' -> '24600'; '0,01' (decimal comma) -> '0.01'; '1.41' stays."""
    if _THOUSANDS.match(token):
        return token.replace(",", "")
    return token.replace(",", ".")


def _numbers(text: str) -> set[str]:
    """Numbers in the text. A colon between digits ("p <0 :001") is read as a decimal point, in
    the answer and the sources alike (number_check.COLON_DECIMAL). Spaces around < > = need no
    rule here: numbers are taken as separate tokens."""
    text = COLON_DECIMAL.sub(".", text.translate(_DIGITS))
    return {_canonical_number(token) for token in _NUMBER.findall(text)}


def _names(text: str) -> set[str]:
    names = {match.group(1) for match in _ET_AL.finditer(text)}
    for match in _CAPITALISED.finditer(text):
        word = match.group(0).strip("'’-")
        if word.lower() in _COMMON_WORDS:
            continue
        if _SENTENCE_START.search(text[: match.start()]):
            continue  # Capitalised only because it starts a sentence.
        names.add(word)
    return names


def _missing(answer: str, question: str, text: str) -> tuple[list[str], list[str]]:
    """Numbers and names in the answer that occur in neither `text` nor the question."""
    numbers = sorted(_numbers(answer) - _numbers(text) - _numbers(question), key=lambda n: answer.find(n))
    words = {word.lower() for word in _WORD.findall(text + "\n" + question)}
    names = sorted((name for name in _names(answer) if name.lower() not in words), key=lambda name: answer.find(name))
    return numbers, names


def check_grounding(
    question: str,
    answer: str,
    evidence_texts: list[str],
    cited_count: int,
    context_texts: list[str] | None = None,
) -> GroundingResult:
    """Compare the answer's numbers and names with the evidence (and the question).

    `evidence_texts` are the cited sources plus their immediate neighbours: the
    model often cites the section next to the one holding the fact, because a
    section boundary can fall inside the passage. `cited_count` is how many
    sources the answer actually cited.

    With `context_texts` (RAG mode: everything sent to the model), a fact
    missing from the evidence is split in two: absent from the whole context
    -> unsupported (grounding warning); present in the context but not near
    the citation -> mismatched (citation_mismatch). Without it (full mode),
    every fact missing from the evidence is unsupported.
    """
    answer = normalize_text(answer)
    question = normalize_text(question)
    numbers, names = _missing(answer, question, normalize_text("\n".join(evidence_texts)))

    mismatched_numbers, mismatched_names = [], []
    if context_texts is not None and (numbers or names):
        absent_numbers, absent_names = _missing(answer, question, normalize_text("\n".join(context_texts)))
        mismatched_numbers = [n for n in numbers if n not in absent_numbers]
        mismatched_names = [n for n in names if n not in absent_names]
        numbers, names = absent_numbers, absent_names

    return GroundingResult(
        unsupported_numbers=numbers,
        unsupported_names=names,
        too_many_citations=cited_count > config.MAX_CITED_SOURCES,
        mismatched_numbers=mismatched_numbers,
        mismatched_names=mismatched_names,
    )
