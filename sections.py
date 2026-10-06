"""Detect a paper's section headings (Abstract, Methods, Results, References, ...).

Headings are found as whole lines matching known names, optionally numbered
("2 Methods", "2.1 Study Design"). Subsections map to their parent section
(e.g. "Search strategy" -> Methods). Guards against false headings seen in real
PDFs:
- a "heading" followed by a line starting in lowercase is the start of a
  sentence or a table header ("Method / for evaluation of 25(OH)D");
- "References" counts only if the text after it looks like a reference list
  (many publication years), since flowcharts and tables can contain the word;
- appendix lines must be real titles ("Appendix A. Supplementary data"), not
  in-text pointers like "Appendix C .".
"""

import re
import unicodedata

FRONT_MATTER = "Front matter"
REFERENCES = "References"

_SECTION_NAMES = {
    "Abstract": ["abstract", "summary"],
    "Introduction": ["introduction", "background"],
    "Methods": [
        "methods", "method", "materials and methods", "methodology", "patients and methods", "study design",
        "search strategy", "eligibility criteria", "data collection", "data extraction", "statistical analysis",
        "statistical analyses",
    ],
    "Results": ["results", "findings"],
    "Discussion": ["discussion"],
    "Limitations": ["limitations", "strengths and limitations", "study limitations"],
    "Conclusion": ["conclusion", "conclusions"],
    REFERENCES: ["references", "bibliography", "literature cited"],
    "Declarations": [
        "acknowledgements", "acknowledgments", "funding", "conflicts of interest", "conflict of interest",
        "declaration of interests", "declaration of interest", "competing interests", "author contributions",
        "data availability", "data availability statement",
    ],
    "Appendix": [
        "appendix a. supplementary data", "appendix. supplementary data", "supplementary data",
        "supplementary material", "supplementary materials", "supplementary information",
    ],
}
_CANONICAL = {name.lower(): section for section, names in _SECTION_NAMES.items() for name in names}
_NUMBERING = re.compile(r"^(?:\d+(?:\.\d+)*\.?|[IVX]+\.)\s+")
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
REFERENCE_CHECK_CHARS = 1_500
MIN_YEARS_IN_REFERENCES = 5


def _heading_name(line: str) -> str | None:
    text = unicodedata.normalize("NFKC", line).strip()
    if not text or len(text) > 60 or text.endswith((",", ":", ";")):
        return None
    text = _NUMBERING.sub("", text)
    return _CANONICAL.get(" ".join(text.split()).lower().rstrip("."))


def _next_line_starts_lowercase(lines: list[str], index: int) -> bool:
    for line in lines[index + 1 :]:
        stripped = line.strip()
        if stripped:
            return stripped[0].islower()
    return False


def detect_sections(text: str) -> list[tuple[int, str]]:
    """Return [(character offset where the section starts, section name)], in order.

    The first entry is always (0, "Front matter") for text before the first heading.
    """
    sections = [(0, FRONT_MATTER)]
    offset = 0
    lines = text.split("\n")
    for index, line in enumerate(lines):
        name = _heading_name(line)
        if name and not _next_line_starts_lowercase(lines, index):
            if name == REFERENCES and len(_YEAR.findall(text[offset : offset + REFERENCE_CHECK_CHARS])) < MIN_YEARS_IN_REFERENCES:
                name = None  # "References" inside a figure or table, not a reference list.
            if name and name != sections[-1][1]:
                sections.append((offset, name))
        offset += len(line) + 1
    return sections


def section_ranges(text: str, sections: list[tuple[int, str]]) -> list[tuple[int, int, str]]:
    """[(start, end, name)] covering the whole text."""
    bounds = [start for start, _ in sections[1:]] + [len(text)]
    return [(start, end, name) for (start, name), end in zip(sections, bounds) if end > start]


# Numbered headings of any name: "2.1 Study design", "II. MATERIALS AND METHOD", "A. Components",
# "A.A.2 Models". Letters and roman numerals need a dot ("A large ..." is a sentence, not a heading).
_NUMBERED_HEADING = re.compile(r"^(?:\d+(?:\.\d+)*\.?|(?:[A-Z]{1,4}\.)+(?:\d+(?:\.\d+)*)?)\s+(?P<title>[A-Z].*)$")
MAX_HEADING_WORDS = 8


def heading_lines(text: str) -> list[tuple[int, str]]:
    """[(character offset of the line, heading as written)] for every heading line, in order.

    Unlike detect_sections, subsections are kept with their own names (e.g. "A.A.2 Models",
    "Search strategy"), so a passage can be labelled with the heading right above it.
    A heading is a known section name (as in detect_sections) or a numbered line of at most
    MAX_HEADING_WORDS words, with no comma, that does not end like a sentence or a table row
    (".", ";", ":", a digit), and is not followed by a line starting in lowercase.
    """
    headings, offset = [], 0
    lines = text.split("\n")
    for index, line in enumerate(lines):
        stripped = " ".join(unicodedata.normalize("NFKC", line).split())
        numbered = _NUMBERED_HEADING.match(stripped)
        is_heading = _heading_name(line) is not None or (
            numbered is not None
            and len(stripped.split()) <= MAX_HEADING_WORDS
            and "," not in stripped  # Reference entries: "21. Elizondo-Montemayor L, Ugalde-Casas PA, ..."
            and not stripped.endswith((".", ";", ":"))
            and not stripped[-1].isdigit()
        )
        if is_heading and not _next_line_starts_lowercase(lines, index):
            headings.append((offset, stripped))
        offset += len(line) + 1
    return headings
