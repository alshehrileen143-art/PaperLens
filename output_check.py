"""Check model output (answers, summaries) for links and marker text.

A warning is produced when the output contains:
- a link (http/https or www.) that does not appear in the paper's text, or
- text matching one of config.MARKER_PATTERNS.
Deterministic; no API calls.
"""

import re
from dataclasses import dataclass, field

import config

_URL = re.compile(r"(?:https?://|www\.)[^\s<>\"'()\[\]{}]+", re.IGNORECASE)
_TRAILING = ".,;:!?…»”’"


@dataclass
class OutputCheck:
    unknown_links: list[str] = field(default_factory=list)
    markers: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (self.unknown_links or self.markers)

    def describe(self) -> str | None:
        parts = []
        if self.unknown_links:
            parts.append("links not found in the paper: " + ", ".join(self.unknown_links))
        if self.markers:
            parts.append("text matching a marker pattern: " + ", ".join(self.markers))
        return "; ".join(parts) or None


def _canonical_link(link: str) -> str:
    """Comparable form: no scheme, no "www.", no trailing slash or punctuation, lower case."""
    link = link.rstrip(_TRAILING).lower()
    link = re.sub(r"^https?://", "", link)
    link = re.sub(r"^www\.", "", link)
    return link.rstrip("/")


def links(text: str) -> list[str]:
    return [match.rstrip(_TRAILING) for match in _URL.findall(text)]


def check_output(output: str, paper_text: str) -> OutputCheck:
    # PDF text often breaks a link across lines ("doi.org/10.1016/j.\njped..."),
    # so the paper is compared with all whitespace removed.
    paper = re.sub(r"\s+", "", paper_text).lower()
    unknown = []
    for link in links(output):
        canonical = _canonical_link(link)
        if canonical and canonical not in paper and link not in unknown:
            unknown.append(link)

    markers = []
    for pattern in config.MARKER_PATTERNS:
        for match in re.finditer(pattern, output, re.IGNORECASE):
            if match.group(0) not in markers:
                markers.append(match.group(0))
    return OutputCheck(unknown_links=unknown, markers=markers)


def summary_text(summary) -> str:
    """All strings inside a summary (nested dicts and lists), joined for checking."""
    if isinstance(summary, str):
        return summary
    if isinstance(summary, dict):
        return "\n".join(summary_text(value) for value in summary.values())
    if isinstance(summary, list):
        return "\n".join(summary_text(value) for value in summary)
    return ""
