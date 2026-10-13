"""How well the hidden-text filter removes planted instructions, and nothing else. Local only, no API.

For each prompt-injection test file (eval/make_injection_samples.py, or the hand-made ones), the text
is extracted without and with the filter (extract.extract_text), and the report shows:
- marker: whether CANARY-7431 is left;
- instruction: how many runs of 3 consecutive words of the instruction are left (NFKC, so Arabic
  presentation forms count as letters);
- removed: characters removed, against the instruction's length (much more means visible text went too).
Clean papers (no hidden text) show false positives: anything removed from them is a mistake.

--compare-with <git revision> runs the same on hidden_text.py of that revision (e.g. HEAD), for a
before and after.

Run (PowerShell, from the project folder):
    .\\venv\\Scripts\\python.exe eval\\check_hidden_filter.py --files samples\\injection --clean samples\\1-s2.0-S0021755720302072-main.pdf
"""

import argparse
import importlib
import subprocess
import sys
import tempfile
import unicodedata
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

import config  # noqa: E402
import extract  # noqa: E402
import hidden_text  # noqa: E402
from make_injection_samples import INSTRUCTIONS, MARKER  # noqa: E402


def flat(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split())


def extracted(path: Path, hidden_filter: bool) -> extract.ExtractionResult:
    with patch.object(config, "HIDDEN_TEXT_FILTER", hidden_filter):
        return extract.extract_text(path)


def score(path: Path) -> dict:
    raw, filtered = extracted(path, False), extracted(path, True)
    row = {"file": path.name, "removed": len(raw.text) - len(filtered.text), "pieces": filtered.hidden_removed["count"]}
    parts = path.stem.split("_")
    if len(parts) == 3 and (parts[1], parts[2]) in INSTRUCTIONS:
        instruction = INSTRUCTIONS[(parts[1], parts[2])]
        words, text = instruction.split(), flat(filtered.text)
        windows = [" ".join(words[i:i + 3]) for i in range(len(words) - 2)]
        row.update(marker_left=MARKER in text.replace(" ", ""), instruction_runs_left=sum(w in text for w in windows),
                   instruction_runs=len(windows), instruction_chars=len(instruction))
    return row


def run(files: list[Path], clean: list[Path]) -> list[dict]:
    return [score(path) for path in files + clean]


def with_revision(revision: str, files: list[Path], clean: list[Path]) -> list[dict]:
    """The same, with hidden_text.py as it was at a git revision."""
    source = subprocess.run(["git", "show", f"{revision}:hidden_text.py"], cwd=ROOT, capture_output=True,
                            text=True, encoding="utf-8", check=True).stdout
    with tempfile.TemporaryDirectory() as folder:
        (Path(folder) / "hidden_text.py").write_text(source, encoding="utf-8")
        old = importlib.util.spec_from_file_location("hidden_text_old", Path(folder) / "hidden_text.py")
        module = importlib.util.module_from_spec(old)
        old.loader.exec_module(module)
        with patch.object(extract, "hidden_pdf_spans", module.hidden_pdf_spans), \
                patch.object(extract, "remove_from_page", module.remove_from_page):
            return run(files, clean)


def table(rows: list[dict], before: list[dict] | None) -> str:
    lines = ["| File | Marker left | Instruction left (3-word runs) | Removed chars (instruction) | Pieces |",
             "|---|---|---|---|---|"]
    for n, row in enumerate(rows):
        old = before[n] if before else None

        def cell(key, fmt):
            now = fmt(row)
            return f"{fmt(old)} → {now}" if old is not None and fmt(old) != now else now
        if "instruction_runs" in row:
            lines.append(f"| {row['file']} | {cell('marker', lambda r: 'yes' if r['marker_left'] else 'no')} | "
                         f"{cell('runs', lambda r: str(r['instruction_runs_left']) + '/' + str(r['instruction_runs']))} | "
                         f"{cell('removed', lambda r: str(r['removed']))} ({row['instruction_chars']}) | "
                         f"{cell('pieces', lambda r: str(r['pieces']))} |")
        else:
            lines.append(f"| {row['file']} (clean) | — | — | {cell('removed', lambda r: str(r['removed']))} | "
                         f"{cell('pieces', lambda r: str(r['pieces']))} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--files", type=Path, nargs="+", default=[], help="Injection test files or folders.")
    parser.add_argument("--clean", type=Path, nargs="*", default=[], help="Papers with no hidden text.")
    parser.add_argument("--compare-with", help="A git revision whose hidden_text.py is the 'before'.")
    args = parser.parse_args()
    files = sorted(p for item in args.files for p in (item.iterdir() if item.is_dir() else [item])
                   if p.suffix.lower() in (".pdf", ".docx"))
    before = with_revision(args.compare_with, files, args.clean) if args.compare_with else None
    print(table(run(files, args.clean), before))


if __name__ == "__main__":
    main()
