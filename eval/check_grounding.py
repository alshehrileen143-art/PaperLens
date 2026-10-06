"""Evaluate the grounding check (grounding.py) on saved results. No API calls.

For every answer with found=True, the cited sources are rebuilt exactly as the
model saw them, the check is applied, and its flag is compared with the judge's
verdict:
  caught        flagged and the judge said incorrect   (good)
  false alarm   flagged but the judge said correct     (bad)
  missed        not flagged but the judge said incorrect
It also simulates both actions: "warn" keeps every answer (accuracy unchanged,
warnings added); "reject" turns flagged answers into refusals, which is right
for questions that should be refused and wrong otherwise.

Usage (from the project root):
    venv\\Scripts\\python eval\\check_grounding.py [--show-flags]
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "eval"))

import config  # noqa: E402
import rag  # noqa: E402
import run_needle  # noqa: E402
from config import FULL_CONTEXT_SECTION_CHARS  # noqa: E402
from extract import extract_text  # noqa: E402
from grounding import check_grounding  # noqa: E402

NEEDLE_RESULTS = PROJECT_ROOT / "eval" / "results" / "needle_20261003_190613.json"
RUN2_RESULTS = PROJECT_ROOT / "eval" / "results" / "run2_approved_judge_rules.json"
NEIGHBORS = 1  # Set from --neighbors.


def _needle_sources(rows: list[dict], dataset: dict) -> dict:
    """Rebuild each needle haystack's sections (full) and chunks (hybrid)."""
    needle = extract_text(PROJECT_ROOT / dataset["paper"]).text.strip()
    fillers, _ = run_needle.load_fillers(PROJECT_ROOT / "samples" / "filler")
    pool = "\n\n".join(fillers)
    units = {}
    for length, position, tokens in {(r["length_target"], r["position"], r["length_tokens"]) for r in rows}:
        haystack = run_needle.build_haystack(needle, pool, length, position)
        if haystack["tokens"] != tokens:
            sys.exit(f"Haystack {length}/{position} rebuilt as {haystack['tokens']} tokens, not {tokens}: filler changed?")
        units[(length, position, "full")] = rag.chunk_text(haystack["text"], size=FULL_CONTEXT_SECTION_CHARS, overlap=0)
        units[(length, position, "hybrid")] = rag.chunk_text(haystack["text"])
    return units


def _run2_sources(dataset: dict) -> dict:
    paper = extract_text(PROJECT_ROOT / dataset["paper"])
    chunks = rag.chunk_text(paper.text, paper.pages)
    whole = [rag.Chunk(index=0, text=paper.text)]  # The full method sent the paper as one source.
    return {"full": whole, "dense": chunks, "hybrid": chunks}


def _evaluate(name: str, rows: list[dict], sources_for, show_flags: bool) -> list[dict]:
    results = []
    for row in rows:
        if not row["found"]:
            continue
        units = sources_for(row)
        wanted = {n for i in row["cited_chunks"] for n in range(i - NEIGHBORS, i + NEIGHBORS + 1)}
        evidence = [units[n].text for n in sorted(wanted) if 0 <= n < len(units)]
        grounding = check_grounding(row["question"], row["answer"], evidence, len(row["cited_chunks"]))
        results.append({**row, "dataset": name, "flagged": not grounding.ok, "why": grounding.describe()})
    return results


def _report(title: str, results: list[dict], all_rows: list[dict]) -> None:
    flagged = [r for r in results if r["flagged"]]
    caught = [r for r in flagged if r["verdict"] != "correct"]
    false_alarms = [r for r in flagged if r["verdict"] == "correct"]
    missed = [r for r in results if not r["flagged"] and r["verdict"] != "correct"]
    incorrect_found = len(caught) + len(missed)
    print(f"\n=== {title} ===")
    print(f"  answers with found=True: {len(results)}  (incorrect among them: {incorrect_found})")
    print(f"  caught: {len(caught)}/{incorrect_found}   false alarms: {len(false_alarms)}/{len(results) - incorrect_found}   "
          f"missed: {len(missed)}")

    # Accuracy over ALL answers (found=False rows unchanged) under each action.
    correct_now = sum(r["verdict"] == "correct" for r in all_rows)
    by_key = {(r.get("length_target"), r["method"], r["id"]): r for r in flagged}
    after_reject = 0
    for row in all_rows:
        f = by_key.get((row.get("length_target"), row["method"], row["id"]))
        if f is None:
            after_reject += row["verdict"] == "correct"
        else:
            after_reject += row["expected_behavior"] == "refuse"  # A refusal is right only for these.
    print(f"  accuracy: now / with 'warn' {correct_now}/{len(all_rows)}   with 'reject' {after_reject}/{len(all_rows)}")
    reasons = Counter()
    for r in flagged:
        reasons.update(part.split(":")[0] for part in r["why"].split("; "))
    print(f"  flag reasons: {dict(reasons)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--show-flags", action="store_true", help="List every flagged and every missed answer.")
    parser.add_argument("--neighbors", type=int, default=config.GROUNDING_NEIGHBORS,
                        help="Sources on each side of a cited one that also count as evidence.")
    args = parser.parse_args()
    global NEIGHBORS
    NEIGHBORS = args.neighbors
    print(f"Evidence = cited sources +/- {NEIGHBORS} neighbour(s); suspicious above {config.MAX_CITED_SOURCES} citations.")

    dataset = json.loads((PROJECT_ROOT / "eval" / "questions.json").read_text(encoding="utf-8"))
    needle_rows = json.loads(NEEDLE_RESULTS.read_text(encoding="utf-8"))
    run2_rows = json.loads(RUN2_RESULTS.read_text(encoding="utf-8"))

    needle_units = _needle_sources(needle_rows, dataset)
    run2_units = _run2_sources(dataset)
    needle_results = _evaluate("needle", needle_rows,
                               lambda r: needle_units[(r["length_target"], r["position"], r["method"])], args.show_flags)
    run2_results = _evaluate("run2", run2_rows, lambda r: run2_units[r["method"]], args.show_flags)

    for method in ("full", "hybrid"):
        _report(f"needle, {method} (all lengths)", [r for r in needle_results if r["method"] == method],
                [r for r in needle_rows if r["method"] == method])
    for method in ("full", "dense", "hybrid"):
        _report(f"run 2 (vitamin D paper alone), {method}", [r for r in run2_results if r["method"] == method],
                [r for r in run2_rows if r["method"] == method])

    if args.show_flags:
        for label, keep in [("FLAGGED", lambda r: r["flagged"]), ("MISSED (incorrect, not flagged)",
                                                                  lambda r: not r["flagged"] and r["verdict"] != "correct")]:
            print(f"\n=== {label} ===")
            for r in [r for r in needle_results + run2_results if keep(r)]:
                where = f"{r['dataset']} {r['method']} {r.get('length_target', '')}".strip()
                print(f"\n[{where}] {r['id']} verdict={r['verdict']}  {r['why']}")
                print(f"   {r['answer'][:220]}")


if __name__ == "__main__":
    main()
