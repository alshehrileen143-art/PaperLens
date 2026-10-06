"""Needle in a haystack: how long can a paper be before sending it whole stops working?

The vitamin D paper (the needle) is placed at the beginning, middle or end of
filler text from unrelated medical papers, at several total lengths. Each
question in eval/questions.json (its anchored `needle_question`) is answered two
ways, through the same rag.generate_answer and prompt:
  full    the whole haystack as consecutive, non-overlapping numbered sections
  hybrid  RAG over the haystack: retrieve (dense + BM25, RRF) -> rerank -> expand_context
Answers are graded by eval/judge.py, which sees only excerpts: the text around
the expected evidence plus the passages the answer cited.

The result suggests the token threshold for /ask: below it send the paper whole,
above it use RAG. A length qualifies when, at every position, full is at most
--tolerance questions behind hybrid, and the first (uncached) question on the
paper stays within --max-first-seconds and --max-first-cost.

Filler papers go in samples/filler/ (PDF, DOCX or TXT; real text, not scans).
Files mentioning vitamin D, obesity, or names from the needle paper are rejected.

Usage (from the project root):
    venv\\Scripts\\python eval\\run_needle.py --dry-run          # no API calls: checks fillers, prints sizes and cost
    venv\\Scripts\\python eval\\run_needle.py                    # full run (calls OpenAI)
    venv\\Scripts\\python eval\\run_needle.py --lengths 8000 --ids q2 q3   # small trial
"""

import argparse
import json
import math
import re
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import openai

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import chromadb  # noqa: E402
from chromadb.config import Settings  # noqa: E402

import rag  # noqa: E402
import config  # noqa: E402
from extract import SUPPORTED_EXTENSIONS, ExtractionResult, extract_text  # noqa: E402
from grounding import check_grounding  # noqa: E402
from human_review import accuracy_line, apply_overrides  # noqa: E402
from judge import judge  # noqa: E402
from llm import _encoding, count_tokens, get_client  # noqa: E402
from normalize import normalize_text  # noqa: E402

DEFAULT_LENGTHS = (8_000, 16_000, 32_000, 64_000, 100_000)
POSITIONS = ("beginning", "middle", "end")
METHODS = ("full", "hybrid")
REFUSE_CATEGORIES = ("unanswerable_related", "figure_only", "out_of_scope")

# Filler must not contain the needle's topic or anything that could answer its questions.
FORBIDDEN_FILLER = re.compile(
    r"vitamin\s*d|25\s*\(?\s*oh\s*\)?\s*d|obes|turer|shulhai|jari\b|wakayo|khor\b|fiamenghi|crd42019137788",
    re.IGNORECASE,
)
EVIDENCE_WINDOW_CHARS = 2_000  # Text on each side of the evidence shown to the judge.
EMBEDDING_PRICE_PER_M = 0.02  # text-embedding-3-small, USD per 1M tokens (verify current pricing).


def _flat(text: str) -> str:
    return " ".join(text.split())


# --- Building haystacks ---


def load_fillers(filler_dir: Path) -> tuple[list[str], list[dict]]:
    """Return (accepted filler texts, report rows for every file)."""
    accepted, report = [], []
    for path in sorted(filler_dir.iterdir()):
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        row = {"file": path.name}
        try:
            result = extract_text(path)
        except Exception as exc:  # noqa: BLE001 - report and skip unreadable files
            row.update(status="rejected", reason=f"could not read: {exc}")
            report.append(row)
            continue
        text = normalize_text(result.text)
        match = FORBIDDEN_FILLER.search(text)
        row["tokens"] = count_tokens(result.text)
        if result.ocr_used:
            row.update(status="rejected", reason="scanned (needed OCR)")
        elif match:
            row.update(status="rejected", reason=f"mentions {match.group(0)!r}")
        else:
            row.update(status="accepted", reason="")
            accepted.append(result.text.strip())
        report.append(row)
    return accepted, report


def _cut_at_newline(text: str) -> str:
    """Drop a trailing partial line so the filler does not end mid-sentence."""
    position = text.rfind("\n")
    return text[:position] if position > len(text) // 2 else text


def build_haystack(needle: str, filler_pool: str, length: int, position: str) -> dict:
    """Needle + filler totalling about `length` tokens, needle at the given position."""
    needle_tokens = count_tokens(needle)
    filler_tokens = max(length - needle_tokens, 0)
    encoded = _encoding().encode(filler_pool, disallowed_special=())
    if len(encoded) < filler_tokens:
        raise ValueError(f"Not enough filler for {length} tokens: have {len(encoded)}, need {filler_tokens}.")
    filler = _cut_at_newline(_encoding().decode(encoded[:filler_tokens])) if filler_tokens else ""

    if position == "beginning":
        parts = [needle, filler]
    elif position == "end":
        parts = [filler, needle]
    else:
        middle = len(filler) // 2
        split = filler.rfind("\n", 0, middle)
        split = split if split > 0 else middle
        parts = [filler[:split], needle, filler[split:]]
    text = "\n\n".join(part.strip() for part in parts if part.strip())
    return {
        "text": text,
        "tokens": count_tokens(text),
        "needle_start": round(text.find(needle.strip()[:200]) / len(text), 3),
    }


# --- Answering and grading ---


def _full_context(haystack: str) -> list[rag.RetrievedChunk]:
    # Same as /ask: numbered sections, with section metadata and reference exclusion when configured.
    return rag.full_context({"text": haystack, "pages": None})


def _judge_excerpts(question: dict, haystack_flat: str, cited: list[rag.RetrievedChunk]) -> list[str]:
    """Text around each evidence phrase, plus every passage the answer cited."""
    excerpts = []
    for phrase in question["evidence"]:
        position = haystack_flat.find(_flat(phrase))
        if position != -1:
            start = max(0, position - EVIDENCE_WINDOW_CHARS)
            excerpts.append(haystack_flat[start : position + len(phrase) + EVIDENCE_WINDOW_CHARS])
    for chunk in cited:
        text = _flat(chunk.text)
        if not any(text in excerpt for excerpt in excerpts):
            excerpts.append(text)
    return excerpts


def _cost(row: dict, prices: argparse.Namespace) -> float:
    uncached = row["prompt_tokens"] - row["cached_tokens"]
    return (uncached * prices.price_input + row["cached_tokens"] * prices.price_cached
            + row["completion_tokens"] * prices.price_output) / 1_000_000


class _Timed:
    """Result of a call made with retries: the value and how long the successful attempt took."""

    def __init__(self, value, seconds: float, retries: int):
        self.value, self.seconds, self.retries = value, seconds, retries


def _is_connection_problem(exc: BaseException) -> bool:
    # Our OpenAI wrappers raise LLMError "from" the SDK error, so look at the cause too.
    # APITimeoutError is a subclass of APIConnectionError.
    return isinstance(exc, openai.APIConnectionError) or isinstance(exc.__cause__, openai.APIConnectionError)


def _with_retries(action, label: str, args) -> _Timed:
    """Run `action`, retrying connection errors and timeouts with growing waits.

    Only the successful attempt is timed, so latency figures are not inflated
    by waiting. Any other error, or the last failed retry, is raised.
    """
    for attempt in range(args.retries + 1):
        started = time.perf_counter()
        try:
            value = action()
            return _Timed(value, time.perf_counter() - started, attempt)
        except Exception as exc:
            if not _is_connection_problem(exc) or attempt == args.retries:
                raise
            wait = args.retry_wait * 3**attempt
            cause = exc if isinstance(exc, openai.APIConnectionError) else exc.__cause__
            print(f"    connection problem during {label} ({type(cause).__name__}); "
                  f"retry {attempt + 1}/{args.retries} in {wait:.0f}s", flush=True)
            time.sleep(wait)


def _display(path: Path) -> str:
    """Path relative to the project when possible, for messages and commands."""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _row_key(row: dict) -> tuple:
    return (row["length_target"], row["position"], row["method"], row["id"])


def run_haystack(args, client, questions, haystack: dict, length: int, position: str, save, done: set) -> list[dict]:
    """Answer and grade every question on one haystack, skipping those in `done`.

    `save(rows)` is called after each answer.
    """
    rows = []
    haystack_flat = _flat(haystack["text"])
    pending = {
        method: [(n, q) for n, q in enumerate(questions) if (length, position, method, q["id"]) not in done]
        for method in args.methods
    }
    skipped = sum(len(questions) - len(items) for items in pending.values())
    if skipped:
        print(f"  (resuming: {skipped} answers already in the results file are skipped)")

    paper_id = None
    if pending.get("hybrid"):
        paper_id, _, _ = _with_retries(
            lambda: rag.index_paper(ExtractionResult(haystack["text"], ocr_used=False), f"haystack-{length}-{position}", store_text=False),
            "indexing", args,
        ).value

    for method in args.methods:
        full_context = _full_context(haystack["text"]) if method == "full" else None
        for number, question in pending[method]:
            text = question["needle_question"]
            rewrite = {}  # Query-rewriting call, when config.QUERY_REWRITE is on.
            if method == "full":
                context, retrieval = full_context, _Timed(None, 0.0, 0)
            else:

                def retrieve_once():
                    rewrite.clear()
                    started = time.perf_counter()
                    extra = rag.extra_retrieval_queries(text, rewrite)  # [] unless QUERY_REWRITE is on.
                    if extra:
                        rewrite.update(query=extra[0], seconds=time.perf_counter() - started)
                    results = rag.retrieve(paper_id, text, method="hybrid", extra_queries=extra)
                    return rag.expand_context(paper_id, rag.rerank(text, results))

                retrieval = _with_retries(retrieve_once, f"retrieval {question['id']}", args)
                context = retrieval.value
            usage = {}

            def answer_once():
                usage.clear()
                return rag.generate_answer(text, context, usage_out=usage)

            try:
                generation = _with_retries(answer_once, f"answer {question['id']}", args)
                answer, found, used = generation.value
                error = None
            except rag.AnswerError as exc:  # Unusable model output (not a connection problem).
                generation = _Timed(None, 0.0, 0)
                answer, found, used, error = "", False, [], str(exc)

            row = {
                "length_target": length,
                "length_tokens": haystack["tokens"],
                "position": position,
                "needle_start": haystack["needle_start"],
                "method": method,
                "first_question": number == 0,
                "id": question["id"],
                "language": question["language"],
                "category": question["category"],
                "expected_behavior": question["expected_behavior"],
                "question": text,
                "answer": answer,
                "found": found,
                "context_chunks": len(context),
                "context_chunk_ids": [chunk.index for chunk in context],
                "matched_units": [chunk.matched for chunk in context if chunk.matched],
                "cited_chunks": [chunk.index for chunk in used],
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "cached_tokens": usage.get("cached_tokens", 0),
                "retrieval_s": round(retrieval.seconds, 3),
                "generation_s": round(generation.seconds, 3),
                "total_s": round(retrieval.seconds + generation.seconds, 3),
                "retries": retrieval.retries + generation.retries,
                "retrieval_query": rewrite.get("query"),
                "rewrite_prompt_tokens": rewrite.get("prompt_tokens", 0),
                "rewrite_completion_tokens": rewrite.get("completion_tokens", 0),
                "rewrite_s": round(rewrite["seconds"], 3) if rewrite.get("seconds") else None,
            }
            # The same grounding check as /ask: RAG splits warning / citation mismatch
            # using the whole context; full context uses the cited sections +/- 1.
            row["grounding_warning"], row["citation_mismatch"] = None, None
            if found and used:
                rag_mode = method == "hybrid"
                grounding = check_grounding(
                    text, answer,
                    rag._evidence_texts(paper_id, context, used, "rag" if rag_mode else "full"),
                    len(used),
                    context_texts=[chunk.text for chunk in context] if rag_mode else None,
                )
                row["grounding_warning"] = None if grounding.ok else grounding.describe()
                row["citation_mismatch"] = grounding.describe_mismatch() or None
            row["cost_usd"] = round(_cost(row, args), 6)
            row["rewrite_cost_usd"] = round((row["rewrite_prompt_tokens"] * args.price_input
                                             + row["rewrite_completion_tokens"] * args.price_output) / 1e6, 6)
            if error:
                row.update(verdict="error", reason=error, unverified_claims=[], judge_prompt_tokens=0, judge_completion_tokens=0)
            else:
                excerpts = _judge_excerpts(question, haystack_flat, used)
                grading = _with_retries(lambda: judge(client, question, text, answer, excerpts), f"judge {question['id']}", args)
                row.update(grading.value)
                row["retries"] += grading.retries
            rows.append(row)
            save(rows)
            print(f"  [{method:6}] {question['id']:4} {row['verdict']:9} found={found!s:5} "
                  f"tokens={row['prompt_tokens']:>6} (cached {row['cached_tokens']:>6}) {row['total_s']:5.1f}s "
                  f"${row['cost_usd']:.4f}", flush=True)

    if paper_id:
        rag._chroma().delete_collection(f"paper_{paper_id}")
        try:  # Only exists when small-to-big was on.
            rag._chroma().delete_collection(rag._small_collection_name(paper_id))
        except Exception:  # noqa: BLE001 - chromadb raises its own NotFoundError types
            pass
    return rows


# --- Reporting ---


def _acc(rows: list[dict]) -> str:
    if not rows:
        return "-"
    correct = sum(row["verdict"] == "correct" for row in rows)
    return f"{correct}/{len(rows)} {100 * correct / len(rows):.0f}%"


def _correct(rows: list[dict]) -> int:
    return sum(row["verdict"] == "correct" for row in rows)


def report(rows: list[dict], args, results_name: str | None = None) -> None:
    lengths = sorted({row["length_target"] for row in rows})
    methods = [m for m in METHODS if any(row["method"] == m for row in rows)]

    # Accuracy as judged, then after the human corrections in eval/human_overrides.json.
    # Everything below uses the corrected verdicts.
    judged = rows
    if results_name:
        rows, applied = apply_overrides(rows, results_name)
        print(f"\n=== Judge vs human review ({applied} override(s) applied) ===\n")
        for length in lengths:
            for method in methods:
                keep = lambda rs: [r for r in rs if r["length_target"] == length and r["method"] == method]  # noqa: E731
                print(accuracy_line(f"  {length:>7} {method:7}", keep(judged), keep(rows)))
    pick = lambda **kw: [r for r in rows if all(r[k] == v for k, v in kw.items())]  # noqa: E731

    flagged = [r for r in rows if r.get("grounding_warning") or r.get("citation_mismatch")]
    if any("grounding_warning" in r for r in rows):
        print("\nGrounding check (as in /ask):")
        for r in flagged:
            kind = "grounding_warning" if r.get("grounding_warning") else "citation_mismatch"
            detail = r.get("grounding_warning") or r.get("citation_mismatch")
            print(f"  {r['length_target']:>7} {r['method']:6} {r['id']:4} {kind:18} verdict={r['verdict']:9} {detail}")
        if not flagged:
            print("  nothing flagged")

    rewritten = [r for r in rows if r.get("rewrite_s")]
    if rewritten:
        seconds = sorted(r["rewrite_s"] for r in rewritten)
        print(f"\nQuery rewriting: {len(rewritten)} calls, median {seconds[len(seconds) // 2]:.2f}s extra, "
              f"${sum(r['rewrite_cost_usd'] for r in rewritten):.4f} in total")

    print("\n=== Accuracy by length and position ===\n")
    header = f"{'length':>8} {'method':7}" + "".join(f"{p:>13}" for p in POSITIONS) + f"{'all':>13}"
    print(header)
    print("-" * len(header))
    for length in lengths:
        for method in methods:
            cells = [_acc(pick(length_target=length, method=method, position=p)) for p in POSITIONS]
            print(f"{length:>8} {method:7}" + "".join(f"{c:>13}" for c in cells)
                  + f"{_acc(pick(length_target=length, method=method)):>13}")

    print("\n=== Accuracy by expected behaviour and language (all positions) ===\n")
    groups = [("answer", "expected_behavior", "answer"), ("no_consensus", "expected_behavior", "no_consensus")]
    groups += [(c, "category", c) for c in REFUSE_CATEGORIES] + [("lang=en", "language", "en"), ("lang=ar", "language", "ar")]
    print(f"{'length':>8} {'method':7}" + "".join(f"{name:>22}" for name, _, _ in groups))
    for length in lengths:
        for method in methods:
            base = pick(length_target=length, method=method)
            print(f"{length:>8} {method:7}" + "".join(
                f"{_acc([r for r in base if r[key] == value]):>22}" for _, key, value in groups))

    print("\n=== First question on the paper (no cache yet): worst case over positions ===\n")
    print(f"{'length':>8} {'method':7} {'tokens':>8} {'max seconds':>12} {'max cost $':>11} {'within caps':>12}")
    for length in lengths:
        for method in methods:
            first = pick(length_target=length, method=method, first_question=True)
            if not first:
                continue
            seconds, cost = max(r["total_s"] for r in first), max(r["cost_usd"] for r in first)
            ok = seconds <= args.max_first_seconds and cost <= args.max_first_cost
            print(f"{length:>8} {method:7} {max(r['prompt_tokens'] for r in first):>8} {seconds:>12.2f} "
                  f"{cost:>11.4f} {'yes' if ok else 'NO':>12}")

    print("\n=== Per question (all questions): tokens, time, cost, unverified claims ===\n")
    print(f"{'length':>8} {'method':7} {'prompt':>8} {'cached':>8} {'median s':>9} {'p90 s':>7} {'mean $':>9} {'unverified':>11}")
    for length in lengths:
        for method in methods:
            base = pick(length_target=length, method=method)
            mean = lambda key: sum(r[key] for r in base) / len(base)  # noqa: E731
            seconds = sorted(r["total_s"] for r in base)
            unverified = sum(len(r["unverified_claims"]) for r in base)
            print(f"{length:>8} {method:7} {mean('prompt_tokens'):>8.0f} {mean('cached_tokens'):>8.0f} "
                  f"{statistics.median(seconds):>9.2f} {seconds[max(0, math.ceil(0.9 * len(seconds)) - 1)]:>7.2f} "
                  f"{mean('cost_usd'):>9.5f} {unverified:>11}")

    if set(methods) == set(METHODS):
        print("\n=== Suggested threshold ===\n")
        positions_run = [p for p in POSITIONS if any(row["position"] == p for row in rows)]
        print(f"  Rule: full is at most {args.tolerance} question(s) behind hybrid at every position, the median "
              f"time of ALL full answers is <= {args.max_median_seconds}s, and the first (uncached) full answer "
              f"costs <= ${args.max_first_cost}. Positions tested: {', '.join(positions_run)}.\n")
        qualifying, contiguous = [], True
        for length in lengths:
            full_rows = pick(length_target=length, method="full")
            accuracy_ok = all(
                _correct(pick(length_target=length, method="full", position=p))
                >= _correct(pick(length_target=length, method="hybrid", position=p)) - args.tolerance
                for p in positions_run
            )
            median_s = statistics.median(r["total_s"] for r in full_rows)
            time_ok = median_s <= args.max_median_seconds
            first = [r for r in full_rows if r["first_question"]]
            cost_ok = bool(first) and all(r["cost_usd"] <= args.max_first_cost for r in first)
            passed = accuracy_ok and time_ok and cost_ok
            print(f"  {length:>7} tokens: accuracy {'ok' if accuracy_ok else 'WORSE'}, "
                  f"median {median_s:.2f}s {'ok' if time_ok else 'TOO SLOW'}, "
                  f"first-answer cost {'ok' if cost_ok else 'TOO HIGH'}  -> {'pass' if passed else 'FAIL'}")
            # The threshold must hold for every shorter length too.
            contiguous = contiguous and passed
            if contiguous:
                qualifying.append(length)
        if qualifying:
            print(f"\n  Full context holds up to {qualifying[-1]} tokens (every shorter length passes too). "
                  "Pick a threshold a bit below it.")
        else:
            print("\n  The shortest length already fails: use RAG for every paper.")


# --- Main ---


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions", default=str(PROJECT_ROOT / "eval" / "questions.json"))
    parser.add_argument("--filler-dir", default=str(PROJECT_ROOT / "samples" / "filler"))
    parser.add_argument("--lengths", type=int, nargs="+", default=list(DEFAULT_LENGTHS))
    parser.add_argument("--positions", nargs="+", default=list(POSITIONS), choices=POSITIONS)
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    parser.add_argument("--ids", nargs="+", help="Only these question ids (for a small trial).")
    parser.add_argument("--dry-run", action="store_true", help="No API calls: check fillers, print sizes and cost.")
    parser.add_argument("--tolerance", type=int, default=1, help="Questions full may trail hybrid by, per position.")
    parser.add_argument("--max-first-seconds", type=float, default=5.0,
                        help="Shown in the first-question table; the threshold uses --max-median-seconds.")
    parser.add_argument("--max-median-seconds", type=float, default=5.0,
                        help="Threshold rule: median time of all full-context answers at a length.")
    parser.add_argument("--max-first-cost", type=float, default=0.02, help="USD.")
    parser.add_argument("--report-only", metavar="RESULTS_JSON",
                        help="Print the report for an existing results file. No API calls, nothing is run.")
    # gpt-4o-mini prices in USD per 1M tokens; verify against current OpenAI pricing.
    parser.add_argument("--price-input", type=float, default=0.15)
    parser.add_argument("--price-cached", type=float, default=0.075)
    parser.add_argument("--price-output", type=float, default=0.60)
    parser.add_argument("--resume", metavar="RESULTS_JSON",
                        help="Continue a stopped run: append to this results file and skip answers already in it.")
    parser.add_argument("--retries", type=int, default=3, help="Retries for connection errors and timeouts.")
    parser.add_argument("--query-rewrite", choices=("on", "off"), help="Override config.QUERY_REWRITE (hybrid only).")
    parser.add_argument("--full-sections", choices=("on", "off"), help="Override config.FULL_CONTEXT_SECTIONS.")
    parser.add_argument("--rag-sections", choices=("on", "off"), help="Override config.RAG_SECTIONS.")
    parser.add_argument("--rag-tag-sections", choices=("on", "off"), help="Override config.RAG_TAG_SECTIONS (needs a fresh index).")
    parser.add_argument("--exclude-references", choices=("on", "off"), help="Override config.EXCLUDE_REFERENCES.")
    parser.add_argument("--small-to-big", choices=("on", "off"), help="Override config.SMALL_TO_BIG (hybrid only).")
    parser.add_argument("--retry-wait", type=float, default=5.0,
                        help="Seconds before the first retry; each later wait is 3x longer (5, 15, 45).")
    args = parser.parse_args()

    if args.query_rewrite:
        config.QUERY_REWRITE = args.query_rewrite == "on"
    if args.full_sections:
        config.FULL_CONTEXT_SECTIONS = args.full_sections == "on"
    if args.rag_sections:
        config.RAG_SECTIONS = args.rag_sections == "on"
    if args.rag_tag_sections:
        config.RAG_TAG_SECTIONS = args.rag_tag_sections == "on"
    if args.exclude_references:
        config.EXCLUDE_REFERENCES = args.exclude_references == "on"
    if args.small_to_big:
        config.SMALL_TO_BIG = args.small_to_big == "on"

    if args.report_only:
        report(json.loads(Path(args.report_only).read_text(encoding="utf-8")), args, Path(args.report_only).name)
        return

    dataset = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    questions = [q for q in dataset["questions"] if not args.ids or q["id"] in args.ids]
    missing = [q["id"] for q in questions if "needle_question" not in q]
    if missing:
        parser.error(f"Questions without needle_question: {missing}")

    needle = extract_text(PROJECT_ROOT / dataset["paper"]).text.strip()
    filler_dir = Path(args.filler_dir)
    if not filler_dir.is_dir():
        parser.error(f"Filler folder not found: {filler_dir}")
    fillers, filler_report = load_fillers(filler_dir)

    print(f"Needle: {dataset['paper']} ({count_tokens(needle)} tokens)\n\nFiller files:")
    for row in filler_report:
        print(f"  {row['status']:9} {row.get('tokens', '-'):>7} tokens  {row['file']}  {row['reason']}")
    filler_pool = "\n\n".join(fillers)
    available = count_tokens(filler_pool)
    needed = max(args.lengths) - count_tokens(needle)
    print(f"\nAccepted filler: {available} tokens (need {needed} for {max(args.lengths)}).")
    if available < needed:
        sys.exit("Not enough filler text. Add more papers to the filler folder.")

    haystacks = {(length, position): build_haystack(needle, filler_pool, length, position)
                 for length in args.lengths for position in args.positions}
    print("\nHaystacks (actual tokens / where the needle starts):")
    for (length, position), haystack in haystacks.items():
        found = sum(_flat(e) in _flat(haystack["text"]) for q in questions for e in q["evidence"])
        total = sum(len(q["evidence"]) for q in questions)
        print(f"  {length:>7} {position:9} {haystack['tokens']:>7} tokens, needle at {haystack['needle_start']:.0%}, "
              f"evidence found {found}/{total}")

    if args.dry_run:
        n = len(questions)
        full_in = sum(h["tokens"] for h in haystacks.values())  # One uncached question per haystack...
        full_cached = full_in * (n - 1)  # ...the rest mostly cached.
        hybrid_in = 2_500 * n * len(haystacks)
        judge_in = 3_000 * n * len(haystacks) * len(args.methods)
        cost = 0.0
        if "full" in args.methods:
            cost += (full_in * args.price_input + full_cached * args.price_cached) / 1e6
        if "hybrid" in args.methods:
            cost += hybrid_in * args.price_input / 1e6 + full_in * EMBEDDING_PRICE_PER_M / 1e6
        cost += judge_in * args.price_input / 1e6
        print(f"\nPlanned: {n} questions x {len(haystacks)} haystacks x {len(args.methods)} methods "
              f"= {n * len(haystacks) * len(args.methods)} answers and as many judgments.")
        print(f"Estimated cost: about ${cost:.2f} (assumes most full-context calls hit the prompt cache).")
        print("Dry run: no API calls were made.")
        return

    results_dir = PROJECT_ROOT / "eval" / "results"
    results_dir.mkdir(exist_ok=True)
    if args.resume:
        output = Path(args.resume)
        if not output.is_absolute():
            output = PROJECT_ROOT / output
        if not output.is_file():
            parser.error(f"Results file to resume not found: {output}")
        rows = json.loads(output.read_text(encoding="utf-8"))
        # The haystacks must be rebuilt identically, or old and new answers are not comparable.
        for row in rows:
            haystack = haystacks.get((row["length_target"], row["position"]))
            if haystack and haystack["tokens"] != row["length_tokens"]:
                sys.exit(f"Cannot resume: the {row['length_target']}/{row['position']} haystack is now "
                         f"{haystack['tokens']} tokens but was {row['length_tokens']}. Did the filler or needle change?")
        print(f"\nResuming {_display(output)}: {len(rows)} answers already done.")
    else:
        output = results_dir / f"needle_{datetime.now():%Y%m%d_%H%M%S}.json"
        rows = []
    done = {_row_key(row) for row in rows}

    rag._client = chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))
    client = get_client()

    def save(haystack_rows: list[dict]) -> None:
        # Written after every answer, so a run that is stopped keeps what it finished.
        output.write_text(json.dumps(rows + haystack_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        for (length, position), haystack in haystacks.items():
            print(f"\n--- {length} tokens, needle at {position} ---")
            rows += run_haystack(args, client, questions, haystack, length, position, save, done)
    except Exception as exc:
        if not _is_connection_problem(exc):
            raise
        argv = sys.argv[1:]
        if "--resume" in argv:  # Drop the previous --resume and its file.
            position = argv.index("--resume")
            argv = argv[:position] + argv[position + 2 :]
        options = " ".join(argv)
        print(f"\nStopped: still no connection after {args.retries} retries ({exc}).")
        print("Everything finished so far is saved. Continue with:")
        print(f"  .\\venv\\Scripts\\python.exe -u eval\\run_needle.py {options} --resume {_display(output)}")
        sys.exit(1)

    report(rows, args, output.name)
    print(f"\nWrote {_display(output)}")


if __name__ == "__main__":
    main()
