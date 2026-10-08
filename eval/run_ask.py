"""End-to-end check of /ask exactly as a user calls it.

Uploads the paper through /upload (real index in chroma_db/ and paper_store/),
or reuses an uploaded one with --paper-id, then sends every question in
eval/questions.json to /ask and grades each answer with the same judge as run 2
of eval/run_rag_vs_full.py (whole paper as the reference), so results are
comparable with run 2. Reports accuracy as judged and after the human
corrections in eval/human_overrides.json.

Token usage is recorded by wrapping rag.generate_answer and rag.rewrite_query
from outside; the production code path is otherwise unchanged. Calls the API.

Usage (from the project root):
    venv\\Scripts\\python eval\\run_ask.py                              # upload, then ask
    venv\\Scripts\\python eval\\run_ask.py --paper-id ID --force-rag --query-rewrite on
    venv\\Scripts\\python eval\\run_ask.py --report-only eval\\results\\ask_....json   # no API calls
"""

import argparse
import json
import re
import sys
import time
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "eval"))

import config  # noqa: E402
import rag  # noqa: E402
from extract import extract_text  # noqa: E402
from llm import count_tokens  # noqa: E402
from human_review import accuracy_line, apply_overrides  # noqa: E402

RUN2 = PROJECT_ROOT / "eval" / "results" / "run2_approved_judge_rules.json"
QUESTIONS = PROJECT_ROOT / "eval" / "questions.json"
CATEGORIES = ("answerable", "no_consensus", "unanswerable_related", "figure_only", "out_of_scope")
# USD per 1M tokens (gpt-4o-mini; text-embedding-3-small), as confirmed by the user.
PRICE_INPUT, PRICE_CACHED, PRICE_OUTPUT, PRICE_EMBEDDING = 0.15, 0.075, 0.60, 0.02


def _cost(prompt: int, cached: int, completion: int) -> float:
    return ((prompt - cached) * PRICE_INPUT + cached * PRICE_CACHED + completion * PRICE_OUTPUT) / 1e6


_MARKER = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_ABBREVIATIONS = re.compile(r"\b(?:et al|e\.g|i\.e|vs|Fig|approx)\.", re.IGNORECASE)
_SENTENCE_BREAK = re.compile(r"(?<=[.!?؟])\s+")


def sentence_count(text: str) -> int:
    """Approximate: text split after ". ! ? ؟" followed by a space, once citation markers and
    common abbreviations are removed (a decimal such as 1.41 has no space after its dot)."""
    text = _ABBREVIATIONS.sub("", _MARKER.sub("", text)).strip()
    return len([part for part in _SENTENCE_BREAK.split(text) if part.strip()]) if text else 0


_ARABIC = re.compile(r"[؀-ۿ]")


def _model_quotes(row: dict) -> list[tuple[str, bool]]:
    """[(quote the model wrote, verified in the /ask response)] for one row.

    From the raw model output when it was saved (every quote, including ones dropped
    before the response); otherwise from the response's sources and quotes_removed.
    """
    verified = {s["chunk_index"] for s in row.get("sources", []) if s.get("quote")}
    if row.get("raw_model_output"):
        try:
            citations = json.loads(row["raw_model_output"]).get("citations") or []
        except ValueError:
            citations = []
        return [(c["quote"], c.get("source_id") in verified) for c in citations
                if isinstance(c, dict) and isinstance(c.get("quote"), str) and c["quote"].strip()]
    quotes = [(s.get("quote") or s.get("rejected_quote"), bool(s.get("quote"))) for s in row.get("sources", [])]
    quotes += [(q["quote"], False) for q in row.get("quotes_removed", [])]
    return [(quote, ok) for quote, ok in quotes if quote]


def marker_numbers(text: str) -> set[int]:
    return {int(n) for group in _MARKER.findall(text) for n in re.split(r"\s*,\s*", group)}


# Local check (no API) of where each must_include item of eval/questions.json appears: in the
# short answer, only in "details", or nowhere. must_include items are concepts, not exact text
# ("RR 1.41"), so each item is matched by key terms: an item is present when every group has
# at least one of its regular expressions (case-insensitive) in the text. Diagnostic only: the
# verdict stays the judge's.
_NO_CONSENSUS = [r"no consensus", r"lack of consensus", r"not? (?:a )?consensus", r"no universally",
                 r"لا (?:يوجد )?إجماع", r"عدم وجود إجماع", r"غير متفق"]
MUST_INCLUDE_TERMS = {
    "q1": [[[r"\b20\b"], [r"stud", r"دراس"]]],
    "q2": [[[r"1\.41"]], [[r"1\.26"], [r"1\.59"]]],
    "q3": [[[r"turer"]]],
    "q4": [[[r"pubmed"]], [[r"embase"]], [[r"lilacs"]]],
    "q5": [[[r"77\.5"]], [[r"66\.1"]]],
    "q8": [[_NO_CONSENSUS, [r"dose", r"جرع"]]],
    "q10": [[[r"publication bias", r"تحيز النشر"]]],
    "q11": [[[r"3,?155"]]],
    "q12": [[[r"\bR\b"]], [[r"\bmeta\b"]]],
    "q13": [[[r"relative risk", r"\bRR\b", r"الخطر النسبي"]], [[r"random", r"عشوائي"]]],
    "q14": [[[r"CRD42019137788"]]],
    "q15": [[[r"(?<![\d.])10 ?ng"]]],
    "q16": [[[r"2,?478"]]],
    "q17": [[[r"zhang"]], [[r"wakayo"]]],
    "q18": [[[r"cross-sectional", r"مقطعي"], [r"meta-analys", r"تحليل (?:تجميعي|تلوي|شمولي)"]]],
    "q19": [[[r"shulhai"]], [[r"0\.027"]]],
    "q20": [[_NO_CONSENSUS], [[r"adipose", r"fat[- ]soluble", r"fat tissue", r"الدهن"]]],
}
_UNIFY_DASHES = str.maketrans({"‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "−": "-"})


def _has_item(text: str, groups: list[list[str]]) -> bool:
    text = unicodedata.normalize("NFKC", _MARKER.sub("", text)).translate(_UNIFY_DASHES)
    return all(any(re.search(term, text, re.IGNORECASE) for term in group) for group in groups)


def must_include_places(row: dict, items: list[str]) -> list[tuple[str, str]]:
    """[(item, "answer" | "details only" | "missing" | "no terms")] for one row."""
    terms = MUST_INCLUDE_TERMS.get(row["id"], [])
    places = []
    for index, item in enumerate(items):
        if index >= len(terms):
            places.append((item, "no terms"))
        elif _has_item(row["answer"], terms[index]):
            places.append((item, "answer"))
        elif _has_item(row.get("details") or "", terms[index]):
            places.append((item, "details only"))
        else:
            places.append((item, "missing"))
    return places


def report(data: dict, results_name: str) -> None:
    judged = data["rows"]
    corrected, applied = apply_overrides(judged, results_name)
    settings = data.get("settings", {})
    print(f"\n=== {results_name}  settings: {settings} ===")
    print(f"modes: {dict(Counter(r['mode'] for r in judged))}   human overrides applied: {applied}\n")
    print(accuracy_line("overall", judged, corrected))
    for category in CATEGORIES:
        keep = lambda rows, c=category: [r for r in rows if r["category"] == c]  # noqa: E731
        print(accuracy_line(f"  {category}", keep(judged), keep(corrected)))
    for language in ("en", "ar"):
        keep = lambda rows, lang=language: [r for r in rows if r["language"] == lang]  # noqa: E731
        print(accuracy_line(f"  lang={language}", keep(judged), keep(corrected)))

    baseline_method = "hybrid" if settings.get("force_rag") else "full"
    run2_rows, _ = apply_overrides(json.loads(RUN2.read_text(encoding="utf-8")), RUN2.name)
    run2 = {r["id"]: r for r in run2_rows if r["method"] == baseline_method}
    print(f"\nCompared with run 2, {baseline_method} (after human review: "
          f"{sum(r['verdict'] == 'correct' for r in run2.values())}/{len(run2)}):")
    for row in corrected:
        if row["id"] in run2 and run2[row["id"]]["verdict"] != row["verdict"]:
            print(f"  {row['id']}: {run2[row['id']]['verdict']} -> {row['verdict']}")

    print("\nIncorrect after human review:")
    for row in [r for r in corrected if r["verdict"] != "correct"]:
        query = f"\n    search query: {row['retrieval_query']}" if row.get("retrieval_query") else ""
        print(f"  {row['id']} ({row['expected_behavior']}): {row['answer']}\n    judge: {row['reason']}{query}")

    found = [r for r in judged if r["found"]]
    lengths = sorted(len(r["answer"]) for r in found)
    if lengths:
        print(f"\nAnswer length (found answers, characters, without details): "
              f"median {lengths[len(lengths) // 2]}, max {lengths[-1]}")
    if found and any("details" in r for r in found):
        sentences = {r["id"]: sentence_count(r["answer"]) for r in found}
        over = {question_id: n for question_id, n in sentences.items() if n > 3}
        print(f"Sentences in the answer (approximate): max {max(sentences.values())}; more than 3: "
              f"{len(over)} {over if over else ''}")
        print(f"Details given: {sum(bool(r['details']) for r in found)} of {len(found)} found answers")
        counts = Counter(len(r["sources"]) for r in found)
        print(f"Sources per answer: {dict(sorted(counts.items()))}")
        if any(r.get("markers") for r in found):
            added = [r["id"] for r in found if r.get("markers") == "added"]
            print(f"Citation numbers in the short answer: {len(found) - len(added)} from the model itself, "
                  f"{len(added)} added at the end by the code {added if added else ''}")
        for row in found:
            numbers = marker_numbers(row["answer"] + " " + (row["details"] or ""))
            expected = set(range(1, len(row["sources"]) + 1))
            if numbers != expected:
                print(f"  markers {row['id']}: in text {sorted(numbers)}, sources {sorted(expected)}")
        must_include = {q["id"]: q.get("must_include") or [] for q in json.loads(QUESTIONS.read_text(encoding="utf-8"))["questions"]}
        places = {r["id"]: must_include_places(r, must_include[r["id"]]) for r in found if must_include.get(r["id"])}
        tally = Counter(place for row_places in places.values() for _, place in row_places)
        print(f"must_include (local term check, not the judge): {sum(tally.values())} items -> "
              f"{tally['answer']} in the short answer, {tally['details only']} only in details, "
              f"{tally['missing']} missing" + (f", {tally['no terms']} without terms" if tally["no terms"] else ""))
        for question_id, row_places in places.items():
            flagged = [(item, place) for item, place in row_places if place != "answer"]
            if flagged or question_id in ("q8", "q20"):
                print(f"  {question_id}: " + "; ".join(f"{item!r} -> {place}" for item, place in row_places))
        print("Where the sources point (page of the quote, nearest heading):")
        for row in found:
            print(f"  {row['id']}: " + " | ".join(
                f"[{s['number']}] p{s['page']}{'' if s['page_end'] in (None, s['page']) else '-' + str(s['page_end'])} "
                f"{s['title']}{'' if s['quote'] else ' (no quote)'}" for s in row["sources"]))
    if any("quotes_cited" in r for r in judged):
        cited = sum(r["quotes_cited"] for r in judged)
        kept = sum(r["quotes_kept"] for r in judged)
        removed = [(r["id"], quote) for r in judged for quote in r["quotes_removed"]]
        missing = sum(len(r["quotes_missing"]) for r in judged)
        dropped = sum(len(r.get("quotes_dropped", [])) for r in judged)
        print(f"Quotes: {cited} cited sources -> {kept} kept, {len(removed)} removed (not verbatim), "
              f"{missing} missing (no quote given); citations dropped (unknown id or past the limit): {dropped}")
        for question_id, quote in removed:
            print(f"  removed {question_id} source {quote['source_id']}: {quote['quote']}")
        # Verbatim quotes rejected afterwards by the number check (number_check.py); they are
        # counted in "kept" above, which is measured before that check.
        mismatched = [(r["id"], s) for r in judged for s in r.get("sources", []) if s.get("quote_issue")]
        if any("quote_issue" in s for r in judged for s in r.get("sources", [])):
            print(f"Number check: {len(mismatched)} of {kept} verbatim quotes rejected (number_mismatch)")
            for question_id, source in mismatched:
                print(f"  {question_id} [{source['number']}] missing {source['missing_numbers']}: "
                      f"{source['rejected_quote']}")
        # Quotes for Arabic questions: translated into Arabic (any Arabic letter, while the paper
        # is in English) or copied in the source's language, and how many of each were verified.
        arabic = [(r["id"], quote, verified) for r in found if r["language"] == "ar"
                  for quote, verified in _model_quotes(r)]
        if arabic:
            translated = [(qid, quote, ok) for qid, quote, ok in arabic if _ARABIC.search(quote)]
            original = [item for item in arabic if not _ARABIC.search(item[1])]
            print(f"Quotes for Arabic questions: {len(arabic)}; translated (Arabic letters): {len(translated)}, "
                  f"verified {sum(ok for *_, ok in translated)}; in the source's language: {len(original)}, "
                  f"verified {sum(ok for *_, ok in original)}")
            for question_id, quote, ok in translated:
                print(f"  translated {question_id} ({'verified' if ok else 'not verified'}): {quote}")

    for field in ("grounding_warning", "citation_mismatch"):
        flagged = [r for r in corrected if r.get(field)]
        print(f"\n{field}: {len(flagged)} answer(s)")
        for row in flagged:
            print(f"  {row['id']} verdict={row['verdict']}: {row[field]}\n    answer: {row['answer']}")

    seconds = sorted(r["seconds"] for r in judged)
    rewrite_s = [r["rewrite_seconds"] for r in judged if r.get("rewrite_seconds")]
    answer_cost = sum(_cost(r["prompt_tokens"], r["cached_tokens"], r["completion_tokens"]) for r in judged)
    rewrite_cost = sum(_cost(r.get("rewrite_prompt_tokens", 0), 0, r.get("rewrite_completion_tokens", 0)) for r in judged)
    judge_cost = sum((r["judge_prompt_tokens"] - r["judge_cached_tokens"]) * PRICE_INPUT
                     + r["judge_cached_tokens"] * PRICE_CACHED for r in judged) / 1e6
    print(f"\nTime per question: first {judged[0]['seconds']}s, median {seconds[len(seconds) // 2]}s, max {seconds[-1]}s")
    if rewrite_s:
        print(f"Query rewriting: median {sorted(rewrite_s)[len(rewrite_s) // 2]:.2f}s extra per question, "
              f"${rewrite_cost:.4f} for {len(rewrite_s)} rewrites")
    print(f"Cost: answers ${answer_cost:.4f} | rewriting ${rewrite_cost:.4f} | judge ${judge_cost:.4f} | "
          f"upload embeddings ~${data.get('embedding_cost', 0):.5f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--paper-id", help="Ask about an already uploaded paper instead of uploading it again.")
    parser.add_argument("--force-rag", action="store_true", help="Use RAG even for short papers (sets the full-context limit to 0).")
    parser.add_argument("--query-rewrite", choices=("on", "off"), help="Override config.QUERY_REWRITE.")
    parser.add_argument("--report-only", metavar="RESULTS_JSON", help="Print the report for a saved run. No API calls.")
    parser.add_argument("--full-sections", choices=("on", "off"), help="Override config.FULL_CONTEXT_SECTIONS.")
    parser.add_argument("--rag-sections", choices=("on", "off"), help="Override config.RAG_SECTIONS "
                        "(RAG needs a fresh upload to tag chunks, so do not combine 'on' with --paper-id).")
    parser.add_argument("--rag-tag-sections", choices=("on", "off"), help="Override config.RAG_TAG_SECTIONS (needs a fresh index).")
    parser.add_argument("--exclude-references", choices=("on", "off"), help="Override config.EXCLUDE_REFERENCES.")
    parser.add_argument("--small-to-big", choices=("on", "off"), help="Override config.SMALL_TO_BIG (RAG only).")
    args = parser.parse_args()

    if args.report_only:
        path = Path(args.report_only)
        report(json.loads(path.read_text(encoding="utf-8")), path.name)
        return

    # Imported here so --report-only needs no API key or server setup.
    from fastapi.testclient import TestClient

    import run_rag_vs_full  # Its judge is the run-2 judge.
    from llm import get_client
    from main import app

    if args.force_rag:
        config.FULL_CONTEXT_MAX_TOKENS = 0
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
    settings = {"force_rag": args.force_rag, "query_rewrite": config.QUERY_REWRITE,
                "full_sections": config.FULL_CONTEXT_SECTIONS, "rag_sections": config.RAG_SECTIONS,
                "rag_tag_sections": config.RAG_TAG_SECTIONS,
                "exclude_references": config.EXCLUDE_REFERENCES, "small_to_big": config.SMALL_TO_BIG,
                "answer_temperature": config.ANSWER_TEMPERATURE, "full_context_max_tokens": config.FULL_CONTEXT_MAX_TOKENS,
                "grounding_action": config.GROUNDING_ACTION, "judge": run_rag_vs_full.JUDGE_VERSION}
    print(f"Settings: {settings}")

    dataset = json.loads((PROJECT_ROOT / "eval" / "questions.json").read_text(encoding="utf-8"))
    paper_path = PROJECT_ROOT / dataset["paper"]
    paper_text = extract_text(paper_path).text
    client = TestClient(app)

    embedding_cost = 0.0
    if args.paper_id:
        paper_id = args.paper_id
    else:
        response = client.post("/upload", files={"file": (paper_path.name, paper_path.read_bytes())})
        response.raise_for_status()
        upload = response.json()
        paper_id = upload["paper_id"]
        embedding_cost = count_tokens(paper_text) * PRICE_EMBEDDING / 1e6  # Approximate: chunks overlap a little.
        print(f"Uploaded: paper_id={paper_id}, {upload['chunk_count']} chunks")

    calls = {}
    original_generate, original_rewrite = rag.generate_answer, rag.rewrite_query

    def logged_generate_answer(question, chunks, usage_out=None, extra_out=None):
        # extra_out is passed through: /ask needs its "details".
        usage, extra = {}, extra_out if extra_out is not None else {}
        result = original_generate(question, chunks, usage_out=usage, extra_out=extra)
        calls["answer"], calls["quotes"] = usage, extra
        calls["context_ids"] = [chunk.index for chunk in chunks]
        calls["matched"] = [chunk.matched for chunk in chunks if chunk.matched]
        return result

    def logged_rewrite_query(question, usage_out=None):
        usage, started = {}, time.perf_counter()
        query = original_rewrite(question, usage_out=usage)
        calls["rewrite"] = {**usage, "seconds": time.perf_counter() - started, "query": query}
        return query

    rag.generate_answer, rag.rewrite_query = logged_generate_answer, logged_rewrite_query
    judge_client = get_client()
    judge_prompt = run_rag_vs_full.JUDGE_RULES % paper_text
    rows = []
    try:
        for question in dataset["questions"]:
            calls.clear()
            started = time.perf_counter()
            response = client.post("/ask", json={"paper_id": paper_id, "question": question["question"]})
            seconds = time.perf_counter() - started
            response.raise_for_status()
            result = response.json()
            answer_usage, rewrite, quotes = calls.get("answer", {}), calls.get("rewrite", {}), calls.get("quotes", {})
            judged_text = result["answer"] + (f"\n\n{result['details']}" if result["details"] else "")
            row = {
                "method": "ask",
                "id": question["id"],
                "language": question["language"],
                "category": question["category"],
                "expected_behavior": question["expected_behavior"],
                "question": question["question"],
                "mode": result["mode"],
                "retrieval_query": rewrite.get("query"),
                "answer": result["answer"],
                "details": result["details"],
                # What the judge reads: everything the user can read, as in the app
                # (the short answer, then the details folded under it).
                "judged_text": judged_text,
                "found": result["found"],
                "grounding_warning": result["grounding_warning"],
                "citation_mismatch": result.get("citation_mismatch"),
                "context_chunk_ids": calls.get("context_ids", []),
                "matched_units": calls.get("matched", []),
                "cited_chunks": [source["chunk_index"] for source in result["sources"]],
                "quotes": {source["chunk_index"]: source["quote"] for source in result["sources"] if source["quote"]},
                "sources": [{key: source[key] for key in ("number", "chunk_index", "page", "page_end", "title", "quote",
                                                          "quote_issue", "missing_numbers", "rejected_quote")}
                            for source in result["sources"]],
                "quotes_cited": quotes.get("cited", 0),
                "quotes_kept": quotes.get("kept", 0),
                "quotes_removed": quotes.get("removed", []),
                "quotes_missing": quotes.get("missing", []),
                "quotes_dropped": quotes.get("dropped", []),
                # "model": the answer kept the model's own markers; "added": none left, so "[1].." was appended.
                "markers": quotes.get("markers"),
                "raw_model_output": quotes.get("raw"),
                "prompt_tokens": answer_usage.get("prompt_tokens", 0),
                "cached_tokens": answer_usage.get("cached_tokens", 0),
                "completion_tokens": answer_usage.get("completion_tokens", 0),
                "rewrite_prompt_tokens": rewrite.get("prompt_tokens", 0),
                "rewrite_completion_tokens": rewrite.get("completion_tokens", 0),
                "rewrite_seconds": round(rewrite["seconds"], 2) if rewrite else None,
                "seconds": round(seconds, 2),
            }
            row.update(run_rag_vs_full._judge(judge_client, judge_prompt, question, judged_text))
            rows.append(row)
            flag = "  [grounding_warning]" if row["grounding_warning"] else ""
            query = f"  query: {row['retrieval_query'][:70]}" if row["retrieval_query"] else ""
            print(f"  {row['id']:4} {row['mode']:4} {row['verdict']:9} found={row['found']!s:5} {row['seconds']:5.1f}s{flag}{query}",
                  flush=True)
    finally:
        rag.generate_answer, rag.rewrite_query = original_generate, original_rewrite

    output = PROJECT_ROOT / "eval" / "results" / f"ask_{datetime.now():%Y%m%d_%H%M%S}.json"
    data = {"paper_id": paper_id, "settings": settings, "embedding_cost": embedding_cost, "rows": rows}
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    report(data, output.name)
    print(f"\nWrote {output.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
