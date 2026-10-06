"""Experiment: do we need RAG, or can we send the whole paper?

Answers every question in eval/questions.json three ways, all through the same
rag.generate_answer and the same prompt; only the context differs:
  full    the whole paper as a single source (no retrieval)
  dense   retrieve (embeddings) -> rerank -> expand_context
  hybrid  retrieve (embeddings + BM25, RRF) -> rerank -> expand_context

Each answer is graded by an LLM judge (gpt-4o-mini, temperature 0) against the
question's expected_behavior and must_include, with the paper as reference.

Calls the OpenAI API (embeddings, answers, judge). Run from the project root:
    venv\\Scripts\\python eval\\run_rag_vs_full.py [--sample-size 10] [--seed 7]
Writes eval/results/rag_vs_full.json and eval/judge_review.md.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import chromadb  # noqa: E402
from chromadb.config import Settings  # noqa: E402

import rag  # noqa: E402
from config import ANSWER_MODEL  # noqa: E402
from extract import extract_text  # noqa: E402
from human_review import accuracy_line, apply_overrides  # noqa: E402
from llm import get_client, openai_errors  # noqa: E402

METHODS = ("full", "dense", "hybrid")
CATEGORIES = ("answerable", "no_consensus", "unanswerable_related", "figure_only", "out_of_scope")
JUDGE_MODEL = ANSWER_MODEL

# v1: original rules (run 1). v2: approved refuse/no_consensus fixes (run 2), the current rules.
# Known limitation, accepted by the user: v2 sometimes marks a correct no_consensus answer incorrect
# for containing true extra details (q20, three times). A reworded v3 and a structured variant did
# not fix it reliably, so judge changes stopped here; such cases are corrected in
# eval/human_overrides.json instead.
JUDGE_VERSION = "whole-paper v2"

JUDGE_RULES = """You are a strict grader for a question-answering system that answers questions about one research paper.
You get the paper text (the reference), the question, the expected behaviour, the facts a correct answer must include,
a reference answer, and the system's answer. Decide if the system's answer is correct.

Only must_include is required. The reference answer is context to help you understand the question; never mark an
answer incorrect for leaving out something that is in the reference answer but not in must_include.

Rules by expected behaviour:
- "answer": correct only if the answer contains EVERY must_include fact. Formatting differences are fine: different
  dash characters, thousands separators, a missing unit when it is clear from context, a different language.
  Extra details are fine if they are true according to the paper. Any wrong, contradicted, or invented number or
  claim makes the answer incorrect. Saying the information is not in the paper is incorrect.
- "refuse": correct if the answer says, in any wording or language, that the paper does not contain (or does not
  give enough information for) what was asked. It does NOT need to explain why, say the question is off-topic, or
  mention related facts. Mentioning related true facts from the paper is allowed (e.g. the 0-18 age range when
  asked for the mean age). It is incorrect only if it gives a number or an answer to the question itself, or states
  something false.
- "no_consensus": correct only if the answer says there is no consensus (or no definitive answer) AND contains every
  must_include fact. If must_include holds only the no-consensus statement, saying there is no consensus is enough.
  Extra details must be true according to the paper. A plain refusal is incorrect.

Respond ONLY with JSON: {"verdict": "correct" | "incorrect", "reason": "<one short sentence>"}

<paper>
%s
</paper>"""


def _flat(text: str) -> str:
    return " ".join(text.split())


def _evidence_position(question: dict, paper_flat: str) -> str | None:
    """Where the first evidence phrase sits in the paper: beginning, middle or end."""
    for phrase in question["evidence"]:
        position = paper_flat.find(_flat(phrase))
        if position != -1:
            fraction = position / len(paper_flat)
            return "beginning" if fraction < 1 / 3 else "middle" if fraction < 2 / 3 else "end"
    return None


def _context(method: str, question: str, paper_id: str, paper_text: str) -> list[rag.RetrievedChunk]:
    if method == "full":
        return [rag.RetrievedChunk(index=0, text=paper_text, score=None)]
    results = rag.retrieve(paper_id, question, method=method)
    results = rag.rerank(question, results)
    return rag.expand_context(paper_id, results)


def _judge(client, system_prompt: str, question: dict, answer: str) -> dict:
    item = {
        "question": question["question"],
        "expected_behavior": question["expected_behavior"],
        "must_include": question["must_include"],
        "reference_answer": question["expected_answer"],
        "system_answer": answer,
    }
    with openai_errors():
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(item, ensure_ascii=False, indent=1)},
            ],
        )
    data = json.loads(response.choices[0].message.content or "{}")
    verdict = data.get("verdict")
    return {
        "verdict": verdict if verdict in ("correct", "incorrect") else "invalid",
        "reason": str(data.get("reason", "")),
        "judge_prompt_tokens": response.usage.prompt_tokens,
        "judge_cached_tokens": getattr(response.usage.prompt_tokens_details, "cached_tokens", 0) or 0,
    }


def _accuracy(rows: list[dict]) -> str:
    if not rows:
        return "   -   "
    correct = sum(row["verdict"] == "correct" for row in rows)
    return f"{correct:2}/{len(rows):<2} {100 * correct / len(rows):3.0f}%"


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _print_report(rows: list[dict], results_name: str | None = None) -> None:
    # Accuracy as judged, then after the human corrections in eval/human_overrides.json;
    # the tables below use the corrected verdicts.
    if results_name:
        judged = rows
        rows, applied = apply_overrides(rows, results_name)
        print(f"\n=== Judge vs human review ({applied} override(s) applied) ===\n")
        for method in METHODS:
            keep = lambda rs, m=method: [r for r in rs if r["method"] == m]  # noqa: E731
            print(accuracy_line(f"  {method:7}", keep(judged), keep(rows)))
    by_method = {method: [row for row in rows if row["method"] == method] for method in METHODS}

    print("\n=== Accuracy (judge: correct / total) ===\n")
    groups = [("overall", lambda r: True)]
    groups += [(category, lambda r, c=category: r["category"] == c) for category in CATEGORIES]
    groups += [(f"lang={language}", lambda r, lang=language: r["language"] == lang) for language in ("en", "ar")]
    print(f"{'group':22}" + "".join(f"{method:>14}" for method in METHODS))
    for name, keep in groups:
        print(f"{name:22}" + "".join(f"{_accuracy([r for r in by_method[m] if keep(r)]):>14}" for m in METHODS))

    print("\n=== Cost and speed (means per question) ===\n")
    print(f"{'metric':22}" + "".join(f"{method:>14}" for method in METHODS))
    for metric, key in [("prompt tokens", "prompt_tokens"), ("cached tokens", "cached_tokens"),
                        ("completion tokens", "completion_tokens"), ("retrieval seconds", "retrieval_s"),
                        ("generation seconds", "generation_s"), ("total seconds", "total_s")]:
        print(f"{metric:22}" + "".join(f"{_mean([r[key] for r in by_method[m]]):>14.2f}" for m in METHODS))

    print("\n=== Accuracy by evidence position in the paper (questions with an answer) ===\n")
    print(f"{'position':22}" + "".join(f"{method:>14}" for method in METHODS))
    for position in ("beginning", "middle", "end"):
        print(f"{position:22}" + "".join(
            f"{_accuracy([r for r in by_method[m] if r['evidence_position'] == position]):>14}" for m in METHODS))


def _write_judge_review(rows: list[dict], sample_size: int, seed: int, path: Path) -> None:
    sample = random.Random(seed).sample(rows, min(sample_size, len(rows)))
    lines = [
        "# مراجعة أحكام الـ judge",
        "",
        f"{len(sample)} أحكام مختارة عشوائيًا (seed={seed}) من {len(rows)} حكم في `eval/results/rag_vs_full.json`. "
        "يتولّد من `eval/run_rag_vs_full.py`.",
        "",
        "**وش تراجعين:** هل الحكم صحيح حسب قواعد التقييم؟ وهل السبب منطقي؟ أضيفي ✅ أو ❌ جنب كل حكم.",
        "",
    ]
    for number, row in enumerate(sample, start=1):
        must = "، ".join(row["must_include"]) or "فاضي (المفروض يرفض)"
        lines += [
            f"## {number}. {row['id']} — الطريقة: `{row['method']}`",
            "",
            f"- **السؤال:** {row['question']}",
            f"- **الفئة / السلوك المتوقع:** {row['category']} / `{row['expected_behavior']}`",
            f"- **must_include:** {must}",
            f"- **found:** {row['found']}",
            f"- **جواب النظام:** {row['answer']}",
            f"- **حكم الـ judge:** **{row['verdict']}**",
            f"- **السبب:** {row['reason']}",
            "- **مراجعتك:** ",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions", default=str(PROJECT_ROOT / "eval" / "questions.json"))
    parser.add_argument("--sample-size", type=int, default=10, help="Judgments to copy into judge_review.md.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--report-only", metavar="RESULTS_JSON", help="Print the report for a saved run. No API calls.")
    args = parser.parse_args()

    if args.report_only:
        path = Path(args.report_only)
        _print_report(json.loads(path.read_text(encoding="utf-8")), path.name)
        return

    dataset = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    extraction = extract_text(PROJECT_ROOT / dataset["paper"])
    paper_flat = _flat(extraction.text)

    # Index into memory so the experiment never adds collections to chroma_db/.
    rag._client = chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))
    paper_id, _, _ = rag.index_paper(extraction, Path(dataset["paper"]).name, store_text=False)

    client = get_client()
    judge_prompt = JUDGE_RULES % extraction.text
    rows = []
    for method in METHODS:
        for question in dataset["questions"]:
            started = time.perf_counter()
            context = _context(method, question["question"], paper_id, extraction.text)
            retrieved = time.perf_counter()
            usage = {}
            try:
                answer, found, used = rag.generate_answer(question["question"], context, usage_out=usage)
                error = None
            except rag.AnswerError as exc:
                # Record and keep going; the row counts as incorrect.
                answer, found, used, error = "", False, [], str(exc)
            finished = time.perf_counter()

            row = {
                "method": method,
                "id": question["id"],
                "question": question["question"],
                "language": question["language"],
                "category": question["category"],
                "expected_behavior": question["expected_behavior"],
                "must_include": question["must_include"],
                "evidence_position": _evidence_position(question, paper_flat),
                "answer": answer,
                "found": found,
                "context_chunks": [chunk.index for chunk in context],
                "cited_chunks": [chunk.index for chunk in used],
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "cached_tokens": usage.get("cached_tokens", 0),
                "retrieval_s": round(retrieved - started, 3),
                "generation_s": round(finished - retrieved, 3),
                "total_s": round(finished - started, 3),
            }
            if error:
                row.update(verdict="error", reason=error, judge_prompt_tokens=0, judge_cached_tokens=0)
            else:
                row.update(_judge(client, judge_prompt, question, answer))
            rows.append(row)
            print(f"[{method:6}] {question['id']:4} {row['verdict']:9} found={found!s:5} "
                  f"tokens={row['prompt_tokens']:>5} (cached {row['cached_tokens']:>5}) {row['total_s']:5.1f}s", flush=True)

    _print_report(rows, "rag_vs_full.json")

    results_dir = PROJECT_ROOT / "eval" / "results"
    results_dir.mkdir(exist_ok=True)
    (results_dir / "rag_vs_full.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_judge_review(rows, args.sample_size, args.seed, PROJECT_ROOT / "eval" / "judge_review.md")
    judge_tokens = sum(row["judge_prompt_tokens"] for row in rows)
    judge_cached = sum(row["judge_cached_tokens"] for row in rows)
    print(f"\nJudge prompt tokens: {judge_tokens} (cached {judge_cached})")
    print("Wrote eval/results/rag_vs_full.json and eval/judge_review.md")


if __name__ == "__main__":
    main()
