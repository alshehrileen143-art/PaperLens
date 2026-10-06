"""Measure retrieval quality (recall@k) on the fixed evaluation set.

Retrieval only: the paper is indexed into an in-memory ChromaDB (the project's
chroma_db/ is not touched) and each question goes through retrieve + rerank.
Only the embeddings API is called, plus gpt-4o-mini once per question (shared
by all configs) when a ":rw" config rewrites the queries.

Several configurations can be compared in one run. Each is "<method>:<norm>[:rw|:rw2]":
  method  dense | hybrid       (see config.SEARCH_METHOD)
  norm    raw   | norm         (all normalization rules off / as set in config.py)
  rw      optional: search with the rewritten query (rag.rewrite_query) instead of the question
  rw2     optional: search with both the question and its rewrite, merged with RRF (what /ask does)

Usage (from the project root):
    venv\\Scripts\\python eval\\run_retrieval.py
    venv\\Scripts\\python eval\\run_retrieval.py --configs dense:norm hybrid:norm --json results.json
"""

import argparse
import json
import time
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import chromadb  # noqa: E402
from chromadb.config import Settings  # noqa: E402

import config  # noqa: E402
import rag  # noqa: E402
from extract import extract_text  # noqa: E402
from normalize import build_vocabulary, normalize_text  # noqa: E402

K_VALUES = (3, 5, 10)
DEFAULT_CONFIGS = ("dense:raw", "dense:norm", "hybrid:norm")
REFUSE_CATEGORIES = ("unanswerable_related", "figure_only", "out_of_scope")
NORMALIZE_FLAGS = ("NORMALIZE_NFKC", "NORMALIZE_DASHES", "NORMALIZE_DEHYPHENATE")
CONFIGURED_NORMALIZATION = {flag: getattr(config, flag) for flag in NORMALIZE_FLAGS}


def _flat(text: str) -> str:
    return " ".join(text.split())


def _set_normalization(enabled: bool) -> None:
    for flag in NORMALIZE_FLAGS:
        setattr(config, flag, CONFIGURED_NORMALIZATION[flag] if enabled else False)


def _check_settings(dataset: dict) -> None:
    current = {
        "chunk_size_chars": config.CHUNK_SIZE_CHARS,
        "chunk_overlap_chars": config.CHUNK_OVERLAP_CHARS,
        "embedding_model": config.EMBEDDING_MODEL,
    }
    for key, value in dataset.get("chunking", {}).items():
        if current.get(key) != value:
            print(f"NOTE: {key} is {current.get(key)} now, but the dataset was labelled with {value}.")


def _relevant_chunks(question: dict, normalized_chunks: dict[int, str], vocabulary: set[str], label: str) -> set[int]:
    """Chunks containing any evidence phrase, both passed through normalize_text."""
    phrases = [_flat(normalize_text(phrase, vocabulary)) for phrase in question["evidence"]]
    found = {index for index, text in normalized_chunks.items() for phrase in phrases if phrase in text}
    stored = set(question["relevant_chunks"])
    if question["evidence"] and found != stored:
        print(f"WARNING [{label}] {question['id']}: labelled chunks {sorted(stored)} but evidence is now in "
              f"{sorted(found)}. Using the evidence-based chunks.")
    return found


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _fmt(value: float | None) -> str:
    return "  -  " if value is None else f"{value:5.2f}"


def _index(dataset: dict, extraction, normalize: bool) -> dict:
    """Index the paper with normalization on or off; returns what the questions need."""
    _set_normalization(normalize)
    paper_id, chunks, stats = rag.index_paper(extraction, Path(dataset["paper"]).name, store_text=False)
    vocabulary = build_vocabulary(extraction.text)
    normalized_chunks = {chunk.index: _flat(normalize_text(chunk.text, vocabulary)) for chunk in chunks}
    return {"paper_id": paper_id, "chunks": chunks, "stats": stats, "vocabulary": vocabulary,
            "normalized_chunks": normalized_chunks}


REWRITES: dict[str, dict] = {}  # Question id -> rewritten query, made once and shared by all :rw configs.


def _rewritten(question: dict) -> str:
    if question["id"] not in REWRITES:
        usage, started = {}, time.perf_counter()
        query = rag.rewrite_query(question["question"], usage)
        REWRITES[question["id"]] = {"query": query, "seconds": time.perf_counter() - started, **usage}
    return REWRITES[question["id"]]["query"]


def _evaluate(dataset: dict, index: dict, method: str, label: str, rewrite: str = "") -> list[dict]:
    """rewrite: "" = the question only; "rw" = the rewritten query only; "rw2" = both, merged with RRF."""
    rows = []
    for question in dataset["questions"]:
        query, extra = question["question"], []
        if rewrite == "rw":
            query = _rewritten(question)
        elif rewrite == "rw2":
            extra = [_rewritten(question)]
        # Full ranking (every chunk), so the rank of the right chunk is known even past k.
        results = rag.retrieve(index["paper_id"], query, top_k=len(index["chunks"]), method=method, extra_queries=extra)
        results = rag.rerank(query, results)
        ranking = [result.index for result in results]
        relevant = _relevant_chunks(question, index["normalized_chunks"], index["vocabulary"], label)
        row = {
            "id": question["id"],
            "language": question["language"],
            "category": question["category"],
            "answerable": question["answerable"],
            "relevant_chunks": sorted(relevant),
            "ranking": ranking,
            # Highest cosine similarity of any chunk: a candidate signal for refusing.
            "top_similarity": max(result.score for result in results),
        }
        if question["answerable"]:
            ranks = [ranking.index(chunk) + 1 for chunk in relevant if chunk in ranking]
            row["rank"] = min(ranks) if ranks else None
            for k in K_VALUES:
                row[f"recall@{k}"] = 1.0 if row["rank"] and row["rank"] <= k else 0.0
        rows.append(row)
    return rows


def _print_report(dataset: dict, results: dict[str, list[dict]], stats: dict | None) -> None:
    labels = list(results)
    questions = dataset["questions"]

    if stats:
        print("Normalization rules applied to the paper:", ", ".join(f"{rule}={count}" for rule, count in stats.items()))

    print("\nRank of the first correct chunk (lower is better; questions to refuse show their top similarity):\n")
    header = f"{'id':4} {'lang':4} {'category':21} {'chunks':10} " + " ".join(f"{label:>12}" for label in labels)
    print(header)
    print("-" * len(header))
    for position, question in enumerate(questions):
        cells = []
        for label in labels:
            row = results[label][position]
            if row["answerable"]:
                cells.append(f"{row['rank'] if row['rank'] else '-':>12}")
            else:
                cells.append(f"{'(' + format(row['top_similarity'], '.3f') + ')':>12}")
        print(f"{question['id']:4} {question['language']:4} {question['category']:21} "
              f"{str(results[labels[0]][position]['relevant_chunks']):10} " + " ".join(cells))

    print("\nMean recall over questions with an answer (answerable + no_consensus):\n")
    print(f"{'config':12} {'group':4} {'n':>3}  " + "  ".join(f"R@{k:<3}" for k in K_VALUES))
    for label in labels:
        for group in ("all", "en", "ar"):
            rows = [r for r in results[label] if r["answerable"] and (group == "all" or r["language"] == group)]
            recalls = "  ".join(_fmt(_mean([r[f"recall@{k}"] for r in rows])) for k in K_VALUES)
            print(f"{label:12} {group:4} {len(rows):3}  {recalls}")
        print()

    print("Top similarity by category (for tuning a future refusal threshold):")
    for label in labels:
        rows = results[label]
        answerable = [r["top_similarity"] for r in rows if r["answerable"]]
        parts = [f"answerable min {min(answerable):.3f}"]
        for category in REFUSE_CATEGORIES:
            scores = [r["top_similarity"] for r in rows if r["category"] == category]
            if scores:
                parts.append(f"{category} max {max(scores):.3f}")
        print(f"  {label:12} " + " | ".join(parts))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions", default=str(PROJECT_ROOT / "eval" / "questions.json"))
    parser.add_argument("--configs", nargs="+", default=list(DEFAULT_CONFIGS),
                        help="Configurations to compare, e.g. dense:raw dense:norm hybrid:norm")
    parser.add_argument("--json", help="Also write per-question results to this file.")
    args = parser.parse_args()

    for label in args.configs:
        method, norm, *extra = label.split(":") + [""]
        if method not in ("dense", "hybrid") or norm not in ("raw", "norm") or extra[0] not in ("", "rw", "rw2"):
            parser.error(f"Bad config {label!r}; expected <dense|hybrid>:<raw|norm>[:rw|:rw2].")

    dataset = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    _check_settings(dataset)

    # Index into memory so evaluation never adds collections to chroma_db/.
    rag._client = chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))
    extraction = extract_text(PROJECT_ROOT / dataset["paper"])

    indexes, results = {}, {}
    for label in args.configs:
        method, norm, *extra = label.split(":") + [""]
        normalize = norm == "norm"
        if norm not in indexes:
            indexes[norm] = _index(dataset, extraction, normalize)
        _set_normalization(normalize)  # Questions must be normalized the same way as the chunks.
        results[label] = _evaluate(dataset, indexes[norm], method, label, rewrite=extra[0])
    _set_normalization(True)

    print(f"Indexed {len(next(iter(indexes.values()))['chunks'])} chunks from {dataset['paper']}\n")
    _print_report(dataset, results, indexes["norm"]["stats"] if "norm" in indexes else None)

    if REWRITES:
        seconds = sorted(r["seconds"] for r in REWRITES.values())
        prompt = sum(r.get("prompt_tokens", 0) for r in REWRITES.values())
        completion = sum(r.get("completion_tokens", 0) for r in REWRITES.values())
        cost = (prompt * 0.15 + completion * 0.60) / 1e6
        print(f"\nQuery rewriting: {len(REWRITES)} calls, median {seconds[len(seconds) // 2]:.2f}s each, "
              f"{prompt} + {completion} tokens, ${cost:.4f}")
        print("\nRewritten queries:")
        for qid, r in REWRITES.items():
            print(f"  {qid:4} {r['query']}")

    if args.json:
        Path(args.json).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
