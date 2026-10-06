"""Human reviews of judge verdicts (eval/human_overrides.json).

apply_overrides() returns result rows with the human verdict where a human
disagreed with the judge, keeping the judge's verdict in "judge_verdict".
Running this file prints the judge's reliability on the reviewed judgments:
    venv\\Scripts\\python eval\\human_review.py
"""

import json
from collections import defaultdict
from pathlib import Path

OVERRIDES_FILE = Path(__file__).resolve().parent / "human_overrides.json"


def load_reviews() -> list[dict]:
    return json.loads(OVERRIDES_FILE.read_text(encoding="utf-8"))["reviews"]


def _matches(row: dict, match: dict) -> bool:
    # Rows from eval/run_ask.py have no "method"; they are /ask answers.
    return all(row.get(key, "ask" if key == "method" else None) == value for key, value in match.items())


def apply_overrides(rows: list[dict], results_file: str | Path) -> tuple[list[dict], int]:
    """Copies of `rows` with human verdicts applied; returns (rows, number of overrides applied)."""
    name = Path(results_file).name
    overrides = [r for r in load_reviews() if r["results_file"] == name and r["human_verdict"] != r["judge_verdict"]]
    corrected, applied = [], 0
    for row in rows:
        row = {**row, "judge_verdict": row["verdict"]}
        for override in overrides:
            if _matches(row, override["match"]):
                row["verdict"] = override["human_verdict"]
                row["override_reason"] = override["reason"]
                applied += 1
        corrected.append(row)
    return corrected, applied


def accuracy_line(label: str, judged: list[dict], corrected: list[dict]) -> str:
    judge_ok = sum(r["verdict"] == "correct" for r in judged)
    human_ok = sum(r["verdict"] == "correct" for r in corrected)
    return f"{label}: judge {judge_ok}/{len(judged)}  |  after human review {human_ok}/{len(corrected)}"


def reliability() -> None:
    by_judge = defaultdict(list)
    for review in load_reviews():
        by_judge[review["judge"]].append(review)
    print("Judge reliability on human-reviewed judgments (biased toward suspicious cases, see notes):\n")
    for judge, reviews in by_judge.items():
        agree = sum(r["human_verdict"] == r["judge_verdict"] for r in reviews)
        wrong_correct = sum(r["judge_verdict"] == "correct" and r["human_verdict"] == "incorrect" for r in reviews)
        wrong_incorrect = sum(r["judge_verdict"] == "incorrect" and r["human_verdict"] == "correct" for r in reviews)
        print(f"  {judge:15} agreed {agree}/{len(reviews)} ({100 * agree / len(reviews):.0f}%)  "
              f"| judged correct but wrong: {wrong_correct}  | judged incorrect but right: {wrong_incorrect}")
    current = [r for r in load_reviews() if r["judge"] != "whole-paper v1"]
    agree = sum(r["human_verdict"] == r["judge_verdict"] for r in current)
    print(f"\n  current judges (v2 + excerpts): agreed {agree}/{len(current)} ({100 * agree / len(current):.0f}%)")


if __name__ == "__main__":
    reliability()
