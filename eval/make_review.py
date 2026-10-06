"""Build eval/review.md from eval/questions.json for a human to check the labels.

For each question it shows the category, expected behaviour, must_include facts,
expected answer, and every evidence phrase in its surrounding paper text.
No API calls. Run from the project root:
    venv\\Scripts\\python eval\\make_review.py
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from extract import extract_text  # noqa: E402
from rag import chunk_text  # noqa: E402

CATEGORY_LABELS = {
    "answerable": "له جواب",
    "no_consensus": "لا إجماع",
    "unanswerable_related": "مرتبط وما له جواب",
    "figure_only": "الجواب داخل رسم فقط",
    "out_of_scope": "خارج الموضوع",
}
BEHAVIOR_LABELS = {"answer": "يجاوب", "no_consensus": "يقول ما فيه إجماع + اللي تقترحه الورقة", "refuse": "يرفض"}
CONTEXT_CHARS = 200


def _flat(text: str) -> str:
    return " ".join(text.split())


def main() -> None:
    dataset = json.loads((PROJECT_ROOT / "eval" / "questions.json").read_text(encoding="utf-8"))
    extraction = extract_text(PROJECT_ROOT / dataset["paper"])
    chunks = chunk_text(extraction.text, extraction.pages)
    paper = _flat(extraction.text)

    lines = [
        "# مراجعة مجموعة التقييم (eval/questions.json)",
        "",
        f"الورقة: `{dataset['paper']}`. هذا الملف يتولّد من `eval/make_review.py`؛ عدّلي `questions.json` ثم أعيدي توليده.",
        "",
        "**وش تراجعين:** هل التصنيف والسلوك المتوقع صحيحين؟ هل `must_include` فيه الحقائق الأساسية بس (لا ناقص ولا زيادة)؟ "
        "هل الجواب المتوقع صحيح؟ هل الأسئلة اللي المفروض يرفضها النظام فعلًا ما لها جواب في النص المستخرج؟",
        "",
        "## ملخص",
        "",
        "| # | اللغة | التصنيف | السلوك المتوقع | must_include | السؤال |",
        "|---|---|---|---|---|---|",
    ]
    for question in dataset["questions"]:
        must = "؛ ".join(question["must_include"]) or "—"
        lines.append(
            f"| {question['id']} | {question['language']} | {CATEGORY_LABELS[question['category']]} | "
            f"{BEHAVIOR_LABELS[question['expected_behavior']]} | {must} | {question['question']} |"
        )

    lines += ["", "## التفاصيل", ""]
    for question in dataset["questions"]:
        lines += [
            f"### {question['id']} — {CATEGORY_LABELS[question['category']]} ({question['language']})",
            "",
            f"**السؤال:** {question['question']}",
            "",
            f"**السلوك المتوقع:** `{question['expected_behavior']}` ({BEHAVIOR_LABELS[question['expected_behavior']]})",
            "",
        ]
        if question["must_include"]:
            lines.append("**must_include:**")
            lines += [f"- {fact}" for fact in question["must_include"]]
        else:
            lines.append("**must_include:** فاضي (المفروض يرفض)")
        lines += ["", f"**الجواب المتوقع:** {question['expected_answer']}", ""]

        for phrase in question["evidence"]:
            phrase = _flat(phrase)
            position = paper.find(phrase)
            if position == -1:
                lines += [f"**Evidence (غير موجود في النص!):** {phrase}", ""]
                continue
            before = paper[max(0, position - CONTEXT_CHARS) : position]
            after = paper[position + len(phrase) : position + len(phrase) + CONTEXT_CHARS]
            lines += ["**Evidence:**", "", f"> …{before}**{phrase}**{after}…", ""]

        if question["relevant_chunks"]:
            pages = sorted({chunks[i].page for i in question["relevant_chunks"]} | {chunks[i].page_end for i in question["relevant_chunks"]})
            lines.append(f"**الأجزاء:** {question['relevant_chunks']} · **الصفحات:** {pages}")
        if question.get("note"):
            lines += ["", f"**ملاحظة:** {question['note']}"]
        lines += ["", "---", ""]

    output = PROJECT_ROOT / "eval" / "review.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {output} ({len(dataset['questions'])} questions)")


if __name__ == "__main__":
    main()
