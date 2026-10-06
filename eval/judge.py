"""LLM-as-a-judge that grades an answer against excerpts of the paper (not the whole paper).

Used by eval/run_needle.py. Rules approved by the user: the judge sees only the
passages around the expected evidence plus the passages the answer cited;
equivalent wording counts as the same fact; claims the excerpts cannot verify do
not change the verdict but are listed in unverified_claims.
(eval/run_rag_vs_full.py keeps its own earlier judge, which saw the whole paper.)
"""

import json

from config import ANSWER_MODEL
from llm import openai_errors

JUDGE_MODEL = ANSWER_MODEL

JUDGE_RULES = """You are a strict grader for a question-answering system that answers questions about one research paper.
You get excerpts from the paper (the reference): the passages around the expected evidence and the passages the
system cited. You also get the question, the expected behaviour, the facts a correct answer must include, a
reference answer, and the system's answer. Decide if the system's answer is correct.

Only must_include is required. The reference answer is context to help you understand the question; never mark an
answer incorrect for leaving out something that is in the reference answer but not in must_include.

Equivalent wording counts as the same fact: synonyms, paraphrases, translations, and alternative technical terms
in any language (e.g. "تحليل تلوي" = "تحليل تجميعي" = "meta-analysis"; "دراسات عرضية" = "دراسات مقطعية" =
"cross-sectional studies").

Rules by expected behaviour:
- "answer": correct only if the answer contains EVERY must_include fact. Formatting differences are fine: different
  dash characters, thousands separators, a missing unit when it is clear from context, a different language.
  Any number or claim that the excerpts contradict makes the answer incorrect. Saying the information is not in
  the paper is incorrect.
- "refuse": correct if the answer says, in any wording or language, that the paper does not contain (or does not
  give enough information for) what was asked. It does NOT need to explain why, say the question is off-topic, or
  mention related facts. Mentioning related facts is allowed. It is incorrect if it gives a number or an answer to
  the question itself, or states something the excerpts contradict.
- "no_consensus": correct only if the answer says there is no consensus (or no definitive answer) AND contains every
  must_include fact. If must_include holds only the no-consensus statement, saying there is no consensus is enough.
  Any claim the excerpts contradict makes it incorrect. A plain refusal is incorrect.

Unverifiable details: if the answer states a specific fact that the excerpts neither support nor contradict, do NOT
change the verdict because of it. List each such fact in "unverified_claims" (a short quote or paraphrase).

Respond ONLY with JSON:
{"verdict": "correct" | "incorrect", "reason": "<one short sentence>", "unverified_claims": ["..."]}"""


def judge(client, question: dict, question_text: str, answer: str, excerpts: list[str]) -> dict:
    """Grade one answer. `excerpts` are the paper passages the judge may rely on."""
    item = {
        "question": question_text,
        "expected_behavior": question["expected_behavior"],
        "must_include": question["must_include"],
        "reference_answer": question["expected_answer"],
        "system_answer": answer,
        "excerpts": excerpts,
    }
    with openai_errors():
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": JUDGE_RULES},
                {"role": "user", "content": json.dumps(item, ensure_ascii=False, indent=1)},
            ],
        )
    data = json.loads(response.choices[0].message.content or "{}")
    verdict = data.get("verdict")
    unverified = data.get("unverified_claims") or []
    return {
        "verdict": verdict if verdict in ("correct", "incorrect") else "invalid",
        "reason": str(data.get("reason", "")),
        "unverified_claims": [str(claim) for claim in unverified] if isinstance(unverified, list) else [str(unverified)],
        "judge_prompt_tokens": response.usage.prompt_tokens,
        "judge_completion_tokens": response.usage.completion_tokens,
    }
