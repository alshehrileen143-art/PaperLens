"""Tests for the number check on quotes (number_check.py, rag.check_quote_numbers, /ask). No API.

Run: .\\venv\\Scripts\\python.exe -m unittest tests.test_number_check -v
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

import main  # noqa: E402
import rag  # noqa: E402
from number_check import NUMBER_MISMATCH, cited_sentences, missing_numbers, numbers_in  # noqa: E402
from rag import AnswerResult, RetrievedChunk  # noqa: E402

CHILDREN = "A total of 24,600 children were included."


def check(answer: str, quote: str, details: str | None = None, added: bool = False) -> dict:
    return missing_numbers(answer, details, {1: quote}, markers_added=added)


class NumbersInTests(unittest.TestCase):
    def test_normalization(self):
        self.assertEqual(numbers_in("٢٠ دراسة و٢٤٬٦٠٠ طفل"), {"20", "24600"})
        self.assertEqual(numbers_in("A total of 24,600 children"), {"24600"})
        self.assertEqual(numbers_in("RR 1·41 (95% CI 1.26–1.59)"), {"1.41", "95", "1.26", "1.59"})
        self.assertEqual(numbers_in("p = ٠٫٠٢٧"), {"0.027"})

    def test_citation_markers_are_not_numbers(self):
        self.assertEqual(numbers_in("Included [1] and [2,3] and [2, 3]."), set())
        self.assertEqual(numbers_in("20 studies [1]."), {"20"})

    def test_thousands_or_decimal_comma(self):
        self.assertEqual(numbers_in("24,600 and 1,234,567"), {"24600", "1234567"})
        self.assertEqual(numbers_in("p < 0,01 and 1,5 and 12,3456"), {"0.01", "1.5", "12.3456"})
        self.assertEqual(numbers_in("٢٤٬٦٠٠"), {"24600"})

    def test_reference_numbers_after_al_are_ignored(self):
        self.assertEqual(numbers_in("Turer et al.27 reported 2478"), {"2478"})
        self.assertEqual(numbers_in("Zhang et al12 and Wakayo et al.13"), set())
        self.assertEqual(numbers_in("et al. 27 children"), {"27"})  # Only numbers stuck to "al".

    def test_decimal_and_integer_differ(self):
        self.assertEqual(numbers_in("1.41"), {"1.41"})
        self.assertEqual(numbers_in("141"), {"141"})


class MissingNumbersTests(unittest.TestCase):
    # The cases from the request.

    def test_real_case_arabic_answer_children_quote(self):
        self.assertEqual(check("تم تضمين 20 دراسة في التحليل [1]", CHILDREN), {1: ["20"]})

    def test_supported_number(self):
        self.assertEqual(check("20 studies were included [1]", "20 cross-sectional studies"), {})

    def test_arabic_indic_digits(self):
        self.assertEqual(check("٢٠ دراسة [1]", "20 studies"), {})

    def test_thousands_comma(self):
        self.assertEqual(check("24600 children were included [1].", CHILDREN), {})

    def test_citation_number_not_counted(self):
        self.assertEqual(check("The studies agree [1].", "Several studies agree."), {})

    def test_decimal_vs_integer_mismatch(self):
        self.assertEqual(check("The RR was 1.41 [1].", "RR 141 in total"), {1: ["1.41"]})

    def test_sentence_without_numbers_unchanged(self):
        self.assertEqual(check("Publication bias was present [1].", CHILDREN), {})

    # Sentences, details and the end-of-answer fallback.

    def test_only_the_citing_sentence_counts(self):
        answer = "The review included 20 studies. Publication bias was present [1]."
        self.assertEqual(check(answer, "A funnel plot showed publication bias."), {})

    def test_marker_after_the_full_stop_belongs_to_the_sentence_before(self):
        self.assertEqual(check("A total of 20 studies were included. [1]", CHILDREN), {1: ["20"]})
        self.assertEqual(cited_sentences("First 1.2. Second 3.4. [1]"), {1: ["Second 3.4. [1]"]})

    def test_abbreviations_do_not_end_a_sentence(self):
        answer = "عدد الأطفال في دراسة Turer et al. كان 2478 طفلاً [1]."
        self.assertEqual(cited_sentences(answer), {1: [answer]})
        self.assertEqual(check("In 2018 Turer et al. found it [1].", "Turer et al. found it."), {1: ["2018"]})

    def test_fallback_markers_use_the_whole_answer(self):
        answer = "The review included 20 studies. It found bias. [1]"
        self.assertEqual(check(answer, "Bias was found.", added=True), {1: ["20"]})
        self.assertEqual(check("It found bias. Also 7 more. [1]", "Bias was found.", added=False), {1: ["7"]})

    def test_details_are_checked(self):
        self.assertEqual(check("Bias was present [1].", "Publication bias was present.",
                               details="Heterogeneity was 89% [1]."), {1: ["89"]})

    def test_decimal_comma_matches_decimal_point(self):
        self.assertEqual(check("The difference was significant (p < 0.01) [1].", "(I2 = 89%, p < 0,01)"), {})

    def test_reference_number_in_the_answer_is_not_required(self):
        answer = "The study that contributed most was Turer et al.27 [1]."
        self.assertEqual(check(answer, "The study that contributed the most was Turer et al."), {})

    def test_digits_inside_words_count(self):
        # As specified, every digit run is a number: "I2" gives 2, so the quote must have it too.
        self.assertEqual(check("The I2 was 89% [1].", "high heterogeneity (I2 = 89%)"), {})
        self.assertEqual(check("The I2 was 89% [1].", "heterogeneity of 89%"), {1: ["2"]})

    def test_several_sources_in_one_sentence(self):
        quotes = {1: "RR 1.41 (95% CI: 1.26–1.59)", 2: CHILDREN, 3: "20 studies were included"}
        answer = "The RR was 1.41 (1.26–1.59) [1] among 24,600 children [2] from 20 studies [3]."
        self.assertEqual(missing_numbers(answer, None, quotes), {})
        answer = "The RR was 1.41 [1] among 20 children [2] from 20 studies [3]."
        self.assertEqual(missing_numbers(answer, None, quotes), {2: ["20"]})
        # Words after the last marker go to it.
        self.assertEqual(missing_numbers("The RR was 1.41 [1], in 7 trials [3] of 9.", None, quotes),
                         {3: ["7", "9"]})

    def test_group_real_case_quote_without_any_number_fails(self):
        # "20 دراسة [1][2][3]": [1] is about 24,600 children, [2] and [3] hold 20.
        quotes = {1: CHILDREN, 2: "20 studies were selected", 3: "The 20 studies were cross-sectional."}
        self.assertEqual(missing_numbers("تم تضمين 20 دراسة في التحليل [1][2][3]", None, quotes), {1: ["20"]})

    def test_group_each_quote_holds_part_of_the_numbers(self):
        quotes = {1: "RR 1.41 (95% CI: 1.26–1.59)", 2: CHILDREN}
        self.assertEqual(missing_numbers("RR 1.41 in 24,600 children [1][2].", None, quotes), {})
        self.assertEqual(missing_numbers("RR 1.41 in 24,600 children [1, 2].", None, quotes), {})

    def test_group_quote_that_contributes_nothing_fails(self):
        quotes = {1: "RR 1.41 (95% CI: 1.26–1.59)", 2: CHILDREN}
        # [1] has neither 1.5 nor 24,600: it fails with all the claim's numbers missing.
        self.assertEqual(missing_numbers("An RR of 1.5 among 24,600 children [1][2].", None, quotes),
                         {1: ["1.5", "24600"]})

    def test_group_gap_when_every_quote_contributes(self):
        # Known gap (as specified): both quotes contribute a number, so neither fails,
        # although together they do not cover 20.
        quotes = {1: "RR 1.41 (95% CI: 1.26–1.59)", 2: CHILDREN}
        self.assertEqual(missing_numbers("RR 1.41 in 24,600 children from 20 studies [1][2].", None, quotes), {})

    def test_single_citation_unchanged(self):
        self.assertEqual(missing_numbers("RR 1.41 in 24,600 children [1].", None, {1: "RR 1.41"}), {1: ["24600"]})
        self.assertEqual(missing_numbers("RR 1.41 [1].", None, {1: "RR 1.41"}), {})

    def test_source_without_quote_is_not_checked(self):
        quotes = {1: "RR 1.41", 2: None}
        self.assertEqual(missing_numbers("The RR was 1.41 [1]. And 3 more [2].", None, quotes), {})
        # A group with an unquoted source: that source may hold the other numbers, so the
        # quoted one only needs one of them.
        self.assertEqual(missing_numbers("RR 1.41 in 3 trials [1][2].", None, quotes), {})
        self.assertEqual(missing_numbers("RR 1.5 in 3 trials [1][2].", None, quotes), {1: ["1.5", "3"]})


class CheckQuoteNumbersTests(unittest.TestCase):
    def source(self, quote):
        return RetrievedChunk(index=4, text="Results text.", score=None, page=2, page_end=2, quote=quote)

    def test_mismatch_makes_the_quote_unverified(self):
        [checked] = rag.check_quote_numbers("A total of 20 studies were included. [1]", None,
                                            [self.source(CHILDREN)], markers_added=True)
        self.assertIsNone(checked.quote)
        self.assertEqual((checked.quote_issue, checked.missing_numbers, checked.rejected_quote),
                         (NUMBER_MISMATCH, ["20"], CHILDREN))

    def test_group_real_case_through_ask(self):
        sources = [self.source(CHILDREN), self.source("20 studies were selected"),
                   self.source("The 20 studies were cross-sectional.")]
        checked = rag.check_quote_numbers("تم تضمين 20 دراسة في التحليل [1][2][3]", None, sources)
        self.assertEqual([(c.quote, c.quote_issue) for c in checked],
                         [(None, NUMBER_MISMATCH), ("20 studies were selected", None),
                          ("The 20 studies were cross-sectional.", None)])
        self.assertEqual(checked[0].rejected_quote, CHILDREN)

    def test_supported_quote_unchanged(self):
        source = self.source("A total of 20 studies were included.")
        self.assertEqual(rag.check_quote_numbers("20 studies were included [1].", None, [source]), [source])


class AskResponseTests(unittest.TestCase):
    def test_fields_in_the_response(self):
        source = RetrievedChunk(index=4, text="A total of 24,600 children were included.", score=None, page=2,
                                page_end=2, title="Results", quote=None, quote_issue=NUMBER_MISMATCH,
                                missing_numbers=["20"], rejected_quote=CHILDREN)
        result = AnswerResult("A total of 20 studies were included. [1]", True, [source], "full")
        with patch.object(main, "answer_question", return_value=result):
            body = TestClient(main.app).post("/ask", json={"paper_id": "x" * 32, "question": "How many?"}).json()
        [returned] = body["sources"]
        self.assertIsNone(returned["quote"])
        self.assertEqual((returned["quote_issue"], returned["missing_numbers"], returned["rejected_quote"]),
                         (NUMBER_MISMATCH, ["20"], CHILDREN))


if __name__ == "__main__":
    unittest.main()
