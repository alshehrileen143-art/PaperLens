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

    def test_colon_as_decimal_point(self):
        self.assertEqual(numbers_in("(p <0 :001) and OR 2:49"), {"0.001", "2.49"})
        self.assertEqual(numbers_in("p < 0.001"), {"0.001"})
        # A space after the colon is a label, not a decimal: "Table 1: 20 patients".
        self.assertEqual(numbers_in("Table 1: 20 patients"), {"1", "20"})
        # Times and ratios read the same on both sides, so they still match themselves.
        self.assertEqual(check("At 10:30 the ratio was 1:2 [1].", "measured at 10:30, ratio 1:2"), {})

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


class GroundingColonTests(unittest.TestCase):
    """The false grounding_warning on the migraine paper: "0.001" against "p <0 :001"."""

    def test_colon_decimal_in_the_source_is_not_a_missing_number(self):
        from grounding import check_grounding
        source = ("Females were shown to be signi ﬁcantly more likely to\nhave both migraine and IBS in the "
                  "bivariate analysis\n(p <0 :001), and in the multivariate regression analysis,\n")
        # As in rag._answer, the citation markers are removed before the check.
        answer = rag.strip_citations("The association was significant (p < 0.001) [1].")
        result = check_grounding("What was the p-value for the association?", answer, [source], 1)
        self.assertTrue(result.ok, result.describe())
        self.assertEqual(result.describe(), "")

    def test_a_really_missing_number_still_warns(self):
        from grounding import check_grounding
        result = check_grounding("What was the p-value?", rag.strip_citations("The p-value was 0.002 [1]."),
                                 ["(p <0 :001)"], 1)
        self.assertFalse(result.ok)
        self.assertIn("0.002", result.describe())


class NumberWordsAndNamesTests(unittest.TestCase):
    """Numbers written in English words, and digits inside a name such as "R4DQ"."""

    def test_number_words(self):
        self.assertEqual(numbers_in("Two thousand eight hundred two participants"), {"2802"})
        self.assertEqual(numbers_in("Seven hundred sixty-seven were found"), {"767"})
        self.assertEqual(numbers_in("twenty studies"), {"20"})
        self.assertEqual(numbers_in("twenty-one, one hundred and five"), {"21", "105"})
        self.assertEqual(numbers_in("two and three groups"), {"2", "3"})  # "and" joins only after a scale.
        self.assertEqual(numbers_in("one of the studies, no one"), set())  # A lone "one" is a word.
        self.assertEqual(numbers_in("hundreds of studies"), set())

    def test_twenty_does_not_match_21(self):
        self.assertEqual(check("There were 21 studies [1].", "twenty studies were selected"), {1: ["21"]})
        self.assertEqual(check("There were 20 studies [1].", "twenty studies were selected"), {})

    def test_digits_inside_a_word_are_a_name(self):
        self.assertEqual(numbers_in("the Rome IV Diagnostic Questionnaire (R4DQ)"), set())
        self.assertEqual(numbers_in("H2O"), set())
        # Digits at the end of a word still count.
        self.assertEqual(numbers_in("I2 = 89%"), {"2", "89"})
        self.assertEqual(numbers_in("PP-OCRv6"), {"6"})
        self.assertEqual(check("The I2 was 89% [1].", "heterogeneity of 89%"), {1: ["2"]})

    def test_real_case_q2_number_in_words(self):
        quote = ("Two thousand eight hundred two participants submitted the survey, and all participants "
                 "were eligible and accounted for in the analysis.")
        self.assertEqual(check("عدد المشاركين في الدراسة كان 2802. [1]", quote, added=True), {})

    def test_real_case_q4_acronym(self):
        answer = ("استخدمت الدراسة استبيانات تشمل استبيان شاشة الصداع النصفي (MS-Q)، ومقياس شدة الصداع النصفي "
                  "(MIGSEV)، واستبيان روما الرابع (R4DQ). [1]")
        self.assertEqual(check(answer, "The survey was divided into three sections.", added=True), {})


class SeveralQuotesForOneSourceTests(unittest.TestCase):
    """_cite keeps every verbatim quote of a source; check_quote_numbers uses the first that passes."""

    SOURCE = ("A total of 24,600 children were included. During the search 3155 articles were retrieved, "
              "of which 20 studies were relevant. The search covered three databases.")

    def answer(self, citations, answer_text):
        import json as _json
        from types import SimpleNamespace
        payload = {"answer": answer_text, "details": "", "found": True, "citations": citations}
        message = SimpleNamespace(content=_json.dumps(payload))
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kwargs: SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None))))
        chunks = [RetrievedChunk(index=4, text=self.SOURCE, score=None)]
        extra = {}
        with patch.object(rag, "get_client", return_value=client):
            final, found, used = rag.generate_answer("How many studies?", chunks, extra_out=extra)
        return final, used, extra

    def test_second_quote_used_when_the_first_lacks_the_number(self):
        final, used, extra = self.answer([
            {"source_id": 4, "quote": "A total of 24,600 children were included."},
            {"source_id": 4, "quote": "of which 20 studies were relevant"},
        ], "20 studies were included [4].")
        self.assertEqual(used[0].quote, "A total of 24,600 children were included.")  # First verbatim.
        self.assertEqual(used[0].quote_candidates, ["A total of 24,600 children were included.",
                                                    "of which 20 studies were relevant"])
        [checked] = rag.check_quote_numbers(final, None, used)
        self.assertEqual((checked.quote, checked.quote_issue), ("of which 20 studies were relevant", None))

    def test_first_verbatim_quote_wins_over_a_non_verbatim_first(self):
        _, used, extra = self.answer([
            {"source_id": 4, "quote": "A total of 20 studies were included."},  # Not in the source.
            {"source_id": 4, "quote": "of which 20 studies were relevant"},
        ], "20 studies were included [4].")
        self.assertEqual(used[0].quote, "of which 20 studies were relevant")
        self.assertEqual(extra["removed"], [])

    def test_no_candidate_passes_keeps_the_first_as_rejected(self):
        final, used, _ = self.answer([
            {"source_id": 4, "quote": "A total of 24,600 children were included."},
            {"source_id": 4, "quote": "The search covered three databases."},
        ], "20 studies were included [4].")
        [checked] = rag.check_quote_numbers(final, None, used)
        self.assertEqual((checked.quote, checked.quote_issue, checked.rejected_quote),
                         (None, NUMBER_MISMATCH, "A total of 24,600 children were included."))

    def test_none_verbatim_reports_the_first_as_removed(self):
        _, used, extra = self.answer([
            {"source_id": 4, "quote": "A total of 20 studies were included."},
            {"source_id": 4, "quote": "There were twenty trials."},
        ], "20 studies were included [4].")
        self.assertIsNone(used[0].quote)
        self.assertEqual(extra["removed"], [{"source_id": 4, "quote": "A total of 20 studies were included."}])

    def test_changed_word_is_still_rejected(self):
        _, used, extra = self.answer([{"source_id": 4, "quote": "of which 20 studies were irrelevant"}],
                                     "20 studies were included [4].")
        self.assertIsNone(used[0].quote)
        self.assertEqual(len(extra["removed"]), 1)
