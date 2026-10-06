"""Tests for answer quotes (rag.quote_in_source, rag.generate_answer). OpenAI is mocked.

Run: .\\venv\\Scripts\\python.exe -m unittest tests.test_quotes -v
"""

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rag  # noqa: E402
from rag import RetrievedChunk, quote_in_source  # noqa: E402

SOURCE = ("Serum 25(OH)D levels were signiﬁcantly lower in the\npatient group (p < 0.001) than in con-\n"
          "trols. The ages ranged from 0- - -18 years.")


def fake_client(payload: dict):
    message = SimpleNamespace(content=json.dumps(payload, ensure_ascii=False))
    response = SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)
    create = lambda **kwargs: response  # noqa: E731
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


class QuoteInSourceTests(unittest.TestCase):
    def test_verbatim_after_normalization(self):
        for quote in (
            "levels were significantly lower in the patient group",  # Ligature ﬁ and a line break.
            "than in controls",  # Line-end hyphenation.
            "ranged from 0–18 years",  # Broken PDF dash.
            "“Serum 25(OH)D levels were significantly lower”",  # Wrapped in quotation marks.
            "than in controls.",  # Full stop added to close a part of a sentence.
            "ranged from 0-18 years",  # Hyphen where the source has a dash.
            "serum 25(oh)d levels",  # Case (casefold).
        ):
            with self.subTest(quote):
                self.assertTrue(quote_in_source(quote, SOURCE))

    def test_not_verbatim(self):
        for quote in (
            "levels were much lower in the patient group",  # Paraphrased.
            "levels were ... lower",  # Ellipsis.
            "كانت المستويات أقل",  # Translated.
            "than in controls;",  # Only a full stop is forgiven.
            "lower in the patient group , than",  # Missing words: the source has "(p < 0.001)" here.
            "levels were significantly lower in the patient group (p < 0.01)",  # Number changed.
            "  ",
        ):
            with self.subTest(quote):
                self.assertFalse(quote_in_source(quote, SOURCE))


# Sources and quotes from eval/results/ask_20261005_233225.json (as the PDF text reads).
Q16_SOURCE = ("Studies performed in tropical countries. The study that contributed the most for heterogeneity "
              "was Turer et al.27 A sensitivity analysis was performed.")
Q19_SOURCE = ("The study by Shulhai et al.31 with Ukrainian children and adolescents reported a signiﬁcant effect "
              "in the development of vit D deﬁciency in individuals who spend more than four hours a day in front "
              "of computer or television (p = 0.027). Tw o other studies")
Q8_SOURCE = ("There is no consensus on the universally recommended dose for the treatment of hypovitaminosis D in "
             "this population; however , authors suggest an increase in the dose used regularly.36,37")
Q12_SOURCE = "The statistical analysis was performed using the ‘‘meta’’ package of the R program (R Core Team)."
Q13_SOURCE = ("The summary measure used was relative risk (RR) with 95% conﬁdence interval, weighted by the study "
              "power , using the random effect model.")
Q15_SOURCE = "Regarding the cutoff point for 25(OH)D deﬁciency, the value of 20 ng/mL was the most frequently used."


class RelaxedComparisonTests(unittest.TestCase):
    """Case, curly quotation marks, ligatures and reference numbers after "al." are forgiven; words never."""

    def test_real_cases_that_now_pass(self):
        cases = {
            # PDF artifacts in the source: a space before a comma; doubled quotation marks.
            "q8": ("there is no consensus on the universally recommended dose for the treatment of hypovitaminosis D "
                   "in this population; however, authors suggest an increase in the dose used regularly.", Q8_SOURCE),
            "q12": ("The statistical analysis was performed using the ‘meta’ package of the R program.", Q12_SOURCE),
            "q13": ("The summary measure used was relative risk (RR) with 95% confidence interval, weighted by the "
                    "study power, using the random effect model.", Q13_SOURCE),
            "q16": ("the study that contributed the most for heterogeneity was Turer et al.27", Q16_SOURCE),
            "q19": ("the study by Shulhai et al.31 with Ukrainian children and adolescents reported a significant "
                    "effect in the development of vit D deficiency in individuals who spend more than four hours a "
                    "day in front of computer or television (p = 0.027).", Q19_SOURCE),
        }
        for name, (quote, source) in cases.items():
            with self.subTest(name):
                self.assertTrue(quote_in_source(quote, source))

    def test_real_cases_still_rejected(self):
        cases = {
            # Not verbatim: a different number and different words (an invented quote).
            "q15": ("the value of 10 ng/mL was used by Jari et al.", Q15_SOURCE),
        }
        for name, (quote, source) in cases.items():
            with self.subTest(name):
                self.assertFalse(quote_in_source(quote, source))

    def test_form_differences_pass(self):
        source = "The authors wrote “no consensus” and ‘meta’ in Turer et al.27 with ﬁve signiﬁcant effects."
        for quote in ('the authors wrote "no consensus"', "and 'meta' in turer et al.", "Turer et al. with five",
                      "SIGNIFICANT EFFECTS"):
            with self.subTest(quote):
                self.assertTrue(quote_in_source(quote, source))

    def test_pdf_artifacts_pass(self):
        cases = [
            ("The relative risk (RR) was high [95% CI], and", "The relative risk ( RR ) was high [ 95% CI ] , and"),
            ('the "meta" package; R was used.', 'the ""meta"" package ; R was used .'),
            ("the 'meta' package", "the ‘‘meta’’ package"),
            ("Is it? Yes! Done: now", "Is it ? Yes ! Done : now"),
        ]
        for quote, source in cases:
            with self.subTest(quote):
                self.assertTrue(quote_in_source(quote, source))

    def test_pdf_artifact_rules_do_not_forgive_words(self):
        source = "weighted by the study power , using the random effect model."
        for quote in ("weighted by study power, using the random effect model",  # Missing word.
                      "weighted by the study power, using a random effect model",  # Changed word.
                      "weighted by the study power, and using the random effect model"):  # Extra word.
            with self.subTest(quote):
                self.assertFalse(quote_in_source(quote, source))

    def test_word_changes_are_rejected(self):
        source = "There is no consensus on the dose; the relative risk was 1.41 in obese children."
        for quote in ("There is a consensus on the dose",  # Changed word.
                      "There is no consensus on dose",  # Missing word.
                      "There is no clear consensus on the dose",  # Extra word.
                      "the relative risk was 1.42",  # Changed number.
                      "the relative risk was 1.41 in obese adolescents"):
            with self.subTest(quote):
                self.assertFalse(quote_in_source(quote, source))


class GenerateAnswerTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [RetrievedChunk(index=i, text=text, score=None) for i, text in
                       [(0, "Intro text."), (1, SOURCE), (2, "Methods: a case-control study of 120 children."),
                        (3, "Results: 20 studies."), (4, "Discussion text.")]]

    def answer(self, payload: dict):
        extra = {}
        with patch.object(rag, "get_client", return_value=fake_client(payload)):
            result = rag.generate_answer("What was found?", self.chunks, extra_out=extra)
        return result, extra

    def test_quotes_kept_removed_and_missing(self):
        (answer, found, used), extra = self.answer({
            "answer": "Levels were lower [1], in a case-control study [2] [0].", "details": "", "found": True,
            "citations": [{"source_id": 1, "quote": "“than in controls”"},
                          {"source_id": 2, "quote": "a cohort study of 120 children"},
                          {"source_id": 0}],
        })
        self.assertTrue(found)
        self.assertEqual([(c.index, c.quote) for c in used], [(1, "than in controls"), (2, None), (0, None)])
        self.assertEqual(answer, "Levels were lower [1], in a case-control study [2] [3].")
        self.assertEqual(extra["details"], "")
        self.assertEqual({k: extra[k] for k in ("requested", "cited", "kept", "missing", "removed", "dropped")},
                         {"requested": 3, "cited": 3, "kept": 1, "missing": [0], "dropped": [],
                          "removed": [{"source_id": 2, "quote": "a cohort study of 120 children"}]})
        # The context chunks themselves are not changed.
        self.assertTrue(all(chunk.quote is None for chunk in self.chunks))

    def test_source_ids_become_1_to_n_and_at_most_three(self):
        (answer, _, used), extra = self.answer({
            "answer": "A [2]. B [2, 99]. C [3]. D [4].", "details": "More on B [2] and D [4], not [1].",
            "found": True,
            "citations": [{"source_id": 2, "quote": "a case-control study"},
                          {"source_id": 99, "quote": "unknown id"},  # Not provided: dropped.
                          {"source_id": 3, "quote": "20 studies"},
                          {"source_id": 4, "quote": "Discussion text"},
                          {"source_id": 2, "quote": "120 children"}],  # Same source again: keeps [1].
        })
        self.assertEqual([c.index for c in used], [2, 3, 4])
        self.assertEqual(answer, "A [1]. B [1]. C [2]. D [3].")
        self.assertEqual(extra["details"], "More on B [1] and D [3], not.")  # Source 1 was not cited.
        self.assertEqual(extra["markers"], "model")
        self.assertEqual(extra["requested"], 5)
        self.assertEqual([c["source_id"] for c in extra["dropped"]], [99])
        self.assertIn('"source_id": 99', extra["raw"])

    def test_fourth_source_is_dropped(self):
        (answer, _, used), extra = self.answer({
            "answer": "W [0] X [1] Y [2] Z [3].", "found": True,
            "citations": [{"source_id": i, "quote": ""} for i in (0, 1, 2, 3)],
        })
        self.assertEqual([c.index for c in used], [0, 1, 2])
        self.assertEqual(answer, "W [1] X [2] Y [3] Z.")
        self.assertEqual(len(extra["dropped"]), 1)

    def test_numbers_added_when_the_answer_has_none(self):
        for model_answer in ("20 studies were included.", "20 studies were included [7]."):  # [7]: not cited.
            with self.subTest(model_answer):
                (answer, _, used), extra = self.answer({
                    "answer": model_answer, "details": "From the search [3].", "found": True,
                    "citations": [{"source_id": 3, "quote": "20 studies"}, {"source_id": 2, "quote": "120 children"}],
                })
                self.assertEqual(answer, "20 studies were included. [1][2]")
                self.assertEqual(extra["markers"], "added")
                self.assertEqual(extra["details"], "From the search [1].")

    def test_no_valid_citation_is_not_found(self):
        (answer, found, used), _ = self.answer({"answer": "A [42].", "found": True,
                                                "citations": [{"source_id": 42, "quote": "x"}]})
        self.assertFalse(found)
        self.assertEqual(used, [])

    def test_not_found(self):
        (_, found, used), extra = self.answer({"answer": "", "details": "", "found": False, "citations": []})
        self.assertFalse(found)
        self.assertEqual(used, [])
        self.assertNotIn("details", extra)
        self.assertIn('"found": false', extra["raw"])

    def test_unexpected_format(self):
        for payload in ({"answer": "A", "found": True, "citations": {"1": "x"}},
                        {"answer": "A", "found": True, "citations": [], "details": 3}):
            with self.subTest(payload), self.assertRaises(rag.AnswerError):
                self.answer(payload)


# A paper laid out like the LangChain primer: the quote is on page 3, under a heading from page 2.
PAGES = [
    "Front page\nA. Components\nThe main components are explained.",
    "More about prompts.\nA.A.2 Models\nLarge Language Models (LLMs) are the main\ntype of models. The embed-\ndings are required when we want to work",
    "with our own documents, as discussed in the following section.\nA.A.3 Chains\nThe chain is the key building block.",
]
PAPER = {"text": "\n".join(PAGES), "pages": PAGES}


class LocateSourcesTests(unittest.TestCase):
    def chunk(self, quote, section="Methods"):
        start = PAPER["text"].index("A. Components")
        return RetrievedChunk(index=0, text=PAPER["text"][start:], score=None, page=1, page_end=3,
                              section=section, quote=quote)

    def test_page_and_heading_of_the_quote(self):
        cases = {
            "with our own documents": (3, "A.A.2 Models"),  # Page 3, but the heading is on page 2.
            "the embeddings are required": (2, "A.A.2 Models"),  # Case is forgiven.
            "the embeddings are needed": None,  # Not verbatim (a different word).
            "The embeddings are required when we want to work with our own documents": (2, "A.A.2 Models"),
            "The chain is the key building block.": (3, "A.A.3 Chains"),
            "The main components are explained": (1, "A. Components"),
        }
        for quote, expected in cases.items():
            with self.subTest(quote):
                if expected is None:
                    self.assertIsNone(rag.quote_span(quote, self.chunk(quote).text))
                    continue
                [located] = rag.locate_sources([self.chunk(quote)], PAPER)
                self.assertEqual((located.page, located.page_end, located.title), (expected[0], expected[0], expected[1]))

    def test_without_quote_keeps_page_range(self):
        [located] = rag.locate_sources([self.chunk(None)], PAPER)
        self.assertEqual((located.page, located.page_end, located.title), (1, 3, "A. Components"))

    def test_no_heading_before_uses_section(self):
        chunk = RetrievedChunk(index=0, text="Front page", score=None, page=1, page_end=1, section="Front matter",
                               quote="Front page")
        [located] = rag.locate_sources([chunk], PAPER)
        self.assertEqual((located.page, located.title), (1, "Front matter"))

    def test_without_stored_paper(self):
        chunk = self.chunk("with our own documents")
        self.assertEqual(rag.locate_sources([chunk], None), [chunk])


class HeadingLinesTests(unittest.TestCase):
    def test_headings(self):
        from sections import heading_lines
        text = "\n".join([
            "II. MATERIALS AND METHOD", "LangChain is a framework.",
            "A.A.2 Models", "Large Language Models are used.",
            "A large language model (LLM) is a subtype of", "artificial intelligence.",
            "Table 1. Examples of prompts.", "Task Prompt",
            "2.1 Study design", "We searched three databases.",
            "Search strategy", "The studies were identified.",
            "12 Brazil Cross-sectional 2015",
            "21. Elizondo-Montemayor L, Ugalde-Casas PA", "Serum levels.",
            "3 Results", "and then lowercase continues",
        ])
        self.assertEqual([title for _, title in heading_lines(text)],
                         ["II. MATERIALS AND METHOD", "A.A.2 Models", "2.1 Study design", "Search strategy"])


if __name__ == "__main__":
    unittest.main()
