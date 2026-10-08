"""Tests for the display-only text helpers in ui_text.py (no API, no Streamlit).

Run: .\\venv\\Scripts\\python.exe -m unittest tests.test_ui_text -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ui_text import (TEXTS, clip, isolate, list_html, nearest_sentence, paper_html, pdf_paragraphs,  # noqa: E402
                     rich_html, shorten_filename, shown_citations, source_html, table_html)


class ShortenFilenameTests(unittest.TestCase):
    def test_long_name_keeps_start_and_extension(self):
        name = "CreatingLargeLanguageModelApplicationsUtilizingLangChain-APrimeronDevelopingLLMAppsFast.pdf"
        short = shorten_filename(name)
        self.assertLessEqual(len(short), 36)
        self.assertTrue(short.startswith("CreatingLarge"))
        self.assertTrue(short.endswith("AppsFast.pdf"))
        self.assertIn("…", short)

    def test_short_name_unchanged(self):
        self.assertEqual(shorten_filename("paper.pdf"), "paper.pdf")

    def test_no_extension(self):
        self.assertEqual(len(shorten_filename("x" * 80)), 36)


class PdfParagraphsTests(unittest.TestCase):
    def test_joins_wrapped_lines_and_hyphenated_words(self):
        text = ("The past decade has witnessed an unparalleled evolution in the realm of artificial\n"
                "intelligence (AI). This period, characterized by the ascendancy of deep learning con-\n"
                "trols the field. GPT-1 (June 2018), GPT-\n"
                "2 (February 2019), and GPT-3 had many parameters in their language models, respectively.\n"
                "Short last line.\n"
                "A new paragraph starts here and goes on for a while to fill a typical line width.\n"
                "• Document Loaders: load documents from many sources\n"
                "and formats.\n"
                "• Retrievers: query your data.")
        paragraphs = pdf_paragraphs(text)
        self.assertEqual(len(paragraphs), 4)
        self.assertIn("artificial intelligence (AI).", paragraphs[0])
        self.assertIn("deep learning controls the field", paragraphs[0])
        self.assertIn("GPT-2 (February 2019)", paragraphs[0])
        self.assertTrue(paragraphs[0].endswith("Short last line."))
        self.assertTrue(paragraphs[1].startswith("A new paragraph"))
        self.assertIn("• Document Loaders: load documents from many sources and formats.", paragraphs[1:])
        self.assertEqual(paragraphs[-1], "• Retrievers: query your data.")

    def test_blank_lines_split(self):
        self.assertEqual(pdf_paragraphs("One line\n\n  \nTwo   spaced   line"), ["One line", "Two spaced line"])

    def test_empty(self):
        self.assertEqual(pdf_paragraphs(" \n \n"), [])


class RichHtmlTests(unittest.TestCase):
    def test_paper_text_cannot_add_markup(self):
        for render in (lambda s: rich_html(s, citations=True), lambda s: list_html([s]), lambda s: paper_html([s]),
                       isolate):
            out = render('<script>alert(1)</script> <a href="http://x">x</a> <img src=x onerror=alert(1)>')
            self.assertNotIn("<script", out)
            self.assertNotIn("<a ", out)
            self.assertNotIn("<img", out)
            self.assertIn("&lt;script&gt;", out)

    def test_citations_become_badges(self):
        out = rich_html("Levels were lower [1] and higher [2].", citations=True)
        self.assertIn('lower<sup class="cite-ref">1</sup>', out)
        self.assertIn('<sup class="cite-ref">2</sup>.', out)
        self.assertNotIn("[1]", out)
        self.assertIn("[1]", rich_html("A [1].", citations=False))

    def test_latin_runs_in_arabic_text_are_isolated_with_their_brackets(self):
        out = rich_html('المقياس هو "نسبة المخاطر" (relative risk) مع فترة ثقة 95% [1].', citations=True)
        self.assertIn("<bdi>(relative risk)</bdi>", out)
        self.assertIn("<bdi>95%</bdi>", out)
        self.assertIn('<p dir="auto">', out)
        out = rich_html("تم استخدام قواعد بيانات PubMed وEmbase وLILACS.")
        for name in ("PubMed", "Embase", "LILACS"):
            self.assertIn(f"<bdi>{name}</bdi>", out)

    def test_english_text_is_not_wrapped(self):
        self.assertNotIn("<bdi>", rich_html("The relative risk (RR) was 1.41."))

    def test_paragraphs_and_bold(self):
        out = rich_html("First **bold** line.\n\nSecond paragraph.")
        self.assertEqual(out.count("<p "), 2)
        self.assertIn("<strong>bold</strong>", out)

    def test_list_html(self):
        self.assertIn("<li dir=\"auto\"><strong>Clinicians:</strong> Screening</li>",
                      list_html(["**Clinicians:** Screening"]))
        self.assertIn("لا شيء", list_html([], "لا شيء"))
        self.assertEqual(list_html([]), "")


class ExcerptTests(unittest.TestCase):
    def test_clip_at_sentence_or_word(self):
        self.assertEqual(clip("Short."), "Short.")
        text = "First sentence is here. " * 20
        clipped = clip(text, 250)
        self.assertLessEqual(len(clipped), 250)
        self.assertTrue(clipped.endswith("here."))
        words = clip("word " * 100, 50)
        self.assertTrue(words.endswith("…"))
        self.assertLessEqual(len(words), 50)

    def test_nearest_sentence(self):
        source = ("LangChain is a framework.\nLarge Language Models (LLMs) accept a text string (prompt) and\n"
                  "output a text string. Chains combine an LLM with a prompt.")
        self.assertEqual(nearest_sentence(source, "An LLM outputs a text string [1]."),
                         "Large Language Models (LLMs) accept a text string (prompt) and output a text string.")
        # Nothing in common (e.g. an Arabic answer with no names or numbers): the first sentence.
        self.assertEqual(nearest_sentence(source, "جواب عربي"), "LangChain is a framework.")
        self.assertEqual(nearest_sentence("", "x"), "")

    def test_nearest_sentence_prefers_a_sentence_with_every_number(self):
        # The migraine paper: an answer of 27.4% must not get a sentence about 16.4% or a figure.
        source = ("Figure 1 depicts the prevalence of migraine and IBS in each region of\nSaudi Arabia. "
                  "Seven hundred sixty-seven were found to be migraine-\npositive, showing a prevalence of 27.4%. "
                  "IBS was less common than migraine in our sample, showing a prevalence of 16.4%.")
        self.assertTrue(nearest_sentence(source, "The prevalence of migraine was 27.4% [1].")
                        .startswith("Seven hundred sixty-seven were found to be"))

    def test_nearest_sentence_without_such_a_sentence_is_unchanged(self):
        source = "LangChain is a framework. Chains combine an LLM with a prompt."
        self.assertEqual(nearest_sentence(source, "Chains combine an LLM with a prompt, in 3 steps [1]."),
                         "Chains combine an LLM with a prompt.")

    def test_table_html_is_escaped(self):
        out = table_html(["Type", "Input"], [["<b>x</b>", "نص (text)"]])
        self.assertIn("<th>Type</th><th>Input</th>", out)
        self.assertIn("&lt;b&gt;x&lt;/b&gt;", out)
        self.assertIn("<bdi>(text)</bdi>", out)


class ShownCitationsTests(unittest.TestCase):
    def source(self, number, quote):
        return {"number": number, "quote": quote, "text": f"Source {number}."}

    def test_hides_unverified_and_renumbers(self):
        answer, details, shown = shown_citations(
            "A [1]. B [2]. C [3]. Both [2, 3].", "More [3] and [2].",
            [self.source(1, None), self.source(2, "q2"), self.source(3, "q3")])
        self.assertEqual([(s["number"], s["quote"]) for s in shown], [(1, "q2"), (2, "q3")])
        self.assertEqual(answer, "A. B [1]. C [2]. Both [1][2].")
        self.assertEqual(details, "More [2] and [1].")

    def test_all_verified_unchanged(self):
        sources = [self.source(1, "q1"), self.source(2, "q2")]
        self.assertEqual(shown_citations("A [1] B [2].", None, sources), ("A [1] B [2].", None, sources))

    def test_none_verified_keeps_the_first(self):
        answer, details, shown = shown_citations("A [2]. B [1].", "", [self.source(1, None), self.source(2, None)])
        self.assertEqual([s["number"] for s in shown], [1])
        self.assertIsNone(shown[0]["quote"])
        self.assertEqual((answer, details), ("A. B [1].", ""))

    def test_answer_left_without_markers_gets_them_at_the_end(self):
        # Only the hidden source was cited in the answer.
        answer, details, shown = shown_citations("A [1].", "More [2].",
                                                 [self.source(1, None), self.source(2, "q2")])
        self.assertEqual((answer, details), ("A. [1]", "More [1]."))
        # Several sources shown: all their numbers are added, as the API does.
        answer, _, _ = shown_citations("A [1].", None,
                                       [self.source(1, None), self.source(2, "q2"), self.source(3, "q3")])
        self.assertEqual(answer, "A. [1][2]")

    def test_no_sources(self):
        self.assertEqual(shown_citations("A.", None, []), ("A.", None, []))

    def test_response_is_not_changed(self):
        sources = [self.source(1, None), self.source(2, "q2")]
        shown_citations("A [2].", None, sources)
        self.assertEqual([s["number"] for s in sources], [1, 2])


class SourceHtmlTests(unittest.TestCase):
    """The full text of a source: the verified quote highlighted, the answer's numbers in bold."""

    MIGRAINE = ("Results. Seven hundred sixty-seven were found to be migraine-\npositive, showing a prevalence of "
                "27.4%. Females had a con-\nsiderably higher prevalence of migraine.\n\n"
                "Other paragraph about the survey, with 2 groups and p <0 :001 in the bivariate analysis.")

    def test_quote_highlighted_inside_text_with_pdf_artifacts(self):
        quote = "Seven hundred sixty-seven were found to be migraine-positive, showing a prevalence of 27.4%."
        out = source_html(self.MIGRAINE, quote, "The prevalence of migraine was 27.4% [1].")
        highlighted = out.split('<span class="quote-mark">')[1].split("</span></p>")[0]
        self.assertIn("migrainepositive", highlighted)  # Display text keeps the artifact.
        self.assertIn('<span class="answer-number">27.4</span>', highlighted)
        self.assertEqual(out.count('class="quote-mark"'), 1)
        # Its paragraph keeps the normal colour; the other one is muted.
        self.assertIn('<p dir="auto">Results.', out)
        self.assertIn('<p dir="auto" class="muted">Other paragraph', out)

    def test_colon_decimal_number_in_bold(self):
        out = source_html(self.MIGRAINE, None, "The association was significant (p < 0.001) [1].")
        self.assertIn('p &lt;<span class="answer-number">0 :001</span>', out)

    def test_no_verified_quote_no_highlight(self):
        out = source_html(self.MIGRAINE, None, "The prevalence was 27.4% [1].")
        self.assertNotIn("quote-mark", out)
        self.assertNotIn("muted", out)
        self.assertIn('<span class="answer-number">27.4</span>', out)  # Numbers are still bold.

    def test_quote_that_cannot_be_located_highlights_nothing(self):
        out = source_html(self.MIGRAINE, "A sentence that is not in this text.", "27.4% [1]")
        self.assertNotIn("quote-mark", out)
        self.assertNotIn("muted", out)

    def test_single_digit_numbers_are_not_bold(self):
        out = source_html(self.MIGRAINE, None, "There were 2 groups and 1 survey [1].")
        self.assertNotIn("answer-number", out)
        self.assertIn("with 2 groups", out)

    def test_number_words_in_the_source_are_bold(self):
        out = source_html(self.MIGRAINE, None, "767 were migraine-positive [1].")
        self.assertIn('<span class="answer-number">Seven hundred sixty-seven</span>', out)

    def test_paper_text_cannot_add_markup_even_when_highlighted(self):
        text = 'Bad <script>alert(1)</script> and <img src=x onerror=alert(1)> with 27.4% here. Next line.'
        out = source_html(text, "Bad <script>alert(1)</script> and <img src=x onerror=alert(1)> with 27.4% here.",
                          "27.4% [1]")
        self.assertNotIn("<script", out)
        self.assertNotIn("<img", out)
        self.assertIn('<span class="quote-mark">Bad &lt;script&gt;alert(1)&lt;/script&gt;', out)
        self.assertIn('<span class="answer-number">27.4</span>', out)


class TextsTests(unittest.TestCase):
    def test_both_languages_have_the_same_keys(self):
        self.assertEqual(set(TEXTS["ar"]), set(TEXTS["en"]))


if __name__ == "__main__":
    unittest.main()
