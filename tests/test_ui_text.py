"""Tests for the display-only text helpers in ui_text.py (no API, no Streamlit).

Run: .\\venv\\Scripts\\python.exe -m unittest tests.test_ui_text -v
"""

import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ui_text import (CONTRAST_MINIMUM, CONTRAST_PAIRS, PALETTES, TEXTS, clip, contrast_ratio,  # noqa: E402
                     isolate, list_html, nearest_sentence, palette_css, paper_html, pdf_paragraphs,
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


class ThemeTests(unittest.TestCase):
    """Dark mode: every colour is a variable with a light and a dark value, light mode unchanged,
    and every text at 4.5:1 or more in both modes."""

    ROOT = Path(__file__).resolve().parent.parent

    def style_block(self) -> str:
        source = (self.ROOT / "app.py").read_text(encoding="utf-8")
        start = source.index('STYLE = """')
        return source[start:source.index('"""', source.index("<style>", start)) + 3]

    def test_no_hard_coded_colours_left_in_style(self):
        self.assertEqual(re.findall(r"#[0-9A-Fa-f]{3,8}\b", self.style_block()), [])
        used = set(re.findall(r"var\(--([a-z-]+)\)", self.style_block()))
        self.assertTrue(used <= set(PALETTES["light"]), used - set(PALETTES["light"]))

    def test_both_modes_define_the_same_variables(self):
        self.assertEqual(set(PALETTES["light"]), set(PALETTES["dark"]))

    def test_light_mode_keeps_the_original_colours(self):
        original = {  # As in app.py before dark mode, and .streamlit/config.toml.
            "page-bg": "#FFFFFF", "page-text": "#1C2533", "muted-text": "#6B7585", "paper-text": "#3A4556",
            "accent": "#1F3A68", "on-accent": "#FFFFFF", "card-bg": "#FFFFFF", "data-bg": "#FFFFFF",
            "summary-bg": "#F7F9FC", "summary-border": "#C9D3E3", "border": "#DDE2EA", "chip-bg": "#EEF2F8",
            "chip-border": "#D5DDEA", "badge-bg": "#E3EAF4", "quote-bg": "#F4F6F9", "highlight-bg": "#E8EDF5",
            "highlight-text": "#3A4556", "table-head-bg": "#E8EDF5", "table-head-text": "#1F3A68",
            "unverified-text": "#4A5568", "unverified-border": "#A3ACBA", "warm-bg": "#FFF8EC",
            "warm-border": "#E2B860", "warm-heading": "#8A5A00", "warm-text": "#5C4A2A",
            # The approved fix: the excerpt note (#7A8495) and "not stated" (#8A93A3) now use this colour.
            "text-secondary": "#5B6678",
        }
        self.assertEqual(PALETTES["light"], original)
        self.assertIn(".cite-note { color: var(--text-secondary);", self.style_block())
        self.assertIn(".not-stated { color: var(--text-secondary);", self.style_block())

    def test_dark_mode_colours(self):
        dark = PALETTES["dark"]
        self.assertEqual((dark["page-bg"], dark["card-bg"], dark["page-text"], dark["text-secondary"], dark["accent"]),
                         ("#0F172A", "#1E293B", "#E5E7EB", "#9CA3AF", "#8AB4F8"))
        self.assertEqual((dark["warm-bg"], dark["warm-text"], dark["highlight-bg"]), ("#3A2E14", "#F5D08A", "#2B4C7E"))

    def test_every_text_reaches_wcag_aa_in_both_modes(self):
        for mode, palette in PALETTES.items():
            for foreground, background in CONTRAST_PAIRS:
                with self.subTest(mode=mode, pair=(foreground, background)):
                    ratio = contrast_ratio(palette[foreground], palette[background])
                    self.assertGreaterEqual(ratio, CONTRAST_MINIMUM, f"{palette[foreground]} on {palette[background]}")

    def test_contrast_ratio_formula(self):
        self.assertAlmostEqual(contrast_ratio("#000000", "#FFFFFF"), 21.0, places=2)
        self.assertAlmostEqual(contrast_ratio("#E5E7EB", "#0F172A"), 14.42, places=1)

    def test_config_toml_matches_the_palettes(self):
        import tomllib
        theme = tomllib.loads((self.ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))["theme"]
        for mode in ("light", "dark"):
            with self.subTest(mode):
                self.assertEqual(theme[mode]["backgroundColor"], PALETTES[mode]["page-bg"])
                self.assertEqual(theme[mode]["textColor"], PALETTES[mode]["page-text"])
                self.assertEqual(theme[mode]["primaryColor"], PALETTES[mode]["accent"])
        self.assertNotIn("base", theme)  # base would tie the app to one mode.

    def test_palette_css_follows_the_device_then_streamlit(self):
        css = palette_css(None)
        self.assertTrue(css.startswith(":root { --page-bg: #FFFFFF;"))
        self.assertIn("@media (prefers-color-scheme: dark) { :root { --page-bg: #0F172A;", css)
        self.assertTrue(palette_css("dark").endswith("--warm-text: #F5D08A; }"))  # Reported mode comes last.
        self.assertTrue(palette_css("light").endswith("--warm-text: #5C4A2A; }"))
        self.assertEqual(palette_css("sepia"), css)


class TextsTests(unittest.TestCase):
    def test_both_languages_have_the_same_keys(self):
        self.assertEqual(set(TEXTS["ar"]), set(TEXTS["en"]))


if __name__ == "__main__":
    unittest.main()
