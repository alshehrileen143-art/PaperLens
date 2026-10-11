"""Tests for the answer and summary warnings as data (*_items), which the interface writes in its
own language, and for warning_markdown, which writes them. No OpenAI calls: answers and summaries
are mocked.

Run: .\\venv\\Scripts\\python.exe -m unittest tests.test_warnings -v
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

import config  # noqa: E402
import main  # noqa: E402
import rag  # noqa: E402
from grounding import GroundingResult, check_grounding  # noqa: E402
from output_check import check_output  # noqa: E402
from rag import AnswerResult  # noqa: E402
from tests.test_summarize import VALID  # noqa: E402
from ui_text import TEXTS, warning_markdown  # noqa: E402

FSI, PDI = "⁨", "⁩"


class ItemsTests(unittest.TestCase):
    def test_grounding_items_hold_what_describe_says(self):
        result = check_grounding("What was the p-value?", rag.strip_citations("The p-value was 0.002 [1]."),
                                 ["(p <0 :001)"], 1)
        self.assertEqual(result.describe(), "numbers not found in the sources: 0.002")  # Unchanged.
        self.assertEqual(result.items(), {"numbers": ["0.002"], "names": [], "too_many_citations": False,
                                          "max_cited_sources": config.MAX_CITED_SOURCES})

    def test_no_warning_no_items(self):
        self.assertIsNone(GroundingResult().items())
        self.assertIsNone(GroundingResult().mismatch_items())
        self.assertIsNone(check_output("No links here.", "Paper text.").items())

    def test_mismatch_and_too_many_citations(self):
        result = GroundingResult(unsupported_names=["Turer"], too_many_citations=True,
                                 mismatched_numbers=["9.2"], mismatched_names=["Jari"])
        self.assertEqual(result.items()["names"], ["Turer"])
        self.assertTrue(result.items()["too_many_citations"])
        self.assertEqual(result.mismatch_items(), {"numbers": ["9.2"], "names": ["Jari"]})
        self.assertEqual(result.describe_mismatch(), "numbers taken from sources it did not cite: 9.2; "
                                                     "names taken from sources it did not cite: Jari")

    def test_output_items(self):
        check = check_output("See http://evil.example and CANARY-7.", "Paper text.")
        self.assertEqual(check.items(), {"links": ["http://evil.example"], "markers": ["CANARY-7"]})
        self.assertIn("links not found in the paper: http://evil.example", check.describe())


class ResponseTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main.app)

    def test_ask_returns_the_english_strings_and_the_items(self):
        result = AnswerResult("The rate was 9.2% [1].", True, [], "full",
                              grounding_warning="numbers not found in the sources: 9.2",
                              grounding_warning_items={"numbers": ["9.2"], "names": [],
                                                       "too_many_citations": False, "max_cited_sources": 8},
                              output_warning="links not found in the paper: http://evil.example",
                              output_warning_items={"links": ["http://evil.example"], "markers": []})
        with patch.object(main, "answer_question", return_value=result):
            body = self.client.post("/ask", json={"paper_id": "x" * 32, "question": "Rate?"}).json()
        self.assertEqual(body["grounding_warning"], "numbers not found in the sources: 9.2")
        self.assertEqual(body["grounding_warning_items"]["numbers"], ["9.2"])
        self.assertEqual(body["output_warning"], "links not found in the paper: http://evil.example")
        self.assertEqual(body["output_warning_items"], {"links": ["http://evil.example"], "markers": []})
        self.assertIsNone(body["citation_mismatch"])
        self.assertIsNone(body["citation_mismatch_items"])

    def test_answer_question_fills_the_items(self):
        answered = AnswerResult("See http://evil.example [1].", True, [], "full")
        with patch.object(rag, "_answer", return_value=answered), \
                patch.object(rag, "load_paper_text", return_value=None), \
                patch.object(rag, "_paper_full_text", return_value="Paper text."):
            result = rag.answer_question("x" * 32, "Link?")
        self.assertEqual(result.output_warning, "links not found in the paper: http://evil.example")
        self.assertEqual(result.output_warning_items, {"links": ["http://evil.example"], "markers": []})

    def test_summarize_returns_the_english_string_and_the_items(self):
        summary = {**VALID, "objective": "Explain LangChain (see http://evil.example)."}
        with patch.object(main, "summarize_paper", return_value=summary):
            response = self.client.post("/summarize", files={"file": ("p.txt", b"LangChain paper text.", "text/plain")})
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["output_warning"], "links not found in the paper: http://evil.example")
        self.assertEqual(body["output_warning_items"], {"links": ["http://evil.example"], "markers": []})
        with patch.object(main, "summarize_paper", return_value=VALID):
            body = self.client.post("/summarize", files={"file": ("p.txt", b"Text.", "text/plain")}).json()
        self.assertNotIn("output_warning_items", body)  # Left out when there is no warning.


class WarningMarkdownTests(unittest.TestCase):
    ITEMS = {"numbers": ["9.2%", "24,600"], "names": ["Turer"], "too_many_citations": True, "max_cited_sources": 8}

    def test_arabic(self):
        self.assertEqual(warning_markdown("تنبيه", self.ITEMS, TEXTS["ar"]),
                         f"**تنبيه**: الأرقام: {FSI}9\\.2%{PDI}، {FSI}24,600{PDI}؛ "
                         f"الأسماء: {FSI}Turer{PDI}؛ تستشهد بأكثر من 8 مصادر")

    def test_english(self):
        self.assertEqual(warning_markdown("Warning", self.ITEMS, TEXTS["en"]),
                         f"**Warning**: Numbers: {FSI}9\\.2%{PDI}, {FSI}24,600{PDI}; "
                         f"Names: {FSI}Turer{PDI}; Cites more than 8 sources")

    def test_links_and_markers_cannot_format_the_warning(self):
        out = warning_markdown("L", {"links": ["http://x.example/a_b*c*"], "markers": ["$x$ [y](z)"]}, TEXTS["en"])
        self.assertIn(f"Links not found in the paper: {FSI}http://x\\.example/a\\_b\\*c\\*{PDI}", out)
        self.assertIn(f"Text matching a marker pattern: {FSI}\\$x\\$ \\[y\\]\\(z\\){PDI}", out)

    def test_no_items_gives_the_label_alone(self):
        for items in (None, {}, {"numbers": [], "names": [], "too_many_citations": False}):
            self.assertEqual(warning_markdown("تنبيه", items, TEXTS["ar"]), "**تنبيه**")

    def test_no_english_words_in_arabic(self):
        out = warning_markdown("تنبيه", {**self.ITEMS, "links": ["http://a.example"], "markers": ["CANARY-1"]},
                               TEXTS["ar"])
        values = ("9\\.2%", "24,600", "Turer", "http://a\\.example", "CANARY\\-1")
        for value in values:
            out = out.replace(FSI + value + PDI, "")
        self.assertNotRegex(out, "[A-Za-z]")


if __name__ == "__main__":
    unittest.main()
