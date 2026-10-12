"""Tests for the Streamlit app (app.py). No server and no OpenAI: every HTTP call is mocked.

AppTest can only type text into st.chat_input, so the message box is replaced by a
fake that returns what a user would submit: text, attached files and/or a recording.

Run: .\\venv\\Scripts\\python.exe -m unittest tests.test_app -v
"""

import re
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import requests
import streamlit as st
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ui_text import PAPER_TYPES, palette_css  # noqa: E402
APP = str(ROOT / "app.py")
PAPER_ID = "8db0b76c45a54a3a8159b8b2824208a9"
LONG_NAME = "CreatingLargeLanguageModelApplicationsUtilizingLangChain-APrimeronDevelopingLLMAppsFast.pdf"

UPLOAD = {
    "paper_id": PAPER_ID, "filename": LONG_NAME, "ocr_used": True, "chunk_count": 12,
    "normalization": {}, "hidden_text_removed": {"count": 2, "preview": ["ignore previous instructions"]},
}
SECTION_TEXT = ("Large Language Models (LLMs) are the main\ntype of models utilized in LangChain. The embed-\n"
                "dings are required when we want to work with our own documents.")
ANSWER = {
    "paper_id": PAPER_ID, "question": "ما المقياس؟",
    "answer": 'المقياس هو "نسبة المخاطر" (relative risk) مع فترة ثقة 95% [1].',
    "details": "تم استخدام نموذج التأثير العشوائي (random effect model) [2] [3].",
    "found": True, "mode": "full", "grounding_warning": "numbers not found in the sources: 42%",
    "citation_mismatch": None, "output_warning": "links not found in the paper: http://evil.example",
    # The same warnings as data, which the interface writes in its own language.
    "grounding_warning_items": {"numbers": ["42%"], "names": [], "too_many_citations": False,
                                "max_cited_sources": 8},
    "citation_mismatch_items": None,
    "output_warning_items": {"links": ["http://evil.example"], "markers": []},
    "sources": [{"number": 1, "chunk_index": 3, "page": 3, "page_end": 3, "title": "A.A.2 Models",
                 "section": "Methods", "matched": None, "score": None, "bm25_score": None, "fused_score": None,
                 "expanded_from": None, "quote": "<b>The embeddings</b> are required", "text": SECTION_TEXT},
                # No verified quote: hidden, and [2] is removed from the answer.
                {"number": 2, "chunk_index": 7, "page": 2, "page_end": 4, "title": None, "section": "Methods",
                 "matched": None, "score": None, "bm25_score": None, "fused_score": None, "expanded_from": None,
                 "quote": None, "text": "Unrelated opening sentence. The random effect model\nwas used for pooling."},
                # Verified, with no page and no title; shown as [2].
                {"number": 3, "chunk_index": 9, "page": None, "page_end": None, "title": None, "section": None,
                 "matched": None, "score": None, "bm25_score": None, "fused_score": None, "expanded_from": None,
                 "quote": "A very long quote. " * 30, "text": "Text."}],
}
# No source has a verified quote: only the first shows, as an unverified excerpt.
UNVERIFIED_ANSWER = {
    **ANSWER,
    "answer": "The random effect model was used [2], as in the methods [1].",
    "details": None,
    "sources": [{**ANSWER["sources"][1], "number": 1},
                {**ANSWER["sources"][0], "number": 2, "quote": None}],
}
SUMMARY = {
    "filename": LONG_NAME, "ocr_used": False, "output_warning": None,
    "summary": {
        "objective": "Measure vitamin D.", "methodology": "Case-control study.",
        "key_findings": ["Lower levels in cases"], "limitations": ["Small sample"],
        "use_cases": [{"audience": "Clinicians", "application": "Screening"}],
        "technical_use_cases": ["Risk model"],
        "ai_ml_tasks": [{"task_type": "classification", "input": "labs", "output": "risk"}],
        "data_and_tools": {"datasets": ["Hospital cohort"], "variables": ["25(OH)D"], "databases": [], "tools": ["SPSS"]},
    },
}


def response(body: dict, status: int = 200) -> MagicMock:
    mock = MagicMock(ok=status < 400, status_code=status, reason="Error")
    mock.json.return_value = body
    return mock


TRANSCRIPT = "ما المقياس المستخدم؟"


def fake_post(url, **kwargs):
    endpoint = url.rsplit("/", 1)[-1]
    if endpoint == "upload":
        return response(UPLOAD)
    if endpoint == "summarize":
        return response(SUMMARY)
    if endpoint == "ask":
        return response({**ANSWER, "question": kwargs["json"]["question"]})
    if endpoint == "transcribe":
        return response({"transcript": TRANSCRIPT})
    if endpoint == "speak":
        audio = response({})
        audio.content = MP3
        return audio
    raise AssertionError(url)


MP3 = b"ID3fake-mp3"


def uploaded_file(name: str, content: bytes, type_: str = "application/pdf") -> MagicMock:
    file = MagicMock(type=type_)
    file.name = name
    file.getvalue.return_value = content
    return file


def submission(text: str = "", files=(), audio=None) -> SimpleNamespace:
    return SimpleNamespace(text=text, files=list(files), audio=audio)


def texts(elements) -> str:
    return "\n".join(str(element.value) for element in elements)


class AppTests(unittest.TestCase):
    def setUp(self):
        self.get = patch("requests.get", return_value=response({"status": "ok"})).start()
        self.post = patch("requests.post", side_effect=fake_post).start()
        # The message box: each run returns the next queued submission, then nothing.
        self.queue = []
        self.chat_input = patch.object(st, "chat_input", side_effect=self._next_submission).start()
        self.addCleanup(patch.stopall)
        self.at = AppTest.from_file(APP, default_timeout=10)

    def _next_submission(self, *args, **kwargs):
        self.chat_args, self.chat_kwargs = args, kwargs
        return self.queue.pop(0) if self.queue else None

    def submit(self, value) -> AppTest:
        self.queue.append(value)
        return self.at.run()

    def html(self) -> str:
        return texts(self.at.markdown)

    def endpoints(self) -> list[str]:
        return [call.args[0].rsplit("/", 1)[-1] for call in self.post.call_args_list]

    def upload(self) -> AppTest:
        """Open the paper from the upload area shown before any paper."""
        self.at.run()
        self.at.file_uploader[0].set_value((LONG_NAME, b"%PDF-1.7 test", "application/pdf"))
        return self.at.run()

    # --- Page, language, server ---

    def test_welcome_in_arabic_right_to_left(self):
        self.at.run()
        self.assertFalse(self.at.exception)
        self.assertEqual(self.at.title[0].value, "🔍 PaperLens")
        self.assertIn('<p class="app-tagline">مساعد الأوراق البحثية بإجابات موثّقة</p>', self.html())
        self.assertIn("direction: rtl", self.html())
        # Colour variables for both modes are on the page, each following Streamlit's mode.
        self.assertIn(":root { --page-bg: light-dark(#FFFFFF, #0F172A);", self.html())
        self.assertNotIn("prefers-color-scheme", self.html())
        # One description line under the name (the old second line is gone), then a rule.
        self.assertNotIn("app-subtitle", self.html())
        self.assertIn('<hr class="header-rule">', self.html())
        # Room for Streamlit's top bar, which is transparent in both modes.
        self.assertIn(".block-container { padding-top: 4.75rem; }", self.html())
        self.assertIn('[data-testid="stHeader"] { background: transparent; }', self.html())
        # Welcome page: three cards (icon, bold title, grey line), not the old steps box.
        html = self.html()
        self.assertEqual(html.count('<div class="welcome-card">'), 3)
        self.assertIn('<div class="card-icon">📎</div><div class="card-title">ارفع ورقة</div>'
                      '<div class="card-text"><bdi>PDF</bdi> و<bdi>Word</bdi> و<bdi>TXT</bdi> وصور</div>', html)
        self.assertIn('<div class="card-title">ملخص منظم</div>', html)
        self.assertIn('<div class="card-title">اسأل بالكتابة أو الصوت</div>'
                      '<div class="card-text">إجابات مختصرة مع استشهادات متحقق منها</div>', html)
        self.assertNotIn('class="welcome"', html)
        # Before any paper: an upload area under the cards, and no message box (no 📎, no 🎤).
        self.chat_input.assert_not_called()
        [area] = self.at.file_uploader
        self.assertEqual(area.label, "اسحب الورقة وأفلتها هنا، أو اخترها من جهازك")
        self.assertEqual(sorted(area.allowed_type), sorted(f".{kind}" for kind in PAPER_TYPES))
        self.assertFalse(area.accept_multiple_files)
        # No brackets in the hint: around two isolated English words they would split.
        self.assertIn('<p class="upload-hint"><bdi>PDF</bdi> و<bdi>Word</bdi> و<bdi>TXT</bdi> وصور '
                      '<bdi>PNG</bdi> و<bdi>JPG</bdi>، حتى <bdi>20</bdi> ميغابايت.</p>', html)

    def test_the_message_box_appears_with_the_paper(self):
        self.upload()
        self.assertFalse(self.at.exception)
        self.assertEqual(len(self.at.file_uploader), 0)  # The upload area is gone.
        # Neutral Modern Standard Arabic in the message box.
        self.assertEqual(self.chat_args[0], "اكتب سؤالك عن الورقة، أو أرفق ورقة بزر 📎")
        # Files (the same types as the upload area) and recordings, inside the message box itself.
        self.assertTrue(self.chat_kwargs["accept_file"])
        self.assertEqual(self.chat_kwargs["file_type"], list(PAPER_TYPES))
        self.assertTrue(self.chat_kwargs["accept_audio"])
        self.assertEqual(self.chat_kwargs["max_upload_size"], 20)
        self.assertNotIn('<div class="welcome-cards">', self.html())

    def test_paper_types_are_the_ones_the_api_extracts(self):
        from extract import SUPPORTED_EXTENSIONS
        self.assertEqual(sorted(f".{kind}" for kind in PAPER_TYPES), sorted(SUPPORTED_EXTENSIONS))

    def test_a_failed_upload_keeps_the_upload_area(self):
        self.post.side_effect = lambda url, **kw: response({"detail": "No text found in 'a.pdf', even after OCR."}, 400)
        self.upload()
        self.assertFalse(self.at.exception)
        self.assertIn("خطأ من الخادم (400): No text found", texts(self.at.error))
        self.assertNotIn("paper", self.at.session_state)
        self.assertEqual(self.at.session_state.messages, [])  # Nothing in a conversation yet.
        self.chat_input.assert_not_called()
        [area] = self.at.file_uploader  # Still there, and empty, so the file is not sent again.
        self.assertFalse(area.value)
        self.assertEqual(self.endpoints(), ["upload"])
        self.at.run()
        self.assertEqual(self.endpoints(), ["upload"])
        # The error is redrawn in the new language.
        self.at.segmented_control(key="lang").set_value("en").run()
        self.assertIn("Server error (400): No text found", texts(self.at.error))
        self.assertEqual(self.at.file_uploader[0].label, "Drag and drop the paper here, or choose it from your device")
        # The next upload works and clears the error.
        self.post.side_effect = fake_post
        self.at.file_uploader[0].set_value((LONG_NAME, b"%PDF-1.7 test", "application/pdf"))
        self.at.run()
        self.assertEqual(len(self.at.error), 0)
        self.assertEqual(self.at.session_state.paper["paper_id"], PAPER_ID)
        self.assertEqual(len(self.at.file_uploader), 0)

    def test_language_switch_redraws_everything_left_to_right(self):
        self.upload()
        self.at.segmented_control(key="lang").set_value("en").run()
        self.assertFalse(self.at.exception)
        self.assertEqual(self.at.title[0].value, "🔍 PaperLens")
        self.assertIn('<p class="app-tagline">Research paper assistant with verified citations</p>', self.html())
        self.assertNotIn("direction: rtl", self.html())
        # Earlier messages are redrawn in English too.
        self.assertIn("Uploaded **", self.html())
        self.assertEqual(self.at.button[0].label, "Summarize the paper")
        self.at.segmented_control(key="lang").set_value("ar").run()
        self.assertIn("direction: rtl", self.html())
        self.assertEqual(self.at.button[0].label, "لخّص الورقة")

    def test_server_down(self):
        self.get.side_effect = requests.ConnectionError()
        self.at.run()
        self.assertIn("الخادم لا يعمل", texts(self.at.error))
        self.assertIn("uvicorn main:app", self.at.code[0].value)
        self.chat_input.assert_not_called()

    def test_server_down_mid_session(self):
        self.upload()
        self.post.side_effect = requests.ConnectionError()
        self.submit(submission(text="سؤال"))
        self.assertIn("الخادم لا يعمل", texts(self.at.error))

    # --- Upload and summary ---

    def test_upload_from_the_message_box(self):
        self.upload()
        self.assertFalse(self.at.exception)
        self.assertEqual(self.endpoints(), ["upload"])
        self.assertEqual(self.post.call_args.kwargs["files"]["file"], (LONG_NAME, b"%PDF-1.7 test"))
        self.assertEqual(self.at.session_state.paper["paper_id"], PAPER_ID)
        html = self.html()
        # Long file names are shortened everywhere they are shown.
        self.assertNotIn(LONG_NAME, html)
        self.assertIn("CreatingLargeLanguag…", html)
        self.assertIn("paper-chip", html)
        warnings = texts(self.at.warning)
        self.assertIn("OCR", warnings)
        self.assertIn("ignore previous instructions", html)
        self.assertEqual([m.name for m in self.at.chat_message], ["user", "assistant"])
        # Custom avatars instead of the default icons.
        self.assertEqual([m.avatar for m in self.at.chat_message], ["👤", "🔍"])
        self.assertIn("تم رفع الورقة **", self.html())
        self.assertNotIn("مرفوعة مسبقًا", self.html())

    def test_a_paper_uploaded_before_is_opened(self):
        self.post.side_effect = lambda url, **kw: (response({**UPLOAD, "duplicate": True}) if url.endswith("/upload")
                                                   else fake_post(url, **kw))
        self.upload()
        self.assertFalse(self.at.exception)
        self.assertEqual(self.at.session_state.paper["paper_id"], PAPER_ID)
        html = self.html()
        self.assertIn("هذه الورقة مرفوعة مسبقًا، تم فتحها: **CreatingLargeLanguag…", html)
        self.assertIn("(عدد الأجزاء: 12). اسأل عنها، أو اطلب ملخصًا.", html)
        self.assertNotIn("تم رفع الورقة", html)
        self.assertEqual(self.at.button[0].label, "لخّص الورقة")  # Everything else as for a new paper.
        self.at.segmented_control(key="lang").set_value("en").run()
        self.assertIn("This paper was already uploaded; it has been opened: **", self.html())
        self.assertNotIn("Uploaded **", self.html())

    def test_only_the_first_file_is_uploaded(self):
        self.upload()
        self.submit(submission(files=[uploaded_file("a.pdf", b"A"), uploaded_file("b.pdf", b"B")]))
        self.assertEqual(self.endpoints(), ["upload", "upload"])
        self.assertEqual(self.post.call_args.kwargs["files"]["file"], ("a.pdf", b"A"))
        self.assertIn("b.pdf", texts(self.at.caption))

    def test_upload_error_from_the_message_box_shows_api_detail(self):
        self.upload()
        self.post.side_effect = lambda url, **kw: response({"detail": "Unsupported file type '.xyz'."}, 400)
        self.submit(submission(files=[uploaded_file("a.xyz", b"x")]))
        self.assertIn("Unsupported file type", texts(self.at.error))
        self.assertEqual(self.at.session_state.paper["paper_id"], PAPER_ID)  # The open paper stays.
        self.assertEqual(self.at.session_state.messages[-2:],
                         [{"role": "user", "kind": "file", "name": "a.xyz"},
                          {"role": "assistant", "kind": "error", "key": "server_error",
                           "values": {"status": 400, "detail": "Unsupported file type '.xyz'."}}])

    def test_summary_is_a_chat_message(self):
        self.upload()
        self.at.button[0].click().run()
        self.assertFalse(self.at.exception)
        self.assertEqual(self.endpoints(), ["upload", "summarize"])
        self.assertEqual(self.post.call_args.kwargs["files"]["file"], (LONG_NAME, b"%PDF-1.7 test"))
        self.assertEqual(len(self.at.json), 0)
        html = self.html()
        self.assertIn('<div class="summary-title">📋 ملخص الورقة</div>', html)
        for heading in ("🎯 الهدف", "🔬 المنهجية", "📊 أبرز النتائج", "⚠️ القيود", "💡 حالات الاستخدام",
                        "🛠️ حالات الاستخدام التقنية", "🤖 مهام الذكاء الاصطناعي", "🗂️ البيانات والأدوات"):
            self.assertIn(f'<div class="summary-heading">{heading}</div>', html)
        self.assertIn('<li dir="auto">Lower levels in cases</li>', html)
        self.assertIn("<strong>Clinicians:</strong> Screening", html)
        # AI tasks as a small table; empty data groups (databases) are hidden.
        self.assertIn("<th>النوع</th><th>المدخلات</th><th>المخرجات</th>", html)
        self.assertIn('<td dir="auto">classification</td><td dir="auto">labs</td><td dir="auto">risk</td>', html)
        # A data group the paper does not mention says so instead of being hidden.
        self.assertIn('<div class="data-group-title">قواعد البيانات</div><div class="not-stated">لم تذكرها الورقة</div>',
                      html)
        # No suggested limitations in this summary: no such section.
        self.assertNotIn("قيود محتملة", html)
        self.at.segmented_control(key="lang").set_value("en").run()
        self.assertIn('<div class="summary-heading">🤖 AI tasks</div>', self.html())
        self.assertIn("<th>Type</th><th>Input</th><th>Output</th>", self.html())
        self.assertEqual(self.at.chat_message[-1].name, "assistant")
        # The button is gone once the paper is summarized.
        self.assertEqual(len(self.at.button), 0)

    def test_summary_fields_the_paper_does_not_state(self):
        empty = {**SUMMARY, "summary": {**SUMMARY["summary"], "methodology": "", "key_findings": [],
                                        "limitations": [],
                                        "suggested_limitations": ["No empirical evaluation of the framework."]}}
        self.upload()
        self.post.side_effect = lambda url, **kw: (response(empty) if url.endswith("/summarize")
                                                   else fake_post(url, **kw))
        self.at.button[0].click().run()
        html = self.html()
        for heading in ("🔬 المنهجية", "📊 أبرز النتائج", "⚠️ القيود"):
            self.assertIn(f'<div class="summary-heading">{heading}</div><div class="not-stated">لم تذكرها الورقة</div>',
                          html)
        # The model's own limitations: right after the authors' ones, in their own coloured section.
        self.assertIn('<div class="summary-section suggested"><div class="summary-heading">'
                      '💭 قيود محتملة (استنتاج، ليست من الورقة)</div>', html)
        self.assertIn("No empirical evaluation of the framework.", html)
        self.assertLess(html.index("⚠️ القيود"), html.index("💭 قيود محتملة"))
        self.assertLess(html.index("💭 قيود محتملة"), html.index("💡 حالات الاستخدام"))
        self.at.segmented_control(key="lang").set_value("en").run()
        self.assertIn("💭 Possible limitations (inferred, not from the paper)", self.html())
        self.assertIn("Not stated in the paper", self.html())

    # --- Questions ---

    def test_no_question_before_any_paper(self):
        # Neither typed nor spoken: there is no message box until a paper is open.
        self.at.run()
        self.chat_input.assert_not_called()
        self.post.assert_not_called()
        self.assertEqual(self.at.session_state.messages, [])

    def test_answer_with_numbered_source_cards(self):
        self.upload()
        self.submit(submission(text="  ما المقياس؟ "))
        self.assertFalse(self.at.exception)
        self.assertEqual(self.post.call_args.kwargs["json"], {"paper_id": PAPER_ID, "question": "ما المقياس؟"})
        html = self.html()
        # Answer: citation badges, and English inside Arabic isolated with its brackets.
        self.assertIn('<sup class="cite-ref">1</sup>', html)
        self.assertIn("<bdi>(relative risk)</bdi>", html)
        # Card: [1] page · bold heading, then the quote (escaped), then the full text folded.
        self.assertIn('<span class="cite-num">1</span><span>صفحة 3 · <strong><bdi dir="auto">A.A.2 Models</bdi>'
                      '</strong></span>', html)
        self.assertIn("📌 &lt;b&gt;The embeddings&lt;/b&gt; are required", html)
        # The source without a verified quote is hidden; the next one becomes [2].
        self.assertNotIn("Methods", html)
        self.assertNotIn("أقرب جملة في القسم — لم يُتحقق من اقتباس النموذج", html)
        self.assertNotIn('<span class="cite-num">3</span>', html)
        # No page and no title at all; a long quote is clipped to about 250 characters.
        self.assertIn('<span class="cite-num">2</span><span>صفحة غير معروفة · بدون عنوان</span>', html)
        quote2 = html.split("cite-num\">2")[1].split("📌 ")[1].split("</div>")[0]
        self.assertLessEqual(len(quote2), 250)
        labels = [e.label for e in self.at.expander]
        self.assertEqual(labels, ["تفاصيل أكثر", "عرض النص الكامل", "عرض النص الكامل"])
        self.assertFalse(any(e.proto.expanded for e in self.at.expander))
        # Details: the hidden source's [2] is removed, and [3] became [2].
        details = texts(self.at.expander[0].markdown)
        self.assertIn("نموذج التأثير العشوائي", details)
        self.assertIn('</bdi><sup class="cite-ref">2</sup>.', details)
        self.assertEqual(details.count("cite-ref"), 1)
        full_text = texts(self.at.expander[1].markdown)
        self.assertIn("the main type of models", full_text)  # PDF lines joined.
        self.assertIn("The embeddings are required", full_text)  # Hyphenated word joined.
        # Warnings in the interface's language only: the label, then the values the checks found.
        warnings = [element.value for element in self.at.warning]
        # Each value is isolated (FSI ... PDI) and escaped for markdown ("evil\.example").
        self.assertIn("**أرقام أو أسماء في الإجابة لم ترد في المصادر**: الأرقام: ⁨42%⁩", warnings)
        self.assertIn("**تحتوي الإجابة على رابط أو نص مريب**: روابط غير موجودة في الورقة: "
                      "⁨http://evil\\.example⁩", warnings)
        self.assertNotIn("not found in the", "\n".join(warnings))  # The API's English sentence.
        self.at.segmented_control(key="lang").set_value("en").run()
        warnings = [element.value for element in self.at.warning]
        self.assertIn("**Numbers or names in the answer were not found in the sources**: Numbers: "
                      "⁨42%⁩", warnings)
        self.assertIn("**The answer contains a link or suspicious text**: Links not found in the paper: "
                      "⁨http://evil\\.example⁩", warnings)

    def test_summary_warning_in_the_interface_language(self):
        flagged = {**SUMMARY, "output_warning": "links not found in the paper: http://evil.example",
                   "output_warning_items": {"links": ["http://evil.example"], "markers": []}}
        self.post.side_effect = lambda url, **kw: (response(flagged) if url.endswith("/summarize")
                                                   else fake_post(url, **kw))
        self.upload()
        self.at.button[0].click().run()
        self.assertFalse(self.at.exception)
        warnings = [element.value for element in self.at.warning]
        self.assertIn("**تحذير في الملخص**: روابط غير موجودة في الورقة: ⁨http://evil\\.example⁩", warnings)
        self.assertNotIn("not found in the", "\n".join(warnings))
        self.at.segmented_control(key="lang").set_value("en").run()
        self.assertIn("**Warning in the summary**: Links not found in the paper: ⁨http://evil\\.example⁩",
                      [element.value for element in self.at.warning])

    def test_warnings_from_an_older_server_show_the_label_only(self):
        # No *_items fields: the translated label alone, never the API's English sentence.
        old = {key: value for key, value in ANSWER.items() if not key.endswith("_items")}
        self.upload()
        self.post.side_effect = lambda url, **kw: (response(old) if url.endswith("/ask") else fake_post(url, **kw))
        self.submit(submission(text="ما المقياس؟"))
        self.assertFalse(self.at.exception)
        warnings = [element.value for element in self.at.warning]
        self.assertIn("**أرقام أو أسماء في الإجابة لم ترد في المصادر**", warnings)
        self.assertIn("**تحتوي الإجابة على رابط أو نص مريب**", warnings)
        self.assertNotIn("42%", "\n".join(warnings))

    def test_no_colour_written_outside_the_palettes(self):
        # Every colour of STYLE and of the HTML the app draws (welcome cards, summary, source cards,
        # quotes, full text) is a var(--...) from ui_text.PALETTES, so it follows the mode.
        # Hex and colour functions anywhere; and a named colour ("white") in a colour property.
        written = re.compile(r"#[0-9A-Fa-f]{3,8}\b|\b(?:rgba?|hsla?|hwb|lab|lch|oklab|oklch|color-mix)\(",
                             re.IGNORECASE)
        declaration = re.compile(r"\b(color|background(?:-color)?|border(?:-[a-z]+){0,2}?(?:-color)?|outline"
                                 r"|fill|stroke)\s*:\s*([^;}\"]+)", re.IGNORECASE)
        keywords = {"none", "transparent", "inherit", "currentcolor", "solid", "dashed", "dotted", "!important"}

        def colours_written(page: str) -> list[str]:
            found = [match.group() for match in written.finditer(page)]
            for match in declaration.finditer(page):
                if match.group(1).lower() in ("border-collapse", "border-radius", "border-spacing"):
                    continue
                value = re.sub(r"var\(--[\w-]+\)", "", match.group(2))
                words = [word for word in value.lower().split()
                         if word not in keywords and not re.fullmatch(r"[\d.]+(?:px|rem|em|%)?", word)]
                found += [match.group()] if words else []
            return found

        pages = []
        self.upload()
        pages.append(self.html())
        self.at.button[0].click().run()
        pages.append(self.html())
        self.submit(submission(text="ما المقياس؟"))
        pages.append(self.html())
        self.post.side_effect = lambda url, **kw: (fake_post(url, **kw) if not url.endswith("/ask")
                                                   else response(UNVERIFIED_ANSWER))
        self.submit(submission(text="ما المقياس؟"))
        pages.append(self.html())
        self.assertFalse(self.at.exception)
        for page in pages:
            # The palette itself is the one place colours are written.
            palette = f"<style>\n{palette_css()}\n</style>"
            self.assertIn(palette, page)
            self.assertEqual(colours_written(page.replace(palette, "")), [])
        # The check itself catches each kind of written colour.
        for sample in ("color: #1F3A68;", "background: rgb(0, 0, 0);", "color: white;",
                       "border: 1px solid red;", "border-inline-start: 3px solid navy;"):
            self.assertTrue(colours_written(sample), sample)
        self.assertEqual(colours_written("border-collapse: collapse; border: 1px solid var(--border);"), [])
        self.assertIn('class="summary-title"', pages[1])
        self.assertIn('class="cite-quote" dir="auto">📌', pages[2])
        self.assertIn('<div class="paper-text">', pages[2])  # Inside "عرض النص الكامل" too.
        self.assertIn('class="cite-quote unverified"', pages[3])

    def test_no_verified_quote_shows_only_the_first_source(self):
        self.upload()
        self.post.side_effect = lambda url, **kw: (fake_post(url, **kw) if not url.endswith("/ask")
                                                   else response(UNVERIFIED_ANSWER))
        self.submit(submission(text="ما المقياس؟"))
        self.assertFalse(self.at.exception)
        html = self.html()
        self.assertEqual(html.count('<span class="cite-num">'), 1)
        # No verified quote and a source over pages 2–4: the whole range, not just the first page.
        self.assertIn('<span class="cite-num">1</span><span>صفحات 2–4 · <strong><bdi dir="auto">Methods</bdi>', html)
        self.assertIn('unverified" dir="auto">The random effect model was used for pooling.</div>', html)
        self.assertIn("أقرب جملة في القسم — لم يُتحقق من اقتباس النموذج", html)
        self.assertNotIn("A.A.2 Models", html)
        # The answer keeps [1] only; the hidden source's [2] is removed.
        self.assertIn('The random effect model was used, as in the methods<sup class="cite-ref">1</sup>.', html)
        self.assertEqual([e.label for e in self.at.expander], ["عرض النص الكامل"])
        # The closest sentence is not shown as a verified quote: no 📌, and no highlight in the full text.
        self.assertNotIn("📌", html)
        self.assertNotIn("quote-mark", texts(self.at.expander[0].markdown))
        self.assertNotIn("muted", texts(self.at.expander[0].markdown))
        self.at.segmented_control(key="lang").set_value("en").run()
        self.assertIn("Closest sentence in the section — the model&#x27;s quote was not verified", self.html())

    def test_page_range_only_without_a_verified_quote(self):
        verified_long = {**ANSWER["sources"][0], "page": 3, "page_end": 5}
        unverified_one_page = {**ANSWER["sources"][1], "number": 2, "quote": None, "page": 4, "page_end": 4}
        mismatch_long = {**ANSWER["sources"][1], "number": 1, "quote": None, "page": 1, "page_end": 2,
                         "quote_issue": "number_mismatch", "missing_numbers": ["20"]}
        cases = [
            ({**ANSWER, "sources": [verified_long]}, "صفحة 3 · "),  # A verified quote is on one page.
            ({**ANSWER, "answer": "A [1].", "details": None, "sources": [unverified_one_page]}, "صفحة 4 · "),
            ({**ANSWER, "answer": "A [1].", "details": None, "sources": [mismatch_long]}, "صفحات 1–2 · "),
        ]
        self.upload()
        for body, expected in cases:
            with self.subTest(expected):
                self.post.side_effect = lambda url, b=body, **kw: (response(b) if url.endswith("/ask")
                                                                  else fake_post(url, **kw))
                self.submit(submission(text="سؤال"))
                last = texts(self.at.chat_message[-1].markdown)
                self.assertIn(f'<span class="cite-num">1</span><span>{expected}', last)
        self.at.segmented_control(key="lang").set_value("en").run()
        self.assertIn("Pages 1–2 · ", texts(self.at.chat_message[-1].markdown))

    def test_user_question_bubble_takes_its_own_direction(self):
        self.upload()
        self.submit(submission(text="What is the prevalence?"))
        # In the Arabic (right-to-left) interface, an English question stays left to right with "?" at the end.
        self.assertIn('<div class="user-text" dir="auto"><div class="rich"><p dir="auto">What is the prevalence?</p>'
                      '</div></div>', self.html())
        self.assertIn("direction: rtl", self.html())

    def test_number_mismatch_quote_is_not_shown_as_verified(self):
        # The real case: the quote is verbatim but lacks the answer's number, so the API sends quote=None.
        children = "A total of 24,600 children were included."
        mismatch = {**ANSWER, "answer": "A total of 20 studies were included. [1]", "details": None,
                    "sources": [{**ANSWER["sources"][0], "title": "Results", "page": 2, "quote": None,
                                 "quote_issue": "number_mismatch", "missing_numbers": ["20"],
                                 "rejected_quote": children,
                                 "text": "Studies. " + children + " The 20 studies were cross-sectional."}]}
        self.upload()
        self.post.side_effect = lambda url, **kw: (response(mismatch) if url.endswith("/ask")
                                                   else fake_post(url, **kw))
        self.submit(submission(text="How many studies?"))
        html = self.html()
        self.assertEqual(html.count('<span class="cite-num">'), 1)
        # No excerpt at all (not even the closest sentence, which here would be the rejected quote).
        self.assertNotIn("📌", html)
        self.assertNotIn("أقرب جملة في القسم — لم يُتحقق من اقتباس النموذج", html)
        self.assertNotIn('class="cite-quote', html)
        self.assertIn('<div class="cite-missing">لم يُعثر على مقتطف يدعم الرقم — راجع النص الكامل</div>', html)
        # The full text stays available.
        self.assertEqual([e.label for e in self.at.expander], ["عرض النص الكامل"])
        self.assertIn("A total of 24,600 children were included.", texts(self.at.expander[0].markdown))
        self.at.segmented_control(key="lang").set_value("en").run()
        self.assertIn("No excerpt supporting the number was found — see the full text", self.html())

    def test_upload_and_question_in_one_submission(self):
        self.upload()
        self.submit(submission(text="ما المقياس؟", files=[uploaded_file("p.pdf", b"P")]))
        self.assertEqual(self.endpoints(), ["upload", "upload", "ask"])
        self.assertEqual([m.name for m in self.at.chat_message], ["user", "assistant"] * 3)

    def test_history_survives_and_is_redrawn(self):
        self.upload()
        self.submit(submission(text="سؤال أول"))
        self.submit(submission(text="سؤال ثاني"))
        self.assertEqual(len(self.at.chat_message), 6)
        self.at.run()  # A plain rerun sends nothing again.
        self.assertEqual(self.endpoints(), ["upload", "ask", "ask"])

    # --- Reading answers aloud (TTS) ---

    def audio_players(self) -> list:
        return list(self.at.get("audio"))

    def test_listen_button_only_under_answers(self):
        self.upload()
        self.at.button[0].click().run()  # Summary card: no 🔊 button.
        self.assertEqual([b.label for b in self.at.button if "🔊" in b.label], [])
        self.submit(submission(text="ما المقياس؟"))
        self.assertEqual([b.label for b in self.at.button if "🔊" in b.label], ["🔊 استمع إلى الإجابة"])
        # Not played automatically: no request and no player until it is pressed.
        self.assertNotIn("speak", self.endpoints())
        self.assertEqual(self.audio_players(), [])

    def test_listen_reads_the_short_answer_once(self):
        self.upload()
        self.submit(submission(text="ما المقياس؟"))
        listen = [b for b in self.at.button if "🔊" in b.label][0]
        listen.click().run()
        self.assertFalse(self.at.exception)
        self.assertEqual(self.endpoints().count("speak"), 1)
        sent = self.post.call_args.kwargs["json"]["text"]
        # Only the short answer (as displayed, numbers renumbered), not the details or sources.
        self.assertEqual(sent, ANSWER["answer"])
        self.assertNotIn("نموذج التأثير العشوائي", sent)
        self.assertEqual(len(self.audio_players()), 1)
        # Pressing again, or redrawing the page, uses the saved audio.
        [b for b in self.at.button if "🔊" in b.label][0].click().run()
        self.at.run()
        self.assertEqual(self.endpoints().count("speak"), 1)
        self.assertEqual(len(self.audio_players()), 1)
        self.assertEqual(list(self.at.session_state.tts_audio.values()), [MP3])

    def test_listen_error_is_shown(self):
        self.upload()
        self.submit(submission(text="ما المقياس؟"))
        self.post.side_effect = lambda url, **kw: response({"detail": "OpenAI returned an error (HTTP 500)."}, 502)
        [b for b in self.at.button if "🔊" in b.label][0].click().run()
        self.assertIn("تعذّر تحويل الإجابة إلى صوت: OpenAI returned an error (HTTP 500).", texts(self.at.error))
        self.assertEqual(self.audio_players(), [])
        self.at.segmented_control(key="lang").set_value("en").run()
        self.assertEqual([b.label for b in self.at.button if "🔊" in b.label], ["🔊 Listen to the answer"])

    # --- Voice ---

    def record(self) -> AppTest:
        self.upload()
        return self.submit(submission(audio=uploaded_file("audio.wav", b"RIFF....WAVEfmt ", "audio/wav")))

    def user_texts(self) -> list[str]:
        return [m["text"] for m in self.at.session_state.messages if m["role"] == "user" and m["kind"] == "text"]

    def test_voice_is_transcribed_into_the_message_box_for_review(self):
        self.record()
        self.assertFalse(self.at.exception)
        # Transcribed only (no answer, no paper_id): the text waits in the message box.
        self.assertEqual(self.endpoints(), ["upload", "transcribe"])
        kwargs = self.post.call_args.kwargs
        self.assertEqual(kwargs["files"]["file"], ("audio.wav", b"RIFF....WAVEfmt ", "audio/wav"))
        self.assertNotIn("data", kwargs)
        self.assertEqual(self.at.session_state.chat, TRANSCRIPT)
        self.assertEqual(self.at.session_state.voice_draft, TRANSCRIPT)
        self.assertEqual(self.user_texts(), [])
        self.assertIn("راجع نص السؤال المفرَّغ في خانة الكتابة وعدّله إن لزم، ثم أرسله.", texts(self.at.caption))
        self.assertEqual(self.at.button(key="cancel-voice-draft").label, "إلغاء")
        self.at.segmented_control(key="lang").set_value("en").run()
        self.assertIn("Review the transcribed question in the message box", texts(self.at.caption))
        self.assertEqual(self.at.button(key="cancel-voice-draft").label, "Cancel")

    def test_the_reviewed_question_is_asked_like_a_typed_one(self):
        self.record()
        self.submit(submission(text="ما المقياس المستخدم في الدراسة؟"))  # Edited in the box.
        self.assertFalse(self.at.exception)
        self.assertEqual(self.endpoints(), ["upload", "transcribe", "ask"])
        self.assertEqual(self.post.call_args.kwargs["json"],
                         {"paper_id": PAPER_ID, "question": "ما المقياس المستخدم في الدراسة؟"})
        self.assertEqual(self.user_texts(), ["ما المقياس المستخدم في الدراسة؟"])
        self.assertIn("مفرّغ من الصوت", texts(self.at.caption))  # Marked 🎤, as edited.
        self.assertNotIn("voice_draft", self.at.session_state)
        self.assertNotIn("راجع نص السؤال", texts(self.at.caption))
        # The next typed question is not marked.
        self.submit(submission(text="سؤال مكتوب"))
        self.assertEqual(texts(self.at.caption).count("مفرّغ من الصوت"), 1)

    def test_cancel_empties_the_box_and_adds_nothing(self):
        self.record()
        messages = len(self.at.session_state.messages)
        self.at.button(key="cancel-voice-draft").click().run()
        self.assertFalse(self.at.exception)
        self.assertNotIn("voice_draft", self.at.session_state)
        self.assertEqual(self.at.session_state.chat, "")
        self.assertEqual(len(self.at.session_state.messages), messages)
        self.assertEqual(self.endpoints(), ["upload", "transcribe"])
        self.assertNotIn("راجع نص السؤال", texts(self.at.caption))

    def test_recording_again_replaces_the_draft(self):
        self.record()
        self.post.side_effect = lambda url, **kw: (response({"transcript": "سؤال جديد؟"}) if url.endswith("/transcribe")
                                                   else fake_post(url, **kw))
        # The old draft still in the box comes with the new recording: it is dropped, not asked.
        self.submit(submission(text=TRANSCRIPT, audio=uploaded_file("again.wav", b"RIFF", "audio/wav")))
        self.assertFalse(self.at.exception)
        self.assertEqual(self.endpoints(), ["upload", "transcribe", "transcribe"])
        self.assertEqual(self.at.session_state.chat, "سؤال جديد؟")
        self.assertEqual(self.at.session_state.voice_draft, "سؤال جديد؟")
        self.assertEqual(self.user_texts(), [])

    def test_a_new_paper_alone_drops_the_draft(self):
        self.record()
        self.submit(submission(files=[uploaded_file("other.pdf", b"%PDF-1.7 other")]))
        self.assertFalse(self.at.exception)
        self.assertEqual(self.endpoints(), ["upload", "transcribe", "upload"])
        self.assertNotIn("voice_draft", self.at.session_state)
        self.assertEqual(self.user_texts(), [])

    def test_voice_error_is_shown(self):
        self.upload()
        self.post.side_effect = lambda url, **kw: response({"detail": "No speech was recognized in the audio."}, 400)
        self.submit(submission(audio=uploaded_file("audio.wav", b"RIFF", "audio/wav")))
        self.assertIn("No speech was recognized", texts(self.at.error))
        self.assertIn("سؤال صوتي", self.html())


if __name__ == "__main__":
    unittest.main()
