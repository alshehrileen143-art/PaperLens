"""Tests for text to speech: tts.clean_for_speech, tts.synthesize and the /speak endpoint. OpenAI is mocked.

Run: .\\venv\\Scripts\\python.exe -m unittest tests.test_tts -v
"""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

import config  # noqa: E402
import main  # noqa: E402
import tts  # noqa: E402
from llm import LLMError, MissingAPIKeyError  # noqa: E402
from tts import clean_for_speech  # noqa: E402

MP3 = b"ID3\x04\x00fake-mp3"


class CleanForSpeechTests(unittest.TestCase):
    def test_removes_citation_markers(self):
        self.assertEqual(clean_for_speech("The RR was 1.41 [1]. Both studies agree [2,3] and [2, 3]."),
                         "The RR was 1.41. Both studies agree and.")

    def test_removes_markdown(self):
        self.assertEqual(clean_for_speech("# Result\n**Bold** and *italic* with `code`."),
                         "Result Bold and italic with code.")

    def test_keeps_numbers_brackets_and_arabic(self):
        self.assertEqual(clean_for_speech("نسبة المخاطر (RR) 1.41 [1]، وفترة الثقة 95%."),
                         "نسبة المخاطر (RR) 1.41، وفترة الثقة 95%.")
        self.assertEqual(clean_for_speech("Levels of 25(OH)D [note]"), "Levels of 25(OH)D [note]")

    def test_collapses_whitespace(self):
        self.assertEqual(clean_for_speech("  a \n\n b\t c  "), "a b c")

    def test_only_markers_is_empty(self):
        self.assertEqual(clean_for_speech(" [1] ** [2,3] `"), "")


class SynthesizeTests(unittest.TestCase):
    def test_uses_configured_settings(self):
        calls = {}

        def create(**kwargs):
            calls.update(kwargs)
            return SimpleNamespace(content=MP3)

        client = SimpleNamespace(audio=SimpleNamespace(speech=SimpleNamespace(create=create)))
        with patch.object(tts, "get_client", return_value=client):
            self.assertEqual(tts.synthesize("Hello."), MP3)
        self.assertEqual(calls, {"model": "gpt-4o-mini-tts", "voice": "alloy", "input": "Hello.",
                                 "instructions": config.TTS_INSTRUCTIONS, "response_format": "mp3"})


class SpeakEndpointTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main.app)

    def speak(self, text: str, **synth):
        with patch.object(main, "synthesize", **synth) as synthesize:
            return self.client.post("/speak", json={"text": text}), synthesize

    def test_ok_returns_mp3(self):
        response, synthesize = self.speak("The RR was **1.41** [1].", return_value=MP3)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "audio/mpeg")
        self.assertEqual(response.content, MP3)
        synthesize.assert_called_once_with("The RR was 1.41.")  # Cleaned before speaking.

    def test_empty_after_cleaning_is_400(self):
        for text in ("", "   ", "[1] [2,3] **"):
            with self.subTest(repr(text)):
                response, synthesize = self.speak(text, return_value=MP3)
                self.assertEqual(response.status_code, 400)
                synthesize.assert_not_called()

    def test_too_long_is_413(self):
        response, synthesize = self.speak("a" * (config.TTS_MAX_CHARS + 1), return_value=MP3)
        self.assertEqual(response.status_code, 413)
        self.assertIn(str(config.TTS_MAX_CHARS), response.json()["detail"])
        synthesize.assert_not_called()

    def test_limit_counts_the_cleaned_text(self):
        text = "a" * config.TTS_MAX_CHARS + " [1]" * 50  # Over the limit only because of markers.
        response, _ = self.speak(text, return_value=MP3)
        self.assertEqual(response.status_code, 200)

    def test_openai_error_is_502_without_internals(self):
        response, _ = self.speak("Hello.", side_effect=LLMError("OpenAI returned an error (HTTP 500)."))
        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.json(), {"detail": "OpenAI returned an error (HTTP 500)."})

    def test_missing_key_is_500(self):
        response, _ = self.speak("Hello.", side_effect=MissingAPIKeyError("OPENAI_API_KEY is not set. Add it to the .env file."))
        self.assertEqual(response.status_code, 500)

    def test_openai_sdk_error_becomes_502(self):
        """A real SDK exception goes through llm.openai_errors: no stack trace or request details leak."""
        import httpx
        import openai

        request = httpx.Request("POST", "https://api.openai.com/v1/audio/speech")
        error = openai.APIStatusError("internal secret detail", response=httpx.Response(500, request=request), body=None)

        def create(**kwargs):
            raise error

        client = SimpleNamespace(audio=SimpleNamespace(speech=SimpleNamespace(create=create)))
        with patch.object(tts, "get_client", return_value=client):
            response = self.client.post("/speak", json={"text": "Hello."})
        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.json(), {"detail": "OpenAI returned an error (HTTP 500)."})
        self.assertNotIn("secret", response.text)


if __name__ == "__main__":
    unittest.main()
