"""Tests for /ask_voice and audio validation. No OpenAI calls: transcription and answering are mocked.

Run: .\\venv\\Scripts\\python.exe -m unittest tests.test_ask_voice -v
"""

import io
import struct
import sys
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

import main  # noqa: E402
from config import MAX_AUDIO_BYTES, MAX_QUESTION_CHARS  # noqa: E402
from llm import LLMError  # noqa: E402
from rag import AnswerResult, PaperNotFoundError  # noqa: E402

PAPER_ID = "8db0b76c45a54a3a8159b8b2824208a9"


# --- Audio builders ---

def make_wav(seconds: float, rate: int = 8000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(1)
        wav.setframerate(rate)
        wav.writeframes(b"\x80" * int(seconds * rate))
    return buffer.getvalue()


def make_mp3(seconds: float, prefix: bytes = b"") -> bytes:
    # MPEG-1 Layer III, 128 kbps, 44.1 kHz: 417-byte frames of 1152 samples.
    frame = b"\xff\xfb\x90\x64" + b"\x00" * 413
    return prefix + frame * int(seconds * 44100 / 1152)


def box(box_type: bytes, body: bytes) -> bytes:
    return struct.pack(">I4s", 8 + len(body), box_type) + body


def make_m4a(header_seconds: float, sample_seconds: float, handler: bytes = b"soun") -> bytes:
    timescale = 1000
    mdhd = box(b"mdhd", struct.pack(">IIIII", 0, 0, 0, timescale, int(header_seconds * timescale)) + b"\x00" * 4)
    hdlr = box(b"hdlr", b"\x00" * 8 + handler + b"\x00" * 12 + b"audio\x00")
    # 1024-sample AAC frames at the track's timescale.
    count = int(sample_seconds * timescale / 1024)
    stts = box(b"stts", struct.pack(">III", 0, 1, count) + struct.pack(">I", 1024))
    stbl = box(b"stbl", stts)
    trak = box(b"trak", box(b"mdia", mdhd + hdlr + box(b"minf", stbl)))
    ftyp = box(b"ftyp", b"M4A \x00\x00\x00\x00M4A isom")
    return ftyp + box(b"moov", trak) + box(b"mdat", b"\x00" * 100)


def element(element_id: int, body: bytes, unknown_size: bool = False) -> bytes:
    id_bytes = element_id.to_bytes((element_id.bit_length() + 7) // 8, "big")
    size = b"\x01" + (b"\xff" * 7 if unknown_size else len(body).to_bytes(7, "big"))
    return id_bytes + size + body


def make_webm(block_ms: list[int], declared_ms: float | None = None, doctype: bytes = b"webm") -> bytes:
    header = element(0x1A45DFA3, element(0x4282, doctype))
    info_body = element(0x2AD7B1, (1_000_000).to_bytes(3, "big"))
    if declared_ms is not None:
        info_body += element(0x4489, struct.pack(">d", declared_ms))
    clusters = b""
    for start in range(0, len(block_ms), 10):  # Ten blocks per cluster, timestamps relative to it.
        group = block_ms[start:start + 10]
        blocks = b"".join(element(0xA3, b"\x81" + struct.pack(">h", ms - group[0]) + b"\x80" + b"\x00" * 20)
                          for ms in group)
        clusters += element(0x1F43B675, element(0xE7, group[0].to_bytes(4, "big")) + blocks, unknown_size=True)
    # Browser recordings (MediaRecorder) use unknown sizes and often omit Duration.
    return header + element(0x18538067, element(0x1549A966, info_body) + clusters, unknown_size=True)


def answer(text: str = "Vitamin D levels were lower.") -> AnswerResult:
    return AnswerResult(text, True, [], "full")


class AskVoiceTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main.app)
        self.paper = patch.object(main, "get_paper_collection")
        self.paper.start()
        self.addCleanup(self.paper.stop)

    def post(self, filename: str, content: bytes, paper_id: str = PAPER_ID):
        return self.client.post("/ask_voice", data={"paper_id": paper_id},
                                files={"file": (filename, content, "application/octet-stream")})

    # --- Successful path ---

    def test_returns_ask_response_plus_transcript(self):
        seen = {}

        def fake_transcribe(path):
            seen["suffix"], seen["exists"] = path.suffix, path.exists()
            seen["path"] = path
            return "  What was the main finding?  "

        with patch.object(main, "transcribe", side_effect=fake_transcribe), \
                patch.object(main, "answer_question", return_value=answer()) as ask:
            response = self.post("q.wav", make_wav(3))
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["transcript"], "What was the main finding?")
        self.assertEqual(body["question"], "What was the main finding?")
        self.assertEqual(body["answer"], "Vitamin D levels were lower.")
        self.assertEqual(set(body) - {"transcript"}, set(main.AskResponse.model_fields))
        ask.assert_called_once_with(PAPER_ID, "What was the main finding?")
        # The extension is kept for OpenAI, and the temp file is deleted afterwards.
        self.assertEqual(seen["suffix"], ".wav")
        self.assertTrue(seen["exists"])
        self.assertFalse(seen["path"].exists())

    def test_every_supported_format_is_accepted(self):
        files = {
            "q.mp3": make_mp3(5),
            "q.mp3 (ID3 tag first)": make_mp3(5, prefix=b"ID3\x04\x00\x00\x00\x00\x00\x0a" + b"\x00" * 10),
            "q.wav": make_wav(5),
            "q.m4a": make_m4a(5, 5),
            "q.webm": make_webm(list(range(0, 5000, 20))),
            "q.webm (with Duration)": make_webm(list(range(0, 5000, 20)), declared_ms=5000),
        }
        for name, content in files.items():
            with self.subTest(name), patch.object(main, "transcribe", return_value="Hi?"), \
                    patch.object(main, "answer_question", return_value=answer()):
                filename = name.split(" ")[0]
                response = self.post(filename.upper() if "ID3" in name else filename, content)
                self.assertEqual(response.status_code, 200, response.text)

    def test_ask_still_works(self):
        with patch.object(main, "answer_question", return_value=answer()):
            response = self.client.post("/ask", json={"paper_id": PAPER_ID, "question": " Q? "})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["question"], "Q?")
        self.assertNotIn("transcript", response.json())

    # --- Rejections that must happen before transcription ---

    def assert_rejected(self, filename, content, status, detail_part, paper_id=PAPER_ID):
        with patch.object(main, "transcribe") as transcribe, patch.object(main, "answer_question") as ask:
            response = self.post(filename, content, paper_id)
        self.assertEqual(response.status_code, status, response.text)
        self.assertIn(detail_part, response.json()["detail"])
        transcribe.assert_not_called()
        ask.assert_not_called()

    def test_unsupported_extension(self):
        self.assert_rejected("q.ogg", b"OggS" + b"\x00" * 100, 400, "Unsupported audio type")

    def test_content_does_not_match_extension(self):
        self.assert_rejected("q.wav", make_mp3(2), 400, "does not match")
        self.assert_rejected("q.mp3", b"%PDF-1.7 not audio", 400, "does not match")
        self.assert_rejected("q.m4a", make_wav(2), 400, "does not match")
        self.assert_rejected("q.webm", make_webm([0, 20], doctype=b"matroska"), 400, "does not match")

    def test_too_large(self):
        self.assert_rejected("q.wav", make_wav(1) + b"\x00" * MAX_AUDIO_BYTES, 413, "Maximum size is 5 MB")

    def test_too_long_in_every_format(self):
        for filename, content in {
            "q.wav": make_wav(61),
            "q.mp3": make_mp3(61),
            "q.m4a": make_m4a(61, 61),
            "q.webm": make_webm(list(range(0, 61_000, 100))),
        }.items():
            with self.subTest(filename):
                self.assert_rejected(filename, content, 413, "Maximum is 60 seconds")

    def test_headers_cannot_understate_duration(self):
        # Header says 10 s, the samples / blocks actually last 120 s.
        self.assert_rejected("q.m4a", make_m4a(10, 120), 413, "Maximum is 60 seconds")
        self.assert_rejected("q.webm", make_webm(list(range(0, 120_000, 1000)), declared_ms=10_000),
                             413, "Maximum is 60 seconds")

    def test_corrupted_or_empty(self):
        self.assert_rejected("q.mp3", b"ID3\x04\x00\x00\x00\x00\x00\x0a" + b"\x00" * 50, 400, "Could not read")
        self.assert_rejected("q.m4a", make_m4a(5, 5, handler=b"vide"), 400, "no audio track")
        self.assert_rejected("q.webm", make_webm([0, 20])[:-10], 400, "Could not read")
        self.assert_rejected("q.wav", make_wav(0), 400, "Could not read")

    def test_unknown_paper_is_checked_before_transcription(self):
        self.paper.stop()
        with patch.object(main, "get_paper_collection", side_effect=PaperNotFoundError("x")):
            self.assert_rejected("q.wav", make_wav(2), 404, "Paper not found", paper_id="0" * 32)
        self.paper.start()

    # --- Transcription results ---

    def test_empty_transcript(self):
        for transcript in ("", "   "):
            with self.subTest(repr(transcript)), \
                    patch.object(main, "transcribe", return_value=transcript.strip()), \
                    patch.object(main, "answer_question") as ask:
                response = self.post("q.wav", make_wav(2))
            self.assertEqual(response.status_code, 400)
            self.assertIn("No speech was recognized", response.json()["detail"])
            ask.assert_not_called()

    def test_transcript_too_long(self):
        with patch.object(main, "transcribe", return_value="a" * (MAX_QUESTION_CHARS + 1)), \
                patch.object(main, "answer_question") as ask:
            response = self.post("q.wav", make_wav(2))
        self.assertEqual(response.status_code, 400)
        ask.assert_not_called()

    def test_openai_failure_is_502(self):
        with patch.object(main, "transcribe", side_effect=LLMError("OpenAI returned an error (HTTP 500).")):
            response = self.post("q.wav", make_wav(2))
        self.assertEqual(response.status_code, 502)


class TranscribeEndpointTests(unittest.TestCase):
    """/transcribe: the same audio checks and limits as /ask_voice, and the text only (no answer)."""

    def setUp(self):
        self.client = TestClient(main.app)
        self.answer = patch.object(main, "answer_question").start()
        self.addCleanup(patch.stopall)

    def post(self, filename: str, content: bytes):
        return self.client.post("/transcribe", files={"file": (filename, content, "application/octet-stream")})

    def test_returns_the_transcript_only(self):
        with patch.object(main, "transcribe", return_value="  What was the main finding?  "):
            response = self.post("q.webm", make_webm(list(range(0, 3000, 20))))
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json(), {"transcript": "What was the main finding?"})
        self.answer.assert_not_called()

    def test_no_paper_needed(self):
        with patch.object(main, "transcribe", return_value="Q?"),                 patch.object(main, "get_paper_collection") as paper:
            self.assertEqual(self.post("q.wav", make_wav(2)).status_code, 200)
        paper.assert_not_called()

    def test_audio_is_checked_before_transcription(self):
        for filename, content, status, detail in (
                ("q.ogg", b"OggS" + b"\x00" * 100, 400, "Unsupported audio type"),
                ("q.wav", make_mp3(2), 400, "does not match"),
                ("q.wav", make_wav(1) + b"\x00" * MAX_AUDIO_BYTES, 413, "Maximum size is 5 MB"),
                ("q.wav", make_wav(61), 413, "Maximum is 60 seconds"),
                ("q.wav", make_wav(0), 400, "Could not read")):
            with self.subTest(detail), patch.object(main, "transcribe") as transcribe:
                response = self.post(filename, content)
            self.assertEqual(response.status_code, status, response.text)
            self.assertIn(detail, response.json()["detail"])
            transcribe.assert_not_called()

    def test_empty_or_too_long_transcript(self):
        for transcript, detail in (("   ", "No speech was recognized"),
                                   ("a" * (MAX_QUESTION_CHARS + 1), f"longer than {MAX_QUESTION_CHARS}")):
            with self.subTest(detail), patch.object(main, "transcribe", return_value=transcript):
                response = self.post("q.wav", make_wav(2))
            self.assertEqual(response.status_code, 400)
            self.assertIn(detail, response.json()["detail"])

    def test_openai_failure_is_502(self):
        with patch.object(main, "transcribe", side_effect=LLMError("OpenAI returned an error (HTTP 500).")):
            self.assertEqual(self.post("q.wav", make_wav(2)).status_code, 502)


class TranscribeTests(unittest.TestCase):
    def test_uses_configured_model(self):
        import stt

        class FakeClient:
            class audio:
                class transcriptions:
                    @staticmethod
                    def create(model, file):
                        FakeClient.called = (model, Path(file.name).suffix)
                        return type("R", (), {"text": " hello \n"})()

        path = Path(__file__).with_name("_tmp_test.wav")
        path.write_bytes(make_wav(1))
        try:
            with patch.object(stt, "get_client", return_value=FakeClient):
                self.assertEqual(stt.transcribe(path), " hello \n")
        finally:
            path.unlink()
        self.assertEqual(FakeClient.called, ("whisper-1", ".wav"))


if __name__ == "__main__":
    unittest.main()
