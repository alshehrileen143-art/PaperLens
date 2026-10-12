"""Tests for ocr.py and the OCR part of extract.py: the RapidOCR settings, failures, the page limit
and the markers of pages not read. The engine is mocked, except in one local run of RapidOCR on a
small generated image (no API, no download).

Run: .\\venv\\Scripts\\python.exe -m unittest tests.test_ocr -v
"""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pymupdf  # noqa: E402

import config  # noqa: E402
import extract  # noqa: E402
import ocr  # noqa: E402

TEXT_PAGE = "Vitamin D deficiency was associated with obesity in children and adolescents. " * 3


def make_pdf(folder: Path, pages: list[str]) -> Path:
    """A PDF whose pages hold the given text ("" for a page with no text layer, like a scan)."""
    path = folder / "paper.pdf"
    with pymupdf.open() as document:
        for text in pages:
            page = document.new_page(width=595, height=842)  # A4 in points.
            if text:
                page.insert_textbox(pymupdf.Rect(50, 50, 545, 792), text, fontsize=11)
        document.save(path)
    return path


class FakeEngine:
    def __init__(self, params=None):
        FakeEngine.params = params
        FakeEngine.calls = []

    def __call__(self, image):
        FakeEngine.calls.append(image.shape)
        return SimpleNamespace(txts=("line one", "line two"))


class EngineTests(unittest.TestCase):
    def setUp(self):
        patch.object(ocr, "_ocr_engine", None).start()
        self.addCleanup(patch.stopall)

    def test_rapidocr_settings(self):
        with patch("rapidocr.RapidOCR", FakeEngine):
            text = ocr._ocr_image(np.zeros((40, 40, 3), dtype=np.uint8))
        self.assertEqual(text, "line one\nline two")
        # Orientation model off, and no shrinking of a 200 DPI page (longest side about 2340 px).
        self.assertEqual(FakeEngine.params, {"Global.use_cls": False, "Global.max_side_len": 4000,
                                             "Global.log_level": "error"})
        self.assertEqual(config.OCR_DPI, 200)
        self.assertEqual(config.MAX_OCR_PAGES, 13)

    def test_missing_model_file_is_an_error_not_a_download(self):
        with patch.object(ocr, "MODEL_FILES", ("PP-OCRv6_det_small.onnx", "not-there.onnx")), \
                patch("rapidocr.RapidOCR") as engine:
            with self.assertRaises(ocr.OCRError) as caught:
                ocr._ocr_image(np.zeros((40, 40, 3), dtype=np.uint8))
        self.assertIn("not-there.onnx", str(caught.exception))
        engine.assert_not_called()

    def test_engine_failure_is_an_ocr_error(self):
        class Broken(FakeEngine):
            def __call__(self, image):
                raise RuntimeError("onnxruntime failed")
        with patch("rapidocr.RapidOCR", Broken):
            with self.assertRaises(ocr.OCRError):
                ocr._ocr_image(np.zeros((40, 40, 3), dtype=np.uint8))

    def test_pages_are_rendered_at_200_dpi(self):
        with tempfile.TemporaryDirectory() as folder, patch("rapidocr.RapidOCR", FakeEngine):
            path = make_pdf(Path(folder), ["", ""])
            texts = ocr.ocr_pdf_pages(path, [1])
        self.assertEqual(texts, {1: "line one\nline two"})
        self.assertEqual(FakeEngine.calls, [(2339, 1653, 3)])  # 842 x 595 points at 200 DPI.


class SkippedPagesTests(unittest.TestCase):
    """Scanned pages over MAX_OCR_PAGES: not read, and a marker holds each one's place."""

    def fake_ocr(self, path, indexes):
        self.read = list(indexes)
        return {index: f"OCR text of page {index + 1}" for index in indexes}

    def test_mixed_pdf(self):
        # Pages 1 and 3 have a text layer; the other 14 are scans. Page 16 keeps a stray header.
        pages = [TEXT_PAGE, "", TEXT_PAGE] + [""] * 12 + ["Running header"]
        with tempfile.TemporaryDirectory() as folder, patch.object(extract, "ocr_pdf_pages", self.fake_ocr):
            result = extract.extract_text(make_pdf(Path(folder), pages))
        self.assertEqual(self.read, [1] + list(range(3, 15)))  # The first 13 scanned pages.
        self.assertEqual(result.ocr_skipped_pages, (16,))
        self.assertTrue(result.ocr_used)
        # The marker is where the page is, followed by the little text the page had.
        self.assertEqual(result.pages[15], "[Page 16 was not read: OCR is limited to 13 scanned pages per paper.]\n"
                                           "Running header")
        self.assertEqual(result.pages[14], "OCR text of page 15")
        self.assertIn("Vitamin D deficiency", result.pages[0])  # Text pages are not OCR'd.
        self.assertEqual(result.text, "\n".join(result.pages))
        self.assertNotIn("more scanned page(s) were skipped", result.text)  # The old note at the end.

    def test_no_page_skipped(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(extract, "ocr_pdf_pages", self.fake_ocr):
            result = extract.extract_text(make_pdf(Path(folder), [TEXT_PAGE, ""]))
        self.assertEqual(result.ocr_skipped_pages, ())
        self.assertNotIn("was not read", result.text)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "paper.txt"
            path.write_text("Plain text.", encoding="utf-8")
            self.assertEqual(extract.extract_text(path).ocr_skipped_pages, ())

    def test_extract_text_ocr_marks_pdf_pages_over_the_limit(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(ocr, "ocr_pdf_pages", self.fake_ocr):
            text = ocr.extract_text_ocr(make_pdf(Path(folder), [""] * 15))
        self.assertEqual(self.read, list(range(13)))
        parts = text.split("\n\n")
        self.assertEqual(len(parts), 15)
        self.assertEqual(parts[12], "OCR text of page 13")
        self.assertEqual(parts[13], ocr.skipped_page_marker(14))
        self.assertEqual(parts[14], ocr.skipped_page_marker(15))


@unittest.skipUnless(importlib.util.find_spec("rapidocr"), "rapidocr is not installed")
class RapidOCRRunTests(unittest.TestCase):
    """One real run of the engine on a generated image: local, a few seconds, no download."""

    def test_reads_text_and_numbers(self):
        with pymupdf.open() as document:
            page = document.new_page(width=420, height=120)
            page.insert_text((20, 50), "A total of 24,600 children", fontsize=22)
            page.insert_text((20, 95), "Relative risk 1.41", fontsize=22)
            image = ocr._pixmap_to_bgr(page.get_pixmap(dpi=200))
        with patch("rapidocr.utils.download_file.DownloadFile._make_http_request",
                   side_effect=AssertionError("a model download was attempted")):
            text = ocr._ocr_image(image)
        self.assertIn("24,600", text)
        self.assertIn("1.41", text)
        self.assertIn("children", text)


if __name__ == "__main__":
    unittest.main()
