"""Tests for eval/make_injection_samples.py: the 8 prompt-injection files of eval/run_markers.py,
made from the original paper (they are not in git). They are made once, in a temporary folder.
No API.

Run: .\\venv\\Scripts\\python.exe -m unittest tests.test_injection_samples -v
"""

import sys
import tempfile
import unicodedata
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from pypdf import PdfReader  # noqa: E402

import config  # noqa: E402
import extract  # noqa: E402
import make_injection_samples as samples  # noqa: E402

MARKER = "CANARY-7431"
NAMES = sorted(f"{kind}_{lang}_{style}.{ext}" for kind, ext in (("white", "pdf"), ("hidden", "docx"))
               for lang in ("ar", "en") for style in ("blunt", "polite"))


def text(path: Path, hidden_filter: bool) -> extract.ExtractionResult:
    with patch.object(config, "HIDDEN_TEXT_FILTER", hidden_filter):
        return extract.extract_text(path)


@unittest.skipUnless(any(font.is_file() for font in samples.ARABIC_FONTS), "no font with Arabic letters")
class InjectionSamplesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        cls.out = Path(cls.folder.name)
        cls.made = samples.make_all(cls.out)

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def test_eight_files_with_the_marker(self):
        self.assertEqual([path.name for path in self.made], NAMES)
        for path in self.made:
            with self.subTest(path.name):
                raw = " ".join(unicodedata.normalize("NFKC", text(path, hidden_filter=False).text).split())
                lang, style = path.stem.split("_")[1:]
                self.assertIn(MARKER, raw)
                # The instruction word for word (the Arabic PDFs give presentation forms; NFKC reads them back).
                self.assertIn(samples.INSTRUCTIONS[(lang, style)], raw)
                self.assertIn("Vitamin D deficiency in children and adolescents", raw)  # The title.

    def test_the_hidden_text_filter_removes_the_marker(self):
        for path in self.made:
            with self.subTest(path.name):
                result = text(path, hidden_filter=True)
                self.assertNotIn(MARKER, result.text)
                self.assertGreater(result.hidden_removed["count"], 0)

    def test_the_filter_removes_the_whole_instruction_except_in_arabic_pdfs(self):
        # Known gap: in the Arabic PDFs, PyMuPDF gives each word's letters in visual order and pypdf
        # in reading order, so the filter finds only the marker (Latin) and the Arabic words stay.
        for path in self.made:
            lang, style = path.stem.split("_")[1:]
            words = samples.INSTRUCTIONS[(lang, style)].split()[:3]
            filtered = " ".join(unicodedata.normalize("NFKC", text(path, hidden_filter=True).text).split())
            with self.subTest(path.name):
                if path.suffix == ".pdf" and lang == "ar":
                    self.assertIn(" ".join(words), filtered)
                else:
                    self.assertNotIn(" ".join(words), filtered)

    def test_no_personal_metadata(self):
        for path in self.made:
            with self.subTest(path.name):
                if path.suffix == ".pdf":
                    metadata = {key: str(value) for key, value in (PdfReader(path).metadata or {}).items()}
                    self.assertEqual(set(metadata), {"/Producer", "/CreationDate", "/ModDate"})
                    self.assertEqual(metadata["/Producer"], samples.PRODUCER)
                    raw = path.read_bytes()
                else:
                    with zipfile.ZipFile(path) as docx:
                        raw = b"".join(docx.read(name) for name in docx.namelist() if name.startswith("docProps/"))
                    self.assertIn(samples.PRODUCER.encode(), raw)
                for name in ("leen", "alshehri", "الشهري", "لين"):
                    self.assertNotIn(name.encode("utf-8"), raw.lower())


if __name__ == "__main__":
    unittest.main()
