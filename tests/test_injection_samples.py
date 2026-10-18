"""Tests for eval/make_injection_samples.py: the 8 prompt-injection files of eval/run_markers.py,
made from the original paper (they are not in git). They are made once, in a temporary folder.
No API.

Run: .\\venv\\Scripts\\python.exe -m unittest tests.test_injection_samples -v
"""

import sys
import tempfile
import unicodedata
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from docx import Document  # noqa: E402
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

    def test_the_filter_removes_the_whole_instruction(self):
        # Every run of 3 words of the instruction is gone, in the Arabic PDFs too (stored in visual
        # order: see hidden_text, "Arabic hidden text"), and only about the instruction's length goes.
        for path in self.made:
            lang, style = path.stem.split("_")[1:]
            words = samples.INSTRUCTIONS[(lang, style)].split()
            raw, filtered = text(path, hidden_filter=False), text(path, hidden_filter=True)
            flat = " ".join(unicodedata.normalize("NFKC", filtered.text).split())
            with self.subTest(path.name):
                self.assertEqual([" ".join(words[i:i + 3]) for i in range(len(words) - 2)
                                  if " ".join(words[i:i + 3]) in flat], [])
                removed = len(raw.text) - len(filtered.text)
                self.assertLessEqual(removed, len(samples.INSTRUCTIONS[(lang, style)]) + 5)

    def test_no_personal_metadata(self):
        for path in self.made:
            with self.subTest(path.name):
                if path.suffix == ".pdf":
                    metadata = {key: str(value) for key, value in (PdfReader(path).metadata or {}).items()}
                    self.assertEqual(set(metadata), {"/Producer", "/CreationDate", "/ModDate"})
                    self.assertEqual(metadata["/Producer"], samples.PRODUCER)
                    self.assertIsNone(PdfReader(path).xmp_metadata)  # No second, XMP copy of the metadata.
                else:
                    properties = Document(path).core_properties
                    # Empty, or the script's fixed value: never the name of whoever ran it.
                    self.assertIn(properties.author, ("", samples.PRODUCER))
                    self.assertIn(properties.last_modified_by, ("", samples.PRODUCER))


if __name__ == "__main__":
    unittest.main()
