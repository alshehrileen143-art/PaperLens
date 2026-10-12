"""Tests for /upload's duplicate check (one stored copy per paper). No OpenAI calls:
embeddings are mocked, and the paper store and Chroma are temporary folders.

Run: .\\venv\\Scripts\\python.exe -m unittest tests.test_upload -v
"""

import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chromadb  # noqa: E402
from chromadb.config import Settings  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import config  # noqa: E402
import main  # noqa: E402
import rag  # noqa: E402
from extract import EXTRACTOR_VERSION, ExtractionResult  # noqa: E402
from llm import LLMError  # noqa: E402

PAPER = ("Vitamin D and asthma in children.\n\nMethods\nWe searched PubMed and Embase for studies "
         "published before 2020. A total of 24,600 children were included.\n\nResults\nThe pooled "
         "relative risk was 1.41 (95% CI 1.26-1.59).\n\nDiscussion\n"
         + " ".join(f"Study {n} measured serum 25(OH)D in {100 + n} children." for n in range(60)) + "\n")
OTHER = PAPER.replace("1.41", "1.52")


def fake_embeddings(texts: list[str]) -> list[list[float]]:
    return [[float(index + 1), 1.0, 0.5] for index, _ in enumerate(texts)]


class UploadTestCase(unittest.TestCase):
    """A temporary paper store and ChromaDB, mocked embeddings, and helpers. No tests of its own."""
    def setUp(self):
        folder = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)  # Chroma may keep its files open.
        self.addCleanup(folder.cleanup)
        self.store = Path(folder.name) / "paper_store"
        self.chroma = chromadb.PersistentClient(path=str(Path(folder.name) / "chroma"),
                                                settings=Settings(anonymized_telemetry=False))
        patch.object(config, "PAPER_STORE_DIR", str(self.store)).start()
        patch.object(rag, "_client", self.chroma).start()
        self.embed = patch.object(rag, "embed_texts", side_effect=fake_embeddings).start()
        self.addCleanup(patch.stopall)
        rag._bm25_index.cache_clear()
        self.client = TestClient(main.app)

    def upload(self, text: str = PAPER, name: str = "paper.txt") -> dict:
        response = self.client.post("/upload", files={"file": (name, text.encode("utf-8"), "text/plain")})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def stored(self) -> list[str]:
        return sorted(path.stem for path in self.store.glob("*.json"))

    def data(self, paper_id: str) -> dict:
        return json.loads((self.store / f"{paper_id}.json").read_text(encoding="utf-8"))

    def rewrite(self, paper_id: str, **changes) -> None:
        """Change a stored paper's file; a None value removes the field (a paper stored by older code)."""
        data = {**self.data(paper_id), **changes}
        data = {key: value for key, value in data.items() if value is not None}
        (self.store / f"{paper_id}.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def extracted(self):
        """A spy on extraction: its call_count tells whether a file was extracted (and OCR'd)."""
        return patch.object(main, "extract_text", wraps=main.extract_text)

    def collection(self, paper_id: str):
        return self.chroma.get_collection(f"paper_{paper_id}")


class UploadDuplicateTests(UploadTestCase):
    def test_same_file_twice_returns_the_same_paper(self):
        first = self.upload()
        self.assertFalse(first["duplicate"])
        second = self.upload()
        self.assertTrue(second["duplicate"])
        self.assertEqual(second["paper_id"], first["paper_id"])
        self.assertEqual(second["chunk_count"], first["chunk_count"])
        self.assertEqual(second["normalization"], first["normalization"])
        self.assertEqual(self.embed.call_count, 1)  # No second embeddings.
        self.assertEqual(self.stored(), [first["paper_id"]])
        self.assertEqual(len(self.chroma.list_collections()), 1)
        # The new paper's file holds its fingerprint.
        data = json.loads((self.store / f"{first['paper_id']}.json").read_text(encoding="utf-8"))
        self.assertEqual(data["text_sha256"], rag.text_fingerprint(PAPER))

    def test_a_duplicate_is_neither_chunked_nor_indexed(self):
        first = self.upload()
        with patch.object(rag, "chunk_text", wraps=rag.chunk_text) as chunk, \
                patch.object(rag, "index_paper", wraps=rag.index_paper) as index:
            second = self.upload()
        self.assertEqual(second["paper_id"], first["paper_id"])
        chunk.assert_not_called()
        index.assert_not_called()
        self.assertEqual(self.embed.call_count, 1)

    def test_a_different_paper_gets_a_new_id(self):
        first = self.upload()
        second = self.upload(OTHER)
        self.assertFalse(second["duplicate"])
        self.assertNotEqual(second["paper_id"], first["paper_id"])
        self.assertEqual(self.embed.call_count, 2)
        self.assertEqual(len(self.stored()), 2)

    def test_the_fingerprint_is_the_text_not_the_file(self):
        # Another file name and other whitespace: the same paper.
        first = self.upload()
        second = self.upload(PAPER.replace("\n\n", "\n \n  ").replace(" were ", "   were "), "copy (2).txt")
        self.assertTrue(second["duplicate"])
        self.assertEqual(second["paper_id"], first["paper_id"])
        self.assertEqual(second["filename"], "copy (2).txt")
        # Only whitespace is normalized: case is not.
        self.assertNotEqual(rag.text_fingerprint(PAPER), rag.text_fingerprint(PAPER.upper()))

    def test_papers_stored_before_fingerprints_are_found(self):
        first = self.upload()
        # As stored before fingerprints, versions and file hashes existed.
        self.rewrite(first["paper_id"], text_sha256=None, extractor=None, file_sha256s=None)
        second = self.upload()
        self.assertTrue(second["duplicate"])
        self.assertEqual(second["paper_id"], first["paper_id"])
        self.assertEqual(self.embed.call_count, 1)

    def test_a_missing_collection_is_indexed_again_under_the_same_id(self):
        first = self.upload()
        self.chroma.delete_collection(f"paper_{first['paper_id']}")
        second = self.upload()
        self.assertTrue(second["duplicate"])
        self.assertEqual(second["paper_id"], first["paper_id"])
        self.assertEqual(second["chunk_count"], first["chunk_count"])
        self.assertEqual(self.embed.call_count, 2)
        self.assertEqual(self.collection(first["paper_id"]).count(), first["chunk_count"])
        self.assertEqual(self.stored(), [first["paper_id"]])

    def test_an_empty_or_incomplete_collection_is_indexed_again(self):
        first = self.upload()
        paper_id = first["paper_id"]
        self.assertGreater(first["chunk_count"], 1)
        self.collection(paper_id).delete(ids=["0"])  # One chunk lost.
        self.assertTrue(self.upload()["duplicate"])
        self.assertEqual(self.collection(paper_id).count(), first["chunk_count"])
        collection = self.collection(paper_id)
        collection.delete(ids=collection.get()["ids"])  # Empty.
        self.assertEqual(self.upload()["paper_id"], paper_id)
        self.assertEqual(self.collection(paper_id).count(), first["chunk_count"])
        self.assertEqual(self.embed.call_count, 3)

    def test_a_copy_indexed_with_other_settings_is_indexed_again(self):
        first = self.upload()
        with patch.object(config, "RAG_TAG_SECTIONS", not config.RAG_TAG_SECTIONS), \
                patch.object(config, "RAG_SECTIONS", False):
            second = self.upload()
        self.assertEqual(second["paper_id"], first["paper_id"])
        self.assertEqual(self.embed.call_count, 2)

    def test_failed_embeddings_leave_the_broken_copy_as_it_was(self):
        first = self.upload()
        paper_id = first["paper_id"]
        self.collection(paper_id).delete(ids=["0"])
        before = self.collection(paper_id).count()
        stored = (self.store / f"{paper_id}.json").read_text(encoding="utf-8")
        self.embed.side_effect = LLMError("OpenAI request failed.")
        response = self.client.post("/upload", files={"file": ("paper.txt", PAPER.encode("utf-8"), "text/plain")})
        self.assertEqual(response.status_code, 502)
        self.assertEqual(self.collection(paper_id).count(), before)
        self.assertEqual((self.store / f"{paper_id}.json").read_text(encoding="utf-8"), stored)
        self.assertEqual(self.stored(), [paper_id])

    def test_the_oldest_usable_copy_is_chosen(self):
        # Three copies of one paper, as uploads made before this check left behind.
        result = ExtractionResult(PAPER, ocr_used=False)
        ids = [rag.index_paper(result, "paper.txt")[0] for _ in range(3)]
        for age, paper_id in zip((300, 200, 100), ids):  # ids[0] is the oldest.
            path = self.store / f"{paper_id}.json"
            os.utime(path, (path.stat().st_atime, path.stat().st_mtime - age))
        self.assertEqual(self.upload()["paper_id"], ids[0])
        self.chroma.delete_collection(f"paper_{ids[0]}")  # The oldest is broken: the next one.
        self.assertEqual(self.upload()["paper_id"], ids[1])
        self.assertEqual(self.embed.call_count, 3)  # Only the three copies made above.
        for paper_id in ids[1:]:  # None usable: the oldest is indexed again.
            self.chroma.delete_collection(f"paper_{paper_id}")
        self.assertEqual(self.upload()["paper_id"], ids[0])
        self.assertEqual(self.embed.call_count, 4)

    def test_evaluation_indexing_does_not_look_for_copies(self):
        first = self.upload()
        with patch.object(rag, "_stored_papers_with") as lookup:
            paper_id, _, _ = rag.index_paper(ExtractionResult(PAPER, ocr_used=False), "paper.txt", store_text=False)
        lookup.assert_not_called()
        self.assertNotEqual(paper_id, first["paper_id"])
        self.assertEqual(self.stored(), [first["paper_id"]])



SUMMARY = {
    "objective": "Explain vitamin D.", "methodology": "", "key_findings": [], "limitations": [],
    "use_cases": [], "technical_use_cases": [], "ai_ml_tasks": [],
    "data_and_tools": {"datasets": [], "variables": [], "databases": [], "tools": []},
    "suggested_limitations": [],
}


class FileFastPathTests(UploadTestCase):
    """The same file bytes as a stored paper's: answered from the store, without extraction or OCR."""

    def test_new_paper_stores_what_the_fast_path_needs(self):
        first = self.upload()
        data = self.data(first["paper_id"])
        self.assertEqual(data["extractor"], EXTRACTOR_VERSION)
        self.assertEqual(data["file_sha256s"], [hashlib.sha256(PAPER.encode("utf-8")).hexdigest()])
        self.assertEqual((data["ocr_used"], data["ocr_skipped_pages"]), (False, []))
        self.assertIn("hidden_text_removed", data)
        self.assertIn("stored_at", data)
        self.assertEqual(first["ocr_skipped_pages"], [])

    def test_same_file_again_is_not_extracted(self):
        first = self.upload()
        with self.extracted() as extract:
            second = self.upload(name="renamed.txt")
        extract.assert_not_called()
        self.assertTrue(second["duplicate"])
        self.assertEqual((second["paper_id"], second["filename"]), (first["paper_id"], "renamed.txt"))
        self.assertEqual(second["chunk_count"], first["chunk_count"])
        self.assertEqual(second["normalization"], first["normalization"])
        self.assertEqual(self.embed.call_count, 1)

    def test_another_extraction_version_is_extracted_again(self):
        first = self.upload()
        self.rewrite(first["paper_id"], extractor="an older extraction")
        with self.extracted() as extract:
            second = self.upload()
        self.assertEqual(extract.call_count, 1)
        self.assertEqual(second["paper_id"], first["paper_id"])  # Still the same paper, by its text.

    def test_old_paper_with_the_same_text_is_stamped(self):
        first = self.upload()
        paper_id = first["paper_id"]
        file_hash = hashlib.sha256(PAPER.encode("utf-8")).hexdigest()
        # Stored before versions were recorded: no version, no file hashes, no extraction fields.
        self.rewrite(paper_id, extractor=None, file_sha256s=None, ocr_used=None, hidden_text_removed=None,
                     ocr_skipped_pages=None, stored_at=None)
        with self.extracted() as extract:
            self.upload()  # Extracted once more: its text is exactly the stored one.
        self.assertEqual(extract.call_count, 1)
        data = self.data(paper_id)
        self.assertEqual(data["extractor"], EXTRACTOR_VERSION)
        self.assertEqual(data["file_sha256s"], [file_hash])
        self.assertEqual((data["ocr_used"], data["ocr_skipped_pages"]), (False, []))
        self.assertIn("stored_at", data)  # Kept from the file's time, so the oldest copy stays first.
        with self.extracted() as extract:
            third = self.upload()
        extract.assert_not_called()  # Now the fast path finds it.
        self.assertEqual(third["paper_id"], paper_id)
        self.assertEqual(self.embed.call_count, 1)

    def test_an_older_version_is_stamped_and_its_old_files_dropped(self):
        first = self.upload()
        self.rewrite(first["paper_id"], extractor="an older extraction", file_sha256s=["0" * 64])
        self.upload()
        self.assertEqual(self.data(first["paper_id"])["file_sha256s"],
                         [hashlib.sha256(PAPER.encode("utf-8")).hexdigest()])

    def test_not_stamped_when_the_text_is_not_exactly_the_same(self):
        first = self.upload()
        # Another file of the same paper, with other whitespace: the same paper by its fingerprint,
        # but not the same text, so its file is not recorded.
        respaced = PAPER.replace(" were ", "   were ")
        self.assertTrue(self.upload(respaced, "copy.txt")["duplicate"])
        self.assertEqual(len(self.data(first["paper_id"])["file_sha256s"]), 1)
        with self.extracted() as extract:
            self.upload(respaced, "copy.txt")
        self.assertEqual(extract.call_count, 1)

    def test_a_new_file_with_the_same_text_is_recorded(self):
        first = self.upload()
        # The same text from another file (another name, another format later): recorded too.
        self.rewrite(first["paper_id"], file_sha256s=["f" * 64])
        self.upload()
        self.assertEqual(self.data(first["paper_id"])["file_sha256s"],
                         ["f" * 64, hashlib.sha256(PAPER.encode("utf-8")).hexdigest()])

    def test_fast_path_with_a_broken_collection_indexes_again(self):
        first = self.upload()
        self.chroma.delete_collection(f"paper_{first['paper_id']}")
        with self.extracted() as extract:
            second = self.upload()
        self.assertEqual(extract.call_count, 1)
        self.assertEqual(second["paper_id"], first["paper_id"])
        self.assertEqual(self.collection(first["paper_id"]).count(), first["chunk_count"])
        self.assertEqual(self.embed.call_count, 2)

    def test_a_duplicate_carries_the_extraction_warnings(self):
        scanned = ExtractionResult(PAPER, ocr_used=True, pages=[PAPER], hidden_removed={"count": 1, "preview": ["x"]},
                                   ocr_skipped_pages=(14, 15))
        with patch.object(main, "extract_text", return_value=scanned):
            first = self.upload()
            by_text = self.upload(name="other-file.txt")  # Another file: extracted, same text.
        for response in (first, by_text):
            self.assertEqual((response["ocr_used"], response["ocr_skipped_pages"]), (True, [14, 15]))
            self.assertEqual(response["hidden_text_removed"], {"count": 1, "preview": ["x"]})
        with self.extracted() as extract:
            by_file = self.upload()  # The fast path: from the store.
        extract.assert_not_called()
        self.assertTrue(by_file["duplicate"])
        self.assertEqual((by_file["ocr_used"], by_file["ocr_skipped_pages"]), (True, [14, 15]))
        self.assertEqual(by_file["hidden_text_removed"], {"count": 1, "preview": ["x"]})

    def test_summarize_uses_the_stored_text(self):
        scanned = ExtractionResult(PAPER, ocr_used=True, pages=[PAPER], ocr_skipped_pages=(14,))
        with patch.object(main, "extract_text", return_value=scanned):
            self.upload()
        with self.extracted() as extract, patch.object(main, "summarize_paper", return_value=SUMMARY) as summarize:
            response = self.client.post("/summarize", files={"file": ("paper.txt", PAPER.encode("utf-8"), "text/plain")})
        self.assertEqual(response.status_code, 200, response.text)
        extract.assert_not_called()  # No second extraction, so no second OCR.
        summarize.assert_called_once_with(PAPER, ocr_used=True)
        body = response.json()
        self.assertEqual((body["ocr_used"], body["ocr_skipped_pages"]), (True, [14]))
        self.assertIn("ocr_warning", body)

    def test_summarize_an_unknown_file_extracts_it(self):
        with self.extracted() as extract, patch.object(main, "summarize_paper", return_value=SUMMARY):
            response = self.client.post("/summarize", files={"file": ("p.txt", b"Some paper text.", "text/plain")})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(extract.call_count, 1)
        self.assertEqual(response.json()["ocr_skipped_pages"], [])

    def test_extract_returns_the_skipped_pages(self):
        scanned = ExtractionResult("Text.", ocr_used=True, pages=["Text."], ocr_skipped_pages=(2,))
        with patch.object(main, "extract_text", return_value=scanned):
            body = self.client.post("/extract", files={"file": ("p.txt", b"x", "text/plain")}).json()
        self.assertEqual(body["ocr_skipped_pages"], [2])


if __name__ == "__main__":
    unittest.main()
