"""Tests for /upload's duplicate check (one stored copy per paper). No OpenAI calls:
embeddings are mocked, and the paper store and Chroma are temporary folders.

Run: .\\venv\\Scripts\\python.exe -m unittest tests.test_upload -v
"""

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
from extract import ExtractionResult  # noqa: E402
from llm import LLMError  # noqa: E402

PAPER = ("Vitamin D and asthma in children.\n\nMethods\nWe searched PubMed and Embase for studies "
         "published before 2020. A total of 24,600 children were included.\n\nResults\nThe pooled "
         "relative risk was 1.41 (95% CI 1.26-1.59).\n\nDiscussion\n"
         + " ".join(f"Study {n} measured serum 25(OH)D in {100 + n} children." for n in range(60)) + "\n")
OTHER = PAPER.replace("1.41", "1.52")


def fake_embeddings(texts: list[str]) -> list[list[float]]:
    return [[float(index + 1), 1.0, 0.5] for index, _ in enumerate(texts)]


class UploadDuplicateTests(unittest.TestCase):
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

    def collection(self, paper_id: str):
        return self.chroma.get_collection(f"paper_{paper_id}")

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
        path = self.store / f"{first['paper_id']}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        del data["text_sha256"]
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
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


if __name__ == "__main__":
    unittest.main()
