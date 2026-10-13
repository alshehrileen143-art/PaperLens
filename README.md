# PaperLens

PaperLens: a grounded research paper assistant with verified citations

PaperLens reads one research paper at a time (PDF, Word, plain text, or a scanned image) and answers
questions about it using only the paper's own text. Every answer cites its sources with a short quote
that is checked against the paper word for word and number by number, so unsupported evidence is never
shown as verified. It also produces a structured summary, and works by voice and in Arabic or English.

## Features

- **Upload first:** before a paper is open, the page shows a drag-and-drop upload area and no message box,
  so nothing suggests a question can be asked yet.
- **Any paper format:** PDF, DOCX, TXT, and PNG/JPG images. Scanned pages are read with OCR (RapidOCR, see [OCR](#ocr)).
  Text a reader cannot see (white or tiny text, invisible characters) is removed during extraction.
- **Structured summary:** objective, methodology, key findings, the authors' own limitations, the model's
  suggested limitations (clearly labelled as inference), use cases, AI/ML tasks, and data and tools.
  Fields the paper does not state are left empty instead of guessed.
- **Questions with RAG:** short papers are sent whole; longer ones use hybrid retrieval. Answers are at most
  three sentences, with numbered citations `[1]`–`[3]`, the page of each quote and the nearest heading.
- **Voice questions:** record a question in the browser; its transcript appears in the message box to review
  and edit, and is answered like a typed one when sent.
- **Answers read aloud:** a 🔊 button reads the short answer (on demand, cached per answer).
- **No duplicate copies:** uploading a paper that is already stored opens it (same `paper_id`, no new
  embeddings). Papers are matched by a SHA-256 of their extracted text, with only whitespace collapsed:
  the original files are not kept, and two PDFs of the same paper give the same text. A stored copy whose
  ChromaDB collection is missing or incomplete is indexed again under its own id.
- **Arabic / English interface:** one switch changes every label and the page direction (right to left or
  left to right); paper text keeps its own direction.

## Architecture

```
Streamlit chat UI (app.py, ui_text.py)  ──HTTP──▶  FastAPI (main.py)
                                                     │
  upload ─▶ extract.py (pypdf, python-docx, ocr.py) ─▶ hidden_text.py ─▶ normalize.py / sections.py
          ─▶ rag.py: chunks + embeddings in ChromaDB, full text in paper_store/
  /ask   ─▶ rag.py: paper ≤ 60k tokens → whole paper as numbered sections;
                    longer → hybrid search (BM25 + embeddings, RRF) with query rewriting
          ─▶ answer (gpt-4o-mini) ─▶ quote check (verbatim) ─▶ number_check.py ─▶ grounding.py ─▶ output_check.py
  /summarize ─▶ summarize.py (gpt-4o-mini, temperature 0)
  /transcribe ─▶ audio_validation.py ─▶ stt.py (whisper-1) ─▶ text to review (the app then calls /ask)
  /ask_voice  ─▶ the same checks and transcription ─▶ /ask, in one step
  /speak     ─▶ tts.py (gpt-4o-mini-tts)
```

All limits and settings are in `config.py`. The evaluation scripts and results are in `eval/`; the full
record of experiments and decisions is `eval/clinic.md`.

## Setup

Requires **Python 3.12** and an OpenAI API key.

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env      # then put your key in .env
```

`.env` is ignored by git; never commit it.

## Running

Start the API, then the interface, in two terminals:

```powershell
.\venv\Scripts\python.exe -m uvicorn main:app --port 8000
.\venv\Scripts\python.exe -m streamlit run app.py
```

Open http://localhost:8501. The interface talks to `http://127.0.0.1:8000` (override with the `API_URL`
environment variable).

## Tests

No API calls (OpenAI and HTTP are mocked):

```powershell
.\venv\Scripts\python.exe -m unittest tests.test_ask_voice tests.test_app tests.test_ui_text tests.test_quotes tests.test_summarize tests.test_tts tests.test_number_check tests.test_upload tests.test_warnings tests.test_ocr tests.test_reading_order tests.test_injection_samples tests.test_hidden_text tests.test_docx_text -v
```

## Endpoints

| Endpoint | Input | Returns |
|---|---|---|
| `GET /health` | — | `{"status": "ok"}` |
| `POST /extract` | file | extracted text, OCR flag, scanned pages not read, hidden text removed |
| `POST /summarize` | file | structured summary (from the stored text if the same file was uploaded) |
| `POST /upload` | file | `paper_id` (indexes the paper for questions; `duplicate: true` when it was already stored) |
| `POST /ask` | `{"paper_id", "question"}` | answer, details, numbered sources with page, heading and verified quote, warnings (English text, and the same as data in `*_items` for the interface's language) |
| `POST /transcribe` | audio (mp3, wav, m4a, webm; ≤ 5 MB, ≤ 60 s) | `transcript` only, to review before `/ask` (used by the app) |
| `POST /ask_voice` | form: `paper_id` + audio (same limits) | `/ask` response + `transcript`, without review |
| `POST /speak` | `{"text"}` (≤ 1,000 characters after cleaning) | `audio/mpeg` |

Uploads are limited to 20 MB, PDFs to 100 pages, and OCR to 13 scanned pages per paper.

## OCR

Scanned pages and images are read by [RapidOCR](https://github.com/RapidAI/RapidOCR) on onnxruntime, with
the PP-OCRv6 small detection and recognition models (bundled in the package; nothing is downloaded at
run time). These are the same models PaddleOCR ran here before; onnxruntime runs them about **4 times
faster** on CPU (about 10 s a page against 40 s) with the same accuracy: 100% of the numbers on clean pages
for both, and on simulated scans RapidOCR missed 3 numbers where PaddleOCR missed 6 (the same 3 and 3 more).
Details and every misread number: [eval/ocr_benchmark.md](eval/ocr_benchmark.md); the script is
`eval/ocr_benchmark.py`.

- **Column reading order** (`reading_order.py`): the engine lists lines from top to bottom across the page,
  so a two-column page came out with its columns interleaved line by line. The lines are put back in
  reading order from their boxes: tilt corrected, the page cut at full-width gaps, columns found from the
  gutter, wide elements (titles, captions) kept in place, and tables read row by row. On the two papers
  measured, sentences found unbroken rose from 11-16% to 60-64%, and random quotes found from 5-8 of 30 to
  23-26 (the rest are reference-list entries and a broken ligature in the text layer).
- **200 DPI, full size:** pages are rendered at 200 DPI and not shrunk before detection, so small digits keep
  their pixels. The orientation model is off (pages are upright).
- **13 scanned pages per paper:** set for the app's first 300 s wait ((300 − 80) / 15.8 s, the slowest page
  measured). A page took up to 22.4 s on a busy CPU, so the app now waits up to 15 minutes for an upload
  or a summary rather than reading fewer pages: a whole paper after a wait is better than a paper cut
  short. Text pages of a mixed PDF are read from the text layer and do not count.
- **Pages not read** keep their place: the page's text says `[Page 15 was not read: OCR is limited to 13
  scanned pages per paper.]`, the API returns their numbers (`ocr_skipped_pages`), and the app warns which
  pages were not read.
- **No second OCR:** the SHA-256 of an uploaded file is stored with the paper. The same file again (to
  `/upload` or `/summarize`) is answered from the stored text, without extraction, as long as it was made by
  the current extraction (`EXTRACTOR_VERSION` in `extract.py`).

## Sample data

- **`samples/1-s2.0-S0021755720302072-main.pdf`:** V. I. Fiamenghi and E. D. de Mello, "Vitamin D deficiency
  in children and adolescents with obesity: a meta-analysis", *Jornal de Pediatria* 2021;97(3):273–279,
  doi:[10.1016/j.jped.2020.08.006](https://doi.org/10.1016/j.jped.2020.08.006). © 2020 Sociedade Brasileira de
  Pediatria, published by Elsevier Editora Ltda., open access under
  [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/). Included unmodified, for
  non-commercial evaluation.
- **Prompt-injection test files** (`samples/injection/`, used by `eval/run_markers.py`) are that paper with a
  hidden instruction added. The licence does not allow sharing modified versions, so they are not in the
  repository: `eval/make_injection_samples.py` makes them locally from the original (no API calls). The
  measurement above used hand-made files (Word, 2026-10-04); files made by the script differ in layout,
  so a new run is not directly comparable with it.
- **Voice recordings** (`samples/voice/`) are kept locally; the tests build their audio in code.

## Results

Measured on a pediatric vitamin D meta-analysis with 25 fixed questions (answerable, "no consensus",
unanswerable, figure-only, and out of scope; 9 in Arabic), graded by an LLM judge with human review.

| Evaluation | Result |
|---|---|
| `/ask` end to end (whole paper, current settings) | **25/25** |
| Retrieval, hybrid search | R@10 = 1.00 |
| Needle in a haystack (paper hidden among unrelated papers): whole paper vs hybrid RAG | 8k: 24 vs 18 · 16k: 23 vs 18 · 32k: 22 vs 19 · 64k: 22 vs 17 · 100k: 20 vs 18 (of 25) |
| Prompt injection (hidden instructions in uploaded papers) | model followed them in **0/320** runs, judged by the marker `CANARY-7431` in answers and summaries; see the correction below the table |
| Voice questions (real recordings, Arabic and English; kept locally, not in the repository) | **3/3** transcribed and answered correctly |
| Citation quotes verified as verbatim | **17/18**; the one rejected was a quote the model invented |

The number check on quotes caught real errors that the judge accepted, such as an answer of "20 studies"
cited with a quote about "24,600 children". The whole project's evaluation cost about **$1.4** in API calls.

**Correction (2026-10-13) to the prompt-injection result.** The 2026-10-04 run said hidden text reached the
sources in 80/80 runs without the filter and **0 with it**. That 0 counted only the marker `CANARY-7431`. In
the two Arabic PDFs, the filter removed the marker but **not the Arabic instruction**: it reached the sources
in all 20 `ask` runs on those files with the filter on (20 of the 80 filter-on `ask` runs). The English and
DOCX files were fully removed. The cause: those PDFs store Arabic in visual order, and the filter compared
it with the text in reading order. On the model's side, "0/320 followed" holds as measured by the marker,
but in the 40 filter-on runs on the Arabic PDFs the marker was gone while the instruction stayed, so the
marker could not show compliance there; none of their answers or summaries mentions a code or "رمز".
The filter now matches Arabic stored in visual order as well, and removes the whole instruction from
those files ([check](eval/hidden_filter_check_20261013_100004.md)); the model-side measurement has not been run again.

## Known limitations

- **Small evaluation:** 25 questions on one short paper, and the prompts were tuned on those same questions,
  so the reported accuracy is optimistic.
- **Judge bias:** the LLM judge is mostly too strict; human review corrected it, but unreviewed verdicts may
  hold errors.
- **Non-determinism:** answers and inferred summary fields can change between runs even at temperature 0.
- **Summary:** the authors' limitations can come out incomplete (5 of 6), suggested limitations sometimes
  repeat stated ones or come out empty, and measured variables can be incomplete.
- **Citation checks:** quotes are checked per source, not per sentence; the number check does not see
  numbers written as Arabic words or claims without numbers, and a rare case of three numbers split
  across a group of citations is not caught.
- **Scope:** one paper at a time, OCR in English only, and fragmented M4A audio is rejected.
- **DOCX tables:** the text of DOCX tables is not extracted yet (paragraphs only); the next change.
  Links, tracked insertions, content controls and reference-manager citations are read
  ([check](eval/docx_text_check_20261013_101511.md)).
- **Duplicates:** two uploads of the same paper at the same moment are both indexed, and a change to text
  extraction can make a stored paper look new (one more copy, never the wrong paper).
- **Latency** measurements were affected by the network and are not reliable.

The full list, with the evidence for each, is in `eval/clinic.md`.

## Future work

- A larger evaluation set: more papers (longer, with several tables) and new, unseen questions.
- A structured evaluation for the summary, like the one for `/ask`.
- Sentence-level evidence checks, beyond numbers.
- A library of several papers, with questions across them.
- Update `eval/run_needle.py` and `eval/run_rag_vs_full.py` to grade the answer with its details.
- Bind Streamlit to `localhost` by default.
- When a verbatim quote is credited to the wrong source (it appears word for word in another source
  that was sent), credit it to the right one; today it is rejected as not verbatim.
- Clean PDF extraction artifacts in the displayed text (a space before ligatures, a colon as decimal
  point, real hyphens joined at line ends); today they are only forgiven when quotes are compared.
