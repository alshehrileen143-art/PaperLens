# PaperLens

PaperLens: a grounded research paper assistant with verified citations

PaperLens reads one research paper at a time (PDF, Word, plain text, or a scanned image) and answers
questions about it using only the paper's own text. Every answer cites its sources with a short quote
that is checked against the paper word for word and number by number, so unsupported evidence is never
shown as verified. It also produces a structured summary, and works by voice and in Arabic or English.

## Features

- **Any paper format:** PDF, DOCX, TXT, and PNG/JPG images. Scanned pages are read with OCR (PaddleOCR).
  Text a reader cannot see (white or tiny text, invisible characters) is removed during extraction.
- **Structured summary:** objective, methodology, key findings, the authors' own limitations, the model's
  suggested limitations (clearly labelled as inference), use cases, AI/ML tasks, and data and tools.
  Fields the paper does not state are left empty instead of guessed.
- **Questions with RAG:** short papers are sent whole; longer ones use hybrid retrieval. Answers are at most
  three sentences, with numbered citations `[1]`–`[3]`, the page of each quote and the nearest heading.
- **Voice questions:** record a question in the browser; it is transcribed and answered like a typed one.
- **Answers read aloud:** a 🔊 button reads the short answer (on demand, cached per answer).
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
  /ask_voice ─▶ audio_validation.py ─▶ stt.py (whisper-1) ─▶ /ask
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
.\venv\Scripts\python.exe -m unittest tests.test_ask_voice tests.test_app tests.test_ui_text tests.test_quotes tests.test_summarize tests.test_tts tests.test_number_check -v
```

## Endpoints

| Endpoint | Input | Returns |
|---|---|---|
| `GET /health` | — | `{"status": "ok"}` |
| `POST /extract` | file | extracted text, OCR flag, hidden text removed |
| `POST /summarize` | file | structured summary |
| `POST /upload` | file | `paper_id` (indexes the paper for questions) |
| `POST /ask` | `{"paper_id", "question"}` | answer, details, numbered sources with page, heading and verified quote, warnings |
| `POST /ask_voice` | form: `paper_id` + audio (mp3, wav, m4a, webm; ≤ 5 MB, ≤ 60 s) | `/ask` response + `transcript` |
| `POST /speak` | `{"text"}` (≤ 1,000 characters after cleaning) | `audio/mpeg` |

Uploads are limited to 20 MB, PDFs to 100 pages, and OCR to 10 pages.

## Results

Measured on a pediatric vitamin D meta-analysis with 25 fixed questions (answerable, "no consensus",
unanswerable, figure-only, and out of scope; 9 in Arabic), graded by an LLM judge with human review.

| Evaluation | Result |
|---|---|
| `/ask` end to end (whole paper, current settings) | **25/25** |
| Retrieval, hybrid search | R@10 = 1.00 |
| Needle in a haystack (paper hidden among unrelated papers): whole paper vs hybrid RAG | 8k: 24 vs 18 · 16k: 23 vs 18 · 32k: 22 vs 19 · 64k: 22 vs 17 · 100k: 20 vs 18 (of 25) |
| Prompt injection (hidden instructions in uploaded papers) | model followed them in **0/320** runs; hidden text reached the sources in 80/80 runs without the filter, **0** with it |
| Voice questions (real recordings, Arabic and English) | **3/3** transcribed and answered correctly |
| Citation quotes verified as verbatim | **17/18**; the one rejected was a quote the model invented |

The number check on quotes caught real errors that the judge accepted, such as an answer of "20 studies"
cited with a quote about "24,600 children". The whole project's evaluation cost about **$1.4** in API calls.

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
- **Latency** measurements were affected by the network and are not reliable.

The full list, with the evidence for each, is in `eval/clinic.md`.

## Future work

- A larger evaluation set: more papers (longer, with several tables) and new, unseen questions.
- A structured evaluation for the summary, like the one for `/ask`.
- Sentence-level evidence checks, beyond numbers.
- A library of several papers, with questions across them.
- Update `eval/run_needle.py` and `eval/run_rag_vs_full.py` to grade the answer with its details.
- Bind Streamlit to `localhost` by default.
- Detect duplicate uploads by a hash of the paper's text and reuse its `paper_id` (today every upload
  creates a new copy in `paper_store/` and ChromaDB).
- Clean PDF extraction artifacts in the displayed text (a space before ligatures, a colon as decimal
  point, real hyphens joined at line ends); today they are only forgiven when quotes are compared.
