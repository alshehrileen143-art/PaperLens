"""All size and processing limits in one place."""

from pathlib import Path

MB = 1024 * 1024

# --- Uploads ---
MAX_UPLOAD_BYTES = 20 * MB
UPLOAD_CHUNK_BYTES = 1 * MB  # Uploads are copied to disk in chunks of this size.
# Whole-request limit checked by the middleware before the body is received.
# A multipart request carries boundaries and part headers on top of the file,
# so allow a small margin; the exact file limit is enforced per upload.
MAX_REQUEST_BYTES = MAX_UPLOAD_BYTES + 1 * MB

# --- Images (checked from the file header, before decoding) ---
MAX_IMAGE_SIDE_PX = 10_000  # Maximum width and maximum height.

# --- DOCX (a zip archive; checked before python-docx opens it) ---
MAX_DOCX_UNCOMPRESSED_BYTES = 100 * MB
MAX_DOCX_ENTRIES = 1_000

# --- PDF ---
MAX_PDF_PAGES = 100

# --- Hidden text (see hidden_text.py) ---
# Remove text a reader cannot see: white / near-white or tiny (< 2 pt) text,
# invisible render mode and off-page text in PDFs; hidden, white or tiny runs
# in DOCX; and invisible Unicode characters (zero-width, bidi controls, tags)
# in every format. What was removed is reported as hidden_text_removed.
HIDDEN_TEXT_FILTER = True
HIDDEN_MIN_FONT_SIZE = 2.0  # Points.
HIDDEN_NEAR_WHITE = 0.94  # A colour whose red, green and blue are all >= this (0-1) counts as white.

# --- OCR ---
# 200 DPI reads normal body text accurately and is noticeably faster than 300 on CPU.
OCR_DPI = 200
# Scanned pages read per paper; pages beyond it are marked as not read. RapidOCR takes about 10 s a
# page on this CPU, 15.8 s at worst on an idle CPU and 22.4 s on a busy one (eval/ocr_benchmark.md and
# eval/ocr_benchmark_20261012_192153.md). 13 was set for a 300 s wait in the app; the app now waits
# 900 s (app.LONG_TIMEOUT_SECONDS) instead of reading fewer pages: a whole paper after a wait is
# better than a paper cut short. No time limit is put on OCR itself.
MAX_OCR_PAGES = 13
# A real text page has thousands of characters; scanned pages usually yield
# nothing, or only a stray header/page number. Pages below this use OCR.
MIN_CHARS_PER_PAGE = 100

# --- RAG (question answering over an uploaded paper) ---
CHROMA_DIR = str(Path(__file__).parent / "chroma_db")
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_BATCH_SIZE = 100  # Chunks sent per embeddings request.
ANSWER_MODEL = "gpt-4o-mini"
# Per-request timeout for every OpenAI call. The library default is 10 minutes;
# one stalled connection once took 651 s. A timeout raises APITimeoutError,
# which the SDK and the eval scripts retry.
OPENAI_TIMEOUT_SECONDS = 60
# 0 makes answers repeatable, so evaluation runs can be compared. Read at call
# time, so tests (e.g. prompt-injection tests) can raise it.
ANSWER_TEMPERATURE = 0.0
# Tokenizer used by gpt-4o / gpt-4o-mini, for counting a paper's tokens.
TOKENIZER_ENCODING = "o200k_base"
# ~1,200 characters is ~300 tokens: a paragraph or two, small enough to stay on
# one topic, large enough to keep a finding together with its numbers.
CHUNK_SIZE_CHARS = 1_200
CHUNK_OVERLAP_CHARS = 200
RETRIEVAL_TOP_K = 5
# When a whole paper is sent instead of retrieved chunks, it is split into
# consecutive, non-overlapping numbered sections of about this size, so the
# answer can still cite where it came from. ~4,000 characters is ~1,000 tokens.
FULL_CONTEXT_SECTION_CHARS = 4_000

# Adaptive /ask: papers up to this many tokens are sent whole (as numbered
# sections); longer papers use hybrid RAG. Preliminary value from the
# needle-in-a-haystack run (eval/results/needle_20261003_190613.json): full
# context beat hybrid at every length, but its accuracy dropped at 100k.
# Known limitation: latency in that run was distorted by network problems
# (16k was slower than 64k), so time was not measured reliably; re-measure
# before relying on the 5-second budget.
FULL_CONTEXT_MAX_TOKENS = 60_000
# Where the text of each uploaded paper is kept, so it can be sent whole.
PAPER_STORE_DIR = str(Path(__file__).parent / "paper_store")

# Query rewriting (RAG mode only): before searching, the model turns the
# question into an English search query using the paper's vocabulary
# (translating Arabic questions). The original question and the rewrite are
# both searched and merged with RRF. The answer is still written in the
# language of the original question. (Measured with eval/run_retrieval.py,
# run_ask.py and run_needle.py.)
QUERY_REWRITE = True
QUERY_REWRITE_MODEL = "gpt-4o-mini"

# --- Final retrieval settings (frozen after the evaluation in eval/clinic.md) ---
# Section metadata (see sections.py): parts never cross a section heading, each
# is tagged with its section (Abstract, Methods, Results, ...), and every source
# in the prompt starts with a "[Section: ...]" line. Set separately per mode:
# - full mode: on. Same accuracy (25/25) with 34% fewer tokens, since the
#   reference list is left out (ask_20261003_231256).
# - RAG mode: off. Section-aware chunking shrank the context and lost table and
#   discussion chunks (hybrid 23/25 vs 25/25; 64k haystack 18/25 vs 21/25).
#   RAG papers must be re-uploaded for their chunks to be tagged.
FULL_CONTEXT_SECTIONS = True
RAG_SECTIONS = False
# RAG alternative that keeps the usual chunk boundaries: tag each chunk with the
# section it starts in, only so the reference list can be excluded from search
# (no "[Section: ...]" headers). Needs a fresh upload. Ignored if RAG_SECTIONS is on.
# On: same accuracy as without it (25/25 and 21/25 after human review) and
# removes a known error source (answers taken from the reference list).
RAG_TAG_SECTIONS = True
# Where sections are on: leave the reference list out of the context / search.
EXCLUDE_REFERENCES = True

# Small-to-big retrieval (RAG mode only): search small units (one or two
# sentences; each table row on its own) and send the big chunks they belong
# to, RETRIEVAL_TOP_K distinct big chunks as before, so the context does not
# shrink. Small units are indexed only while this is on (at upload, or on first
# use for papers uploaded without them).
# Off: worse in all four comparisons, with or without reference exclusion
# (vitamin D 24 and 24 vs 25/25; 64k haystack 19 and 19 vs 21/25). Small units
# scattered retrieval and lost paragraph-level answers such as q8.
SMALL_TO_BIG = False
SMALL_UNIT_MIN_CHARS = 120  # A shorter sentence is joined with the next one.
SMALL_UNIT_MAX_CHARS = 400  # Longer run-on text (e.g. flattened tables) is split.

# Prompt delimiters: paper text inside <paper> / <source id="..."> tags, the
# question inside <question>, and a line saying the paper is data, not
# instructions. Keep on; turning it off exists only to measure what it adds
# (eval/run_markers.py).
PROMPT_DELIMITERS = True
# Temperature for /summarize. 0 makes summaries repeatable, so prompt changes can be
# compared run to run (until 2026-10-05 it was None: not sent, the API default).
# None still means "not sent"; eval/run_markers.py sets it to compare 0.7 and 0.
SUMMARY_TEMPERATURE = 0

# Output check on answers and summaries (see output_check.py): a warning is
# added when the output contains a link that is not in the paper's text, or
# text matching one of these regular expressions (case-insensitive).
MARKER_PATTERNS = [r"CANARY-\d+"]

# Grounding check after generation (see grounding.py).
# An answer citing more sources than this is treated as suspect: correct
# answers in tests cited at most 5; a guessed one cited 46.
MAX_CITED_SOURCES = 8
# Sources shown with an answer (numbered [1]..[3] in its text). The grounding check
# above still counts every source the model asked to cite, before this limit.
MAX_ANSWER_SOURCES = 3
# Sources on each side of a cited one that also count as evidence: the model
# often cites the section next to the one holding the fact.
GROUNDING_NEIGHBORS = 1
# What to do with an answer that fails the check: "warn" (keep it, add
# grounding_warning) or "reject" (replace it with the not-found answer).
GROUNDING_ACTION = "warn"
# expand_context adds at most one neighbouring chunk to each of the first N results.
EXPAND_TOP_N = 3

# Search method: "dense" (embeddings only) or "hybrid" (embeddings + BM25 keyword
# search, merged with Reciprocal Rank Fusion).
SEARCH_METHOD = "hybrid"
RRF_K = 60  # Standard RRF constant: score = sum(1 / (RRF_K + rank)).

# Text normalization rules, applied by normalize_text() to chunks and questions
# alike before embedding and BM25. Each rule can be switched off on its own.
NORMALIZE_NFKC = True  # Unicode NFKC: splits ligatures (ﬁ -> fi), I² -> I2, etc.
NORMALIZE_DASHES = True  # Broken PDF dashes: "0- - -18" -> "0–18".
NORMALIZE_DEHYPHENATE = True  # Line-end hyphenation: "con-\ntrols" -> "controls".
MAX_QUESTION_CHARS = 1_000

# --- Voice questions (/ask_voice) ---
# The audio is transcribed, then answered exactly like a /ask question.
STT_MODEL = "whisper-1"
MAX_AUDIO_BYTES = 5 * MB
# Measured from the audio stream itself (frames, samples, block timestamps),
# not from header fields a file could fake; see audio_validation.py.
MAX_AUDIO_SECONDS = 60

# --- Spoken answers (/speak) ---
# The app's 🔊 button reads the short /ask answer aloud (not its details or sources).
TTS_MODEL = "gpt-4o-mini-tts"
TTS_VOICE = "alloy"
TTS_MAX_CHARS = 1000  # Checked after citation markers and Markdown are removed.
TTS_INSTRUCTIONS = "Speak clearly and calmly in a neutral academic tone. Read numbers and statistics naturally."

# --- Summarization ---
# gpt-4o-mini has a 128k-token context; ~4 characters per token, so 100k
# characters (~25k tokens) leaves plenty of room while keeping costs low.
MAX_INPUT_CHARS = 100_000
