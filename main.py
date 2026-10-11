"""FastAPI server for extracting, summarizing, and asking questions about research papers."""

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

from audio_validation import AUDIO_EXTENSIONS, validate_saved_audio
from config import MAX_AUDIO_BYTES, MAX_QUESTION_CHARS, MAX_UPLOAD_BYTES, MB, TTS_MAX_CHARS
from extract import SUPPORTED_EXTENSIONS, ExtractionResult, extract_text
from middleware import RequestSizeLimitMiddleware
from llm import LLMError, MissingAPIKeyError
from ocr import OCRError
from rag import PaperNotFoundError, answer_question, find_or_index_paper, get_paper_collection
from stt import transcribe
from tts import clean_for_speech, synthesize
from summarize import summarize_paper
from output_check import check_output, summary_text
from upload_validation import UploadRejected, save_upload_limited, validate_saved_upload

app = FastAPI(title="PaperLens")
app.add_middleware(RequestSizeLimitMiddleware)


class UseCase(BaseModel):
    audience: str
    application: str


class AIMLTask(BaseModel):
    task_type: str
    input: str
    output: str


class DataAndTools(BaseModel):
    datasets: list[str]
    variables: list[str]
    databases: list[str]
    tools: list[str]


class Summary(BaseModel):
    # From the paper only; empty ("" or []) when the paper does not state them.
    objective: str
    methodology: str
    key_findings: list[str]
    limitations: list[str]  # Only limitations the authors state themselves.
    use_cases: list[UseCase]
    technical_use_cases: list[str]
    ai_ml_tasks: list[AIMLTask]
    data_and_tools: DataAndTools
    # The model's own inference, not stated by the paper; shown apart from "limitations".
    suggested_limitations: list[str] = []


class HiddenTextRemoved(BaseModel):
    count: int  # Hidden text pieces plus invisible characters removed during extraction.
    preview: list[str]  # Up to 5 short excerpts of what was removed.


class GroundingItems(BaseModel):
    numbers: list[str]  # Not found where the model could have read them.
    names: list[str]
    too_many_citations: bool  # Cites more than max_cited_sources sources.
    max_cited_sources: int


class MismatchItems(BaseModel):
    numbers: list[str]  # In the context sent, but not in or next to the cited sources.
    names: list[str]


class OutputItems(BaseModel):
    links: list[str]  # Not in the paper.
    markers: list[str]  # Text matching config.MARKER_PATTERNS.


class SummarizeResponse(BaseModel):
    filename: str
    ocr_used: bool
    ocr_warning: str | None = None
    hidden_text_removed: HiddenTextRemoved | None = None
    # Set when the summary has a link that is not in the paper, or marker-like text.
    output_warning: str | None = None
    output_warning_items: OutputItems | None = None  # The same warning as data, for the interface's language.
    summary: Summary


class UploadResponse(BaseModel):
    paper_id: str
    filename: str
    ocr_used: bool
    chunk_count: int
    normalization: dict[str, int]  # How many times each normalization rule changed the text.
    hidden_text_removed: HiddenTextRemoved | None  # None when HIDDEN_TEXT_FILTER is off.
    # True when this paper's text was already stored: its paper_id is returned, without new embeddings.
    duplicate: bool = False


class AskRequest(BaseModel):
    paper_id: str
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)


class SpeakRequest(BaseModel):
    # Length is checked after cleaning (citation markers and Markdown removed), so a
    # too-long text gets 413 rather than a validation error.
    text: str


class Source(BaseModel):
    number: int  # The [n] that cites this source in the answer.
    chunk_index: int
    # Page of the quote when there is one (page_end == page); otherwise the chunk's page range.
    page: int | None
    page_end: int | None
    title: str | None  # Nearest heading before the quote (or chunk), e.g. "A.A.2 Models"; else the section.
    section: str | None  # e.g. "Methods", when section metadata is on.
    matched: str | None  # Small-to-big: the sentence or table row that found this chunk.
    score: float | None  # None for chunks added as context (see expanded_from).
    bm25_score: float | None  # Hybrid search only.
    fused_score: float | None  # Hybrid search only.
    expanded_from: int | None
    # A short sentence copied from text that supports the answer; checked to be
    # verbatim (after normalization), otherwise None.
    quote: str | None
    # Set when a verbatim quote was still rejected: "number_mismatch" means the sentences
    # citing this source have numbers (missing_numbers) that the quote (rejected_quote) lacks.
    quote_issue: str | None = None
    missing_numbers: list[str] = []
    rejected_quote: str | None = None
    text: str


class AskResponse(BaseModel):
    paper_id: str
    question: str
    answer: str  # At most three sentences, with [n] after the information each source supports.
    details: str | None  # Important details beyond the short answer; None if there are none.
    found: bool
    mode: str  # "full" (whole paper sent) or "rag" (retrieved chunks sent).
    # Set when numbers or names in the answer were not found in the sources
    # (RAG: anywhere in the context sent; full: in or next to the cited sections).
    grounding_warning: str | None
    # RAG only: numbers or names that are in the context but not in or next to
    # the cited sources, i.e. the citation probably points to the wrong chunk.
    citation_mismatch: str | None
    # Set when the answer has a link that is not in the paper, or marker-like text.
    output_warning: str | None
    # The three warnings above as data, so an interface can write them in its own language
    # (the strings above stay in English). None when there is no warning.
    grounding_warning_items: GroundingItems | None = None
    citation_mismatch_items: MismatchItems | None = None
    output_warning_items: OutputItems | None = None
    sources: list[Source]


class AskVoiceResponse(AskResponse):
    transcript: str  # What the audio was transcribed to; answered as the question.


OCR_WARNING = "Text was extracted via OCR. Please verify numbers against the original document."


@contextmanager
def _llm_http_errors():
    """Map OpenAI failures to HTTP errors: missing key is a server setup problem (500),
    anything else is an upstream failure (502)."""
    try:
        yield
    except MissingAPIKeyError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


def _require_text(result: ExtractionResult, filename: str | None) -> None:
    if not result.text.strip():
        raise HTTPException(status_code=400, detail=f"No text found in '{filename}', even after OCR.")


def _transcribe_upload(file: UploadFile) -> str:
    """Save the audio to a temp file, check it, transcribe it, and always clean up."""
    filename = file.filename or ""
    extension = Path(filename).suffix.lower()
    if extension not in AUDIO_EXTENSIONS:
        supported = ", ".join(AUDIO_EXTENSIONS)
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio type '{extension or filename}'. Supported: {supported}",
        )
    if file.size is not None and file.size > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail=f"File is too large. Maximum size is {MAX_AUDIO_BYTES // MB} MB.")

    fd, temp_path = tempfile.mkstemp(prefix="audio_", suffix=extension)
    try:
        try:
            with os.fdopen(fd, "wb") as temp_file:
                save_upload_limited(file.file, temp_file, max_bytes=MAX_AUDIO_BYTES)
            validate_saved_audio(Path(temp_path), extension)
        except UploadRejected as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
        with _llm_http_errors():
            return transcribe(Path(temp_path))
    finally:
        file.file.close()
        os.remove(temp_path)


def _extract_upload(file: UploadFile) -> ExtractionResult:
    """Save the upload to a temp file, extract its text, and always clean up."""
    filename = file.filename or ""
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(SUPPORTED_EXTENSIONS)
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{extension or filename}'. Supported: {supported}",
        )

    # Starlette records the size as it receives the upload; reject oversized
    # files before copying anything. The chunked copy below enforces it again.
    if file.size is not None and file.size > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File is too large. Maximum size is {MAX_UPLOAD_BYTES // MB} MB.")

    # The temp name is random; only the already-validated extension is taken from
    # the user's filename, so extract_text() can pick the right parser.
    fd, temp_path = tempfile.mkstemp(prefix="upload_", suffix=extension)
    try:
        try:
            with os.fdopen(fd, "wb") as temp_file:
                save_upload_limited(file.file, temp_file)
            validate_saved_upload(Path(temp_path), extension)
        except UploadRejected as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

        try:
            return extract_text(temp_path)
        except OCRError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except Exception as exc:
            # pypdf / python-docx / UTF-8 decoding raise many different errors
            # for corrupted or mislabeled files; report them all as bad input.
            raise HTTPException(
                status_code=400,
                detail=f"Could not read '{filename}'. The file may be corrupted or not a valid {extension} file.",
            ) from exc
    finally:
        file.file.close()
        os.remove(temp_path)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/extract")
def extract(file: UploadFile = File(...)) -> dict:
    result = _extract_upload(file)
    return {"filename": file.filename, "text": result.text, "ocr_used": result.ocr_used,
            "hidden_text_removed": result.hidden_removed}


# exclude_none drops ocr_warning entirely when OCR was not used.
@app.post("/summarize", response_model=SummarizeResponse, response_model_exclude_none=True)
def summarize(file: UploadFile = File(...)) -> dict:
    result = _extract_upload(file)
    _require_text(result, file.filename)
    with _llm_http_errors():
        summary = summarize_paper(result.text, ocr_used=result.ocr_used)

    output = check_output(summary_text(summary), result.text)
    response = {"filename": file.filename, "ocr_used": result.ocr_used, "summary": summary,
                "hidden_text_removed": result.hidden_removed,
                "output_warning": output.describe(), "output_warning_items": output.items()}
    if result.ocr_used:
        response["ocr_warning"] = OCR_WARNING
    return response


@app.post("/upload", response_model=UploadResponse)
def upload(file: UploadFile = File(...)) -> dict:
    """Index a paper for question answering and return its paper_id (the stored one, if the
    same text was uploaded before)."""
    result = _extract_upload(file)
    _require_text(result, file.filename)
    with _llm_http_errors():
        paper_id, chunk_count, normalization, duplicate = find_or_index_paper(result, file.filename or "")
    return {
        "paper_id": paper_id,
        "filename": file.filename,
        "ocr_used": result.ocr_used,
        "chunk_count": chunk_count,
        "normalization": normalization,
        "hidden_text_removed": result.hidden_removed,
        "duplicate": duplicate,
    }


def _paper_not_found() -> HTTPException:
    return HTTPException(status_code=404, detail="Paper not found. Upload it first with /upload.")


def _answer_response(paper_id: str, question: str) -> dict:
    """The /ask response for a question that is already stripped and non-empty."""
    try:
        with _llm_http_errors():
            result = answer_question(paper_id, question)
    except PaperNotFoundError as exc:
        raise _paper_not_found() from exc

    return {
        "paper_id": paper_id,
        "question": question,
        "answer": result.answer,
        "details": result.details,
        "found": result.found,
        "mode": result.mode,
        "grounding_warning": result.grounding_warning,
        "citation_mismatch": result.citation_mismatch,
        "output_warning": result.output_warning,
        "grounding_warning_items": result.grounding_warning_items,
        "citation_mismatch_items": result.citation_mismatch_items,
        "output_warning_items": result.output_warning_items,
        "sources": [
            {
                "number": number,
                "chunk_index": chunk.index,
                "page": chunk.page,
                "page_end": chunk.page_end,
                "title": chunk.title,
                "score": chunk.score,
                "bm25_score": chunk.bm25_score,
                "fused_score": chunk.fused_score,
                "section": chunk.section,
                "matched": chunk.matched,
                "expanded_from": chunk.expanded_from,
                "quote": chunk.quote,
                "quote_issue": chunk.quote_issue,
                "missing_numbers": chunk.missing_numbers,
                "rejected_quote": chunk.rejected_quote,
                "text": chunk.text,
            }
            for number, chunk in enumerate(result.sources, start=1)
        ],
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> dict:
    """Answer a question using only the most relevant parts of an uploaded paper."""
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is empty.")
    return _answer_response(request.paper_id, question)


@app.post("/ask_voice", response_model=AskVoiceResponse)
def ask_voice(paper_id: str = Form(...), file: UploadFile = File(...)) -> dict:
    """Transcribe a spoken question, then answer it exactly like /ask."""
    # Check the paper first, so audio for an unknown paper is never sent for transcription.
    try:
        get_paper_collection(paper_id)
    except PaperNotFoundError as exc:
        file.file.close()
        raise _paper_not_found() from exc

    transcript = _transcribe_upload(file).strip()
    if not transcript:
        raise HTTPException(
            status_code=400,
            detail="No speech was recognized in the audio. Record the question again, closer to the microphone.",
        )
    if len(transcript) > MAX_QUESTION_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"The transcribed question is longer than {MAX_QUESTION_CHARS} characters.",
        )
    return {**_answer_response(paper_id, transcript), "transcript": transcript}


@app.post("/speak", response_class=Response, responses={200: {"content": {"audio/mpeg": {}}}})
def speak(request: SpeakRequest) -> Response:
    """Read a text (the app sends the short /ask answer) aloud; returns MP3.

    Citation markers and Markdown are removed first. OpenAI failures map like every
    other endpoint: 502 with a message safe to show (no internal details), 500 if the
    server has no API key.
    """
    text = clean_for_speech(request.text)
    if not text:
        raise HTTPException(status_code=400, detail="There is no text to read aloud.")
    if len(text) > TTS_MAX_CHARS:
        raise HTTPException(status_code=413, detail=f"Text is too long to read aloud. Maximum is {TTS_MAX_CHARS} characters.")
    with _llm_http_errors():
        audio = synthesize(text)
    return Response(content=audio, media_type="audio/mpeg")
