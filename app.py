"""Streamlit chat front end for the FastAPI server (main.py).

Every action is an HTTP call to the API; no extraction, RAG or OpenAI logic
lives here. A paper is attached with the 📎 button of the message box, a
question is typed or recorded with its 🎤 button, and everything (upload,
summary, answers) appears as messages in one conversation. The interface is
in Arabic (right to left) or English; paper text keeps its own direction.

Start the API first (uvicorn main:app), then: streamlit run app.py
Theme and font: .streamlit/config.toml.
"""

import hashlib
import html
import os

import requests
import streamlit as st

from config import MAX_UPLOAD_BYTES, MB
from ui_text import (LANGUAGES, TEXTS, clip, interface_html, isolate, list_html, nearest_sentence, rich_html,
                     palette_css, shorten_filename, shown_citations, source_html, table_html)

# The API address; the API_URL environment variable overrides it (e.g. to test a second server).
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")
# Upload and summarize can run OCR and several OpenAI calls.
LONG_TIMEOUT_SECONDS = 300
ASK_TIMEOUT_SECONDS = 120
SERVER_COMMAND = ".\\venv\\Scripts\\python.exe -m uvicorn main:app --port 8000"
AUDIO_EXTENSIONS = (".wav", ".webm", ".mp3", ".m4a")
AVATARS = {"user": "👤", "assistant": "🔍"}
MAX_SOURCE_CARDS = 3  # The API sends at most 3 (config.MAX_ANSWER_SOURCES).

STYLE = """
<style>
/* Every colour is a variable from ui_text.PALETTES (light and dark), set by palette_css(). */
/* One width for the header, the welcome cards, the messages and the message box. */
.block-container, [data-testid="stBottomBlockContainer"] { max-width: 820px; margin-left: auto;
                                                             margin-right: auto; }
/* Streamlit's top bar (the ⋮ menu, shown with toolbarMode = "viewer") is about 3.75rem high and
   sits over the page: leave room for it, and keep it transparent so it shows each mode's background. */
.block-container { padding-top: 4.75rem; }
[data-testid="stHeader"] { background: transparent; }
.st-key-app-header h1 { font-size: 1.9rem; color: var(--accent); padding: 0; margin: 0; }
.app-tagline { color: var(--text-secondary); font-size: 0.92rem; margin: 0.15rem 0 0 0; }
.header-rule { border: none; border-top: 1px solid var(--border); margin: 0.9rem 0 1.1rem 0; }
.welcome-cards { display: flex; flex-wrap: wrap; gap: 0.8rem; margin: 1.2rem 0; }
.welcome-card { flex: 1 1 13rem; background: var(--card-bg); border: 1px solid var(--border); border-radius: 0.75rem;
                padding: 1rem 1.1rem; }
.welcome-card .card-icon { font-size: 1.5rem; line-height: 1; margin-bottom: 0.5rem; }
.welcome-card .card-title { color: var(--accent); font-weight: 600; margin-bottom: 0.25rem; }
.welcome-card .card-text { color: var(--text-secondary); font-size: 0.86rem; line-height: 1.6; }
.paper-chip { display: inline-block; background: var(--chip-bg); color: var(--accent);
              border: 1px solid var(--chip-border);
              border-radius: 999px; padding: 0.15rem 0.8rem; font-size: 0.85rem; }
.rich p, .rich li, .paper-text p { unicode-bidi: plaintext; text-align: start; }.rich p { margin: 0 0 0.45rem 0; line-height: 1.75; }
.rich ul { margin: 0.2rem 0 0.6rem 0; padding-inline-start: 1.3rem; }
.cite-ref { display: inline-block; min-width: 1.25em; padding: 0 0.32em; margin-inline-start: 0.2em;
            border-radius: 0.6em; background: var(--badge-bg); color: var(--accent); font-size: 0.7em; font-weight: 600;
            text-align: center; line-height: 1.5; vertical-align: super; }
.cite-head { display: flex; align-items: center; gap: 0.5rem; color: var(--accent); font-size: 0.9rem; }
.cite-num { display: inline-flex; align-items: center; justify-content: center; width: 1.6rem; height: 1.6rem;
            min-width: 1.6rem; border-radius: 50%; background: var(--accent); color: var(--on-accent); font-size: 0.8rem;
            font-weight: 600; }
.cite-quote { border-inline-start: 3px solid var(--accent); background: var(--quote-bg); border-radius: 0.35rem;
              padding: 0.5rem 0.8rem; margin: 0.45rem 0 0.2rem 0; font-size: 0.93rem; line-height: 1.7;
              unicode-bidi: plaintext; text-align: start; }
.paper-text p { font-size: 0.88rem; color: var(--paper-text); line-height: 1.65; margin: 0 0 0.6rem 0; }
.paper-text p.muted { color: var(--muted-text); }
.paper-text .quote-mark { background: var(--highlight-bg); color: var(--highlight-text); font-weight: 600; border-radius: 0.25rem;
                          padding: 0.05rem 0.2rem; box-decoration-break: clone; -webkit-box-decoration-break: clone; }
.paper-text .answer-number { font-weight: 700; }
.paper-text .rest-of-section { border-top: 1px solid var(--border); margin-top: 0.4rem; padding-top: 0.4rem; }
.paper-text .rest-of-section summary { color: var(--accent); cursor: pointer; font-size: 0.85rem;
                                       font-weight: 600; margin-bottom: 0.4rem; }
.sources-label { color: var(--text-secondary); font-size: 0.82rem; margin: 0.6rem 0 0.2rem 0; }
.cite-quote.unverified { border-inline-start-color: var(--unverified-border); color: var(--unverified-text); }
.cite-note { color: var(--text-secondary); font-size: 0.75rem; margin-top: 0.15rem; }
.cite-missing { color: var(--warm-heading); background: var(--warm-bg); border-radius: 0.35rem; padding: 0.4rem 0.7rem;
                margin: 0.45rem 0 0.2rem 0; font-size: 0.88rem; }
[class*="st-key-cite-"] { background: var(--card-bg); }
[class*="st-key-summary-"] { background: var(--summary-bg); border-color: var(--summary-border) !important; }
.summary-title { color: var(--accent); font-size: 1.15rem; font-weight: 700; padding-bottom: 0.5rem;
                 border-bottom: 1px solid var(--border); margin-bottom: 0.6rem; }
.summary-section { margin: 0.75rem 0 0.25rem 0; }
.summary-heading { color: var(--accent); font-weight: 600; font-size: 0.98rem; margin-bottom: 0.25rem; }
.mini-table { border-collapse: collapse; width: 100%; font-size: 0.88rem; }
.mini-table th { background: var(--table-head-bg); color: var(--table-head-text); font-weight: 600; text-align: start; padding: 0.35rem 0.6rem; }
.mini-table td { border-top: 1px solid var(--border); padding: 0.35rem 0.6rem; unicode-bidi: plaintext; text-align: start; }
.data-groups { display: flex; flex-wrap: wrap; gap: 0.6rem; }
.data-group { flex: 1 1 11rem; background: var(--data-bg); border: 1px solid var(--border); border-radius: 0.5rem;
              padding: 0.45rem 0.7rem; }
.data-group-title { color: var(--text-secondary); font-size: 0.8rem; font-weight: 600; }
.not-stated { color: var(--text-secondary); font-style: italic; font-size: 0.88rem; margin: 0.1rem 0 0.3rem 0; }
.summary-section.suggested { background: var(--warm-bg); border: 1px dashed var(--warm-border); border-radius: 0.5rem;
                             padding: 0.5rem 0.8rem; }
.summary-section.suggested .summary-heading { color: var(--warm-heading); }
.summary-section.suggested li { color: var(--warm-text); }
</style>
"""
# Arabic: the whole interface runs right to left. Paper text inside it keeps its own
# direction (dir="auto" and unicode-bidi: plaintext on every paragraph).
STYLE_RTL = """
<style>
.stApp { direction: rtl; }
[data-testid="stChatInput"] textarea { direction: rtl; text-align: right; }
</style>
"""


class ApiError(Exception):
    """A failed API call. Kept as a message key and values, so it can be shown in either language."""

    def __init__(self, key: str, **values):
        super().__init__(key)
        self.key, self.values = key, values


def t(key: str, **values) -> str:
    text = TEXTS[st.session_state.get("lang", "ar")][key]
    return text.format(**values) if values else text


def _error_detail(response: requests.Response) -> str:
    try:
        detail = response.json().get("detail")
    except ValueError:
        return response.text[:300] or response.reason
    if isinstance(detail, list):  # FastAPI validation errors (422).
        return "; ".join(item.get("msg", str(item)) for item in detail)
    return str(detail)


def _post(endpoint: str, timeout: int, **kwargs) -> requests.Response:
    """POST to the API and return the successful response, or raise ApiError."""
    try:
        response = requests.post(f"{API_URL}{endpoint}", timeout=timeout, **kwargs)
    except requests.ConnectionError as exc:
        raise ApiError("server_down", url=API_URL) from exc
    except requests.Timeout as exc:
        raise ApiError("timeout", seconds=timeout) from exc
    if not response.ok:
        raise ApiError("server_error", status=response.status_code, detail=_error_detail(response))
    return response


def api_post(endpoint: str, timeout: int, **kwargs) -> dict:
    """POST to the API and return its JSON, or raise ApiError."""
    return _post(endpoint, timeout, **kwargs).json()


def api_post_audio(endpoint: str, timeout: int, **kwargs) -> bytes:
    """POST to the API and return the audio it sends back, or raise ApiError."""
    return _post(endpoint, timeout, **kwargs).content


def server_is_up() -> bool:
    try:
        return requests.get(f"{API_URL}/health", timeout=3).ok
    except requests.RequestException:
        return False


def html_block(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


# --- Messages ---
# Messages are stored as data (keys, API responses), not as text, so the whole
# conversation is redrawn in the new language when the language changes.

def show_error(key: str, values: dict) -> None:
    st.error(t(key, **values))
    if key == "server_down":
        st.code(SERVER_COMMAND, language="powershell")


def show_hidden_text(hidden: dict | None) -> None:
    if hidden and hidden.get("count"):
        st.warning(t("hidden_removed", count=hidden["count"]))
        html_block(list_html(hidden.get("preview", [])))


def show_uploaded(index: int, message: dict) -> None:
    paper = message["paper"]
    st.markdown(t("uploaded", name=shorten_filename(paper["filename"]), chunks=paper["chunk_count"]))
    if message.get("first_of"):
        st.caption(t("one_file", name=shorten_filename(message["first_of"])))
    if paper.get("ocr_used"):
        st.warning(t("ocr_warning"))
    show_hidden_text(paper.get("hidden_text_removed"))
    current = st.session_state.get("paper") or {}
    if paper["paper_id"] == current.get("paper_id") and not message.get("summarized"):
        if st.button(t("summarize"), key=f"summarize-{index}", type="primary", icon="📝"):
            summarize(message)


def _section(key: str, body: str, css_class: str = "summary-section") -> str:
    return f'<div class="{css_class}"><div class="summary-heading">{html.escape(t(key))}</div>{body}</div>'


def _not_stated() -> str:
    return f'<div class="not-stated">{html.escape(t("not_stated"))}</div>'


def _from_paper_text(text: str) -> str:
    return rich_html(text) if text.strip() else _not_stated()


def _from_paper_list(items: list[str]) -> str:
    return list_html(items) if items else _not_stated()


def show_summary(index: int, response: dict) -> None:
    """The summary as its own card: a "📋" title, then one section per field with an icon heading.

    Fields taken from the paper only (objective, methodology, key findings, limitations,
    data and tools) are empty when the paper does not state them, and show "not stated in
    the paper" instead of being hidden. The model's own suggested limitations come right
    after the authors' limitations, in a differently coloured section that says so.
    """
    summary = response["summary"]
    with st.container(border=True, key=f"summary-{index}"):
        html_block(f'<div class="summary-title">{html.escape(t("summary_title"))}</div>')
        if response.get("ocr_warning"):
            st.warning(t("ocr_warning"))
        show_hidden_text(response.get("hidden_text_removed"))
        if response.get("output_warning"):
            st.warning(f"**{t('summary_output_warning')}:** {response['output_warning']}")
        tasks = summary["ai_ml_tasks"]
        suggested = summary.get("suggested_limitations") or []
        html_block("".join([
            _section("objective", _from_paper_text(summary["objective"])),
            _section("methodology", _from_paper_text(summary["methodology"])),
            _section("key_findings", _from_paper_list(summary["key_findings"])),
            _section("limitations", _from_paper_list(summary["limitations"])),
            _section("suggested_limitations", list_html(suggested), "summary-section suggested") if suggested else "",
            _section("use_cases", list_html([f"**{case['audience']}:** {case['application']}"
                                             for case in summary["use_cases"]], t("nothing"))),
            _section("technical_use_cases", list_html(summary["technical_use_cases"], t("nothing"))),
            _section("ai_ml_tasks", table_html([t("task_type"), t("task_input"), t("task_output")],
                                               [[task["task_type"], task["input"], task["output"]] for task in tasks])
                     if tasks else list_html([], t("nothing"))),
        ]))
        # Data and tools: four small groups; a group the paper does not mention says so.
        data = summary["data_and_tools"]
        cells = "".join(f'<div class="data-group"><div class="data-group-title">{html.escape(t(key))}</div>'
                        f"{_from_paper_list(data.get(key) or [])}</div>"
                        for key in ("datasets", "variables", "databases", "tools"))
        html_block(_section("data_and_tools", f'<div class="data-groups">{cells}</div>'))


def show_sources(index: int, sources: list[dict], answer: str) -> None:
    """One small card per source (at most MAX_SOURCE_CARDS), in this order:
    a heading line "[n] page · title", the quote 📌 (or, when no source has a verified
    quote, the sentence closest to the answer, marked as an unverified excerpt), and
    the full text folded under "show full text".

    The page is the quote's page; the API falls back to the chunk's first page.
    The title falls back to the section name.
    """
    html_block(f'<div class="sources-label">{html.escape(t("sources"))}</div>')
    for position, source in enumerate(sources[:MAX_SOURCE_CARDS], start=1):
        number = source.get("number", position)
        page = source.get("page")
        title = source.get("title") or source.get("section")
        page_end = source.get("page_end")
        if page is None:
            page_label = t("page_unknown")
        elif not source.get("quote") and page_end is not None and page_end > page:
            # No verified quote to place on one page: show the source's whole page range.
            page_label = t("pages", page=page, page_end=page_end)
        else:
            page_label = t("page", page=page)
        page_label = html.escape(page_label)
        title_label = f"<strong>{isolate(title)}</strong>" if title else html.escape(t("no_title"))
        with st.container(border=True, key=f"cite-{index}-{number}"):
            html_block(f'<div class="cite-head"><span class="cite-num">{number}</span>'
                       f'<span>{page_label} · {title_label}</span></div>')
            if source.get("quote"):
                html_block(f'<div class="cite-quote" dir="auto">📌 {html.escape(clip(source["quote"]))}</div>')
            elif source.get("quote_issue") == "number_mismatch":
                # The quote lacked a number of the answer: no excerpt is shown, since the
                # closest sentence by words could be that same rejected quote.
                html_block(f'<div class="cite-missing">{html.escape(t("number_not_found"))}</div>')
            else:
                excerpt = nearest_sentence(source["text"], answer)
                html_block(f'<div class="cite-quote unverified" dir="auto">{html.escape(excerpt)}</div>'
                           f'<div class="cite-note">{html.escape(t("excerpt_unverified"))}</div>')
            with st.expander(t("full_text"), expanded=False):
                # The verified quote highlighted where it is, the answer's numbers in bold.
                html_block(source_html(source["text"], source.get("quote"), answer, t("rest_of_section")))


def show_speak(index: int, text: str) -> None:
    """The 🔊 button under an answer: reads the short answer aloud, only when pressed.

    The audio is kept in st.session_state under a hash of the text, so pressing again
    (or redrawing the page) plays it without calling the API again.
    """
    key = hashlib.sha256(text.encode("utf-8")).hexdigest()
    cache = st.session_state.setdefault("tts_audio", {})
    if st.button(t("listen"), key=f"speak-{index}") and key not in cache:
        try:
            with st.spinner(t("speaking")):
                cache[key] = api_post_audio("/speak", ASK_TIMEOUT_SECONDS, json={"text": text})
        except ApiError as exc:
            if exc.key == "server_error":
                st.error(t("speak_failed", detail=exc.values["detail"]))
            else:
                show_error(exc.key, exc.values)
    if key in cache:
        st.audio(cache[key], format="audio/mpeg")


def show_answer(index: int, response: dict) -> None:
    # Only sources with a verified quote are shown, renumbered 1..n in the answer too
    # (or the first source alone when none has one); see ui_text.shown_citations.
    answer, details, sources = shown_citations(response["answer"], response.get("details"),
                                               response.get("sources") or [])
    html_block(rich_html(answer, citations=True))
    # Only the short answer is read aloud: not the details or the sources.
    show_speak(index, answer)
    if details:
        with st.expander(t("more_details")):
            html_block(rich_html(details, citations=True))
    for key in ("grounding_warning", "citation_mismatch", "output_warning"):
        if response.get(key):
            st.warning(f"**{t(key)}:** {response[key]}")
    if sources:
        show_sources(index, sources, f"{answer} {details or ''}")


def show_message(index: int, message: dict) -> None:
    with st.chat_message(message["role"], avatar=AVATARS[message["role"]]):
        kind = message["kind"]
        if kind == "text":
            # The bubble follows its own text, so an English question keeps its "?" at the end.
            html_block(f'<div class="user-text" dir="auto">{rich_html(message["text"])}</div>')
            if message.get("voice"):
                st.caption(t("voice_caption"))
        elif kind == "file":
            st.markdown(f"📎 {isolate(shorten_filename(message['name']))}", unsafe_allow_html=True)
        elif kind == "voice_failed":
            st.markdown(t("voice_question"))
        elif kind == "uploaded":
            show_uploaded(index, message)
        elif kind == "summary":
            show_summary(index, message["response"])
        elif kind == "answer":
            show_answer(index, message["response"])
        elif kind == "info":
            st.info(t(message["key"]))
        elif kind == "error":
            show_error(message["key"], message["values"])


# --- Actions ---

def add(role: str, kind: str, **fields) -> None:
    st.session_state.messages.append({"role": role, "kind": kind, **fields})


def add_error(exc: ApiError) -> None:
    add("assistant", "error", key=exc.key, values=exc.values)


def upload(files: list) -> bool:
    file = files[0]
    add("user", "file", name=file.name)
    content = file.getvalue()
    try:
        with st.spinner(t("uploading")):
            response = api_post("/upload", LONG_TIMEOUT_SECONDS, files={"file": (file.name, content)})
    except ApiError as exc:
        add_error(exc)
        return False
    st.session_state.paper = {**response, "content": content}
    add("assistant", "uploaded", paper={k: v for k, v in response.items() if k != "normalization"},
        first_of=files[1].name if len(files) > 1 else None)
    return True


def summarize(message: dict) -> None:
    paper = st.session_state.paper
    try:
        with st.spinner(t("summarizing")):
            response = api_post("/summarize", LONG_TIMEOUT_SECONDS,
                                files={"file": (paper["filename"], paper["content"])})
    except ApiError as exc:
        add_error(exc)
    else:
        message["summarized"] = True
        add("assistant", "summary", response=response)
    st.rerun()


def ask_text(question: str) -> None:
    add("user", "text", text=question)
    paper = st.session_state.get("paper")
    if not paper:
        add("assistant", "info", key="need_paper")
        return
    try:
        with st.spinner(t("searching")):
            response = api_post("/ask", ASK_TIMEOUT_SECONDS, json={"paper_id": paper["paper_id"], "question": question})
    except ApiError as exc:
        add_error(exc)
    else:
        add("assistant", "answer", response=response)


def ask_voice(audio) -> None:
    paper = st.session_state.get("paper")
    if not paper:
        add("user", "voice_failed")
        add("assistant", "info", key="need_paper")
        return
    filename = audio.name if audio.name.lower().endswith(AUDIO_EXTENSIONS) else "question.wav"
    try:
        with st.spinner(t("transcribing")):
            response = api_post("/ask_voice", ASK_TIMEOUT_SECONDS, data={"paper_id": paper["paper_id"]},
                                files={"file": (filename, audio.getvalue(), audio.type)})
    except ApiError as exc:
        add("user", "voice_failed")
        add_error(exc)
        return
    add("user", "text", text=response["transcript"], voice=True)
    add("assistant", "answer", response=response)


def handle(value) -> None:
    """One submission of the message box: attached paper first, then the typed or spoken question."""
    files = getattr(value, "files", None) or []
    text = (getattr(value, "text", None) or "").strip()
    audio = getattr(value, "audio", None)
    if files and not upload(files):
        return
    if text:
        ask_text(text)
    if audio is not None:
        ask_voice(audio)


# --- Page ---

st.set_page_config(page_title="PaperLens", page_icon="🔍", layout="centered")
st.session_state.setdefault("messages", [])

with st.container(key="app-header"):
    header, switch = st.columns([4, 1.4], vertical_alignment="center")
    with switch:
        st.segmented_control(TEXTS["en"]["language"], options=list(LANGUAGES), format_func=LANGUAGES.get,
                             default="ar", required=True, key="lang", label_visibility="collapsed")
    # The colour variables first (light, dark, and the mode Streamlit reports), then the styles.
    html_block(f"<style>\n{palette_css(st.context.theme.type)}\n</style>" + STYLE
               + (STYLE_RTL if st.session_state.get("lang", "ar") == "ar" else ""))
    with header:
        st.title(f"🔍 {t('title')}")
        html_block(f'<p class="app-tagline">{html.escape(t("tagline"))}</p>')
html_block('<hr class="header-rule">')

if not server_is_up():
    show_error("server_down", {"url": API_URL})
    st.stop()

paper = st.session_state.get("paper")
if paper:
    html_block(f'<span class="paper-chip">📄 {html.escape(t("current_paper"))}: '
               f'{isolate(shorten_filename(paper["filename"]))}</span>')

if not st.session_state.messages:
    cards = "".join(
        f'<div class="welcome-card"><div class="card-icon">{icon}</div>'
        f'<div class="card-title">{interface_html(t(key + "_title"))}</div>'
        f'<div class="card-text">{interface_html(t(key + "_text"))}</div></div>'
        for icon, key in (("📎", "card_upload"), ("📝", "card_summary"), ("🎤", "card_ask"))
    )
    html_block(f'<div class="welcome-cards">{cards}</div>')

for index, message in enumerate(st.session_state.messages):
    show_message(index, message)

value = st.chat_input(
    t("placeholder") if paper else t("placeholder_no_paper"),
    key="chat",
    accept_file=True,
    accept_audio=True,
    max_upload_size=MAX_UPLOAD_BYTES // MB,
)
if value:
    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        handle(value)
    st.rerun()
