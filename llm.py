"""Shared OpenAI client and error handling for summarization and RAG."""

import os
from contextlib import contextmanager
from functools import lru_cache

import openai
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI

from config import OPENAI_TIMEOUT_SECONDS, TOKENIZER_ENCODING

load_dotenv()


@lru_cache(maxsize=1)
def _encoding():
    return tiktoken.get_encoding(TOKENIZER_ENCODING)


def count_tokens(text: str) -> int:
    """Exact token count for gpt-4o / gpt-4o-mini (works for Arabic, unlike chars / 4)."""
    return len(_encoding().encode(text, disallowed_special=()))


class LLMError(Exception):
    """Raised when an OpenAI call fails; the message is safe to show users."""


class MissingAPIKeyError(LLMError):
    """Raised when OPENAI_API_KEY is not configured on the server."""


def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise MissingAPIKeyError("OPENAI_API_KEY is not set. Add it to the .env file.")
    return OpenAI(api_key=api_key, timeout=OPENAI_TIMEOUT_SECONDS)


@contextmanager
def openai_errors():
    """Translate OpenAI SDK exceptions into LLMError with a clear message."""
    try:
        yield
    except openai.AuthenticationError as exc:
        raise LLMError("OpenAI rejected the API key. Check OPENAI_API_KEY in .env.") from exc
    except openai.RateLimitError as exc:
        raise LLMError("OpenAI rate limit or quota exceeded. Try again later.") from exc
    except openai.APIConnectionError as exc:
        raise LLMError("Could not connect to OpenAI. Check your internet connection.") from exc
    except openai.APIStatusError as exc:
        raise LLMError(f"OpenAI returned an error (HTTP {exc.status_code}).") from exc
