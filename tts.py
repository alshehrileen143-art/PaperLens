"""Text to speech for /speak: an answer read aloud as MP3."""

import re

import config
from llm import get_client, openai_errors

# Citation markers: "[1]", "[2,3]", "[2, 3]".
_CITATION = re.compile(r"\s*\[\d+(?:\s*,\s*\d+)*\]")
# Markdown that would be read out as symbols: bold/italic stars, headings, code ticks.
_MARKDOWN = re.compile(r"\*+|^\s*#+\s*|`+", re.MULTILINE)
_WHITESPACE = re.compile(r"\s+")


def clean_for_speech(text: str) -> str:
    """The text without citation markers and Markdown symbols, with runs of whitespace as one space."""
    text = _CITATION.sub("", text)
    text = _MARKDOWN.sub("", text)
    return _WHITESPACE.sub(" ", text).strip()


def synthesize(text: str) -> bytes:
    """Speak `text` with OpenAI and return the audio as MP3 bytes. The caller cleans and limits the text."""
    client = get_client()
    with openai_errors():
        response = client.audio.speech.create(
            model=config.TTS_MODEL,
            voice=config.TTS_VOICE,
            input=text,
            instructions=config.TTS_INSTRUCTIONS,
            response_format="mp3",
        )
        return response.content
