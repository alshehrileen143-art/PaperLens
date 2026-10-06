"""Speech to text for /ask_voice."""

from pathlib import Path

import config
from llm import get_client, openai_errors


def transcribe(path: Path) -> str:
    """Transcribe an audio file with OpenAI. The language is detected automatically.

    The file's extension tells OpenAI its format, so the path must keep it.
    """
    client = get_client()
    with openai_errors(), open(path, "rb") as audio:
        result = client.audio.transcriptions.create(model=config.STT_MODEL, file=audio)
    return result.text
