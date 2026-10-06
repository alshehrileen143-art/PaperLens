"""Safety checks for uploaded files, run before any parser touches them.

Each check raises UploadRejected with an HTTP status code and a message that
is safe to show to the user.
"""

import codecs
import zipfile
from pathlib import Path
from typing import BinaryIO

import pymupdf
from PIL import Image

from config import (
    MAX_DOCX_ENTRIES,
    MAX_DOCX_UNCOMPRESSED_BYTES,
    MAX_IMAGE_SIDE_PX,
    MAX_PDF_PAGES,
    MAX_UPLOAD_BYTES,
    MB,
    UPLOAD_CHUNK_BYTES,
)

# The PDF spec lets "%PDF-" appear anywhere in the first 1024 bytes.
_PDF_HEADER_WINDOW = 1024

_MAGIC_BYTES = {
    ".docx": (b"PK\x03\x04",),
    ".png": (b"\x89PNG\r\n\x1a\n",),
    ".jpg": (b"\xff\xd8\xff",),
    ".jpeg": (b"\xff\xd8\xff",),
}

_PIL_FORMATS = {".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG"}


class UploadRejected(Exception):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def save_upload_limited(source: BinaryIO, destination: BinaryIO, max_bytes: int = MAX_UPLOAD_BYTES) -> None:
    """Copy the upload in chunks, stopping as soon as it exceeds max_bytes."""
    total = 0
    while chunk := source.read(UPLOAD_CHUNK_BYTES):
        total += len(chunk)
        if total > max_bytes:
            raise UploadRejected(413, f"File is too large. Maximum size is {max_bytes // MB} MB.")
        destination.write(chunk)


def _check_magic_bytes(path: Path, extension: str) -> None:
    with open(path, "rb") as file:
        head = file.read(_PDF_HEADER_WINDOW)
    if extension == ".pdf":
        matches = b"%PDF-" in head
    else:
        matches = head.startswith(_MAGIC_BYTES[extension])
    if not matches:
        raise UploadRejected(400, f"File content does not match its '{extension}' extension.")


def _check_txt(path: Path) -> None:
    """Require valid UTF-8 text, decoded incrementally so large files use little memory."""
    decoder = codecs.getincrementaldecoder("utf-8")()
    try:
        with open(path, "rb") as file:
            while chunk := file.read(UPLOAD_CHUNK_BYTES):
                if b"\x00" in chunk:
                    raise UploadRejected(400, "File content does not match its '.txt' extension.")
                decoder.decode(chunk)
            decoder.decode(b"", final=True)
    except UnicodeDecodeError as exc:
        raise UploadRejected(400, "Text file is not valid UTF-8.") from exc


def _check_image(path: Path, extension: str) -> None:
    # Image.open only parses the header; pixels are not decoded here.
    try:
        with Image.open(path) as image:
            image_format, (width, height) = image.format, image.size
    except Image.DecompressionBombError as exc:
        raise UploadRejected(413, "Image dimensions are too large.") from exc
    except Exception as exc:
        raise UploadRejected(400, "Could not read the image. The file may be corrupted.") from exc

    if image_format != _PIL_FORMATS[extension]:
        raise UploadRejected(400, f"File content does not match its '{extension}' extension.")
    if width > MAX_IMAGE_SIDE_PX or height > MAX_IMAGE_SIDE_PX:
        raise UploadRejected(
            413,
            f"Image is {width}x{height} px. Maximum is {MAX_IMAGE_SIDE_PX}x{MAX_IMAGE_SIDE_PX} px.",
        )


def _check_docx(path: Path) -> None:
    """Guard against zip bombs using the sizes declared in the zip directory.

    Python's zipfile never returns more bytes than an entry's declared size,
    so a file that lies about its size cannot expand beyond this limit.
    """
    try:
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            if len(entries) > MAX_DOCX_ENTRIES:
                raise UploadRejected(413, f"DOCX contains too many internal files (max {MAX_DOCX_ENTRIES}).")
            uncompressed = sum(entry.file_size for entry in entries)
            if uncompressed > MAX_DOCX_UNCOMPRESSED_BYTES:
                raise UploadRejected(
                    413,
                    f"DOCX expands to more than {MAX_DOCX_UNCOMPRESSED_BYTES // MB} MB when unpacked.",
                )
            if "word/document.xml" not in archive.namelist():
                raise UploadRejected(400, "File content does not match its '.docx' extension.")
    except zipfile.BadZipFile as exc:
        raise UploadRejected(400, "Could not read the DOCX file. The file may be corrupted.") from exc


def _check_pdf(path: Path) -> None:
    try:
        with pymupdf.open(path) as document:
            page_count = document.page_count
    except Exception as exc:
        raise UploadRejected(400, "Could not read the PDF file. The file may be corrupted.") from exc
    if page_count > MAX_PDF_PAGES:
        raise UploadRejected(413, f"PDF has {page_count} pages. Maximum is {MAX_PDF_PAGES}.")


def validate_saved_upload(path: Path, extension: str) -> None:
    """Verify the saved file's real type matches its extension and is within limits."""
    if extension == ".txt":
        _check_txt(path)
        return

    _check_magic_bytes(path, extension)
    if extension in _PIL_FORMATS:
        _check_image(path, extension)
    elif extension == ".docx":
        _check_docx(path)
    elif extension == ".pdf":
        _check_pdf(path)
