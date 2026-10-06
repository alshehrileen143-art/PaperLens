"""OCR for images and scanned PDFs using PaddleOCR (English only)."""

import threading
from pathlib import Path

import numpy as np
import pymupdf

from config import MAX_IMAGE_SIDE_PX, MAX_OCR_PAGES, OCR_DPI

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")

# "small" is ~5x faster than the default "medium" on CPU (~40s vs ~190s per
# full page) with the same accuracy in our tests; "tiny" misread numbers.
OCR_MODEL_SIZE = "small"

_ocr_engine = None
_ocr_lock = threading.Lock()


class OCRError(Exception):
    """Raised when the OCR engine itself fails (not when the input file is bad)."""


def _get_engine():
    """Load the PaddleOCR model once, on first use, and reuse it afterwards."""
    global _ocr_engine
    if _ocr_engine is None:
        # Imported here because loading Paddle takes several seconds and is only
        # needed when a file actually requires OCR.
        from paddleocr import PaddleOCR

        _ocr_engine = PaddleOCR(
            lang="en",
            # paddlepaddle 3.3 on Windows CPU crashes during inference when
            # oneDNN is enabled (ConvertPirAttribute2RuntimeAttribute error).
            enable_mkldnn=False,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_detection_model_name=f"PP-OCRv6_{OCR_MODEL_SIZE}_det",
            text_recognition_model_name=f"PP-OCRv6_{OCR_MODEL_SIZE}_rec",
        )
    return _ocr_engine


def _pixmap_to_bgr(pix: pymupdf.Pixmap) -> np.ndarray:
    """Convert a PyMuPDF pixmap to the BGR uint8 array PaddleOCR expects."""
    if pix.alpha:
        pix = pymupdf.Pixmap(pix, 0)
    if pix.colorspace is None or pix.colorspace.n != 3:
        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
    rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    return np.ascontiguousarray(rgb[:, :, ::-1])


def _ocr_image(image: np.ndarray) -> str:
    # PaddleOCR is not documented as thread-safe, and FastAPI runs sync
    # endpoints in a thread pool, so only one OCR call runs at a time.
    with _ocr_lock:
        try:
            engine = _get_engine()
            results = engine.predict(image)
        except Exception as exc:
            raise OCRError("The OCR engine failed to process the file.") from exc
    lines = []
    for result in results:
        lines.extend(result["rec_texts"])
    return "\n".join(lines)


def ocr_pdf_pages(file_path: str | Path, page_indexes: list[int]) -> dict[int, str]:
    """OCR the given 0-based PDF pages; returns {page_index: text}."""
    texts = {}
    with pymupdf.open(file_path) as document:
        for index in page_indexes:
            page = document[index]
            # A PDF can declare huge page sizes; lower the DPI so the rendered
            # image stays within the same limit as uploaded images.
            longest_side_pt = max(page.rect.width, page.rect.height, 1)
            dpi = min(OCR_DPI, int(MAX_IMAGE_SIDE_PX * 72 / longest_side_pt))
            pixmap = page.get_pixmap(dpi=max(dpi, 1))
            texts[index] = _ocr_image(_pixmap_to_bgr(pixmap))
    return texts


def limit_note(skipped: int) -> str:
    return f"[OCR limited to {MAX_OCR_PAGES} pages; {skipped} more scanned page(s) were skipped.]"


def extract_text_ocr(file_path: str | Path) -> str:
    """Return OCR text from a PNG/JPG image or a scanned PDF (first MAX_OCR_PAGES pages)."""
    path = Path(file_path)
    extension = path.suffix.lower()

    if extension in IMAGE_EXTENSIONS:
        return _ocr_image(_pixmap_to_bgr(pymupdf.Pixmap(str(path))))

    if extension == ".pdf":
        with pymupdf.open(path) as document:
            page_count = document.page_count
        texts = ocr_pdf_pages(path, list(range(min(page_count, MAX_OCR_PAGES))))
        pages = [texts[index] for index in sorted(texts)]
        if page_count > MAX_OCR_PAGES:
            pages.append(limit_note(page_count - MAX_OCR_PAGES))
        return "\n\n".join(pages)

    raise ValueError(f"OCR does not support '{extension}' files.")
