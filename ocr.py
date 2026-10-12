"""OCR for images and scanned PDFs: RapidOCR (onnxruntime) with the PP-OCRv6 small models.

The models are the ones PaddleOCR used here before; onnxruntime runs them about 4 times faster on
this CPU (about 10 s a page against 40 s) with the same accuracy on numbers: eval/ocr_benchmark.md.
The models ship inside the rapidocr package; nothing is downloaded (see _get_engine).
"""

import threading
from pathlib import Path

import numpy as np
import pymupdf

from config import MAX_IMAGE_SIDE_PX, MAX_OCR_PAGES, OCR_DPI

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")

# RapidOCR's defaults use these PP-OCRv6 small files, bundled in the package (rapidocr/models).
MODEL_FILES = ("PP-OCRv6_det_small.onnx", "PP-OCRv6_rec_small.onnx")
ENGINE_PARAMS = {
    # Pages are upright scans; the orientation model only costs time (it was off with PaddleOCR too).
    "Global.use_cls": False,
    # RapidOCR shrinks images whose longest side is over 2000 px; a 200 DPI A4 page is about 2340.
    # Kept at full size: no slower in the benchmark, and small digits keep their pixels.
    "Global.max_side_len": 4000,
    "Global.log_level": "error",
}

_ocr_engine = None
_ocr_lock = threading.Lock()


class OCRError(Exception):
    """Raised when the OCR engine itself fails (not when the input file is bad)."""


def _get_engine():
    """Load the OCR models once, on first use, and reuse them afterwards.

    If a bundled model file is missing, RapidOCR would download it; the server must not fetch
    anything at run time, so that is an OCRError instead."""
    global _ocr_engine
    if _ocr_engine is None:
        # Imported here: only needed when a file actually requires OCR.
        import rapidocr
        from rapidocr import RapidOCR

        models = Path(rapidocr.__file__).parent / "models"
        missing = [name for name in MODEL_FILES if not (models / name).is_file()]
        if missing:
            raise OCRError(f"OCR model files are missing: {', '.join(missing)}. Reinstall rapidocr.")
        _ocr_engine = RapidOCR(params=dict(ENGINE_PARAMS))
    return _ocr_engine


def _pixmap_to_bgr(pix: pymupdf.Pixmap) -> np.ndarray:
    """Convert a PyMuPDF pixmap to the BGR uint8 array the OCR engine expects."""
    if pix.alpha:
        pix = pymupdf.Pixmap(pix, 0)
    if pix.colorspace is None or pix.colorspace.n != 3:
        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
    rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    return np.ascontiguousarray(rgb[:, :, ::-1])


def _ocr_image(image: np.ndarray) -> str:
    # One OCR call at a time: FastAPI runs sync endpoints in a thread pool, and a second call
    # would only compete for the same CPU cores.
    with _ocr_lock:
        try:
            engine = _get_engine()
            result = engine(image)
        except OCRError:
            raise
        except Exception as exc:
            raise OCRError("The OCR engine failed to process the file.") from exc
    return "\n".join(result.txts or ())


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


def skipped_page_marker(page_number: int) -> str:
    """Put in the text where a scanned page was not read (1-based page number), so the gap shows
    where it is: in the full text of a source, and to the model."""
    return f"[Page {page_number} was not read: OCR is limited to {MAX_OCR_PAGES} scanned pages per paper.]"


def extract_text_ocr(file_path: str | Path) -> str:
    """Return OCR text from a PNG/JPG image or a scanned PDF (first MAX_OCR_PAGES pages; each
    page after them is a skipped_page_marker)."""
    path = Path(file_path)
    extension = path.suffix.lower()

    if extension in IMAGE_EXTENSIONS:
        return _ocr_image(_pixmap_to_bgr(pymupdf.Pixmap(str(path))))

    if extension == ".pdf":
        with pymupdf.open(path) as document:
            page_count = document.page_count
        texts = ocr_pdf_pages(path, list(range(min(page_count, MAX_OCR_PAGES))))
        pages = [texts[index] if index in texts else skipped_page_marker(index + 1) for index in range(page_count)]
        return "\n\n".join(pages)

    raise ValueError(f"OCR does not support '{extension}' files.")
