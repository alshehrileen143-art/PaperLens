"""OCR benchmark: speed and accuracy of the project's OCR (ocr.py) on every page of a PDF that has a
text layer, clean and with simulated scan damage. pypdf's text of each page is the reference.
Local only: no API, and no model download (blocked).

The engine runs once per page; its boxes are kept, so the report scores the same recognition in
two orders: the engine's (top to bottom) and the columns one of reading_order.py. Measures:
- word and number recall: how many of the reference's words and numbers OCR read, in any order;
- sentence recall and 5-gram recall: how many of the reference's sentences (8 words or more) and
  runs of 5 words appear unbroken, so they show lines read out of order;
- quotes: how many of 30 random sentences of the reference quote_in_source finds in the OCR text
  (what a quote check in /ask meets); the reference against itself is the check of the measure;
- sections: the headings detect_sections finds, against those it finds in the reference.

Each run writes new files, never over old ones:
    eval/results/ocr_benchmark_<time>.json   the OCR lines, boxes and time of every page (not in
                                             git: it holds the paper's text)
    eval/ocr_benchmark_<time>.md             the report
eval/ocr_benchmark.md is the run of 2026-10-12 (PaddleOCR vs RapidOCR) that chose RapidOCR; the
script of that run, with PaddleOCR, is in the git history.

Run (PowerShell, from the project folder):
    .\\venv\\Scripts\\python.exe -u eval\\ocr_benchmark.py
    .\\venv\\Scripts\\python.exe -u eval\\ocr_benchmark.py --pdf <file.pdf> --seed 1234 --jpeg-quality 30
    .\\venv\\Scripts\\python.exe eval\\ocr_benchmark.py --report-only eval\\results\\ocr_benchmark_<time>.json
"""

import argparse
import json
import random
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DEFAULT_PDF = ROOT / "samples" / "1-s2.0-S0021755720302072-main.pdf"
VERSIONS = ("clean", "scanned")
ORDERS = {"engine": "engine order", "columns": "column order"}
QUOTE_SAMPLE = 30
MIN_SENTENCE_WORDS, MAX_SENTENCE_WORDS = 8, 40

# Superscript digits ("et al.²⁷", "I²") and the middle dot ("128·9") are written differently by the
# text layer and by OCR; both sides are read the same way, so only real misreadings count.
_SAME_DIGITS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹·", "0123456789.")


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.translate(_SAME_DIGITS).lower())


def words(text: str) -> Counter:
    return Counter(re.findall(r"[a-z]+", text.lower()))


def numbers(text: str) -> Counter:
    return Counter(re.findall(r"\d+(?:[.,]\d+)?", text.translate(_SAME_DIGITS)))


def recall(found: Counter, reference: Counter) -> tuple[int, int]:
    return sum(min(count, found[item]) for item, count in reference.items()), sum(reference.values())


def sentences(text: str) -> list[str]:
    """The reference's sentences of MIN_SENTENCE_WORDS words or more, as written (line breaks kept)."""
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [part.strip() for part in parts if len(tokens(part)) >= MIN_SENTENCE_WORDS]


def unbroken(reference: str, ocr_text: str) -> tuple[int, int, int, int]:
    """(sentences found unbroken, sentences, 5-grams found, 5-grams) of the reference in the OCR text."""
    ocr_tokens = tokens(ocr_text)
    joined = " " + " ".join(ocr_tokens) + " "
    found = [sentence for sentence in sentences(reference) if " " + " ".join(tokens(sentence)) + " " in joined]
    grams = Counter(zip(*(tokens(reference)[i:] for i in range(5))))
    ocr_grams = Counter(zip(*(ocr_tokens[i:] for i in range(5))))
    hits, total = recall(ocr_grams, grams)
    return len(found), len(sentences(reference)), hits, total


def make_images(pdf: Path, folder: Path, dpi: int, seed: int, noise: float, quality: int) -> list[float]:
    """Each page as a clean PNG and as a damaged JPEG (tilt 1-2 degrees, noise, JPEG compression);
    returns the tilt of each page. The seed is seed + page number, so a run can be made again."""
    import numpy as np
    import pymupdf
    from PIL import Image

    tilts = []
    with pymupdf.open(pdf) as document:
        for index, page in enumerate(document):
            pixmap = page.get_pixmap(dpi=dpi)
            clean = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
            clean.save(folder / f"p{index + 1}_clean.png")
            rng = np.random.default_rng(seed + index + 1)
            angle = rng.uniform(1.0, 2.0) * rng.choice([-1, 1])
            tilted = clean.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=(255, 255, 255))
            noisy = np.asarray(tilted, dtype=np.float32) + rng.normal(0, noise, (tilted.height, tilted.width, 1))
            Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8)).save(folder / f"p{index + 1}_scanned.jpg",
                                                                          quality=quality)
            tilts.append(float(angle))
    return tilts


def run_engine(folder: Path) -> dict:
    """The OCR engine of ocr.py on every image: {image name: {"seconds", "width", "lines", "boxes"}},
    the lines in the engine's own order, after one warm-up call."""
    import numpy as np
    from PIL import Image
    from rapidocr.utils import download_file

    import ocr

    def blocked(cls, url, logger):
        raise SystemExit(f"DOWNLOAD BLOCKED: {url}")
    download_file.DownloadFile._make_http_request = classmethod(blocked)
    engine = ocr._get_engine()

    def bgr(path: Path):
        return np.ascontiguousarray(np.asarray(Image.open(path).convert("RGB"))[:, :, ::-1])

    images = sorted(folder.iterdir())
    engine(bgr(images[0]))  # Warm-up (model load), not timed.
    results = {}
    for path in images:
        image = bgr(path)
        start = time.perf_counter()
        found = engine(image)
        seconds = time.perf_counter() - start
        boxes = [] if found.boxes is None else np.asarray(found.boxes).tolist()
        results[path.stem] = {"seconds": seconds, "width": image.shape[1], "lines": list(found.txts or ()),
                              "boxes": boxes}
        print(f"{path.stem}: {seconds:.1f} s", file=sys.stderr, flush=True)
    return results


def page_text(result: dict, order: str, stats: dict | None = None) -> str:
    from reading_order import reading_order
    lines = result["lines"]
    if order == "engine" or not result["boxes"]:
        return "\n".join(lines)
    return "\n".join(lines[index] for index in reading_order(result["boxes"], result["width"], stats))


def report(run: dict) -> str:
    from pypdf import PdfReader

    from quote_match import quote_in_source
    from sections import detect_sections

    references = [page.extract_text() for page in PdfReader(run["pdf"]).pages]
    reference_doc = "\n".join(references)
    rng = random.Random(run["settings"]["seed"])
    pool = [s for s in sentences(reference_doc) if len(tokens(s)) <= MAX_SENTENCE_WORDS]
    quotes = rng.sample(pool, min(QUOTE_SAMPLE, len(pool)))
    reference_sections = [name for _, name in detect_sections(reference_doc)]
    settings = run["settings"]
    lines = [
        f"# OCR benchmark ({run['time']})",
        "",
        f"`{Path(run['pdf']).name}`, {len(references)} pages at {settings['dpi']} DPI; pypdf's text is the reference.",
        f"- **clean:** the page as rendered. **scanned:** tilted 1 to 2 degrees, Gaussian noise (sigma "
        f"{settings['noise']}), JPEG quality {settings['jpeg_quality']}; seed {settings['seed']} + page number.",
        "- The engine runs once per page; both orders score the same recognition. Time per page after one",
        "  warm-up call (ordering takes milliseconds). Word and number recall do not depend on the order.",
        f"- Sentences: the reference's sentences of {MIN_SENTENCE_WORDS} words or more found unbroken; 5-grams: runs",
        f"  of 5 words found unbroken. Quotes: {len(quotes)} random sentences of the reference ({MIN_SENTENCE_WORDS} to "
        f"{MAX_SENTENCE_WORDS} words, seed {settings['seed']}) that quote_in_source finds; against the reference itself: "
        f"{sum(quote_in_source(q, reference_doc) for q in quotes)}/{len(quotes)}.",
        f"- Sections in the reference ({len(reference_sections)}): {', '.join(reference_sections)}.",
        f"- Tilts: {', '.join(f'{tilt:+.2f}°' for tilt in run['tilts'])}.",
        "",
        "| Version | Order | s/page | Words | Numbers | Sentences | 5-grams | Quotes | Sections found |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    per_page = []
    results = run["pages"]
    for version in VERSIONS:
        seconds = [results[f"p{page}_{version}"]["seconds"] for page in range(1, len(references) + 1)]
        for order in ORDERS:
            stats: dict = {}
            texts = [page_text(results[f"p{page}_{version}"], order, stats) for page in range(1, len(references) + 1)]
            doc = "\n".join(texts)
            word_hits = number_hits = word_total = number_total = 0
            sentence_hits = sentence_total = gram_hits = gram_total = 0
            for page, (reference, text) in enumerate(zip(references, texts), start=1):
                hits, total = recall(words(text), words(reference))
                word_hits, word_total = word_hits + hits, word_total + total
                hits, total = recall(numbers(text), numbers(reference))
                number_hits, number_total = number_hits + hits, number_total + total
                found, count, grams, gram_count = unbroken(reference, text)
                sentence_hits, sentence_total = sentence_hits + found, sentence_total + count
                gram_hits, gram_total = gram_hits + grams, gram_total + gram_count
                per_page.append((version, order, page, found, count, grams, gram_count))
            quoted = sum(quote_in_source(quote, doc) for quote in quotes)
            found_sections = [name for _, name in detect_sections(doc)]
            matched = sum((Counter(found_sections) & Counter(reference_sections)).values())
            lines.append(
                f"| {version} | {ORDERS[order]} | {sum(seconds) / len(seconds):.1f} | {word_hits / word_total:.1%} | "
                f"{number_hits / number_total:.1%} | {sentence_hits / sentence_total:.1%} ({sentence_hits}/{sentence_total}) | "
                f"{gram_hits / gram_total:.1%} | {quoted}/{len(quotes)} | {matched}/{len(reference_sections)} |")
            if order == "columns":
                lines.append(f"|  |  |  |  |  | layout: {stats.get('column_bands', 0)} bands in columns, "
                             f"{stats.get('row_bands', 0)} in rows, {stats.get('crossing', 0)} wide boxes |  |  |  |")
            per_page.append((version, order, "sections", found_sections))
    lines += ["", "## Per page: sentences and 5-grams found unbroken", "",
              "| Page | " + " | ".join(f"{version}, {ORDERS[order]}" for version in VERSIONS for order in ORDERS) + " |",
              "|---" * (1 + len(VERSIONS) * len(ORDERS)) + "|"]
    for page in range(1, len(references) + 1):
        cells = []
        for version in VERSIONS:
            for order in ORDERS:
                row = next(r for r in per_page if r[:3] == (version, order, page))
                cells.append(f"{row[3]}/{row[4]} · {row[5] / row[6]:.0%}" if row[6] else "—")
        lines.append(f"| {page} | " + " | ".join(cells) + " |")
    lines += ["", "## Sections found", ""]
    for row in per_page:
        if row[2] == "sections":
            lines.append(f"- **{row[0]}, {ORDERS[row[1]]}:** {', '.join(row[3])}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--noise", type=float, default=8)
    parser.add_argument("--jpeg-quality", type=int, default=30)
    parser.add_argument("--report-only", type=Path, help="Write the report of a saved results file; no OCR.")
    parser.add_argument("--run-engine", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.run_engine:  # The engine in its own process.
        print("RESULT " + json.dumps(run_engine(args.run_engine)))
        return
    if args.report_only:
        run = json.loads(args.report_only.read_text(encoding="utf-8"))
        out = ROOT / "eval" / f"ocr_benchmark_{run['time']}.md"
        if out.exists():  # A new report of the same run (after a change to the code): a new file.
            out = out.with_name(f"ocr_benchmark_{run['time']}_report_{datetime.now():%Y%m%d_%H%M%S}.md")
        out.write_text(report(run) + "\n", encoding="utf-8")
        print(f"written {out}")
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = ROOT / "eval" / "results" / f"ocr_benchmark_{stamp}.json"
    report_path = ROOT / "eval" / f"ocr_benchmark_{stamp}.md"
    with tempfile.TemporaryDirectory(prefix="ocr_benchmark_") as folder:  # Images stay out of the project.
        tilts = make_images(args.pdf, Path(folder), args.dpi, args.seed, args.noise, args.jpeg_quality)
        done = subprocess.run([sys.executable, __file__, "--run-engine", folder], capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=3600)
    if "DOWNLOAD BLOCKED" in done.stdout + done.stderr:
        sys.exit("Stopped: the OCR engine tried to download a model.")
    found = [line for line in done.stdout.splitlines() if line.startswith("RESULT ")]
    if not found:
        sys.exit(f"OCR failed:\n{done.stderr[-1500:]}")
    run = {"time": stamp, "pdf": str(args.pdf), "tilts": tilts, "pages": json.loads(found[-1][7:]),
           "settings": {"dpi": args.dpi, "seed": args.seed, "noise": args.noise, "jpeg_quality": args.jpeg_quality}}
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(run, ensure_ascii=False), encoding="utf-8")
    report_path.write_text(report(run) + "\n", encoding="utf-8")
    print(f"written {results_path}\nwritten {report_path}")


if __name__ == "__main__":
    main()
