"""OCR benchmark: speed and accuracy of the project's OCR engine (ocr.py), and of PaddleOCR if it is
installed, on every page of a PDF that has a text layer, clean and with simulated scan damage.
pypdf's text of each page is the reference. Local only: no API, and no model download (blocked).

Each run writes new files, never over old ones:
    eval/results/ocr_benchmark_<time>.json   the OCR text and time of every page, per engine (not in git:
                                             it holds the paper's text)
    eval/ocr_benchmark_<time>.md             the report
eval/ocr_benchmark.md is the run of 2026-10-12 (PaddleOCR vs RapidOCR) that chose RapidOCR.

Run (PowerShell, from the project folder):
    .\\venv\\Scripts\\python.exe -u eval\\ocr_benchmark.py
    .\\venv\\Scripts\\python.exe -u eval\\ocr_benchmark.py --engines project --seed 1234 --jpeg-quality 30
    .\\venv\\Scripts\\python.exe eval\\ocr_benchmark.py --report-only eval\\results\\ocr_benchmark_<time>.json
"""

import argparse
import importlib.util
import json
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PDF = ROOT / "samples" / "1-s2.0-S0021755720302072-main.pdf"
VERSIONS = ("clean", "scanned")
ENGINE_NAMES = {"project": "Project OCR (ocr.py)", "paddle": "PaddleOCR small (old setup)"}

# Superscript digits ("et al.²⁷", "I²") and the middle dot ("128·9") are written differently by the
# text layer and by OCR; both sides are read the same way, so only real misreadings count.
_SAME_DIGITS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹·", "0123456789.")


def words(text: str) -> Counter:
    return Counter(re.findall(r"[a-z]+", text.lower()))


def numbers(text: str) -> Counter:
    return Counter(re.findall(r"\d+(?:[.,]\d+)?", text.translate(_SAME_DIGITS)))


def recall(found: Counter, reference: Counter) -> tuple[int, int]:
    return sum(min(count, found[item]) for item, count in reference.items()), sum(reference.values())


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


def run_engine(engine: str, folder: Path) -> dict:
    """OCR of every image in the folder: {image name: {"seconds", "text"}}, after one warm-up call."""
    sys.path.insert(0, str(ROOT))
    import numpy as np
    from PIL import Image

    if engine == "project":
        import ocr
        if importlib.util.find_spec("rapidocr"):
            from rapidocr.utils import download_file

            def blocked(cls, url, logger):
                raise SystemExit(f"DOWNLOAD BLOCKED: {url}")
            download_file.DownloadFile._make_http_request = classmethod(blocked)
        read = ocr._ocr_image
    else:  # PaddleOCR with the settings ocr.py used before RapidOCR.
        from paddleocr import PaddleOCR
        paddle = PaddleOCR(lang="en", enable_mkldnn=False, use_doc_orientation_classify=False,
                           use_doc_unwarping=False, use_textline_orientation=False,
                           text_detection_model_name="PP-OCRv6_small_det",
                           text_recognition_model_name="PP-OCRv6_small_rec")

        def read(image):
            return "\n".join(line for result in paddle.predict(image) for line in result["rec_texts"])

    def bgr(path: Path):
        return np.ascontiguousarray(np.asarray(Image.open(path).convert("RGB"))[:, :, ::-1])

    images = sorted(folder.iterdir())
    read(bgr(images[0]))  # Warm-up (model load), not timed.
    results = {}
    for path in images:
        image = bgr(path)
        start = time.perf_counter()
        text = read(image)
        results[path.stem] = {"seconds": time.perf_counter() - start, "text": text}
        print(f"{engine} {path.stem}: {results[path.stem]['seconds']:.1f} s", file=sys.stderr, flush=True)
    return results


def report(run: dict) -> str:
    from pypdf import PdfReader
    references = [page.extract_text() for page in PdfReader(run["pdf"]).pages]
    settings = run["settings"]
    lines = [
        f"# OCR benchmark ({run['time']})",
        "",
        f"`{Path(run['pdf']).name}`, {len(references)} pages at {settings['dpi']} DPI. pypdf's text of each page "
        "is the reference.",
        f"- **clean:** the page as rendered. **scanned:** tilted 1 to 2 degrees, Gaussian noise (sigma "
        f"{settings['noise']}), JPEG quality {settings['jpeg_quality']}; seed {settings['seed']} + page number.",
        "- Time per page after one warm-up call. Recall counts the reference's words and numbers (with repeats)",
        "  found in the OCR text, in any order. Superscript digits and the middle dot read as plain digits and a",
        "  point on both sides. \"Extra\" numbers include those in figures, which the text layer does not have.",
        f"- Tilts: {', '.join(f'{tilt:+.2f}°' for tilt in run['tilts'])}.",
        "",
        "| Engine | Version | s/page (mean) | s/page (max) | Word recall | Number recall | Missing | Extra |",
        "|---|---|---|---|---|---|---|---|",
    ]
    details = []
    for engine, results in run["engines"].items():
        for version in VERSIONS:
            seconds, word_hits, word_total, number_hits, number_total, missing_all, extra_all = [], 0, 0, 0, 0, 0, 0
            for page, reference in enumerate(references, start=1):
                result = results[f"p{page}_{version}"]
                seconds.append(result["seconds"])
                hits, total = recall(words(result["text"]), words(reference))
                word_hits, word_total = word_hits + hits, word_total + total
                found, expected = numbers(result["text"]), numbers(reference)
                hits, total = recall(found, expected)
                number_hits, number_total = number_hits + hits, number_total + total
                missing, extra = expected - found, found - expected
                missing_all, extra_all = missing_all + sum(missing.values()), extra_all + sum(extra.values())
                details.append((engine, version, page, result["seconds"], hits, total, missing, extra))
            lines.append(f"| {ENGINE_NAMES[engine]} | {version} | {sum(seconds) / len(seconds):.1f} | "
                         f"{max(seconds):.1f} | {word_hits / word_total:.1%} | "
                         f"{number_hits / number_total:.1%} ({number_hits}/{number_total}) | {missing_all} | {extra_all} |")

    def show(counter: Counter) -> str:
        return ", ".join(f"`{n}`" + (f" ×{c}" if c > 1 else "") for n, c in sorted(counter.items())) or "—"

    lines += ["", "## Numbers that differ from the reference, per page", ""]
    for engine in run["engines"]:
        for version in VERSIONS:
            lines += [f"### {ENGINE_NAMES[engine]}, {version}", "",
                      "| Page | Seconds | Number recall | Missing (in reference, not read) | Extra (read, not in reference) |",
                      "|---|---|---|---|---|"]
            lines += [f"| {page} | {seconds:.1f} | {hits}/{total} | {show(missing)} | {show(extra)} |"
                      for row_engine, row_version, page, seconds, hits, total, missing, extra in details
                      if (row_engine, row_version) == (engine, version)]
            lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--engines", nargs="+", choices=list(ENGINE_NAMES), default=list(ENGINE_NAMES))
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--noise", type=float, default=8)
    parser.add_argument("--jpeg-quality", type=int, default=30)
    parser.add_argument("--report-only", type=Path, help="Write the report of a saved results file; no OCR.")
    parser.add_argument("--run-engine", nargs=2, metavar=("ENGINE", "FOLDER"), help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.run_engine:  # One engine in its own process, so the two never share memory or threads.
        print("RESULT " + json.dumps(run_engine(args.run_engine[0], Path(args.run_engine[1]))))
        return
    if args.report_only:
        run = json.loads(args.report_only.read_text(encoding="utf-8"))
        out = ROOT / "eval" / f"ocr_benchmark_{run['time']}.md"
        if out.exists():
            sys.exit(f"{out} exists; not overwritten.")
        out.write_text(report(run) + "\n", encoding="utf-8")
        print(f"written {out}")
        return

    engines = [engine for engine in args.engines
               if engine != "paddle" or importlib.util.find_spec("paddleocr")]
    if "paddle" in args.engines and "paddle" not in engines:
        print("PaddleOCR is not installed: skipped.")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = ROOT / "eval" / "results" / f"ocr_benchmark_{stamp}.json"
    report_path = ROOT / "eval" / f"ocr_benchmark_{stamp}.md"
    with tempfile.TemporaryDirectory(prefix="ocr_benchmark_") as folder:  # Images stay out of the project.
        tilts = make_images(args.pdf, Path(folder), args.dpi, args.seed, args.noise, args.jpeg_quality)
        run = {"time": stamp, "pdf": str(args.pdf), "tilts": tilts, "engines": {},
               "settings": {"dpi": args.dpi, "seed": args.seed, "noise": args.noise,
                            "jpeg_quality": args.jpeg_quality}}
        for engine in engines:
            done = subprocess.run([sys.executable, __file__, "--run-engine", engine, folder], capture_output=True,
                                  text=True, encoding="utf-8", errors="replace", timeout=3600)
            if "DOWNLOAD BLOCKED" in done.stdout + done.stderr:
                sys.exit("Stopped: the OCR engine tried to download a model.")
            lines = [line for line in done.stdout.splitlines() if line.startswith("RESULT ")]
            if not lines:
                sys.exit(f"{engine} failed:\n{done.stderr[-1500:]}")
            run["engines"][engine] = json.loads(lines[-1][7:])
            print(f"{engine}: done", flush=True)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(run, ensure_ascii=False), encoding="utf-8")
    report_path.write_text(report(run) + "\n", encoding="utf-8")
    print(f"written {results_path}\nwritten {report_path}")


if __name__ == "__main__":
    main()
