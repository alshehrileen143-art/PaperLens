"""Tests for reading_order.py: OCR lines read column by column, from box coordinates only.

Boxes are made by hand, at the scale of a 200 DPI page (1650 px wide, lines 30 px high, 3 px
apart). No OCR engine and no API.

Run: .\\venv\\Scripts\\python.exe -m unittest tests.test_reading_order -v
"""

import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

import ocr  # noqa: E402
from reading_order import page_tilt, reading_order  # noqa: E402

WIDTH = 1650
LINE, STEP = 30, 33  # Line height, and line height plus the gap between lines.
LEFT, RIGHT = (130, 810), (845, 1525)  # Two columns with a 35 px gutter.


def box(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def column(name, span, count, top=200, width_share=1.0):
    """count lines of a column, each (label, box); the last line of a paragraph is shorter."""
    x0, x1 = span
    lines = []
    for n in range(count):
        right = x0 + (x1 - x0) * (0.6 if n == count - 1 else width_share)
        lines.append((f"{name}{n + 1}", box(x0, top + n * STEP, right, top + n * STEP + LINE)))
    return lines


def engine_order(lines):
    """As the engine gives them: top to bottom, then left to right."""
    return sorted(lines, key=lambda line: (line[1][0][1], line[1][0][0]))


def read(lines, width=WIDTH):
    lines = engine_order(lines)
    return [lines[index][0] for index in reading_order(np.array([b for _, b in lines], dtype=float), width)]


class ReadingOrderTests(unittest.TestCase):
    def test_one_column_stays_top_to_bottom(self):
        lines = column("L", (130, 1525), 6)
        self.assertEqual(read(lines), ["L1", "L2", "L3", "L4", "L5", "L6"])

    def test_two_columns_left_then_right(self):
        lines = column("L", LEFT, 5) + column("R", RIGHT, 5)
        self.assertEqual([label for label, _ in engine_order(lines)][:4], ["L1", "R1", "L2", "R2"])  # The bug.
        self.assertEqual(read(lines), ["L1", "L2", "L3", "L4", "L5", "R1", "R2", "R3", "R4", "R5"])

    def test_narrow_gutter(self):
        # 12 px between the columns, as on the vitamin D paper's pages.
        lines = column("L", (130, 819), 4) + column("R", (831, 1525), 4)
        self.assertEqual(read(lines), ["L1", "L2", "L3", "L4", "R1", "R2", "R3", "R4"])

    def test_three_columns(self):
        lines = column("A", (100, 600), 4) + column("B", (630, 1100), 4) + column("C", (1130, 1600), 4)
        self.assertEqual(read(lines), ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4", "C1", "C2", "C3", "C4"])

    def test_title_above_two_columns(self):
        # A title across the gutter, close above the columns (no full-line gap): read first.
        lines = [("T", box(300, 160, 1350, 190))] + column("L", LEFT, 4) + column("R", RIGHT, 4)
        self.assertEqual(read(lines), ["T", "L1", "L2", "L3", "L4", "R1", "R2", "R3", "R4"])

    def test_wide_element_between_columns_keeps_its_place(self):
        # A figure caption across both columns: the columns above it, the caption, the columns below.
        top = column("L", LEFT, 4) + column("R", RIGHT, 4)
        caption = [("CAPTION", box(300, 200 + 4 * STEP, 1350, 200 + 4 * STEP + LINE))]
        below = column("l", LEFT, 4, top=200 + 5 * STEP) + column("r", RIGHT, 4, top=200 + 5 * STEP)
        self.assertEqual(read(top + caption + below),
                         ["L1", "L2", "L3", "L4", "R1", "R2", "R3", "R4", "CAPTION",
                          "l1", "l2", "l3", "l4", "r1", "r2", "r3", "r4"])

    def test_running_header_comes_first(self):
        # A page number on the left and the journal on the right, set off by a gap of two lines.
        header = [("2", box(134, 120, 155, 147)), ("Journal", box(1180, 120, 1525, 147))]
        lines = header + column("L", LEFT, 4, top=210) + column("R", RIGHT, 4, top=210)
        self.assertEqual(read(lines), ["2", "Journal", "L1", "L2", "L3", "L4", "R1", "R2", "R3", "R4"])

    def test_tilted_scan(self):
        # Turned 2 degrees: a line drifts about 50 px across the page, more than the gutter.
        angle = math.radians(2)
        lines = column("L", LEFT, 12) + column("R", RIGHT, 12)
        turned = [(label, [[x * math.cos(angle) - y * math.sin(angle), x * math.sin(angle) + y * math.cos(angle)]
                           for x, y in points]) for label, points in lines]
        self.assertAlmostEqual(page_tilt([points for _, points in turned]), angle, places=6)
        self.assertEqual(read(turned), [f"L{n}" for n in range(1, 13)] + [f"R{n}" for n in range(1, 13)])

    def test_table_is_read_row_by_row(self):
        # Short cells in four columns: not text columns, so each row stays together.
        lines = []
        for row in range(5):
            for col, x in enumerate((130, 520, 900, 1280)):
                lines.append((f"r{row + 1}c{col + 1}", box(x, 200 + row * STEP, x + 120, 200 + row * STEP + LINE)))
        self.assertEqual(read(lines), [f"r{r}c{c}" for r in range(1, 6) for c in range(1, 5)])

    def test_a_table_beside_a_text_column(self):
        # Text on the left, a table on the right (as on page 3 of the migraine paper): the text is
        # read as a column, then the table on its own, row by row.
        table = [(f"r{row + 1}c{col + 1}", box(x, 200 + row * STEP, x + 90, 200 + row * STEP + LINE))
                 for row in range(4) for col, x in enumerate((900, 1200, 1420))]
        self.assertEqual(read(column("L", LEFT, 4) + table),
                         ["L1", "L2", "L3", "L4"] + [f"r{r}c{c}" for r in range(1, 5) for c in range(1, 4)])

    def test_a_table_across_the_page_keeps_the_engine_rows(self):
        # A wide table whose cells leave a gap, with a cell across it: no text columns, so the
        # engine's rows stay as they are.
        lines = [(f"r{row + 1}c{col + 1}", box(x, 200 + row * STEP, x + 90, 200 + row * STEP + LINE))
                 for row in range(5) for col, x in enumerate((130, 500, 900, 1300))]
        lines.append(("wide", box(560, 200 + 5 * STEP, 1000, 200 + 5 * STEP + LINE)))
        self.assertEqual(read(lines), [f"r{r}c{c}" for r in range(1, 6) for c in range(1, 5)] + ["wide"])

    def test_lines_split_into_several_boxes(self):
        # One line found as two boxes: still read left to right, before the next line.
        lines = [("a", box(130, 200, 500, 230)), ("b", box(520, 202, 1500, 231)), ("c", box(130, 233, 1500, 263))]
        self.assertEqual(read(lines), ["a", "b", "c"])

    def test_nothing(self):
        self.assertEqual(reading_order([], WIDTH), [])
        self.assertEqual(reading_order(None, WIDTH), [])
        self.assertEqual(reading_order(np.array([box(10, 10, 100, 40)], dtype=float), WIDTH), [0])


class OCRImageTests(unittest.TestCase):
    """ocr._ocr_image returns the lines in reading order, not the engine's."""

    def setUp(self):
        patch.object(ocr, "_ocr_engine", None).start()
        self.addCleanup(patch.stopall)

    def run_engine(self, boxes, texts):
        class Engine:
            def __init__(self, params=None):
                pass

            def __call__(self, image):
                return SimpleNamespace(boxes=boxes, txts=texts)
        ocr._ocr_engine = None  # A new engine for each call (the module keeps the first one).
        with patch("rapidocr.RapidOCR", Engine):
            return ocr._ocr_image(np.zeros((2339, WIDTH, 3), dtype=np.uint8))

    def test_columns(self):
        lines = engine_order(column("L", LEFT, 3) + column("R", RIGHT, 3))
        text = self.run_engine(np.array([b for _, b in lines], dtype=float), tuple(label for label, _ in lines))
        self.assertEqual(text, "L1\nL2\nL3\nR1\nR2\nR3")

    def test_without_boxes_the_engine_order_is_kept(self):
        self.assertEqual(self.run_engine(None, ("one", "two")), "one\ntwo")
        self.assertEqual(self.run_engine(None, None), "")


if __name__ == "__main__":
    unittest.main()
