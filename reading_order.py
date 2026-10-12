"""Reading order of OCR lines on a page, column by column.

The OCR engine finds one box per line and returns them from top to bottom, so on a two-column
page a line of the left column is followed by the line beside it in the right column ("MS-Q has a
hypothesized that serotonin..."). This puts the boxes back in reading order from their
coordinates (a recursive XY-cut):

1. The page's tilt is measured from the boxes' top edges and the coordinates are turned back
   (a scan tilted 2 degrees shifts a line by more than a column gap from top to bottom).
2. A region is cut across where a gap with no box runs the full width and is at least a line
   high: a running header, a title block and the body come apart.
3. Otherwise a gutter is looked for: a vertical strip, away from the edges, that (almost) no box
   covers. The few boxes that cross it (a centred heading, a figure caption, a page number) are
   wide elements: they stay where they are and cut the columns into bands above and below them.
4. In a band, the left side is read before the right one, if both sides look like text columns
   (3 lines or more, most of them filling the column). Otherwise (a table, figure labels) the
   band is read row by row, as the engine did.
5. Each side is a region again, so three columns and columns inside a part are handled the same way.

Thresholds come from the pages measured (eval/ocr_benchmark.md): at 200 DPI a line is about 30 px
high, gutters were 12 to 37 px wide (0.7% to 2.2% of the page), and gaps between lines 1 to 3 px.
Not handled: text in irregular frames (wrapped around a figure), rotated pages, right-to-left
scripts (OCR is English only).
"""

import math
from dataclasses import dataclass
from statistics import median

# A full-width gap at least this many line heights tall cuts a region (gaps between lines are
# about 0.1 of a line; a header or a title block is set off by one line or more).
CUT_GAP_LINES = 1.0
# A gutter is at least this wide, as a share of the page width, and lies between these shares of
# the region's width (a margin is not a gutter).
MIN_GUTTER = 0.005
GUTTER_SPAN = (0.2, 0.8)
# At most this share of a region's boxes, and at least one box, may cross a gutter (they become
# wide elements; a gutter only counts if text columns lie on both sides of it).
MAX_CROSSING = 0.1
# A side is a text column if it has this many lines, its median line fills this share of it, and
# that line is at least this share of the page wide (text lines are about 0.4 of the page in two
# columns and 0.3 in three; table cells and figure labels are much narrower).
MIN_COLUMN_LINES = 3
MIN_COLUMN_FILL = 0.5
MIN_LINE_WIDTH = 0.15
# Boxes this much wider than tall measure the page's tilt (short boxes give noisy angles).
TILT_MIN_ASPECT = 3.0


@dataclass(frozen=True)
class _Box:
    index: int
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def width(self) -> float:
        return self.x1 - self.x0


def page_tilt(boxes) -> float:
    """The page's tilt in radians: the median angle of the long boxes' top edges (0 if none)."""
    angles = []
    for points in boxes:
        (left_x, left_y), (right_x, right_y) = points[0], points[1]
        xs, ys = [point[0] for point in points], [point[1] for point in points]
        if (max(xs) - min(xs)) >= TILT_MIN_ASPECT * max(max(ys) - min(ys), 1):
            angles.append(math.atan2(right_y - left_y, right_x - left_x))
    return median(angles) if angles else 0.0


def _upright(boxes) -> list[_Box]:
    """Each box's bounds after turning the page back by its tilt."""
    tilt = page_tilt(boxes)
    cos, sin = math.cos(-tilt), math.sin(-tilt)
    result = []
    for index, points in enumerate(boxes):
        turned = [(x * cos - y * sin, x * sin + y * cos) for x, y in points]
        xs, ys = [x for x, _ in turned], [y for _, y in turned]
        result.append(_Box(index, min(xs), min(ys), max(xs), max(ys)))
    return result


def _rows(boxes: list[_Box]) -> list[_Box]:
    """Row by row, as the engine listed them: its own grouping of lines into rows keeps a table's
    tall header cells better than grouping by box heights did (eval/ocr_benchmark.md, page 3 of
    the vitamin D paper)."""
    return sorted(boxes, key=lambda box: box.index)


def _cuts(boxes: list[_Box], line_height: float) -> list[list[_Box]]:
    """The region split at full-width gaps of CUT_GAP_LINES lines or more, top to bottom."""
    parts, current, bottom = [], [], None
    for box in sorted(boxes, key=lambda box: box.y0):
        if current and box.y0 - bottom >= CUT_GAP_LINES * line_height:
            parts.append(current)
            current = []
        current.append(box)
        bottom = box.y1 if bottom is None or not current[:-1] else max(bottom, box.y1)
    return parts + [current]


def _gutters(boxes: list[_Box], page_width: float) -> list[tuple[float, float]]:
    """(left, right) of every strip in the region's middle that no more than MAX_CROSSING of its
    boxes cover, at least MIN_GUTTER of the page wide."""
    left, right = min(box.x0 for box in boxes), max(box.x1 for box in boxes)
    low, high = left + GUTTER_SPAN[0] * (right - left), left + GUTTER_SPAN[1] * (right - left)
    allowed = max(1, int(MAX_CROSSING * len(boxes)))
    # Coverage changes only at box edges: count the boxes over each interval between edges.
    edges = sorted({low, high, *(x for box in boxes for x in (box.x0, box.x1) if low < x < high)})
    runs, start_of_run = [], None
    for start, end in zip(edges, edges[1:]):
        middle = (start + end) / 2
        if sum(box.x0 < middle < box.x1 for box in boxes) <= allowed:
            start_of_run = start if start_of_run is None else start_of_run
            if end == edges[-1]:
                runs.append((start_of_run, end))
        elif start_of_run is not None:
            runs.append((start_of_run, start))
            start_of_run = None
    return [run for run in runs if run[1] - run[0] >= MIN_GUTTER * page_width]


def _sides(boxes: list[_Box], gutter: tuple[float, float]) -> tuple[list[_Box], list[_Box], float]:
    """The boxes left and right of the gutter (those crossing it are on neither side), and its middle."""
    middle = sum(gutter) / 2
    return [box for box in boxes if box.x1 <= middle], [box for box in boxes if box.x0 >= middle], middle


def _best_gutter(boxes: list[_Box], page_width: float) -> tuple[float, float] | None:
    """The gutter with text columns on both sides, else on one side (a table beside text: the gap
    between two table columns can be wider than the one beside the text), else the widest."""
    def score(gutter):
        left, right, _ = _sides(boxes, gutter)
        return _is_text(left, page_width) + _is_text(right, page_width), gutter[1] - gutter[0]
    candidates = _gutters(boxes, page_width)
    return max(candidates, key=score) if candidates else None


def _is_text(boxes: list[_Box], page_width: float) -> bool:
    """A text column (MIN_COLUMN_LINES lines or more, most of them wide and filling it), or
    columns of text side by side (the right side of a three-column page holds two of them)."""
    if len(boxes) < MIN_COLUMN_LINES:
        return False
    width = max(box.x1 for box in boxes) - min(box.x0 for box in boxes)
    line = median(box.width for box in boxes)
    if line < MIN_LINE_WIDTH * page_width:
        return False
    if line >= MIN_COLUMN_FILL * width:
        return True
    for gutter in _gutters(boxes, page_width):
        left, right, _ = _sides(boxes, gutter)
        if _is_text(left, page_width) and _is_text(right, page_width):
            return True
    return False


def _order(boxes: list[_Box], line_height: float, page_width: float, stats: dict) -> list[_Box]:
    if len(boxes) <= 1:
        return list(boxes)
    parts = _cuts(boxes, line_height)
    if len(parts) > 1:
        return [box for part in parts for box in _order(part, line_height, page_width, stats)]
    gutter = _best_gutter(boxes, page_width)
    if gutter is None:
        return _rows(boxes)
    middle = sum(gutter) / 2
    crossing = [box for box in boxes if box.x0 < middle < box.x1]
    stats["crossing"] = stats.get("crossing", 0) + len(crossing)
    # Bands between the boxes that cross the gutter; each crossing box stays between its bands.
    ordered, band, columns = [], [], False
    for box in sorted(boxes, key=lambda box: box.y0) + [None]:
        if box is None or box in crossing:
            band_order, in_columns = _band(band, middle, line_height, page_width, stats)
            ordered += band_order + ([box] if box else [])
            columns, band = columns or in_columns, []
        else:
            band.append(box)
    # No band holds text columns (a table across the page, whose cells leave a gap): the engine's
    # rows, since moving the cells that cross the gap would only reshuffle the table.
    return ordered if columns else _rows(boxes)


def _band(boxes: list[_Box], middle: float, line_height: float, page_width: float,
          stats: dict) -> tuple[list[_Box], bool]:
    """(the band in reading order, whether it was read as columns)."""
    left = [box for box in boxes if box.x1 <= middle]
    right = [box for box in boxes if box.x0 >= middle]
    if not left or not right:
        return _order(left or right, line_height, page_width, stats), False
    if not (_is_text(left, page_width) or _is_text(right, page_width)):
        stats["row_bands"] = stats.get("row_bands", 0) + 1
        return _rows(boxes), False  # A table or figure labels on both sides: rows keep their cells together.
    # Text on at least one side: the left side, then the right one (a table beside a text column is
    # read on its own, in rows, after or before the text).
    stats["column_bands"] = stats.get("column_bands", 0) + 1
    return _order(left, line_height, page_width, stats) + _order(right, line_height, page_width, stats), True


def reading_order(boxes, page_width: float, stats: dict | None = None) -> list[int]:
    """The indexes of the OCR boxes in reading order. boxes: one per line, 4 corner points each
    ([top-left, top-right, bottom-right, bottom-left], as RapidOCR gives them); page_width in the
    same pixels. stats, if given, counts what was found (for eval/ocr_benchmark.py): boxes crossing
    a gutter, bands read column by column, and bands read row by row."""
    if boxes is None or len(boxes) == 0:
        return []
    upright = _upright(boxes)
    line_height = median(box.height for box in upright)
    return [box.index for box in _order(upright, line_height, page_width, {} if stats is None else stats)]
