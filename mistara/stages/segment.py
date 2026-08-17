"""Classical line segmentation — zero model weights, fully deterministic.

Used two ways: as the geometry that anchors VLM transcription (so we can detect
silent line omission), and later in S2 as an independent cross-check against the
hosted layout model.

Nastaliq complicates this — descenders from one line overlap the next, and
baselines slope — so the profile is smoothed and bands are merged conservatively.
This is a baseline, not the final layout analyser.
"""

from __future__ import annotations

import cv2
import numpy as np
from pydantic import BaseModel

from mistara.core.config import DEFAULT, SegmentConfig
from mistara.core.model import BBox, Issue, IssueKind


def binarize(gray: np.ndarray, config: SegmentConfig | None = None) -> np.ndarray:
    """Adaptive threshold to ink=255 / background=0."""
    cfg = (config or DEFAULT.segment).binarize
    block = cfg.block_size if cfg.block_size % 2 else cfg.block_size + 1
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, cfg.c
    )


def ink_ratio(gray: np.ndarray, config: SegmentConfig | None = None) -> float:
    return float((binarize(gray, config) > 0).mean())


def _smooth(profile: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return profile
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(profile, kernel, mode="same")


def detect_text_rows(
    gray: np.ndarray, config: SegmentConfig | None = None
) -> list[BBox]:
    """Find text lines by horizontal projection of ink.

    Returns boxes tightened to the actual horizontal ink extent of each band, so
    a short marginal note does not get a full-width box.
    """
    if gray.ndim != 2:
        raise ValueError("detect_text_rows expects a single-channel image")

    cfg = config or DEFAULT.segment
    lines = cfg.lines
    binary = binarize(gray, cfg)
    profile = _smooth((binary > 0).sum(axis=1).astype(float), lines.smooth_window)
    if profile.max() <= 0:
        return []

    threshold = max(1.0, float(profile[profile > 0].mean()) * lines.threshold_scale)

    bands: list[tuple[int, int]] = []
    start: int | None = None
    gap = 0
    for y, value in enumerate(profile):
        if value >= threshold:
            if start is None:
                start = y
            gap = 0
        elif start is not None:
            gap += 1
            if gap >= lines.min_gap_px:
                bands.append((start, y - gap))
                start = None
    if start is not None:
        bands.append((start, len(profile) - 1))

    boxes: list[BBox] = []
    height, width = binary.shape
    for y0, y1 in bands:
        if y1 - y0 + 1 < lines.min_height_px:
            continue
        strip = binary[y0 : y1 + 1]
        cols = np.flatnonzero((strip > 0).sum(axis=0) > 0)
        if cols.size == 0:
            continue
        x0, x1 = int(cols[0]), int(cols[-1])
        boxes.append(
            BBox(
                x=float(x0),
                y=float(y0),
                w=float(max(1, x1 - x0 + 1)),
                h=float(max(1, y1 - y0 + 1)),
            )
        )

    # Guard against a pathological page where every row crosses threshold.
    if len(boxes) > height // max(lines.min_height_px, 1):
        return []
    return boxes


def estimate_line_height(boxes: list[BBox]) -> float:
    if not boxes:
        return 0.0
    return float(np.median([b.h for b in boxes]))


# --------------------------------------------------------------------------- #
# Side-by-side columns
# --------------------------------------------------------------------------- #


class ColumnBlock(BaseModel):
    """A run of consecutive rows split by a shared vertical whitespace gutter.

    These books set Arabic and its Urdu translation side by side — Arabic in the
    right cell, Urdu in the left — and also use the same shape for tables. A
    horizontal projection cannot see this: each such row is a normal-height,
    regularly-pitched box, so it looks perfectly healthy by every vertical
    statistic while actually holding two different texts in two languages.

    What gives it away is a column of near-zero ink running through the middle
    of several *consecutive* rows at the same x. One row might have a wide word
    gap by chance; four rows sharing a gutter to within a few pixels do not.
    """

    row_indices: list[int]
    gutter_x: float
    gutter_width: float
    y0: float
    y1: float

    @property
    def rows(self) -> int:
        return len(self.row_indices)


def _widest_interior_gutter(
    binary: np.ndarray, box: BBox, *, margin_px: int
) -> tuple[float, float] | None:
    """Return the widest interior ink-free run as an (x_start, x_end) interval.

    An interval rather than a centre: gutter widths vary a lot down a block
    (one row's cells may nearly touch while another's are far apart), and a wide
    gutter's centre sits far from a narrow one's even when they plainly belong to
    the same column boundary. Overlap is stable where centre distance is not.
    """
    y0, y1 = int(box.y), int(np.ceil(box.y + box.h))
    x0, x1 = int(box.x), int(np.ceil(box.x + box.w))
    strip = binary[y0:y1, x0:x1]
    if strip.size == 0:
        return None

    columns = (strip > 0).sum(axis=0)
    best: tuple[float, float] | None = None
    best_w = 0.0
    start: int | None = None
    for i, value in enumerate(columns):
        if value == 0:
            if start is None:
                start = i
        elif start is not None:
            width = i - start
            # Ignore runs touching either edge: those are page margins, not
            # gutters between two cells of the same row.
            if start > margin_px and i < len(columns) - margin_px and width > best_w:
                best_w, best = float(width), (float(x0 + start), float(x0 + i))
            start = None
    return best


def detect_column_blocks(
    gray: np.ndarray, boxes: list[BBox], config: SegmentConfig | None = None
) -> list[ColumnBlock]:
    """Find runs of consecutive rows sharing a vertical gutter."""
    cfg = config or DEFAULT.segment
    col = cfg.columns
    if len(boxes) < col.min_rows:
        return []

    binary = binarize(gray, cfg)
    ordered = sorted(boxes, key=lambda b: b.y)
    median_h = float(np.median([b.h for b in ordered]))

    gutters = [
        _widest_interior_gutter(binary, b, margin_px=col.edge_margin_px)
        for b in ordered
    ]
    widths = np.array([(g[1] - g[0]) if g else 0.0 for g in gutters], dtype=float)
    # A page's typical inter-word gap sets the baseline; a real gutter has to be
    # several times that, and also wide relative to the line height.
    typical = float(np.median(widths)) if widths.size else 0.0
    threshold = max(
        col.min_gutter_ratio * max(typical, 1.0),
        col.min_gutter_frac_of_height * median_h,
    )

    blocks: list[ColumnBlock] = []
    run: list[int] = []

    def flush() -> None:
        if len(run) >= col.min_rows:
            # The shared boundary is where every row's gutter agrees, so cut at
            # the centre of their intersection rather than any one row's middle.
            lo = max(gutters[i][0] for i in run)  # type: ignore[index]
            hi = min(gutters[i][1] for i in run)  # type: ignore[index]
            blocks.append(
                ColumnBlock(
                    row_indices=list(run),
                    gutter_x=float((lo + hi) / 2),
                    gutter_width=float(np.median(widths[run])),
                    y0=min(ordered[i].y for i in run),
                    y1=max(ordered[i].y + ordered[i].h for i in run),
                )
            )
        run.clear()

    for i, gutter in enumerate(gutters):
        if gutter is None or widths[i] < threshold:
            flush()
            continue
        if run:
            prev = gutters[run[-1]]
            assert prev is not None
            overlap = min(prev[1], gutter[1]) - max(prev[0], gutter[0])
            if overlap < col.align_overlap_px:
                flush()
        run.append(i)
    flush()
    return blocks


def split_rows_at_gutters(
    gray: np.ndarray,
    boxes: list[BBox],
    blocks: list[ColumnBlock] | None = None,
    config: SegmentConfig | None = None,
) -> list[BBox]:
    """Split multi-column rows into per-cell boxes.

    Cells are emitted right-to-left within a row, matching how these pages read:
    the Arabic sits in the right cell and its Urdu translation in the left.
    """
    cfg = config or DEFAULT.segment
    ordered = sorted(boxes, key=lambda b: b.y)
    blocks = detect_column_blocks(gray, ordered, cfg) if blocks is None else blocks
    if not blocks:
        return ordered

    binary = binarize(gray, cfg)
    split_at = {i: b.gutter_x for b in blocks for i in b.row_indices}

    out: list[BBox] = []
    for i, box in enumerate(ordered):
        gutter = split_at.get(i)
        if gutter is None or not (box.x < gutter < box.x + box.w):
            out.append(box)
            continue
        for x0, x1 in ((gutter, box.x + box.w), (box.x, gutter)):  # right cell first
            cell = _tighten(binary, BBox(x=x0, y=box.y, w=x1 - x0, h=box.h))
            if cell is not None:
                out.append(cell)
    return out


def _tighten(binary: np.ndarray, box: BBox) -> BBox | None:
    """Shrink a box to the ink it actually contains; None if it holds none."""
    y0, y1 = int(box.y), int(np.ceil(box.y + box.h))
    x0, x1 = int(box.x), int(np.ceil(box.x + box.w))
    region = binary[y0:y1, x0:x1] > 0
    if not region.any():
        return None
    rows = np.flatnonzero(region.any(axis=1))
    cols = np.flatnonzero(region.any(axis=0))
    return BBox(
        x=float(x0 + cols[0]),
        y=float(y0 + rows[0]),
        w=float(cols[-1] - cols[0] + 1),
        h=float(rows[-1] - rows[0] + 1),
    )


# --------------------------------------------------------------------------- #
# Assessing the segmentation — how we know it went wrong, without ground truth
# --------------------------------------------------------------------------- #


class RowAssessment(BaseModel):
    """Evidence that line segmentation missed or merged lines.

    The leverage here comes from typesetting: a printed book is set on a regular
    baseline grid, so line pitch is nearly constant down a column. Both failure
    modes break that regularity in *characteristic* ways, which makes them
    detectable with no ground truth at all:

    * a **merged** pair leaves one box roughly twice the median height;
    * a **missed** line leaves a gap roughly twice the median pitch, and — the
      decisive part — leaves ink stranded inside that gap.

    The ink test is what makes gap detection precise. A page's header/body gap
    is also ~2x pitch but contains no stranded ink, so requiring both conditions
    suppresses that false positive.

    Note that ink coverage alone is a **bad** objective: it is maximized by
    emitting one box for the whole page. Regularity is the thing to optimize;
    coverage is only meaningful alongside a line count.
    """

    rows: int
    median_height: float
    median_pitch: float
    pitch_cv: float  # coefficient of variation — lower is more regular
    ink_covered: float  # share of ink inside some box
    tall_boxes: int
    stranded_gaps: int
    est_merged_lines: int
    est_missed_lines: int
    #: Rows that actually hold two or more side-by-side cells. Invisible to
    #: every vertical statistic above — see ColumnBlock.
    column_blocks: int = 0
    multi_column_rows: int = 0
    #: Carried so findings() is self-contained rather than reaching for config.
    max_pitch_cv: float = 0.35
    #: Where the suspicions are, so they can be drawn rather than merely counted.
    merged_boxes: list[BBox] = []
    missed_gaps: list[BBox] = []
    gutters: list[BBox] = []

    def issues(self) -> list[Issue]:
        out = [
            Issue(
                kind=IssueKind.MERGED_LINES,
                bbox=b,
                note="box height implies more than one line",
            )
            for b in self.merged_boxes
        ]
        out += [
            Issue(
                kind=IssueKind.MISSED_LINE,
                bbox=b,
                note="ink here belongs to no detected line",
            )
            for b in self.missed_gaps
        ]
        out += [
            Issue(
                kind=IssueKind.COLUMN_GUTTER,
                bbox=b,
                note="rows split here into side-by-side cells",
            )
            for b in self.gutters
        ]
        return out

    @property
    def suspect_lines(self) -> int:
        return self.est_merged_lines + self.est_missed_lines + self.multi_column_rows

    def findings(self) -> list[str]:
        out: list[str] = []
        if self.est_merged_lines:
            out.append(
                f"{self.est_merged_lines} line(s) look merged "
                f"({self.tall_boxes} box(es) over 1.6x median height)"
            )
        if self.est_missed_lines:
            out.append(
                f"{self.est_missed_lines} line(s) look missed "
                f"({self.stranded_gaps} gap(s) hold unassigned ink)"
            )
        if self.multi_column_rows:
            out.append(
                f"{self.multi_column_rows} row(s) span {self.column_blocks} "
                f"side-by-side column block(s) and need splitting"
            )
        if self.pitch_cv > self.max_pitch_cv:
            out.append(
                f"line pitch is irregular (cv {self.pitch_cv:.2f}) — segmentation "
                f"may not be tracking the baseline grid"
            )
        return out


def assess_rows(
    gray: np.ndarray, boxes: list[BBox], config: SegmentConfig | None = None
) -> RowAssessment | None:
    """Judge a row segmentation against the page's own baseline regularity."""
    cfg = config or DEFAULT.segment
    a = cfg.assess
    if len(boxes) < a.min_rows_for_assessment:
        return None

    binary = binarize(gray, cfg)
    ink = binary > 0
    ordered = sorted(boxes, key=lambda b: b.y)

    heights = np.array([b.h for b in ordered], dtype=float)
    centres = np.array([b.y + b.h / 2 for b in ordered], dtype=float)
    pitch = np.diff(centres)
    median_h = float(np.median(heights))
    median_pitch = float(np.median(pitch)) if pitch.size else 0.0
    pitch_cv = float(pitch.std() / pitch.mean()) if pitch.size and pitch.mean() else 0.0

    covered = np.zeros_like(ink)
    row_ink: list[int] = []
    for b in ordered:
        y0, y1 = int(b.y), int(np.ceil(b.y + b.h))
        x0, x1 = int(b.x), int(np.ceil(b.x + b.w))
        covered[y0:y1, x0:x1] = True
        row_ink.append(int(ink[y0:y1, x0:x1].sum()))
    total_ink = int(ink.sum())
    ink_covered = float((ink & covered).sum()) / total_ink if total_ink else 1.0
    typical_row_ink = float(np.median(row_ink)) if row_ink else 0.0

    # -- merges ---------------------------------------------------------------
    # A box holding n lines is not n x the line height — it also swallows the
    # (n-1) gaps between them, so its height is (n-1)*pitch + line_height.
    # Inverting that is what gives the right count; dividing by height alone
    # over-counts (a 2-line box is ~3x a line's height, not 2x).
    tall = [b for b in ordered if b.h > a.tall_factor * median_h]
    est_merged = 0
    merged_boxes: list[BBox] = []
    for b in tall:
        if median_pitch <= 0:
            continue
        lines_in_box = int(round((b.h - median_h) / median_pitch)) + 1
        if lines_in_box > 1:
            merged_boxes.append(b)
        est_merged += max(0, lines_in_box - 1)

    # -- misses: an outsized gap that still holds ink -------------------------
    stranded_gaps = 0
    est_missed = 0
    missed_gaps: list[BBox] = []
    for i in range(1, len(ordered)):
        gap = pitch[i - 1]
        if median_pitch <= 0 or gap <= a.gap_factor * median_pitch:
            continue
        y0 = int(ordered[i - 1].y + ordered[i - 1].h)
        y1 = int(ordered[i].y)
        if y1 <= y0:
            continue
        gap_ink = int(ink[y0:y1].sum())
        if typical_row_ink > 0 and gap_ink > a.stranded_ink_frac * typical_row_ink:
            stranded_gaps += 1
            est_missed += max(1, int(round(gap / median_pitch)) - 1)
            cols = np.flatnonzero(ink[y0:y1].any(axis=0))
            if cols.size:
                missed_gaps.append(
                    BBox(
                        x=float(cols[0]),
                        y=float(y0),
                        w=float(cols[-1] - cols[0] + 1),
                        h=float(y1 - y0),
                    )
                )

    blocks = detect_column_blocks(gray, ordered, cfg)

    return RowAssessment(
        rows=len(ordered),
        max_pitch_cv=a.max_pitch_cv,
        column_blocks=len(blocks),
        multi_column_rows=sum(b.rows for b in blocks),
        merged_boxes=merged_boxes,
        missed_gaps=missed_gaps,
        gutters=[
            BBox(x=b.gutter_x - 1, y=b.y0, w=2.0, h=b.y1 - b.y0) for b in blocks
        ],
        median_height=median_h,
        median_pitch=median_pitch,
        pitch_cv=pitch_cv,
        ink_covered=ink_covered,
        tall_boxes=len(tall),
        stranded_gaps=stranded_gaps,
        est_merged_lines=est_merged,
        est_missed_lines=est_missed,
    )
