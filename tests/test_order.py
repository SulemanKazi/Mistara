"""S3 reading order.

Geometry here is synthetic on purpose: the point of these tests is that the
right answer is known by construction, which it never is on a real scan.
"""

from __future__ import annotations

import pytest

from mistara.core.config import OrderConfig, SegmentConfig
from mistara.core.model import BBox, Language, Line, Provenance, Span
from mistara.stages.s3_order import (
    COLUMN_MAJOR,
    ROW_MAJOR,
    _bands,
    _gutters,
    group_page,
)

CFG = OrderConfig()
SEG = SegmentConfig()


def line(
    line_id: str,
    x: float,
    y: float,
    w: float,
    h: float = 30,
    text: str = "متن",
    lang: Language = Language.UNKNOWN,
) -> Line:
    return Line(
        line_id=line_id,
        bbox=BBox(x=x, y=y, w=w, h=h),
        text=text,
        spans=[
            Span(
                span_id=f"{line_id}s0",
                char_start=0,
                char_end=len(text),
                language=lang,
                provenance=Provenance(),
            )
        ],
    )


def two_column_page(rows: int = 3) -> list[Line]:
    """Arabic in the right cell, Urdu in the left — the shape sample 2 exhibits."""
    out: list[Line] = []
    for i in range(rows):
        y = 100 + i * 40
        out.append(line(f"R{i}", x=320, y=y, w=260, lang=Language.ARABIC))
        out.append(line(f"L{i}", x=20, y=y, w=260, lang=Language.URDU))
    return out


# --------------------------------------------------------------------------- #
# Banding
# --------------------------------------------------------------------------- #


def test_bands_group_cells_of_one_row_and_order_them_rtl():
    bands = _bands(two_column_page(rows=3), CFG.band_overlap_frac)
    assert len(bands) == 3
    for band in bands:
        assert [ln.line_id[0] for ln in band] == ["R", "L"]  # rightmost first


def test_bands_tolerate_the_y_jitter_real_cells_have():
    # Splitting a row tightens each cell to its own ink, so the two halves of one
    # row rarely share a y to the pixel.
    lines = [line("R", x=320, y=100, w=260, h=30), line("L", x=20, y=104, w=260, h=26)]
    assert len(_bands(lines, CFG.band_overlap_frac)) == 1


def test_separate_rows_do_not_merge():
    lines = [line("a", x=20, y=100, w=560), line("b", x=20, y=140, w=560)]
    assert len(_bands(lines, CFG.band_overlap_frac)) == 2


def test_gutters_reject_horizontally_overlapping_cells():
    overlapping = [line("a", x=20, y=100, w=300), line("b", x=200, y=100, w=300)]
    assert _gutters(overlapping) is None


# --------------------------------------------------------------------------- #
# Grouping and order
# --------------------------------------------------------------------------- #


def test_column_major_reads_each_column_to_its_end_right_to_left():
    groups = group_page(two_column_page(rows=3), CFG, SEG)
    assert len(groups) == 2
    assert [ln.line_id for ln in groups[0].lines] == ["R0", "R1", "R2"]
    assert [ln.line_id for ln in groups[1].lines] == ["L0", "L1", "L2"]
    assert groups[0].column_index == 0  # 0 is the rightmost column
    assert all(g.columns == 2 for g in groups)


def test_row_major_keeps_the_interleaving_but_still_marks_the_block():
    cfg = CFG.model_copy(update={"column_read_order": ROW_MAJOR})
    groups = group_page(two_column_page(rows=3), cfg, SEG)
    assert len(groups) == 1
    assert [ln.line_id for ln in groups[0].lines] == [
        "R0", "L0", "R1", "L1", "R2", "L2",
    ]
    assert groups[0].columns == 2


def test_prose_stays_one_region_in_page_order():
    lines = [line(f"p{i}", x=20, y=100 + i * 40, w=560) for i in range(4)]
    groups = group_page(lines, CFG, SEG)
    assert len(groups) == 1
    assert [ln.line_id for ln in groups[0].lines] == ["p0", "p1", "p2", "p3"]
    assert groups[0].columns == 1


def test_prose_above_and_below_a_block_stays_separate_from_it():
    lines = [line("head", x=20, y=40, w=560)]
    lines += two_column_page(rows=3)
    lines += [line("foot", x=20, y=260, w=560)]
    groups = group_page(lines, CFG, SEG)
    # header · right column · left column · footer
    assert [len(g.lines) for g in groups] == [1, 3, 3, 1]
    assert groups[0].lines[0].line_id == "head"
    assert groups[-1].lines[0].line_id == "foot"


def test_a_run_shorter_than_min_rows_is_not_a_column_block():
    # One row with a wide word gap is not evidence of a column; several rows
    # agreeing on the same gap are.
    groups = group_page(two_column_page(rows=1), CFG, SEG)
    assert len(groups) == 1
    assert groups[0].columns == 1


def test_misaligned_gutters_do_not_form_a_block():
    lines = [
        line("R0", x=320, y=100, w=260),
        line("L0", x=20, y=100, w=260),
        line("R1", x=180, y=140, w=400),  # gutter somewhere else entirely
        line("L1", x=20, y=140, w=120),
    ]
    groups = group_page(lines, CFG, SEG)
    assert len(groups) == 1
    assert groups[0].columns == 1


def test_a_narrow_column_is_marginalia_and_reads_after_the_body():
    lines: list[Line] = []
    for i in range(3):
        y = 100 + i * 40
        lines.append(line(f"note{i}", x=520, y=y, w=60))  # narrow, on the right
        lines.append(line(f"body{i}", x=20, y=y, w=460))
    groups = group_page(lines, CFG, SEG)
    assert len(groups) == 2
    # Body reads first even though the note sits to its right.
    assert [ln.line_id for ln in groups[0].lines] == ["body0", "body1", "body2"]
    assert groups[0].marginalia is False
    assert groups[1].marginalia is True


# --------------------------------------------------------------------------- #
# The invariant
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("read_order", [COLUMN_MAJOR, ROW_MAJOR])
def test_grouping_is_a_permutation_of_its_input(read_order):
    """Nothing lost, nothing duplicated — the check the verifier enforces."""
    lines = [line("head", x=20, y=40, w=560)] + two_column_page(rows=4)
    cfg = CFG.model_copy(update={"column_read_order": read_order})
    grouped = [ln.line_id for g in group_page(lines, cfg, SEG) for ln in g.lines]
    assert sorted(grouped) == sorted(ln.line_id for ln in lines)


def test_column_purity_flags_a_column_holding_two_languages():
    clean = group_page(two_column_page(rows=3), CFG, SEG)
    assert clean[0].purity() == 1.0

    mixed = two_column_page(rows=3)
    # One Urdu line stranded in the Arabic column: what a misplaced gutter does.
    mixed[0] = line("R0", x=320, y=100, w=260, lang=Language.URDU)
    groups = group_page(mixed, CFG, SEG)
    assert groups[0].purity() == pytest.approx(2 / 3)


def test_empty_page_yields_no_groups():
    assert group_page([], CFG, SEG) == []
