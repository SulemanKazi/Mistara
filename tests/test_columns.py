"""Side-by-side columns: Arabic/Urdu pairs, tables, and margin notes.

This failure mode is invisible to every vertical statistic — such a row has
normal height and sits on the regular pitch, so the row assessment calls it
healthy while it actually holds two texts in two languages. The giveaway is
horizontal: an ink-free gutter running through several *consecutive* rows at the
same x.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from mistara.core.model import IssueKind
from mistara.core.store import Store
from mistara.stages.s0_ingest import IngestParams, ingest_document, load_view_array
from mistara.stages.segment import (
    assess_rows,
    detect_column_blocks,
    detect_text_rows,
    split_rows_at_gutters,
)

SAMPLE2 = Path("data/sample_pdf_2_columns.pdf")
needs_columns = pytest.mark.skipif(
    not SAMPLE2.exists(), reason="two-column sample not present"
)

PITCH, LINE_H, WIDTH = 40, 20, 600
GUTTER_X = 300


def two_column_page(rows: int = 5, *, total_rows: int = 12) -> np.ndarray:
    """`rows` side-by-side rows on top, then single-column prose beneath."""
    img = np.full((PITCH * (total_rows + 1), WIDTH), 255, np.uint8)
    for i in range(total_rows):
        y = PITCH * (i + 1)
        if i < rows:  # two cells with a clear gutter between them
            cv2.rectangle(img, (20, y), (GUTTER_X - 40, y + LINE_H), 0, -1)
            cv2.rectangle(img, (GUTTER_X + 40, y), (WIDTH - 20, y + LINE_H), 0, -1)
        else:
            cv2.rectangle(img, (20, y), (WIDTH - 20, y + LINE_H), 0, -1)
    return img


class TestDetection:
    def test_a_two_column_block_is_found(self):
        img = two_column_page(rows=5)
        blocks = detect_column_blocks(img, detect_text_rows(img))
        assert len(blocks) == 1
        assert blocks[0].rows == 5
        assert blocks[0].gutter_x == pytest.approx(GUTTER_X, abs=15)

    def test_single_column_prose_yields_no_blocks(self):
        img = two_column_page(rows=0)
        assert detect_column_blocks(img, detect_text_rows(img)) == []

    def test_one_isolated_wide_gap_is_not_a_column(self):
        """A single row with a big word gap must not be mistaken for a table.

        Requiring the gutter to repeat across consecutive rows is what makes the
        signal trustworthy — one row can gap by chance, several cannot.
        """
        img = two_column_page(rows=1)
        assert detect_column_blocks(img, detect_text_rows(img)) == []

    def test_assessment_reports_multi_column_rows(self):
        img = two_column_page(rows=4)
        a = assess_rows(img, detect_text_rows(img))
        assert a.column_blocks == 1
        assert a.multi_column_rows == 4
        assert any("side-by-side" in f for f in a.findings())
        # Vertical statistics alone see nothing wrong — that is the whole point.
        assert a.est_merged_lines == 0
        assert a.est_missed_lines == 0


class TestSplitting:
    def test_split_adds_one_box_per_extra_cell(self):
        img = two_column_page(rows=5, total_rows=12)
        rows = detect_text_rows(img)
        cells = split_rows_at_gutters(img, rows)
        assert len(cells) == len(rows) + 5

    def test_cells_are_emitted_right_to_left(self):
        """RTL reading order: the Arabic cell precedes its Urdu translation."""
        img = two_column_page(rows=3)
        rows = sorted(detect_text_rows(img), key=lambda b: b.y)
        cells = split_rows_at_gutters(img, rows)
        first_row_cells = [c for c in cells if abs(c.y - rows[0].y) < LINE_H]
        assert len(first_row_cells) == 2
        assert first_row_cells[0].x > first_row_cells[1].x

    def test_cells_stay_inside_their_row_and_hold_ink(self):
        img = two_column_page(rows=4)
        rows = sorted(detect_text_rows(img), key=lambda b: b.y)
        for cell in split_rows_at_gutters(img, rows):
            assert any(
                r.y - 2 <= cell.y and cell.y + cell.h <= r.y + r.h + 2 for r in rows
            )
            patch = img[
                int(cell.y) : int(cell.y + cell.h), int(cell.x) : int(cell.x + cell.w)
            ]
            assert (patch < 128).any(), "a split cell must contain ink"

    def test_splitting_is_idempotent(self):
        img = two_column_page(rows=4)
        once = split_rows_at_gutters(img, detect_text_rows(img))
        twice = split_rows_at_gutters(img, once)
        assert len(once) == len(twice)


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    return Store(tmp_path_factory.mktemp("store"))


@needs_columns
class TestOnTheRealSamples:
    def test_arabic_urdu_pairs_are_detected_mid_page(self, store):
        doc, _ = ingest_document(SAMPLE2, store, IngestParams())
        gray = load_view_array(store, doc.page(2))
        blocks = detect_column_blocks(gray, detect_text_rows(gray))
        width = gray.shape[1]
        # One gutter near the middle (the Arabic/Urdu table) and one near the
        # outer margin (the annotations) — different structures, same mechanism.
        assert any(0.3 * width < b.gutter_x < 0.7 * width for b in blocks)
        assert any(b.gutter_x > 0.8 * width for b in blocks)

    def test_margin_notes_are_a_lopsided_split_tables_are_not(self, store):
        doc, _ = ingest_document(SAMPLE2, store, IngestParams())
        gray = load_view_array(store, doc.page(2))
        rows = sorted(detect_text_rows(gray), key=lambda b: b.y)
        ratios = []
        for block in detect_column_blocks(gray, rows):
            row = rows[block.row_indices[0]]
            left = block.gutter_x - row.x
            right = row.x + row.w - block.gutter_x
            ratios.append(min(left, right) / max(left, right))
        assert min(ratios) < 0.35  # a margin note: one cell far narrower
        assert max(ratios) > 0.35  # a table: two comparable cells


class TestIssueGeometry:
    """Issues carry location, not just counts — that is what makes them useful.

    A scalar says something is wrong; a box says where, which is what a human
    eyeballing the render and an agent choosing a region to re-run both need.
    """

    def test_a_column_block_yields_a_gutter_issue_with_geometry(self):
        img = two_column_page(rows=4)
        a = assess_rows(img, detect_text_rows(img))
        gutters = [i for i in a.issues() if i.kind is IssueKind.COLUMN_GUTTER]
        assert len(gutters) == 1
        g = gutters[0]
        assert g.bbox.x == pytest.approx(GUTTER_X, abs=15)
        assert g.bbox.h > PITCH * 3  # spans the block, not one row
        assert g.note

    def test_a_clean_page_reports_no_issues(self):
        img = two_column_page(rows=0)
        assert assess_rows(img, detect_text_rows(img)).issues() == []

    def test_every_issue_lands_inside_the_page(self):
        img = two_column_page(rows=4)
        h, w = img.shape
        for issue in assess_rows(img, detect_text_rows(img)).issues():
            assert issue.bbox.x >= 0 and issue.bbox.x + issue.bbox.w <= w + 1
            assert issue.bbox.y >= 0 and issue.bbox.y + issue.bbox.h <= h + 1


@needs_columns
class TestIssuesReachTheViewer:
    def test_issues_are_persisted_and_rendered(self, store):
        from mistara.render.html import render_document
        from mistara.stages.s4_extract import ExtractParams, ExtractStage

        doc, _ = ingest_document(SAMPLE2, store, IngestParams(max_pages=3))
        doc, _ = ExtractStage(store).execute(doc, ExtractParams(provider="stub"), store)

        n_issues = sum(len(p.issues) for p in doc.pages)
        assert n_issues > 0, "the two-column sample must produce gutter issues"

        html = render_document(doc, store)
        assert html.count('class="issue') == n_issues
        assert 'data-issues="on"' in html
        assert "data-toggle-issues" in html
