"""Detecting missed and merged lines without ground truth.

The detector leans on typesetting regularity: printed pages sit on a baseline
grid, so a merge shows up as a double-height box and a miss as a double-height
gap that still contains ink. These tests build synthetic pages where the answer
is known by construction, then confirm the real sample behaves as measured.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from mistara.core.model import BBox
from mistara.core.store import Store
from mistara.stages.s0_ingest import IngestParams, ingest_document, load_view_array
from mistara.stages.segment import assess_rows, detect_text_rows

SAMPLE = Path("data/sample_pdf_1.pdf")
needs_sample = pytest.mark.skipif(not SAMPLE.exists(), reason="sample PDF not present")

PITCH = 40
LINE_H = 20


def page_with_lines(n: int = 12, *, widths: list[int] | None = None) -> np.ndarray:
    """A synthetic page: n evenly pitched lines of controllable width."""
    img = np.full((PITCH * (n + 1), 400), 255, np.uint8)
    for i in range(n):
        w = widths[i] if widths else 360
        y = PITCH * (i + 1)
        cv2.rectangle(img, (20, y), (20 + w, y + LINE_H), 0, -1)
    return img


class TestCleanPage:
    def test_regular_page_is_reported_clean(self):
        img = page_with_lines(12)
        boxes = detect_text_rows(img)
        a = assess_rows(img, boxes)
        assert a is not None
        assert a.est_merged_lines == 0
        assert a.est_missed_lines == 0
        assert a.findings() == []
        assert a.pitch_cv < 0.1  # evenly spaced by construction

    def test_pitch_matches_construction(self):
        a = assess_rows(page_with_lines(10), detect_text_rows(page_with_lines(10)))
        assert a.median_pitch == pytest.approx(PITCH, abs=2)

    def test_too_few_lines_yields_no_assessment(self):
        # Regularity is meaningless with two lines; say nothing rather than guess.
        assert assess_rows(page_with_lines(2), detect_text_rows(page_with_lines(2))) is None


class TestMergeDetection:
    def test_a_double_height_box_is_flagged_as_one_merge(self):
        img = page_with_lines(10)
        boxes = detect_text_rows(img)
        # Fuse boxes 3 and 4 into one, as a marginal note bridging them would.
        merged = BBox(
            x=boxes[3].x,
            y=boxes[3].y,
            w=boxes[3].w,
            h=boxes[4].y + boxes[4].h - boxes[3].y,
        )
        mangled = boxes[:3] + [merged] + boxes[5:]
        a = assess_rows(img, mangled)
        assert a.est_merged_lines == 1
        assert a.tall_boxes == 1
        assert any("merged" in f for f in a.findings())

    def test_a_triple_merge_counts_two_lost_lines(self):
        img = page_with_lines(12)
        boxes = detect_text_rows(img)
        merged = BBox(
            x=boxes[4].x,
            y=boxes[4].y,
            w=boxes[4].w,
            h=boxes[6].y + boxes[6].h - boxes[4].y,
        )
        a = assess_rows(img, boxes[:4] + [merged] + boxes[7:])
        assert a.est_merged_lines == 2


class TestMissDetection:
    def test_a_dropped_line_leaves_stranded_ink_and_is_flagged(self):
        img = page_with_lines(10)
        boxes = detect_text_rows(img)
        a = assess_rows(img, boxes[:5] + boxes[6:])  # drop box 5; its ink remains
        assert a.est_missed_lines == 1
        assert a.stranded_gaps == 1
        assert any("missed" in f for f in a.findings())

    def test_a_wide_but_empty_gap_is_not_flagged(self):
        """The header/body gap is also ~2x pitch — but holds no ink.

        Requiring stranded ink is what keeps this from being a false positive,
        and it is the reason gap width alone is not a usable signal.
        """
        img = page_with_lines(10)
        boxes = detect_text_rows(img)
        blank = boxes[5]
        img[int(blank.y) - 2 : int(blank.y + blank.h) + 2, :] = 255  # erase the ink
        a = assess_rows(img, boxes[:5] + boxes[6:])
        assert a.est_missed_lines == 0
        assert a.findings() == []

    def test_a_short_line_is_dropped_by_detection_and_caught_by_assessment(self):
        """The real-world failure mode, reproduced end to end.

        A paragraph-final clause carries a small fraction of a full row's ink,
        so a *global* row-sum threshold drops it. That is a genuine limitation of
        projection segmentation — the point of the assessment is that we notice.
        """
        img = page_with_lines(10, widths=[360] * 5 + [30] + [360] * 4)
        boxes = detect_text_rows(img)

        assert len(boxes) == 9, "the short line should defeat the global threshold"
        a = assess_rows(img, boxes)
        assert a.est_missed_lines == 1
        assert a.stranded_gaps == 1


@pytest.fixture(scope="module")
def doc_and_store(tmp_path_factory):
    store = Store(tmp_path_factory.mktemp("store"))
    doc, _ = ingest_document(SAMPLE, store, IngestParams())
    return doc, store


@needs_sample
class TestOnTheRealSample:
    def test_vertical_failures_are_rare_and_localized(self, doc_and_store):
        """Merges and misses are the *vertical* failures, tracked separately.

        Column structure is common and expected (these pages carry margin
        notes); merges and misses are genuine defects. Lumping them would hide
        the defects in the noise, so the counts are asserted apart.
        """
        doc, store = doc_and_store
        vertical = 0
        for page in doc.pages:
            gray = load_view_array(store, page)
            a = assess_rows(gray, detect_text_rows(gray))
            if a and (a.est_merged_lines or a.est_missed_lines):
                vertical += 1
        # Pages 0 and 5 only; if this moves, segmentation changed and the
        # change should be deliberate.
        assert vertical == 2

    def test_marginalia_shows_up_as_column_structure(self, doc_and_store):
        doc, store = doc_and_store
        with_columns = 0
        for page in doc.pages:
            gray = load_view_array(store, page)
            a = assess_rows(gray, detect_text_rows(gray))
            if a and a.column_blocks:
                with_columns += 1
        assert with_columns >= 4

    def test_typesetting_regularity_holds(self, doc_and_store):
        doc, store = doc_and_store
        for page in doc.pages:
            gray = load_view_array(store, page)
            a = assess_rows(gray, detect_text_rows(gray))
            assert a is not None
            assert 35 < a.median_pitch < 50  # a consistent baseline grid
            assert a.pitch_cv < 0.35
            assert a.ink_covered > 0.90
