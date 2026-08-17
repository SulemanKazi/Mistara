"""End-to-end: ingest a real PDF, transcribe with the offline stub, render.

These run against `data/sample_pdf_1.pdf` so the geometry paths are exercised on
a genuine scan rather than a synthetic fixture.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from mistara.core.model import ViewName
from mistara.core.store import Store
from mistara.render.html import render_document
from mistara.stages.s0_ingest import IngestParams, ingest_document, load_view_array
from mistara.stages.s4_extract import ExtractParams, ExtractStage
from mistara.stages.segment import (
    detect_text_rows,
    estimate_line_height,
    split_rows_at_gutters,
)

SAMPLE = Path("data/sample_pdf_1.pdf")
needs_sample = pytest.mark.skipif(not SAMPLE.exists(), reason="sample PDF not present")


@pytest.fixture
def store(tmp_path) -> Store:
    return Store(tmp_path / "store")


@pytest.fixture
def ingested(store):
    doc, report = ingest_document(SAMPLE, store, IngestParams(max_pages=2))
    return store, doc, report


class TestSegmentation:
    def test_synthetic_rows_are_found(self):
        img = np.full((300, 200), 255, np.uint8)
        for y in (20, 70, 120, 170, 220):
            cv2.rectangle(img, (20, y), (180, y + 18), 0, -1)
        boxes = detect_text_rows(img)
        assert len(boxes) == 5
        assert all(b.h >= 10 for b in boxes)
        # Boxes are tightened to the ink, not stretched to full page width.
        assert all(b.x >= 15 and b.x + b.w <= 190 for b in boxes)

    def test_blank_image_yields_no_rows(self):
        assert detect_text_rows(np.full((200, 200), 255, np.uint8)) == []

    def test_rows_are_returned_in_reading_order(self):
        img = np.full((300, 200), 255, np.uint8)
        for y in (20, 100, 180):
            cv2.rectangle(img, (20, y), (180, y + 18), 0, -1)
        boxes = detect_text_rows(img)
        assert [b.y for b in boxes] == sorted(b.y for b in boxes)

    def test_colour_input_is_rejected(self):
        with pytest.raises(ValueError, match="single-channel"):
            detect_text_rows(np.zeros((10, 10, 3), np.uint8))


@needs_sample
class TestIngest:
    def test_extracts_embedded_scan_natively(self, ingested):
        _, doc, report = ingested
        assert len(doc.pages) == 2
        # The sample is a wrapper around embedded PNGs; we must not rasterize it.
        assert report.counts["native"] == 2
        assert all(p.views[ViewName.RAW].note == "native" for p in doc.pages)

    def test_reports_true_source_dpi_not_requested_dpi(self, ingested):
        _, doc, _ = ingested
        # Requested target_dpi is 400, but the embedded scan is ~106 DPI. If we
        # ever report 400 here, we are upsampling and inventing detail.
        assert all(80 < p.source_dpi < 200 for p in doc.pages)

    def test_low_resolution_is_surfaced_as_a_finding(self, ingested):
        _, _, report = ingested
        assert report.verdict.passed  # not repairable, so not a failure
        assert any("resolution is low" in f for f in report.verdict.findings)

    def test_both_views_are_stored_and_decodable(self, ingested):
        store, doc, _ = ingested
        page = doc.pages[0]
        assert {ViewName.RAW, ViewName.GRAY} <= set(page.views)
        gray = load_view_array(store, page, ViewName.GRAY)
        assert gray.ndim == 2
        assert gray.shape == (page.height_px, page.width_px)

    def test_doc_id_is_derived_from_content(self, store):
        a, _ = ingest_document(SAMPLE, store, IngestParams(max_pages=1))
        b, _ = ingest_document(SAMPLE, store, IngestParams(max_pages=1))
        assert a.doc_id == b.doc_id  # same bytes, same document


@needs_sample
class TestExtractWithStub:
    def test_lines_are_anchored_to_detected_geometry(self, ingested):
        store, doc, _ = ingested
        doc, report = ExtractStage(store).execute(
            doc, ExtractParams(provider="stub"), store
        )
        page = doc.pages[0]
        assert len(page.regions) == 1
        lines = page.regions[0].lines
        assert lines
        # Every line must sit inside its page, or the spatial output is a lie.
        assert all(page.bbox.contains(ln.bbox) for ln in lines)
        assert report.verdict.metrics["coverage.line_count"] == pytest.approx(1.0)

    def test_verifier_detects_the_stubs_repetition(self, ingested):
        store, doc, _ = ingested
        _, report = ExtractStage(store).execute(
            doc, ExtractParams(provider="stub"), store
        )
        # The stub emits repeated filler; the degeneracy check must catch it.
        assert report.verdict.metrics["degeneracy.repeat_rate"] > 0.25
        assert any("decoding loop" in f for f in report.verdict.findings)
        assert not report.verdict.passed

    def test_run_is_recorded_in_the_ledger(self, ingested):
        store, doc, _ = ingested
        doc, _ = ExtractStage(store).execute(doc, ExtractParams(provider="stub"), store)
        stages = [r["stage"] for r in store.runs_for(doc.doc_id)]
        assert "s0_ingest" in stages and "s4_extract" in stages

    def test_line_count_matches_detected_cells(self, ingested):
        store, doc, _ = ingested
        gray = load_view_array(store, doc.pages[0], ViewName.GRAY)
        rows = detect_text_rows(gray)
        cells = split_rows_at_gutters(gray, rows)
        doc, _ = ExtractStage(store).execute(doc, ExtractParams(provider="stub"), store)
        # Cells, not rows: a row holding a margin note or an Arabic/Urdu pair is
        # split before transcription, so each cell is transcribed separately.
        assert len(doc.pages[0].regions[0].lines) == len(cells)
        assert len(cells) >= len(rows)
        assert estimate_line_height(rows) > 5

    def test_splitting_can_be_disabled(self, ingested):
        store, doc, _ = ingested
        gray = load_view_array(store, doc.pages[0], ViewName.GRAY)
        doc, _ = ExtractStage(store).execute(
            doc, ExtractParams(provider="stub", split_columns=False), store
        )
        assert len(doc.pages[0].regions[0].lines) == len(detect_text_rows(gray))


@needs_sample
class TestRender:
    def test_html_is_self_contained_and_places_every_line(self, ingested):
        store, doc, _ = ingested
        doc, _ = ExtractStage(store).execute(doc, ExtractParams(provider="stub"), store)
        html = render_document(doc, store)

        assert html.startswith("<!doctype html>")
        assert "data:image/png;base64," in html  # portable, no external files
        n_lines = sum(len(r.lines) for p in doc.pages for r in p.regions)
        assert html.count('class="box"') == n_lines
        # RTL is a correctness property here, not styling.
        assert 'direction:rtl' in html

    def test_render_survives_a_document_with_no_text(self, ingested):
        store, doc, _ = ingested
        html = render_document(doc, store)
        assert "<main>" in html
