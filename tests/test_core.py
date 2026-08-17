"""Document model, artifact store, and metric behaviour."""

from __future__ import annotations

import pytest

from mistara.core.model import BBox, Document, Line, Page, Span
from mistara.core.store import Store, stable_hash
from mistara.text.metrics import agreement, cer, wer


class TestBBox:
    def test_iou_of_identical_boxes_is_one(self):
        b = BBox(x=10, y=10, w=100, h=20)
        assert b.iou(b) == pytest.approx(1.0)

    def test_disjoint_boxes_have_zero_iou(self):
        a = BBox(x=0, y=0, w=10, h=10)
        b = BBox(x=100, y=100, w=10, h=10)
        assert a.iou(b) == 0.0

    def test_half_overlap(self):
        a = BBox(x=0, y=0, w=10, h=10)
        b = BBox(x=5, y=0, w=10, h=10)
        assert a.iou(b) == pytest.approx(50 / 150)

    def test_containment(self):
        outer = BBox(x=0, y=0, w=100, h=100)
        assert outer.contains(BBox(x=10, y=10, w=10, h=10))
        assert not outer.contains(BBox(x=90, y=90, w=50, h=50))


class TestLineValidation:
    def test_span_outside_text_is_rejected(self):
        with pytest.raises(ValueError, match="outside line text"):
            Line(
                line_id="l1",
                bbox=BBox(x=0, y=0, w=10, h=10),
                text="abc",
                spans=[Span(span_id="s1", char_start=0, char_end=99)],
            )

    def test_valid_span_is_accepted(self):
        line = Line(
            line_id="l1",
            bbox=BBox(x=0, y=0, w=10, h=10),
            text="سلام",
            spans=[Span(span_id="s1", char_start=0, char_end=4)],
        )
        assert line.spans[0].text_of(line.text) == "سلام"


class TestStore:
    def test_blob_roundtrip_is_content_addressed(self, tmp_path):
        store = Store(tmp_path)
        sha_a = store.put_blob(b"hello")
        sha_b = store.put_blob(b"hello")
        assert sha_a == sha_b  # same bytes, one blob
        assert store.get_blob(sha_a) == b"hello"
        assert store.has_blob(sha_a)

    def test_missing_blob_raises(self, tmp_path):
        with pytest.raises(KeyError):
            Store(tmp_path).get_blob("0" * 64)

    def test_document_roundtrip(self, tmp_path):
        store = Store(tmp_path)
        doc = Document(
            doc_id="abc123",
            source_path="x.pdf",
            source_sha256="deadbeef",
            pages=[Page(page_no=0, width_px=700, height_px=1150)],
        )
        store.save_document(doc)
        loaded = store.load_document("abc123")
        assert loaded.doc_id == doc.doc_id
        assert loaded.pages[0].width_px == 700

    def test_doc_id_prefix_resolution(self, tmp_path):
        store = Store(tmp_path)
        store.save_document(
            Document(doc_id="abc123", source_path="x", source_sha256="y")
        )
        assert store.resolve_doc_id("abc") == "abc123"
        with pytest.raises(KeyError, match="no document"):
            store.resolve_doc_id("zzz")

    def test_stable_hash_is_key_order_independent(self):
        assert stable_hash({"a": 1, "b": 2}) == stable_hash({"b": 2, "a": 1})


class TestMetrics:
    def test_identical_text_has_zero_error(self):
        text = "یہ معاہدہ قرآن مجید"
        assert cer(text, text).rate == 0.0
        assert wer(text, text).rate == 0.0

    def test_cer_counts_character_edits_against_reference_length(self):
        result = cer("کتاب", "کتب")
        assert result.edits == 1
        assert result.reference_units == 3

    def test_normalization_mode_changes_the_verdict(self):
        # Tashkeel differences are errors in strict mode, invisible in loose.
        assert cer("بِسْمِ", "بسم", "loose").rate == 0.0
        assert cer("بِسْمِ", "بسم", "strict").rate > 0.0

    def test_agreement_is_symmetric_and_bounded(self):
        a, b = "کتاب اچھا ہے", "کتاب برا ہے"
        assert agreement(a, b) == pytest.approx(agreement(b, a))
        assert 0.0 <= agreement(a, b) <= 1.0
        assert agreement(a, a) == 1.0

    def test_agreement_of_unrelated_text_is_low(self):
        assert agreement("کتاب اچھا ہے", "Something else entirely") < 0.3

    def test_empty_reference_is_handled(self):
        assert cer("", "").rate == 0.0
        assert cer("something", "").rate == 1.0
