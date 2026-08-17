"""The reading view.

Synthetic geometry throughout: every assertion here is about a transformation
whose right answer is known by construction (this region is 60px wide on a
700px page, these three lines fill the measure, this span is corpus text). The
one thing these tests cannot check is whether the result *looks* finished —
that is what the headless-Chrome screenshot loop in CLAUDE.md is for.
"""

from __future__ import annotations

import re

from mistara.core.model import (
    BBox,
    Document,
    Line,
    Page,
    Provenance,
    ProvenanceKind,
    Region,
    RegionType,
    Span,
)
from mistara.render.reader import (
    _body_cells,
    _hashiya,
    _region_html,
    _rows,
    _split_band,
    render_reader,
)

PAGE_W, PAGE_H = 700, 1124


def line(y: float, w: float, text: str = "متن", x: float = 6, h: float = 30) -> Line:
    return Line(line_id=f"l{y:.0f}", bbox=BBox(x=x, y=y, w=w, h=h), text=text)


def quran_line(y: float, w: float, text: str, ref: str = "quran:2:64") -> Line:
    return Line(
        line_id=f"q{y:.0f}",
        bbox=BBox(x=6, y=y, w=w, h=30),
        text=text,
        spans=[
            Span(
                span_id="s0",
                char_start=0,
                char_end=len(text),
                provenance=Provenance(
                    kind=ProvenanceKind.QURAN, source_ref=ref, original_text="raw"
                ),
            )
        ],
    )


def region(
    region_id: str,
    bbox: BBox,
    lines: list[Line],
    *,
    columns: float = 1,
    column_index: float = -1,
    block: float = -1,
    kind: RegionType = RegionType.UNKNOWN,
) -> Region:
    return Region(
        region_id=region_id,
        bbox=bbox,
        region_type=kind,
        lines=lines,
        signals={"columns": columns, "column_index": column_index, "block": block},
    )


def page(regions: list[Region]) -> Page:
    return Page(
        page_no=0,
        width_px=PAGE_W,
        height_px=PAGE_H,
        regions=regions,
        reading_order=[r.region_id for r in regions],
    )


def document(pages: list[Page]) -> Document:
    return Document(
        doc_id="deadbeef",
        source_path="data/x.pdf",
        source_sha256="0" * 64,
        pages=pages,
    )


#: The shape sample 2 exhibits: a body column and a narrow margin note sharing
#: one block, between two full-width prose regions.
def sample_page() -> Page:
    return page(
        [
            region("r0", BBox(x=6, y=14, w=680, h=662), [line(20, 660, "اول")]),
            region(
                "r1",
                BBox(x=6, y=689, w=599, h=71),
                [line(700, 590), line(740, 590)],
                columns=2,
                column_index=1,
                block=0,
            ),
            region(
                "r2",
                BBox(x=637, y=689, w=60, h=68),
                [line(700, 55, "نہی"), line(740, 55, "قید")],
                columns=2,
                column_index=0,
                block=0,
                kind=RegionType.MARGINALIA,
            ),
            region("r3", BBox(x=2, y=770, w=608, h=349), [line(780, 600, "آخر")]),
        ]
    )


class TestBands:
    def test_a_block_becomes_one_band(self):
        rows = _rows(sample_page())
        assert [[r.region_id for r in row.regions] for row in rows] == [
            ["r0"],
            ["r2", "r1"],  # column_index 0 first — RTL, rightmost leads
            ["r3"],
        ]

    def test_band_extent_spans_every_region_in_it(self):
        band = _rows(sample_page())[1]
        assert band.top == 689 / PAGE_H
        assert band.bottom == (689 + 71) / PAGE_H  # the taller of the two

    def test_regions_of_different_blocks_do_not_merge(self):
        p = page(
            [
                region("a", BBox(x=0, y=0, w=300, h=50), [line(0, 300)], block=0),
                region("b", BBox(x=0, y=60, w=300, h=50), [line(60, 300)], block=1),
            ]
        )
        assert len(_rows(p)) == 2


class TestGeometry:
    def test_the_note_is_split_out_of_the_body(self):
        body, notes = _split_band(_rows(sample_page())[1])
        assert [r.region_id for r in body] == ["r1"]
        assert [r.region_id for r in notes] == ["r2"]

    def test_body_text_always_fills_the_measure(self):
        """The reserved hashiya is the whole point: a page *with* a note and a
        page *without* one must set their body text to the same width."""
        with_note, _ = _split_band(_rows(sample_page())[1])
        alone, _ = _split_band(_rows(sample_page())[0])
        assert _body_cells(with_note)[0][1] == 100.0
        assert _body_cells(alone)[0][1] == 100.0
        assert _body_cells(with_note)[0][2] == 0.0

    def test_parallel_columns_keep_their_proportions_inside_the_measure(self):
        """Two body columns still divide the measure the way the page did."""
        band = [
            region(
                "right",
                BBox(x=360, y=0, w=240, h=50),
                [line(0, 240)],
                columns=2,
                column_index=0,
                block=0,
            ),
            region(
                "left",
                BBox(x=0, y=0, w=300, h=50),
                [line(0, 300)],
                columns=2,
                column_index=1,
                block=0,
            ),
        ]
        cells = _body_cells(band)
        assert [r.region_id for r, _, _ in cells] == ["right", "left"]
        # The span is 600px: 240 and 300 wide, with a 60px gutter between them.
        assert [round(w, 1) for _, w, _ in cells] == [40.0, 50.0]
        assert [round(g, 1) for _, _, g in cells] == [0.0, 10.0]


class TestHashiya:
    def test_the_reserved_column_is_measured_from_the_notes(self):
        h = _hashiya(document([sample_page()]))
        # 60px of a 700px page is 8.6%, floored to the readable minimum.
        assert h.width == 12.0
        # The gutter is the gap between the note and the body beside it.
        assert round(h.gutter, 2) == round(100 * (637 - 605) / 700, 2)

    def test_a_document_with_no_notes_still_reserves_the_column(self):
        p = page([region("r0", BBox(x=0, y=0, w=600, h=50), [line(0, 600)])])
        h = _hashiya(document([p]))
        assert h.width > 0 and h.gutter > 0
        assert h.side(0) in ("right", "left")

    def test_the_side_follows_the_outer_margin_and_alternates(self):
        """These books put the hashiya on the outer margin, so it flips with the
        leaf. A page that carries a note uses its own side; a page that does not
        inherits the parity the rest of the document votes for."""
        right = page(
            [
                region("b", BBox(x=0, y=0, w=600, h=50), [line(0, 600)], block=0),
                region(
                    "n",
                    BBox(x=637, y=0, w=60, h=50),
                    [line(0, 55)],
                    block=0,
                    kind=RegionType.MARGINALIA,
                ),
            ]
        )
        left = Page(
            page_no=1,
            width_px=PAGE_W,
            height_px=PAGE_H,
            regions=[
                region("b", BBox(x=100, y=0, w=600, h=50), [line(0, 600)], block=0),
                region(
                    "n",
                    BBox(x=6, y=0, w=60, h=50),
                    [line(0, 55)],
                    block=0,
                    kind=RegionType.MARGINALIA,
                ),
            ],
            reading_order=["b", "n"],
        )
        blank = Page(
            page_no=2,
            width_px=PAGE_W,
            height_px=PAGE_H,
            regions=[region("b", BBox(x=0, y=0, w=600, h=50), [line(0, 600)])],
            reading_order=["b"],
        )
        h = _hashiya(document([right, left, blank]))
        assert h.side(0) == "right"  # observed
        assert h.side(1) == "left"  # observed
        assert h.side(2) == "right"  # predicted from the parity of the others


class TestParagraphs:
    def test_full_measure_lines_run_together(self):
        r = region(
            "r",
            BBox(x=0, y=0, w=600, h=200),
            [line(0, 600, "الف"), line(40, 600, "ب"), line(80, 600, "ج")],
        )
        assert _region_html(r, pitch=40).count("<p") == 1

    def test_a_short_line_ends_the_paragraph(self):
        r = region(
            "r",
            BBox(x=0, y=0, w=600, h=200),
            [line(0, 600, "الف"), line(40, 200, "ب"), line(80, 600, "ج")],
        )
        assert _region_html(r, pitch=40).count("<p") == 2

    def test_a_wide_vertical_gap_ends_the_paragraph(self):
        """The running-head case: full measure, but detached from the text."""
        r = region(
            "r",
            BBox(x=0, y=0, w=600, h=200),
            [line(0, 600, "سرصفحہ"), line(90, 600, "متن"), line(130, 600, "متن")],
        )
        html = _region_html(r, pitch=40)
        assert html.count("<p") == 2
        assert html.index("سرصفحہ") < html.index("</p>")

    def test_every_line_survives_the_grouping(self):
        r = region(
            "r",
            BBox(x=0, y=0, w=600, h=200),
            [line(0, 600, "الف"), line(40, 200, "ب"), line(80, 600, "ج")],
        )
        html = _region_html(r, pitch=40)
        assert [t for t in ("الف", "ب", "ج") if t in html] == ["الف", "ب", "ج"]

    def test_lines_stay_individually_addressable(self):
        """Paragraph mode is a CSS state, so the line elements must survive it."""
        r = region("r", BBox(x=0, y=0, w=600, h=200), [line(0, 600), line(40, 600)])
        assert _region_html(r, pitch=40).count('class="ln"') == 2


class TestProvenance:
    def test_a_verse_line_becomes_a_display_block_with_one_reference(self):
        r = region(
            "r",
            BBox(x=0, y=0, w=600, h=100),
            [quran_line(0, 600, "ثم توليتم من بعد ذلك")],
        )
        html = _region_html(r, pitch=40)
        assert 'class="ayah"' in html
        # The block carries the citation once; the inline chip is suppressed.
        assert html.count('class="ref"') == 1
        assert "2:64" in html

    def test_an_inline_quote_stays_in_the_prose(self):
        text = "اور فرمایا لا تشتروا بآياتي ثمنا قليلا اس کا مطلب"
        start, end = text.index("لا"), text.index("قليلا") + len("قليلا")
        ln = Line(
            line_id="l0",
            bbox=BBox(x=0, y=0, w=600, h=30),
            text=text,
            spans=[
                Span(
                    span_id="s0",
                    char_start=start,
                    char_end=end,
                    provenance=Provenance(
                        kind=ProvenanceKind.QURAN, source_ref="quran:2:41"
                    ),
                )
            ],
        )
        html = _region_html(region("r", BBox(x=0, y=0, w=600, h=100), [ln]), pitch=40)
        assert 'class="ayah"' not in html
        assert 'class="ar sacred"' in html
        # Text outside the span is untouched prose.
        assert "اور فرمایا" in html.split('<span class="ar sacred"')[0]

    def test_identified_text_is_marked_but_not_recoloured(self):
        """`cited` means the source is known and the text is still the OCR
        reading. Setting it in the corpus colour would claim a replacement."""
        ln = Line(
            line_id="l0",
            bbox=BBox(x=0, y=0, w=600, h=30),
            text="آیت",
            spans=[
                Span(
                    span_id="s0",
                    char_start=0,
                    char_end=3,
                    provenance=Provenance(
                        kind=ProvenanceKind.OCR, source_ref="quran:5:44"
                    ),
                )
            ],
        )
        html = _region_html(region("r", BBox(x=0, y=0, w=600, h=100), [ln]), pitch=40)
        # Blue and naskh like all Arabic (`ar`), but never the verified shade.
        assert 'class="ar cited"' in html
        assert "sacred" not in html
        assert 'class="ayah"' not in html

    def test_text_outside_any_span_is_still_rendered(self):
        ln = Line(
            line_id="l0",
            bbox=BBox(x=0, y=0, w=600, h=30),
            text="ابجد",
            spans=[Span(span_id="s0", char_start=0, char_end=2)],
        )
        assert "ابجد" in re.sub(
            r"<[^>]+>", "", _region_html(region("r", BBox(x=0, y=0, w=600, h=100), [ln]))
        )

    def test_markup_in_the_text_is_escaped(self):
        r = region("r", BBox(x=0, y=0, w=600, h=100), [line(0, 600, "<script>")])
        assert "<script>" not in _region_html(r)


class TestDocument:
    def test_regions_appear_in_reading_order(self):
        p = sample_page()
        html = render_reader(document([p]), embed_fonts=False)
        body = html[html.index("<main>") :]
        # r2 (the margin note) precedes r1 in the source because RTL puts the
        # rightmost column first, but the band as a whole sits between r0 and r3.
        assert body.index("اول") < body.index("نہی") < body.index("آخر")

    def test_every_page_reserves_the_hashiya_cell(self):
        """Even the pages with no note — otherwise the measure moves."""
        blank = Page(
            page_no=1,
            width_px=PAGE_W,
            height_px=PAGE_H,
            regions=[region("b", BBox(x=0, y=0, w=600, h=50), [line(0, 600)])],
            reading_order=["b"],
        )
        html = render_reader(document([sample_page(), blank]), embed_fonts=False)
        assert html.count('class="hashiya"') == html.count('class="row"')
        assert "--hw:" in html

    def test_a_note_and_the_text_it_annotates_share_one_row(self):
        """They are one band, so they must be one grid row — a note that lands
        in a row of its own drops below the text it belongs beside."""
        html = render_reader(document([sample_page()]), embed_fonts=False)
        rows = re.findall(r'<div class="row".*?</aside></div>', html, re.S)
        band = [r for r in rows if "نہی" in r]
        assert len(band) == 1
        assert "متن" in band[0]

    def test_the_page_number_is_in_the_footer(self):
        html = render_reader(document([sample_page()]), embed_fonts=False)
        assert '<div class="folio">1</div>' in html
        assert html.index('class="leaf"') < html.index('class="folio"')

    def test_the_scan_is_not_embedded(self):
        """This view is the text, not the page image — no store, no bitmap."""
        html = render_reader(document([sample_page()]), embed_fonts=False)
        assert "<img" not in html
        assert "data:image" not in html

    def test_fonts_can_be_left_out(self):
        html = render_reader(document([sample_page()]), embed_fonts=False)
        assert "@font-face" not in html
        assert "system fonts" in html

    def test_an_empty_document_renders(self):
        html = render_reader(document([]), embed_fonts=False)
        assert "<main></main>" in html
