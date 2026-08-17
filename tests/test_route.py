"""S5 routing.

Two things are worth stating up front, because they shape every test here:

* In this corpus Arabic is set in an Urdu typeface, so a verse contains ی/ہ —
  codepoints `urdu_affinity` reads as Urdu. Text evidence therefore *cannot*
  reliably rescue a voweled verse on its own; the corpus hit is the decisive
  path, and the headline test exercises exactly that.
* The text-evidence path is only asked to handle the clear cases: unvoweled
  Urdu prose, and Arabic that happens to carry no Urdu-form letters.
"""

from __future__ import annotations

from mistara.core.config import RouteConfig
from mistara.core.model import (
    BBox,
    Document,
    Language,
    Line,
    Page,
    Provenance,
    ProvenanceKind,
    Region,
    Span,
)
from mistara.stages.s3_order import language_purity
from mistara.stages.s5_route import (
    RouteParams,
    RouteStage,
    _is_corpus_hit,
    route_span,
)
from mistara.text.canonical import urdu_affinity

CFG = RouteConfig()


# --------------------------------------------------------------------------- #
# route_span — text evidence only, the clear cases
# --------------------------------------------------------------------------- #


def test_unvocalized_urdu_prose_routes_urdu():
    lang, posterior = route_span("یہ ایک سادہ اردو عبارت ہے", CFG)
    assert lang is Language.URDU
    assert posterior > 0


def test_vocalized_arabic_without_urdu_letters_routes_arabic():
    # No Urdu-form letters at all, so orthography is silent (affinity 0) and the
    # tashkeel alone must carry it — exactly the signal Urdu prose never has.
    text = "رَبَّنَا وَتَقَبَّلْ"
    assert urdu_affinity(text) == 0.0
    lang, _ = route_span(text, CFG)
    assert lang is Language.ARABIC


def test_quranic_ornament_brackets_route_arabic():
    lang, _ = route_span("﴿ الم ﴾", CFG)
    assert lang is Language.ARABIC


def test_empty_text_is_unknown():
    lang, posterior = route_span("   ", CFG)
    assert lang is Language.UNKNOWN
    assert posterior == 0.0


def test_ambiguous_span_is_mixed_not_a_false_certainty():
    # Urdu-form letters (Urdu evidence) and heavy tashkeel (Arabic evidence)
    # cancel: the honest answer is Mixed, at low confidence.
    lang, posterior = route_span("یَحْکُمْ", CFG)
    assert lang is Language.MIXED
    assert posterior < CFG.urdu_cut


# --------------------------------------------------------------------------- #
# Corpus hits are decisive
# --------------------------------------------------------------------------- #


def test_a_replaced_span_is_a_corpus_hit():
    span = Span(
        span_id="s",
        char_start=0,
        char_end=3,
        provenance=Provenance(kind=ProvenanceKind.QURAN, source_ref="quran:2:2"),
    )
    assert _is_corpus_hit(span)


def test_a_cited_span_is_a_corpus_hit_even_though_its_text_is_still_ocr():
    span = Span(
        span_id="s",
        char_start=0,
        char_end=3,
        provenance=Provenance(kind=ProvenanceKind.OCR, source_ref="quran:2:2"),
    )
    assert _is_corpus_hit(span)


def test_corpus_hit_overrides_urdu_orthography():
    # The span reads Urdu by orthography, but the matcher verified it: Arabic.
    line = Line(
        line_id="L0",
        bbox=BBox(x=0, y=0, w=100, h=30),
        text="یہ اردو",
        spans=[
            Span(
                span_id="L0s0",
                char_start=0,
                char_end=7,
                language=Language.URDU,
                provenance=Provenance(source_ref="quran:2:2"),
            )
        ],
    )
    doc = _doc([Region(region_id="r0", bbox=line.bbox, lines=[line])])
    out = RouteStage().run(doc, RouteParams())
    assert len(out.spans) == 1
    assert out.spans[0].language is Language.ARABIC
    assert out.spans[0].from_corpus
    assert out.spans[0].posterior == 1.0
    assert out.spans[0].changed  # it was URDU coming in


# --------------------------------------------------------------------------- #
# The headline: corpus hits repair a column the S4 guess mis-tagged
# --------------------------------------------------------------------------- #


def test_corpus_hits_lift_a_mixed_column_to_pure_arabic():
    """An Arabic column whose lines the S4 guess tagged inconsistently — some
    URDU, some ARABIC — but which the matcher verified line by line. Routing
    replaces the guess with the corpus evidence, and the column reads as one
    language. This test fails on the pre-S5 tags."""
    region = _corpus_verified_column(
        # measured shape: a real Arabic column comes back with the guess split,
        # because per-line orthography swings on how many ی/ہ each verse holds.
        guesses=[Language.URDU, Language.ARABIC, Language.URDU, Language.ARABIC],
    )
    pre = language_purity([ln.spans[0].language for ln in region.lines])
    assert pre < 1.0  # the guess, left alone, looks like a mixed column

    doc = _doc([region])
    stage = RouteStage()
    out = stage.run(doc, RouteParams())
    new_doc = stage.apply(doc, out)

    routed = new_doc.page(0).region("c0")
    post = language_purity([ln.spans[0].language for ln in routed.lines])
    assert post == 1.0
    assert all(s.language is Language.ARABIC for s in out.spans)

    verdict = stage.verifier.verify(doc, out, RouteParams())
    assert verdict.metrics["route.column_purity"] == 1.0
    assert verdict.passed


def test_a_column_that_still_mixes_after_routing_is_flagged_advisorily():
    """A column that still mixes after routing is surfaced as a located finding
    for review — but it does not fail the stage, because after corpus-informed
    routing a mixing column is expected (an un-matched verse tail or an inline
    quote), not a re-run trigger."""
    lines = [
        _line("u0", "یہ سادہ اردو عبارت ہے", Language.UNKNOWN),
        _line("u1", "اور یہ بھی اردو ہے", Language.UNKNOWN),
        _line("a0", "رَبَّنَا وَتَقَبَّلْ", Language.UNKNOWN),
    ]
    region = Region(
        region_id="c0",
        bbox=BBox(x=0, y=0, w=200, h=200),
        lines=lines,
        signals={"columns": 2.0},
    )
    doc = _doc([region])
    stage = RouteStage()
    out = stage.run(doc, RouteParams())
    verdict = stage.verifier.verify(doc, out, RouteParams())
    assert verdict.metrics["route.column_purity"] < 1.0
    assert verdict.passed  # advisory, not a failure
    assert any("still mixes" in f for f in verdict.findings)


def test_posterior_is_written_onto_each_span():
    region = Region(
        region_id="r0",
        bbox=BBox(x=0, y=0, w=200, h=40),
        lines=[_line("l0", "یہ سادہ اردو عبارت ہے", Language.UNKNOWN)],
    )
    doc = _doc([region])
    stage = RouteStage()
    new_doc = stage.apply(doc, stage.run(doc, RouteParams()))
    span = new_doc.page(0).region("r0").lines[0].spans[0]
    assert "lang_posterior" in span.signals
    assert span.language is Language.URDU


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _line(line_id: str, text: str, lang: Language) -> Line:
    return Line(
        line_id=line_id,
        bbox=BBox(x=0, y=0, w=200, h=30),
        text=text,
        spans=[
            Span(span_id=f"{line_id}s0", char_start=0, char_end=len(text), language=lang)
        ],
    )


def _corpus_verified_column(guesses: list[Language]) -> Region:
    lines: list[Line] = []
    for i, guess in enumerate(guesses):
        text = "قول"  # the text is immaterial; the source_ref is the evidence
        lines.append(
            Line(
                line_id=f"a{i}",
                bbox=BBox(x=300, y=100 + i * 40, w=200, h=30),
                text=text,
                spans=[
                    Span(
                        span_id=f"a{i}s0",
                        char_start=0,
                        char_end=len(text),
                        language=guess,
                        provenance=Provenance(
                            kind=ProvenanceKind.QURAN, source_ref="quran:2:2"
                        ),
                    )
                ],
            )
        )
    return Region(
        region_id="c0",
        bbox=BBox(x=300, y=100, w=200, h=len(guesses) * 40),
        lines=lines,
        signals={"columns": 2.0},
    )


def _doc(regions: list[Region]) -> Document:
    page = Page(page_no=0, width_px=600, height_px=800, regions=regions)
    return Document(
        doc_id="d0", source_path="x.pdf", source_sha256="0" * 64, pages=[page]
    )
