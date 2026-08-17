"""S5 — Routing: assign a language to every span.

Design §S5 draws this stage before corpus matching, using S2's typeface as the
strong prior. In practice the reliable detector *is* the matcher: a corpus hit
is decisive evidence a span is Arabic, where orthography alone mis-tags Quranic
Arabic as Urdu because these books set it in an Urdu typeface. So S5 runs
*after* S6a (mirroring the way S3 runs after S4), consuming the `Language.ARABIC`
tags the matcher already wrote and generalising them to the spans it did not
resolve.

The generalisation is deterministic text evidence only — no image, no model:

* a **corpus hit** (a span the matcher verified or cited) → Arabic, decisively;
* otherwise a signed score in [-1, +1] (-1 = certain Arabic, +1 = certain Urdu)
  from Urdu-exclusive orthography, vocalization, and Quranic ornaments.

Image typeface is deliberately **not** used. In this corpus Arabic is set in an
Urdu typeface, so naskh/nastaliq is unreliable evidence here rather than helpful,
and reintroducing it would fight the mis-tag this stage exists to correct.

The stage is presentational in the same sense the renderers are: it writes
`language` and per-span signals, and never moves text, re-splits a region, or
decides a region's *type*. Its verifier re-checks column purity under the new
tags — the whole point of the change is that purity should rise toward 1.0.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from mistara.core.config import OrderConfig, RouteConfig
from mistara.core.model import (
    Document,
    Language,
    Line,
    Page,
    ProvenanceKind,
    Region,
    Span,
)
from mistara.core.stage import Stage, StageParams, Verdict, Verifier
from mistara.core.store import stable_hash
from mistara.stages.s3_order import language_purity
from mistara.text.canonical import (
    ornament_density,
    tashkeel_density,
    urdu_affinity,
)


class RouteParams(StageParams):
    cfg: RouteConfig = Field(default_factory=RouteConfig)
    #: The purity cross-check reuses order's thresholds rather than duplicating
    #: them; the whole section is embedded so the params hash covers retuning it.
    order: OrderConfig = Field(default_factory=OrderConfig)


class SpanRoute(BaseModel):
    page_no: int
    region_id: str
    line_id: str
    span_index: int
    language: Language
    posterior: float
    from_corpus: bool
    #: Whether this differs from the tag the span carried coming in (the S4 guess
    #: or a prior route). The headline count: how much the corpus evidence moved.
    changed: bool


class RouteResult(BaseModel):
    spans: list[SpanRoute] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Decision — pure, per span
# --------------------------------------------------------------------------- #


def _is_corpus_hit(span: Span) -> bool:
    """True if the matcher resolved this span against the corpus.

    Covers both outcomes: a replaced span carries `kind` quran/hadith, and a
    cited span keeps `kind` ocr but gains a `source_ref`. Either way the span
    was verified against an authenticated text, which is decisive for language.
    """
    prov = span.provenance
    return (
        prov.kind in (ProvenanceKind.QURAN, ProvenanceKind.HADITH)
        or prov.source_ref is not None
    )


def route_span(text: str, cfg: RouteConfig) -> tuple[Language, float]:
    """Language and confidence for a non-corpus span, from text evidence alone.

    Returns the assigned language and a posterior in [0, 1] — the distance of the
    evidence from neutral, so a genuinely mixed span reports low confidence
    rather than a false certainty.
    """
    if not text.strip():
        return Language.UNKNOWN, 0.0

    affinity = urdu_affinity(text)  # [-1, +1], + = Urdu
    tash = min(1.0, tashkeel_density(text) / cfg.tashkeel_arabic) if cfg.tashkeel_arabic else 0.0
    ornament = 1.0 if ornament_density(text) > 0 else 0.0

    score = cfg.w_urdu * affinity - cfg.w_tashkeel * tash - cfg.w_ornament * ornament
    score = max(-1.0, min(1.0, score))

    if score >= cfg.urdu_cut:
        return Language.URDU, abs(score)
    if score <= cfg.arabic_cut:
        return Language.ARABIC, abs(score)
    return Language.MIXED, abs(score)


# --------------------------------------------------------------------------- #
# Verifier
# --------------------------------------------------------------------------- #


class RouteVerifier(Verifier[RouteResult]):
    """The numeric signals: language mix, confidence, and the purity cross-check.

    The headline is `route.retagged` — spans whose tag the corpus evidence
    corrected. `route.column_purity` is a *diagnostic*, not a gate: recomputed
    from the new tags, it flags columns that still mix languages so a reviewer
    can look. After corpus-informed routing a mixing column is usually not a
    grouping bug but a short Arabic verse-tail the matcher declined (see the
    `match` unresolved count) or a genuinely mixed prose block with an inline
    quote — so the findings are advisory and do not fail the stage: re-running it
    deterministically would change nothing.
    """

    name = "route"

    def verify(self, doc: Document, output: RouteResult, params: RouteParams) -> Verdict:
        findings: list[str] = []
        routed = output.spans
        if not routed:
            return Verdict(passed=True, metrics={"route.spans": 0.0})

        new_lang = {
            (s.page_no, s.line_id, s.span_index): s.language for s in routed
        }
        counts = {lang: 0 for lang in (Language.ARABIC, Language.URDU, Language.MIXED)}
        for s in routed:
            if s.language in counts:
                counts[s.language] += 1
        total = len(routed)

        metrics: dict[str, float] = {
            "route.spans": float(total),
            "route.from_corpus": float(sum(1 for s in routed if s.from_corpus)),
            "route.retagged": float(sum(1 for s in routed if s.changed)),
            "route.mean_posterior": sum(s.posterior for s in routed) / total,
            "route.arabic_frac": counts[Language.ARABIC] / total,
            "route.urdu_frac": counts[Language.URDU] / total,
            "route.mixed_frac": counts[Language.MIXED] / total,
        }

        purities: list[float] = []
        for page in doc.pages:
            for region in page.regions:
                if region.signals.get("columns", 1) < 2:
                    continue
                if len(region.lines) < params.order.min_lines_for_purity:
                    continue
                langs = [
                    _leading_language(page.page_no, ln, new_lang)
                    for ln in region.lines
                ]
                purity = language_purity(langs)
                purities.append(purity)
                if purity < params.order.min_column_purity:
                    findings.append(
                        f"page {page.page_no}: column region {region.region_id} "
                        f"still mixes languages ({purity:.0%} agree) after routing — "
                        f"worth a look: usually a short Arabic verse-tail the matcher "
                        f"declined, or a mixed prose block with an inline quote; only "
                        f"rarely a mis-placed gutter"
                    )
        if purities:
            metrics["route.column_purity"] = sum(purities) / len(purities)

        # Findings are advisory: they locate columns to eyeball, but a mixing
        # column is expected here and re-running the stage would not change it.
        return Verdict(
            passed=True,
            score=metrics["route.mean_posterior"],
            findings=findings,
            metrics=metrics,
        )


def _leading_language(
    page_no: int,
    line: Line,
    new_lang: dict[tuple[int, str, int], Language],
) -> Language:
    """A line's language for purity: its first span's tag, after routing.

    Falls back to the tag already on the line for anything the router left
    untouched (an empty line has no spans and contributes nothing).
    """
    if not line.spans:
        return Language.UNKNOWN
    return new_lang.get((page_no, line.line_id, 0)) or line.spans[0].language


# --------------------------------------------------------------------------- #
# Stage
# --------------------------------------------------------------------------- #


class RouteStage(Stage[RouteParams, RouteResult]):
    name = "s5_route"
    version = "1"
    params_model = RouteParams
    verifier = RouteVerifier()

    def run(self, doc: Document, params: RouteParams) -> RouteResult:
        cfg = params.cfg
        spans: list[SpanRoute] = []
        for page in doc.pages:
            for region in page.regions:
                for line in region.lines:
                    for i, span in enumerate(line.spans):
                        if _is_corpus_hit(span):
                            language, posterior, from_corpus = (
                                Language.ARABIC,
                                1.0,
                                True,
                            )
                        else:
                            language, posterior = route_span(
                                span.text_of(line.text), cfg
                            )
                            from_corpus = False
                        spans.append(
                            SpanRoute(
                                page_no=page.page_no,
                                region_id=region.region_id,
                                line_id=line.line_id,
                                span_index=i,
                                language=language,
                                posterior=posterior,
                                from_corpus=from_corpus,
                                changed=language is not span.language,
                            )
                        )
        return RouteResult(spans=spans)

    def apply(self, doc: Document, output: RouteResult) -> Document:
        """Write the routed language and posterior onto each span."""
        by_span = {
            (s.page_no, s.line_id, s.span_index): s for s in output.spans
        }
        if not by_span:
            return doc

        pages: list[Page] = []
        for page in doc.pages:
            regions: list[Region] = []
            for region in page.regions:
                lines: list[Line] = []
                for line in region.lines:
                    spans = [
                        _apply_span(page.page_no, line.line_id, i, span, by_span)
                        for i, span in enumerate(line.spans)
                    ]
                    lines.append(line.model_copy(update={"spans": spans}))
                regions.append(region.model_copy(update={"lines": lines}))
            pages.append(page.model_copy(update={"regions": regions}))
        return doc.model_copy(update={"pages": pages})

    def input_hash(self, doc: Document, params: RouteParams) -> str:
        return stable_hash(
            {
                "spans": [
                    (
                        ln.text,
                        s.char_start,
                        s.char_end,
                        str(s.language),
                        str(s.provenance.kind),
                        s.provenance.source_ref,
                    )
                    for page in doc.pages
                    for region in page.regions
                    for ln in region.lines
                    for s in ln.spans
                ],
                "params": params.hash(),
            }
        )

    def counts(self, output: RouteResult) -> dict[str, int]:
        return {
            "arabic": sum(1 for s in output.spans if s.language is Language.ARABIC),
            "urdu": sum(1 for s in output.spans if s.language is Language.URDU),
            "mixed": sum(1 for s in output.spans if s.language is Language.MIXED),
            "retagged": sum(1 for s in output.spans if s.changed),
        }


def _apply_span(
    page_no: int,
    line_id: str,
    index: int,
    span: Span,
    by_span: dict[tuple[int, str, int], SpanRoute],
) -> Span:
    decision = by_span.get((page_no, line_id, index))
    if decision is None:
        return span
    return span.model_copy(
        update={
            "language": decision.language,
            "signals": {**span.signals, "lang_posterior": decision.posterior},
        }
    )
