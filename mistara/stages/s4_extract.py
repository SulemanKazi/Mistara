"""S4 — Extraction: line-anchored VLM transcription.

We detect line geometry locally first, then ask the model to return exactly one
entry per detected line. That buys three things from one call: the text, a
text-to-geometry alignment (which is what makes spatially-faithful output
possible), and a structural check — if the model returns 11 entries for 14
detected lines, we have caught a silent omission with no ground truth involved.

Silent line-skipping is the most common VLM OCR failure, and it is invisible to
any metric that only looks at the returned text.
"""

from __future__ import annotations

import time

from pydantic import BaseModel, Field

from mistara.core.config import ExtractConfig, SegmentConfig
from mistara.core.model import (
    Document,
    Issue,
    Language,
    Line,
    Page,
    Provenance,
    ProvenanceKind,
    Region,
    RegionType,
    ScriptStyle,
    Span,
    ViewName,
)
from mistara.core.stage import Cost, Stage, StageParams, Verdict, Verifier
from mistara.core.store import Store, sha256_bytes
from mistara.providers.registry import get_vlm
from mistara.providers.vlm.base import TranscriptionRequest
from mistara.stages.s0_ingest import load_view, load_view_array
from mistara.stages.segment import (
    RowAssessment,
    assess_rows,
    detect_text_rows,
    split_rows_at_gutters,
)
from mistara.text.canonical import (
    canonical,
    repeat_rate,
    script_purity,
    tashkeel_density,
    urdu_affinity,
)

#: Bump when the prompt changes materially — cached responses were produced by
#: the old wording and should not be reused against new expectations.
PROMPT_VERSION = "2-geometry-grounded"


class ExtractParams(StageParams):
    provider: str = Field(
        default="gemini:gemini-3.7-flash",
        description="Provider spec, e.g. 'gemini:gemini-3.7-flash' or 'stub'.",
    )
    view: ViewName = Field(
        default=ViewName.GRAY,
        description="Which page view to send. VLMs generally do worse on binarized input.",
    )
    line_anchored: bool = Field(
        default=True,
        description="Anchor output to locally detected line boxes (enables omission detection).",
    )
    pages: list[int] | None = Field(default=None, description="Restrict to these page numbers.")
    split_columns: bool = Field(
        default=True,
        description=(
            "Split rows that hold side-by-side cells (Arabic/Urdu pairs, tables, "
            "margin notes) into one box per cell, right-to-left."
        ),
    )
    #: Tunables from the active profile. Embedded rather than looked up so that
    #: `params.hash()` covers them — retuning a threshold re-runs the stage.
    cfg: ExtractConfig = Field(default_factory=ExtractConfig)
    segment: SegmentConfig = Field(default_factory=SegmentConfig)


class PageExtraction(BaseModel):
    page_no: int
    region: Region
    issues: list[Issue] = []
    #: Cells whose returned text is implausible for the box it claims to fill.
    #: Catches text/geometry MISALIGNMENT, which no count-based signal sees: a
    #: model that returns the right lines against the wrong ids produces a
    #: perfect line count and completely wrong spatial output.
    implausible_cells: int = 0
    detected_lines: int
    returned_lines: int
    empty_lines: int
    #: Judgement on the *geometry* we anchored to. Distinct from the judgement
    #: on the text: a page can be transcribed perfectly against line boxes that
    #: were themselves wrong, and that failure is invisible to any text metric.
    rows: RowAssessment | None = None


class ExtractResult(BaseModel):
    pages: list[PageExtraction]
    cost: Cost = Field(default_factory=Cost)
    model_id: str = ""


class ExtractVerifier(Verifier[ExtractResult]):
    """Post-step judgement: did the read work, and did it miss anything?"""

    name = "extract"

    def verify(self, doc: Document, output: ExtractResult, params: ExtractParams) -> Verdict:
        findings: list[str] = []
        metrics: dict[str, float] = {}
        if not output.pages:
            return Verdict.fail("no pages transcribed")

        coverages, purities, repeats = [], [], []
        suspect_geometry = 0
        implausible = 0
        pitch_cvs: list[float] = []

        for pe in output.pages:
            text = pe.region.text

            # -- alignment: is the text in the cell it claims to occupy? ------
            if pe.implausible_cells:
                implausible += pe.implausible_cells
                findings.append(
                    f"page {pe.page_no}: {pe.implausible_cells} cell(s) hold more "
                    f"text than their width can carry — text is probably mapped to "
                    f"the wrong regions"
                )

            # -- geometry: were the line boxes we anchored to actually right? --
            if pe.rows is not None:
                suspect_geometry += pe.rows.suspect_lines
                pitch_cvs.append(pe.rows.pitch_cv)
                findings.extend(
                    f"page {pe.page_no}: {f_}" for f_ in pe.rows.findings()
                )
            # -- coverage: the structural check that catches silent omission --
            coverage = (
                pe.returned_lines / pe.detected_lines if pe.detected_lines else 1.0
            )
            coverages.append(min(coverage, 1.0))
            if pe.detected_lines and coverage < params.cfg.min_line_coverage:
                findings.append(
                    f"page {pe.page_no}: {pe.returned_lines}/{pe.detected_lines} lines "
                    f"returned — likely dropped lines"
                )
            if pe.empty_lines > max(
                1, pe.detected_lines * params.cfg.max_empty_line_frac
            ):
                findings.append(
                    f"page {pe.page_no}: {pe.empty_lines} lines came back empty"
                )

            # -- lexical plausibility: catches mojibake, Latin, markup ---------
            purity = script_purity(text)
            purities.append(purity)
            if text and purity < params.cfg.min_script_purity:
                findings.append(
                    f"page {pe.page_no}: only {purity:.0%} of characters are Arabic-script"
                )

            # -- degeneracy: catches decoding loops ---------------------------
            rep = repeat_rate(text, params.cfg.repeat_ngram)
            repeats.append(rep)
            if rep > params.cfg.max_repeat_rate:
                findings.append(
                    f"page {pe.page_no}: {rep:.0%} of text is one repeated n-gram "
                    f"— probable decoding loop"
                )

        if pitch_cvs:
            metrics["layout.pitch_cv"] = sum(pitch_cvs) / len(pitch_cvs)
            metrics["layout.suspect_lines"] = float(suspect_geometry)
        metrics["alignment.implausible_cells"] = float(implausible)
        metrics["coverage.line_count"] = sum(coverages) / len(coverages)
        metrics["lexical.script_purity"] = sum(purities) / len(purities)
        metrics["degeneracy.repeat_rate"] = max(repeats) if repeats else 0.0
        metrics["pages"] = float(len(output.pages))

        score = min(metrics["coverage.line_count"], metrics["lexical.script_purity"])
        return Verdict(
            passed=not findings, score=score, findings=findings, metrics=metrics
        )


class ExtractStage(Stage[ExtractParams, ExtractResult]):
    name = "s4_extract"
    version = "1"
    params_model = ExtractParams
    verifier = ExtractVerifier()

    def __init__(self, store: Store) -> None:
        self.store = store

    def run(self, doc: Document, params: ExtractParams) -> ExtractResult:
        client = get_vlm(params.provider)
        wanted = params.pages
        results: list[PageExtraction] = []
        total = Cost()

        for page in doc.pages:
            if wanted is not None and page.page_no not in wanted:
                continue
            t0 = time.perf_counter()
            image = load_view(self.store, page, params.view)

            line_boxes = {}
            assessment = None
            if params.line_anchored:
                gray = load_view_array(self.store, page, params.view)
                boxes = detect_text_rows(gray, params.segment)
                # Assess the raw rows: the assessment is what reports *why* a
                # split was needed, so it must run before we apply one.
                assessment = assess_rows(gray, boxes, params.segment)
                if params.split_columns:
                    boxes = split_rows_at_gutters(gray, boxes, config=params.segment)
                line_boxes = {f"L{i:04d}": b for i, b in enumerate(boxes)}

            request = TranscriptionRequest(
                image_png=image,
                line_boxes=line_boxes,
                image_width=page.width_px,
                image_height=page.height_px,
            )
            # Model output is the expensive part of this pipeline and cannot be
            # regenerated for free. Cache the raw response against the exact
            # inputs that produced it, so re-parsing — or a change to how we
            # interpret it — costs nothing.
            result = self._cached_transcribe(client, request, params)
            total = total + Cost(
                provider_calls=1,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                wall_seconds=time.perf_counter() - t0,
            )

            lines = _build_lines(result.lines, line_boxes, page)
            region = Region(
                region_id=f"p{page.page_no:03d}r000",
                bbox=page.bbox,
                region_type=RegionType.PAGE,
                script_style=ScriptStyle.UNKNOWN,
                lines=lines,
                signals={
                    "detected_lines": float(len(line_boxes)),
                    "returned_lines": float(len(lines)),
                },
            )
            results.append(
                PageExtraction(
                    page_no=page.page_no,
                    region=region,
                    detected_lines=len(line_boxes),
                    returned_lines=sum(1 for ln in lines if ln.text.strip()),
                    empty_lines=sum(1 for ln in lines if not ln.text.strip()),
                    rows=assessment,
                    issues=assessment.issues() if assessment else [],
                    implausible_cells=_count_implausible(lines, params),
                )
            )

        return ExtractResult(pages=results, cost=total, model_id=client.model_id)

    def _response_key(self, client, request: TranscriptionRequest, params) -> str:
        from mistara.core.store import stable_hash

        return stable_hash(
            {
                "model": client.model_id,
                "image": sha256_bytes(request.image_png),
                "boxes": {k: v.model_dump() for k, v in request.line_boxes.items()},
                "prompt_version": PROMPT_VERSION,
            }
        )

    def _cached_transcribe(self, client, request: TranscriptionRequest, params):
        key = self._response_key(client, request, params)
        cached = self.store.get_response(key)
        if cached is not None:
            return client.parse_response(cached, request)
        result = client.transcribe(request)
        if result.raw_response:
            self.store.put_response(key, result.raw_response)
        return result

    def apply(self, doc: Document, output: ExtractResult) -> Document:
        by_page = {pe.page_no: pe for pe in output.pages}
        pages: list[Page] = []
        for page in doc.pages:
            pe = by_page.get(page.page_no)
            if pe is None:
                pages.append(page)
                continue
            pages.append(
                page.model_copy(
                    update={
                        "regions": [pe.region],
                        "reading_order": [pe.region.region_id],
                        "issues": pe.issues,
                    }
                )
            )
        return doc.model_copy(update={"pages": pages})

    def counts(self, output: ExtractResult) -> dict[str, int]:
        return {
            "pages": len(output.pages),
            "lines": sum(p.returned_lines for p in output.pages),
            "empty": sum(p.empty_lines for p in output.pages),
        }


def _count_implausible(lines: list[Line], params: ExtractParams) -> int:
    """Cells holding far more text than their width could physically carry.

    A 55px-wide margin note cannot hold a 40-character sentence. When the model
    returns text in its own reading order rather than per region, narrow cells
    fill with body text and wide cells come back empty — and every count-based
    check still passes. This is the signal that catches it.
    """
    widths = [ln.bbox.w for ln in lines if ln.text.strip()]
    texts = [len(ln.text.strip()) for ln in lines if ln.text.strip()]
    if len(widths) < 4:
        return 0
    # Pixels per character, estimated from the page's own typical line.
    ppc = sorted(w / max(t, 1) for w, t in zip(widths, texts, strict=True))
    typical_ppc = ppc[len(ppc) // 2]
    if typical_ppc <= 0:
        return 0

    bad = 0
    for ln in lines:
        chars = len(ln.text.strip())
        if not chars:
            continue
        capacity = ln.bbox.w / typical_ppc
        if chars > capacity * params.cfg.max_capacity_overrun:
            bad += 1
    return bad


def _build_lines(reads, line_boxes: dict, page: Page) -> list[Line]:
    """Attach transcribed text to geometry, and tag each line's language."""
    lines: list[Line] = []
    ids = list(line_boxes)

    for i, read in enumerate(reads):
        box = line_boxes.get(read.line_id)
        if box is None:
            # Whole-page mode, or an id the model invented: fall back to
            # positional matching, then to a zero-height placeholder box so the
            # text is never silently dropped from the output.
            box = line_boxes[ids[i]] if i < len(ids) else page.bbox
        text = canonical(read.text, "strict", keep_newlines=False)
        lines.append(
            Line(
                line_id=read.line_id,
                bbox=box,
                text=text,
                spans=[
                    Span(
                        span_id=f"{read.line_id}s0",
                        char_start=0,
                        char_end=len(text),
                        language=_guess_language(text),
                        provenance=Provenance(kind=ProvenanceKind.OCR),
                    )
                ]
                if text
                else [],
                signals={
                    "script_purity": script_purity(text),
                    "urdu_affinity": urdu_affinity(text),
                    "tashkeel_density": tashkeel_density(text),
                },
            )
        )
    return lines


def _guess_language(text: str) -> Language:
    """Provisional tag only — S5 (`stages/s5_route.py`) is authoritative and
    supersedes this once it runs. It exists so `order`, which runs before
    `match`, has *a* language to check column purity against; it reads
    orthography alone, so it mis-tags Quranic Arabic set in an Urdu typeface."""
    if not text.strip():
        return Language.UNKNOWN
    affinity = urdu_affinity(text)
    vocalized = tashkeel_density(text)
    if vocalized > 0.25 and affinity <= 0:
        return Language.ARABIC
    if affinity > 0.2:
        return Language.URDU
    if affinity < -0.2:
        return Language.ARABIC
    return Language.MIXED
