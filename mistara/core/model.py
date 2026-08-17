"""The document model — the single accumulating state for a run.

Every stage declares its own typed input/output models (see `stage.py`); the
runtime merges a stage's output into the `Document` below. So stages stay
independently testable while there is still exactly one canonical state object.

Coordinates are always **pixels in the page image's own frame**, origin at the
top-left. Text is always stored in **logical Unicode order** — never visually
reordered. Bidi is a rendering concern, not a storage concern.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "1"


class Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=False, validate_assignment=False)


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #


class BBox(Frozen):
    """Axis-aligned box in page-image pixel space, origin top-left."""

    x: float
    y: float
    w: float
    h: float

    @property
    def x1(self) -> float:
        return self.x + self.w

    @property
    def y1(self) -> float:
        return self.y + self.h

    @property
    def area(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)

    def iou(self, other: BBox) -> float:
        ix = max(0.0, min(self.x1, other.x1) - max(self.x, other.x))
        iy = max(0.0, min(self.y1, other.y1) - max(self.y, other.y))
        inter = ix * iy
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def contains(self, other: BBox, tol: float = 2.0) -> bool:
        return (
            other.x >= self.x - tol
            and other.y >= self.y - tol
            and other.x1 <= self.x1 + tol
            and other.y1 <= self.y1 + tol
        )


# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #


class RegionType(StrEnum):
    MAIN_TEXT = "main_text"
    INTERLINEAR_TRANSLATION = "interlinear_translation"
    MARGINALIA = "marginalia"
    HEADER = "header"
    FOOTER = "footer"
    FOOTNOTE = "footnote"
    ORNAMENT = "ornament"
    PAGE = "page"  # whole-page baseline, before real layout analysis
    UNKNOWN = "unknown"


class ScriptStyle(StrEnum):
    NASTALIQ = "nastaliq"
    NASKH = "naskh"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class Language(StrEnum):
    URDU = "ur"
    ARABIC = "ar"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class LinkType(StrEnum):
    TRANSLATION = "translation"
    COMMENTARY = "commentary"
    CONTINUATION = "continuation"
    FOOTNOTE_REF = "footnote_ref"


class ProvenanceKind(StrEnum):
    """Where a span's text came from. v1 keeps this deliberately coarse."""

    OCR = "ocr"
    QURAN = "quran"
    HADITH = "hadith"
    LLM_CORRECTED = "llm_corrected"
    UNRESOLVED = "unresolved"


class ViewName(StrEnum):
    RAW = "raw"
    GRAY = "gray"
    DESKEWED = "deskewed"
    NORMALIZED = "normalized"
    BINARIZED = "binarized"


# --------------------------------------------------------------------------- #
# Content
# --------------------------------------------------------------------------- #


class Provenance(Frozen):
    """Coarse source tag for a span.

    `original_text` is retained whenever text is replaced or corrected. It is the
    one piece of the audit story that is load-bearing rather than nice-to-have:
    without it, a wrong corpus match silently destroys the OCR reading with no
    way to recover or diagnose it.
    """

    kind: ProvenanceKind = ProvenanceKind.OCR
    source_ref: str | None = None  # "quran:2:255", "bukhari:1:1"
    original_text: str | None = None
    score: float | None = None  # match identity, edit confidence, etc.


class Span(Frozen):
    span_id: str
    char_start: int
    char_end: int
    language: Language = Language.UNKNOWN
    provenance: Provenance = Field(default_factory=Provenance)
    signals: dict[str, float] = Field(default_factory=dict)

    def text_of(self, line_text: str) -> str:
        return line_text[self.char_start : self.char_end]


class Line(Frozen):
    line_id: str
    bbox: BBox
    text: str = ""
    spans: list[Span] = Field(default_factory=list)
    signals: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_spans(self) -> Self:
        for s in self.spans:
            if not (0 <= s.char_start <= s.char_end <= len(self.text)):
                raise ValueError(
                    f"span {s.span_id} range [{s.char_start},{s.char_end}] "
                    f"outside line text of length {len(self.text)}"
                )
        return self


class Region(Frozen):
    region_id: str
    bbox: BBox
    region_type: RegionType = RegionType.UNKNOWN
    script_style: ScriptStyle = ScriptStyle.UNKNOWN
    lines: list[Line] = Field(default_factory=list)
    signals: dict[str, float] = Field(default_factory=dict)

    @property
    def text(self) -> str:
        return "\n".join(ln.text for ln in self.lines)


class Link(Frozen):
    src_region_id: str
    dst_region_id: str
    link_type: LinkType
    confidence: float = 0.0


class ArtifactRef(Frozen):
    """A pointer into the content-addressed store."""

    sha256: str
    media_type: str = "image/png"
    width: int | None = None
    height: int | None = None
    note: str | None = None


class IssueKind(StrEnum):
    """Things a verifier suspects are wrong, with the geometry to show them."""

    MERGED_LINES = "merged_lines"  # one box holding two or more lines
    MISSED_LINE = "missed_line"  # ink no box claims
    COLUMN_GUTTER = "column_gutter"  # a row split into side-by-side cells
    LOW_CONFIDENCE = "low_confidence"  # text-level doubt


class Issue(Frozen):
    """A suspected defect, located on the page.

    Verifiers produce these alongside their scalar metrics. A number tells you
    *that* something is wrong; this tells you *where*, which is what makes the
    finding actionable — for a human eyeballing the render, and for an agent
    deciding which region to re-run.
    """

    kind: IssueKind
    bbox: BBox
    note: str = ""


class Page(Frozen):
    page_no: int  # 0-based
    width_px: int
    height_px: int
    source_dpi: float | None = None  # estimated DPI of the *source* pixels
    views: dict[ViewName, ArtifactRef] = Field(default_factory=dict)
    regions: list[Region] = Field(default_factory=list)
    reading_order: list[str] = Field(default_factory=list)
    links: list[Link] = Field(default_factory=list)
    issues: list[Issue] = Field(default_factory=list)
    signals: dict[str, float] = Field(default_factory=dict)

    @property
    def bbox(self) -> BBox:
        return BBox(x=0, y=0, w=self.width_px, h=self.height_px)

    def region(self, region_id: str) -> Region | None:
        return next((r for r in self.regions if r.region_id == region_id), None)

    def ordered_regions(self) -> list[Region]:
        """Regions in reading order; any not listed are appended in raw order."""
        by_id = {r.region_id: r for r in self.regions}
        out = [by_id[rid] for rid in self.reading_order if rid in by_id]
        seen = {r.region_id for r in out}
        out.extend(r for r in self.regions if r.region_id not in seen)
        return out

    @property
    def text(self) -> str:
        return "\n\n".join(r.text for r in self.ordered_regions() if r.text)


class HistoryEntry(Frozen):
    """One stage invocation. Enough to reproduce, not a full audit record."""

    stage: str
    version: str
    params_hash: str
    at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    passed: bool | None = None
    note: str | None = None


class Document(Frozen):
    doc_id: str
    source_path: str
    source_sha256: str
    schema_version: str = SCHEMA_VERSION
    pages: list[Page] = Field(default_factory=list)
    history: list[HistoryEntry] = Field(default_factory=list)

    def page(self, page_no: int) -> Page:
        for p in self.pages:
            if p.page_no == page_no:
                return p
        raise KeyError(f"page {page_no} not in document {self.doc_id}")

    @property
    def text(self) -> str:
        return "\n\n".join(p.text for p in self.pages)
