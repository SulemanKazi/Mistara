"""S3 — Reading order: group cells into regions, then sequence them.

S4 hands back one flat list of cells per page, in the order geometry produced
them: top to bottom, and right-cell-before-left-cell within each row. For a page
of plain prose that is already correct. For a side-by-side block it is not — it
alternates between two columns, so an Arabic passage and its Urdu translation
come out interleaved line by line and neither reads continuously.

This stage fixes that by recovering the column structure the cells came from and
grouping each column into its own region:

    row-major (what geometry gives us)   column-major (what these pages mean)
    ┌──────────┬──────────┐              ┌──────────┬──────────┐
    │ ar 1  ①  │ ur 1  ②  │              │ ar 1  ①  │ ur 1  ④  │
    │ ar 2  ③  │ ur 2  ④  │      →       │ ar 2  ②  │ ur 2  ⑤  │
    │ ar 3  ⑤  │ ur 3  ⑥  │              │ ar 3  ③  │ ur 3  ⑥  │
    └──────────┴──────────┘              └──────────┴──────────┘

It works entirely on the boxes already in the `Document` — no image, no store,
no model. That is deliberate: it makes the stage trivially testable against
synthetic geometry, and it runs against a dumped result file with no cache
present.

**What this stage does not know.** Column-major is right for parallel text and
wrong for a table, and the two are geometrically identical — both are a run of
rows sharing a gutter. Telling them apart is region typing, which is S2 and is
not built. So the choice is a per-book config setting (`order.column_read_order`)
with a default fitted to the observed sample, not a detection. The verifier
reports the language evidence that would support or undermine it rather than
silently assuming it was right.
"""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel, Field

from mistara.core.config import OrderConfig, SegmentConfig
from mistara.core.model import (
    BBox,
    Document,
    Language,
    Line,
    Page,
    Region,
    RegionType,
)
from mistara.core.stage import Stage, StageParams, Verdict, Verifier
from mistara.core.store import stable_hash

COLUMN_MAJOR = "column_major"
ROW_MAJOR = "row_major"


def language_purity(languages: Iterable[Language]) -> float:
    """Share of tagged lines agreeing with the majority language.

    Shared by S3 (checking its own column grouping) and S5 (checking that the
    grouping still holds under the corpus-informed tags it produces). `UNKNOWN`
    lines carry no evidence and are skipped; no tagged line at all is vacuously
    pure.
    """
    langs = [lang for lang in languages if lang is not Language.UNKNOWN]
    if not langs:
        return 1.0
    top = max(set(langs), key=langs.count)
    return langs.count(top) / len(langs)


class OrderParams(StageParams):
    cfg: OrderConfig = Field(default_factory=OrderConfig)
    #: Only `columns` is read, but the whole section is embedded so the params
    #: hash covers any retuning of it.
    segment: SegmentConfig = Field(default_factory=SegmentConfig)


# --------------------------------------------------------------------------- #
# Grouping — pure geometry over cell boxes
# --------------------------------------------------------------------------- #


class Group(BaseModel):
    """One region's worth of lines, plus why they were grouped together."""

    lines: list[Line]
    #: 0 is the rightmost column. None for a single-column band.
    column_index: int | None = None
    #: How many columns the block this came from had. 1 for ordinary prose.
    columns: int = 1
    #: Which side-by-side block this came from, so the columns of one block can
    #: be told from the columns of the next. None for ordinary prose.
    block: int | None = None
    marginalia: bool = False

    @property
    def languages(self) -> list[Language]:
        return [
            ln.spans[0].language
            for ln in self.lines
            if ln.spans and ln.spans[0].language is not Language.UNKNOWN
        ]

    def purity(self) -> float:
        """Share of lines agreeing with this group's majority language.

        Text evidence for a decision made on geometry: a column that holds both
        Arabic and Urdu was probably split in the wrong place.
        """
        return language_purity(self.languages)


def _bands(lines: list[Line], overlap_frac: float) -> list[list[Line]]:
    """Group cells into rows by vertical overlap; cells ordered right-to-left.

    Cells split from a single row share a y-extent almost exactly, so this is
    a recovery of information S4 had and discarded, not an inference.
    """
    bands: list[list[Line]] = []
    current: list[Line] = []
    y0 = y1 = 0.0

    for line in sorted(lines, key=lambda ln: (ln.bbox.y, -ln.bbox.x)):
        b = line.bbox
        if current:
            overlap = min(y1, b.y1) - max(y0, b.y)
            reference = min(b.h, y1 - y0) or 1.0
            if overlap / reference >= overlap_frac:
                current.append(line)
                y0, y1 = min(y0, b.y), max(y1, b.y1)
                continue
            bands.append(current)
        current = [line]
        y0, y1 = b.y, b.y1
    if current:
        bands.append(current)

    # Right-to-left within a row: these pages read RTL, so the rightmost cell
    # is the first one.
    return [sorted(band, key=lambda ln: -ln.bbox.x) for band in bands]


def _gutters(band: list[Line]) -> list[tuple[float, float]] | None:
    """Ink-free intervals between adjacent cells, left to right.

    `None` if the cells overlap horizontally, which means they are not a clean
    set of side-by-side columns and should not be treated as one.
    """
    cells = sorted(band, key=lambda ln: ln.bbox.x)
    out: list[tuple[float, float]] = []
    for left, right in zip(cells, cells[1:], strict=False):
        if right.bbox.x <= left.bbox.x1:
            return None
        out.append((left.bbox.x1, right.bbox.x))
    return out


def _column_runs(
    bands: list[list[Line]], *, align_overlap_px: float, min_rows: int
) -> dict[int, tuple[int, int]]:
    """Map band index → (run_id, column_count) for bands in a column block.

    A run is consecutive bands with the same cell count whose gutters line up.
    Alignment is tested by **interval overlap**, not by distance between gutter
    centres: gutter width varies down a block (one row's cells nearly touch
    while another's are far apart), which moves the centre a long way without
    moving the boundary at all.
    """
    assigned: dict[int, tuple[int, int]] = {}
    run: list[int] = []
    run_id = 0

    def flush() -> None:
        nonlocal run_id
        if len(run) >= min_rows:
            ncols = len(bands[run[0]])
            for i in run:
                assigned[i] = (run_id, ncols)
            run_id += 1
        run.clear()

    for i, band in enumerate(bands):
        gutters = _gutters(band) if len(band) > 1 else None
        if gutters is None:
            flush()
            continue
        if run:
            previous = _gutters(bands[run[-1]])
            aligned = previous is not None and len(previous) == len(gutters) and all(
                min(p[1], g[1]) - max(p[0], g[0]) >= align_overlap_px
                for p, g in zip(previous, gutters, strict=True)
            )
            if not aligned:
                flush()
        run.append(i)
    flush()
    return assigned


def group_page(
    lines: list[Line], cfg: OrderConfig, segment: SegmentConfig
) -> list[Group]:
    """Group a page's cells into regions, in reading order."""
    if not lines:
        return []

    col = segment.columns
    bands = _bands(lines, cfg.band_overlap_frac)
    runs = _column_runs(
        bands, align_overlap_px=col.align_overlap_px, min_rows=col.min_rows
    )

    groups: list[Group] = []
    prose: list[Line] = []

    def flush_prose() -> None:
        if prose:
            groups.append(Group(lines=list(prose)))
            prose.clear()

    i = 0
    while i < len(bands):
        entry = runs.get(i)
        if entry is None:
            prose.extend(bands[i])
            i += 1
            continue

        run_id, ncols = entry
        block = []
        while i < len(bands) and runs.get(i) == (run_id, ncols):
            block.append(bands[i])
            i += 1

        flush_prose()
        groups.extend(_block_groups(block, ncols, run_id, cfg, segment))

    flush_prose()
    return groups


def _block_groups(
    block: list[list[Line]],
    ncols: int,
    block_id: int,
    cfg: OrderConfig,
    segment: SegmentConfig,
) -> list[Group]:
    """Turn one side-by-side block into regions, in reading order."""
    if cfg.column_read_order == ROW_MAJOR:
        # Read across each row. Already how the cells arrive, so this is just a
        # flatten — but it is still emitted as its own region, so the block
        # remains visible as a unit rather than dissolving into the prose.
        return [
            Group(
                lines=[ln for band in block for ln in band],
                columns=ncols,
                block=block_id,
            )
        ]

    columns = [[band[c] for band in block] for c in range(ncols)]  # 0 = rightmost
    widths = [
        sorted(ln.bbox.w for ln in column)[len(column) // 2] for column in columns
    ]
    # A column much narrower than its neighbours is a margin note, not half of a
    # parallel-text pair. It reads after the body it sits beside — a convention,
    # not something the geometry can tell us.
    widest = max(widths) or 1.0
    marginal = [w / widest < segment.columns.margin_note_ratio for w in widths]

    groups = [
        Group(
            lines=column,
            column_index=c,
            columns=ncols,
            block=block_id,
            marginalia=marginal[c],
        )
        for c, column in enumerate(columns)
    ]
    return sorted(groups, key=lambda g: (g.marginalia, g.column_index or 0))


def regions_for_page(
    page: Page, cfg: OrderConfig, segment: SegmentConfig
) -> list[Region]:
    lines = [ln for region in page.regions for ln in region.lines]
    groups = group_page(lines, cfg, segment)
    return [
        Region(
            region_id=f"p{page.page_no:03d}r{i:03d}",
            bbox=_hull(g.lines),
            region_type=(
                RegionType.MARGINALIA if g.marginalia else RegionType.UNKNOWN
            ),
            lines=g.lines,
            signals={
                "columns": float(g.columns),
                "column_index": float(
                    g.column_index if g.column_index is not None else -1
                ),
                "block": float(g.block if g.block is not None else -1),
                "column_purity": g.purity(),
                "lines": float(len(g.lines)),
            },
        )
        for i, g in enumerate(groups)
    ]


def _hull(lines: list[Line]) -> BBox:
    xs = [ln.bbox.x for ln in lines]
    ys = [ln.bbox.y for ln in lines]
    x1s = [ln.bbox.x1 for ln in lines]
    y1s = [ln.bbox.y1 for ln in lines]
    return BBox(x=min(xs), y=min(ys), w=max(x1s) - min(xs), h=max(y1s) - min(ys))


# --------------------------------------------------------------------------- #
# Stage
# --------------------------------------------------------------------------- #


class PageOrder(BaseModel):
    page_no: int
    regions: list[Region]
    lines_in: int
    lines_out: int
    column_blocks: int


class OrderResult(BaseModel):
    pages: list[PageOrder] = Field(default_factory=list)


class OrderVerifier(Verifier[OrderResult]):
    """Two checks: nothing was lost, and the columns look like real columns."""

    name = "order"

    def verify(self, doc: Document, output: OrderResult, params: OrderParams) -> Verdict:
        if not output.pages:
            return Verdict.fail("no transcribed lines to order — run extract first")

        findings: list[str] = []
        purities: list[float] = []
        lines_in = sum(p.lines_in for p in output.pages)
        lines_out = sum(p.lines_out for p in output.pages)
        blocks = sum(p.column_blocks for p in output.pages)

        for page in output.pages:
            # Structural invariant. Grouping must be a permutation of its input:
            # a bug here silently drops or duplicates transcribed text, which no
            # text metric would attribute to the ordering stage.
            if page.lines_in != page.lines_out:
                findings.append(
                    f"page {page.page_no}: grouped {page.lines_out} of "
                    f"{page.lines_in} lines — ordering lost or duplicated text"
                )
            for region in page.regions:
                if region.signals.get("columns", 1) < 2:
                    continue
                if len(region.lines) < params.cfg.min_lines_for_purity:
                    continue
                purity = region.signals.get("column_purity", 1.0)
                purities.append(purity)
                if purity < params.cfg.min_column_purity:
                    findings.append(
                        f"page {page.page_no}: column region {region.region_id} "
                        f"mixes languages ({purity:.0%} agree) — either the gutter "
                        f"is in the wrong place or the language tags are (S5 "
                        f"routing is provisional; these books set Arabic in an "
                        f"Urdu typeface, which fools orthography alone)"
                    )

        conservation = lines_out / lines_in if lines_in else 1.0
        metrics = {
            "order.line_conservation": conservation,
            "order.regions": float(sum(len(p.regions) for p in output.pages)),
            "order.column_blocks": float(blocks),
        }
        if purities:
            metrics["order.column_purity"] = sum(purities) / len(purities)

        return Verdict(
            passed=not findings,
            score=min(conservation, metrics.get("order.column_purity", 1.0)),
            findings=findings,
            metrics=metrics,
        )


class OrderStage(Stage[OrderParams, OrderResult]):
    name = "s3_order"
    version = "1"
    params_model = OrderParams
    verifier = OrderVerifier()

    def run(self, doc: Document, params: OrderParams) -> OrderResult:
        pages: list[PageOrder] = []
        for page in doc.pages:
            lines_in = sum(len(r.lines) for r in page.regions)
            if not lines_in:
                continue
            regions = regions_for_page(page, params.cfg, params.segment)
            pages.append(
                PageOrder(
                    page_no=page.page_no,
                    regions=regions,
                    lines_in=lines_in,
                    lines_out=sum(len(r.lines) for r in regions),
                    column_blocks=len(
                        {
                            r.signals["block"]
                            for r in regions
                            if r.signals.get("block", -1) >= 0
                        }
                    ),
                )
            )
        return OrderResult(pages=pages)

    def apply(self, doc: Document, output: OrderResult) -> Document:
        by_page = {p.page_no: p for p in output.pages}
        pages = [
            page.model_copy(
                update={
                    "regions": by_page[page.page_no].regions,
                    "reading_order": [
                        r.region_id for r in by_page[page.page_no].regions
                    ],
                }
            )
            if page.page_no in by_page
            else page
            for page in doc.pages
        ]
        return doc.model_copy(update={"pages": pages})

    def input_hash(self, doc: Document, params: OrderParams) -> str:
        """Keyed on the geometry we group, not on the source PDF.

        The default key would not change when a re-extraction produced different
        cells, so an ordering result would be served from cache against boxes it
        never saw.
        """
        return stable_hash(
            {
                "boxes": [
                    [ln.line_id, *ln.bbox.model_dump().values()]
                    for page in doc.pages
                    for region in page.regions
                    for ln in region.lines
                ],
                "params": params.hash(),
            }
        )

    def counts(self, output: OrderResult) -> dict[str, int]:
        return {
            "pages": len(output.pages),
            "regions": sum(len(p.regions) for p in output.pages),
            "blocks": sum(p.column_blocks for p in output.pages),
        }
