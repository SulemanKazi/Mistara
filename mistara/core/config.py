"""Tunable parameters, in one place, overridable per book.

Every threshold in this project was fitted to one book at one scan quality. A
different print run, a different scanner, or a different typesetter will move
them. Keeping them here means retuning is editing a TOML file rather than
hunting literals through the source.

Defaults live in the models below; a TOML file overrides only what it names, so
a per-book config can be three lines long. Config values are embedded in each
stage's params, which means they are part of the cache key and are recorded in
the run ledger — change a threshold and affected stages re-run, and you can
always see which settings produced a given result.

    from mistara.core.config import MistaraConfig
    cfg = MistaraConfig.load("config/tafsir.toml")
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field

CONFIG_DIR = Path("config")


class Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- #
# Image preparation
# --------------------------------------------------------------------------- #


class BinarizeConfig(Section):
    """Adaptive threshold separating ink from paper.

    The single most scan-dependent setting here. Faded or unevenly lit scans
    want a larger block and a smaller constant; crisp high-contrast scans
    tolerate the reverse.
    """

    block_size: int = Field(
        default=31, description="Neighbourhood size, odd. Roughly a character's width."
    )
    c: float = Field(
        default=10.0, description="Subtracted from the local mean. Higher keeps less ink."
    )


class IngestConfig(Section):
    target_dpi: float = Field(
        default=400.0,
        description="Rasterization DPI, used ONLY for pages with no full-page image.",
    )
    native_image_coverage: float = Field(
        default=0.55,
        description=(
            "Fraction of page area one image must cover to be extracted natively. "
            "Scans are often letterboxed inside a Letter/A4 page, so this is not near 1."
        ),
    )
    low_dpi_warning: float = Field(
        default=150.0,
        description="Below this, nastaliq dot placement (i'jam) starts to be unrecoverable.",
    )
    blank_page_ink_ratio: float = Field(
        default=0.002, description="Ink share below which a page is called blank."
    )
    ink_level: int = Field(
        default=200, description="Grey level below which a pixel counts as ink (0-255)."
    )


# --------------------------------------------------------------------------- #
# Geometry — vertical axis
# --------------------------------------------------------------------------- #


class LineConfig(Section):
    """Finding text lines by horizontal projection of ink."""

    min_gap_px: int = Field(
        default=3, description="Ink-free rows needed to end a line band."
    )
    min_height_px: int = Field(default=5, description="Bands shorter than this are noise.")
    threshold_scale: float = Field(
        default=0.30,
        description=(
            "Fraction of the mean row-ink a row must reach to count as text. "
            "LOWERING THIS DOES NOT RECOVER SHORT LINES — it lifts the valleys "
            "between lines too, so bands fuse and the line count drops."
        ),
    )
    smooth_window: int = Field(
        default=3, description="Rows of smoothing applied to the projection profile."
    )


class AssessConfig(Section):
    """Judging a line segmentation against the page's own baseline regularity."""

    tall_factor: float = Field(
        default=1.6,
        description="Box height over this multiple of the median implies a merge.",
    )
    gap_factor: float = Field(
        default=1.5,
        description="Gap over this multiple of the median pitch is a candidate miss.",
    )
    stranded_ink_frac: float = Field(
        default=0.08,
        description=(
            "Share of a typical row's ink that must sit in an outsized gap for it "
            "to count as a missed line. Separates a real short line (~12% on the "
            "sample) from header whitespace speckle (~3%)."
        ),
    )
    max_pitch_cv: float = Field(
        default=0.35,
        description="Pitch variation above this suggests we are not tracking the grid.",
    )
    min_rows_for_assessment: int = Field(
        default=4, description="Below this many lines, regularity is meaningless."
    )


# --------------------------------------------------------------------------- #
# Geometry — horizontal axis
# --------------------------------------------------------------------------- #


class ColumnConfig(Section):
    """Finding side-by-side cells: Arabic/Urdu pairs, tables, margin notes."""

    min_rows: int = Field(
        default=2,
        description=(
            "Consecutive rows that must share a gutter. Raising this is safer but "
            "misses genuine single-row side-by-side layouts by construction."
        ),
    )
    align_overlap_px: float = Field(
        default=4.0,
        description=(
            "Overlap required between consecutive rows' gutters. Overlap, not "
            "centre distance: a wide gutter's centre sits far from a narrow one's "
            "even at the same column boundary."
        ),
    )
    min_gutter_ratio: float = Field(
        default=3.0,
        description="Gutter must exceed this multiple of the page's typical word gap.",
    )
    min_gutter_frac_of_height: float = Field(
        default=0.4, description="...and this fraction of the median line height."
    )
    edge_margin_px: int = Field(
        default=5,
        description="Ink-free runs this close to a box edge are margins, not gutters.",
    )
    margin_note_ratio: float = Field(
        default=0.35,
        description=(
            "Narrower/wider cell ratio below which a split is a margin note rather "
            "than a table of comparable columns."
        ),
    )


class SegmentConfig(Section):
    binarize: BinarizeConfig = Field(default_factory=BinarizeConfig)
    lines: LineConfig = Field(default_factory=LineConfig)
    assess: AssessConfig = Field(default_factory=AssessConfig)
    columns: ColumnConfig = Field(default_factory=ColumnConfig)


# --------------------------------------------------------------------------- #
# Reading order
# --------------------------------------------------------------------------- #


class OrderConfig(Section):
    """Grouping cells into regions and putting those regions in reading order.

    Deliberately small: the geometric thresholds this needs (how well two rows'
    gutters must line up, when a narrow cell is a margin note, how many rows make
    a column block) already exist under `segment.columns` and are reused rather
    than duplicated. A threshold defined twice is a threshold that will be tuned
    in one place and not the other.
    """

    band_overlap_frac: float = Field(
        default=0.5,
        description=(
            "Share of the shorter cell's height that two cells must overlap "
            "vertically to count as the same row. Cells split from one row share "
            "a y-extent almost exactly, so this is not a sensitive setting."
        ),
    )
    column_read_order: str = Field(
        default="column_major",
        description=(
            "'column_major' reads each column of a side-by-side block to its end "
            "before starting the next, right to left — correct for parallel text, "
            "where the Arabic passage is followed by its Urdu translation. "
            "'row_major' reads across each row instead, which is how a table "
            "reads. We cannot yet tell the two structures apart (that is region "
            "typing, S2), so this is a per-book choice, not a detection."
        ),
    )
    min_column_purity: float = Field(
        default=0.8,
        description=(
            "Share of a column region's lines that must share its majority "
            "language before the grouping is trusted. A column holding both "
            "Arabic and Urdu suggests the gutter was in the wrong place — text "
            "evidence for a decision made purely on geometry. Provisional."
        ),
    )
    min_lines_for_purity: int = Field(
        default=3,
        description=(
            "Columns shorter than this are not purity-checked. Purity over two "
            "lines can only be 0, 0.5 or 1 — it carries no information, and "
            "margin notes (which are short by nature, and whose two-word "
            "fragments the language heuristic reads least reliably) would "
            "otherwise generate a finding each. Same reasoning as "
            "`segment.assess.min_rows_for_assessment`."
        ),
    )


class RouteConfig(Section):
    """S5 — per-span language routing from corpus hits and text evidence.

    A corpus hit (a span the matcher verified against the Quran/Hadith) is
    decisive and needs no weighting: it is Arabic by construction. Everything
    below only scores the spans the matcher did *not* resolve, combining
    orthography, vocalization, and Quranic ornaments into one signed score
    (−1 = certain Arabic, +1 = certain Urdu). Image typeface is deliberately not
    used: these books set Arabic in an Urdu typeface, so naskh/nastaliq is
    unreliable evidence here rather than helpful.

    All fitted to the two sample documents — provisional, like every threshold.
    """

    w_urdu: float = Field(
        default=1.0,
        description="Weight on Urdu-exclusive orthography (`urdu_affinity`).",
    )
    w_tashkeel: float = Field(
        default=1.0,
        description="Weight on vocalization as evidence of Arabic.",
    )
    w_ornament: float = Field(
        default=1.0,
        description="Weight on Quranic ornamental brackets (﴾ ﴿) as evidence of Arabic.",
    )
    tashkeel_arabic: float = Field(
        default=0.25,
        description=(
            "Tashkeel density at which vocalization counts as full evidence of "
            "Arabic. Density is ramped linearly to 1.0 at this value. Matches "
            "the `VOCALIZED_HINT` the matcher uses to count unresolved lines."
        ),
    )
    urdu_cut: float = Field(
        default=0.2,
        description="Score at or above this routes the span Urdu.",
    )
    arabic_cut: float = Field(
        default=-0.2,
        description=(
            "Score at or below this routes the span Arabic. Between the two cuts "
            "the evidence is genuinely mixed and the span is left Mixed."
        ),
    )


# --------------------------------------------------------------------------- #
# Text quality
# --------------------------------------------------------------------------- #


class ExtractConfig(Section):
    """Thresholds the extraction verifier judges returned text against."""

    min_script_purity: float = Field(
        default=0.55,
        description="Share of characters that must be Arabic-script for a sane read.",
    )
    max_repeat_rate: float = Field(
        default=0.25,
        description="Share of text one repeated n-gram may occupy before it reads "
        "as a decoding loop.",
    )
    repeat_ngram: int = Field(
        default=12, description="n-gram length used to measure repetition."
    )
    min_line_coverage: float = Field(
        default=0.95,
        description="Returned lines / detected lines below this means dropped lines.",
    )
    max_empty_line_frac: float = Field(
        default=0.10, description="Share of lines that may come back empty."
    )
    max_capacity_overrun: float = Field(
        default=2.5,
        description=(
            "How far a cell's character count may exceed what its width could "
            "physically carry before it is called misaligned. Catches text "
            "mapped to the wrong region, which no count-based check sees."
        ),
    )


# --------------------------------------------------------------------------- #
# Corpus matching
# --------------------------------------------------------------------------- #


class MatchConfig(Section):
    """When a corpus alignment is good enough to act on.

    The governing target is a **false replacement rate of zero**: it is far
    better to leave a verse as OCR than to overwrite the author's page with the
    wrong text. Every default here is set on that side.
    """

    min_identity: float = Field(
        default=0.92,
        description=(
            "Character identity within the aligned span. Governs whether the "
            "TEXT is trustworthy."
        ),
    )
    min_coverage: float = Field(
        default=0.85,
        description=(
            "Share of the query the alignment consumed. Distinguishes 'this line "
            "is a verse' from 'this line contains a verse'."
        ),
    )
    min_span_tokens: int = Field(
        default=4,
        description=(
            "Shortest alignment that may be acted on. Without this a single "
            "common word shared with the Quran matches Urdu prose at identity "
            "1.0 — measured, not hypothetical. Coverage alone does not catch it, "
            "because an inline quote is a low-coverage match by definition."
        ),
    )
    min_query_tokens: int = Field(
        default=3, description="Queries shorter than this are not looked up at all."
    )
    min_continuation_tokens: int = Field(
        default=1,
        description=(
            "A shorter floor, allowed ONLY for a line that contiguously continues "
            "the passage the previous accepted line matched. A 2-3 word verse tail "
            "like `مُہْتَدُوْنَ (82 انعام)` falls below `min_span_tokens` and would "
            "otherwise resolve to nothing. Continuity is the guard the length floor "
            "gives up: a random short line will not also align to the exact next "
            "corpus position, so this does not reopen the false-replacement risk "
            "that `min_span_tokens` closes for isolated lines."
        ),
    )
    continuation_max_gap: int = Field(
        default=3,
        description=(
            "How far a match may start past where the previous accepted line "
            "ended and still count as continuing it — a line break can drop a "
            "word. Same tolerance the run-continuity verifier uses."
        ),
    )
    min_margin: float = Field(
        default=0.05,
        description=(
            "Best identity minus the best identity elsewhere in the corpus. "
            "Below this the WORDING is certain but the CITATION is a guess — "
            "Quranic phrases repeat verbatim (e.g. 5:44, 5:45 and 5:47 share a "
            "clause). Reported, not blocking: it does not make the text wrong."
        ),
    )
    replace_inline: bool = Field(
        default=False,
        description=(
            "Whether a quote found *inside* a line of prose may be rewritten "
            "from the corpus, or only cited. Off by default: the span boundary "
            "is inferred, and a boundary error would overwrite the author's own "
            "words rather than a verse."
        ),
    )
    candidates: int = Field(
        default=8, description="Corpus windows aligned per query."
    )
    window_slack: int = Field(
        default=6, description="Tokens of padding around a candidate window."
    )
    gap_penalty: float = Field(
        default=-1.0, description="Smith-Waterman gap penalty, in token units."
    )
    min_token_similarity: float = Field(
        default=0.5,
        description=(
            "Below this, two tokens count as unrelated rather than partly alike. "
            "Stops a run of vaguely-similar Urdu words accumulating a score."
        ),
    )
    tie_epsilon: float = Field(
        default=0.75,
        description=(
            "Alignment scores within this of the best are treated as tied, and "
            "broken in favour of the candidate that continues the passage the "
            "previous line matched. Reading order is evidence the aligner cannot "
            "see on its own."
        ),
    )


# --------------------------------------------------------------------------- #
# Root
# --------------------------------------------------------------------------- #


class MistaraConfig(Section):
    name: str = Field(default="default", description="Identifies this profile in reports.")
    notes: str = Field(default="", description="Free text: which book, which scan batch.")
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    segment: SegmentConfig = Field(default_factory=SegmentConfig)
    extract: ExtractConfig = Field(default_factory=ExtractConfig)
    order: OrderConfig = Field(default_factory=OrderConfig)
    match: MatchConfig = Field(default_factory=MatchConfig)
    route: RouteConfig = Field(default_factory=RouteConfig)

    @classmethod
    def load(cls, path: Path | str | None) -> Self:
        """Load a TOML profile. `None` yields defaults.

        Unknown keys are rejected rather than ignored — a typo in a threshold
        name would otherwise silently leave the default in place, which is the
        worst possible failure for a tuning file.
        """
        if path is None:
            return cls()
        p = Path(path)
        if not p.exists() and not p.is_absolute() and p.parent == Path():
            p = CONFIG_DIR / p  # allow `--config tafsir.toml`
        if not p.exists():
            raise FileNotFoundError(f"no config at {path!r} (looked in {p})")
        with p.open("rb") as fh:
            return cls.model_validate(tomllib.load(fh))

    def to_toml(self) -> str:
        """Serialize back to TOML. Round-trips through `load`."""
        data = self.model_dump(mode="json")
        lines: list[str] = []
        for key in ("name", "notes"):
            lines.append(f'{key} = "{data.pop(key)}"')
        lines.append("")
        _emit(data, [], lines)
        return "\n".join(lines) + "\n"


def _emit(node: dict[str, Any], path: list[str], out: list[str]) -> None:
    scalars = {k: v for k, v in node.items() if not isinstance(v, dict)}
    tables = {k: v for k, v in node.items() if isinstance(v, dict)}
    if scalars:
        out.append(f"[{'.'.join(path)}]")
        for k, v in scalars.items():
            out.append(f"{k} = {_fmt(v)}")
        out.append("")
    for k, v in tables.items():
        _emit(v, [*path, k], out)


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return f'"{value}"'
    return str(value)


#: Used when no profile is supplied.
DEFAULT = MistaraConfig()
