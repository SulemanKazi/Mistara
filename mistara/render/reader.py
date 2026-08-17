"""The reading view — the same document, typeset rather than debugged.

`render/html.py` answers "did the pipeline get this right?": it paints each line
at the exact pixel box the ink occupied, scaled to fit, over the scan. That is
the wrong artifact to hand someone who wants to *read* the book, because the
thing it faithfully reproduces — every line at its own arbitrary size — is
exactly what makes a page look unfinished.

This module answers "what does the book say?" It keeps the structure that came
out of the pipeline and throws away only the accidents of the scan:

* **kept** — reading order, the band structure (a margin note stays a note in
  the margin, a two-column band stays two columns in RTL order), which margin
  the hashiya sits in, and the separation of corpus-verified sacred text
  from OCR;
* **regularised** — the body measure. Every page reserves a hashiya column
  whether or not it carries a note, so the text column is the same width
  throughout; a page's own note width no longer decides its measure;
* **discarded** — per-line pixel geometry, per-line scale, the scan itself.

Urdu is set in nastaliq and ordinary ink, Arabic in naskh and blue, with the
saturated blue reserved for text that actually came from the corpus.

Everything here is *presentational*. Nothing in this file decides what a region
is; it renders what S2/S3 already decided. The two places where it does make a
typographic inference — merging consecutive full-measure lines into a paragraph,
and setting a region in a smaller size when its lines are visibly smaller — are
reversible from the UI and are never written back into the document.
"""

from __future__ import annotations

import base64
import html
from dataclasses import dataclass
from pathlib import Path
from statistics import median

from mistara.core.model import (
    Document,
    Language,
    Line,
    ProvenanceKind,
    Region,
    RegionType,
)

#: A line whose replaced-corpus characters reach this share of it is set as a
#: display verse rather than run into the surrounding prose.
_AYAH_LINE_FRAC = 0.6

#: A line reaching this share of its region's width was a *full measure* line in
#: the book, so the paragraph continues through it. A shorter line ended a
#: paragraph. This is typography, not structure — the Lines toggle undoes it.
_FULL_MEASURE = 0.8

#: …and so did a line followed by a vertical gap this much larger than the
#: *page's* line pitch. The width test alone misses a running head, whose box
#: spans the full measure because its two halves sit at opposite margins.
#:
#: The yardstick is the page's pitch rather than the region's on purpose: a
#: region of two or three lines has too few gaps for its own median to mean
#: anything — with two lines the single gap *is* the median, so no gap can ever
#: exceed it and a running head silently runs into the text below.
#:
#: Both tests are presentational: they decide where a paragraph breaks in this
#: view, and assert nothing about the document.
_PARA_GAP = 1.25

#: A region needs this many gaps before it may contribute to the page's pitch.
_MIN_LINES_FOR_PITCH = 4

#: A region needs this many lines before its median line height is allowed to
#: demote it a size step — the same reasoning as `order.min_lines_for_purity`.
#: Over one or two lines the median is noise, and a two-line region set in the
#: smaller size reads as a caption when it is body text.
_MIN_LINES_FOR_TIER = 3

#: Region type scale. A region whose median line height falls below this share
#: of the document's median is set one step down. Two tiers, not a continuum:
#: the point of this view is that sizes are consistent.
#:
#: Deliberately conservative, and provisional: marginalia are already demoted by
#: their region *type*, so this only catches type S2 has not classified yet
#: (footnote blocks). At 0.85 it fired on a four-line band whose median was
#: pulled down by a running head — body text set as a footnote, which reads far
#: worse than a footnote set as body text.
_SMALL_TIER = 0.7

_FONT_FILES = {
    "Mistara Nastaliq": ("NotoNastaliqUrdu-Regular.ttf", "NotoNastaliqUrdu[wght].ttf"),
    "Mistara Naskh": ("NotoNaskhArabic-Regular.ttf", "NotoNaskhArabic[wght].ttf"),
}

_FONT_DIRS = (
    "/usr/share/fonts",
    "/usr/local/share/fonts",
    "~/.local/share/fonts",
    "~/.fonts",
    "/Library/Fonts",
    "~/Library/Fonts",
    "C:/Windows/Fonts",
)


def _pretty_ref(ref: str) -> str:
    """`quran:2:255-256` → `2:255-256`, `bukhari:1:1` → `bukhari 1:1`."""
    book, _, rest = ref.partition(":")
    return rest if book == "quran" else f"{book} {rest}"


# ---------------------------------------------------------------------------
# fonts


def _find_font(names: tuple[str, ...]) -> Path | None:
    for root in _FONT_DIRS:
        base = Path(root).expanduser()
        if not base.is_dir():
            continue
        for name in names:
            hit = next(base.rglob(name), None)
            if hit is not None:
                return hit
    return None


def _font_faces() -> tuple[str, list[str]]:
    """Embedded `@font-face` rules, plus the families actually found.

    Embedding is what makes the file a deliverable: a single self-contained HTML
    that renders identically on a machine with no Urdu fonts installed, which is
    the common case outside this repo. When a face is missing locally we simply
    fall back to the CSS stack rather than failing.
    """
    faces: list[str] = []
    found: list[str] = []
    for family, names in _FONT_FILES.items():
        path = _find_font(names)
        if path is None:
            continue
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        fmt = "opentype" if path.suffix.lower() == ".otf" else "truetype"
        faces.append(
            f"@font-face{{font-family:'{family}';font-display:swap;"
            f"src:url(data:font/ttf;base64,{b64}) format('{fmt}')}}"
        )
        found.append(family)
    return "".join(faces), found


# ---------------------------------------------------------------------------
# layout


@dataclass(frozen=True)
class _Row:
    """One horizontal band of the page: a block of columns, or a lone region."""

    regions: list[Region]
    top: float  # page-fraction of the band's top edge
    bottom: float


def _rows(page) -> list[_Row]:
    """Regions in reading order, grouped into bands by their S3 block id.

    A block is precisely the thing that has to stay side by side — a parallel
    column pair, or body text beside a margin note. Everything else is its own
    full-width band.
    """
    h = page.height_px or 1
    bands: list[list[Region]] = []
    block: float | None = None

    for region in page.ordered_regions():
        b = region.signals.get("block", -1.0)
        if b >= 0 and b == block and bands:
            bands[-1].append(region)
            continue
        bands.append([region])
        block = b if b >= 0 else None

    rows: list[_Row] = []
    for band in bands:
        # RTL: column_index 0 is the rightmost column, which is also the order
        # a `direction:rtl` flex row lays its children out in.
        band.sort(key=lambda r: r.signals.get("column_index", 0))
        rows.append(
            _Row(
                band,
                min(r.bbox.y for r in band) / h,
                max(r.bbox.y + r.bbox.h for r in band) / h,
            )
        )
    return rows


def _is_note(region: Region) -> bool:
    return region.region_type is RegionType.MARGINALIA


def _split_band(row: _Row) -> tuple[list[Region], list[Region]]:
    """A band's regions as `(body, hashiya)`."""
    return (
        [r for r in row.regions if not _is_note(r)],
        [r for r in row.regions if _is_note(r)],
    )


def _body_cells(body: list[Region]) -> list[tuple[Region, float, float]]:
    """`(region, width%, leading-gap%)` *within the body measure*.

    Only meaningful when a band holds more than one body region — a genuine
    parallel-column pair. A lone region always fills the measure, which is the
    whole point of reserving the hashiya: the body column is the same width on
    every page whether or not that page carries a note.
    """
    if len(body) == 1:
        return [(body[0], 100.0, 0.0)]
    left = min(r.bbox.x for r in body)
    span = max(r.bbox.x + r.bbox.w for r in body) - left or 1.0
    out: list[tuple[Region, float, float]] = []
    prev_left_edge = left + span  # RTL: walk right → left
    for region in body:
        right = region.bbox.x + region.bbox.w
        gap = max(0.0, prev_left_edge - right) / span * 100
        out.append((region, region.bbox.w / span * 100, gap))
        prev_left_edge = region.bbox.x
    return out


@dataclass(frozen=True)
class _Hashiya:
    """The reserved margin column: how wide, and which side of which page.

    Reserving it on *every* page is the point — without it the body measure is
    whatever the widest region on that page happened to be, so the text column
    changes width from page to page and the book stops looking like a book.

    The side is not fixed, because in these books the hashiya sits on the
    **outer** margin and therefore alternates: in `sample_pdf_1` the notes are
    on the left on pages 1, 3, 5 and on the right on pages 2, 6. So the side is
    taken from the page's own notes where it has any, and elsewhere from the
    parity the rest of the document votes for.
    """

    width: float  # % of the sheet measure
    gutter: float  # % of the sheet measure
    right_on_even: bool  # parity fitted from the pages that carry notes
    sides: dict[int, str]  # page_no → "right" | "left", where observed

    def side(self, page_no: int) -> str:
        if page_no in self.sides:
            return self.sides[page_no]
        even = page_no % 2 == 0
        return "right" if even == self.right_on_even else "left"


#: Reserved-column geometry, as a share of the sheet measure. The width is the
#: median observed note width, floored so a two-word note is not set one letter
#: per line, and capped so it never eats the body measure.
_HASHIYA_MIN, _HASHIYA_MAX = 12.0, 22.0
_GUTTER_MIN, _GUTTER_MAX = 2.5, 6.0
_HASHIYA_DEFAULT, _GUTTER_DEFAULT = 14.0, 3.5


def _hashiya(doc: Document) -> _Hashiya:
    widths: list[float] = []
    gutters: list[float] = []
    sides: dict[int, str] = {}
    votes = 0

    for page in doc.pages:
        w = page.width_px or 1
        notes = [r for r in page.regions if _is_note(r)]
        if not notes:
            continue
        widths += [100 * r.bbox.w / w for r in notes]
        centre = median([r.bbox.x + r.bbox.w / 2 for r in notes])
        right = centre > w / 2
        sides[page.page_no] = "right" if right else "left"
        votes += 1 if right == (page.page_no % 2 == 0) else -1

        # The gutter is the gap between the notes and the body text beside them.
        body = [r for r in page.regions if not _is_note(r)]
        for note in notes:
            near = [
                (note.bbox.x - (r.bbox.x + r.bbox.w))
                if right
                else (r.bbox.x - (note.bbox.x + note.bbox.w))
                for r in body
                if r.bbox.y < note.bbox.y + note.bbox.h
                and note.bbox.y < r.bbox.y + r.bbox.h
            ]
            gutters += [100 * g / w for g in near if g > 0]

    return _Hashiya(
        width=min(_HASHIYA_MAX, max(_HASHIYA_MIN, median(widths)))
        if widths
        else _HASHIYA_DEFAULT,
        gutter=min(_GUTTER_MAX, max(_GUTTER_MIN, median(gutters)))
        if gutters
        else _GUTTER_DEFAULT,
        right_on_even=votes >= 0,
        sides=sides,
    )


def _median_line_height(lines: list[Line]) -> float:
    return median([ln.bbox.h for ln in lines]) if lines else 0.0


def _size_tier(region: Region, body_h: float) -> str:
    """Presentational size class. Two tiers only — see `_SMALL_TIER`."""
    if region.region_type is RegionType.MARGINALIA:
        return "small"
    if len(region.lines) < _MIN_LINES_FOR_TIER:
        return "body"
    h = _median_line_height(region.lines)
    if body_h and h and h < _SMALL_TIER * body_h:
        return "small"
    return "body"


def _line_pitch(lines: list[Line]) -> float:
    """Median baseline-to-baseline distance, used to size the paragraph gap."""
    tops = [ln.bbox.y for ln in lines]
    gaps = [b - a for a, b in zip(tops, tops[1:], strict=False) if b > a]
    return median(gaps) if gaps else 0.0


# ---------------------------------------------------------------------------
# text


def _replaced_frac(line: Line) -> float:
    if not line.text:
        return 0.0
    covered = sum(
        span.char_end - span.char_start
        for span in line.spans
        if span.provenance.kind in (ProvenanceKind.QURAN, ProvenanceKind.HADITH)
    )
    return covered / len(line.text)


def _line_ref(line: Line) -> str | None:
    for span in line.spans:
        if span.provenance.source_ref:
            return span.provenance.source_ref
    return None


def _chip(ref: str | None) -> str:
    if not ref:
        return ""
    return f'<span class="ref">{html.escape(_pretty_ref(ref))}</span>'


def _span_html(span, text: str, *, refs: bool) -> str:
    """One provenance run.

    Arabic is set in naskh and in blue; Urdu is the page's ordinary ink. Within
    the Arabic, three states stay distinguishable because they are three
    different claims:

    * **verified** — the characters on screen came from the corpus (full blue);
    * **identified** — the source is known but the text is still the OCR
      reading, so it keeps the plainer blue and takes a dotted underline;
    * **as read** — Arabic the corpus has not matched at all (plainer blue).

    Colouring an identified span like a verified one would assert a replacement
    that never happened, which is the whole distinction this view exists to keep.
    """
    escaped = html.escape(text)
    kind = span.provenance.kind
    ref = span.provenance.source_ref
    chip = _chip(ref) if refs else ""

    if kind in (ProvenanceKind.QURAN, ProvenanceKind.HADITH):
        return f'<span class="ar sacred">{escaped}</span>{chip}'
    if kind is ProvenanceKind.LLM_CORRECTED:
        return f'<span class="edited">{escaped}</span>'
    if ref:
        return f'<span class="ar cited">{escaped}</span>{chip}'
    if span.language in (Language.ARABIC, Language.MIXED):
        return f'<span class="ar">{escaped}</span>'
    return escaped


def _inline_html(line: Line, *, refs: bool = True) -> str:
    """A line's text with its provenance runs marked.

    Character ranges no span claims are emitted as plain text rather than
    dropped, so a line always renders in full even when span coverage is partial.
    """
    text = line.text
    if not line.spans:
        return html.escape(text)
    parts: list[str] = []
    cursor = 0
    for span in sorted(line.spans, key=lambda s: s.char_start):
        start = max(cursor, span.char_start)
        end = min(len(text), span.char_end)
        if start > cursor:
            parts.append(html.escape(text[cursor:start]))
        if end > start:
            parts.append(_span_html(span, text[start:end], refs=refs))
            cursor = end
    if cursor < len(text):
        parts.append(html.escape(text[cursor:]))
    return "".join(parts)


def _region_html(region: Region, *, pitch: float = 0.0) -> str:
    """A region's lines, run into paragraphs.

    Lines stay individually addressable (`<span class="ln">`) so the Lines
    toggle is a CSS change rather than a re-render: in paragraph mode they are
    inline and flow together, in line mode each becomes a block.
    """
    measure = region.bbox.w or 1
    lines = [ln for ln in region.lines if ln.text.strip()]
    out: list[str] = []
    para: list[str] = []

    def flush() -> None:
        if para:
            out.append(f'<p class="para">{"".join(para)}</p>')
            para.clear()

    for i, line in enumerate(lines):
        if _replaced_frac(line) >= _AYAH_LINE_FRAC:
            # A display verse carries its citation once, at the end of the
            # block — not inline after each run, which is where it belongs when
            # the quote is embedded in prose.
            flush()
            body = _inline_html(line, refs=False)
            out.append(
                f'<p class="ayah">{body}{_chip(_line_ref(line))}</p>'
            )
            continue

        para.append(f'<span class="ln">{_inline_html(line)}</span> ')
        nxt = lines[i + 1] if i + 1 < len(lines) else None
        gap = (nxt.bbox.y - line.bbox.y) if nxt else 0.0
        if line.bbox.w < _FULL_MEASURE * measure or (
            pitch and gap > _PARA_GAP * pitch
        ):
            flush()
    flush()
    return "".join(out)


# ---------------------------------------------------------------------------
# page


_CSS = """
/* White paper, ink-grey Urdu, blue Arabic. Light is the base palette; dark
   redefines only the tokens. The theme has three states — explicit light,
   explicit dark, and "follow the system", which sets no attribute at all — so
   the media query has to be guarded against an explicit light choice or the
   button cannot turn dark mode off. */
:root{
  --paper:#ffffff; --ink:#2c2c2c; --muted:#767472; --rule:#e4e2de;
  --edge:#e8e6e2; --bg:#f1efec; --chrome:#ffffff;
  --ar:#12508f;        /* verified Arabic — the characters came from the corpus */
  --ar-plain:#4a678a;  /* Arabic as read off the page */
  --accent:#12508f; --gold:#9a7b3f;
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
         --paper:#17181a; --ink:#e6e4e0; --muted:#8f8d8a; --rule:#2f3134;
         --edge:#26282b; --bg:#0d0e10; --chrome:#1b1d1f;
         --ar:#7fb2e8; --ar-plain:#93a8c2; --accent:#7fb2e8; --gold:#c3a468; }
}
:root[data-theme="dark"]{
  --paper:#17181a; --ink:#e6e4e0; --muted:#8f8d8a; --rule:#2f3134;
  --edge:#26282b; --bg:#0d0e10; --chrome:#1b1d1f;
  --ar:#7fb2e8; --ar-plain:#93a8c2; --accent:#7fb2e8; --gold:#c3a468;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
     font:14px/1.5 ui-sans-serif,system-ui,-apple-system,sans-serif}
header{position:sticky;top:0;z-index:10;background:var(--chrome);
       border-bottom:1px solid var(--rule);padding:.7rem 1.2rem;display:flex;
       gap:1rem;align-items:center;flex-wrap:wrap}
h1{font-size:.95rem;margin:0;font-weight:600;letter-spacing:.01em}
.meta{color:var(--muted);font-size:.78rem}
.legend{display:flex;gap:.9rem;align-items:center;font-size:.78rem;
        color:var(--muted)}
.legend b{font-weight:600}
.swatch{display:inline-block;width:.6rem;height:.6rem;border-radius:2px;
        vertical-align:-1px;margin-inline-end:.3rem}
.controls{margin-inline-start:auto;display:flex;gap:.4rem;flex-wrap:wrap}
button{font:inherit;font-size:.82rem;padding:.3rem .7rem;border:1px solid var(--rule);
       background:var(--chrome);color:var(--ink);border-radius:.45rem;cursor:pointer}
button[aria-pressed="true"]{background:var(--accent);border-color:var(--accent);
                            color:var(--paper)}
main{padding:2.5rem 1rem 4rem;display:flex;flex-direction:column;
     align-items:center;gap:2.5rem}

.sheet{position:relative;width:min(100%,940px);background:var(--paper);
       padding:1.6rem 2.6rem 1.4rem;border:1px solid var(--edge);
       border-radius:2px;box-shadow:0 1px 2px #0000000f,0 12px 30px -12px #00000024;
       direction:rtl;text-align:right;display:flex;flex-direction:column}
.leaf{flex:1;padding:1.9rem 0 2.1rem}

/* Header and footer ornament. Built from CSS rules and a rotated square rather
   than a glyph: the embedded faces carry no ornament characters, and a missing
   glyph would render as tofu on exactly the machines the embedding exists for. */
.orn{display:flex;align-items:center;gap:.7rem;color:var(--gold)}
.orn::before,.orn::after{content:"";flex:1;height:1px;background:currentColor;
                         opacity:.55}
.orn i{width:5px;height:5px;border:1px solid currentColor;transform:rotate(45deg);
       flex:none}
.orn.thin{gap:.5rem}
.orn.thin::before,.orn.thin::after{opacity:.3}
.head{display:flex;flex-direction:column;gap:3px}
.foot{display:flex;flex-direction:column;gap:5px;
      /* The folio sits in the *page* footer, centred, not in the text. */
      align-items:stretch}
.folio{display:flex;justify-content:center;align-items:center;gap:.6rem;
       font:600 11px/1 ui-sans-serif,system-ui,sans-serif;letter-spacing:.16em;
       color:var(--muted);direction:ltr}
.folio::before,.folio::after{content:"";width:14px;height:1px;
                             background:var(--gold);opacity:.55}

/* Every page reserves the hashiya column, whether or not it carries a note, so
   the body measure never changes width from page to page. --hw and --hg are the
   reserved width and its gutter, both measured off the document. */
.row{display:grid;column-gap:var(--hg);direction:rtl;align-items:flex-start}
/* Both cells are pinned to the first grid row. Without this, a page whose
   hashiya is the *leading* track gets the note auto-placed into a second row —
   grid auto-placement never moves backwards — and the note drops below the
   text it annotates instead of sitting beside it. */
.measure,.hashiya{grid-row:1}
.sheet[data-hashiya="right"] .row{grid-template-columns:var(--hw) 1fr}
.sheet[data-hashiya="right"] .hashiya{grid-column:1}
.sheet[data-hashiya="right"] .measure{grid-column:2}
.sheet[data-hashiya="left"] .row{grid-template-columns:1fr var(--hw)}
.sheet[data-hashiya="left"] .measure{grid-column:1}
.sheet[data-hashiya="left"] .hashiya{grid-column:2}
/* NB `.measure`, not `.body` — the size tier puts `body` on the section
   itself, and a `.body{display:flex}` rule would then turn every region into a
   flex row and lay its paragraphs out side by side. It did exactly that. */
.measure{display:flex;direction:rtl;align-items:flex-start;min-width:0}
.hashiya{min-width:0}
.rg{min-width:0}

/* One family for Urdu, one for Arabic, one size per tier — this is the whole
   point of the view. Nastaliq needs the generous leading; at 1.5 the descenders
   of one line collide with the next. */
.rg{font-family:'Mistara Nastaliq','Noto Nastaliq Urdu','Jameel Noori Nastaleeq',
    'Nafees Nastaleeq',serif;line-height:2.35;font-size:15px}
.rg.small{font-size:12.5px;line-height:2.15}
.rg.note{color:var(--muted);font-size:12.5px;line-height:2.15}
.sheet[data-hashiya="right"] .rg.note{border-inline-start:1px solid var(--rule);
                                      padding-inline-start:.8rem}
.sheet[data-hashiya="left"] .rg.note{border-inline-end:1px solid var(--rule);
                                     padding-inline-end:.8rem}
.para{margin:0 0 .85rem;text-align:right}
body[data-flow="para"] .para{text-align:justify;text-align-last:right}
/* Never justify a narrow measure: a three-word sidenote stretched to its
   column turns into three widely-spaced columns of one word each. */
body[data-flow="para"] .small .para,
body[data-flow="para"] .note .para{text-align:right}
body[data-flow="lines"] .ln{display:block}
body[data-flow="lines"] .para{margin-bottom:.5rem}

/* Arabic: naskh and blue. The saturated blue is reserved for text that came
   from the corpus; Arabic merely read off the page takes the plainer one. */
.ar,.ayah{font-family:'Mistara Naskh','Noto Naskh Arabic','Amiri',serif;
          color:var(--ar-plain);font-size:1.06em;line-height:1.9}
.ar.sacred,.ayah{color:var(--ar)}
.ayah{margin:.9rem 0;padding:.5rem 1.1rem;text-align:center;
      border-inline-start:2px solid color-mix(in srgb,var(--ar) 30%,transparent);
      border-inline-end:2px solid color-mix(in srgb,var(--ar) 30%,transparent);
      background:color-mix(in srgb,var(--ar) 4%,transparent);border-radius:3px}
.cited{text-decoration:underline dotted var(--ar);text-decoration-thickness:1.5px;
       text-underline-offset:5px}
.edited{border-bottom:1px dashed #b4700f}
.ref{font:600 10px/1 ui-sans-serif,system-ui,sans-serif;direction:ltr;
     display:inline-block;vertical-align:.35em;margin:0 .25em;padding:2px 5px;
     border-radius:3px;background:color-mix(in srgb,var(--accent) 12%,transparent);
     color:var(--accent);letter-spacing:.02em;white-space:nowrap}
.ayah .ref{vertical-align:.15em;margin-inline-start:.6em}
body:not([data-refs="on"]) .ref{display:none}
body:not([data-sacred="on"]) .ar.sacred{color:var(--ar-plain)}
body:not([data-sacred="on"]) .ayah{background:none;border-color:var(--rule)}
body:not([data-sacred="on"]) .cited{text-decoration:none}

@media print{
  header{display:none}
  body{background:#fff}
  main{padding:0;gap:0}
  .sheet{box-shadow:none;border:none;width:100%;break-after:page}
}
@media (max-width:640px){
  .sheet{padding:1rem 1.2rem}
  .row,.sheet[data-hashiya="right"] .row,.sheet[data-hashiya="left"] .row{
       grid-template-columns:1fr}
  .row .measure,.row .hashiya{grid-column:1}
  .measure{flex-wrap:wrap}
  .measure .rg{width:100%!important;margin-inline-start:0!important}
  .rg.note{border:none!important;border-top:1px solid var(--rule)!important;
           padding:.6rem 0 0!important;margin-top:.6rem}
}
"""

_JS = """
const body=document.body;
function toggle(attr,btn,on,off){
  const cur=body.dataset[attr];
  body.dataset[attr]=cur===on?off:on;
  btn.setAttribute('aria-pressed',body.dataset[attr]===on);
}
document.querySelectorAll('[data-flag]').forEach(btn=>{
  const [attr,on,off]=btn.dataset.flag.split(':');
  btn.setAttribute('aria-pressed',body.dataset[attr]===on);
  btn.addEventListener('click',()=>toggle(attr,btn,on,off));
});
// Theme starts unset — "follow the system". The first press resolves it to the
// opposite of whatever is currently showing, so the button always does the
// visible thing rather than the thing the attribute says.
const root=document.documentElement, themeBtn=document.querySelector('[data-theme-btn]');
function paintTheme(){
  const dark = root.dataset.theme
    ? root.dataset.theme==='dark'
    : matchMedia('(prefers-color-scheme:dark)').matches;
  themeBtn.textContent = dark ? 'Light' : 'Dark';
}
themeBtn.addEventListener('click',()=>{
  const dark = root.dataset.theme
    ? root.dataset.theme==='dark'
    : matchMedia('(prefers-color-scheme:dark)').matches;
  root.dataset.theme = dark ? 'light' : 'dark';
  paintTheme();
});
paintTheme();
"""


def render_reader(doc: Document, *, embed_fonts: bool = True) -> str:
    faces, families = _font_faces() if embed_fonts else ("", [])
    hashiya = _hashiya(doc)

    body_h = median(
        [
            _median_line_height(r.lines)
            for p in doc.pages
            for r in p.regions
            if r.lines and r.region_type is not RegionType.MARGINALIA
        ]
        or [0.0]
    )

    sheets: list[str] = []
    for page in doc.pages:
        # One pitch for the whole page, measured only on the regions long
        # enough to have a meaningful one. See `_PARA_GAP`.
        page_pitch = median(
            [
                _line_pitch(r.lines)
                for r in page.regions
                if len(r.lines) >= _MIN_LINES_FOR_PITCH
            ]
            or [0.0]
        )
        rows_html: list[str] = []
        prev_bottom = 0.0
        for row in _rows(page):
            # Vertical gaps are the page's, compressed: the typeset text no
            # longer has the scan's heights, so an absolute gap would be a lie.
            # The ordering and the *presence* of a break are what carry meaning.
            gap = max(0.0, row.top - prev_bottom)
            space = min(2.2, max(0.3, gap * 22))
            prev_bottom = row.bottom

            body, notes = _split_band(row)
            body_html: list[str] = []
            for region, width, lead in _body_cells(body):
                text = _region_html(region, pitch=page_pitch)
                if not text:
                    continue
                body_html.append(
                    f'<section class="rg {_size_tier(region, body_h)}" '
                    f'style="width:{width:.2f}%;margin-inline-start:{lead:.2f}%">'
                    f"{text}</section>"
                )
            notes_html = [
                f'<section class="rg note">{text}</section>'
                for text in (
                    _region_html(r, pitch=page_pitch) for r in notes
                )
                if text
            ]
            if body_html or notes_html:
                # The hashiya cell is emitted even when empty: it is what holds
                # the reserved column open on a page with no note.
                rows_html.append(
                    f'<div class="row" style="margin-top:{space:.2f}em">'
                    f'<div class="measure">{"".join(body_html)}</div>'
                    f'<aside class="hashiya">{"".join(notes_html)}</aside>'
                    f"</div>"
                )
        sheets.append(
            f'<article class="sheet" data-hashiya="{hashiya.side(page.page_no)}">'
            f'<div class="head"><div class="orn"><i></i></div>'
            f'<div class="orn thin"><i></i></div></div>'
            f'<div class="leaf">{"".join(rows_html)}</div>'
            f'<div class="foot"><div class="orn thin"><i></i></div>'
            f'<div class="folio">{page.page_no + 1}</div></div>'
            f"</article>"
        )

    title = html.escape(Path(doc.source_path).name)
    n_sacred = sum(
        1
        for p in doc.pages
        for r in p.regions
        for ln in r.lines
        for s in ln.spans
        if s.provenance.kind in (ProvenanceKind.QURAN, ProvenanceKind.HADITH)
    )
    font_note = (
        f" · fonts embedded ({len(families)})" if families else " · system fonts"
    )

    return f"""<!doctype html>
<html lang="ur" dir="rtl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>{faces}{_CSS}</style>
</head>
<body dir="ltr" data-flow="para" data-refs="on" data-sacred="on"
      style="--hw:{hashiya.width:.2f}%;--hg:{hashiya.gutter:.2f}%">
<header>
  <h1>{title}</h1>
  <span class="meta">{len(doc.pages)} pages · {n_sacred} verified spans
    · doc {doc.doc_id}{font_note}</span>
  <span class="legend">
    <span><span class="swatch" style="background:var(--ar)"></span><b>corpus</b>
      — verified text</span>
    <span><span class="swatch" style="background:var(--ar-plain)"></span>Arabic
      — as read</span>
    <span><span class="cited">identified</span> — source known, text still OCR</span>
  </span>
  <div class="controls">
    <button data-flag="flow:lines:para">Lines</button>
    <button data-flag="refs:on:off">References</button>
    <button data-flag="sacred:on:off">Highlight corpus</button>
    <button data-theme-btn>Dark</button>
  </div>
</header>
<main>{"".join(sheets)}</main>
<script>{_JS}</script>
</body>
</html>
"""


def write_reader(
    doc: Document, path: Path | str, *, embed_fonts: bool = True
) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_reader(doc, embed_fonts=embed_fonts), encoding="utf-8")
    return out
