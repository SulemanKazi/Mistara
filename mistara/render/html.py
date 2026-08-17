"""Spatial HTML output.

This is deliberately not a debug throwaway. Requirement #4 says the extracted
text must live where it lived on the page, and the eventual deliverable is a
viewer or a PDF. That is the same artifact as the thing we need to eyeball
results — so it is built once, properly, and used for both.

The output is a single self-contained file: page images are embedded as data
URIs, so it can be opened anywhere or sent to someone with no server.
"""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path

from mistara.core.model import (
    Document,
    IssueKind,
    Language,
    ProvenanceKind,
    RegionType,
    ViewName,
)
from mistara.core.store import Store

_PROVENANCE_COLOURS = {
    ProvenanceKind.OCR: "#3b82f6",
    ProvenanceKind.QURAN: "#16a34a",
    ProvenanceKind.HADITH: "#0d9488",
    ProvenanceKind.LLM_CORRECTED: "#d97706",
    ProvenanceKind.UNRESOLVED: "#dc2626",
}

_ISSUE_STYLE = {
    IssueKind.MERGED_LINES: ("#dc2626", "merged"),
    IssueKind.MISSED_LINE: ("#ea580c", "missed"),
    IssueKind.COLUMN_GUTTER: ("#7c3aed", "gutter"),
    IssueKind.LOW_CONFIDENCE: ("#ca8a04", "low conf"),
}

#: Regions are coloured by the *role the grouping assigned them*, not by type —
#: S3 decides order, S2 will decide type. The badge carries the reading
#: sequence, so a wrong order is visible at a glance rather than only in a dump.
_REGION_STYLE = {
    "prose": ("#64748b", ""),
    "column": ("#0f766e", "col"),
    "marginalia": ("#b45309", "note"),
}

_CSS = """
:root {
  --bg: #f6f5f2; --panel: #fff; --ink: #1c1917; --muted: #78716c;
  --line: #e7e5e4; --accent: #0f766e;
}
@media (prefers-color-scheme: dark) {
  :root { --bg:#1c1917; --panel:#292524; --ink:#f5f5f4; --muted:#a8a29e;
          --line:#44403c; --accent:#5eead4; }
}
* { box-sizing: border-box; }
body { margin:0; font:14px/1.5 ui-sans-serif,system-ui,sans-serif;
       background:var(--bg); color:var(--ink); }
header { position:sticky; top:0; z-index:10; background:var(--panel);
         border-bottom:1px solid var(--line); padding:.75rem 1rem;
         display:flex; gap:1rem; align-items:center; flex-wrap:wrap; }
h1 { font-size:1rem; margin:0; font-weight:600; }
.meta { color:var(--muted); font-size:.8rem; }
.controls { margin-inline-start:auto; display:flex; gap:.5rem; flex-wrap:wrap; }
button { font:inherit; padding:.35rem .7rem; border:1px solid var(--line);
         background:var(--panel); color:var(--ink); border-radius:.4rem;
         cursor:pointer; }
button[aria-pressed="true"] { background:var(--accent); color:var(--panel);
                              border-color:var(--accent); }
main { padding:1.5rem; display:flex; flex-direction:column; gap:2rem;
       align-items:center; }
.page { position:relative; background:var(--panel); box-shadow:0 1px 3px #0002;
        max-width:100%; }
.page img { display:block; width:100%; height:auto; }
.layer { position:absolute; inset:0; }
/* Clip transcription to the page. `fit()` scales a line to its box but stops at
   0.4, so a box holding far more text than its width can carry still overflows —
   and without this it spills off the page onto the backdrop, next to a
   neighbouring page, which reads as if it belonged there. */
.layer.text { overflow:hidden; }
/* justify-content:flex-start, NOT flex-end. The container is direction:rtl, so
   flex-start IS the right edge — where RTL text begins. flex-end packs the span
   to the left, and since fit() scales it about `right center`, the anchor then
   sits at the span's *overflowed* right edge rather than the box's: the line
   gets scaled to exactly the right width in exactly the wrong place, landing
   hundreds of pixels off the page. */
.line { position:absolute; display:flex; align-items:center;
        justify-content:flex-start; direction:rtl; unicode-bidi:isolate;
        font-family:"Noto Nastaliq Urdu","Jameel Noori Nastaleeq","Nafees Nastaleeq",
                    "Noto Naskh Arabic","Amiri",serif;
        line-height:1.9; white-space:nowrap; overflow:visible; }
.line > span { transform-origin:right center; }
/* Text whose source is verified is coloured; text merely *identified* keeps the
   ordinary ink and is underlined instead. Colouring the latter would claim a
   verification that never happened. */
.pv { color:var(--c); }
.pv.cited { color:inherit; text-decoration:underline dotted var(--c);
            text-decoration-thickness:2px; text-underline-offset:4px; }
.box { position:absolute; border:1px solid currentColor; opacity:.55;
       pointer-events:none; }
/* Issues sit above regions — they are the thing you are hunting for.
   Colour arrives as --c for the same reason it does on .region: a label that
   sets its own `color` cannot also use currentColor for its background. */
.layer.issues { z-index:5; }
.issue { position:absolute; border:2px solid var(--c); border-radius:3px;
         cursor:help; box-shadow:0 0 0 1px #ffffffcc; }
/* Fill as a pseudo-element so only the fill is translucent. Putting opacity on
   .issue itself fades its border and its label with it, which is what made
   these invisible. */
.issue::before { content:""; position:absolute; inset:0; background:var(--c);
                 opacity:.18; }
.issue:hover::before { opacity:.38; }
.issue::after { content:attr(data-label); position:absolute; top:-1.15rem;
                inset-inline-start:0; font:700 10px/1.3 ui-sans-serif,sans-serif;
                color:#fff; background:var(--c); padding:2px 6px;
                border-radius:3px; white-space:nowrap;
                box-shadow:0 1px 3px #00000066; }
.issue.gutter::after { top:auto; bottom:-1.25rem; }
/* Regions sit above the text layer: the sequence badge is the whole point, and
   it is useless if a line of transcription lands on top of it.
   The region colour arrives as --c, NOT as `color`. Using currentColor here is
   a trap: the badge sets its own `color` for the text, which would redefine
   currentColor for its background in the same rule — white on white. */
.layer.regions { z-index:4; }
.region { position:absolute; border:2px solid var(--c); border-radius:4px;
          pointer-events:none;
          /* A light halo inside and out, so the outline stays legible over both
             dark ink and bare paper. */
          box-shadow:0 0 0 1px #ffffffcc, inset 0 0 0 1px #ffffffcc; }
/* Tint as its own layer: opacity on .region would fade the border with it. */
.region::before { content:""; position:absolute; inset:0; border-radius:2px;
                  background:var(--c); opacity:.10; }
/* ::after so the badge paints over the tint. Inset, not hanging off the corner,
   so clipping the text layer never eats it. */
.region::after { content:attr(data-seq); position:absolute; top:0; right:0;
                 font:700 12px/1.2 ui-sans-serif,sans-serif; color:#fff;
                 background:var(--c); padding:3px 8px;
                 border-radius:0 2px 0 7px; white-space:nowrap;
                 box-shadow:0 1px 4px #00000066; }
.badge { position:absolute; top:-1.4rem; inset-inline-end:0; font-size:.72rem;
         font-weight:600; padding:1px 6px; border-radius:.6rem;
         background:#dc2626; color:#fff; }
.pagelabel { position:absolute; top:-1.4rem; inset-inline-start:0;
             color:var(--muted); font-size:.75rem; }
.warn { color:#b45309; font-size:.8rem; }

body[data-mode="image"] .layer.text { display:none; }
body[data-mode="text"] .page img { visibility:hidden; }
body[data-mode="text"] .page { background:var(--panel); }
body:not([data-boxes="on"]) .box { display:none; }
body:not([data-regions="on"]) .layer.regions { display:none; }
body:not([data-issues="on"]) .layer.issues, 
body:not([data-issues="on"]) .badge { display:none; }
"""

_JS = """
const body = document.body;
function setMode(m){ body.dataset.mode = m;
  document.querySelectorAll('[data-set-mode]').forEach(b =>
    b.setAttribute('aria-pressed', b.dataset.setMode === m)); }
function toggleBoxes(){
  const on = body.dataset.boxes === 'on';
  body.dataset.boxes = on ? 'off' : 'on';
  document.querySelector('[data-toggle-boxes]').setAttribute('aria-pressed', !on); }
document.querySelectorAll('[data-set-mode]').forEach(b =>
  b.addEventListener('click', () => setMode(b.dataset.setMode)));
document.querySelector('[data-toggle-boxes]').addEventListener('click', toggleBoxes);
function toggleIssues(){
  const on = body.dataset.issues === 'on';
  body.dataset.issues = on ? 'off' : 'on';
  document.querySelector('[data-toggle-issues]').setAttribute('aria-pressed', !on); }
document.querySelector('[data-toggle-issues]').addEventListener('click', toggleIssues);
function toggleRegions(){
  const on = body.dataset.regions === 'on';
  body.dataset.regions = on ? 'off' : 'on';
  document.querySelector('[data-toggle-regions]').setAttribute('aria-pressed', !on); }
document.querySelector('[data-toggle-regions]').addEventListener('click', toggleRegions);
setMode('overlay');

// Shrink each line to fit the width of the box it was detected in, so text
// sits where the ink sat rather than overflowing into its neighbours.
function fit(){
  document.querySelectorAll('.line').forEach(el => {
    const span = el.firstElementChild;
    if(!span) return;
    span.style.transform = 'scale(1)';
    const avail = el.clientWidth, actual = span.scrollWidth;
    if(actual > avail && actual > 0){
      span.style.transform = `scale(${Math.max(0.4, avail/actual)})`;
    }
  });
}
window.addEventListener('load', fit);
window.addEventListener('resize', fit);
"""


def _data_uri(store: Store, sha: str, media_type: str) -> str:
    b64 = base64.b64encode(store.get_blob(sha)).decode("ascii")
    return f"data:{media_type};base64,{b64}"


def _leading_span(line):
    """The span that should speak for the line.

    Prefers a corpus-verified or corrected span over plain OCR, then a span that
    at least carries a citation. Otherwise the first span, or none.
    """
    if not line.spans:
        return None
    for span in line.spans:
        if span.provenance.kind is not ProvenanceKind.OCR:
            return span
    for span in line.spans:
        if span.provenance.source_ref:
            return span
    return line.spans[0]


def _span_html(span, text: str) -> str:
    """One span of a line, coloured by where its text came from.

    Two visually distinct states, because they are different claims:

    * **replaced** — the text on screen is corpus text, so it is *coloured*;
    * **cited** — the text is still the OCR reading and only its source is
      known, so it keeps the ordinary ink and gets a dotted underline.

    Colouring a cited span would assert a verification that did not happen.
    """
    escaped = html.escape(text)
    kind = span.provenance.kind
    ref = span.provenance.source_ref

    if kind is not ProvenanceKind.OCR:
        colour = _PROVENANCE_COLOURS.get(kind, "#3b82f6")
        return f'<span class="pv" style="--c:{colour}">{escaped}</span>'
    if ref:
        colour = _PROVENANCE_COLOURS.get(_corpus_kind(ref), "#16a34a")
        return f'<span class="pv cited" style="--c:{colour}">{escaped}</span>'
    return escaped


def _corpus_kind(ref: str) -> ProvenanceKind:
    return ProvenanceKind.QURAN if ref.startswith("quran") else ProvenanceKind.HADITH


def _line_html(line) -> str:
    """A line's text, split into its provenance spans.

    Everything sits inside one wrapper element because `fit()` scales
    `firstElementChild` to the box width — several siblings would leave all but
    the first unscaled.

    Ranges no span claims are emitted as plain text rather than skipped, so the
    line always renders in full even when span coverage is partial.
    """
    text = line.text
    if not line.spans:
        return f"<span>{html.escape(text)}</span>"

    parts: list[str] = []
    cursor = 0
    for span in sorted(line.spans, key=lambda s: s.char_start):
        start = max(cursor, span.char_start)
        end = min(len(text), span.char_end)
        if start > cursor:
            parts.append(html.escape(text[cursor:start]))
        if end > start:
            parts.append(_span_html(span, text[start:end]))
            cursor = end
    if cursor < len(text):
        parts.append(html.escape(text[cursor:]))
    return f"<span>{''.join(parts)}</span>"


def _region_role(region) -> str:
    if region.region_type is RegionType.MARGINALIA:
        return "marginalia"
    return "column" if region.signals.get("columns", 1) >= 2 else "prose"


def render_document(
    doc: Document,
    store: Store,
    *,
    view: ViewName = ViewName.RAW,
    show_boxes: bool = True,
    show_issues: bool = True,
    show_regions: bool = True,
) -> str:
    pages_html: list[str] = []

    for page in doc.pages:
        ref = page.views.get(view) or page.views.get(ViewName.RAW)
        img = _data_uri(store, ref.sha256, ref.media_type) if ref else ""
        w, h = page.width_px or 1, page.height_px or 1

        lines_html: list[str] = []
        boxes_html: list[str] = []
        regions_html: list[str] = []
        for seq, region in enumerate(page.ordered_regions(), start=1):
            role = _region_role(region)
            colour, tag = _REGION_STYLE[role]
            label = f"{seq}"
            if role == "column":
                label += f" {tag}{region.signals.get('column_index', 0):+.0f}"
            elif tag:
                label += f" {tag}"
            regions_html.append(
                f'<div class="region" style="left:{100 * region.bbox.x / w:.3f}%;'
                f"top:{100 * region.bbox.y / h:.3f}%;"
                f"width:{100 * region.bbox.w / w:.3f}%;"
                f'height:{100 * region.bbox.h / h:.3f}%;--c:{colour}" '
                f'data-seq="{html.escape(label)}"></div>'
            )
            for line in region.lines:
                left = 100 * line.bbox.x / w
                top = 100 * line.bbox.y / h
                width = 100 * line.bbox.w / w
                height = 100 * line.bbox.h / h
                # A matched line carries several spans — OCR prefix, corpus text,
                # OCR suffix — so the *first* span is not the interesting one.
                # Colour by the strongest provenance present.
                span = _leading_span(line)
                kind = span.provenance.kind if span else ProvenanceKind.OCR
                colour = _PROVENANCE_COLOURS.get(kind, "#3b82f6")
                lang = span.language if span else Language.UNKNOWN
                ref = span.provenance.source_ref if span else None
                title = html.escape(
                    f"{line.line_id} · {lang} · {kind}"
                    + (f" · {ref}" if ref else "")
                    + (
                        f" · identity {span.provenance.score:.2f}"
                        if span and span.provenance.score is not None
                        else ""
                    )
                    + (
                        f" · purity {line.signals.get('script_purity', 0):.2f}"
                        if line.signals
                        else ""
                    )
                )
                style = (
                    f"left:{left:.3f}%;top:{top:.3f}%;"
                    f"width:{width:.3f}%;height:{height:.3f}%;"
                )
                if line.text:
                    px = height * h / 100 * 0.72
                    lines_html.append(
                        f'<div class="line" style="{style}font-size:{px:.1f}px" '
                        f'title="{title}">{_line_html(line)}</div>'
                    )
                boxes_html.append(
                    f'<div class="box" style="{style}color:{colour}"></div>'
                )

        issues_html: list[str] = []
        for issue in page.issues:
            colour, label = _ISSUE_STYLE.get(issue.kind, ("#dc2626", issue.kind))
            cls = "issue gutter" if issue.kind is IssueKind.COLUMN_GUTTER else "issue"
            issues_html.append(
                f'<div class="{cls}" style="left:{100 * issue.bbox.x / w:.3f}%;'
                f"top:{100 * issue.bbox.y / h:.3f}%;"
                f"width:{max(100 * issue.bbox.w / w, 0.35):.3f}%;"
                f'height:{100 * issue.bbox.h / h:.3f}%;--c:{colour}" '
                f'data-label="{label}" title="{html.escape(issue.note)}"></div>'
            )

        badge = (
            f'<div class="badge">{len(page.issues)} issue'
            f'{"s" if len(page.issues) != 1 else ""}</div>'
            if page.issues
            else ""
        )

        pages_html.append(
            f"""
      <div class="page" style="aspect-ratio:{w}/{h};width:{min(w, 900)}px">
        <div class="pagelabel">page {page.page_no + 1} · {w}×{h}px"""
            + (f" · ~{page.source_dpi:.0f} DPI" if page.source_dpi else "")
            + f"""</div>
        {f'<img src="{img}" alt="page {page.page_no + 1}">' if img else ""}
        <div class="layer regions">{"".join(regions_html)}</div>
        <div class="layer boxes">{"".join(boxes_html)}</div>
        <div class="layer text">{"".join(lines_html)}</div>
        <div class="layer issues">{"".join(issues_html)}</div>
        {badge}
      </div>"""
        )

    n_lines = sum(len(r.lines) for p in doc.pages for r in p.regions)
    n_issues = sum(len(p.issues) for p in doc.pages)
    legend = " ".join(
        f'<span style="color:{c}">■</span> {k}' for k, c in _PROVENANCE_COLOURS.items()
    ) + (
        ' · <span class="pv cited" style="--c:#16a34a">underlined</span>'
        " = source identified, text still OCR"
    )
    issue_legend = " ".join(
        f'<span style="color:{c}">▮</span> {lbl}'
        for c, lbl in _ISSUE_STYLE.values()
    )
    region_legend = " ".join(
        f'<span style="color:{c}">▭</span> {role}'
        for role, (c, _) in _REGION_STYLE.items()
    )
    n_regions = sum(len(p.regions) for p in doc.pages)

    return f"""<!doctype html>
<html lang="ur" dir="rtl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mistara — {html.escape(Path(doc.source_path).name)}</title>
<style>{_CSS}</style>
</head>
<body data-mode="overlay" data-boxes="{"on" if show_boxes else "off"}"
      data-issues="{"on" if show_issues else "off"}"
      data-regions="{"on" if show_regions else "off"}" dir="ltr">
<header>
  <h1>{html.escape(Path(doc.source_path).name)}</h1>
  <span class="meta">{len(doc.pages)} pages · {n_regions} regions · {n_lines} lines
    · {n_issues} issues · doc {doc.doc_id}</span>
  <span class="meta">{legend}</span>
  <span class="meta">{issue_legend}</span>
  <span class="meta">{region_legend}</span>
  <div class="controls">
    <button data-set-mode="image">Image</button>
    <button data-set-mode="overlay">Overlay</button>
    <button data-set-mode="text">Text only</button>
    <button data-toggle-boxes aria-pressed="{"true" if show_boxes else "false"}">Boxes</button>
    <button data-toggle-issues aria-pressed="{"true" if show_issues else "false"}">Issues</button>
    <button data-toggle-regions
            aria-pressed="{"true" if show_regions else "false"}">Regions</button>
  </div>
</header>
<main>{"".join(pages_html)}</main>
<script>{_JS}</script>
</body>
</html>
"""


def write_document(
    doc: Document, store: Store, path: Path | str, **kwargs
) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_document(doc, store, **kwargs), encoding="utf-8")
    return out


def write_text_dump(doc: Document, path: Path | str) -> Path:
    """Plain reading-order text, for diffing and quick inspection.

    Page and region boundaries are marked. Without them a wrong reading order is
    hard to see: the text still looks like text, it just says the wrong thing in
    the wrong sequence, which is precisely the failure this dump exists to catch.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    chunks: list[str] = []
    for page in doc.pages:
        chunks.append(f"{'=' * 60}\n== page {page.page_no + 1}\n{'=' * 60}")
        for seq, region in enumerate(page.ordered_regions(), start=1):
            if not region.text.strip():
                continue
            role = _region_role(region)
            detail = f" {role}"
            if role == "column":
                detail += (
                    f" {region.signals.get('column_index', 0):+.0f}"
                    f"/{region.signals.get('columns', 1):.0f}"
                )
            chunks.append(f"-- [{seq}]{detail} --\n{region.text}")
    out.write_text("\n\n".join(chunks) + "\n", encoding="utf-8")
    return out


def document_summary(doc: Document) -> dict:
    return {
        "doc_id": doc.doc_id,
        "source": doc.source_path,
        "pages": len(doc.pages),
        "lines": sum(len(r.lines) for p in doc.pages for r in p.regions),
        "chars": len(doc.text),
        "history": [json.loads(h.model_dump_json()) for h in doc.history],
    }
