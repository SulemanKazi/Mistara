# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

`uv` lives in `~/.local/bin`; export it if `uv` is not found:
`export PATH="$HOME/.local/bin:$PATH"`

```bash
uv sync                                    # Python 3.12 (3.13+ breaks OpenCV)
uv run pytest                              # full suite
uv run pytest tests/test_columns.py -v     # one file
uv run pytest -k "gutter" -v               # by name
uv run pytest tests/test_core.py::TestBBox::test_half_overlap
uv run ruff check --fix .

# End-to-end, offline — the `stub` provider needs no API key or network
uv run mistara ingest data/sample_pdf_1.pdf
uv run mistara extract <doc_id> --provider stub
uv run mistara order <doc_id>              # cells → regions in reading order
uv run mistara corpus fetch                # download the Quran (once; the only network call)
uv run mistara match <doc_id>              # verify Arabic against the corpus
uv run mistara render <doc_id> -o out/view.html --text out/view.txt
uv run mistara render <doc_id> --reader out/read.html   # typeset reading view
uv run mistara show <doc_id> --json        # state + run ledger
uv run mistara config                      # effective tuning profile
uv run mistara providers                   # which credentials are visible
uv run mistara <cmd> --config config/x.toml
```

Doc ids accept prefixes (`938f`). `data/store/` is a rebuildable cache — deleting it is safe.

Provider keys load from `.env` at the repo root (gitignored; real env vars win over it). `mistara providers` reports what is visible without printing values — use it rather than echoing the environment.

## Read first

- `docs/progress.md` — **what is actually built and what is thin.** Read before assuming a stage works.
- `docs/design.md` — architecture and the reasoning behind it.
- `docs/prd.md` — requirements.

Stage numbering (S0–S7) is used consistently across docs, module names, and CLI output. Several stages are unbuilt; the table at the top of `progress.md` is authoritative.

## Architecture

**Stage = processor + verifier.** Every stage does work *and* judges its own output. This is not decoration: PRD requirement #2 is post-step verification — after a step runs, can we tell whether it worked and whether to re-run? Verifiers return a `Verdict` (`passed`, optional `score`, `findings`, named `metrics`) and may be pure code, a cross-check against a second implementation, or model-backed. `score` is optional by design; some stages can only say pass/fail plus findings.

**Named metrics, not a scalar.** The right repair differs by which signal fired — low `coverage.line_count` means re-read at line level, low `layout.cv_agreement` means re-run layout. A single confidence number cannot express that.

**Findings carry geometry.** Verifiers emit `Issue(kind, bbox, note)` persisted on the `Page`. A number says *that* something is wrong; a box says *where*, which is what makes it actionable for a human eyeballing the render and for an agent choosing a region to re-run. The HTML viewer draws them.

**Typed per-stage I/O, one accumulating state.** Each stage declares its own input/output models so it is testable without constructing a whole `Document`; `Stage.run()` produces typed output and `Stage.apply()` merges it into the canonical `Document`. Keep those separate when adding stages.

**Content-addressed store + SQLite ledger** (`core/store.py`). Blobs keyed by sha256; every stage invocation recorded with params, signals, cost. This is what makes retry cheap and evaluation reproducible.

**The CLI is the tool surface.** There is deliberately no `agent/` package. The orchestrator is Claude Code or a human driving the CLI, so *every stage must stay exactly one command with `--json` output*. When a programmatic agent is eventually built, its tools wrap these commands. Do not add a stage that can only be invoked from Python.

**Providers are role-indirected** (`providers/registry.py`). Stages request `extract.primary`, never a vendor. Adding a provider means implementing `VLMClient` and registering a spec string — no stage changes.

**`text/canonical.py` is load-bearing.** Ground truth, pipeline output, and every metric route through it. If it drifts, every number the project reports drifts. It has two modes: `strict` keeps tashkeel (Arabic/Quranic comparison), `loose` strips it (Urdu CER, since Urdu prose is unvocalized). **A CER number without its mode is meaningless** — always state it.

## Domain traps

These cost real debugging time. They are not discoverable from reading one file.

**Geometry fails on two independent axes.** Vertical signals (pitch regularity, box height, gaps) catch merged and missed lines. Horizontal signals (shared ink-free gutters) catch side-by-side Arabic/Urdu pairs, tables, and margin notes. A side-by-side row has normal height and sits on the regular pitch, so it passes every vertical check while holding two texts in two languages — the vertical detector reported an entire two-column document as clean. Adding a third failure mode? Assume the existing signals will call it healthy.

**Ink coverage is a trap as an objective.** It is maximised by emitting one box for the whole page. Lowering the projection threshold to "capture more ink" *reduces* line count because valleys rise above the line and bands fuse (measured: 27 rows → 19 while coverage hit 100%). Optimise **pitch regularity** instead.

**A parallel-text block and a table are the same geometry.** Both are a run of consecutive rows sharing a gutter, and they read differently: parallel text is column-major (finish the right column, then the left), a table is row-major. Nothing in the geometry distinguishes them — that is region typing (S2, unbuilt). S3 therefore exposes `order.column_read_order` as a **per-book config choice** defaulted to the observed sample, not as a detection. Do not add a heuristic that guesses it and presents the guess as fact.

**Cells are grouped, and the grouping must be a permutation.** S3 regroups S4's flat cell list into regions; `order.line_conservation` must be exactly 1.0. A grouping bug drops or duplicates transcribed text, and no text metric would attribute that to the ordering stage. Also note S4's `apply()` overwrites `page.regions`, so re-running `extract` discards the ordering — run `order` again after.

**`identity` and `margin` answer different questions.** Identity says the *wording* is right; margin says the *citation* is. The Quran repeats phrasing verbatim — `وَمَنْ لَّمْ یَحْکُمْ بِمَآ اَنْزَلَ اللّٰہُ` is identical at 5:44, 5:45 and 5:47 — so identity 1.00 with margin 0.00 is the normal case, not a bug. A low margin must annotate the reference and never block the replacement. About a third of matches on the samples are ambiguous this way.

**`match.min_span_tokens` is a safety floor, not a tuning knob.** A single common word shared with the Quran matches Urdu prose at identity 1.00 — measured, not hypothetical. Coverage cannot catch it, because a genuine inline quote is also a low-coverage match; only span length separates them. Lowering it *globally* trades directly against the zero-false-replacement target. The **one** sanctioned exception is a contiguous continuation: a line that picks up where the previous *accepted* match left off (`0 ≤ corpus_start − prefer_after ≤ continuation_max_gap`) may be acted on down to `min_continuation_tokens` (1), because continuity is independent evidence the length floor cannot see — a random short line will not also land on the exact next corpus position. This is `decide(..., continues=True)`, and it also required lowering the matcher's query-length floor (`min_continuation_query_tokens`) and letting `CorpusIndex.candidates` seed sub-trigram queries from rare unigrams. A continuation is **replaced, not merely cited**, even at low coverage — unlike a generic inline quote (which `replace_inline` gates off by default because its boundary is inferred), a continuation's location is vouched for by continuity and its boundary is the aligner's exact token span, so a trailing tail like `مِّمَّا يَكۡسِبُونَ (79 بقرہ)` restores to corpus text (green) with its citation left intact. It recovered the ~6 short verse tails per 10 pages that the isolated-line floor left unresolved; verified on the samples that the same query still returns nothing when looked up cold, and that every flipped span overwrites verse text only. Do **not** relax the continuation guard to fire without a preceding accepted match — the backward contiguity is the whole safety argument.

**Normalization only has to be consistent, never faithful.** `text/arabic.py` strips the dagger alef, so `کٰفِرُوْنَ` becomes `کفرون` rather than `کافرون`. That is correct because the Uthmani corpus carries the same dagger alef and lands on the same skeleton. The invariant is that corpus and query go through *the identical function* — if they ever diverge, every match score silently becomes meaningless.

**S6a re-runs must not overwrite `original_text`.** On a second pass the line already holds corpus text, so the naive `line.text[start:end]` records the previous replacement as the "OCR reading" and destroys the audit trail. `_rewrite` carries the earliest original forward; there is a test for it.

**A verse that wraps a proclitic across a line break leaves an orphan.** Uthmani fuses proclitics (و ف ب ل ک …) into the following word — `وَٱلۡأَحۡبَارُ` is one corpus token. When the page splits that at the line break (`… وَٱلرَّبَّـٰنِيُّونَ وَ` on one line, `ٱلۡأَحۡبَارُ` on the next), the trailing lone `وَ` has **no standalone corpus token to match** — `match("وَ")` returns nothing even as a continuation — so the aligner correctly bounds the verified span before it and the waw stays un-verified OCR (ungreen). Meanwhile the next line is replaced with the full corpus word, which *reintroduces* the waw — so the final text carries a duplicated proclitic across the break. This is inherent to line-anchored, line-local, non-destructive replacement; it is not a matcher bug, and colouring the orphan green would assert a corpus match that does not exist for that isolated character. Left as-is by decision (2026-08-15); the only clean fix is render-only suppression of a proclitic already covered by an adjacent verified span, never a write-back.

**Never upsample scans.** These PDFs are wrappers: a Letter-sized page object containing one embedded ~700px bitmap. Rasterizing at a nominal 400 DPI invents pixels and burns vision tokens. S0 detects this and extracts natively, reporting the *true* source DPI measured against the image's placed width — not the page width.

**Source scans are ~102–115 DPI.** This caps achievable accuracy regardless of pipeline quality, because nastaliq is distinguished by *i'jam* (dot placement) and dots die first at low resolution. Do not attribute all error to the pipeline.

**`temperature` is rejected on Claude Opus 5 and Sonnet 5** (400 error; sampling params were removed on that family). Self-consistency via k-sampling is therefore not portable. Agreement signals use **input perturbation** (different view, crop padding, prompt phrasing) instead. `VLMClient.supports_temperature` records which providers offer the alternative.

**VLMs have memorized the Quran.** Asked to read a Quranic line from a degraded scan, a model emits the canonical verse regardless of what is printed. Harmless downstream (we replace with corpus text anyway) but it means **VLM Arabic output is not evidence of what the page says** — never use it to validate verse identity, and never let it leak into Urdu prose.

**Urdu and Arabic share the script.** Language routing is not script detection. Evidence combines Urdu-exclusive codepoints, tashkeel density, typeface (naskh vs nastaliq), Quranic ornaments, and corpus retrieval hits — the matcher is the most reliable detector.

**Text is stored in logical Unicode order, never visually reordered.** Bidi is a rendering concern.

## Calibration is thin — do not generalize

Every numeric constant is fitted to **one book, two documents, ten pages** — `stranded_ink_frac = 0.08` (calibrated on two data points), `min_gutter_ratio`, `tall_factor`, `gap_factor`, the `0.35` margin-note-vs-table ratio, and the binarization block/c. There is **no accuracy measurement yet**.

They all live in `core/config.py` with defaults, documented in `config/default.toml`, overridable per book via `--config`. **Add new thresholds there, not as literals in stage code.** Config is embedded in stage params, so it is part of the cache key and is recorded in the run ledger — retuning re-runs affected stages automatically, and any result can be traced back to the settings that produced it. Unknown keys are rejected rather than ignored.

A previous session inferred a rule about page layout from these ten pages and asserted it in a test; it was removed. Treat every threshold as provisional, keep them as named parameters rather than inline literals, and prefer widening the corpus over further tuning.

Reference data is **bootstrapped by correcting pipeline output**, not authored from scratch — the document JSON is already the reference format. Do not treat missing ground truth as a blocker or press for a labelling pass; reviewing stage output by eye is the intended workflow for now, and the issue overlay exists to support it. The one thing to keep in mind: corrections inherit blind spots (a line never detected never gets corrected), so the issue layer matters for spotting what is *missing* rather than wrong.

## Conventions

- Stages live in `stages/sN_name.py` and register params as a `StageParams` subclass (typed, validated, hashed for caching).
- Pure geometry and image work goes in `stages/segment.py`; it must stay deterministic and weight-free so it can serve as the independent cross-check against hosted layout models.
- Phase 1 runs **no local inference**. Anything with model weights is a hosted endpoint. Do not add `torch` or model downloads.
- Tests run against the real sample PDFs in `data/` (guarded with `skipif`) plus synthetic fixtures where ground truth must be known by construction. Prefer both.
- When a test asserts a count measured from sample data, say so in a comment — those assertions are change-detectors, not specifications.
- **Verify the render by looking at it, not by reading the CSS.** `render/html.py` has no test that can catch a visually broken overlay, and reasoning about it has a poor track record here (three real bugs shipped invisibly). The loop is cheap:

  ```bash
  google-chrome --headless --disable-gpu --no-sandbox --hide-scrollbars \
    --virtual-time-budget=5000 --window-size=880,1200 --force-device-scale-factor=2 \
    --screenshot=/tmp/shot.png "file://$PWD/out/view.html"
  ```

  For anything the screenshot cannot explain, inject a `<script>` that writes measurements into `document.title` and read it back with `--dump-dom` — `getBoundingClientRect`, `getComputedStyle`, and `elementFromPoint` settle in one run what CSS reasoning gets wrong. Note that `elementFromPoint` ignores *ink* overflow, so a glyph painted outside its box returns whatever is behind it.

## Two renderers — do not merge them

`render/html.py` is the **debug** surface: every line absolutely positioned at
its ink box over the scan, scaled to fit. `render/reader.py` is the **reading**
surface: uniform type, structure kept (reading order, band arrangement), pixel
geometry dropped, and the body measure *regularised* — every page reserves a
hashiya column so the text column never changes width. They answer different
questions and share only the provenance convention (corpus text is marked
distinctly from text merely identified).

**The hashiya side alternates by page** — it is the outer margin, so it flips
with the leaf, and the two samples run opposite parities. Take the side from the
page's own notes and fall back to the document's fitted parity; never hard-code
a side.

Everything in `reader.py` is presentational; it must never write back to the
`Document` or decide what a region *is*. Its two typographic inferences —
paragraph merging and the size tier — are reversible from the UI, and its
thresholds (`_PARA_GAP`, `_SMALL_TIER`, `_FULL_MEASURE`, `_AYAH_LINE_FRAC`) are
module constants because they are view settings, not pipeline config: they are
not part of any cache key and must not enter `core/config.py`.

Fonts are embedded as base64 data URIs from the system font directories, so the
file renders on a machine with no Urdu fonts. When a face is missing the CSS
stack takes over — never make a missing font an error.

## Render traps

**Never write `background: currentColor` in a rule that also sets `color`.** `currentColor` resolves to that element's own `color` — which the same rule just changed — so the element paints its background in its text colour and becomes invisible. This shipped three separate times (the region badge, the issue label twice). Region and issue colours are therefore passed as a `--c` custom property and referenced as `var(--c)`; keep it that way.

**Opacity on a marker fades its label too.** Translucent fills belong on a `::before` pseudo-element, so the border and the label stay at full strength. An `opacity:.3` on the marker itself is why issue labels were unreadable.

**A class name that is also a layout class will collide.** The reader's size
tier writes `body` onto its section; a `.body{display:flex}` rule meant for the
row wrapper then turned every region into a flex row and laid its paragraphs out
side by side as columns. The wrapper is `.measure` for that reason. Check any
new class name against the ones already in the stylesheet.

**Grid auto-placement never moves backwards.** In the reader's two-track row,
assigning `grid-column` alone put the hashiya in a *second* row on the pages
where it is the leading track — the note dropped below the text it annotates.
Both cells carry `grid-row:1`.

**The line overlay is `direction: rtl`, so `flex-start` is the right edge.** Using `flex-end` packs the text to the *left*; combined with `transform-origin: right center` in `fit()`, the scale then anchors to the span's overflowed right edge instead of the box's, and the line lands hundreds of pixels off the page — correctly sized, completely misplaced.
