"""S6a — Corpus matching: verify sacred text instead of reading it.

The product's central claim lives here. For Quranic Arabic, OCR output is a
*search query*, never the answer: the answer comes from an authenticated corpus
via local alignment. A verse that matches is replaced with the corpus text,
diacritics intact, and the OCR reading is retained in provenance so the
substitution is always reversible and always auditable.

No model is called. The whole stage is deterministic, which is what lets it be
re-run for free and lets a wrong answer be traced to a threshold rather than to
a sampling accident.

**Two different questions, two different signals.** `identity` says whether the
*wording* is right; `margin` says whether the *citation* is. They come apart
constantly, because the Quran repeats phrasing verbatim — `وَمَنْ لَّمْ یَحْکُمْ
بِمَآ اَنْزَلَ اللّٰہُ` occurs identically at 5:44, 5:45 and 5:47. A line like that
can be replaced with total confidence and still be cited wrongly, so a low
margin annotates the citation rather than blocking the replacement.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from mistara.core.config import MatchConfig
from mistara.core.model import (
    Document,
    Language,
    Line,
    Page,
    Provenance,
    ProvenanceKind,
    Region,
    Span,
)
from mistara.core.stage import Stage, StageParams, Verdict, Verifier
from mistara.core.store import stable_hash
from mistara.corpus.quran import CorpusMatcher, Match, load_quran
from mistara.corpus.source import DEFAULT_CORPUS_DIR
from mistara.text.canonical import tashkeel_density

#: Above this share of vocalized characters, a line is probably sacred text.
#: Used only to count what we *failed* to resolve — never to gate matching,
#: which runs against every line regardless. The matcher is the detector.
VOCALIZED_HINT = 0.25

REPLACED = "replaced"
CITED = "cited"


class MatchParams(StageParams):
    corpus_dir: str = Field(
        default=str(DEFAULT_CORPUS_DIR),
        description="Directory holding quran.json (see `mistara corpus fetch`).",
    )
    cfg: MatchConfig = Field(default_factory=MatchConfig)


class LineMatch(BaseModel):
    page_no: int
    region_id: str
    line_id: str
    ref: str
    identity: float
    coverage: float
    margin: float
    tokens_matched: int
    action: str  # replaced | cited
    corpus_start: int
    corpus_end: int
    #: Character range of the match within the line, and the text that replaces
    #: it. Carried on the report so `apply()` needs no corpus, and so the run
    #: ledger records exactly what was substituted — which is the audit trail.
    char_start: int
    char_end: int
    corpus_text: str
    #: Position within the region. Continuity is only meaningful between lines
    #: that are actually adjacent on the page.
    line_index: int = 0


class PageMatch(BaseModel):
    page_no: int
    matches: list[LineMatch] = Field(default_factory=list)
    lines_considered: int = 0
    #: Lines that look vocalized (so probably sacred) but resolved to nothing.
    unresolved: int = 0
    candidates_considered: int = 0


class MatchResult(BaseModel):
    pages: list[PageMatch] = Field(default_factory=list)
    corpus: str = ""
    corpus_tokens: int = 0


# --------------------------------------------------------------------------- #
# Policy
# --------------------------------------------------------------------------- #


def decide(match: Match, cfg: MatchConfig, *, continues: bool = False) -> str | None:
    """What to do with an alignment: replace it, cite it, or ignore it.

    Ordered by how much damage a wrong answer does. A replacement rewrites the
    page, so it needs the whole line to be accounted for; a citation only adds
    a reference, so it needs a long-enough span and nothing more.

    `continues` marks a line that contiguously continues the passage the
    previous accepted line matched. That is independent evidence the length
    floor cannot see, so a continuation may be acted on down to a shorter floor —
    which is how 2-3 word verse tails get resolved instead of dropped.
    """
    if match.identity < cfg.min_identity:
        return None
    # Length floor: a single common word shared with the corpus matches Urdu
    # prose at identity 1.0, and coverage cannot catch that (an inline quote is
    # a low-coverage match by definition). A contiguous continuation is exempt —
    # a random short line will not also land on the exact next corpus position.
    floor = cfg.min_continuation_tokens if continues else cfg.min_span_tokens
    if match.tokens_matched < floor:
        return None
    if match.coverage >= cfg.min_coverage:
        return REPLACED
    # A continuation is restored, not merely cited. Its location is vouched for
    # by continuity and its boundary is the aligner's exact token span (the
    # citation beside a verse tail is left untouched), so it carries the same
    # confidence as a whole-verse match — which is what makes a trailing tail
    # verified/green rather than only source-identified.
    return REPLACED if (continues or cfg.replace_inline) else CITED


# --------------------------------------------------------------------------- #
# Verifier
# --------------------------------------------------------------------------- #


class MatchVerifier(Verifier[MatchResult]):
    name = "match"

    def verify(self, doc: Document, output: MatchResult, params: MatchParams) -> Verdict:
        findings: list[str] = []
        accepted = [m for p in output.pages for m in p.matches]
        considered = sum(p.lines_considered for p in output.pages)
        unresolved = sum(p.unresolved for p in output.pages)

        if not output.corpus_tokens:
            return Verdict.fail("corpus is empty — run `mistara corpus fetch`")

        metrics: dict[str, float] = {
            "match.replaced": float(sum(1 for m in accepted if m.action == REPLACED)),
            "match.cited": float(sum(1 for m in accepted if m.action == CITED)),
            "match.lines_considered": float(considered),
            "match.unresolved": float(unresolved),
            "match.candidates_considered": float(
                sum(p.candidates_considered for p in output.pages)
            ),
        }

        if accepted:
            metrics["match.identity"] = sum(m.identity for m in accepted) / len(accepted)
            metrics["match.margin_min"] = min(m.margin for m in accepted)
            ambiguous = [m for m in accepted if m.margin < params.cfg.min_margin]
            metrics["match.ambiguous_citations"] = float(len(ambiguous))
            for m in ambiguous[:8]:
                findings.append(
                    f"page {m.page_no}: {m.line_id} matched {m.ref} at identity "
                    f"{m.identity:.2f} but margin {m.margin:.2f} — this wording also "
                    f"occurs elsewhere, so the TEXT is safe and the CITATION is not"
                )

        # Independent of the matcher: does the sequence of matches read like a
        # passage? A page quoting a verse produces consecutive corpus spans. A
        # false match lands somewhere unrelated and breaks the run — evidence no
        # single alignment score can provide.
        continuity, runs = _continuity(output)
        if runs:
            metrics["match.run_continuity"] = continuity
            if continuity < 1.0:
                findings.append(
                    f"{runs - int(continuity * runs)} of {runs} consecutive match "
                    f"pair(s) are not contiguous in the corpus — check those lines"
                )

        if unresolved:
            findings.append(
                f"{unresolved} vocalized line(s) matched nothing — either not "
                f"Quranic (Hadith is not indexed yet) or read too poorly to align"
            )

        return Verdict(
            passed=not findings,
            score=metrics.get("match.identity"),
            findings=findings,
            metrics=metrics,
        )


#: Matched lines further apart than this are not one passage being quoted, so
#: their corpus positions carry no expectation of continuity.
ADJACENT_LINES = 2


def _continuity(output: MatchResult) -> tuple[float, int]:
    """Share of adjacent matched lines that continue the same corpus passage.

    Only lines that are *neighbours on the page* are compared. A block of verse
    should run straight down the corpus; a paragraph of prose that happens to
    quote 3:130 and then 5:44 should not, and counting that as a break would
    make the signal fire on correct output — which it did, before this guard.
    """
    good = total = 0
    for page in output.pages:
        by_region: dict[str, list[LineMatch]] = {}
        for m in page.matches:
            by_region.setdefault(m.region_id, []).append(m)
        for matches in by_region.values():
            for prev, nxt in zip(matches, matches[1:], strict=False):
                if nxt.line_index - prev.line_index > ADJACENT_LINES:
                    continue
                total += 1
                # Contiguous, or a short hop — a line break can drop a word.
                if 0 <= nxt.corpus_start - prev.corpus_end <= 3:
                    good += 1
    return (good / total if total else 1.0), total


# --------------------------------------------------------------------------- #
# Stage
# --------------------------------------------------------------------------- #


class MatchStage(Stage[MatchParams, MatchResult]):
    name = "s6a_match"
    version = "1"
    params_model = MatchParams
    verifier = MatchVerifier()

    def run(self, doc: Document, params: MatchParams) -> MatchResult:
        index = load_quran(Path(params.corpus_dir))
        cfg = params.cfg
        matcher = CorpusMatcher(
            index,
            candidates=cfg.candidates,
            slack=cfg.window_slack,
            gap=cfg.gap_penalty,
            min_similarity=cfg.min_token_similarity,
            min_query_tokens=cfg.min_query_tokens,
            min_continuation_query_tokens=cfg.min_continuation_tokens,
            tie_epsilon=cfg.tie_epsilon,
        )

        pages: list[PageMatch] = []
        for page in doc.pages:
            pm = PageMatch(page_no=page.page_no)
            for region in page.ordered_regions():
                # Reading order carries the prior: a line that continues where
                # the previous one stopped is almost certainly the same passage.
                prefer_after: int | None = None
                for line_index, line in enumerate(region.lines):
                    if not line.text.strip():
                        continue
                    pm.lines_considered += 1
                    match = matcher.match(line.text, prefer_after=prefer_after)
                    if match is None:
                        pm.unresolved += _looks_sacred(line)
                        continue
                    pm.candidates_considered += match.candidates
                    # Does this line pick up where the previous accepted one left
                    # off? If so it may be acted on below the isolated-line floor.
                    continues = (
                        prefer_after is not None
                        and 0 <= match.corpus_start - prefer_after <= cfg.continuation_max_gap
                    )
                    action = decide(match, cfg, continues=continues)
                    if action is None:
                        pm.unresolved += _looks_sacred(line)
                        continue
                    prefer_after = match.corpus_end
                    pm.matches.append(
                        LineMatch(
                            page_no=page.page_no,
                            region_id=region.region_id,
                            line_id=line.line_id,
                            ref=match.ref,
                            identity=match.identity,
                            coverage=match.coverage,
                            margin=match.margin,
                            tokens_matched=match.tokens_matched,
                            action=action,
                            corpus_start=match.corpus_start,
                            corpus_end=match.corpus_end,
                            char_start=match.char_start,
                            char_end=match.char_end,
                            corpus_text=match.corpus_text,
                            line_index=line_index,
                        )
                    )
            pages.append(pm)

        return MatchResult(
            pages=pages, corpus=index.name, corpus_tokens=len(index)
        )

    def apply(self, doc: Document, output: MatchResult) -> Document:
        """Rewrite matched spans, keeping the OCR reading in provenance."""
        by_line = {
            (p.page_no, m.line_id): m for p in output.pages for m in p.matches
        }
        if not by_line:
            return doc

        pages: list[Page] = []
        for page in doc.pages:
            regions: list[Region] = []
            for region in page.regions:
                lines = [
                    _rewrite(line, by_line[(page.page_no, line.line_id)])
                    if (page.page_no, line.line_id) in by_line
                    else line
                    for line in region.lines
                ]
                regions.append(region.model_copy(update={"lines": lines}))
            pages.append(page.model_copy(update={"regions": regions}))
        return doc.model_copy(update={"pages": pages})

    def input_hash(self, doc: Document, params: MatchParams) -> str:
        return stable_hash(
            {
                "text": [
                    ln.text
                    for page in doc.pages
                    for region in page.regions
                    for ln in region.lines
                ],
                "params": params.hash(),
            }
        )

    def counts(self, output: MatchResult) -> dict[str, int]:
        return {
            "replaced": sum(
                1 for p in output.pages for m in p.matches if m.action == REPLACED
            ),
            "cited": sum(
                1 for p in output.pages for m in p.matches if m.action == CITED
            ),
            "unresolved": sum(p.unresolved for p in output.pages),
        }


def _looks_sacred(line: Line) -> int:
    """1 if a line is vocalized enough that failing to resolve it is notable."""
    return int(tashkeel_density(line.text) > VOCALIZED_HINT)


def _rewrite(line: Line, match: LineMatch) -> Line:
    """Splice the corpus text into one line, or just annotate it.

    Splicing the matched *range* rather than overwriting the whole line is what
    keeps a line like `الْکٰفِرُوْنَ (5 مائدہ 44)` intact — the verse is restored and
    the book's own citation beside it survives.
    """
    start, end = match.char_start, match.char_end
    if not (0 <= start < end <= len(line.text)):
        return line

    # Re-running the stage must not destroy the audit trail. On a second pass the
    # line already holds corpus text, so `line.text[start:end]` is no longer the
    # OCR reading — carry the earliest one forward instead. Without this, running
    # `match` twice silently overwrites the only record of what the page said.
    original = next(
        (
            span.provenance.original_text
            for span in line.spans
            if span.provenance.kind is ProvenanceKind.QURAN
            and span.provenance.original_text
        ),
        None,
    ) or line.text[start:end]
    corpus_text = match.corpus_text

    if match.action == REPLACED:
        text = line.text[:start] + corpus_text + line.text[end:]
        sacred_end = start + len(corpus_text)
        provenance = Provenance(
            kind=ProvenanceKind.QURAN,
            source_ref=match.ref,
            original_text=original,
            score=match.identity,
        )
    else:
        text = line.text
        sacred_end = end
        provenance = Provenance(
            # Text is still the OCR reading — only the source is now known.
            kind=ProvenanceKind.OCR,
            source_ref=match.ref,
            score=match.identity,
        )

    # The prose either side of a quote keeps the line's pre-existing tag rather
    # than resetting to UNKNOWN — otherwise an inline quotation silently strips
    # the language off the Urdu around it, and `match` output is nonsense on a
    # mixed line even before S5 re-routes. S5 supersedes these tags when it runs.
    prior_lang = line.spans[0].language if line.spans else Language.UNKNOWN

    spans: list[Span] = []
    if start > 0:
        spans.append(
            Span(
                span_id=f"{line.line_id}s0",
                char_start=0,
                char_end=start,
                language=prior_lang,
            )
        )
    spans.append(
        Span(
            span_id=f"{line.line_id}s1",
            char_start=start,
            char_end=sacred_end,
            language=Language.ARABIC,
            provenance=provenance,
            signals={"match.identity": match.identity, "match.margin": match.margin},
        )
    )
    if sacred_end < len(text):
        # Whatever the aligner did not verify stays OCR — including a lone
        # proclitic (و ف ب ل …) left dangling when a verse wraps across a line
        # break. Uthmani fuses that proclitic into the *next* word, so there is
        # no standalone corpus token to match it against; it is honestly
        # un-verified here even though the next line's replacement reintroduces
        # it. See "verse that wraps a proclitic" in CLAUDE.md. Not written back.
        spans.append(
            Span(
                span_id=f"{line.line_id}s2",
                char_start=sacred_end,
                char_end=len(text),
                language=prior_lang,
            )
        )

    return line.model_copy(
        update={
            "text": text,
            "spans": spans,
            "signals": {
                **line.signals,
                "match.identity": match.identity,
                "match.margin": match.margin,
                "match.coverage": match.coverage,
            },
        }
    )


