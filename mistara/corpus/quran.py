"""Matching a noisy OCR line against an authenticated corpus.

Ties the index and the aligner together and reports, for one query: where it
matched, how well, how much of the query it accounted for, and — the part that
matters most for a *zero false replacement* target — whether anywhere else in
the corpus matched nearly as well.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mistara.corpus.align import Alignment, identity, smith_waterman
from mistara.corpus.index import CorpusIndex
from mistara.corpus.source import DEFAULT_CORPUS_DIR
from mistara.text.arabic import tokenize


@dataclass(frozen=True)
class Match:
    """One query's best corpus alignment."""

    ref: str
    identity: float
    coverage: float
    #: Best identity minus the best identity anywhere else in the corpus. A high
    #: identity with a low margin means the phrase is formulaic and the *location*
    #: is a guess, which is a false replacement waiting to happen.
    margin: float
    candidates: int
    score: float
    #: Character span of the match within the original query string. This is what
    #: makes an inline quote attachable to a `Span` instead of merely scored.
    char_start: int
    char_end: int
    corpus_start: int
    corpus_end: int
    #: Verbatim corpus text, diacritics intact — what gets written back.
    corpus_text: str
    tokens_matched: int
    query_tokens: int

    @property
    def is_whole_query(self) -> bool:
        return self.tokens_matched == self.query_tokens


class CorpusMatcher:
    """Finds the best corpus location for a query, or reports nothing."""

    def __init__(
        self,
        index: CorpusIndex,
        *,
        candidates: int = 8,
        slack: int = 6,
        gap: float = -1.0,
        min_similarity: float = 0.5,
        min_query_tokens: int = 3,
        min_continuation_query_tokens: int = 1,
        tie_epsilon: float = 0.75,
    ) -> None:
        self.index = index
        self.candidates = candidates
        self.slack = slack
        self.gap = gap
        self.min_similarity = min_similarity
        self.min_query_tokens = min_query_tokens
        #: A shorter lookup floor, used only when `prefer_after` places us mid-
        #: passage. It only lets a short line be *looked up*; whether the result
        #: is actually a contiguous continuation is decided by the caller.
        self.min_continuation_query_tokens = min_continuation_query_tokens
        self.tie_epsilon = tie_epsilon

    def match(self, text: str, *, prefer_after: int | None = None) -> Match | None:
        """Best corpus match for `text`.

        `prefer_after` is the corpus position where the previous line's match
        ended. Quranic wording repeats verbatim across verses, so alignment
        alone often cannot say *which* occurrence a line is — but a page reads
        in order, and a line that continues where the last one stopped is
        overwhelmingly likely to be the same passage. Only near-ties are broken
        this way; a clearly better match elsewhere still wins.
        """
        tokens = tokenize(text)
        # Mid-passage (prefer_after set), allow a shorter lookup so a verse tail
        # can be found at all; the contiguity guard downstream is what keeps it
        # honest. Cold, a query still has to clear the full floor.
        floor = (
            self.min_continuation_query_tokens
            if prefer_after is not None
            else self.min_query_tokens
        )
        if len(tokens) < floor:
            return None
        query = [t.text for t in tokens]

        windows = self.index.candidates(
            query, k=self.candidates, slack=self.slack
        )
        if not windows:
            return None

        scored: list[tuple[Alignment, int, float]] = []
        for window in windows:
            target = self.index.tokens[window.start : window.end]
            alignment = smith_waterman(
                query, target, gap=self.gap, min_similarity=self.min_similarity
            )
            if alignment is None:
                continue
            t0 = window.start + alignment.t_start
            t1 = window.start + alignment.t_end
            ident = identity(
                query[alignment.q_start : alignment.q_end],
                self.index.tokens[t0:t1],
            )
            scored.append((alignment, window.start, ident))

        if not scored:
            return None

        scored.sort(key=lambda s: -s[0].score)
        if prefer_after is not None and len(scored) > 1:
            cutoff = scored[0][0].score - self.tie_epsilon
            tied = [s for s in scored if s[0].score >= cutoff]
            if len(tied) > 1:
                tied.sort(key=lambda s: _continuity_gap(s, prefer_after))
                scored = tied + [s for s in scored if s not in tied]

        best, offset, best_identity = scored[0]
        b0, b1 = offset + best.t_start, offset + best.t_end

        return Match(
            ref=self.index.ref_for(b0, b1),
            identity=best_identity,
            coverage=best.q_len / len(query),
            margin=best_identity - self._runner_up(scored, b0, b1),
            candidates=len(windows),
            score=best.score,
            char_start=tokens[best.q_start].start,
            char_end=tokens[best.q_end - 1].end,
            corpus_start=b0,
            corpus_end=b1,
            corpus_text=self.index.text_for(b0, b1),
            tokens_matched=best.q_len,
            query_tokens=len(query),
        )

    @staticmethod
    def _runner_up(
        scored: list[tuple[Alignment, int, float]], b0: int, b1: int
    ) -> float:
        """Best identity from a corpus location that does *not* overlap the winner.

        Overlapping candidates are the same match found twice; only a genuinely
        different location tells us the query is ambiguous.
        """
        for alignment, offset, ident in scored[1:]:
            t0, t1 = offset + alignment.t_start, offset + alignment.t_end
            if t1 <= b0 or t0 >= b1:
                return ident
        return 0.0


def _continuity_gap(
    scored: tuple[Alignment, int, float], prefer_after: int
) -> tuple[int, int]:
    """How well a candidate continues from `prefer_after`.

    Sorts forward continuations ahead of backward ones at equal distance: text
    runs on down the page, so a candidate that starts *before* where the last
    line ended is the less likely reading.
    """
    alignment, offset, _ = scored
    delta = (offset + alignment.t_start) - prefer_after
    return (0 if delta >= 0 else 1, abs(delta))


def load_quran(corpus_dir: Path | str = DEFAULT_CORPUS_DIR) -> CorpusIndex:
    """Load the Quran index, with an actionable error when it is missing."""
    path = Path(corpus_dir) / "quran.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"no Quran corpus at {path} — run `mistara corpus fetch` first"
        )
    return CorpusIndex.load(path)
