"""Corpus index: one continuous token stream, seeded by rare n-grams.

Two decisions do most of the work here.

**The corpus is one token stream, not 6,236 ayah records.** A book quotes half a
verse, or runs from the end of one ayah into the next, far more often than it
quotes a whole one. Indexing continuously means a partial quote is just a
substring — no special-casing, no per-ayah stitching — and the reference falls
out of the token offsets afterwards.

**Candidates come from diagonal voting.** Every trigram of the query that occurs
in the corpus votes for the *offset* it implies (`corpus position − query
position`). A true match puts many trigrams on the same diagonal; coincidences
scatter. This is seed-and-extend, and it turns a 77k-token search into a dict
lookup, which is what lets alignment be the expensive-but-exact second pass.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from mistara.text.arabic import tokenize


@dataclass(frozen=True)
class Window:
    """A stretch of the corpus worth aligning against."""

    start: int
    end: int
    votes: float


@dataclass
class CorpusIndex:
    """A searchable corpus. Built once, then read-only."""

    name: str
    #: (surah, ayah) per verse — generalizes to (collection, number) for hadith.
    refs: list[tuple[int, int]]
    #: Verbatim words, whole corpus, in order. What gets written back on a match.
    words: list[str]
    #: Normalized forms of the same words. What gets matched.
    tokens: list[str]
    #: token index -> index into `refs`.
    token_ref: list[int]
    meta: dict = field(default_factory=dict)

    _grams: dict[tuple[str, str, str], list[int]] = field(default_factory=dict, repr=False)
    _unigrams: dict[str, list[int]] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------- building --

    @classmethod
    def from_verses(
        cls, name: str, verses: list[tuple[int, int, str]], meta: dict | None = None
    ) -> CorpusIndex:
        refs: list[tuple[int, int]] = []
        words: list[str] = []
        tokens: list[str] = []
        token_ref: list[int] = []

        for surah, ayah, text in verses:
            ref_id = len(refs)
            refs.append((surah, ayah))
            for token in tokenize(text):
                words.append(text[token.start : token.end])
                tokens.append(token.text)
                token_ref.append(ref_id)

        index = cls(
            name=name,
            refs=refs,
            words=words,
            tokens=tokens,
            token_ref=token_ref,
            meta=meta or {},
        )
        index._build_postings()
        return index

    @classmethod
    def load(cls, path: Path | str) -> CorpusIndex:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        verses = [(int(s), int(a), str(t)) for s, a, t in data["verses"]]
        meta = {k: v for k, v in data.items() if k != "verses"}
        return cls.from_verses(data.get("name", "corpus"), verses, meta)

    def _build_postings(self) -> None:
        toks = self.tokens
        grams: dict[tuple[str, str, str], list[int]] = {}
        for i in range(len(toks) - 2):
            grams.setdefault((toks[i], toks[i + 1], toks[i + 2]), []).append(i)
        self._grams = grams

        unigrams: dict[str, list[int]] = {}
        for i, tok in enumerate(toks):
            unigrams.setdefault(tok, []).append(i)
        self._unigrams = unigrams

    # ------------------------------------------------------------ searching --

    def candidates(
        self,
        query: list[str],
        *,
        k: int = 8,
        slack: int = 6,
        max_postings: int = 2000,
        rare_df: int = 60,
    ) -> list[Window]:
        """Rank corpus windows worth aligning the query against.

        A query of one or two tokens has no trigram, so the loop below finds
        nothing and seeding falls through to rare unigrams. That path exists for
        continuation matching (a short verse tail), which is the only caller that
        sends a query this short — the cold path floors query length well above
        it. A rare word still pins the window down; a common one seeds nothing
        and the query resolves to no candidate, which is the safe outcome.
        """
        if not query:
            return []

        n_tokens = max(len(self.tokens), 1)
        votes: dict[int, float] = {}

        for i in range(len(query) - 2):
            postings = self._grams.get((query[i], query[i + 1], query[i + 2]))
            if not postings or len(postings) > max_postings:
                continue
            # Rare n-grams are the informative ones; weight by inverse frequency
            # so that formulaic phrasing cannot outvote a distinctive phrase.
            weight = math.log(1.0 + n_tokens / len(postings))
            for p in postings:
                votes[p - i] = votes.get(p - i, 0.0) + weight

        if not votes:
            votes = self._unigram_votes(query, rare_df=rare_df, n_tokens=n_tokens)
        if not votes:
            return []

        # Indels shift the diagonal by a token or two, so nearby diagonals are
        # the same candidate. Merge them rather than spending alignments on both.
        ordered = sorted(votes.items(), key=lambda kv: -kv[1])
        chosen: list[tuple[int, float]] = []
        for diagonal, score in ordered:
            if any(abs(diagonal - d) <= slack for d, _ in chosen):
                continue
            chosen.append((diagonal, score))
            if len(chosen) >= k:
                break

        limit = len(self.tokens)
        return [
            Window(
                start=max(0, d - slack),
                end=min(limit, d + len(query) + slack),
                votes=score,
            )
            for d, score in chosen
        ]

    def _unigram_votes(
        self, query: list[str], *, rare_df: int, n_tokens: int
    ) -> dict[int, float]:
        """Fallback seeding on rare single words.

        A short quote with one badly-read word may share no trigram with the
        corpus at all. Rare unigrams still pin it down; common ones are ignored
        because they would vote for everything.
        """
        votes: dict[int, float] = {}
        for i, token in enumerate(query):
            postings = self._unigrams.get(token)
            if not postings or len(postings) > rare_df:
                continue
            weight = math.log(1.0 + n_tokens / len(postings))
            for p in postings:
                votes[p - i] = votes.get(p - i, 0.0) + weight
        return votes

    # ------------------------------------------------------------- readback --

    def text_for(self, start: int, end: int) -> str:
        """Verbatim corpus text for a token span — with its diacritics intact."""
        return " ".join(self.words[start:end])

    def ref_for(self, start: int, end: int) -> str:
        """A citation string for a token span, e.g. `quran:5:44` or `quran:2:255-256`."""
        if start >= end or not self.refs:
            return self.name
        first = self.refs[self.token_ref[start]]
        last = self.refs[self.token_ref[min(end, len(self.token_ref)) - 1]]
        if first == last:
            return f"{self.name}:{first[0]}:{first[1]}"
        if first[0] == last[0]:
            return f"{self.name}:{first[0]}:{first[1]}-{last[1]}"
        return f"{self.name}:{first[0]}:{first[1]}-{last[0]}:{last[1]}"

    def __len__(self) -> int:
        return len(self.tokens)
