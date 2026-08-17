"""Smith-Waterman local alignment over token sequences.

**Local, not global**, because that is precisely the partial-quotation
requirement: local alignment finds the best-matching *substring of both
sequences*, so a half-verse quote lands on its exact span inside the ayah, and
an inline quote lands on its exact span inside an Urdu sentence — with no
special-casing at either end.

**Tokens are compared by character similarity, not equality.** OCR errors are
sub-word: one misplaced dot turns a correctly-read word into a different valid
word. Exact token equality would score that as a total mismatch and throw away
an otherwise perfect verse, so a substitution scores on a graded scale instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from rapidfuzz.distance import Levenshtein

#: Traceback directions.
_STOP, _DIAG, _UP, _LEFT = 0, 1, 2, 3


@lru_cache(maxsize=1 << 18)
def token_similarity(a: str, b: str) -> float:
    """Character-level similarity of two tokens, in [0, 1]."""
    if a == b:
        return 1.0
    return Levenshtein.normalized_similarity(a, b)


@dataclass(frozen=True)
class Alignment:
    score: float
    q_start: int
    q_end: int
    t_start: int
    t_end: int

    @property
    def q_len(self) -> int:
        return self.q_end - self.q_start

    @property
    def t_len(self) -> int:
        return self.t_end - self.t_start


def smith_waterman(
    query: list[str],
    target: list[str],
    *,
    gap: float = -1.0,
    min_similarity: float = 0.5,
) -> Alignment | None:
    """Best local alignment of `query` against `target`.

    A substitution scores `2 * similarity - 1`, so identical tokens score +1,
    tokens half-alike score 0, and unrelated tokens score -1. `min_similarity`
    floors that: below it a pair is treated as unrelated rather than
    fractionally good, which stops a run of vaguely-similar Urdu words from
    accumulating a positive score.
    """
    n, m = len(query), len(target)
    if n == 0 or m == 0:
        return None

    # One extra row/column of zeros: the classic Smith-Waterman sentinel that
    # lets an alignment start anywhere.
    h = [[0.0] * (m + 1) for _ in range(n + 1)]
    back = [[_STOP] * (m + 1) for _ in range(n + 1)]
    best = 0.0
    best_i = best_j = 0

    for i in range(1, n + 1):
        qi = query[i - 1]
        h_prev, h_cur, back_cur = h[i - 1], h[i], back[i]
        for j in range(1, m + 1):
            sim = token_similarity(qi, target[j - 1])
            if sim < min_similarity:
                sim = 0.0
            diagonal = h_prev[j - 1] + (2.0 * sim - 1.0)
            up = h_prev[j] + gap
            left = h_cur[j - 1] + gap

            value = diagonal
            move = _DIAG
            if up > value:
                value, move = up, _UP
            if left > value:
                value, move = left, _LEFT
            if value <= 0.0:
                value, move = 0.0, _STOP

            h_cur[j] = value
            back_cur[j] = move
            if value > best:
                best, best_i, best_j = value, i, j

    if best <= 0.0:
        return None

    i, j = best_i, best_j
    while i > 0 and j > 0 and back[i][j] != _STOP:
        move = back[i][j]
        if move == _DIAG:
            i, j = i - 1, j - 1
        elif move == _UP:
            i -= 1
        else:
            j -= 1

    return Alignment(score=best, q_start=i, q_end=best_i, t_start=j, t_end=best_j)


def identity(query: list[str], target: list[str]) -> float:
    """Character-level identity of two aligned token runs.

    Measured on the joined strings rather than accumulated from the DP score, so
    the number means exactly what it says — matched characters over aligned
    length — independent of whatever gap penalty produced the span.
    """
    if not query or not target:
        return 0.0
    return Levenshtein.normalized_similarity(" ".join(query), " ".join(target))
