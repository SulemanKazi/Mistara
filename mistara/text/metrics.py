"""Error-rate metrics. Every function states the normalization mode it used."""

from __future__ import annotations

from dataclasses import dataclass

from rapidfuzz.distance import Levenshtein

from mistara.text.canonical import Mode, canonical


@dataclass(frozen=True)
class ErrorRate:
    rate: float
    edits: int
    reference_units: int
    mode: Mode

    def __str__(self) -> str:
        return f"{self.rate:.4f} ({self.edits}/{self.reference_units}, {self.mode})"


def cer(hypothesis: str, reference: str, mode: Mode | str = Mode.LOOSE) -> ErrorRate:
    """Character error rate after canonicalization."""
    mode = Mode(mode)
    hyp = canonical(hypothesis, mode, keep_newlines=False)
    ref = canonical(reference, mode, keep_newlines=False)
    if not ref:
        return ErrorRate(0.0 if not hyp else 1.0, len(hyp), 0, mode)
    edits = Levenshtein.distance(hyp, ref)
    return ErrorRate(edits / len(ref), edits, len(ref), mode)


def wer(hypothesis: str, reference: str, mode: Mode | str = Mode.LOOSE) -> ErrorRate:
    """Word error rate after canonicalization."""
    mode = Mode(mode)
    hyp = canonical(hypothesis, mode, keep_newlines=False).split()
    ref = canonical(reference, mode, keep_newlines=False).split()
    if not ref:
        return ErrorRate(0.0 if not hyp else 1.0, len(hyp), 0, mode)
    edits = Levenshtein.distance(hyp, ref)
    return ErrorRate(edits / len(ref), edits, len(ref), mode)


def agreement(a: str, b: str, mode: Mode | str = Mode.LOOSE) -> float:
    """1 - CER between two independent readings of the same image.

    This is the core manufactured confidence signal: two reads that agree are
    probably both right; two reads that diverge mark genuinely ambiguous ink.
    Symmetric by construction (we normalize by the longer string).
    """
    mode = Mode(mode)
    x = canonical(a, mode, keep_newlines=False)
    y = canonical(b, mode, keep_newlines=False)
    if not x and not y:
        return 1.0
    denom = max(len(x), len(y))
    if denom == 0:
        return 1.0
    return max(0.0, 1.0 - Levenshtein.distance(x, y) / denom)
