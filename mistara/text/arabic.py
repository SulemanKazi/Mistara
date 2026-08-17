"""Rasm-level normalization — the form in which Arabic text is *matched*.

Deliberately separate from `canonical.py`. That module produces the form we
**store and score**: it keeps enough distinction to compute a meaningful CER.
This module produces the form we **look up**, and it is far more aggressive,
because a printed book and a corpus edition disagree in ways that carry no
meaning at all:

* the book is set in an Urdu typeface, so Arabic ي ك ه arrive as ی ک ہ;
* the corpus is Uthmani (ٱ, dagger alef, Quranic pause marks); the book is
  usually closer to Imlaei;
* OCR at ~106 DPI loses and invents diacritics freely.

None of that changes which verse is on the page, so all of it is folded away
before matching. What survives is close to the *rasm* — the consonantal
skeleton — which is exactly the level at which two editions of the same verse
genuinely agree.

The same function must be applied to the corpus and the query. If they ever
diverge, every match score silently becomes meaningless.
"""

from __future__ import annotations

import re
import unicodedata
from typing import NamedTuple

from mistara.text.canonical import TATWEEL

#: Fold every letter that two editions (or an Urdu typesetter) might disagree
#: about onto one representative. Targets are arbitrary but must be stable —
#: only agreement between corpus and query matters, not which form wins.
_MATCH_FOLD = {
    # alef family, including the Uthmani wasla
    "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ٲ": "ا", "ٳ": "ا",
    # ya family: Arabic yeh, Farsi yeh, alef maksura, hamza-on-ya, yeh barree
    "ي": "ی", "ى": "ی", "ئ": "ی", "ے": "ی", "ۓ": "ی", "ۍ": "ی", "ې": "ی",
    # waw family
    "ؤ": "و", "ٶ": "و", "ۇ": "و", "ۆ": "و", "ۈ": "و", "ۋ": "و",
    # kaf family
    "ك": "ک", "ڪ": "ک", "ګ": "ک", "ڬ": "ک",
    # heh family, including both teh marbutas
    "ه": "ہ", "ة": "ہ", "ۃ": "ہ", "ۀ": "ہ", "ھ": "ہ", "ﻩ": "ہ",
}

#: Bare hamza is dropped rather than folded. It is the single most volatile
#: character between editions and the first casualty of a low-DPI scan, and one
#: character cannot carry a false match on its own.
_DROP = {"ء", "ٴ"}

_ARABIC_RANGES = ((0x0620, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF))
_WORD_RE = re.compile(r"\S+")


def _is_arabic_letter(ch: str) -> bool:
    if unicodedata.category(ch) != "Lo":
        return False
    code = ord(ch)
    return any(lo <= code <= hi for lo, hi in _ARABIC_RANGES)


def match_form(text: str) -> str:
    """Reduce `text` to the form used for corpus lookup.

    Every combining mark is dropped by Unicode *category* rather than by an
    enumerated list. Uthmani text carries annotation marks that no hand-written
    range quite covers, and a mark we forgot to strip is a mismatch we would
    never explain.
    """
    if not text:
        return ""

    # NFKC also normalizes Arabic presentation forms, which OCR output and some
    # corpus editions still contain.
    out = unicodedata.normalize("NFKC", text).replace(TATWEEL, "")
    out = "".join(ch for ch in out if unicodedata.category(ch) != "Mn")
    out = "".join(_MATCH_FOLD.get(ch, ch) for ch in out)
    out = "".join(
        ch if _is_arabic_letter(ch) and ch not in _DROP else " " for ch in out
    )
    return " ".join(out.split())


class Token(NamedTuple):
    """A matchable word, with its span in the *original* string.

    The offsets are what make partial matches usable: an inline Quranic quote
    inside an Urdu sentence has to come back as a character range we can attach
    a `Span` to, not merely as a score.
    """

    text: str
    start: int
    end: int


def tokenize(text: str) -> list[Token]:
    """Split into matchable tokens, preserving original character offsets."""
    tokens: list[Token] = []
    for m in _WORD_RE.finditer(text):
        normalized = match_form(m.group()).replace(" ", "")
        if normalized:
            tokens.append(Token(normalized, m.start(), m.end()))
    return tokens


def match_tokens(text: str) -> list[str]:
    return [t.text for t in tokenize(text)]
