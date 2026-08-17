"""Unicode canonicalization for Urdu and Arabic.

This module is used by ground-truth authoring, pipeline output, and every
metric. If those three disagree about what counts as "the same string", every
number the project reports is wrong — so this is deliberately explicit rather
than clever.

Two modes:

* ``strict`` — keeps tashkeel. Use for Arabic/Quranic comparison, where the
  vowel marks are part of the text.
* ``loose``  — strips tashkeel. Use for Urdu CER: Urdu prose is unvocalized, so
  a stray fatha that OCR invented should not be scored as a real error.

Every metric must state which mode it used. A CER number without its mode is
meaningless.
"""

from __future__ import annotations

import re
import unicodedata
from enum import StrEnum

# --------------------------------------------------------------------------- #
# Character classes
# --------------------------------------------------------------------------- #

TATWEEL = "ـ"  # kashida — pure typography, never semantic
ZWNJ = "‌"
ZWJ = "‍"

#: Combining marks: fathatan..sukun, superscript alef, and the Urdu/Quran extras.
TASHKEEL = "".join(
    [
        *(chr(c) for c in range(0x064B, 0x0653)),  # fathatan .. sukun
        "ٔ",  # hamza above
        "ٕ",  # hamza below
        "ٖ",  # subscript alef
        "ٗ",  # inverted damma
        "٘",  # mark noon ghunna
        "ٰ",  # superscript alef (dagger alef)
    ]
)

#: Quranic annotation signs: sajda marks, pause marks, ayah separators.
QURANIC_MARKS = "".join(chr(c) for c in range(0x06D6, 0x06EE))

#: Ornamental brackets used around Quranic quotations in Urdu books.
QURAN_BRACKETS = "﴾﴿۞۝"

_TASHKEEL_RE = re.compile(f"[{re.escape(TASHKEEL)}]")
_QURANIC_RE = re.compile(f"[{re.escape(QURANIC_MARKS)}]")
_WS_RE = re.compile(r"[ \t  -​  　]+")
_NEWLINES_RE = re.compile(r"\s*\n\s*")

#: Codepoints only Urdu uses. Strong positive evidence for Urdu over Arabic.
URDU_ONLY = set("ٹڈڑںھےۓگچپژہۃۂڪڱڳڻۋی")

#: Forms characteristic of Arabic rather than Urdu orthography.
ARABIC_ONLY = set("ةيكأإآؤئى")

#: Letters, both scripts, after folding. Used for script-purity measurement.
ARABIC_LETTERS = set(chr(c) for c in range(0x0620, 0x0660)) | set(
    chr(c) for c in range(0x066E, 0x06D4)
)

#: Fold visually-identical-but-distinct codepoints to one canonical choice.
#: We canonicalize toward the *Urdu* forms, because the corpus of running text
#: is Urdu; Arabic quotations get folded further in `arabic.py` for matching.
_FOLD_MAP = {
    "ي": "ی",  # ARABIC YEH        -> FARSI YEH
    "ى": "ی",  # ALEF MAKSURA      -> FARSI YEH
    "ے": "ے",  # YEH BARREE (kept distinct: semantic in Urdu)
    "ك": "ک",  # ARABIC KAF        -> KEHEH
    "ڪ": "ک",  # SWASH KAF         -> KEHEH
    "ه": "ہ",  # ARABIC HEH        -> HEH GOAL
    "ۃ": "ہ",  # TEH MARBUTA GOAL  -> HEH GOAL
    "ة": "ہ",  # TEH MARBUTA       -> HEH GOAL
    "ھ": "ھ",  # DOACHASHMEE HEH (kept: distinct in Urdu digraphs)
    "أ": "ا",  # ALEF WITH HAMZA ABOVE -> ALEF
    "إ": "ا",  # ALEF WITH HAMZA BELOW -> ALEF
    "آ": "آ",  # ALEF WITH MADDA (kept: distinct in Urdu)
    "ٱ": "ا",  # ALEF WASLA        -> ALEF
}

#: Arabic-Indic and Extended Arabic-Indic digits -> ASCII.
_DIGIT_MAP = {
    **{chr(0x0660 + i): str(i) for i in range(10)},
    **{chr(0x06F0 + i): str(i) for i in range(10)},
}

_PUNCT_MAP = {
    "،": "،",  # Arabic comma (kept)
    "؛": "؛",  # Arabic semicolon (kept)
    "؟": "؟",  # Arabic question mark (kept)
    "۔": "۔",  # Urdu full stop (kept)
}


class Mode(StrEnum):
    STRICT = "strict"  # keep tashkeel — Arabic/Quranic comparison
    LOOSE = "loose"  # strip tashkeel — Urdu CER


def fold_letters(text: str) -> str:
    return "".join(_FOLD_MAP.get(ch, ch) for ch in text)


def fold_digits(text: str) -> str:
    return "".join(_DIGIT_MAP.get(ch, ch) for ch in text)


def strip_tashkeel(text: str) -> str:
    return _TASHKEEL_RE.sub("", text)


def strip_quranic_marks(text: str) -> str:
    return _QURANIC_RE.sub("", text)


def canonical(
    text: str,
    mode: Mode | str = Mode.LOOSE,
    *,
    keep_newlines: bool = True,
) -> str:
    """Canonicalize `text` for storage, comparison, or metric computation."""
    mode = Mode(mode)
    if not text:
        return ""

    out = unicodedata.normalize("NFC", text)
    out = out.replace(TATWEEL, "")
    out = strip_quranic_marks(out)
    if mode is Mode.LOOSE:
        out = strip_tashkeel(out)
    out = fold_letters(out)
    out = fold_digits(out)
    out = out.replace(ZWJ, "")  # ZWJ is presentational; ZWNJ is not

    if keep_newlines:
        out = _NEWLINES_RE.sub("\n", out)
        out = "\n".join(_WS_RE.sub(" ", ln).strip() for ln in out.split("\n"))
    else:
        out = _WS_RE.sub(" ", out.replace("\n", " ")).strip()

    return unicodedata.normalize("NFC", out).strip()


# --------------------------------------------------------------------------- #
# Cheap script measurements — used as verification signals, not classification
# --------------------------------------------------------------------------- #


def script_purity(text: str) -> float:
    """Fraction of non-space characters that are Arabic-script letters.

    Low values mean the model emitted Latin text, mojibake, or markup where it
    should have emitted script. A blunt but effective garbage detector.
    """
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    hits = sum(1 for c in chars if c in ARABIC_LETTERS or c in _DIGIT_MAP.values())
    return hits / len(chars)


def urdu_affinity(text: str) -> float:
    """+1 means strongly Urdu, -1 strongly Arabic, 0 undetermined.

    Counts orthographic tells only. Deliberately not a classifier — it is one
    piece of evidence that S5 combines with typeface and corpus retrieval.
    """
    stripped = strip_tashkeel(unicodedata.normalize("NFC", text))
    urdu = sum(1 for c in stripped if c in URDU_ONLY)
    arabic = sum(1 for c in stripped if c in ARABIC_ONLY)
    total = urdu + arabic
    if total == 0:
        return 0.0
    return (urdu - arabic) / total


def tashkeel_density(text: str) -> float:
    """Share of characters that are vowel marks.

    Quranic Arabic is fully vocalized; Urdu prose almost never is. High density
    is strong evidence of a Quranic quotation.
    """
    base = unicodedata.normalize("NFC", text)
    if not base:
        return 0.0
    marks = len(base) - len(strip_tashkeel(base))
    letters = sum(1 for c in strip_tashkeel(base) if not c.isspace())
    return marks / letters if letters else 0.0


def ornament_density(text: str) -> float:
    """Share of characters that are Quranic ornaments — quotation brackets and
    verse marks (﴾ ﴿ ۞ ۝ and the U+06D6–06ED signs).

    Books wrap Quranic quotations in ornate brackets and separate verses with
    these marks. Their presence is strong evidence a span is Arabic sacred text,
    and — crucially — it is independent of the letters, which the Urdu typeface
    makes ambiguous. Note that `canonical()` strips the U+06Dx marks, so in
    stored line text only the brackets survive; the function still counts both so
    it is correct when called on raw input.
    """
    base = unicodedata.normalize("NFC", text)
    chars = [c for c in base if not c.isspace()]
    if not chars:
        return 0.0
    hits = sum(1 for c in chars if c in QURAN_BRACKETS or c in QURANIC_MARKS)
    return hits / len(chars)


def repeat_rate(text: str, n: int = 12) -> float:
    """Largest share of the text taken up by one repeated n-gram.

    Catches decoding loops, where a VLM emits the same phrase until it runs out
    of tokens. Near 0 for healthy text.
    """
    s = _WS_RE.sub(" ", text).strip()
    if len(s) < n * 2:
        return 0.0
    counts: dict[str, int] = {}
    for i in range(len(s) - n + 1):
        g = s[i : i + n]
        counts[g] = counts.get(g, 0) + 1
    top = max(counts.values())
    return (top * n) / len(s) if top > 1 else 0.0
