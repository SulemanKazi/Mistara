"""Fetching and storing the corpus text.

Kept apart from indexing so the download happens once, explicitly, via
`mistara corpus fetch` — no stage reaches for the network.

The stored file holds the verse text **verbatim**, with its provenance beside
it. Normalization happens at load time (`mistara.text.arabic`), never on disk:
a corpus that has been folded on the way in cannot be re-folded differently
later, and cannot be checked against its source.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_CORPUS_DIR = Path("data/corpora")

#: Uthmani text. Indexed as Uthmani deliberately — it is what these books print
#: — while `text.arabic.match_form` folds the Uthmani/Imlaei difference away for
#: matching purposes.
QURAN_URL = "https://raw.githubusercontent.com/risan/quran-json/main/dist/quran.json"

#: Recorded in the corpus file so a result can always be traced to its source.
#: The upstream repository carries no LICENSE file; the Arabic text is credited
#: to The Noble Qur'an Encyclopedia. Terms are therefore **unverified** — see
#: decision D3 in docs/design.md before any distribution.
QURAN_ATTRIBUTION = {
    "text_source": "The Noble Qur'an Encyclopedia (quranenc.com), via risan/quran-json",
    "script": "uthmani",
    "license": "UNVERIFIED — upstream has no LICENSE file; see design decision D3",
}

EXPECTED_SURAHS = 114
EXPECTED_VERSES = 6236


def fetch_quran(
    dest_dir: Path | str = DEFAULT_CORPUS_DIR,
    *,
    url: str = QURAN_URL,
    timeout: float = 60.0,
) -> Path:
    """Download the Quran and write it to `<dest_dir>/quran.json`."""
    import httpx

    response = httpx.get(url, timeout=timeout, follow_redirects=True)
    response.raise_for_status()
    return write_quran(response.json(), dest_dir, url=url)


def write_quran(
    payload: Any, dest_dir: Path | str = DEFAULT_CORPUS_DIR, *, url: str = QURAN_URL
) -> Path:
    """Validate an upstream payload and write it in our own format."""
    verses = _flatten(payload)
    _validate(verses)

    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / "quran.json"
    path.write_text(
        json.dumps(
            {
                "name": "quran",
                **QURAN_ATTRIBUTION,
                "source_url": url,
                "fetched_at": datetime.now(UTC).isoformat(),
                "verses": verses,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def _flatten(payload: Any) -> list[list[Any]]:
    if not isinstance(payload, list):
        raise ValueError("expected a list of surahs from the upstream corpus")
    out: list[list[Any]] = []
    for surah in payload:
        sid = int(surah["id"])
        for verse in surah["verses"]:
            out.append([sid, int(verse["id"]), str(verse["text"])])
    return out


def _validate(verses: list[list[Any]]) -> None:
    """Refuse a short corpus loudly.

    A truncated corpus does not fail — it silently stops matching whatever it is
    missing, and every downstream number quietly degrades instead of breaking.
    """
    surahs = {v[0] for v in verses}
    if len(surahs) != EXPECTED_SURAHS or len(verses) != EXPECTED_VERSES:
        raise ValueError(
            f"corpus looks wrong: {len(surahs)} surahs / {len(verses)} verses, "
            f"expected {EXPECTED_SURAHS} / {EXPECTED_VERSES}"
        )
    if any(not v[2].strip() for v in verses):
        raise ValueError("corpus contains an empty verse")
