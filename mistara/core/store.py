"""Content-addressed artifact store plus a SQLite run ledger.

Two jobs:

1. **Blobs** — page images, crops, debug overlays, raw provider responses. Keyed
   by sha256, so writing the same bytes twice is a no-op.
2. **Ledger** — one row per stage invocation: what ran, with which params, on
   which input, what it produced, whether it passed, what it cost.

The payoff is that re-running a stage with identical inputs is free, which is
what makes retry affordable and evaluation reproducible.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mistara.core.model import ArtifactRef, Document

DEFAULT_STORE = Path("data/store")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS docs (
    doc_id        TEXT PRIMARY KEY,
    source_path   TEXT NOT NULL,
    source_sha256 TEXT NOT NULL,
    state_sha256  TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    run_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id        TEXT NOT NULL,
    stage         TEXT NOT NULL,
    stage_version TEXT NOT NULL,
    params_hash   TEXT NOT NULL,
    input_hash    TEXT NOT NULL,
    output_hash   TEXT,
    status        TEXT NOT NULL,
    passed        INTEGER,
    score         REAL,
    report_json   TEXT NOT NULL,
    started_at    TEXT NOT NULL,
    duration_s    REAL NOT NULL,
    input_tokens  INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    provider_calls INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_runs_cache
    ON runs (stage, stage_version, params_hash, input_hash);
CREATE INDEX IF NOT EXISTS idx_runs_doc ON runs (doc_id, stage);
"""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_hash(obj: Any) -> str:
    """Hash a JSON-able object deterministically (sorted keys, no whitespace)."""
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256_bytes(payload.encode("utf-8"))


class Store:
    def __init__(self, root: Path | str = DEFAULT_STORE) -> None:
        self.root = Path(root)
        self.blobs = self.root / "blobs"
        self.blobs.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "ledger.db"
        self._db = sqlite3.connect(self.db_path)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(_SCHEMA)
        self._db.commit()

    # ---------------------------------------------------------------- blobs --

    def _blob_path(self, sha: str) -> Path:
        return self.blobs / sha[:2] / sha

    def put_blob(self, data: bytes) -> str:
        sha = sha256_bytes(data)
        path = self._blob_path(sha)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(data)
            tmp.rename(path)
        return sha

    def get_blob(self, sha: str) -> bytes:
        path = self._blob_path(sha)
        if not path.exists():
            raise KeyError(f"blob {sha} not in store at {self.root}")
        return path.read_bytes()

    def has_blob(self, sha: str) -> bool:
        return self._blob_path(sha).exists()

    def put_image(
        self, data: bytes, media_type: str = "image/png", **meta: Any
    ) -> ArtifactRef:
        return ArtifactRef(sha256=self.put_blob(data), media_type=media_type, **meta)

    # ------------------------------------------------- provider responses --

    def _response_path(self, key: str) -> Path:
        return self.root / "responses" / key[:2] / f"{key}.txt"

    def put_response(self, key: str, body: str) -> None:
        """Cache a raw provider response against the inputs that produced it."""
        path = self._response_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")

    def get_response(self, key: str) -> str | None:
        path = self._response_path(key)
        return path.read_text(encoding="utf-8") if path.is_file() else None

    # ------------------------------------------------------------ documents --

    def save_document(self, doc: Document) -> str:
        payload = doc.model_dump_json(indent=None).encode("utf-8")
        state_sha = self.put_blob(payload)
        self._db.execute(
            "INSERT INTO docs (doc_id, source_path, source_sha256, state_sha256, updated_at) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(doc_id) DO UPDATE SET "
            "  state_sha256=excluded.state_sha256, updated_at=excluded.updated_at",
            (
                doc.doc_id,
                doc.source_path,
                doc.source_sha256,
                state_sha,
                datetime.now(UTC).isoformat(),
            ),
        )
        self._db.commit()
        return state_sha

    def load_document(self, doc_id: str) -> Document:
        row = self._db.execute(
            "SELECT state_sha256 FROM docs WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"no document {doc_id!r} in store")
        return Document.model_validate_json(self.get_blob(row["state_sha256"]))

    def list_documents(self) -> list[dict[str, Any]]:
        rows = self._db.execute(
            "SELECT doc_id, source_path, updated_at FROM docs ORDER BY updated_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def resolve_doc_id(self, prefix: str) -> str:
        """Allow short doc-id prefixes on the CLI."""
        rows = self._db.execute(
            "SELECT doc_id FROM docs WHERE doc_id LIKE ?", (prefix + "%",)
        ).fetchall()
        if not rows:
            raise KeyError(f"no document matching {prefix!r}")
        if len(rows) > 1:
            raise KeyError(f"{prefix!r} is ambiguous: {[r['doc_id'] for r in rows]}")
        return rows[0]["doc_id"]

    # ----------------------------------------------------------------- runs --

    def record_run(
        self,
        *,
        doc_id: str,
        stage: str,
        stage_version: str,
        params_hash: str,
        input_hash: str,
        output_hash: str | None,
        status: str,
        passed: bool | None,
        score: float | None,
        report: dict[str, Any],
        started_at: datetime,
        duration_s: float,
    ) -> int:
        cost = report.get("cost") or {}
        cur = self._db.execute(
            "INSERT INTO runs (doc_id, stage, stage_version, params_hash, input_hash, "
            "output_hash, status, passed, score, report_json, started_at, duration_s, "
            "input_tokens, output_tokens, provider_calls) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                doc_id,
                stage,
                stage_version,
                params_hash,
                input_hash,
                output_hash,
                status,
                None if passed is None else int(passed),
                score,
                json.dumps(report, ensure_ascii=False, default=str),
                started_at.isoformat(),
                duration_s,
                int(cost.get("input_tokens", 0)),
                int(cost.get("output_tokens", 0)),
                int(cost.get("provider_calls", 0)),
            ),
        )
        self._db.commit()
        return int(cur.lastrowid or 0)

    def find_cached_run(
        self, *, stage: str, stage_version: str, params_hash: str, input_hash: str
    ) -> dict[str, Any] | None:
        row = self._db.execute(
            "SELECT * FROM runs WHERE stage=? AND stage_version=? AND params_hash=? "
            "AND input_hash=? AND status='ok' ORDER BY run_id DESC LIMIT 1",
            (stage, stage_version, params_hash, input_hash),
        ).fetchone()
        return dict(row) if row else None

    def runs_for(self, doc_id: str) -> list[dict[str, Any]]:
        rows = self._db.execute(
            "SELECT run_id, stage, stage_version, status, passed, score, duration_s, "
            "started_at, input_tokens, output_tokens, provider_calls "
            "FROM runs WHERE doc_id=? ORDER BY run_id",
            (doc_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        self._db.close()
