"""S0 — Ingest: PDF to page images in the artifact store.

The important subtlety is resolution. Many scanned PDFs are a thin wrapper: a
US-Letter page object containing one embedded bitmap that is the actual scan.
Rasterizing such a page at "400 DPI" does not produce 400 DPI of information —
it upsamples the embedded bitmap and invents pixels, which costs vision tokens
and adds nothing.

So: when a page is a single full-bleed image, we extract that image at its
native resolution. We rasterize only when we have to, and we never upsample past
what the source actually contains.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import pymupdf
from pydantic import BaseModel, Field

from mistara.core.config import IngestConfig
from mistara.core.model import ArtifactRef, Document, HistoryEntry, Page, ViewName
from mistara.core.stage import Cost, Stage, StageParams, StageReport, Verdict, Verifier
from mistara.core.store import Store, sha256_file


class IngestParams(StageParams):
    """Per-run options plus the tunables from the active config profile.

    Config values are embedded rather than looked up so that `params.hash()`
    covers them: change a threshold and this stage re-runs instead of serving a
    stale cache entry, and the ledger records which settings produced a result.
    """

    max_pages: int | None = Field(default=None, description="Ingest only the first N pages.")
    cfg: IngestConfig = Field(default_factory=IngestConfig)

    @property
    def target_dpi(self) -> float:
        return self.cfg.target_dpi

    @property
    def native_image_coverage(self) -> float:
        return self.cfg.native_image_coverage


class PageImage(BaseModel):
    page_no: int
    width_px: int
    height_px: int
    source_dpi: float | None
    native: bool
    png: bytes
    gray: bytes

    model_config = {"arbitrary_types_allowed": True}


class IngestResult(BaseModel):
    source_path: str
    source_sha256: str
    pages: list[PageImage]
    model_config = {"arbitrary_types_allowed": True}


def _to_png(arr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", arr)
    if not ok:
        raise RuntimeError("failed to PNG-encode page image")
    return bytes(buf)


def _to_gray(png_bytes: bytes) -> tuple[bytes, np.ndarray]:
    arr = cv2.imdecode(np.frombuffer(png_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise RuntimeError("failed to decode page image")
    if arr.ndim == 3:
        channels = arr.shape[2]
        if channels == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2GRAY)
        else:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    return _to_png(arr), arr


def _full_page_image(
    page: pymupdf.Page, coverage: float
) -> tuple[int, pymupdf.Rect] | None:
    """Identify a page that is really just one scanned bitmap.

    Returns (xref, placed_rect) or None. The placed rect matters: scans are
    routinely letterboxed inside a Letter/A4 page, and the *placed* width is
    what determines the scan's true DPI — using the page width understates it.
    """
    images = page.get_images(full=True)
    if len(images) != 1:
        return None
    # A real text layer means this is not a pure scan; do not treat it as one.
    if len(page.get_text().strip()) > 20:
        return None
    xref = images[0][0]
    rects = page.get_image_rects(xref)
    if not rects:
        return None
    page_area = page.rect.get_area()
    covered = sum(r.get_area() for r in rects)
    if page_area <= 0 or covered / page_area < coverage:
        return None
    return xref, max(rects, key=lambda r: r.get_area())


class IngestVerifier(Verifier[IngestResult]):
    name = "ingest"

    def verify(self, doc: Document, output: IngestResult, params: IngestParams) -> Verdict:
        findings: list[str] = []
        metrics: dict[str, float] = {}

        if not output.pages:
            return Verdict.fail("no pages extracted")

        dpis = [p.source_dpi for p in output.pages if p.source_dpi]
        if dpis:
            metrics["min_source_dpi"] = min(dpis)
            metrics["mean_source_dpi"] = sum(dpis) / len(dpis)
            if min(dpis) < params.cfg.low_dpi_warning:
                findings.append(
                    f"source resolution is low (min {min(dpis):.0f} DPI); nastaliq dot "
                    f"placement may be unrecoverable — higher-resolution scans would help"
                )

        blank = 0
        for p in output.pages:
            arr = cv2.imdecode(np.frombuffer(p.gray, np.uint8), cv2.IMREAD_GRAYSCALE)
            ink = (
                float((arr < params.cfg.ink_level).mean()) if arr is not None else 0.0
            )
            if ink < params.cfg.blank_page_ink_ratio:
                blank += 1
        metrics["blank_pages"] = float(blank)
        if blank:
            findings.append(f"{blank} page(s) appear blank")

        metrics["pages"] = float(len(output.pages))
        metrics["native_extraction_ratio"] = sum(p.native for p in output.pages) / len(
            output.pages
        )
        # Low resolution is worth reporting but is a property of the source, not
        # a failure of this stage: it cannot be repaired by re-running.
        return Verdict(passed=True, score=None, findings=findings, metrics=metrics)


class IngestStage(Stage[IngestParams, IngestResult]):
    name = "s0_ingest"
    version = "1"
    params_model = IngestParams
    verifier = IngestVerifier()

    def __init__(self, pdf_path: Path | str) -> None:
        self.pdf_path = Path(pdf_path)

    def run(self, doc: Document, params: IngestParams) -> IngestResult:
        src_sha = sha256_file(self.pdf_path)
        pages: list[PageImage] = []

        with pymupdf.open(self.pdf_path) as pdf:
            count = pdf.page_count
            if params.max_pages is not None:
                count = min(count, params.max_pages)

            for idx in range(count):
                page = pdf[idx]
                found = _full_page_image(page, params.native_image_coverage)

                if found is not None:
                    xref, rect = found
                    info = pdf.extract_image(xref)
                    raw = info["image"]
                    width, height = int(info["width"]), int(info["height"])
                    # Effective DPI of the *source* pixels, measured against the
                    # width the image actually occupies — not the page width.
                    placed_w_in = rect.width / 72.0
                    dpi = width / placed_w_in if placed_w_in > 0 else None
                    native = True
                    if info["ext"].lower() != "png":
                        arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
                        raw = _to_png(arr)
                else:
                    pix = page.get_pixmap(dpi=int(params.target_dpi), colorspace=pymupdf.csGRAY)
                    raw = pix.tobytes("png")
                    width, height = pix.width, pix.height
                    dpi = float(params.target_dpi)
                    native = False

                gray_png, _ = _to_gray(raw)
                pages.append(
                    PageImage(
                        page_no=idx,
                        width_px=width,
                        height_px=height,
                        source_dpi=dpi,
                        native=native,
                        png=raw,
                        gray=gray_png,
                    )
                )

        return IngestResult(
            source_path=str(self.pdf_path), source_sha256=src_sha, pages=pages
        )

    def apply(self, doc: Document, output: IngestResult) -> Document:
        raise NotImplementedError("use ingest_document(); S0 creates rather than mutates")

    def counts(self, output: IngestResult) -> dict[str, int]:
        return {
            "pages": len(output.pages),
            "native": sum(1 for p in output.pages if p.native),
        }


def ingest_document(
    pdf_path: Path | str, store: Store, params: IngestParams | None = None
) -> tuple[Document, object]:
    """Run S0 and persist the resulting Document. Returns (doc, report)."""
    params = params or IngestParams()
    stage = IngestStage(pdf_path)
    started, t0 = datetime.now(UTC), time.perf_counter()

    result = stage.run(Document(doc_id="", source_path="", source_sha256=""), params)
    doc_id = result.source_sha256[:12]

    pages: list[Page] = []
    for pi in result.pages:
        raw_ref = ArtifactRef(
            sha256=store.put_blob(pi.png),
            media_type="image/png",
            width=pi.width_px,
            height=pi.height_px,
            note="native" if pi.native else "rasterized",
        )
        gray_ref = ArtifactRef(
            sha256=store.put_blob(pi.gray),
            media_type="image/png",
            width=pi.width_px,
            height=pi.height_px,
        )
        pages.append(
            Page(
                page_no=pi.page_no,
                width_px=pi.width_px,
                height_px=pi.height_px,
                source_dpi=pi.source_dpi,
                views={ViewName.RAW: raw_ref, ViewName.GRAY: gray_ref},
            )
        )

    doc = Document(
        doc_id=doc_id,
        source_path=result.source_path,
        source_sha256=result.source_sha256,
        pages=pages,
    )

    verdict = stage.verifier.verify(doc, result, params) if stage.verifier else Verdict()
    elapsed = time.perf_counter() - t0

    doc.history.append(
        HistoryEntry(
            stage=stage.name,
            version=stage.version,
            params_hash=params.hash(),
            passed=verdict.passed,
            note="; ".join(verdict.findings) or None,
        )
    )
    state_sha = store.save_document(doc)

    report = StageReport(
        stage=stage.name,
        version=stage.version,
        params_hash=params.hash(),
        status="ok" if verdict.passed else "degraded",
        verdict=verdict,
        counts=stage.counts(result),
        cost=Cost(wall_seconds=elapsed),
    )
    store.record_run(
        doc_id=doc.doc_id,
        stage=stage.name,
        stage_version=stage.version,
        params_hash=params.hash(),
        input_hash=result.source_sha256,
        output_hash=state_sha,
        status=report.status,
        passed=verdict.passed,
        score=verdict.score,
        report=report.model_dump(mode="json"),
        started_at=started,
        duration_s=elapsed,
    )
    return doc, report


def load_view(store: Store, page: Page, view: ViewName = ViewName.GRAY) -> bytes:
    ref = page.views.get(view) or page.views.get(ViewName.RAW)
    if ref is None:
        raise KeyError(f"page {page.page_no} has no {view} view")
    return store.get_blob(ref.sha256)


def load_view_array(store: Store, page: Page, view: ViewName = ViewName.GRAY) -> np.ndarray:
    data = load_view(store, page, view)
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise RuntimeError(f"could not decode {view} view of page {page.page_no}")
    return arr

