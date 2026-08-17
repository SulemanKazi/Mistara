"""Command line interface.

Every stage is exactly one command, with JSON available on demand. This is a
deliberate constraint: the orchestrator agent, when it arrives, gets a tool
surface that is a thin wrapper over these commands — so anything the agent does,
a human can reproduce by hand, and anything a human can do, the agent can too.
"""

from __future__ import annotations

import json as jsonlib
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from mistara.core.config import MistaraConfig
from mistara.core.env import credential_status, load_dotenv
from mistara.core.store import DEFAULT_STORE, Store
from mistara.render.html import document_summary, write_document, write_text_dump
from mistara.render.reader import write_reader
from mistara.stages.s0_ingest import IngestParams, ingest_document
from mistara.stages.s3_order import (
    COLUMN_MAJOR,
    ROW_MAJOR,
    OrderParams,
    OrderStage,
)
from mistara.stages.s4_extract import ExtractParams, ExtractStage
from mistara.stages.s5_route import RouteParams, RouteStage
from mistara.stages.s6a_match import MatchParams, MatchStage

#: Loaded once, before any command runs, so provider clients see the keys.
#: Real environment variables win over the file.
DOTENV_PATH = load_dotenv()

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Layout-preserving OCR and verification for Urdu/Arabic books.",
)
console = Console()

StoreOpt = Annotated[Path, typer.Option("--store", help="Artifact store directory.")]
JsonOpt = Annotated[bool, typer.Option("--json", help="Emit machine-readable JSON.")]
ConfigOpt = Annotated[
    Path | None,
    typer.Option(
        "--config",
        "-c",
        help="Tuning profile (TOML). Bare names resolve inside config/.",
    ),
]


def _config(path: Path | None) -> MistaraConfig:
    try:
        return MistaraConfig.load(path)
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc


def _open(store: Path) -> Store:
    return Store(store)


def _report_panel(report, json_out: bool) -> None:
    if json_out:
        console.print_json(jsonlib.dumps(report.model_dump(mode="json")))
        return
    style = "green" if report.verdict.passed else "yellow"
    console.print(f"[{style}]{report.summary}[/{style}]")
    for metric, value in sorted(report.verdict.metrics.items()):
        console.print(f"  [dim]{metric}[/dim] {value:.4g}")
    for finding in report.verdict.findings:
        console.print(f"  [yellow]![/yellow] {finding}")


@app.command()
def ingest(
    pdf: Annotated[Path, typer.Argument(help="PDF to ingest.")],
    store: StoreOpt = DEFAULT_STORE,
    config: ConfigOpt = None,
    max_pages: Annotated[int | None, typer.Option(help="Ingest only the first N pages.")] = None,
    json: JsonOpt = False,
) -> None:
    """S0 — rasterize or natively extract page images into the store."""
    if not pdf.exists():
        raise typer.BadParameter(f"no such file: {pdf}")
    st = _open(store)
    cfg = _config(config)
    doc, report = ingest_document(
        pdf, st, IngestParams(max_pages=max_pages, cfg=cfg.ingest)
    )
    if not json:
        console.print(f"[bold]{doc.doc_id}[/bold]  {len(doc.pages)} pages")
    _report_panel(report, json)


@app.command()
def extract(
    doc_id: Annotated[str, typer.Argument(help="Document id (a prefix is fine).")],
    store: StoreOpt = DEFAULT_STORE,
    config: ConfigOpt = None,
    provider: Annotated[
        str, typer.Option(help="e.g. gemini:gemini-3.7-flash, anthropic:claude-opus-5, stub")
    ] = "gemini:gemini-3.7-flash",
    pages: Annotated[str | None, typer.Option(help="Comma-separated page numbers.")] = None,
    no_anchor: Annotated[bool, typer.Option("--no-anchor", help="Free transcription.")] = False,
    json: JsonOpt = False,
) -> None:
    """S4 — transcribe pages, anchored to locally detected line geometry."""
    st = _open(store)
    doc = st.load_document(st.resolve_doc_id(doc_id))
    page_list = [int(p) for p in pages.split(",")] if pages else None
    stage = ExtractStage(st)
    cfg = _config(config)
    params = ExtractParams(
        provider=provider,
        pages=page_list,
        line_anchored=not no_anchor,
        cfg=cfg.extract,
        segment=cfg.segment,
    )
    doc, report = stage.execute(doc, params, st)
    _report_panel(report, json)


@app.command()
def order(
    doc_id: Annotated[str, typer.Argument(help="Document id (a prefix is fine).")],
    store: StoreOpt = DEFAULT_STORE,
    config: ConfigOpt = None,
    read_order: Annotated[
        str | None,
        typer.Option(
            "--read-order",
            help="column_major (parallel text) or row_major (table). Overrides config.",
        ),
    ] = None,
    json: JsonOpt = False,
) -> None:
    """S3 — group cells into regions and put them in reading order.

    Pure geometry over boxes already in the document: no image, no model, no
    provider call. Run it after `extract`.
    """
    st = _open(store)
    doc = st.load_document(st.resolve_doc_id(doc_id))
    cfg = _config(config)
    order_cfg = cfg.order
    if read_order is not None:
        if read_order not in (COLUMN_MAJOR, ROW_MAJOR):
            raise typer.BadParameter(
                f"--read-order must be {COLUMN_MAJOR} or {ROW_MAJOR}"
            )
        order_cfg = order_cfg.model_copy(update={"column_read_order": read_order})
    params = OrderParams(cfg=order_cfg, segment=cfg.segment)
    doc, report = OrderStage().execute(doc, params, st)
    _report_panel(report, json)


corpus_app = typer.Typer(no_args_is_help=True, help="Authenticated corpora.")
app.add_typer(corpus_app, name="corpus")


@corpus_app.command("fetch")
def corpus_fetch(
    dest: Annotated[Path, typer.Option("--dest", help="Corpus directory.")] = Path(
        "data/corpora"
    ),
) -> None:
    """Download the Quran text. Run once; no stage reaches for the network."""
    from mistara.corpus.source import QURAN_ATTRIBUTION, fetch_quran

    try:
        path = fetch_quran(dest)
    except Exception as exc:  # network, or a corpus that fails validation
        raise typer.BadParameter(f"could not fetch corpus: {exc}") from exc
    console.print(f"wrote [bold]{path}[/bold]")
    console.print(f"[dim]text:[/dim] {QURAN_ATTRIBUTION['text_source']}")
    console.print(f"[dim]licence:[/dim] {QURAN_ATTRIBUTION['license']}")


@corpus_app.command("info")
def corpus_info(
    corpus: Annotated[Path, typer.Option("--corpus", help="Corpus directory.")] = Path(
        "data/corpora"
    ),
    json: JsonOpt = False,
) -> None:
    """Show what is indexed, and where it came from."""
    from mistara.corpus.quran import load_quran

    try:
        index = load_quran(corpus)
    except FileNotFoundError as exc:
        raise typer.BadParameter(str(exc)) from exc
    info = {
        "name": index.name,
        "verses": len(index.refs),
        "tokens": len(index),
        **{k: v for k, v in index.meta.items() if k != "name"},
    }
    if json:
        console.print_json(jsonlib.dumps(info))
        return
    for key, value in info.items():
        console.print(f"  [dim]{key}[/dim] {value}")


@app.command()
def match(
    doc_id: Annotated[str, typer.Argument(help="Document id (a prefix is fine).")],
    store: StoreOpt = DEFAULT_STORE,
    config: ConfigOpt = None,
    corpus: Annotated[Path, typer.Option("--corpus", help="Corpus directory.")] = Path(
        "data/corpora"
    ),
    replace_inline: Annotated[
        bool,
        typer.Option(
            "--replace-inline",
            help="Also rewrite quotes found inside a line of prose, not just cite them.",
        ),
    ] = False,
    json: JsonOpt = False,
) -> None:
    """S6a — verify Arabic against the corpus; replace or cite what matches.

    Deterministic and free: no provider call. Run it after `order`.
    """
    st = _open(store)
    doc = st.load_document(st.resolve_doc_id(doc_id))
    cfg = _config(config)
    match_cfg = cfg.match
    if replace_inline:
        match_cfg = match_cfg.model_copy(update={"replace_inline": True})
    params = MatchParams(corpus_dir=str(corpus), cfg=match_cfg)
    try:
        doc, report = MatchStage().execute(doc, params, st)
    except FileNotFoundError as exc:
        raise typer.BadParameter(str(exc)) from exc
    _report_panel(report, json)


@app.command()
def route(
    doc_id: Annotated[str, typer.Argument(help="Document id (a prefix is fine).")],
    store: StoreOpt = DEFAULT_STORE,
    config: ConfigOpt = None,
    json: JsonOpt = False,
) -> None:
    """S5 — assign a language to every span from corpus hits and text evidence.

    Deterministic and free: no provider call. Run it after `match`, whose corpus
    hits are the decisive evidence a span is Arabic — orthography alone mis-tags
    Quranic Arabic set in an Urdu typeface.
    """
    st = _open(store)
    doc = st.load_document(st.resolve_doc_id(doc_id))
    cfg = _config(config)
    params = RouteParams(cfg=cfg.route, order=cfg.order)
    doc, report = RouteStage().execute(doc, params, st)
    _report_panel(report, json)


@app.command()
def render(
    doc_id: Annotated[str, typer.Argument(help="Document id (a prefix is fine).")],
    store: StoreOpt = DEFAULT_STORE,
    out: Annotated[Path, typer.Option("-o", "--out", help="Output HTML path.")] = Path(
        "out/view.html"
    ),
    text: Annotated[
        Path | None, typer.Option("--text", help="Also write a plain-text dump.")
    ] = None,
    reader: Annotated[
        Path | None,
        typer.Option(
            "--reader",
            help="Also write the typeset reading view (no scan, unified type).",
        ),
    ] = None,
    boxes: Annotated[bool, typer.Option(help="Draw detected line boxes.")] = True,
    regions: Annotated[
        bool, typer.Option(help="Outline regions and number them in reading order.")
    ] = True,
    embed_fonts: Annotated[
        bool,
        typer.Option(help="Embed Urdu/Arabic fonts in the reading view (portable)."),
    ] = True,
) -> None:
    """Render the spatial view — the debug tool and the viewer are one artifact."""
    st = _open(store)
    doc = st.load_document(st.resolve_doc_id(doc_id))
    path = write_document(doc, st, out, show_boxes=boxes, show_regions=regions)
    console.print(f"wrote [bold]{path}[/bold]")
    if text:
        console.print(f"wrote [bold]{write_text_dump(doc, text)}[/bold]")
    if reader:
        console.print(
            f"wrote [bold]{write_reader(doc, reader, embed_fonts=embed_fonts)}[/bold]"
        )


@app.command()
def providers(json: JsonOpt = False) -> None:
    """Show which provider credentials are visible. Never prints the values."""
    from mistara.providers.registry import DEFAULT_ROLES

    status = credential_status()
    if json:
        console.print_json(
            jsonlib.dumps(
                {
                    "dotenv": str(DOTENV_PATH) if DOTENV_PATH else None,
                    "credentials": status,
                    "roles": DEFAULT_ROLES,
                }
            )
        )
        return

    console.print(
        f"[dim].env:[/dim] {DOTENV_PATH or 'none found'}"
    )
    table = Table("provider", "credential", "implemented")
    implemented = {"anthropic": "yes", "google": "yes", "openai": "no"}
    for name, state in status.items():
        ok = state != "not set"
        table.add_row(
            name,
            f"[green]{state}[/green]" if ok else f"[dim]{state}[/dim]",
            implemented.get(name, "no"),
        )
    console.print(table)
    console.print("[dim]stub[/dim] always available — no credential needed")
    for role, spec in DEFAULT_ROLES.items():
        console.print(f"  [dim]{role}[/dim] → {spec}")


@app.command("config")
def show_config(
    config: ConfigOpt = None,
    json: JsonOpt = False,
) -> None:
    """Print the effective tuning profile — defaults merged with any overrides."""
    cfg = _config(config)
    if json:
        console.print_json(cfg.model_dump_json())
        return
    # markup=False: Rich would otherwise swallow TOML's [section] headers.
    console.print(cfg.to_toml(), markup=False, highlight=False)


RESULTS_DIR = Path("data/results")


@app.command()
def dump(
    doc_id: Annotated[str, typer.Argument(help="Document id (a prefix is fine).")],
    store: StoreOpt = DEFAULT_STORE,
    out: Annotated[
        Path | None, typer.Option("-o", "--out", help="Defaults to data/results/<id>.json")
    ] = None,
) -> None:
    """Write a document to durable JSON, outside the rebuildable store.

    The store is a cache and can be deleted; model output is expensive and
    cannot be regenerated for free. Dump anything worth keeping — the same file
    is what you correct to build reference data.
    """
    st = _open(store)
    resolved = st.resolve_doc_id(doc_id)
    doc = st.load_document(resolved)
    target = out or RESULTS_DIR / f"{resolved}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(doc.model_dump_json(indent=2), encoding="utf-8")
    cells = sum(len(r.lines) for pg in doc.pages for r in pg.regions)
    console.print(
        f"wrote [bold]{target}[/bold] — {len(doc.pages)} pages, "
        f"{cells} cells, {len(doc.text)} chars"
    )


@app.command("load")
def load_document(
    path: Annotated[Path, typer.Argument(help="A JSON document written by `dump`.")],
    store: StoreOpt = DEFAULT_STORE,
) -> None:
    """Restore a dumped document into the store."""
    from mistara.core.model import Document

    if not path.exists():
        raise typer.BadParameter(f"no such file: {path}")
    doc = Document.model_validate_json(path.read_text(encoding="utf-8"))
    st = _open(store)
    st.save_document(doc)
    console.print(f"loaded [bold]{doc.doc_id}[/bold] from {path}")


@app.command("ls")
def list_docs(store: StoreOpt = DEFAULT_STORE, json: JsonOpt = False) -> None:
    """List documents in the store."""
    docs = _open(store).list_documents()
    if json:
        console.print_json(jsonlib.dumps(docs))
        return
    if not docs:
        console.print("[dim]store is empty — run `mistara ingest <pdf>`[/dim]")
        return
    table = Table("doc_id", "source", "updated")
    for d in docs:
        table.add_row(d["doc_id"], d["source_path"], d["updated_at"][:19])
    console.print(table)


@app.command()
def show(
    doc_id: Annotated[str, typer.Argument()],
    store: StoreOpt = DEFAULT_STORE,
    json: JsonOpt = False,
) -> None:
    """Summarize a document and the stages that have run against it."""
    st = _open(store)
    resolved = st.resolve_doc_id(doc_id)
    doc = st.load_document(resolved)
    summary = document_summary(doc)
    if json:
        console.print_json(jsonlib.dumps({**summary, "runs": st.runs_for(resolved)}))
        return

    console.print(f"[bold]{doc.doc_id}[/bold]  {doc.source_path}")
    console.print(
        f"{summary['pages']} pages · {summary['lines']} lines · {summary['chars']} chars"
    )
    table = Table("page", "size", "dpi", "regions", "lines")
    for p in doc.pages:
        table.add_row(
            str(p.page_no),
            f"{p.width_px}×{p.height_px}",
            f"{p.source_dpi:.0f}" if p.source_dpi else "-",
            str(len(p.regions)),
            str(sum(len(r.lines) for r in p.regions)),
        )
    console.print(table)

    runs = st.runs_for(resolved)
    if runs:
        rt = Table("run", "stage", "status", "passed", "score", "secs")
        for r in runs:
            rt.add_row(
                str(r["run_id"]),
                r["stage"],
                r["status"],
                "-" if r["passed"] is None else ("yes" if r["passed"] else "NO"),
                f"{r['score']:.3f}" if r["score"] is not None else "-",
                f"{r['duration_s']:.2f}",
            )
        console.print(rt)


@app.command()
def text(
    doc_id: Annotated[str, typer.Argument()],
    store: StoreOpt = DEFAULT_STORE,
    page: Annotated[int | None, typer.Option(help="Only this page.")] = None,
) -> None:
    """Print the transcription in reading order."""
    st = _open(store)
    doc = st.load_document(st.resolve_doc_id(doc_id))
    console.print(doc.page(page).text if page is not None else doc.text)


if __name__ == "__main__":
    app()
