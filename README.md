# Mistara

Layout-preserving OCR and verification pipeline for printed Urdu books containing
interspersed Arabic (Quran and Hadith).

Mistara does not just read pages. It reads them *and tells you whether the read
worked* — every stage pairs a processor with a verifier, and defects come back
located on the page rather than as a bare count. Sacred text is not recognized
but **verified**: Arabic OCR output is treated as a noisy search query against an
authenticated corpus, and the corpus text is what lands in the output. Extracted
text keeps the position its ink occupied, so the output is spatial rather than a
flat dump.

| Doc | What it covers |
|---|---|
| [`docs/prd.md`](docs/prd.md) | requirements — what and why |
| [`docs/design.md`](docs/design.md) | architecture — how |
| [`docs/progress.md`](docs/progress.md) | **what is actually built, and what is thin** |

## Setup

```bash
uv sync                 # Python 3.12; no CUDA, no model downloads
cp .env.example .env    # then add your ANTHROPIC_API_KEY
uv run mistara providers   # confirm it is picked up (never prints the value)
```

`.env` is gitignored. Real environment variables take precedence over it, so a
one-off override works without editing anything:
`ANTHROPIC_API_KEY=sk-other uv run mistara extract ...`

## Usage

Everything runs offline with the `stub` provider — no API key needed to exercise
the full pipeline.

```bash
# S0 — PDF to page images. Extracts embedded scans natively rather than
# upsampling them, and reports the true source resolution.
uv run mistara ingest data/sample_pdf_1.pdf

# S4 — transcribe, anchored to detected line and cell geometry
uv run mistara extract <doc_id> --provider stub
uv run mistara extract <doc_id> --provider anthropic:claude-opus-5

# Render the spatial view — the debug tool and the viewer are one artifact.
# Toggle Image / Overlay / Text-only, line boxes, and the issue layer.
uv run mistara render <doc_id> -o out/view.html --text out/view.txt

# Tuning — every threshold is per-book configurable
uv run mistara config                            # effective profile
uv run mistara config -c config/mybook.toml      # with overrides
uv run mistara extract <doc_id> -c config/mybook.toml

# Inspect
uv run mistara ls
uv run mistara show <doc_id> --json     # state + run ledger, machine-readable
uv run mistara text <doc_id> --page 0
```

Doc ids accept prefixes (`938f` works). The artifact store lives in `data/store/`
and is safe to delete — `ingest` rebuilds it.

Every stage is exactly one command with `--json` output. That is deliberate: the
orchestrator is Claude Code or a human driving this CLI, so anything an agent does
is reproducible by hand, and when a programmatic agent is worth building its tool
surface is a thin wrapper over commands that already exist.

## Layout

```
mistara/
├── core/      document model, artifact store, stage protocol, config
├── text/      Unicode canonicalization, script signals, CER/WER
├── stages/    s0_ingest, segment (lines + columns + assessment), s4_extract
├── providers/ VLM clients behind a pluggable interface (stub, anthropic)
├── render/    spatial HTML output with located issues
└── cli.py     one command per stage

config/default.toml   documented defaults; copy per book and override a few keys
```

Every detection threshold — binarization, line detection, merge/miss detection,
column gutters, text-quality checks — lives in one config profile rather than
scattered through the source. They were all fitted to one book at one scan
quality, so a different print run will want different values.

## Development

```bash
uv run pytest        # 107 tests
uv run ruff check .
```
