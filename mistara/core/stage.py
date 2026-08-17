"""Stage contract: a PROCESSOR that does the work, and a VERIFIER that judges it.

The split matters. The PRD's "eyes and ears" requirement is not about model-
internal confidence — it is about being able to answer, after a step runs:
*did that work? did it miss anything? should we run it again?* That is a
verification job, and it is a separate, separately-testable component.

A verifier may be pure code (structural checks, geometry, counts), a measurement
against another implementation, or an LLM looking at the crop beside the output.
Not every stage can produce a meaningful number, so `Verdict.score` is optional —
`passed` plus human-readable `findings` is the floor.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from mistara.core.model import Document, HistoryEntry
from mistara.core.store import Store, stable_hash


class StageParams(BaseModel):
    """Base for a stage's typed, hashable parameters."""

    model_config = ConfigDict(extra="forbid")

    def hash(self) -> str:
        return stable_hash(self.model_dump(mode="json"))


class Cost(BaseModel):
    provider_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    wall_seconds: float = 0.0

    def __add__(self, other: Cost) -> Cost:
        return Cost(
            provider_calls=self.provider_calls + other.provider_calls,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            wall_seconds=self.wall_seconds + other.wall_seconds,
        )


class Verdict(BaseModel):
    """The verifier's judgement on one stage's output."""

    passed: bool = True
    score: float | None = None
    findings: list[str] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)

    @classmethod
    def ok(cls, **metrics: float) -> Verdict:
        return cls(passed=True, metrics=metrics)

    @classmethod
    def fail(cls, *findings: str, **metrics: float) -> Verdict:
        return cls(passed=False, findings=list(findings), metrics=metrics)


class StageReport(BaseModel):
    stage: str
    version: str
    params_hash: str
    status: str = "ok"  # ok | degraded | failed
    verdict: Verdict = Field(default_factory=Verdict)
    counts: dict[str, int] = Field(default_factory=dict)
    cost: Cost = Field(default_factory=Cost)
    warnings: list[str] = Field(default_factory=list)
    cached: bool = False

    @property
    def summary(self) -> str:
        bits = [f"{self.stage} {self.status}"]
        if self.counts:
            bits.append(" ".join(f"{k}={v}" for k, v in self.counts.items()))
        if self.verdict.score is not None:
            bits.append(f"score={self.verdict.score:.3f}")
        if not self.verdict.passed:
            bits.append("VERIFY-FAILED")
        return " | ".join(bits)


class Verifier[O: BaseModel](ABC):
    """Judges a stage's output. Pure code or model-backed; both are fine."""

    name: str = "verifier"

    @abstractmethod
    def verify(self, doc: Document, output: O, params: BaseModel) -> Verdict: ...


class Stage[P: StageParams, O: BaseModel](ABC):
    """A processor plus an optional verifier.

    `run()` produces the typed output; `apply()` merges it into the Document.
    Keeping those separate is what lets a stage be tested without constructing a
    whole Document, while still leaving one canonical accumulating state.
    """

    name: str
    version: str
    params_model: type[P]
    verifier: Verifier[O] | None = None

    @abstractmethod
    def run(self, doc: Document, params: P) -> O: ...

    @abstractmethod
    def apply(self, doc: Document, output: O) -> Document: ...

    def input_hash(self, doc: Document, params: P) -> str:
        """What this stage's result depends on. Override to narrow the key."""
        return stable_hash(
            {"doc": doc.source_sha256, "pages": len(doc.pages), "params": params.hash()}
        )

    # -- execution ---------------------------------------------------------- #

    def cost_of(self, output: O) -> Cost | None:
        """Provider spend for this run, if the stage tracked any.

        Override where the output carries it. Without this the ledger records
        zeros for every model-backed stage, which silently destroys the only
        record of what a run actually cost.
        """
        return getattr(output, "cost", None)

    def execute(
        self, doc: Document, params: P, store: Store
    ) -> tuple[Document, StageReport]:
        started = datetime.now(UTC)
        t0 = time.perf_counter()

        output = self.run(doc, params)
        verdict = (
            self.verifier.verify(doc, output, params) if self.verifier else Verdict()
        )
        new_doc = self.apply(doc, output)

        elapsed = time.perf_counter() - t0
        cost = self.cost_of(output) or Cost()
        cost.wall_seconds = elapsed
        report = StageReport(
            stage=self.name,
            version=self.version,
            params_hash=params.hash(),
            status="ok" if verdict.passed else "degraded",
            verdict=verdict,
            counts=self.counts(output),
            cost=cost,
        )

        new_doc.history.append(
            HistoryEntry(
                stage=self.name,
                version=self.version,
                params_hash=params.hash(),
                passed=verdict.passed,
                note="; ".join(verdict.findings) or None,
            )
        )
        state_sha = store.save_document(new_doc)
        store.record_run(
            doc_id=new_doc.doc_id,
            stage=self.name,
            stage_version=self.version,
            params_hash=params.hash(),
            input_hash=self.input_hash(doc, params),
            output_hash=state_sha,
            status=report.status,
            passed=verdict.passed,
            score=verdict.score,
            report=report.model_dump(mode="json"),
            started_at=started,
            duration_s=elapsed,
        )
        return new_doc, report

    def counts(self, output: O) -> dict[str, int]:  # noqa: ARG002
        return {}


def params_from[P: StageParams](model: type[P], overrides: dict[str, Any] | None) -> P:
    return model.model_validate(overrides or {})
