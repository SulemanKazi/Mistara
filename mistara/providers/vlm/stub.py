"""Offline stub provider.

Exists so the whole pipeline — ingest, extract, verify, render — can be run and
tested with no API key and no network. It performs real line segmentation via
horizontal projection, so the geometry is genuine; only the text is synthetic.
That makes it useful for exercising the spatial output and the verifiers.
"""

from __future__ import annotations

import cv2
import numpy as np

from mistara.providers.vlm.base import (
    LineRead,
    TranscriptionRequest,
    TranscriptionResult,
    VLMClient,
)

#: Recognisable filler so stub output is never mistaken for a real transcription.
_FILLER = "نمونہ متن"


class StubVLM(VLMClient):
    model_id = "stub:projection"

    def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult:
        if request.line_boxes:
            lines = [
                LineRead(line_id=lid, text=f"{_FILLER} {i + 1}")
                for i, lid in enumerate(request.line_boxes)
            ]
        else:
            arr = cv2.imdecode(
                np.frombuffer(request.image_png, np.uint8), cv2.IMREAD_GRAYSCALE
            )
            rows = _text_rows(arr) if arr is not None else []
            lines = [
                LineRead(line_id=f"l{i:04d}", text=f"{_FILLER} {i + 1}")
                for i in range(len(rows))
            ]
        return TranscriptionResult(
            lines=lines, raw_response="<stub>", model_id=self.model_id
        )

    def parse_response(
        self, raw: str, request: TranscriptionRequest
    ) -> TranscriptionResult:
        return self.transcribe(request)


def _text_rows(gray: np.ndarray, min_gap: int = 3) -> list[tuple[int, int]]:
    """Segment text rows by horizontal projection of ink.

    Shared with the classical-CV layout path in S2; kept simple here on purpose.
    """
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10
    )
    profile = (binary > 0).sum(axis=1).astype(float)
    if profile.max() == 0:
        return []
    threshold = max(1.0, profile.mean() * 0.35)
    rows: list[tuple[int, int]] = []
    start: int | None = None
    gap = 0
    for y, value in enumerate(profile):
        if value >= threshold:
            if start is None:
                start = y
            gap = 0
        elif start is not None:
            gap += 1
            if gap >= min_gap:
                rows.append((start, y - gap))
                start = None
    if start is not None:
        rows.append((start, len(profile) - 1))
    return [(a, b) for a, b in rows if b - a >= 4]
