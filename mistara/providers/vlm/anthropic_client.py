"""Anthropic vision provider."""

from __future__ import annotations

import base64
import os

from mistara.providers.vlm.base import (
    LINE_ANCHORED_INSTRUCTION,
    TRANSCRIBE_SYSTEM,
    WHOLE_PAGE_INSTRUCTION,
    TranscriptionRequest,
    TranscriptionResult,
    VLMClient,
    parse_line_json,
    parse_plain,
)

DEFAULT_MODEL = "claude-opus-5"


class AnthropicVLM(VLMClient):
    def __init__(self, model: str = DEFAULT_MODEL, *, max_tokens: int = 16000) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.model_id = f"anthropic:{model}"
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            import anthropic

            if not (
                os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
            ):
                # A profile from `ant auth login` also works, so this is a hint
                # rather than a hard failure.
                pass
            self._client = anthropic.Anthropic()
        return self._client

    @property
    def supports_temperature(self) -> bool:
        # Sampling parameters were removed on the Opus 5 / Sonnet 5 family;
        # sending `temperature` returns a 400.
        return False

    def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult:
        client = self._ensure_client()
        b64 = base64.standard_b64encode(request.image_png).decode("ascii")

        if request.line_boxes:
            regions = "\n".join(
                f"  {lid}  x={b.x:.0f} y={b.y:.0f} w={b.w:.0f} h={b.h:.0f}"
                for lid, b in request.line_boxes.items()
            )
            instruction = LINE_ANCHORED_INSTRUCTION.substitute(
                width=request.image_width or 0,
                height=request.image_height or 0,
                regions=regions,
            )
        else:
            instruction = WHOLE_PAGE_INSTRUCTION
        if request.hint:
            instruction = f"{instruction}\n\nContext: {request.hint}"

        message = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=TRANSCRIBE_SYSTEM,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": b64,
                            },
                        },
                        {"type": "text", "text": instruction},
                    ],
                }
            ],
        )

        text = "".join(b.text for b in message.content if b.type == "text")
        result = self.parse_response(text, request)
        result.input_tokens = message.usage.input_tokens
        result.output_tokens = message.usage.output_tokens
        return result

    def parse_response(
        self, raw: str, request: TranscriptionRequest
    ) -> TranscriptionResult:
        lines = (
            parse_line_json(raw, list(request.line_boxes))
            if request.line_boxes
            else parse_plain(raw)
        )
        return TranscriptionResult(
            lines=lines, raw_response=raw, model_id=self.model_id
        )
