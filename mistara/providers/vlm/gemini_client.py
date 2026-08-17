"""Google Gemini vision provider.

Cheaper per token than the Claude Opus family by an order of magnitude, and
Gemini's Flash tier is a strong nastaliq/Arabic reader — which is why it is the
default `extract.primary`. The role indirection means that choice is a registry
line, not a stage change; swap it back with `--provider anthropic:...`.

The same verbatim-transcription caveat as every VLM applies: Gemini has the
Quran memorized and will emit canonical verses, so its Arabic output is never
evidence of what the page says — see `TRANSCRIBE_SYSTEM`.
"""

from __future__ import annotations

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

DEFAULT_MODEL = "gemini-3.7-flash"


class GeminiVLM(VLMClient):
    def __init__(self, model: str = DEFAULT_MODEL, *, max_tokens: int = 16000) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.model_id = f"gemini:{model}"
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from google import genai
            from google.genai import types

            # The SDK reads GOOGLE_API_KEY / GEMINI_API_KEY from the environment;
            # a missing key surfaces as a clear auth error on the first call, so
            # we do not pre-check it here (mirrors the Anthropic client).
            #
            # Transport-level backoff so a transient 429/503 ("high demand" on a
            # freshly released model is common) does not abort a whole multi-page
            # run — the Anthropic SDK retries by default and we match that.
            self._client = genai.Client(
                http_options=types.HttpOptions(
                    retry_options=types.HttpRetryOptions(
                        attempts=6,
                        initial_delay=2.0,
                        max_delay=60.0,
                        http_status_codes=[408, 429, 500, 502, 503, 504],
                    )
                )
            )
        return self._client

    @property
    def supports_temperature(self) -> bool:
        # Gemini accepts sampling params, so k-sample self-consistency is
        # available on this provider (unlike the Opus 5 / Sonnet 5 family).
        return True

    def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult:
        from google.genai import types

        client = self._ensure_client()

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
            # Anchored reads ask for a JSON object; constraining the response
            # mime type makes the parse robust without relying on fence stripping.
            response_mime_type = "application/json"
        else:
            instruction = WHOLE_PAGE_INSTRUCTION
            response_mime_type = None
        if request.hint:
            instruction = f"{instruction}\n\nContext: {request.hint}"

        response = client.models.generate_content(
            model=self.model,
            contents=[
                types.Part.from_bytes(
                    data=request.image_png, mime_type="image/png"
                ),
                instruction,
            ],
            config=types.GenerateContentConfig(
                system_instruction=TRANSCRIBE_SYSTEM,
                max_output_tokens=self.max_tokens,
                response_mime_type=response_mime_type,
            ),
        )

        text = response.text or ""
        result = self.parse_response(text, request)
        usage = getattr(response, "usage_metadata", None)
        if usage is not None:
            result.input_tokens = usage.prompt_token_count or 0
            # Billed output includes thinking tokens on the Gemini 3 family.
            result.output_tokens = (usage.candidates_token_count or 0) + (
                getattr(usage, "thoughts_token_count", 0) or 0
            )
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
