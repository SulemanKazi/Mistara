"""Provider-agnostic vision interface.

Everything with model weights sits behind this. Swapping providers — or moving
from a hosted frontier API to a self-hosted open-weight model — is a config
change, not a code change, which is what makes the bake-off cheap to run.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from string import Template

from pydantic import BaseModel, Field

from mistara.core.model import BBox


class LineRead(BaseModel):
    """One transcribed line, anchored to the line box it came from."""

    line_id: str
    text: str
    illegible: bool = False


class TranscriptionRequest(BaseModel):
    image_png: bytes
    #: Line boxes to anchor against. Empty means whole-image free transcription.
    line_boxes: dict[str, BBox] = Field(default_factory=dict)
    #: Pixel dimensions, so box coordinates in the prompt are interpretable.
    image_width: int | None = None
    image_height: int | None = None
    hint: str | None = None

    model_config = {"arbitrary_types_allowed": True}


class TranscriptionResult(BaseModel):
    lines: list[LineRead]
    raw_response: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    model_id: str = ""

    @property
    def text(self) -> str:
        return "\n".join(ln.text for ln in self.lines)


class VLMClient(ABC):
    """A vision model that can transcribe a page or region image."""

    #: Stable identifier recorded against every read, e.g. "anthropic:claude-opus-5".
    model_id: str

    @abstractmethod
    def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult: ...

    def parse_response(
        self, raw: str, request: TranscriptionRequest
    ) -> TranscriptionResult:
        """Re-interpret a cached raw response without calling the provider.

        Parsing is where we are most likely to change our minds; the model call
        is where the money is. Keeping them separable means a parser fix costs
        nothing to re-apply to work already paid for.
        """
        raise NotImplementedError

    @property
    def supports_temperature(self) -> bool:
        """Whether k-sample self-consistency is available on this provider.

        False for Claude Opus 5 / Sonnet 5, where sampling parameters were
        removed and sending `temperature` is a 400. Agreement signals therefore
        default to input perturbation rather than temperature sampling.
        """
        return False


#: The prompt is deliberately blunt about verbatim transcription. VLMs have
#: memorized the Quran and will happily emit the canonical verse regardless of
#: what is actually printed — useful for us downstream (we replace with corpus
#: text anyway) but fatal if it leaks into Urdu prose or is mistaken for
#: evidence of what the page says.
TRANSCRIBE_SYSTEM = """\
You are a careful OCR transcriber for printed Urdu and Arabic books.

Transcribe EXACTLY what is printed on the page image. Rules:

1. Verbatim only. Do not correct spelling, grammar, or orthography. Do not \
normalize or modernize the text. Do not complete a partially visible word from \
your own knowledge.
2. This applies with particular force to Quranic and Hadith Arabic. You may \
recognize a verse — transcribe the ink that is actually there, including any \
differences from the canonical wording. Never substitute a remembered verse for \
what is printed.
3. Mark unreadable characters with the replacement character �. Never guess.
4. Preserve the reading order of the page: Urdu and Arabic run right to left.
5. Output plain Unicode text. No transliteration, no translation, no commentary, \
no markdown fences.
"""

#: `string.Template` rather than `str.format`: the body contains a literal JSON
#: example, and `{"lines"` would be parsed as a format field.
LINE_ANCHORED_INSTRUCTION = Template("""\
The page image is $width x $height pixels. It has been divided into regions,
each listed below with its pixel box: x,y is the top-left corner, w x h the size.
Note that a row may be split into several side-by-side regions — a narrow region
beside a wide one is usually a margin note beside body text, and the two must NOT
be merged.

Return a JSON object of the form
{"lines": [{"line_id": "<id>", "text": "<verbatim text>"}, ...]}

Transcribe ONLY the text inside each region's box, and return exactly one entry
for EVERY id below, even if a region is blank or unreadable (use an empty string
or the replacement character as appropriate). Match text to the region it
physically occupies — do not simply return the page in reading order. A narrow
box holds few characters; a wide box holds many.

Regions:
$regions
""")

WHOLE_PAGE_INSTRUCTION = """\
Transcribe the entire page. Put each printed line of text on its own output line, \
in reading order. Return only the transcription.
"""


# --- Response parsing -------------------------------------------------------
# Provider-agnostic: the wire format we ask every VLM for is the same (a JSON
# object of line_id/text pairs when anchored, plain lines otherwise), so the
# parsing lives here and each client shares it rather than re-implementing it.


def strip_fences(text: str) -> str:
    return re.sub(r"^\s*```[a-zA-Z]*\s*|\s*```\s*$", "", text.strip())


def parse_plain(text: str) -> list[LineRead]:
    stripped = strip_fences(text)
    return [
        LineRead(line_id=f"l{i:04d}", text=ln)
        for i, ln in enumerate(stripped.split("\n"))
        if ln.strip()
    ]


def parse_line_json(text: str, expected_ids: list[str]) -> list[LineRead]:
    """Parse the line-anchored response, tolerating a missing or partial JSON body.

    Falls back to plain-text parsing rather than raising: a malformed response is
    still evidence, and the verifier is what decides whether it is good enough.
    """
    body = strip_fences(text)
    try:
        start, end = body.index("{"), body.rindex("}") + 1
        data = json.loads(body[start:end])
        got = {
            str(item["line_id"]): str(item.get("text", ""))
            for item in data.get("lines", [])
            if isinstance(item, dict) and "line_id" in item
        }
        # Preserve the order we asked for; missing ids become empty reads, which
        # is what makes silent omission measurable downstream.
        return [LineRead(line_id=i, text=got.get(i, "")) for i in expected_ids]
    except (ValueError, KeyError, TypeError):
        plain = parse_plain(body)
        return [
            LineRead(
                line_id=expected_ids[i] if i < len(expected_ids) else f"x{i:04d}",
                text=p.text,
            )
            for i, p in enumerate(plain)
        ]
