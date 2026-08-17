"""Maps logical model roles to concrete providers.

Stages ask for a *role* (`extract.primary`, `extract.secondary`), never a vendor.
The Phase 0 bake-off therefore ends in a config change rather than a code change,
and cross-model agreement just means asking for two different roles.
"""

from __future__ import annotations

from mistara.providers.vlm.base import VLMClient

#: Role -> provider spec. Override per run with `--provider`.
#: Gemini Flash is the default extractor: ~an order of magnitude cheaper per
#: token than Claude Opus and a strong nastaliq/Arabic reader. Claude backs the
#: secondary role so cross-model agreement stays available.
DEFAULT_ROLES: dict[str, str] = {
    "extract.primary": "gemini:gemini-3.7-flash",
    "extract.secondary": "anthropic:claude-opus-5",
    "judge": "gemini:gemini-3.7-flash",
}


def get_vlm(spec: str) -> VLMClient:
    """Build a client from a spec like ``gemini:gemini-3.7-flash`` or ``stub``."""
    provider, _, model = spec.partition(":")
    provider = provider.strip().lower()

    if provider == "stub":
        from mistara.providers.vlm.stub import StubVLM

        return StubVLM()

    if provider == "anthropic":
        from mistara.providers.vlm.anthropic_client import DEFAULT_MODEL, AnthropicVLM

        return AnthropicVLM(model or DEFAULT_MODEL)

    if provider in ("gemini", "google"):
        from mistara.providers.vlm.gemini_client import DEFAULT_MODEL, GeminiVLM

        return GeminiVLM(model or DEFAULT_MODEL)

    raise ValueError(
        f"unknown provider {provider!r} in spec {spec!r}; "
        f"known providers: stub, anthropic, gemini"
    )


def get_role(role: str) -> VLMClient:
    if role not in DEFAULT_ROLES:
        raise KeyError(f"unknown role {role!r}; known: {sorted(DEFAULT_ROLES)}")
    return get_vlm(DEFAULT_ROLES[role])
