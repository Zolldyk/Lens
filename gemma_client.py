"""Thin wrapper around google-genai for Gemma 4 calls.

Centralises:
  * API key loading from environment
  * Model selection (gemma-4-26b-a4b-it default, gemma-4-31b-it alternative)
  * Thinking-mode handling (SDK-native if available; prompt fallback otherwise)
  * Error mapping (rate-limit, auth, network) to typed result dicts
  * Token / latency capture for the test battery report
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any

from google import genai
from google.genai import types

from prompts import THINKING_FALLBACK_PREFIX
from render import resize_if_needed, split_thinking

DEFAULT_MODEL = "gemma-4-26b-a4b-it"
ALT_MODEL = "gemma-4-31b-it"
SUPPORTED_MODELS = (DEFAULT_MODEL, ALT_MODEL)

# State flag — populated on first call. None = untried, True = SDK supports
# thinking_config, False = use prompt fallback.
_THINKING_SDK_SUPPORTED: bool | None = None


@dataclass
class GenerateResult:
    ok: bool
    text: str = ""
    thoughts: str | None = None
    error: str | None = None
    error_kind: str | None = None  # 'rate_limit' | 'auth' | 'network' | 'other'
    model: str = ""
    latency_ms: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    thinking_mode: str = "off"  # 'off' | 'sdk' | 'prompt'

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _get_client() -> genai.Client:
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key or key == "your-key-here":
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key "
            "from https://aistudio.google.com/apikey"
        )
    return genai.Client(api_key=key)


def _classify_error(err: Exception) -> str:
    msg = str(err).lower()
    if "429" in msg or "rate" in msg or "quota" in msg or "resource_exhausted" in msg:
        return "rate_limit"
    if "401" in msg or "403" in msg or "permission" in msg or "api key" in msg:
        return "auth"
    if any(s in msg for s in ("timeout", "connection", "network", "unreachable")):
        return "network"
    return "other"


def _extract_token_counts(response: Any) -> tuple[int | None, int | None]:
    usage = getattr(response, "usage_metadata", None)
    if not usage:
        return None, None
    input_tokens = getattr(usage, "prompt_token_count", None)
    output_tokens = getattr(usage, "candidates_token_count", None)
    return input_tokens, output_tokens


def generate(
    image_bytes: bytes | None,
    mime: str,
    prompt: str,
    model: str = DEFAULT_MODEL,
    thinking: bool = False,
) -> GenerateResult:
    """Call Gemma 4 with optional image and optional thinking mode.

    On any exception, returns a GenerateResult with ok=False and a populated
    error_kind. Never raises (except for missing API key at first call).
    """
    global _THINKING_SDK_SUPPORTED

    if model not in SUPPORTED_MODELS:
        return GenerateResult(
            ok=False,
            error=f"Unsupported model: {model}",
            error_kind="other",
            model=model,
        )

    final_prompt = prompt
    thinking_mode = "off"
    if thinking:
        if _THINKING_SDK_SUPPORTED is False:
            final_prompt = THINKING_FALLBACK_PREFIX + prompt
            thinking_mode = "prompt"
        else:
            thinking_mode = "sdk"  # tentative; may downgrade on failure

    client = _get_client()

    # Build contents — prepend image part if provided
    parts: list[Any] = []
    if image_bytes:
        resized_bytes, resized_mime = resize_if_needed(image_bytes, mime)
        parts.append(types.Part.from_bytes(data=resized_bytes, mime_type=resized_mime))
    parts.append(final_prompt)

    start = time.perf_counter()
    config = None
    if thinking and _THINKING_SDK_SUPPORTED is not False:
        try:
            config = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(include_thoughts=True)
            )
        except Exception:  # noqa: BLE001 — any SDK shape mismatch falls back
            _THINKING_SDK_SUPPORTED = False
            config = None
            final_prompt = THINKING_FALLBACK_PREFIX + prompt
            thinking_mode = "prompt"
            parts[-1] = final_prompt

    try:
        response = client.models.generate_content(
            model=model,
            contents=parts,
            config=config,
        )
    except Exception as err:  # noqa: BLE001 — we intentionally catch all
        latency_ms = int((time.perf_counter() - start) * 1000)
        return GenerateResult(
            ok=False,
            error=str(err)[:500],
            error_kind=_classify_error(err),
            model=model,
            latency_ms=latency_ms,
            thinking_mode=thinking_mode,
        )

    latency_ms = int((time.perf_counter() - start) * 1000)

    # Try to extract SDK-native thoughts if present
    thoughts: str | None = None
    try:
        answer_text = response.text or ""
    except Exception as err:  # noqa: BLE001 — .text can raise on safety blocks
        return GenerateResult(
            ok=False,
            error=f"Response blocked or empty: {err}"[:500],
            error_kind="other",
            model=model,
            latency_ms=latency_ms,
            thinking_mode=thinking_mode,
        )
    candidates = getattr(response, "candidates", None) or []
    if thinking_mode == "sdk" and candidates:
        cand = candidates[0]
        content = getattr(cand, "content", None)
        parts_out = getattr(content, "parts", None) if content else None
        if parts_out:
            thought_chunks: list[str] = []
            answer_chunks: list[str] = []
            for p in parts_out:
                text_chunk = getattr(p, "text", None) or ""
                if getattr(p, "thought", False):
                    thought_chunks.append(text_chunk)
                else:
                    answer_chunks.append(text_chunk)
            if thought_chunks:
                thoughts = "\n".join(c for c in thought_chunks if c).strip() or None
            if answer_chunks:
                answer_text = "\n".join(c for c in answer_chunks if c).strip()
        # If SDK said thinking but returned no thought parts, mark first-call result
        # but don't toggle the global — let the next call retry.
        if not thoughts and _THINKING_SDK_SUPPORTED is None:
            _THINKING_SDK_SUPPORTED = False
            thinking_mode = "prompt"
            # Re-run once with prompt fallback inline so the user still gets thoughts.
            return generate(image_bytes, mime, prompt, model=model, thinking=True)
        if _THINKING_SDK_SUPPORTED is None:
            _THINKING_SDK_SUPPORTED = True
    elif thinking_mode == "prompt":
        thoughts, answer_text = split_thinking(answer_text)

    input_tokens, output_tokens = _extract_token_counts(response)

    return GenerateResult(
        ok=True,
        text=answer_text,
        thoughts=thoughts,
        model=model,
        latency_ms=latency_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        thinking_mode=thinking_mode,
    )
