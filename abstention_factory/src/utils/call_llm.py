from __future__ import annotations

import json
import os
import re
import threading
import time
from typing import Optional

from dotenv import load_dotenv
import openai
from openai import OpenAI
import anthropic

# Load .env from project root (override=True so .env wins over shell exports)
load_dotenv(override=True)

# Provider: "native" (default) or "bedrock"
_PROVIDER = os.getenv("LLM_PROVIDER", "native").lower()


# ------------------------------------------------------------------ Usage tracking
_usage_lock = threading.Lock()
_usage_records: list[dict] = []


def _record_usage(model: str, input_tokens: int, output_tokens: int) -> None:
    with _usage_lock:
        _usage_records.append({
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        })


def get_usage_summary() -> dict:
    """Return aggregated token usage by model since process start (or last reset)."""
    with _usage_lock:
        by_model: dict[str, dict] = {}
        for rec in _usage_records:
            m = rec["model"]
            if m not in by_model:
                by_model[m] = {"input_tokens": 0, "output_tokens": 0, "calls": 0}
            by_model[m]["input_tokens"] += rec["input_tokens"]
            by_model[m]["output_tokens"] += rec["output_tokens"]
            by_model[m]["calls"] += 1
        return by_model


def reset_usage() -> None:
    """Clear accumulated usage records."""
    with _usage_lock:
        _usage_records.clear()


class EmptyContentError(ValueError):
    """Provider returned an empty content payload (e.g. Bedrock content filter)."""


def _is_claude_model(model: str) -> bool:
    return "claude" in model.lower()


# ------------------------------------------------------------------ OpenAI
def _call_openai(messages: list, model: str, temperature: float, max_tokens: int) -> str:
    """Call OpenAI models via the Responses API."""
    client = OpenAI()

    # Convert messages list to Responses API input format
    # Responses API accepts: string, or list of {role, content} dicts
    input_data = messages

    kwargs: dict = {
        "model": model,
        "input": input_data,
        "max_output_tokens": max_tokens,
    }

    response = client.responses.create(**kwargs)
    usage = getattr(response, "usage", None)
    if usage:
        _record_usage(
            model,
            getattr(usage, "input_tokens", 0),
            getattr(usage, "output_tokens", 0),
        )
    return response.output_text or ""


# ------------------------------------------------------------------ Anthropic
#
# Bedrock auth note: anthropic.AnthropicBedrock uses boto3's default
# credential chain (SigV4 with IAM access keys / role) and IGNORES
# AWS_BEARER_TOKEN_BEDROCK. On accounts where the IAM principal in the
# default chain hasn't filled Anthropic's per-principal use-case form
# (PutUseCaseForModelAccess), every Claude 4.x call returns 404
# "Model use case details have not been submitted for this account",
# even when a fresh Bedrock API key sits in AWS_BEARER_TOKEN_BEDROCK.
# boto3.client("bedrock-runtime").invoke_model DOES honour the bearer
# token automatically (recent botocore picks up AWS_BEARER_TOKEN_BEDROCK
# as a first-class credential), so we route Bedrock-mode calls through
# boto3 directly and shape the response to match the Anthropic SDK
# contract the rest of this module expects.
def _get_anthropic_client():
    """Return an Anthropic client for native (non-Bedrock) use only.

    Bedrock callers use _bedrock_invoke() instead — see comment above.
    """
    return anthropic.Anthropic()


def _call_anthropic_native(messages: list, model: str, temperature: float, max_tokens: int) -> str:
    """Native Anthropic API path — unchanged from the original
    implementation. Exists as the non-Bedrock branch of _call_anthropic.
    """
    client = _get_anthropic_client()

    system = None
    api_messages = []
    for msg in messages:
        if msg["role"] == "system":
            system = msg["content"]
        else:
            api_messages.append(msg)

    kwargs: dict = {
        "model": model,
        "messages": api_messages,
        "max_tokens": max_tokens,
    }
    if system:
        kwargs["system"] = system
    if temperature is not None:
        kwargs["temperature"] = temperature

    response = client.messages.create(**kwargs)
    if not response.content:
        raise EmptyContentError(
            f"Anthropic API returned empty content (likely provider-side "
            f"content filter). model={model} stop_reason="
            f"{getattr(response, 'stop_reason', '?')!r}"
        )
    usage = getattr(response, "usage", None)
    if usage:
        _record_usage(
            model,
            getattr(usage, "input_tokens", 0),
            getattr(usage, "output_tokens", 0),
        )
    return response.content[0].text


# Module-level Bedrock client; created lazily so non-Bedrock users don't
# import boto3 at all.
_bedrock_client = None
_bedrock_client_lock = threading.Lock()


def _get_bedrock_runtime_client():
    global _bedrock_client
    if _bedrock_client is not None:
        return _bedrock_client
    with _bedrock_client_lock:
        if _bedrock_client is None:
            import boto3
            _bedrock_client = boto3.client(
                "bedrock-runtime",
                region_name=os.getenv("AWS_REGION", "us-east-1"),
            )
    return _bedrock_client


def _bedrock_invoke(messages: list, model: str, temperature: float, max_tokens: int) -> str:
    """Bedrock path — call invoke_model directly through boto3 so the
    AWS_BEARER_TOKEN_BEDROCK Bedrock API key is honoured (see module
    note above). The wire body is Anthropic's standard messages-api
    JSON, which boto3 forwards verbatim; the response body is the
    same JSON shape AnthropicBedrock returns, so the rest of this
    module sees identical fields (`content[0].text`, `usage.input_tokens`,
    `stop_reason`).
    """
    client = _get_bedrock_runtime_client()

    system = None
    api_messages = []
    for msg in messages:
        if msg["role"] == "system":
            system = msg["content"]
        else:
            api_messages.append(msg)

    body: dict = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "messages": api_messages,
    }
    if system:
        body["system"] = system
    if temperature is not None:
        body["temperature"] = temperature

    try:
        resp = client.invoke_model(modelId=model, body=json.dumps(body))
    except Exception as exc:
        # Re-raise with the model name in the message so the existing
        # retry-loop logging stays informative. We deliberately do NOT
        # translate to anthropic.* exception types — _is_retryable knows
        # to look at boto3 exceptions too (see _RETRYABLE_BOTO3 below).
        raise RuntimeError(
            f"Failed to call Bedrock model {model!r}: {type(exc).__name__}: {exc}"
        ) from exc

    out = json.loads(resp["body"].read())
    content = out.get("content") or []
    if not content:
        raise EmptyContentError(
            f"Bedrock returned empty content (likely provider-side content "
            f"filter). model={model} stop_reason={out.get('stop_reason')!r}"
        )
    usage = out.get("usage") or {}
    _record_usage(
        model,
        int(usage.get("input_tokens") or 0),
        int(usage.get("output_tokens") or 0),
    )
    # Concatenate every text-typed content block in case the model
    # returns multiple (e.g. mixed thinking + answer blocks). Plain
    # text-only responses are unchanged because content[0].type == 'text'.
    return "".join(
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )


def _call_anthropic(messages: list, model: str, temperature: float, max_tokens: int) -> str:
    """Router: pick Bedrock invoke_model (boto3 + bearer token) or
    native Anthropic API based on the LLM_PROVIDER env var.

    The split exists because anthropic.AnthropicBedrock ignores
    AWS_BEARER_TOKEN_BEDROCK and falls back to whichever IAM principal
    boto3's default chain finds — on accounts where that principal
    hasn't filled Anthropic's per-principal use-case form, every Claude
    4.x call returns 404. Going through boto3 directly fixes this
    without touching any model name, prompt, or critic logic.
    """
    if _PROVIDER == "bedrock":
        return _bedrock_invoke(messages, model, temperature, max_tokens)
    return _call_anthropic_native(messages, model, temperature, max_tokens)


# ------------------------------------------------------------------ Router
_RETRYABLE_NETWORK = (
    anthropic.RateLimitError,
    anthropic.APIConnectionError,
    openai.RateLimitError,
    openai.APIConnectionError,
)
_RETRYABLE_STATUS = (anthropic.APIStatusError, openai.APIStatusError)

# Bedrock path raises botocore exceptions (we wrap them in RuntimeError
# in _bedrock_invoke for legibility, but the original is preserved as
# the cause). These match the same retryable shape as the anthropic
# SDK errors above.
_RETRYABLE_BOTO3_ERROR_CODES = {
    "ThrottlingException",
    "TooManyRequestsException",
    "ServiceUnavailableException",
    "InternalServerException",
    "ModelTimeoutException",
    "ModelStreamErrorException",
    "ModelNotReadyException",
}


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, EmptyContentError):
        return True
    if isinstance(exc, _RETRYABLE_NETWORK):
        return True
    if isinstance(exc, _RETRYABLE_STATUS):
        status = getattr(exc, "status_code", None)
        return status is not None and 500 <= status < 600
    # Bedrock invoke_model wraps boto errors. Walk the cause chain so
    # the wrapper RuntimeError still gets retried for transient errors.
    cur = exc
    seen: set[int] = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        cls_name = type(cur).__name__
        if cls_name in _RETRYABLE_BOTO3_ERROR_CODES:
            return True
        # botocore.exceptions.ClientError carries an error code in
        # its response['Error']['Code']; check that too.
        resp = getattr(cur, "response", None)
        if isinstance(resp, dict):
            code = (resp.get("Error") or {}).get("Code")
            if code in _RETRYABLE_BOTO3_ERROR_CODES:
                return True
        cur = cur.__cause__
    return False


def call_llm(
    model: str,
    prompt: Optional[str] = None,
    messages: Optional[list] = None,
    temperature: float = 0.2,
    max_tokens: int = 2000,
    retries: int = 3,
) -> str:
    """Call an LLM using either a prompt or explicit message list.

    Routing:
      - claude-* models → Anthropic Messages API (native or Bedrock)
      - everything else  → OpenAI Responses API

    Retries (exponential backoff 1s/2s/4s) cover transient provider errors:
    rate limits, connection failures, 5xx, and Bedrock empty-content filter
    misses. Auth errors, 4xx, and input-validation errors fail fast.
    """
    if messages is None:
        if prompt is None:
            raise ValueError("Either prompt or messages must be provided")
        messages = [{"role": "user", "content": prompt}]

    last_exc: Optional[BaseException] = None
    for attempt in range(retries + 1):
        try:
            if _is_claude_model(model):
                return _call_anthropic(messages, model, temperature, max_tokens)
            return _call_openai(messages, model, temperature, max_tokens)
        except Exception as exc:
            if attempt < retries and _is_retryable(exc):
                last_exc = exc
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"Failed to call LLM ({model}): {exc}") from exc

    # Unreachable: loop either returns or raises, but keep mypy happy.
    raise RuntimeError(f"Failed to call LLM ({model}): {last_exc}") from last_exc


def parse_json_response(response_text: str) -> dict:
    """Extract and parse a JSON object from raw model output."""

    text = response_text.strip()

    # Try fenced code blocks anywhere in the text (not just at the start)
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    if fenced:
        # Pick the longest fenced block (most likely the full JSON)
        candidate = max(fenced, key=len).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # Try parsing the whole text as JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: find outermost balanced braces
    start = text.find("{")
    if start == -1:
        raise ValueError(
            "LLM response did not contain a JSON object\n"
            "Response text:\n" + "=" * 50 + "\n" + response_text[:2000] + "\n" + "=" * 50
        )

    # Walk forward to find the matching closing brace (handle nesting)
    depth = 0
    in_string = False
    escape = False
    end = -1
    for i in range(start, len(text)):
        c = text[i]
        if escape:
            escape = False
            continue
        if c == "\\":
            escape = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                end = i
                break

    if end == -1:
        raise ValueError(
            "LLM response contained unbalanced braces\n"
            "Response text:\n" + "=" * 50 + "\n" + response_text[:2000] + "\n" + "=" * 50
        )

    return json.loads(text[start : end + 1])
