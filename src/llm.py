"""LLM and embedding factories. Provider is chosen through environment variables."""
from __future__ import annotations

import logging
import re
import time
from functools import lru_cache

from . import config

log = logging.getLogger(__name__)


def get_llm(temperature: float = 0.2, provider: str | None = None, model: str | None = None):
    p = provider or config.LLM_PROVIDER
    m = model or config.LLM_MODEL
    if p == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(model=m, temperature=temperature)
    if p == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=m, temperature=temperature)
    if p == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=m, temperature=temperature)
    if p in ("google", "gemini"):
        from langchain_google_genai import ChatGoogleGenerativeAI

        # The key is read from GOOGLE_API_KEY (loaded from .env by config).
        return ChatGoogleGenerativeAI(model=m, temperature=temperature)
    if p == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=m, temperature=temperature, base_url=config.OLLAMA_BASE_URL)
    raise ValueError(f"Unknown LLM_PROVIDER '{p}' (use groq | openai | anthropic | google | ollama)")


def _light_choice() -> tuple[str, str]:
    provider = config.LIGHT_PROVIDER or config.LLM_PROVIDER
    # A light provider without its own model must not inherit the writer's model name.
    model = config.LIGHT_MODEL or (config.LLM_MODEL if provider == config.LLM_PROVIDER else "")
    if not model:
        raise ValueError(f"LIGHT_PROVIDER='{provider}' needs LIGHT_MODEL to be set")
    return provider, model


def get_light_llm(temperature: float = 0.2):
    """LLM for job analysis, cover letter and interview questions; the main model if unset."""
    provider, model = _light_choice()
    return get_llm(temperature, provider=provider, model=model)


def light_is_distinct() -> bool:
    """True when the light model differs from the main one (so it can fall back to it)."""
    return _light_choice() != (config.LLM_PROVIDER, config.LLM_MODEL)


def get_judge_llm(temperature: float = 0.0):
    """LLM used by the Critic. Defaults to the main model when no JUDGE_* is configured."""
    provider = config.JUDGE_PROVIDER or config.LLM_PROVIDER
    # A judge provider without its own model must not inherit the writer's model name.
    model = config.JUDGE_MODEL or (config.LLM_MODEL if provider == config.LLM_PROVIDER else None)
    if not model:
        raise ValueError(f"JUDGE_PROVIDER='{provider}' needs JUDGE_MODEL to be set")
    return get_llm(temperature, provider=provider, model=model)


def get_judge_fallback_llm(temperature: float = 0.0):
    """Backup judge, or None when it would be the same model as the primary judge."""
    provider = config.JUDGE_FALLBACK_PROVIDER or config.LLM_PROVIDER
    model = config.JUDGE_FALLBACK_MODEL or (
        config.LLM_MODEL if provider == config.LLM_PROVIDER else None
    )
    if not model:
        raise ValueError(
            f"JUDGE_FALLBACK_PROVIDER='{provider}' needs JUDGE_FALLBACK_MODEL to be set"
        )
    primary_provider = config.JUDGE_PROVIDER or config.LLM_PROVIDER
    primary_model = config.JUDGE_MODEL or config.LLM_MODEL
    if (provider, model) == (primary_provider, primary_model):
        return None
    return get_llm(temperature, provider=provider, model=model)


class QuotaExceeded(RuntimeError):
    """The provider refused the call because a free-tier rate limit was reached."""

    def __init__(self, message: str, daily: bool, wait: float = 0.0):
        super().__init__(message)
        self.daily = daily  # True: the per-day quota is spent, waiting seconds will not help
        self.wait = wait  # seconds the provider asks us to wait (0 if unknown)


def _seconds(text: str) -> float:
    """'33m30.096s' / '1h2m' / '7.5s' -> seconds."""
    m = re.fullmatch(r"(?:(\d+)h)?(?:(\d+)m)?(?:([\d.]+)s)?", text.strip())
    if not m or not any(m.groups()):
        return 0.0
    h, mn, s = m.groups()
    return int(h or 0) * 3600 + int(mn or 0) * 60 + float(s or 0)


def quota_error(error: Exception, model=None) -> QuotaExceeded | None:
    """Turn an HTTP 429 from the provider into a QuotaExceeded with a readable message."""
    # Groq/OpenAI expose status_code, google-genai exposes code; wrappers chain the original.
    status, text, err = None, str(error), error
    for _ in range(4):
        status = status or getattr(err, "status_code", None) or getattr(err, "code", None) or getattr(
            getattr(err, "response", None), "status_code", None
        )
        err = getattr(err, "__cause__", None)
        if err is None:
            break
        text += " " + str(err)
    if status != 429 and "RESOURCE_EXHAUSTED" not in text:
        return None
    daily = bool(re.search(r"per day|PerDay|\((?:TPD|RPD)\)", text, flags=re.I))
    found = re.search(r"(?:try again|retry) in ([0-9hms.]+)", text, flags=re.I) or re.search(
        r"retryDelay\W+([0-9.]+s)", text
    )
    wait = _seconds(found.group(1).rstrip(".")) if found else 0.0
    name = getattr(model, "model_name", None) or getattr(model, "model", None) or "the model"
    if daily:
        when = f" Try again in about {found.group(1).rstrip('.')}." if found else ""
        message = f"The free daily quota of '{name}' is used up.{when}"
    else:
        message = f"Rate limit reached for '{name}'. Try again in a moment."
    return QuotaExceeded(message, daily=daily, wait=wait)


def invoke_structured(prompt, schema, inputs, llm, fallback=None, attempts: int | None = None):
    """prompt | llm.with_structured_output(schema), retried; then retried on `fallback` if given.

    Free models sometimes answer without calling the tool or return invalid JSON; the next
    attempt usually succeeds. Raises the last error when every attempt failed.
    """
    attempts = attempts or config.STRUCTURED_RETRIES
    last: Exception | None = None
    for label, model in (("primary", llm), ("fallback", fallback)):
        if model is None:
            continue
        chain = prompt | model.with_structured_output(schema)
        for i in range(1, attempts + 1):
            try:
                return chain.invoke(inputs)
            except Exception as e:  # provider 400s, invalid JSON, validation errors
                last = e
                log.warning(
                    "%s call failed (%s model, attempt %d/%d): %s",
                    schema.__name__, label, i, attempts, str(e)[:200],
                )
                quota = quota_error(e, model)
                if quota:
                    last = quota  # a readable error if this ends up being the final failure
                    if quota.daily:
                        break  # retrying cannot help today: go to the next model, if any
                if i < attempts:
                    time.sleep(min(quota.wait, 60) if quota and quota.wait else 2 * i)
    raise last


@lru_cache(maxsize=1)
def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )
