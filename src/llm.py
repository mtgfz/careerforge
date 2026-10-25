"""LLM and embedding factories. Provider is chosen through environment variables."""
from __future__ import annotations

import logging
import re
import time
from functools import lru_cache

from langchain_core.runnables.fallbacks import RunnableWithFallbacks

from . import config

log = logging.getLogger(__name__)


_PROVIDERS = {"groq", "openai", "anthropic", "google", "gemini", "ollama"}


class ModelChain(RunnableWithFallbacks):
    """Several chat models tried in order: when one fails (overloaded, unavailable, too slow,
    quota used up, invalid output) the next one answers. Works with `prompt | chain` and with
    `chain.with_structured_output(schema)` like a single chat model."""

    def with_structured_output(self, schema, **kwargs):
        chains = [m.with_structured_output(schema, **kwargs) for m in (self.runnable, *self.fallbacks)]
        return RunnableWithFallbacks(
            runnable=chains[0], fallbacks=chains[1:], exceptions_to_handle=(Exception,)
        )

    @property
    def model_name(self) -> str:
        names = [getattr(m, "model_name", None) or getattr(m, "model", "?") for m in (self.runnable, *self.fallbacks)]
        return " > ".join(map(str, names))


def _build_llm(provider: str, model: str, temperature: float):
    """One chat model. Short timeout and few SDK retries: the chain moves on quickly instead."""
    p = provider
    kw = {"timeout": config.LLM_TIMEOUT, "max_retries": config.LLM_MAX_RETRIES}
    if p == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(model=model, temperature=temperature, **kw)
    if p == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, temperature=temperature, **kw)
    if p == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model, temperature=temperature, **kw)
    if p in ("google", "gemini"):
        from langchain_google_genai import ChatGoogleGenerativeAI

        # The key is read from GOOGLE_API_KEY (loaded from .env by config).
        return ChatGoogleGenerativeAI(model=model, temperature=temperature, **kw)
    if p == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=model, temperature=temperature, base_url=config.OLLAMA_BASE_URL)
    raise ValueError(f"Unknown LLM_PROVIDER '{p}' (use groq | openai | anthropic | google | ollama)")


def _parse_models(spec: str, default_provider: str) -> list[tuple[str, str]]:
    """'gemini-3.8-flash, gemini-3.6-flash, groq:openai/gpt-oss-120b' -> [(provider, model), ...]"""
    out = []
    for entry in (e.strip() for e in spec.split(",")):
        if not entry:
            continue
        head, sep, rest = entry.partition(":")
        if sep and head.lower() in _PROVIDERS:  # 'qwen2.5:7b' is a model name, not a provider
            out.append((head.lower(), rest.strip()))
        else:
            out.append((default_provider, entry))
    return out


def get_llm(temperature: float = 0.2, provider: str | None = None, model: str | None = None):
    """A chat model, or a ModelChain when `model` lists several (comma-separated, in order of
    preference). An entry may name its own provider: 'google:gemini-3.6-flash'."""
    p = (provider or config.LLM_PROVIDER).lower()
    choices = _parse_models(model or config.LLM_MODEL, p)
    if not choices:
        raise ValueError("No model configured (LLM_MODEL is empty)")
    models = [_build_llm(cp, cm, temperature) for cp, cm in choices]
    if len(models) == 1:
        return models[0]
    return ModelChain(runnable=models[0], fallbacks=models[1:], exceptions_to_handle=(Exception,))


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
