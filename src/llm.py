"""LLM and embedding factories. Provider is chosen through environment variables."""
from __future__ import annotations

from functools import lru_cache

from . import config


def get_llm(temperature: float = 0.2):
    p = config.LLM_PROVIDER
    if p == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(model=config.LLM_MODEL, temperature=temperature)
    if p == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=config.LLM_MODEL, temperature=temperature)
    if p == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=config.LLM_MODEL, temperature=temperature)
    if p == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=config.LLM_MODEL, temperature=temperature, base_url=config.OLLAMA_BASE_URL
        )
    raise ValueError(f"Unknown LLM_PROVIDER '{p}' (use groq | openai | anthropic | ollama)")


@lru_cache(maxsize=1)
def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )
