"""Shared LLM factory for the RAG pipeline."""

from __future__ import annotations

from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import get_settings


def get_llm(*, temperature: float = 0.0, max_output_tokens: int | None = None, model: str | None = None):
    """Create a Gemini chat model with the configured API key."""
    settings = get_settings()
    chosen_model = model or settings.LLM_MODEL
    
    # Handle mock model for testing
    if str(chosen_model).strip().lower() == "mock":
        from langchain_community.chat_models import FakeListChatModel
        return FakeListChatModel(
            responses=[
                '{"query_type":"search","domain":"other"}', # Router mock
                "This is a mock search result from the video transcript [00:42].", # Generator mock
                '{"passed": true, "reason": "mock"}', # Guardrail mock
                "What else can you tell me?\nHow does this work?\nCan you summarize?" # Suggestions mock
            ]
        )

    # Handle Groq models
    if any(m in str(chosen_model).lower() for m in ["llama", "mixtral", "gemma", "deepseek"]):
        from langchain_groq import ChatGroq
        groq_key = settings.GROQ_API_KEY.strip()
        if not groq_key:
            raise RuntimeError("GROQ_API_KEY is required for Groq inference")
        
        return ChatGroq(
            model=chosen_model,
            groq_api_key=groq_key,
            temperature=temperature,
            max_tokens=max_output_tokens,
            timeout=float(getattr(settings, "LLM_STEP_TIMEOUT_SECONDS", 15.0) or 15.0),
        )

    api_key = settings.GOOGLE_API_KEY.strip()
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is required for Gemini inference")

    kwargs = {
        "model": chosen_model,
        "google_api_key": api_key,
        "temperature": temperature,
        "timeout": float(getattr(settings, "LLM_STEP_TIMEOUT_SECONDS", 15.0) or 15.0),
    }
    if max_output_tokens is not None:
        kwargs["max_output_tokens"] = max_output_tokens

    return ChatGoogleGenerativeAI(**kwargs)


def get_fallback_llm(*, temperature: float = 0.0, max_output_tokens: int | None = None):
    """Return a fallback LLM if configured, otherwise None."""
    settings = get_settings()
    fb = settings.LLM_FALLBACK_MODEL.strip()
    if not fb:
        return None
    return get_llm(temperature=temperature, max_output_tokens=max_output_tokens, model=fb)