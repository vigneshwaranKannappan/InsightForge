"""Chat-model factory configured from environment variables (.env).

LLM_PROVIDER = openai | anthropic   (blank -> offline mode, no LLM calls)
LLM_MODEL    = optional model override
OPENAI_API_KEY / ANTHROPIC_API_KEY = provider credentials
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel

DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-5-5",
}
API_KEY_VARS = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}


@dataclass(frozen=True)
class LLMSettings:
    provider: str | None
    model: str | None
    temperature: float = 0.0

    @property
    def is_configured(self) -> bool:
        return bool(self.provider and os.getenv(API_KEY_VARS[self.provider]))

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}" if self.is_configured else "offline"


def load_settings() -> LLMSettings:
    load_dotenv()
    provider = (os.getenv("LLM_PROVIDER") or "").strip().lower() or None
    if provider is not None and provider not in DEFAULT_MODELS:
        supported = ", ".join(DEFAULT_MODELS)
        raise ValueError(f"Unsupported LLM_PROVIDER '{provider}'. Choose from: {supported}")
    model = (os.getenv("LLM_MODEL") or "").strip() or (
        DEFAULT_MODELS[provider] if provider else None
    )
    temperature = float(os.getenv("LLM_TEMPERATURE", "0"))
    return LLMSettings(provider=provider, model=model, temperature=temperature)


def get_chat_model(settings: LLMSettings | None = None) -> BaseChatModel | None:
    """Return a configured chat model, or None when running offline."""
    settings = settings or load_settings()
    if not settings.is_configured:
        return None
    if settings.provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=settings.model, temperature=settings.temperature)
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(model=settings.model, temperature=settings.temperature)
